# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank native graph profiler refresh and unchanged-output checks."""
from __future__ import annotations
import argparse
from datetime import timedelta
import collections
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_musa
import v41_verify_collective_test_module as native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    group = native.VerifyCollective(rank, 8, args.native_port)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with torch.inference_mode():
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            base = (torch.arange(4 * 5120).reshape(4, 5120) % 17 - 8 + rank).float() / 16
            if rank in (2, 6):
                base.zero_()
            expected = base.double()
            dist.all_reduce(expected)
            dtypes = [torch.bfloat16, torch.float32] * 40
            data = [base.to(device=f'musa:{rank}', dtype=dtype) for dtype in dtypes]
            outputs = group.capture_many(data, True)
            stream.synchronize()
            for phase in ['before', 'refresh', 'refresh_second_profile']:
                dist.barrier()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1]) as prof:
                    if phase != 'before':
                        assert native.NativeProfileGraph.reinstantiate_registered() == 1
                    dist.barrier()
                    for multiplier in [2, -1]:
                        for tensor, dtype in zip(data, dtypes):
                            tensor.copy_((base * multiplier).to(device=f'musa:{rank}', dtype=dtype))
                        stream.synchronize()
                        dist.barrier()
                        with torch.profiler.record_function('E28/custom_graph/' + phase):
                            group.replay_many()
                        stream.synchronize()
                        for tensor in outputs:
                            assert torch.equal(tensor.cpu().double(), expected * multiplier)
                path = args.output / f'rank{rank}_{phase}.pt.trace.json'
                prof.export_chrome_trace(str(path))
                events = json.loads(path.read_text())['traceEvents']
                counts = collections.Counter(e['name'] for e in events if e.get('ph') == 'X' and 'kernel' in e.get('cat', '').lower())
                results.append({'phase': phase, 'kernel_count': sum(counts.values()), 'kernel_names': dict(counts), 'exact_output_checks': 160})
                (args.output / f'rank{rank}.json').write_text(json.dumps({'rank':rank,'status':'RUNNING','results':results},indent=2))
            assert native.NativeProfileGraph.reinstantiate_registered() == 1
            group.release_graph()
            assert native.NativeProfileGraph.reinstantiate_registered() == 0
        stream.synchronize()
    dist.barrier()
    del group
    dist.destroy_process_group()
    (args.output / f'rank{rank}.json').write_text(json.dumps({'rank':rank,'status':'PASS','results':results,'scope':'80 mixed BF16/FP32 custom reductions, updated inputs, profiler lifecycle, not model performance'},indent=2))


if __name__ == '__main__':
    main()
