# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""C1 real-expert microbenchmark; excludes Attention/shared experts/serving."""
from __future__ import annotations
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from deep_ep import Buffer
from validate_deepep_w4a8 import _load, _inputs, _save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--collective-module', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gemm', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.module), str(args.collective_module)]
    native = importlib.import_module('v41_deep_ep_w4a8_test_module')
    collectives = importlib.import_module('v41_native_collective_test_module')
    rank = int(os.environ['RANK'])
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=180))
    device = torch.device('musa', rank)
    comm = collectives.NativeCollectives(rank, 8, 35691, rank)
    stream = torch.musa.Stream(device=rank)
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    results: list[dict[str, Any]] = []
    buffers = []

    def stage(name: str, **details: Any) -> None:
        _save(args.output / f'rank{rank}_progress.json', {
            'at': datetime.datetime.now().astimezone().isoformat(),
            'stage': name, 'rank': rank, **details})
        _save(args.output / f'rank{rank}_partial.json', results)

    def measure(run: Callable[[], Any]) -> float:
        torch.musa.synchronize()
        dist.barrier()
        with torch.musa.stream(stream):
            for _ in range(10):
                run()
            start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                run()
            end.record()
        torch.musa.synchronize()
        dist.barrier()
        return start.elapsed_time(end) * 10  # ms / 100 -> us

    for prefix, experts, topk in [('layers.0.ffn', 384, 6), ('mtp.0.ffn', 128, 3)]:
        stage('load_real_weights', prefix=prefix)
        tensors, _ = _load(args.model, index, prefix, rank * (experts // 8),
                           experts // 8, device)
        implementation = native.Experts(*tensors, rank * (experts // 8), 10.,
                                         str(args.gemm), 0., 0., True)
        size = Buffer.get_low_latency_rdma_size_hint(16, 5120, 8, experts)
        buffer = Buffer(dist.group.WORLD, num_rdma_bytes=size, low_latency_mode=True,
                        num_qps_per_rank=experts // 8)
        buffers.append(buffer)
        buffer.clean_low_latency_buffer(16, 5120, experts)
        bridge = native.Bridge(buffer, 16, experts)
        for rows in [1, 4, 6]:
            local_values = [v.to(device) for v in _inputs(rank, rows, experts, topk, 0, 0, False, False)]
            replicated = [v.to(device) for v in _inputs(0, rows, experts, topk, 0, 0, False, False)]

            def control() -> torch.Tensor:
                output = implementation.forward(*replicated)
                comm.allreduce(output)
                return output.to(torch.bfloat16)

            def candidate() -> torch.Tensor:
                return bridge.forward(implementation, *local_values)[0]

            torch.musa.synchronize()
            dist.barrier()
            with torch.musa.stream(stream):
                for _ in range(2):
                    expected = control()
                    actual = candidate()
            torch.musa.synchronize()
            reference = expected if rank == 0 else torch.zeros_like(expected)
            if rank == 0:
                rms = ((actual.float() - reference.float()).square().mean().sqrt()
                       / reference.float().square().mean().sqrt().clamp_min(1e-12)).item()
                assert rms < 0.02, rms
            else:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            graphs, captured = {}, {}
            for name, run in [('control', control), ('deepep', candidate)]:
                stage('capture', prefix=prefix, rows=rows, route=name)
                dist.barrier()
                buffer.reset_low_latency_flags(16, 5120, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    captured[name] = run()
                graphs[name] = graph
                buffer.reset_low_latency_flags(16, 5120, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
            for repeat in range(3):
                order = ['control', 'deepep'] if repeat % 2 == 0 else ['deepep', 'control']
                for name in order:
                    stage('timing', prefix=prefix, rows=rows, repeat=repeat, route=name)
                    us = measure(graphs[name].replay)
                    results.append({'kind': 'graph_device_stream_span', 'prefix': prefix,
                                    'rows': rows, 'repeat': repeat, 'route': name,
                                    'us_per_layer': us, 'warmup': 10, 'iterations': 100})
            graphs.clear()
            del graph, captured, local_values, replicated, actual, expected, reference
            torch.musa.synchronize()
            dist.barrier()
        del bridge, implementation, tensors
    comm.close()
    stage('done', measurements=len(results))
    _save(args.output / f'rank{rank}.json', {'passed': True, 'rank': rank,
        'measurements': results, 'scope': 'real target0/MTP0 routed MoE only; no shared/attention/draft acceptance/serving; control uses native MCCL SUM not cumulative-best custom allreduce'})
    buffers.clear()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
