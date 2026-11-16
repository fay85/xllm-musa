# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch/combine lower-bound timing. No model GEMM or request latency."""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa
from deep_ep import Buffer

sys.path.insert(0, '/expt/builds/deepep_ll_bridge_sg_v1')
import v41_deep_ep_ll_bridge_test_module as native


def main() -> None:
    root = Path(os.environ['E35_PROBE_OUTPUT'])
    root.mkdir(exist_ok=True)
    rank = int(os.environ['RANK'])
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=90))
    device = torch.device('musa', rank)
    results = []
    buffers = []
    for experts, topk in [(384, 6), (128, 3)]:
        cap, hidden = 16, 5120
        buffer = Buffer(dist.group.WORLD,
                        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(cap, hidden, 8, experts),
                        low_latency_mode=True, num_qps_per_rank=experts // 8)
        buffers.append(buffer)
        buffer.clean_low_latency_buffer(cap, hidden, experts)
        bridge = native.Bridge(buffer, cap, experts)
        for rows in [1, 4, 6]:
            x = torch.full((rows, hidden), 0.25, device=device, dtype=torch.bfloat16)
            ids = torch.arange(topk, device=device).mul(experts // 8).expand(rows, topk).contiguous().long()
            if rank != 0:
                ids.fill_(-1)
            weights = torch.full((rows, topk), 0.125, device=device, dtype=torch.float32)
            expected = torch.full((rows, hidden), topk * 0.125 * 0.25 if rank == 0 else 0, dtype=torch.bfloat16)

            def pair() -> torch.Tensor:
                recv, _ = bridge.dispatch(x, ids)
                return bridge.combine(recv, ids, weights)

            stream = torch.musa.Stream(device=rank)
            for grouped in [1, 10, 40]:
                with torch.musa.stream(stream):
                    for _ in range(2):
                        pair()
                torch.musa.synchronize()
                dist.barrier()
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    for _ in range(grouped):
                        output = pair()
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                for _ in range(10):
                    graph.replay()
                torch.musa.synchronize()
                dist.barrier()
                start = torch.musa.Event(enable_timing=True)
                end = torch.musa.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                torch.musa.synchronize()
                wall_seconds = time.perf_counter() - wall_start
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                results.append({'experts': experts, 'rows': rows, 'pairs_per_graph': grouped,
                                'iterations': 100, 'warmup': 10,
                                'device_us_per_pair': start.elapsed_time(end) * 10 / grouped,
                                'wall_us_per_pair': wall_seconds * 1e6 / 100 / grouped,
                                'scope': 'identity expert; communication lower bound; not model MoE/TPOT'})
                (root / f'rank{rank}_partial.json').write_text(json.dumps(results, indent=2))
                dist.barrier()
                del graph
    (root / f'rank{rank}.json').write_text(json.dumps({'passed': True, 'results': results}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
