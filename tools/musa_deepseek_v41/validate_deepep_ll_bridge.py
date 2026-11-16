# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Isolated real DeepEP LL routing/weighting/empty-rank/Graph regression.

This is a communication test, not an implementation of the model's MoE.
The reference expert function has a known rank- and expert-dependent factor.
"""
import argparse
import datetime
import json
import importlib
import sys
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_musa
from deep_ep import Buffer


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _inputs(rank: int, rows: int, owner: int, experts: int, topk: int,
            variant: int, negative: bool, all_active: bool,
            true_empty: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = all_active or rank == owner
    count = rows if active or not true_empty else 0
    x = ((torch.arange(count * 5120).reshape(count, 5120) % 17).float()
         + rank + variant) / 32
    ids = torch.empty((count, topk), dtype=torch.int64)
    for row in range(count):
        # Unique within a token; shared experts across different tokens.
        ids[row] = (torch.arange(topk) * (experts // 8) + row * 3
                    + variant * 7 + rank) % experts
    if not active:
        ids.fill_(-1)
    if negative and count:
        ids[:, ::2] = -1
    weights = torch.arange(1, topk + 1).float().expand(count, topk).clone() / 16
    return x.to(torch.bfloat16), ids, weights


def _expected(rank: int, rows: int, owner: int, experts: int, topk: int,
              variant: int, negative: bool, all_active: bool,
              true_empty: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x, ids, weights = _inputs(rank, rows, owner, experts, topk, variant,
                             negative, all_active, true_empty)
    local = experts // 8
    factor = (ids.clamp_min(0) // local + 1).float() + (ids.clamp_min(0) % 3).float() / 4
    expert_outputs = (x.float().unsqueeze(1) * factor.unsqueeze(-1)).to(torch.bfloat16)
    reference = (expert_outputs.float() * (weights * (ids >= 0)).unsqueeze(-1)).sum(1)
    counts = torch.zeros(local, dtype=torch.int32)
    for source in range(8):
        _, source_ids, _ = _inputs(source, rows, owner, experts, topk, variant,
                                  negative, all_active, true_empty)
        valid = source_ids[(source_ids >= rank * local) & (source_ids < (rank + 1) * local)]
        counts += torch.bincount(valid - rank * local, minlength=local).to(torch.int32)
    return reference.to(torch.bfloat16), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--native-module', type=Path)
    parser.add_argument('--mode', choices=['eager', 'graph', 'all'], default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native_module = None
    if args.native_module is not None:
        sys.path.insert(0, str(args.native_module))
        native_module = importlib.import_module('v41_deep_ep_ll_bridge_test_module')
    rank = int(os.environ['RANK'])
    progress = args.output / f'rank{rank}_progress.json'
    locals_outer = {}

    def stage(name: str, **details: Any) -> None:
        _save(progress, {'at': datetime.datetime.now().astimezone().isoformat(),
                         'rank': rank, 'stage': name, **details})
        if 'results' in locals_outer:
            _save(args.output / f'rank{rank}_partial.json', locals_outer['results'])

    stage('initialize_torch')
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=90))
    device = torch.device('musa', rank)
    results = []
    locals_outer['results'] = results
    buffers = []
    for experts, topk in [(384, 6), (128, 3)]:
        cap, hidden = 16, 5120
        size = Buffer.get_low_latency_rdma_size_hint(cap, hidden, 8, experts)
        stage('buffer_init', experts=experts, topk=topk, rdma_bytes=size)
        buffer = Buffer(dist.group.WORLD, num_rdma_bytes=size, low_latency_mode=True,
                        num_qps_per_rank=experts // 8)
        buffers.append(buffer)
        bridge = native_module.Bridge(buffer, cap, experts) if native_module else None
        stage('buffer_clean', experts=experts)
        buffer.clean_low_latency_buffer(cap, hidden, experts)
        torch.musa.synchronize()
        local_ids = torch.arange(rank * (experts // 8), (rank + 1) * (experts // 8), device=device)
        factors = ((local_ids // (experts // 8) + 1).float() + (local_ids % 3).float() / 4).to(torch.bfloat16).view(-1, 1, 1)
        stream = torch.musa.Stream(device=rank)

        def forward(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
            if bridge is not None:
                return bridge.reference_experts(x, ids, weights, factors)
            received, counts, handle, event, _ = buffer.low_latency_dispatch(
                x, ids, cap, experts, use_fp8=False, async_finish=True)
            event.current_stream_wait()
            computed = received * factors
            output, event, _ = buffer.low_latency_combine(
                computed, ids, weights, handle, async_finish=True)
            event.current_stream_wait()
            return output, counts

        def check(output: torch.Tensor, counts: torch.Tensor,
                  expected: torch.Tensor, expected_counts: torch.Tensor) -> float:
            actual = output.cpu()
            torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
            # One BF16 rounding in expert transform, then weighted FP32 combine.
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)
            if not bool(expected.any()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0

        if args.mode in ['eager', 'all']:
            cases = [(q, owner, negative, all_active, empty)
                     for q in [1, 4, 6]
                     for owner, negative, all_active, empty in [
                         (0, False, False, False), (3, True, False, False),
                         (7, False, False, True), (-1, False, False, False),
                         (0, False, True, False)]]
            for case_index, (rows, owner, negative, all_active, empty) in enumerate(cases):
                for nondefault in [False, True]:
                    stage('eager', experts=experts, case=case_index, nondefault=nondefault)
                    values = _inputs(rank, rows, owner, experts, topk, 0, negative, all_active, empty)
                    x, ids, weights = [t.to(device) for t in values]
                    torch.musa.synchronize()
                    dist.barrier()
                    with torch.musa.stream(stream if nondefault else torch.musa.default_stream()):
                        output, counts = forward(x, ids, weights)
                    torch.musa.synchronize()
                    expected, expected_counts = _expected(rank, rows, owner, experts, topk, 0, negative, all_active, empty)
                    error = check(output, counts, expected, expected_counts)
                    results.append({'kind': 'eager', 'experts': experts, 'rows': rows,
                                    'owner': owner, 'negative': negative, 'all_active': all_active,
                                    'true_empty': empty, 'nondefault_stream': nondefault,
                                    'max_abs_error': error})
                    dist.barrier()

        if args.mode in ['graph', 'all']:
            for rows in [1, 4, 6]:
                stage('graph_prepare', experts=experts, rows=rows)
                x, ids, weights = [t.to(device) for t in _inputs(rank, rows, 0, experts, topk, 0, False, False, False)]
                torch.musa.synchronize()
                dist.barrier()
                with torch.musa.stream(stream):
                    for _ in range(2):
                        forward(x, ids, weights)
                torch.musa.synchronize()
                dist.barrier()
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                graph = torch.musa.MUSAGraph()
                stage('graph_capture', experts=experts, rows=rows)
                with torch.musa.graph(graph, stream=stream):
                    captured_output, captured_counts = forward(x, ids, weights)
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                for variant, owner, negative, all_active in [(1, 3, True, False), (2, 7, False, False), (3, -1, False, False), (4, 0, False, True)]:
                    stage('graph_replay', experts=experts, rows=rows, variant=variant)
                    for destination, source in zip((x, ids, weights), _inputs(rank, rows, owner, experts, topk, variant, negative, all_active, False)):
                        destination.copy_(source)
                    torch.musa.synchronize()
                    dist.barrier()
                    graph.replay()
                    torch.musa.synchronize()
                    expected, expected_counts = _expected(rank, rows, owner, experts, topk, variant, negative, all_active, False)
                    error = check(captured_output, captured_counts, expected, expected_counts)
                    results.append({'kind': 'graph', 'experts': experts, 'rows': rows,
                                    'variant': variant, 'owner': owner, 'all_active': all_active,
                                    'max_abs_error': error})
                    dist.barrier()
                if args.benchmark:
                    # Fixed C1-like ownership: rank 0 owns rows, all other ranks
                    # still participate with masked dummy rows. No model GEMM.
                    for destination, source in zip((x, ids, weights), _inputs(rank, rows, 0, experts, topk, 0, False, False, False)):
                        destination.copy_(source)
                    torch.musa.synchronize()
                    dist.barrier()
                    for graph_mode in [False, True]:
                        stage('timing', experts=experts, rows=rows, graph=graph_mode)
                        for _ in range(10):
                            if graph_mode:
                                graph.replay()
                            else:
                                forward(x, ids, weights)
                        torch.musa.synchronize()
                        dist.barrier()
                        start = torch.musa.Event(enable_timing=True)
                        end = torch.musa.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):
                            if graph_mode:
                                graph.replay()
                            else:
                                timed_output, timed_counts = forward(x, ids, weights)
                        end.record()
                        torch.musa.synchronize()
                        expected, expected_counts = _expected(rank, rows, 0, experts, topk, 0, False, False, False)
                        check(captured_output if graph_mode else timed_output,
                              captured_counts if graph_mode else timed_counts,
                              expected, expected_counts)
                        results.append({'kind': 'communication_microbenchmark', 'experts': experts,
                                        'rows': rows, 'graph': graph_mode, 'owner': 0,
                                        'warmup': 10, 'iterations': 100,
                                        'device_span_us_per_round': start.elapsed_time(end) * 10,
                                        'scope': 'dispatch + known expert scaling + combine, not model latency'})
                        dist.barrier()
                del graph
        stage('buffer_done', experts=experts)
    torch.musa.synchronize()
    dist.barrier()
    _save(args.output / f'rank{rank}.json', {'passed': True, 'rank': rank,
        'torch': torch.__version__, 'torch_musa': torch_musa.__version__,
        'native_module': str(args.native_module), 'checks': results})
    stage('done', checks=len(results))
    # Teardown communication while every rank is still alive.
    buffers.clear()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
