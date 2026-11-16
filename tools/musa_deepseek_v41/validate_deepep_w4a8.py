# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint DeepEP/MATE integration, not full-model accuracy/throughput.

The control is the existing GPU expert implementation plus CPU Gloo aggregation
of its final small test outputs. No model projection is computed on the CPU.
Independent quantization checks also cover NaN padding and dynamic graph counts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from deep_ep import Buffer
from safetensors import safe_open


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _inputs(rank: int, rows: int, experts: int, topk: int, variant: int,
            owner: int, negative: bool, all_active: bool,
            true_empty: bool = False) -> tuple[torch.Tensor, ...]:
    active = all_active or rank == owner
    count = rows if active or not true_empty else 0
    x = (((torch.arange(count * 5120).reshape(count, 5120) * 7) % 97).float()
         - 48 + rank + variant) / 32
    ids = ((torch.arange(count)[:, None] * 3 + torch.arange(topk)[None, :]
            * (experts // 8) + variant * 7 + rank) % experts).long()
    if not active:
        ids.fill_(-1)
    if negative:
        ids[:, ::2] = -1
    weights = torch.arange(1, topk + 1).float().expand(count, topk).clone()
    weights *= 1.5 / (topk * (topk + 1) / 2)
    return x.to(torch.bfloat16), ids, weights


def _load(model: Path, index: dict[str, str], prefix: str, first: int,
          count: int, device: torch.device) -> tuple[list[torch.Tensor], list[float]]:
    weights, scales, input_scales = [], [], []
    with ExitStack() as stack:
        handles: dict[str, Any] = {}

        def tensor(name: str) -> torch.Tensor:
            file = index[name]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(str(model / file), framework='pt', device='cpu'))
            return handles[file].get_tensor(name)

        for projection in ['w1', 'w3', 'w2']:
            names = [f'{prefix}.experts.{expert}.{projection}'
                     for expert in range(first, first + count)]
            weights.append(torch.stack([tensor(n + '.weight') for n in names]).to(device))
            scales.append(torch.stack([tensor(n + '.weight_scale_inv') for n in names])
                          .to(device=device, dtype=torch.bfloat16))
            input_scales.append(max(tensor(n + '.input_scale').item() for n in names))
    return [torch.cat(weights[:2], 1).contiguous(),
            torch.cat(scales[:2], 1).contiguous(), weights[2], scales[2]], [
                max(input_scales[:2]), input_scales[2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gemm', type=Path, required=True)
    parser.add_argument('--static-gemm', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.module))
    native = importlib.import_module('v41_deep_ep_w4a8_test_module')
    rank = int(os.environ['RANK'])
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=180))
    device = torch.device('musa', rank)
    stream = torch.musa.Stream(device=rank)
    records: list[dict[str, Any]] = []

    def stage(name: str, **details: Any) -> None:
        _save(args.output / f'rank{rank}_progress.json', {
            'at': datetime.datetime.now().astimezone().isoformat(),
            'stage': name, 'rank': rank, **details})
        _save(args.output / f'rank{rank}_partial.json', records)

    stage('quantization')
    for static in [0.0, 0.125]:
        x = torch.empty((4, 128, 5120), device=device, dtype=torch.bfloat16)
        counts = torch.empty(4, device=device, dtype=torch.int32)
        with torch.musa.stream(stream):
            x.fill_(1)
            counts.fill_(1)
            native.quant_dispatched(x, counts, static)
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            q, scales = native.quant_dispatched(x, counts, static)
        for variant, nums in enumerate([[0, 1, 6, 128], [128, 0, 2, 5], [0, 0, 0, 0]]):
            x.fill_(float('nan'))
            counts.copy_(torch.tensor(nums, dtype=torch.int32))
            for expert, n in enumerate(nums):
                values = ((torch.arange(n * 5120, device=device).reshape(n, 5120)
                           % 257).float() - 128 + variant) / 16
                x[expert, :n].copy_(values)
            torch.musa.synchronize()
            graph.replay()
            torch.musa.synchronize()
            for expert, n in enumerate(nums):
                if not n:
                    continue
                groups = x[expert, :n].float().reshape(n, 40, 128)
                scale = (torch.full((n, 1), static, device=device) if static else
                         groups.abs().amax(-1).clamp_min(1e-4) / 448)
                values = (x[expert, :n].float() / scale.repeat_interleave(
                    5120 if static else 128, -1)).clamp(-448, 448)
                expected = values.to(torch.float8_e4m3fn).float()
                torch.testing.assert_close(scales[expert, :n], scale, rtol=0, atol=0)
                torch.testing.assert_close(q[expert, :n].float(), expected, rtol=0, atol=0)
            records.append({'kind': 'quant_graph', 'static_scale': static,
                            'counts': nums, 'nan_padding': True})
        del graph, x, counts, q, scales
    index_path = args.model / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    buffers = []
    for prefix, experts, topk in [('layers.0.ffn', 384, 6), ('mtp.0.ffn', 128, 3)]:
        count, cap, hidden = experts // 8, 16, 5120
        stage('load_real_weights', prefix=prefix, first=rank * count, count=count)
        tensors, static_scales = _load(args.model, index, prefix, rank * count, count, device)
        stage('buffer_init', prefix=prefix)
        size = Buffer.get_low_latency_rdma_size_hint(cap, hidden, 8, experts)
        buffer = Buffer(dist.group.WORLD, num_rdma_bytes=size, low_latency_mode=True,
                        num_qps_per_rank=count)
        buffers.append(buffer)
        buffer.clean_low_latency_buffer(cap, hidden, experts)
        torch.musa.synchronize()
        bridge = native.Bridge(buffer, cap, experts)
        for use_static in [False, True]:
            scales_pair = static_scales if use_static else [0., 0.]
            library = args.static_gemm if use_static else args.gemm
            implementation = native.Experts(*tensors, rank * count, 10., str(library),
                                            *scales_pair, True)

            def reference(rows: int, variant: int, owner: int, negative: bool,
                          all_active: bool, true_empty: bool) -> tuple[torch.Tensor, torch.Tensor]:
                sources = [_inputs(i, rows, experts, topk, variant, owner, negative,
                                   all_active, true_empty) for i in range(8)]
                merged = [torch.cat([v[k] for v in sources]).to(device) for k in range(3)]
                # Gloo aggregates small completed GPU projections outside capture.
                with torch.musa.stream(stream):
                    local = implementation.forward(*merged)
                torch.musa.synchronize()
                reduced = local.cpu()
                dist.all_reduce(reduced)
                start = sum(v[0].size(0) for v in sources[:rank])
                result = reduced[start:start + sources[rank][0].size(0)].to(torch.bfloat16)
                flat = merged[1].cpu().flatten()
                valid = flat[(flat >= rank * count) & (flat < (rank + 1) * count)]
                counts_ref = torch.bincount(valid - rank * count, minlength=count).int()
                return result, counts_ref

            def check(output: torch.Tensor, counts: torch.Tensor, expected: torch.Tensor,
                      expected_counts: torch.Tensor) -> dict[str, float]:
                actual = output.cpu()
                torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
                if not expected.numel() or not bool(expected.any()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    return {'relative_rms': 0., 'max_abs': 0.}
                delta = actual.float() - expected.float()
                rms = (delta.square().mean().sqrt() /
                       expected.float().square().mean().sqrt().clamp_min(1e-12)).item()
                assert bool(torch.isfinite(actual).all()) and rms <= 0.02, rms
                return {'relative_rms': rms, 'max_abs': delta.abs().max().item()}

            for rows in [1, 4, 6]:
                for owner, negative, active, empty in [(0, False, False, False),
                        (7, True, False, True), (-1, False, False, False),
                        (0, False, True, False)]:
                    stage('eager', prefix=prefix, static=use_static, rows=rows, owner=owner)
                    values = [v.to(device) for v in _inputs(rank, rows, experts, topk,
                                0, owner, negative, active, empty)]
                    torch.musa.synchronize()
                    dist.barrier()
                    with torch.musa.stream(stream):
                        output, counts = bridge.forward(implementation, *values)
                    torch.musa.synchronize()
                    expected, expected_counts = reference(rows, 0, owner, negative, active, empty)
                    error = check(output, counts, expected, expected_counts)
                    records.append({'kind': 'eager', 'prefix': prefix, 'static': use_static,
                                    'rows': rows, 'owner': owner, 'negative': negative,
                                    'all_active': active, 'true_empty': empty, **error})
                    dist.barrier()
                stage('graph_prepare', prefix=prefix, static=use_static, rows=rows)
                values = [v.to(device) for v in _inputs(rank, rows, experts, topk,
                                                       0, 0, False, False)]
                torch.musa.synchronize()
                dist.barrier()
                with torch.musa.stream(stream):
                    for _ in range(2):
                        bridge.forward(implementation, *values)
                torch.musa.synchronize()
                dist.barrier()
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    output, counts = bridge.forward(implementation, *values)
                buffer.reset_low_latency_flags(cap, hidden, experts, start_index=0)
                torch.musa.synchronize()
                dist.barrier()
                for variant, owner, negative, active in [(1, 3, True, False),
                        (2, 7, False, False), (3, -1, False, False), (4, 0, False, True)]:
                    stage('graph_replay', prefix=prefix, static=use_static, rows=rows,
                          variant=variant)
                    for dst, src in zip(values, _inputs(rank, rows, experts, topk,
                                                       variant, owner, negative, active)):
                        dst.copy_(src)
                    torch.musa.synchronize()
                    dist.barrier()
                    graph.replay()
                    torch.musa.synchronize()
                    expected, expected_counts = reference(rows, variant, owner, negative, active, False)
                    error = check(output, counts, expected, expected_counts)
                    records.append({'kind': 'graph', 'prefix': prefix, 'static': use_static,
                                    'rows': rows, 'variant': variant, 'owner': owner,
                                    'all_active': active, **error})
                    dist.barrier()
                del graph, values, output, counts
            del implementation
        del bridge, tensors
        torch.musa.synchronize()
        dist.barrier()
    stage('done', checks=len(records))
    _save(args.output / f'rank{rank}.json', {
        'passed': True, 'rank': rank, 'checks': records,
        'model_index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'module_sha256': hashlib.sha256((args.module / 'v41_deep_ep_w4a8_test_module.so').read_bytes()).hexdigest(),
        'scope': 'real target0/MTP0 expert computation + LL routing/combine/Graph; not full model or shared-expert integration'})
    buffers.clear()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
