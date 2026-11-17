# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate true-empty ranks and consecutive real-expert LL graph calls.

This is a communication/operator regression, not model accuracy or serving
performance. Optional native MCCL isolates mixed communication capture from
the complete model. Attention is deliberately omitted in both modes.
"""
from __future__ import annotations

import argparse
import datetime
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.validate_deepep_w4a8 import _inputs, _load, _save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gemm', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=[1, 40], required=True)
    parser.add_argument('--mccl-port', type=int)
    parser.add_argument('--gather-outside-graph', action='store_true')
    parser.add_argument('--variants', type=int, choices=[3, 12], default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.module))
    native = importlib.import_module('v41_deep_ep_w4a8_test_module')
    rank = int(os.environ['RANK'])
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
    device = torch.device('musa', rank)
    stream = torch.musa.Stream(device=rank)
    capture_stream = torch.musa.Stream(device=rank)
    records: list[dict[str, Any]] = []

    def stage(name: str, **details: Any) -> None:
        _save(args.output / f'rank{rank}_progress.json', {
            'at': datetime.datetime.now().astimezone().isoformat(),
            'stage': name, 'rank': rank, **details})
        logger.info(f'LL real-expert chain rank={rank} stage={name} {details}')

    stage('load_one_real_layer')
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    tensors, scales = _load(args.model, index, 'layers.0.ffn', rank * 48, 48, device)
    experts = native.Experts(*tensors, rank * 48, 10., str(args.gemm), *scales, True)
    size = Buffer.get_low_latency_rdma_size_hint(16, 5120, 8, 384)
    buffer = Buffer(dist.group.WORLD, num_rdma_bytes=size, low_latency_mode=True,
                    num_qps_per_rank=48)
    bridge = native.Bridge(buffer, 16, 384)
    collectives = None
    if args.mccl_port is not None:
        stage('native_mccl_bootstrap')
        collectives = native.Collectives(rank, args.mccl_port)

    def reset() -> None:
        torch.musa.synchronize()
        dist.barrier()
        buffer.reset_low_latency_flags(16, 5120, 384, start_index=0)
        torch.musa.synchronize()
        dist.barrier()

    # Fixed q4 reproduces target verification. Other ranks truly have zero
    # input rows, unlike a padded q4 tensor with all expert IDs set to -1.
    values = [v.to(device) for v in _inputs(rank, 4, 384, 6, 0, 0, False, False, True)]
    embedding_shard = torch.ones((4, 640), device=device, dtype=torch.bfloat16)
    embedding_buffer = torch.empty((4, 5120), device=device, dtype=torch.bfloat16)

    def prepare_embedding() -> None:
        if collectives is not None and args.gather_outside_graph:
            gathered = collectives.allgather(embedding_shard)
            embedding_buffer.copy_(gathered.permute(1, 0, 2).reshape(4, 5120))

    def forward() -> torch.Tensor:
        value = values[0]
        if collectives is not None:
            full_embedding = embedding_buffer
            if not args.gather_outside_graph:
                gathered = collectives.allgather(embedding_shard)
                full_embedding = gathered.permute(1, 0, 2).reshape(4, 5120)
            if rank == 0:
                value = value * full_embedding
        for _ in range(args.layers):
            output, _ = bridge.forward(experts, value, values[1], values[2])
            # Retain a cross-layer data dependency while bounding the test
            # activation. This synthetic recurrence is not a model forward.
            value = output * 0.01 + values[0]
        if collectives is not None and rank != 0:
            # Mirror native target: nonowners return a full zero tensor after
            # their empty-input expert calls. The output is broadcast outside
            # capture/replay, matching the model boundary.
            return torch.zeros((4, 5120), device=device, dtype=torch.bfloat16)
        return value

    def broadcast(value: torch.Tensor) -> None:
        if collectives is not None:
            collectives.broadcast(value, 0)

    buffer.clean_low_latency_buffer(16, 5120, 384)
    reset()
    stage('warmup', layers=args.layers)
    with torch.musa.stream(stream):
        for _ in range(2):
            prepare_embedding()
            output = forward()
            broadcast(output)
    reset()
    with torch.musa.stream(stream):
        prepare_embedding()
    torch.musa.synchronize()
    stage('capture', layers=args.layers)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        captured = forward()
    reset()
    for variant in range(args.variants):
        stage('eager_reference', variant=variant, layers=args.layers)
        with torch.musa.stream(stream):
            for dst, src in zip(values, _inputs(rank, 4, 384, 6, variant, 0,
                                               False, False, True)):
                dst.copy_(src)
            prepare_embedding()
            expected = forward()
            broadcast(expected)
            expected = expected.clone()
        reset()
        stage('graph_replay', variant=variant, layers=args.layers)
        with torch.musa.stream(stream):
            prepare_embedding()
            graph.replay()
            broadcast(captured)
            actual = captured.clone()
        torch.musa.synchronize()
        actual_cpu, expected_cpu = actual.cpu(), expected.cpu()
        assert torch.isfinite(actual_cpu).all()
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0)
        records.append({'variant': variant, 'layers': args.layers,
                        'local_rows': values[0].size(0), 'bitwise_equal': True,
                        'native_mccl': collectives is not None,
                        'gather_outside_graph': args.gather_outside_graph})
        _save(args.output / f'rank{rank}_partial.json', records)
        dist.barrier()
    del graph, captured, bridge
    torch.musa.synchronize()
    dist.barrier()
    stage('done', checks=len(records))
    _save(args.output / f'rank{rank}.json', {
        'rank': rank, 'passed': True, 'checks': records,
        'scope': 'real layer0 W4A8 + true-empty LL + repeated graph; no Attention',
        'native_mccl': collectives is not None})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
