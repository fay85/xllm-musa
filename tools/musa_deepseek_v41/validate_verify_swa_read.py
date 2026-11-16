# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check fused causal SWA rows, picks, slots and production cache commits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa
import v41_verify_read_test_module as native


def _reference(cache: torch.Tensor, current: torch.Tensor, table: torch.Tensor,
               positions: torch.Tensor, window: int,
               compressed: Optional[torch.Tensor], extra: Optional[torch.Tensor]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    previous = positions[:1].long()
    ring_positions = torch.arange(window, device=cache.device) + previous - window
    safe = ring_positions.clamp_min(0)
    blocks = table[0].long()[torch.div(safe, cache.shape[1], rounding_mode="floor") % table.shape[1]]
    slots = blocks.clamp_min(0) * cache.shape[1] + safe % cache.shape[1]
    ring = cache.view(-1, cache.shape[-1])[slots]
    ring = torch.where((ring_positions >= 0)[:, None], ring, 0)
    rows = torch.cat([ring, current])
    desired = positions[:, None] - window + 1 + torch.arange(window, device=cache.device)
    picks = torch.where(desired >= 0, desired - previous + window, -1).long()
    if compressed is not None:
        mapped = torch.where((extra >= 0) & (extra < len(compressed)), extra.long() + len(rows), -1)
        picks = torch.cat([picks, mapped], 1)
        rows = torch.cat([rows, compressed])
    columns = torch.div(positions.long(), cache.shape[1], rounding_mode="floor") % table.shape[1]
    slots = table[0].long()[columns] * cache.shape[1] + positions.long() % cache.shape[1]
    return rows, picks, slots


def _equal(actual: tuple, expected: tuple) -> None:
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-library", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(3401)
    torch.musa.set_device(0)
    stream = torch.musa.Stream()
    checks, graphs, attention_checks, attention_graphs = [], [], [], []
    window, block, width = 128, 32, 512
    with torch.inference_mode(), torch.musa.stream(stream):
        for q in [1, 2, 3, 4, 5, 6]:
            for dtype in [torch.int32, torch.int64]:
                # Strided page tables, positions and extra picks exercise conversion.
                table_storage = torch.tensor([[4, 2, -1, 6, 1, 3, 8, 7] * 2], device="musa", dtype=dtype)
                table = table_storage[:, ::2]
                pos_storage = torch.zeros(2*q, device="musa", dtype=dtype)
                positions = pos_storage[::2]
                cache = torch.randn(12, block, width, device="musa", dtype=torch.bfloat16)
                current = torch.randn(q, width, device="musa", dtype=torch.bfloat16)
                before = cache.clone()
                for capacity in [None, 0, 3, 80]:
                    compressed = None if capacity is None else torch.randn(capacity, width, device="musa", dtype=torch.bfloat16)
                    extra = None if capacity is None else torch.randint(-2, capacity+3, (q, 14), device="musa", dtype=dtype)[:, ::2]
                    for start in [0, 1, 31, 127, 255, 511, 2048]:
                        positions.copy_(torch.arange(start, start+q, device="musa", dtype=dtype))
                        actual = native.read(cache, current, table, positions, window, compressed, extra)
                        _equal(actual, _reference(cache, current, table, positions, window, compressed, extra))
                        torch.testing.assert_close(cache, before, rtol=0, atol=0)
                        if q == 1:
                            _equal(native.decode_read(cache, current, table, positions, window, compressed, extra), actual)
                        checks.append(dict(q=q, dtype=str(dtype), capacity=capacity, start=start))
                    if dtype == torch.int64 and capacity in [None, 80]:
                        for _ in range(2):
                            native.read(cache, current, table, positions, window, compressed, extra)
                        stream.synchronize()
                        graph = torch.musa.MUSAGraph()
                        with torch.musa.graph(graph):
                            actual = native.read(cache, current, table, positions, window, compressed, extra)
                        for start in [0, 31, 511]:
                            positions.copy_(torch.arange(start, start+q, device="musa"))
                            table_storage.copy_(table_storage.roll(1, dims=1))
                            current.copy_(torch.randn_like(current))
                            graph.replay()
                            _equal(actual, _reference(cache, current, table, positions, window, compressed, extra))
                            graphs.append(dict(q=q, capacity=capacity, start=start))
                        del graph, actual
        for selected_stream in [stream, torch.musa.default_stream()]:
            with torch.musa.stream(selected_stream):
                for q in [1, 4, 6]:
                    for capacity in [None, 80]:
                        cache = torch.randn(12, block, width, device="musa", dtype=torch.bfloat16)
                        control_cache, candidate_cache = cache.clone(), cache.clone()
                        table = torch.tensor([[4, 2, -1, 6, 1, 3, 8, 7]], device="musa", dtype=torch.int32)
                        query = torch.randn(q, 8, width, device="musa", dtype=torch.bfloat16)
                        current = torch.randn(q, width, device="musa", dtype=torch.bfloat16)
                        positions = torch.arange(q, device="musa", dtype=torch.int64)
                        sink = torch.randn(8, device="musa", dtype=torch.float32)
                        compressed = None if capacity is None else torch.randn(capacity, width, device="musa", dtype=torch.bfloat16)
                        extra = None if capacity is None else torch.randint(-2, capacity+3, (q, 64), device="musa", dtype=torch.int64)
                        control = native.Paged(control_cache, window, width**-.5, args.sparse_library, True, False, False)
                        candidate = native.Paged(candidate_cache, window, width**-.5, args.sparse_library, True, False, True)
                        for start in [0, 31, 127, 255, 511, 2048]:
                            positions.copy_(torch.arange(start, start+q, device="musa"))
                            control_cache.copy_(cache)
                            candidate_cache.copy_(cache)
                            actual = candidate.forward(query, current, table, start+q, sink, compressed, extra, positions, 0, True)
                            expected = control.forward(query, current, table, start+q, sink, compressed, extra, positions, 0, True)
                            _equal((actual, candidate_cache), (expected, control_cache))
                            attention_checks.append(dict(q=q, capacity=capacity, start=start, default_stream=selected_stream==torch.musa.default_stream()))
                        if selected_stream == stream and q == 4:
                            stream.synchronize()
                            graph = torch.musa.MUSAGraph()
                            with torch.musa.graph(graph):
                                actual = candidate.forward(query, current, table, 4096, sink, compressed, extra, positions, 0, True)
                            for start in [0, 31, 511]:
                                positions.copy_(torch.arange(start, start+q, device="musa"))
                                current.copy_(torch.randn_like(current))
                                control_cache.copy_(cache)
                                candidate_cache.copy_(cache)
                                graph.replay()
                                expected = control.forward(query, current, table, start+q, sink, compressed, extra, positions, 0, True)
                                _equal((actual, candidate_cache), (expected, control_cache))
                                attention_graphs.append(dict(capacity=capacity, start=start))
                            del graph, actual
                selected_stream.synchronize()
    result = dict(status="PASS", metadata_checks=checks, dynamic_graphs=graphs,
                  attention_and_commit_checks=attention_checks, attention_dynamic_graphs=attention_graphs,
                  scope="Native multi-query causal SWA including masked commit; not full model precision")
    (args.output/"result.json").write_text(json.dumps(result, indent=2))
    logger.info("PASS: metadata=%d graph=%d attention/cache=%d attention_graph=%d",
                len(checks), len(graphs), len(attention_checks), len(attention_graphs))


if __name__ == "__main__":
    main()
