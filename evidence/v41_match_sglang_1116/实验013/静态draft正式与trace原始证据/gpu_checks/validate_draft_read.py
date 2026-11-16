# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check noncausal draft reads and MATE attention, including dynamic replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401
import v41_draft_read_test_module as native


def _reference(cache: torch.Tensor, current: torch.Tensor,
               table: torch.Tensor, context: int, window: int) -> tuple:
    positions = torch.arange(window) + context - window
    logical = positions.clamp_min(0)
    blocks = table.flatten()[logical.div(cache.shape[1], rounding_mode="floor")
                             % table.numel()].clamp_min(0)
    slots = blocks * cache.shape[1] + logical % cache.shape[1]
    ring = cache.flatten(0, 1)[slots.long()]
    ring = torch.where(positions[:, None] >= 0, ring, 0)
    rows = torch.cat((ring, current))
    valid = torch.cat((positions >= 0, torch.ones(current.shape[0], dtype=torch.bool)))
    picks = torch.where(valid, torch.arange(rows.shape[0]), -1)
    return rows, picks[None].expand(current.shape[0], -1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-library", required=True)
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111613)
    cases = []
    window = 512
    width = 512
    sparse = native.Sparse(width ** -0.5, 32 * 1024 * 1024, args.sparse_library)
    with torch.inference_mode():
        for nondefault in (False, True):
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for q in range(1, 7):
                    for context in (0, 1, 31, 32, 127, 255, 511, 512, 513, 1024):
                        dtype = torch.int32 if context % 2 else torch.int64
                        cache_cpu = torch.randn(6, 128, width).to(torch.bfloat16)
                        current_cpu = torch.randn(q, width).to(torch.bfloat16)
                        table_cpu = torch.tensor([[3, -1, 1, 4]], dtype=dtype)
                        cache = cache_cpu.to("musa")
                        current = current_cpu.to("musa")
                        table = table_cpu.to("musa")
                        previous = torch.tensor([context], dtype=torch.int64, device="musa")
                        rows, picks = native.draft_read(cache, current, table, previous, window)
                        expected_rows, expected_picks = _reference(
                            cache_cpu, current_cpu, table_cpu, context, window)
                        torch.testing.assert_close(rows.cpu(), expected_rows, atol=0, rtol=0)
                        torch.testing.assert_close(picks.cpu(), expected_picks, atol=0, rtol=0)
                        query = torch.randn(q, 8, width, device="musa", dtype=torch.bfloat16)
                        sink = torch.randn(8, device="musa", dtype=torch.float32)
                        legacy = native.Paged(cache, window, width ** -0.5,
                                              args.sparse_library, True, False)
                        expected = legacy.forward_draft_block(query, current, table, context, sink)
                        actual = sparse.forward(query, rows, picks, sink)
                        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
                        torch.testing.assert_close(cache.cpu(), cache_cpu, atol=0, rtol=0)
                        cases.append({"kind": "eager_exact_and_real_mate", "q": q,
                                      "context": context, "nondefault": nondefault,
                                      "table_dtype": str(dtype), "cache_unchanged": True})
                logger.info("Draft read exact and actual MATE checks passed, nondefault=%s", nondefault)
            stream.synchronize()

        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            for q in range(1, 7):
                cache = torch.randn(6, 128, width, device="musa", dtype=torch.bfloat16)
                current = torch.randn(q, width, device="musa", dtype=torch.bfloat16)
                table = torch.tensor([[3, -1, 1, 4]], device="musa", dtype=torch.int64)
                previous = torch.zeros(1, device="musa", dtype=torch.int64)
                query = torch.randn(q, 8, width, device="musa", dtype=torch.bfloat16)
                sink = torch.randn(8, device="musa", dtype=torch.float32)
                for _ in range(2):
                    rows, picks = native.draft_read(cache, current, table, previous, window)
                    sparse.forward(query, rows, picks, sink)
                stream.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    rows, picks = native.draft_read(cache, current, table, previous, window)
                    actual = sparse.forward(query, rows, picks, sink)
                for context in (0, 255, 512, 33):
                    cache.copy_(torch.randn_like(cache))
                    current.copy_(torch.randn_like(current))
                    query.copy_(torch.randn_like(query))
                    sink.copy_(torch.randn_like(sink))
                    table.copy_(torch.tensor([[2, 4, -1, 1]] if context % 2 else [[3, -1, 1, 4]],
                                             device="musa", dtype=torch.int64))
                    previous.fill_(context)
                    saved_cache = cache.cpu()
                    expected_rows, expected_picks = _reference(
                        saved_cache, current.cpu(), table.cpu(), context, window)
                    legacy = native.Paged(cache, window, width ** -0.5,
                                          args.sparse_library, True, False)
                    expected = legacy.forward_draft_block(query, current, table, context, sink)
                    graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(rows.cpu(), expected_rows, atol=0, rtol=0)
                    torch.testing.assert_close(picks.cpu(), expected_picks, atol=0, rtol=0)
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
                    torch.testing.assert_close(cache.cpu(), saved_cache, atol=0, rtol=0)
                    cases.append({"kind": "real_graph_dynamic_inputs_exact", "q": q,
                                  "context": context, "cache_unchanged": True})
                del graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS_NONCAUSAL_DRAFT_READ_AND_MATE_GRAPH",
                                       "count": len(cases), "cases": cases,
                                       "scope": "Draft SWA rows/picks and actual MATE attention; no full-model accuracy claim"}, indent=2))
    logger.info("Draft read checks passed: %d", len(cases))


if __name__ == "__main__":
    main()
