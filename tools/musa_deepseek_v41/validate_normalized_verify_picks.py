# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare normalized picks against the full validation path in H8 attention."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_musa  # noqa: F401


FLAG = "XLLM_DSV41_VERIFY_NORMALIZED_PICKS"
WINDOW, WIDTH, BLOCK_ROWS, TOPK = 128, 512, 128, 512
DEVICE = "musa:0"
GEOMETRIES = ((1, 10752), (2, 10752), (4, 10752), (6, 10752), (6, 21504),
              (6, 3072), (6, 0))


def verify_library(path: Path, expected: str):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    assert digest.hexdigest() == expected


def tensors(queries, compressed_rows, dtype):
    cache = torch.randn(12, BLOCK_ROWS, WIDTH, dtype=torch.bfloat16).to(DEVICE)
    current = torch.randn(queries, WIDTH, dtype=torch.bfloat16).to(DEVICE)
    page_table = torch.tensor([[2, -1, 7, 9]], dtype=dtype, device=DEVICE)
    positions = torch.arange(20000, 20000 + queries, dtype=dtype, device=DEVICE)
    if compressed_rows:
        compressed = torch.randn(compressed_rows, WIDTH, dtype=torch.bfloat16).to(DEVICE)
        extra = torch.randint(compressed_rows, (queries, TOPK), dtype=dtype)
        extra[:, 1] = -1
        extra[:, 2] = compressed_rows
        extra[:, 3] = extra[:, 0]
        extra = extra.to(DEVICE)
    else:
        compressed, extra = None, None
    return cache, current, page_table, positions, WINDOW, compressed, extra


def read(native, args, selected):
    os.environ[FLAG] = "1" if selected else "0"
    return native.read(*args)


def output(native, readout, query, sink, library):
    rows, picks, _ = readout
    attention = native.attention_prepared if picks.dtype == torch.int32 else native.attention
    return attention(query, rows, picks, sink, WIDTH ** -0.5, library)


def validate_layout(old, new, expected_rows, prepared):
    old_rows, old_picks, old_slots = old
    new_rows, new_picks, new_slots = new
    assert new_rows.shape == (expected_rows, WIDTH)
    assert old_picks.dtype == torch.int64
    assert new_picks.dtype == (torch.int32 if prepared else torch.int64)
    assert torch.equal(old_slots, new_slots)
    assert torch.equal(old_picks, new_picks.long())
    assert torch.equal(old_rows, new_rows)


def combined_ms(native, args, selected, stream, query, sink, library):
    os.environ[FLAG] = "1" if selected else "0"
    measured = []
    for _ in range(5):
        begin = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        begin.record()
        for _ in range(16):
            output(native, native.read(*args), query, sink, library)
        end.record()
        stream.synchronize()
        measured.append(begin.elapsed_time(end) / 16)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--sparse-library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_library(args.sparse_library, args.library_sha256)
    assert torch.musa.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(11222026)
    module_name = "v41_verify_graph_inputs_test_module"
    spec = importlib.util.spec_from_file_location(module_name, args.module)
    assert spec is not None and spec.loader is not None
    native = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = native
    spec.loader.exec_module(native)

    os.environ["XLLM_DSV41_VERIFY_PACKED_COMPRESSED_READ"] = "1"
    checks, graphs, performance = [], [], []
    library = str(args.sparse_library)
    stream = torch.musa.Stream()
    with torch.inference_mode(), torch.musa.stream(stream):
        for queries, compressed_rows in GEOMETRIES:
            for dtype in (torch.int32, torch.int64):
                inputs = tensors(queries, compressed_rows, dtype)
                query = torch.randn(queries, 8, WIDTH, dtype=torch.bfloat16).to(DEVICE)
                sink = torch.randn(8, dtype=torch.float32).to(DEVICE)
                old = read(native, inputs, False)
                new = read(native, inputs, True)
                packed = queries > 1 and compressed_rows > queries * TOPK
                expected_rows = WINDOW + queries + (queries * TOPK if packed else compressed_rows)
                validate_layout(old, new, expected_rows, packed)
                old_attention = output(native, old, query, sink, library)
                new_attention = output(native, new, query, sink, library)
                assert old_attention.isfinite().all() and torch.equal(old_attention, new_attention)
                checks.append(dict(q=queries, source_rows=compressed_rows, dtype=str(dtype),
                                   materialized_rows=expected_rows, packed=packed, normalized_picks=packed,
                                   visible_kv_and_mask_exact=True, attention_exact=True,
                                   commit_slots_exact=True))
                if queries != 6 or compressed_rows not in (10752, 21504) or dtype != torch.int64:
                    continue
                captures, values = [], []
                for selected in (False, True):
                    read(native, inputs, selected)
                    stream.synchronize()
                    graph = torch.musa.MUSAGraph()
                    with torch.musa.graph(graph, stream=stream):
                        captured = read(native, inputs, selected)
                        result = output(native, captured, query, sink, library)
                    captures.append(graph)
                    values.append((captured, result))
                for start in (19967, 20095, 21020):
                    inputs[3].copy_(torch.arange(start, start + queries, dtype=dtype, device=DEVICE))
                    inputs[6].copy_(inputs[6].roll(23, dims=1))
                    inputs[1].mul_(-1)
                    for graph in captures:
                        graph.replay()
                    stream.synchronize()
                    validate_layout(values[0][0], values[1][0], expected_rows, packed)
                    assert torch.equal(values[0][1], values[1][1])
                    graphs.append(dict(q=queries, source_rows=compressed_rows, position=start,
                                       bits_exact=True))
                off = combined_ms(native, inputs, False, stream, query, sink, library)
                on = combined_ms(native, inputs, True, stream, query, sink, library)
                performance.append(dict(q=6, source_rows=compressed_rows, off_ms=off, on_ms=on,
                                        off_median_ms=statistics.median(off),
                                        on_median_ms=statistics.median(on)))
        stream.synchronize()

    with args.output.open("x") as report:
        json.dump(dict(status="PASS_NORMALIZED_VERIFY_PICKS_SCOPE", checks=checks, graphs=graphs,
                       performance=performance, sparse_library_sha256=args.library_sha256), report, indent=2)


if __name__ == "__main__":
    main()
