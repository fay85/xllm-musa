# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify native single-token SWA commits, stream order and ring-wrap semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import statistics
import time
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _time_ms(function: Callable[[], torch.Tensor]) -> float:
    for _ in range(3):
        function()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 20


def _run(args: argparse.Namespace) -> dict:
    import musa_python_kernel_test_module as native

    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    torch.manual_seed(111212)
    records: list[dict] = []
    device = torch.device("musa:0")
    for dimension in (128, 512):
        for other_stream in (False, True):
            stream = (
                torch.musa.Stream() if other_stream else torch.musa.current_stream()
            )
            for slot_id in (-65, -1, 0, 17, 47, 48, 100):
                initial = torch.randn(3, 16, dimension, dtype=torch.bfloat16)
                source = torch.randn(1, dimension, dtype=torch.bfloat16)
                expected = initial.clone()
                produced = (source.float() * 1.25 + 0.125).to(torch.bfloat16)
                if 0 <= slot_id < 48:
                    expected.view(-1, dimension)[slot_id] = produced[0]
                with torch.musa.stream(stream):
                    cache = initial.to(device)
                    # GPU producer and consumer must be ordered around the native
                    # launch on this stream, without a device synchronize.
                    value = (source.to(device).float() * 1.25 + 0.125).to(
                        torch.bfloat16
                    )
                    slot = torch.tensor([slot_id], dtype=torch.int64, device=device)
                    native.deepseek_v4_scatter_single_cache_native(cache, slot, value)
                    consumed = cache.clone()
                    result = consumed.cpu()
                assert torch.equal(result, expected), (dimension, other_stream, slot_id)
                records.append(
                    {
                        "kind": "raw_scatter",
                        "dimension": dimension,
                        "nondefault_stream": other_stream,
                        "slot": slot_id,
                        "cache_bitwise_exact": True,
                    }
                )
    for table_values in ([2, 0], [-1, 1]):
        for other_stream in (False, True):
            stream = (
                torch.musa.Stream() if other_stream else torch.musa.current_stream()
            )
            with torch.musa.stream(stream):
                dimension, window, block_size = 512, 32, 16
                cache_cpu = torch.randn(3, block_size, dimension, dtype=torch.bfloat16)
                cache = cache_cpu.to(device)
                reference_cache = cache.clone()
                table = torch.tensor([table_values], dtype=torch.int32, device=device)
                fast = native.NativeV41PagedAttention(
                    cache, window, dimension**-0.5, str(args.sparse_library), True
                )
                original = native.NativeV41PagedAttention(
                    reference_cache,
                    window,
                    dimension**-0.5,
                    str(args.sparse_library),
                    False,
                )
                sink = torch.randn(8, dtype=torch.float32, device=device)
                context = 0
                # Multi-row chunks exceed the ring: original stable latest-wins
                # ordering must remain active, including a prefill/decode switch.
                for count in (0, 17, 1, 1, 65, 1, 1, 1):
                    query = torch.randn(
                        count, 8, dimension, device=device, dtype=torch.bfloat16
                    )
                    values = torch.randn(
                        count, dimension, device=device, dtype=torch.bfloat16
                    )
                    value_cpu = values.cpu()
                    context += count
                    actual = fast.forward(
                        query, values, table, context, sink, None, None
                    )
                    expected = original.forward(
                        query, values, table, context, sink, None, None
                    )
                    actual_cpu, expected_cpu = actual.cpu(), expected.cpu()
                    for row in range(count):
                        position = context - count + row
                        page = table_values[
                            (position // block_size) % len(table_values)
                        ]
                        slot = page * block_size + position % block_size
                        if 0 <= slot < cache_cpu.numel() // dimension:
                            cache_cpu.view(-1, dimension)[slot] = value_cpu[row]
                    assert torch.equal(actual_cpu, expected_cpu), (
                        table_values,
                        context,
                    )
                    assert torch.equal(cache.cpu(), cache_cpu), (table_values, context)
                    assert torch.equal(reference_cache.cpu(), cache_cpu)
                    records.append(
                        {
                            "kind": "complete_paged_attention",
                            "rows": count,
                            "context": context,
                            "table": table_values,
                            "nondefault_stream": other_stream,
                            "output_and_cache_bitwise_exact": True,
                        }
                    )
    with torch.musa.stream(torch.musa.default_stream()):
        cache = torch.randn(3, 16, 512, device=device, dtype=torch.bfloat16)
        table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
        query = torch.randn(1, 8, 512, device=device, dtype=torch.bfloat16)
        value = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
        sink = torch.randn(8, device=device)
        original = native.NativeV41PagedAttention(
            cache.clone(), 32, 512**-0.5, str(args.sparse_library), False
        )
        fast = native.NativeV41PagedAttention(
            cache.clone(), 32, 512**-0.5, str(args.sparse_library), True
        )
        samples = []
        for round_id in range(5):
            pair = {}
            for label in (
                ("original", "single_scatter")
                if round_id % 2 == 0
                else ("single_scatter", "original")
            ):
                target = original if label == "original" else fast
                time.sleep(0.1)
                before = Path("/sys/fs/cgroup/cpu.stat")
                stat_before = before.read_text() if before.exists() else None
                pair[label] = _time_ms(
                    lambda: target.forward(query, value, table, 47, sink, None, None)
                )
                stat_after = before.read_text() if before.exists() else None
                pair[label + "_cpu_stat"] = {"before": stat_before, "after": stat_after}
            samples.append(pair)
        old_ms = statistics.median(x["original"] for x in samples)
        new_ms = statistics.median(x["single_scatter"] for x in samples)
    return {
        "status": "PASS_NATIVE_SINGLE_SWA_STREAM_ORDER_AND_RING_SEMANTICS",
        "device": torch.musa.get_device_name(0),
        "cases": records,
        "timing_scope": "complete M1 paged attention, GPU events, 3 warmup/20 repeats",
        "alternating_timing_samples": samples,
        "old_complete_ms": old_ms,
        "new_complete_ms": new_ms,
        "complete_speedup": old_ms / new_ms,
        "boundary": "microbenchmark is not full-model TTFT/TPOT; prefill path unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-library", type=Path, required=True)
    args = parser.parse_args()
    if not args.sparse_library.is_file():
        raise ValueError("Require the already validated safe MATE sparse library")
    with torch.inference_mode():
        result = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info(
        "Single-token SWA regression passed %d cases; complete M1 speedup %.3f",
        len(result["cases"]),
        result["complete_speedup"],
    )


if __name__ == "__main__":
    main()
