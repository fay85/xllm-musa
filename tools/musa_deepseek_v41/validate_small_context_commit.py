"""Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0.

Validate the production small-row scatter against sequential latest-wins
semantics and the previous stable-sort implementation. No model accuracy claim.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch_musa


def _reference(cache: torch.Tensor, slots: torch.Tensor,
               values: torch.Tensor) -> None:
    order = torch.argsort(slots, stable=True)
    sorted_slots = slots.index_select(0, order)
    keep = (sorted_slots >= 0) & (sorted_slots < cache.shape[0] * cache.shape[1])
    if slots.numel() > 1:
        keep[:-1] &= sorted_slots[:-1] != sorted_slots[1:]
    latest = order[keep]
    cache.view(-1, cache.shape[-1]).index_copy_(
        0, slots.index_select(0, latest).long(), values.index_select(0, latest))


def _expected(cache: torch.Tensor, slots: torch.Tensor,
              values: torch.Tensor) -> torch.Tensor:
    result = cache.cpu().view(torch.int16).clone()
    flat = result.view(-1, result.shape[-1])
    for row, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < flat.shape[0]:
            flat[slot].copy_(values.cpu().view(torch.int16)[row])
    return result


def _measure(call: Callable[[], object]) -> dict:
    for _ in range(5):
        call()
    torch.musa.synchronize()
    samples = []
    for _ in range(5):
        begin = time.perf_counter()
        for _ in range(100):
            call()
        torch.musa.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6 / 100)
    return {"host_wall_us_per_call": samples,
            "median_host_wall_us": statistics.median(samples)}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import v41_small_context_commit_test_module as native

    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111806)
    checks, graphs, timings = [], [], []
    stream = torch.musa.Stream()

    def save(status: str) -> None:
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "module": native.__file__,
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "BF16 bitwise cache semantics and dynamic Graph; not full model accuracy",
        }, indent=2))

    with torch.inference_mode():
        for width in (1, 17, 512, 576):
            for rows in range(7):
                for dtype in (torch.int32, torch.int64):
                    for pattern in ("unique", "duplicates", "invalid", "mixed"):
                        seed = torch.randint(-32768, 32768, (3, 8, width), dtype=torch.int16)
                        value_bits = torch.randint(-32768, 32768, (rows, width), dtype=torch.int16)
                        # Raw BF16 bits exercise NaN/Inf/signed-zero payload copies as well.
                        if pattern == "unique":
                            slot_values = [(i * 7 + 1) % 24 for i in range(rows)]
                        elif pattern == "duplicates":
                            slot_values = [3] * rows
                        elif pattern == "invalid":
                            slot_values = [-1 if i % 2 else 24 for i in range(rows)]
                        else:
                            slot_values = [23, 0, 23, -1, 24, 0][:rows]
                        slots = torch.tensor(slot_values, dtype=dtype, device="musa")
                        values = value_bits.view(torch.bfloat16).to("musa")
                        actual = seed.view(torch.bfloat16).to("musa")
                        reference = actual.clone()
                        expected = _expected(actual, slots, values)
                        stream.wait_stream(torch.musa.current_stream())
                        chosen = stream if dtype == torch.int32 else torch.musa.current_stream()
                        with torch.musa.stream(chosen):
                            native.scatter(actual, slots, values)
                            _reference(reference, slots, values)
                        chosen.synchronize()
                        assert torch.equal(actual.cpu().view(torch.int16), expected)
                        assert torch.equal(reference.cpu().view(torch.int16), expected)
                        checks.append(dict(width=width, rows=rows, dtype=str(dtype),
                                           pattern=pattern, full_cache_bits_equal=True))
                    save("CHECKING")

        for rows in range(1, 7):
            for dtype in (torch.int32, torch.int64):
                cache = torch.randn(3, 8, 512, dtype=torch.bfloat16, device="musa")
                values = torch.randn(rows, 512, dtype=torch.bfloat16, device="musa")
                slots = torch.arange(rows, dtype=dtype, device="musa")
                stream.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(stream):
                    native.scatter(cache, slots, values)
                    stream.synchronize()
                    graph = torch.musa.MUSAGraph()
                    with torch.musa.graph(graph, stream=stream):
                        native.scatter(cache, slots, values)
                    for offset in (7, 23, -2, 25):
                        cache.normal_()
                        values.normal_()
                        # Duplicate destinations and invalid slots change at replay time.
                        slots.copy_(torch.tensor([(offset + i // 2) for i in range(rows)],
                                                 dtype=dtype, device="musa"))
                        stream.synchronize()
                        expected = _expected(cache, slots, values)
                        graph.replay()
                        stream.synchronize()
                        assert torch.equal(cache.cpu().view(torch.int16), expected)
                        graphs.append(dict(rows=rows, dtype=str(dtype), offset=offset,
                                           full_cache_bits_equal=True))
                torch.musa.current_stream().wait_stream(stream)
        save("CHECKS_PASS")

        for rows in (2, 4, 6):
            slots = torch.tensor([i // 2 for i in range(rows)], device="musa", dtype=torch.int64)
            values = torch.randn(rows, 512, device="musa", dtype=torch.bfloat16)
            cache = torch.zeros(3, 8, 512, device="musa", dtype=torch.bfloat16)
            for iteration in range(3):
                for label in ("reference", "candidate"):
                    call = (lambda: _reference(cache, slots, values)) if label == "reference" else (
                        lambda: native.scatter(cache, slots, values))
                    timings.append(dict(rows=rows, iteration=iteration, path=label,
                                        **_measure(call)))
        save("PASS_BITS_DYNAMIC_GRAPH_AND_EAGER_MICRO")


if __name__ == "__main__":
    _main()
