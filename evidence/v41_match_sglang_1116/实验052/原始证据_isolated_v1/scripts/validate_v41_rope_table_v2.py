"""Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0.

Native partial RoPE reference versus direct FP32 table lookup. Operator only.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch_musa


def _time(call: Callable) -> dict:
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(5):
            call()
        stream.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            outputs = [call() for _ in range(30)]
        graph.replay()
        stream.synchronize()
        samples = []
        for _ in range(5):
            begin = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            begin.record(stream)
            for _ in range(3):
                graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / 90)
    torch.musa.current_stream().wait_stream(stream)
    assert outputs
    return {"us_per_call": samples, "median_us": statistics.median(samples)}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import v41_rope_table_test_module as native

    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111752)
    checks, graphs, timings, differences = [], [], [], []

    def save(status: str) -> None:
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "differences": differences,
            "module": native.__file__,
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "Native operator and dynamic Graph checks; no model accuracy claim",
        }, indent=2))

    def check(actual: torch.Tensor, expected: torch.Tensor, context: dict) -> None:
        a, e = actual.cpu(), expected.cpu()
        assert a.shape == e.shape and a.dtype == e.dtype == torch.bfloat16
        exact = torch.equal(a.view(torch.int16), e.view(torch.int16))
        if not exact:
            finite = torch.isfinite(a) & torch.isfinite(e)
            item = dict(context, mismatch=int((a.view(torch.int16) != e.view(torch.int16)).sum()),
                        max_finite_abs=float((a.float()[finite] - e.float()[finite]).abs().max()) if finite.any() else 0.0)
            differences.append(item)
            save("NUMERICAL_DIFFERENCE")
            assert torch.equal(torch.isnan(a), torch.isnan(e))
            assert torch.equal(torch.isposinf(a), torch.isposinf(e))
            assert torch.equal(torch.isneginf(a), torch.isneginf(e))
            distances = (a.view(torch.int16).to(torch.int32) - e.view(torch.int16).to(torch.int32)).abs()
            both_zero = (a == 0) & (e == 0)
            max_ulp = int(distances[finite & ~both_zero].max()) if (finite & ~both_zero).any() else 0
            differences[-1]["max_bf16_ulp"] = max_ulp
            save("BOUNDED_ROUNDING_DIFFERENCE")
            assert max_ulp <= 1, "More than one BF16 output step is not a rounding-only basic check"
        checks.append(dict(context, exact_bits=exact))

    with torch.inference_mode():
        for rotary in (64, 128):
            angle = torch.randn(4096, rotary // 2, device="musa", dtype=torch.float32)
            table = torch.cat((angle.cos(), angle.sin()), dim=1).contiguous()
            for heads in (1, 8, 64):
                for rows in (0, 1, 4, 6, 256):
                    shape = (rows, 512) if heads == 1 else (rows, heads, 512)
                    x = torch.randn(shape, device="musa", dtype=torch.bfloat16)
                    for dtype in (torch.int32, torch.int64):
                        positions = (torch.arange(rows, device="musa", dtype=dtype) * 127) % 4096
                        for inverse in (False, True):
                            for block in (128, 256):
                                stream = torch.musa.Stream() if block == 128 else torch.musa.current_stream()
                                stream.wait_stream(torch.musa.current_stream())
                                with torch.musa.stream(stream):
                                    expected = native.reference(x, positions, table, inverse)
                                    actual = native.candidate(x, positions, table, inverse, block)
                                stream.synchronize()
                                torch.musa.current_stream().wait_stream(stream)
                                context = dict(rotary=rotary, heads=heads, rows=rows,
                                               dtype=str(dtype), inverse=inverse, block=block)
                                check(actual, expected, context)
                    save("CHECKING")
                    if rotary != 64 or rows not in (1, 4, 6):
                        continue
                    # Actual rotary=64; mutate positions and values with stable addresses.
                    for inverse in (False, True):
                        for block in (128, 256):
                            stream = torch.musa.Stream()
                            stream.wait_stream(torch.musa.current_stream())
                            with torch.musa.stream(stream):
                                native.candidate(x, positions, table, inverse, block)
                                stream.synchronize()
                                graph = torch.musa.MUSAGraph()
                                with torch.musa.graph(graph, stream=stream):
                                    actual = native.candidate(x, positions, table, inverse, block)
                                for offset in (255, 511, 2047, 4095):
                                    x.normal_()
                                    positions.copy_((torch.arange(rows, device="musa", dtype=positions.dtype) + offset) % 4096)
                                    expected = native.reference(x, positions, table, inverse)
                                    graph.replay()
                                    stream.synchronize()
                                    check(actual, expected, dict(graph=True, heads=heads, rows=rows,
                                                                 inverse=inverse, block=block, offset=offset))
                                    graphs.append(dict(heads=heads, rows=rows, inverse=inverse,
                                                       block=block, offset=offset, passed=True))
                            torch.musa.current_stream().wait_stream(stream)
                        control = lambda: native.reference(x, positions, table, inverse)
                        timings.append(dict(heads=heads, rows=rows, inverse=inverse,
                                            control=_time(control),
                                            block128=_time(lambda: native.candidate(x, positions, table, inverse, 128)),
                                            block256=_time(lambda: native.candidate(x, positions, table, inverse, 256)),
                                            reverse_control=_time(control)))
                        save("GRAPH_AND_TIMING")
    save("PASS_BASIC_ROPE_TABLE_ONE_BF16_ULP_AND_DYNAMIC_GRAPH")


if __name__ == "__main__":
    _main()
