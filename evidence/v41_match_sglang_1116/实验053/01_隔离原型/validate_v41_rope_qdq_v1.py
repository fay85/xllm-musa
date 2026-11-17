# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import torch_musa


def _time(call: object) -> dict:
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
    import v41_rope_qdq_test_module as native

    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111753)
    checks, graphs, timings, differences = [], [], [], []

    def save(status: str) -> None:
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "differences": differences,
            "module": native.__file__,
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "Isolated fused RoPE plus original block32 QDQ; no serving or model accuracy claim",
        }, indent=2))

    def check(actual: torch.Tensor, expected: torch.Tensor, context: dict) -> None:
        a, e = actual.cpu(), expected.cpu()
        assert a.shape == e.shape and a.dtype == e.dtype == torch.bfloat16
        same = torch.equal(a.view(torch.int16), e.view(torch.int16))
        if not same:
            differences.append(dict(context,
                                    mismatches=int((a.view(torch.int16) != e.view(torch.int16)).sum()),
                                    max_abs=float((a.float()-e.float()).abs().max())))
            save("BIT_DIFFERENCE")
        assert same, "Fused operator changed the RoPE-BF16-QDQ output"
        checks.append(dict(context, bit_equal=True))

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
                            stream = torch.musa.Stream() if dtype == torch.int32 else torch.musa.current_stream()
                            stream.wait_stream(torch.musa.current_stream())
                            with torch.musa.stream(stream):
                                expected = native.reference(x, positions, table, inverse)
                                actual = native.candidate(x, positions, table, inverse)
                            stream.synchronize()
                            torch.musa.current_stream().wait_stream(stream)
                            check(actual, expected, dict(rotary=rotary, heads=heads, rows=rows,
                                                         dtype=str(dtype), inverse=inverse))
                    save("CHECKING")
                    if rotary != 64 or rows not in (1, 4, 6):
                        continue
                    for inverse in (False, True):
                        stream = torch.musa.Stream()
                        stream.wait_stream(torch.musa.current_stream())
                        with torch.musa.stream(stream):
                            native.candidate(x, positions, table, inverse)
                            stream.synchronize()
                            graph = torch.musa.MUSAGraph()
                            with torch.musa.graph(graph, stream=stream):
                                actual = native.candidate(x, positions, table, inverse)
                            for offset, amplitude in ((255, 0.0), (511, 1.0e-5), (2047, 1.0), (4095, 100.0)):
                                x.normal_().mul_(amplitude)
                                positions.copy_((torch.arange(rows, device="musa", dtype=positions.dtype) + offset) % 4096)
                                expected = native.reference(x, positions, table, inverse)
                                graph.replay()
                                stream.synchronize()
                                context = dict(graph=True, heads=heads, rows=rows, inverse=inverse,
                                               offset=offset, amplitude=amplitude)
                                check(actual, expected, context)
                                graphs.append(dict(context, passed=True))
                        torch.musa.current_stream().wait_stream(stream)
                        control = lambda: native.reference(x, positions, table, inverse)
                        timings.append(dict(heads=heads, rows=rows, inverse=inverse,
                                            control=_time(control),
                                            fused=_time(lambda: native.candidate(x, positions, table, inverse)),
                                            reverse_control=_time(control)))
                        save("GRAPH_AND_TIMING")
        # Identity rotation isolates quantizer thresholds and signed zero.
        table = torch.cat((torch.ones(1, 32, device="musa"), torch.zeros(1, 32, device="musa")), 1)
        positions = torch.zeros(4, device="musa", dtype=torch.int32)
        values = torch.tensor([0.0, -0.0, 1e-5, -1e-5, 1e-4, -1e-4,
                               1.0, 1.0625, 1.1875, 1.9375, 224.0, 448.0, 480.0, -448.0, 0.5, -0.5],
                              device="musa", dtype=torch.bfloat16).repeat(128).reshape(4, 512)
        for inverse in (False, True):
            check(native.candidate(values, positions, table, inverse),
                  native.reference(values, positions, table, inverse), dict(edge_values=True, inverse=inverse))
    save("PASS_BIT_EQUAL_AND_DYNAMIC_GRAPH")


if __name__ == "__main__":
    _main()
