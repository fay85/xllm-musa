"""Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0.

Bounded real-checkpoint Gate projection checks and device microbenchmarks.
This is an isolated operator experiment, not model accuracy acceptance.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch_musa
from safetensors.torch import load_file


def _graph_time(call: Callable) -> dict:
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
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record(stream)
            for _ in range(3):
                graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 90)
    torch.musa.current_stream().wait_stream(stream)
    assert outputs
    return {"us_per_call": samples, "median_us": statistics.median(samples),
            "scope": "30 calls per graph x 3 replays, 5 samples; not E2E"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import v41_gate_gemv_test_module as native

    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111750)
    tensors = load_file(str(args.fixture), device="cpu")
    config = json.loads(args.fixture.with_suffix(".json").read_text())["config"]["text_config"]
    checks, graphs, timings, route_differences = [], [], [], []

    def save(status: str) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "route_differences": route_differences,
            "module": native.__file__,
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "scope": "Real BF16 gate weights, generated inputs, FP32 output; operator only",
            "accuracy_status": "仅通过基础功能检查，未完成模型精度验收",
        }, ensure_ascii=False, indent=2))

    with torch.inference_mode():
        for part, prefix, topk in [
            ("target", "layers.0.ffn.gate", config["num_experts_per_tok"]),
            ("draft", "mtp.0.ffn.gate", config["dspark_num_experts_per_tok"]),
        ]:
            weight = tensors[prefix + ".weight"].to("musa")
            assert weight.dtype == torch.bfloat16
            fp32_weight = weight.float()
            bias = tensors[prefix + ".bias"].to("musa")
            for dtype in (torch.bfloat16, torch.float32):
                for rows in (0, 1, 3, 4, 6):
                    x = torch.randn(rows, weight.shape[1], device="musa", dtype=dtype)
                    control = lambda: native.reference(x, fp32_weight)
                    candidate = lambda: native.gemv(x, weight)

                    def route(logits: torch.Tensor) -> tuple:
                        return native.routing(logits, bias, topk,
                                              config["norm_topk_prob"],
                                              config["routed_scaling_factor"])

                    for magnitude in (0.0, 1e-8, 1.0, 10.0):
                        for nondefault in (False, True):
                            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
                            stream.wait_stream(torch.musa.current_stream())
                            with torch.musa.stream(stream):
                                x.copy_(torch.randn_like(x) * magnitude)
                                before = x.clone()
                                expected, actual = control(), candidate()
                                expected_route, actual_route = route(expected), route(actual)
                            stream.synchronize()
                            torch.musa.current_stream().wait_stream(stream)
                            torch.testing.assert_close(before, x, rtol=0, atol=0)
                            assert actual.dtype == torch.float32 and actual.shape == expected.shape
                            max_abs, relative_rms = 0.0, 0.0
                            if actual.numel():
                                delta = actual - expected
                                max_abs = delta.abs().max().item()
                                relative_rms = delta.square().mean().sqrt().item() / max(
                                    expected.square().mean().sqrt().item(), 1e-30)
                                normalized_max = max_abs / max(expected.abs().max().item(), 1e-30)
                                assert relative_rms < 3e-5 and normalized_max < 5e-5
                            same_ids = torch.equal(actual_route[1], expected_route[1])
                            item = {"part": part, "dtype": str(dtype), "rows": rows,
                                    "magnitude": magnitude, "nondefault": nondefault,
                                    "max_abs": max_abs, "relative_rms": relative_rms,
                                    "route_ids_equal": same_ids, "input_unchanged": True}
                            checks.append(item)
                            if not same_ids:
                                route_differences.append(dict(item,
                                    expected_ids=expected_route[1].cpu().tolist(),
                                    actual_ids=actual_route[1].cpu().tolist()))
                            else:
                                torch.testing.assert_close(actual_route[0], expected_route[0],
                                                           rtol=5e-5, atol=1e-6)
                    save("NUMERICAL_CHECKING")
                    if not rows:
                        continue
                    stream = torch.musa.Stream()
                    stream.wait_stream(torch.musa.current_stream())
                    with torch.musa.stream(stream):
                        candidate()
                        stream.synchronize()
                        graph = torch.musa.MUSAGraph()
                        with torch.musa.graph(graph, stream=stream):
                            actual = candidate()
                            actual_route = route(actual)
                        for trial in range(3):
                            x.normal_()
                            expected = candidate()
                            expected_route = route(expected)
                            graph.replay()
                            stream.synchronize()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(actual_route[0], expected_route[0], rtol=0, atol=0)
                            torch.testing.assert_close(actual_route[1], expected_route[1], rtol=0, atol=0)
                            graphs.append({"part": part, "dtype": str(dtype), "rows": rows,
                                           "trial": trial, "exact": True})
                    torch.musa.current_stream().wait_stream(stream)
                    if dtype == torch.bfloat16:
                        for include_route in (False, True):
                            before_call = (lambda: route(control())) if include_route else control
                            after_call = (lambda: route(candidate())) if include_route else candidate
                            timings.append({"part": part, "rows": rows,
                                            "include_route": include_route,
                                            "control": _graph_time(before_call),
                                            "candidate": _graph_time(after_call),
                                            "control_reverse": _graph_time(before_call)})
                    save("GRAPH_AND_TIMING")
    save("PASS_NATIVE_GATE_GEMV_AND_GRAPH")


if __name__ == "__main__":
    main()
