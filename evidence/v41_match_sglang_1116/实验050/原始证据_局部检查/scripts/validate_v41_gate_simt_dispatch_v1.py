"""Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0.

Check the production Gate class, including declared unsupported shapes/dtypes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_musa
from safetensors.torch import load_file
from validate_v41_gate_gemv_v1 import _graph_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import v41_gate_gemv_test_module as native
    assert torch.musa.device_count() == 1
    torch.manual_seed(1117503)
    torch.set_num_threads(1)
    tensors = load_file(str(args.fixture), device="cpu")
    checks, graphs, timings = [], [], []

    def save(status: str) -> None:
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "module": native.__file__,
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "Production DeepseekV41Gate class, not a model accuracy test",
        }, indent=2))

    with torch.inference_mode():
        for prefix, topk in [("layers.0.ffn.gate", 6), ("mtp.0.ffn.gate", 3)]:
            bias = tensors[prefix + ".bias"].to("musa")
            for weight_dtype in (torch.bfloat16, torch.float32):
                weight = tensors[prefix + ".weight"].to("musa", dtype=weight_dtype)
                for cached in (False, True):
                    control = native.Gate(weight, bias, topk, True, 1.5, False, cached)
                    candidate = native.Gate(weight, bias, topk, True, 1.5, True, cached)
                    for rows in (0, 1, 4, 6, 7, 16):
                        for input_dtype in (torch.bfloat16, torch.float32):
                            x = torch.randn(rows, 5120, device="musa", dtype=input_dtype)
                            stream = torch.musa.Stream()
                            stream.wait_stream(torch.musa.current_stream())
                            with torch.musa.stream(stream):
                                expected = control.forward(x)
                                actual = candidate.forward(x)
                            stream.synchronize()
                            torch.musa.current_stream().wait_stream(stream)
                            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                            torch.testing.assert_close(actual[0], expected[0], rtol=5e-5, atol=1e-6)
                            use_candidate = weight_dtype == torch.bfloat16 and rows <= 6
                            if not use_candidate:
                                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                            checks.append({"prefix": prefix, "weight_dtype": str(weight_dtype),
                                           "input_dtype": str(input_dtype), "cache_fp32": cached,
                                           "rows": rows, "candidate_eligible": use_candidate,
                                           "route_ids_equal": True})
                    save("DISPATCH_CHECKING")
                    if weight_dtype != torch.bfloat16 or not cached:
                        continue
                    for rows in (1, 4, 7):
                        x = torch.randn(rows, 5120, device="musa", dtype=torch.bfloat16)
                        stream = torch.musa.Stream()
                        stream.wait_stream(torch.musa.current_stream())
                        with torch.musa.stream(stream):
                            candidate.forward(x)
                            stream.synchronize()
                            graph = torch.musa.MUSAGraph()
                            with torch.musa.graph(graph, stream=stream):
                                actual = candidate.forward(x)
                            for trial in range(3):
                                x.normal_()
                                expected = candidate.forward(x)
                                graph.replay()
                                stream.synchronize()
                                for a, e in zip(actual, expected):
                                    torch.testing.assert_close(a, e, rtol=0, atol=0)
                                graphs.append({"prefix": prefix, "rows": rows,
                                               "trial": trial, "exact": True})
                        torch.musa.current_stream().wait_stream(stream)
                    x = torch.randn(4, 5120, device="musa", dtype=torch.bfloat16)
                    timings.append({"prefix": prefix, "rows": 4,
                                    "control": _graph_time(lambda: control.forward(x)),
                                    "candidate": _graph_time(lambda: candidate.forward(x)),
                                    "control_reverse": _graph_time(lambda: control.forward(x))})
                    save("PRODUCTION_GRAPH_CHECKING")
    save("PASS_PRODUCTION_GATE_DISPATCH_AND_GRAPH")


if __name__ == "__main__":
    main()
