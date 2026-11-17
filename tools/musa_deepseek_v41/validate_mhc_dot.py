# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native SG dot160/RMS/Sinkhorn against real-weight mHC preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if not torch.isfinite(actual).all().item():
        raise AssertionError("Nonfinite candidate output")
    diff = actual.float() - expected.float()
    return {"max_abs": diff.abs().max().item(),
            "relative_rms": (diff.square().mean() /
                             expected.float().square().mean().clamp_min(1e-30)).sqrt().item()}


def _graph_time(function: Callable, nodes: int = 40) -> dict:
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            function()
        stream.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            for _ in range(nodes):
                output = function()
        for _ in range(3):
            graph.replay()
        samples = []
        for _ in range(5):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / (nodes * 3))
    return {"samples_us": samples, "median_us": statistics.median(samples),
            "nodes": nodes, "output_count": len(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--mate-directory", required=True)
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_config = json.loads(args.model_config.read_text())
    config = raw_config.get("text_config", raw_config)
    norm_eps = config["rms_norm_eps"]
    assert norm_eps == 1e-20 and config["hc_eps"] == 1e-6
    manifest = json.loads((Path(args.candidate_directory) / "manifest.json").read_text())
    assert manifest["reduce_constants"]["EPS"] == norm_eps
    assert manifest["reduce_constants"]["HC_EPS"] == config["hc_eps"]
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    import v41_mhc_dot_test_module as native

    torch.manual_seed(111746)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = load_file(args.fixture, device="cpu")
    checks, graphs, timings = [], [], []

    def save(status: str) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": status, "checks": checks, "graphs": graphs,
            "timings": timings, "module": native.__file__,
            "rms_norm_eps": norm_eps, "hc_eps": config["hc_eps"],
            "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "Real layer0 attn/ffn parameters, native launch, carried_pre, "
                     "changed-input Graph; operator-only, no model accuracy or global speedup claim",
        }, indent=2) + "\n")

    with torch.inference_mode():
        for part in ("attn", "ffn"):
            weights = [source[f"layers.0.hc_{part}_{term}"].to("musa", dtype=torch.float32)
                       for term in ("fn", "scale", "base")]
            for rows in (1, 4, 6, 8):
                residual = torch.randn(rows, 4, 5120, device="musa", dtype=torch.bfloat16)
                carried = torch.rand(rows, 4, device="musa")
                value = torch.randn(rows, 5120, device="musa", dtype=torch.bfloat16)

                def control() -> tuple:
                    return native.control(residual, carried, weights, args.mate_directory)

                def candidate() -> tuple:
                    mix, pre, post, comb = native.forward(
                        residual.flatten(1), *weights, args.candidate_directory)
                    collapsed = native.collapse(residual, carried, None)
                    return collapsed, pre, post, comb, mix

                for magnitude in (0.0, 1e-10, 1e-6, 1.0, 10.0):
                    for nondefault in (False, True):
                        stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
                        stream.wait_stream(torch.musa.current_stream())
                        with torch.musa.stream(stream):
                            residual.copy_(torch.randn_like(residual) * magnitude)
                            carried.uniform_()
                            expected = control()
                            actual = candidate()
                            flat = residual.flatten(1).float()
                            oracle = (flat @ weights[0].T) * torch.rsqrt(
                                flat.square().mean(-1, keepdim=True) + norm_eps)
                            combined = native.combine(value, residual, actual[2], actual[3], None)
                            reference_combined = native.combine(value, residual, expected[2], expected[3], None)
                        stream.synchronize()
                        torch.musa.current_stream().wait_stream(stream)
                        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                        mix_error = _error(actual[4], oracle)
                        metadata_errors = [_error(a, b) for a, b in zip(actual[1:4], expected[1:])]
                        merged_error = _error(combined, reference_combined)
                        assert mix_error["relative_rms"] < .002, mix_error
                        assert all(x["relative_rms"] < .01 for x in metadata_errors), metadata_errors
                        assert merged_error["relative_rms"] < .01, merged_error
                        checks.append({"part": part, "rows": rows, "magnitude": magnitude,
                                       "nondefault_stream": nondefault, "mix_error": mix_error,
                                       "metadata_errors": metadata_errors, "merged_error": merged_error,
                                       "carried_pre_exact": True})
                        save("CHECKING")
                stream = torch.musa.Stream()
                stream.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(stream):
                    candidate()
                    stream.synchronize()
                    graph = torch.musa.MUSAGraph()
                    with torch.musa.graph(graph, stream=stream):
                        actual = candidate()
                    for trial in range(3):
                        residual.normal_()
                        carried.uniform_()
                        expected = candidate()
                        graph.replay()
                        stream.synchronize()
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        graphs.append({"part": part, "rows": rows, "trial": trial,
                                       "all_outputs_exact": True})
                        save("GRAPH_CHECKING")
                torch.musa.current_stream().wait_stream(stream)
                record = {"part": part, "rows": rows, "control": _graph_time(control),
                          "candidate": _graph_time(candidate), "control_reverse": _graph_time(control)}
                timings.append(record)
                save("TIMING")
                logger.info("%s M=%d control %.3f -> dot160 %.3f -> control %.3f us",
                            part, rows, record["control"]["median_us"],
                            record["candidate"]["median_us"], record["control_reverse"]["median_us"])
    save("PASS_NATIVE_MHC_DOT_AND_GRAPH")
    logger.info("Passed %d numerical and %d changed-input Graph cases", len(checks), len(graphs))


if __name__ == "__main__":
    main()
