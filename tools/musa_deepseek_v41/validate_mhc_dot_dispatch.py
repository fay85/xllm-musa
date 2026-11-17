# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify native model-layer selection and the q9 MATE boundary in Graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--mate-directory", required=True)
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    import v41_mhc_dot_test_module as native

    torch.manual_seed(1146)
    torch.set_num_threads(1)
    fixture = load_file(args.fixture, device="cpu")
    checks = []
    stream = torch.musa.Stream()
    with torch.inference_mode(), torch.musa.stream(stream):
        for part in ("attn", "ffn"):
            weights = [fixture[f"layers.0.hc_{part}_{term}"].to("musa", dtype=torch.float32)
                       for term in ("fn", "scale", "base")]
            for rows in (4, 9, 1, 6, 8):
                residual = torch.randn(rows, 4, 5120, device="musa", dtype=torch.bfloat16)
                carried = torch.rand(rows, 4, device="musa")

                def integrated() -> tuple:
                    return native.integrated(residual, carried, weights,
                                             args.mate_directory, args.candidate_directory)

                for _ in range(2):
                    integrated()
                stream.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    actual = integrated()
                for trial in range(3):
                    residual.normal_()
                    carried.uniform_()
                    if rows <= 8:
                        _, pre, post, comb = native.forward(
                            residual.flatten(1), *weights, args.candidate_directory)
                        expected = (native.collapse(residual, carried, None), pre, post, comb)
                    else:
                        expected = native.control(residual, carried, weights, args.mate_directory)
                    graph.replay()
                    stream.synchronize()
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    checks.append({"part": part, "rows": rows, "trial": trial,
                                   "route": "dot160" if rows <= 8 else "original_mate",
                                   "all_outputs_exact": True})
    args.output.write_text(json.dumps({"status": "PASS_NATIVE_MODEL_LAYER_DISPATCH",
                                      "checks": checks}, indent=2) + "\n")
    logger.info("Passed %d native layer dispatch/Graph cases", len(checks))


if __name__ == "__main__":
    main()
