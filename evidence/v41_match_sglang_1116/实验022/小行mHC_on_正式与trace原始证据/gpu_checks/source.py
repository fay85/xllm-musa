# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the existing small-row MHC reduction in updated-input Graphs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch
import torch_musa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("XLLM_DSV41_MHC_SMALL_BATCH_FUSION") != "1":
        raise ValueError("Set the explicit small-row policy before importing")
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.set_num_threads(1)
    torch.manual_seed(1122)
    source = load_file(str(args.fixture), device="cpu")
    cases = []
    stream = torch.musa.Stream()
    with torch.inference_mode(), torch.musa.stream(stream):
        for part in ("attn", "ffn"):
            weights = [source[f"layers.0.hc_{part}_{term}"].to("musa")
                       for term in ("fn", "scale", "base")]
            reference = native.NativeV41HyperConnection(
                *weights, 20, 1e-6, 1e-6, str(args.library_directory), False
            )
            target = native.NativeV41HyperConnection(
                *weights, 20, 1e-6, 1e-6, str(args.library_directory), True
            )
            width = weights[0].shape[1] // 4
            for rows in (4, 6, 9):
                residual = torch.randn(rows, 4, width, device="musa",
                                       dtype=torch.bfloat16)
                carried = torch.rand(rows, 4, device="musa")
                value = torch.randn(rows, width, device="musa",
                                    dtype=torch.bfloat16)
                for _ in range(2):
                    metadata = target.prepare(residual, carried)
                    target.combine(value, residual, metadata[2], metadata[3])
                stream.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    actual = target.prepare(residual, carried)
                    combined = target.combine(value, residual, actual[2], actual[3])
                for trial in range(3):
                    residual.copy_(torch.randn_like(residual))
                    carried.copy_(torch.rand_like(carried))
                    value.copy_(torch.randn_like(value))
                    expected = reference.prepare(residual, carried)
                    expected_combined = reference.combine(
                        value, residual, expected[2], expected[3]
                    )
                    graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    for a, b in zip(actual[1:], expected[1:]):
                        torch.testing.assert_close(a, b, rtol=3e-4, atol=1e-4)
                    torch.testing.assert_close(combined, expected_combined,
                                               rtol=0.01, atol=0.02)
                    cases.append({
                        "part": part, "rows": rows, "trial": trial,
                        "metadata_max_abs": [float((a - b).abs().max().item())
                                             for a, b in zip(actual[1:], expected[1:])],
                        "combined_max_abs": float(
                            (combined - expected_combined).abs().max().item()
                        ),
                        "carried_pre_exact": True,
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "PASS_SMALL_BATCH_MHC_UPDATED_INPUT_GRAPH",
        "cases": cases,
        "scope": "Actual native MATE MHC preparation/combine; q4/q6 and q9 "
                 "normal-path boundary. No full-model precision claim.",
    }, indent=2))
    logger.info("MHC updated-input Graph passed %d cases", len(cases))


if __name__ == "__main__":
    main()
