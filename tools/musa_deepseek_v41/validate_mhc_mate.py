# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight mHC prenorm integration, shifted-pre and local GPU timing.

The reference is FP32 arithmetic, plus the previously checked metadata and
residual kernels. This is a local operator/layer test, not full-model PD
accuracy or serving performance. No CPU forward fallback is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
from safetensors.torch import load_file
from scripts.logger import logger
from tools.musa_deepseek_v41.validate_w4a8_experts import _error, _time_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("musa:0")
    source = load_file(str(args.fixture), device="cpu")
    results = []
    with torch.inference_mode():
        for part in ("attn", "ffn"):
            fn, scale, base = [
                source[f"layers.0.hc_{part}_{term}"].to(
                    device=device, dtype=torch.float32
                )
                for term in ("fn", "scale", "base")
            ]
            hidden = fn.size(1) // 4
            reference = native.NativeV41HyperConnection(fn, scale, base, 20, 1e-6, 1e-6)
            target = native.NativeV41HyperConnection(
                fn, scale, base, 20, 1e-6, 1e-6, str(args.library_directory)
            )
            for rows in (1, 17, 128, 256, 512):
                torch.manual_seed(111230 + rows)
                residual = torch.randn(
                    rows, 4, hidden, device=device, dtype=torch.bfloat16
                )
                carried = torch.rand(rows, 4, device=device, dtype=torch.float32)
                flat = residual.flatten(1).float()
                oracle_mix = (flat @ fn.T) * torch.rsqrt(
                    flat.square().mean(-1, keepdim=True) + 1e-6
                )
                mix = native.v41_mhc_prenorm(
                    residual.flatten(1), fn, str(args.library_directory), 1e-6
                )
                mix_error = _error(mix, oracle_mix)
                if mix_error["relative_rms"] > 0.002:
                    raise AssertionError(
                        f"TF32 mHC mix error exceeds 0.2% RMS: {mix_error}"
                    )
                reference_outputs = reference.prepare(residual, carried)
                outputs = target.prepare(residual, carried)
                # A current sublayer's new pre must not replace carried_pre.
                torch.testing.assert_close(
                    outputs[0], reference_outputs[0], atol=0, rtol=0
                )
                metadata_errors = {
                    name: _error(actual, expected)
                    for name, actual, expected in zip(
                        ("next_pre", "post", "comb"), outputs[1:], reference_outputs[1:]
                    )
                }
                if any(
                    error["relative_rms"] > 0.01 for error in metadata_errors.values()
                ):
                    raise AssertionError(
                        f"mHC metadata error exceeds 1% RMS: {metadata_errors}"
                    )
                value = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
                merged = target.combine(value, residual, outputs[2], outputs[3])
                reference_merged = reference.combine(
                    value, residual, reference_outputs[2], reference_outputs[3]
                )
                merged_error = _error(merged, reference_merged)
                if merged_error["relative_rms"] > 0.01:
                    raise AssertionError(
                        f"mHC residual result exceeds 1% RMS: {merged_error}"
                    )
                baseline_ms = _time_ms(lambda: reference.prepare(residual, carried))
                mate_ms = _time_ms(lambda: target.prepare(residual, carried))
                record = {
                    "part": part,
                    "rows": rows,
                    "hidden": hidden,
                    "mix_error": mix_error,
                    "metadata_errors": metadata_errors,
                    "merged_error": merged_error,
                    "reference_prepare_gpu_ms": baseline_ms,
                    "mate_prepare_gpu_ms": mate_ms,
                    "speedup": baseline_ms / mate_ms,
                    "warmups": 2,
                    "repeats": 10,
                }
                results.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                    + "\n"
                )
                logger.info(
                    "mHC %s M=%d mix_RMS=%.6g prepare %.4f -> %.4f ms",
                    part,
                    rows,
                    mix_error["relative_rms"],
                    baseline_ms,
                    mate_ms,
                )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_REAL_WEIGHT_MHC_PRENORM_LAYER",
                "results": results,
                "scope": "layer0 attention/FFN parameters; native shifted-pre preserved; no claim of full-model PD equivalence or global speedup",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
