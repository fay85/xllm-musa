# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate W4A8 ABI/stream, packing, scale and compiler failures.

Explicit synchronization and a few CPU dot products are diagnostic oracles,
never a serving fallback. This tool reports errors without weakening the
acceptance thresholds in validate_w4a8_experts.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import tvm_ffi
from safetensors.torch import load_file
from scripts.logger import logger
from tools.musa_deepseek_v41.validate_w4a8_experts import (
    _dequantize,
    _error,
    _quantize,
    _weight,
)


def _case(
    native: Any,
    function: Any,
    library: str,
    name: str,
    q: torch.Tensor,
    a_scale: torch.Tensor,
    packed: torch.Tensor,
    b_scale: torch.Tensor,
) -> dict:
    counts = torch.ones(q.size(0), device=q.device, dtype=torch.int32)
    torch.musa.synchronize()
    cpp = native.v41_w4a8_gemm(q, a_scale, packed, b_scale, counts, library)
    torch.musa.synchronize()
    direct = torch.empty_like(cpp)
    properties = torch.musa.get_device_properties(0)
    function(
        q,
        a_scale,
        packed,
        b_scale,
        direct,
        counts,
        q.size(1),
        properties.multi_processor_count,
    )
    torch.musa.synchronize()
    x = _dequantize(q[:, :1], a_scale[:, :1])
    weight = _weight(packed, b_scale)
    oracle = torch.bmm(x, weight.transpose(1, 2)).to(torch.bfloat16)
    # Independent CPU arithmetic for only four output cells, to check that
    # the GPU FP32 oracle itself is trustworthy for this diagnostic.
    cpu_x = x[0, 0].cpu().double()
    raw = packed[0, :4].cpu().to(torch.uint8)
    low, high = raw & 15, raw >> 4
    codes = torch.stack((low, high), -1).flatten(-2).to(torch.int32)
    signed = torch.where(codes >= 8, codes - 16, codes).double()
    cpu_weight = signed * b_scale[0, :4].cpu().double().repeat_interleave(128, -1)
    cpu_dot = (cpu_weight * cpu_x).sum(-1)
    return {
        "name": name,
        "q_shape": list(q.shape),
        "weight_shape": list(packed.shape),
        "a_scale_stride": list(a_scale.stride()),
        "b_scale_stride": list(b_scale.stride()),
        "cpp_vs_oracle": _error(cpp[:, :1], oracle),
        "python_ffi_vs_oracle": _error(direct[:, :1], oracle),
        "cpp_vs_python_ffi": _error(cpp[:, :1], direct[:, :1]),
        "first4": {
            "cpu_fp64_dot": cpu_dot.tolist(),
            "gpu_fp32_oracle_bf16": oracle[0, 0, :4].float().cpu().tolist(),
            "cpp": cpp[0, 0, :4].float().cpu().tolist(),
            "python_ffi": direct[0, 0, :4].float().cpu().tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gemm-so", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    function = tvm_ffi.load_module(str(args.gemm_so)).get_function(
        "mp31_masked_moe_gemm_mixed_dtype_bf16_32x256x256_s3_Apergroup"
    )
    device = torch.device("musa:0")
    source = load_file(str(args.fixture), device="cpu")
    records = []
    with torch.inference_mode():
        for k in (256, 2304, 5120):
            q = torch.ones((1, 128, k), device=device, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            a_scale = torch.ones((1, 128, k // 128), device=device)
            b = torch.full((1, 256, k // 2), 17, device=device, dtype=torch.int8)
            b_scale = torch.ones(
                (1, 256, k // 128), device=device, dtype=torch.bfloat16
            )
            records.append(
                _case(
                    native,
                    function,
                    str(args.gemm_so),
                    f"ones_K{k}",
                    q,
                    a_scale,
                    b,
                    b_scale,
                )
            )
        for projection in ("w1", "w2"):
            packed = torch.stack(
                [
                    source[f"layers.0.ffn.experts.{i}.{projection}.weight"]
                    for i in (0, 1)
                ]
            ).to(device)
            scale = torch.stack(
                [
                    source[f"layers.0.ffn.experts.{i}.{projection}.weight_scale_inv"]
                    for i in (0, 1)
                ]
            ).to(device=device, dtype=torch.bfloat16)
            k = packed.size(2) * 2
            torch.manual_seed(1112 + k)
            inputs = torch.randn((2, 128, k), device=device, dtype=torch.bfloat16)
            q, a_scale = _quantize(inputs, 1e-4)
            records.append(
                _case(
                    native,
                    function,
                    str(args.gemm_so),
                    projection + "_real_scales",
                    q,
                    a_scale,
                    packed,
                    scale,
                )
            )
            records.append(
                _case(
                    native,
                    function,
                    str(args.gemm_so),
                    projection + "_unit_b_scales",
                    q,
                    a_scale,
                    packed,
                    torch.ones_like(scale),
                )
            )
            records.append(
                _case(
                    native,
                    function,
                    str(args.gemm_so),
                    projection + "_unit_all_scales",
                    q,
                    torch.ones_like(a_scale),
                    packed,
                    torch.ones_like(scale),
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"diagnostic_only": True, "records": records}, indent=2)
                + "\n"
            )
    args.output.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "library_sha256": hashlib.sha256(args.gemm_so.read_bytes()).hexdigest(),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "Saved %d synchronized GEMM diagnostics to %s", len(records), args.output
    )


if __name__ == "__main__":
    main()
