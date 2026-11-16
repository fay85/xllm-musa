# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export FP8 projections that quantize activation groups once per call.

Derived from frozen SGLang 7c97bb0e's prequantized_decode_gemm. This is a
capacity prerequisite for DP Attention, not a claim of model speedup. Keep
the current native activation floor (1e-4), original FP8 weights and decoded
FP32 UE8M0 scales. Python runs offline; native serving launches the binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.build_dense_block32_aot import (
    _export,
    dense_block32_reduce,
)

import triton
from triton.backends.compiler import AttrsDescriptor, GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def quantize_block32_once(
    inputs: tl.tensor,
    quantized: tl.tensor,
    scales: tl.tensor,
    GROUPS: tl.constexpr,
):
    groups = tl.program_id(0) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 32)
    offsets = groups[:, None] * 32 + columns[None, :]
    values = tl.load(inputs + offsets, groups[:, None] < GROUPS, 0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(values), 1), 1e-4)
    raw_scale = maximum * (1.0 / 448.0)
    bits = raw_scale.to(tl.int32, bitcast=True)
    scale = ((bits + 0x7FFFFF) & -0x800000).to(tl.float32, bitcast=True)
    values = tl.clamp(values * (1.0 / scale)[:, None], -448.0, 448.0)
    tl.store(quantized + offsets, values, groups[:, None] < GROUPS)
    tl.store(scales + groups, scale, groups < GROUPS)


@triton.jit
def prequantized_block32_gemm(
    inputs: tl.tensor,
    weight: tl.tensor,
    input_scale: tl.tensor,
    weight_scale: tl.tensor,
    partials: tl.tensor,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    block_column = tl.program_id(0)
    split = tl.program_id(1)
    input_rows = tl.arange(0, 32) % ROWS
    output_columns = block_column * 32 + tl.arange(0, 32)
    contraction = tl.arange(0, 32)
    accumulator = tl.zeros((32, 32), tl.float32)
    for local_tile in range(HIDDEN // 32 // SPLITS):
        tile = split * (HIDDEN // 32 // SPLITS) + local_tile
        offsets = tile * 32 + contraction
        values = tl.load(inputs + input_rows[:, None] * HIDDEN + offsets[None, :])
        weights = tl.load(weight + output_columns[None, :] * HIDDEN + offsets[:, None])
        activation_scale = tl.load(input_scale + input_rows * (HIDDEN // 32) + tile)
        scales = tl.load(weight_scale + (output_columns // 32) * (HIDDEN // 32) + tile)
        accumulator += tl.dot(values, weights) * activation_scale[:, None] * scales[None, :]
    stored_rows = tl.arange(0, 32)
    tl.store(
        partials
        + split * ROWS * COLUMNS
        + stored_rows[:, None] * COLUMNS
        + output_columns[None, :],
        accumulator,
        stored_rows[:, None] < ROWS,
    )


def _build(output: Path, shape: list[int]) -> dict[str, Any]:
    rows, columns, hidden, splits = shape
    if not (
        1 <= rows <= 8
        and columns > 0
        and columns % 32 == 0
        and hidden > 0
        and splits in (1, 2, 4, 8, 16, 32)
        and hidden % (32 * splits) == 0
    ):
        raise ValueError(f"Unsupported prequantized projection: {shape}")
    destination = output / f"m{rows}_n{columns}_k{hidden}_s{splits}"
    destination.mkdir(parents=True, exist_ok=False)
    target = GPUTarget("musa", 31, 32)
    options = {"num_warps": 4, "num_stages": 1}
    quant = triton.compile(
        ASTSource(
            quantize_block32_once,
            signature={"inputs": "*bf16", "quantized": "*fp8e4nv", "scales": "*fp32"},
            constants={"GROUPS": rows * hidden // 32},
            attrs=AttrsDescriptor.from_hints({0: 16, 1: 16, 2: 16}),
        ),
        target=target,
        options=options,
    )
    gemm = triton.compile(
        ASTSource(
            prequantized_block32_gemm,
            signature={
                "inputs": "*fp8e4nv",
                "weight": "*fp8e4nv",
                "input_scale": "*fp32",
                "weight_scale": "*fp32",
                "partials": "*fp32",
            },
            constants={"ROWS": rows, "COLUMNS": columns, "HIDDEN": hidden, "SPLITS": splits},
            attrs=AttrsDescriptor.from_hints({i: 16 for i in range(5)}),
        ),
        target=target,
        options=options,
    )
    reduce = triton.compile(
        ASTSource(
            dense_block32_reduce,
            signature={"partials": "*fp32", "output": "*bf16", "elements": "i32"},
            constants={"SPLITS": splits},
            attrs=AttrsDescriptor.from_hints({0: 16, 1: 16, 2: 16}),
        ),
        target=target,
        options=options,
    )
    manifest = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "SGLang 7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6 prequantized_decode_gemm",
        "static_shape": [rows, columns, hidden],
        "split_count": splits,
        "activation_minimum": 1e-4,
        "activation_scale": "block32 power-of-two FP32",
        "alignment_bytes": 16,
        "alignment_contract": "Native caller validates input/weight/scale pointers; output elements divisible by 16",
        "target": {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size},
        "triton_version": triton.__version__,
        "kernels": {
            "quant": _export(quant, destination, "quant"),
            "gemm": _export(gemm, destination, "gemm"),
            "reduce": _export(reduce, destination, "reduce"),
        },
        "status": "EXPORTED_REQUIRES_NATIVE_GPU_AND_GRAPH_VALIDATION",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Exported prequantized block32 shape=%s", shape)
    return {"directory": str(destination), "shape": shape, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4, action="append", required=True,
                        metavar=("M", "N", "K", "SPLITS"))
    args = parser.parse_args()
    records = [_build(args.output, shape) for shape in args.shape]
    (args.output / "export_summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
