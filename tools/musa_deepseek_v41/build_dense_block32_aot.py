# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export SGLang's FP8 block32 quant/GEMM math for native MUSA launches.

Derived from the pinned JD delivery's musa_decode_quant_gemm.py. The kernel
uses the same power-of-two activation scale, per-K32 FP32 accumulation and
split reduction. Runtime rows/columns/inner extend dispatch to native xLLM's
unfused projections; this is a port requiring its own numerical validation.
Python is used only at build time. Serving loads the exported MUSA binaries.
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

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def dense_block32_split(
    inputs: tl.tensor,
    weight: tl.tensor,
    weight_scale: tl.tensor,
    partials: tl.tensor,
    rows: tl.tensor,
    columns: tl.tensor,
    hidden: tl.tensor,
    SPLITS: tl.constexpr,
):
    block_column = tl.program_id(0)
    split = tl.program_id(1)
    input_rows = tl.program_id(2) * 32 + tl.arange(0, 32)
    output_columns = block_column * 32 + tl.arange(0, 32)
    contraction = tl.arange(0, 32)
    accumulator = tl.zeros((32, 32), tl.float32)
    tiles_per_split = hidden // 32 // SPLITS
    for local_tile in range(tiles_per_split):
        tile = split * tiles_per_split + local_tile
        offsets = tile * 32 + contraction
        values = tl.load(
            inputs + input_rows[:, None] * hidden + offsets[None, :],
            input_rows[:, None] < rows,
            0.0,
        ).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(values), 1), 1e-10)
        raw_scale = maximum * (1.0 / 448.0)
        scale_bits = raw_scale.to(tl.int32, bitcast=True)
        scale = ((scale_bits + 0x7FFFFF) & -0x800000).to(tl.float32, bitcast=True)
        quantized = tl.clamp(values * (1.0 / scale)[:, None], -448.0, 448.0).to(
            weight.dtype.element_ty
        )
        weights = tl.load(
            weight + output_columns[None, :] * hidden + offsets[:, None],
            output_columns[None, :] < columns,
            0.0,
        )
        scales = tl.load(
            weight_scale + (output_columns // 32) * (hidden // 32) + tile,
            output_columns < columns,
            0.0,
        )
        accumulator += tl.dot(quantized, weights) * scale[:, None] * scales[None, :]
    tl.store(
        partials
        + split * rows * columns
        + input_rows[:, None] * columns
        + output_columns[None, :],
        accumulator,
        (input_rows[:, None] < rows) & (output_columns[None, :] < columns),
    )


@triton.jit
def dense_block32_reduce(
    partials: tl.tensor,
    output: tl.tensor,
    elements: tl.tensor,
    SPLITS: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        partials + splits[:, None] * elements + offsets[None, :],
        offsets[None, :] < elements,
        0.0,
    )
    tl.store(output + offsets, tl.sum(values, axis=0), offsets < elements)


def _export(kernel: Any, output: Path, key: str) -> dict[str, Any]:
    binary = kernel.asm["mubin"]
    if isinstance(binary, tuple):
        binary = Path(binary[1]).read_bytes()
    if isinstance(binary, str):
        binary = Path(binary).read_bytes()
    path = output / (key + ".mubin")
    path.write_bytes(binary)
    metadata = kernel.metadata
    return {
        "file": path.name,
        "sha256": hashlib.sha256(binary).hexdigest(),
        "bytes": len(binary),
        "symbol": metadata.name,
        "threads": metadata.num_warps * 32,
        "shared": metadata.shared,
        "num_ctas": metadata.num_ctas,
        "cluster_dims": metadata.cluster_dims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = GPUTarget("musa", 31, 32)
    records = {}
    for splits in (4, 16):
        signature = {
            "inputs": "*bf16",
            "weight": "*fp8e4nv",
            "weight_scale": "*fp32",
            "partials": "*fp32",
            "rows": "i32",
            "columns": "i32",
            "hidden": "i32",
        }
        compiled = triton.compile(
            ASTSource(
                dense_block32_split, signature=signature, constants={"SPLITS": splits}
            ),
            target=target,
            options={"num_warps": 4, "num_stages": 1},
        )
        records[f"split{splits}"] = _export(compiled, args.output, f"split{splits}")
        reduced = triton.compile(
            ASTSource(
                dense_block32_reduce,
                signature={"partials": "*fp32", "output": "*bf16", "elements": "i32"},
                constants={"SPLITS": splits},
            ),
            target=target,
            options={"num_warps": 4, "num_stages": 1},
        )
        records[f"reduce{splits}"] = _export(reduced, args.output, f"reduce{splits}")
        logger.info("Exported block32 split=%d", splits)
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": {
            "backend": target.backend,
            "arch": target.arch,
            "warp_size": target.warp_size,
        },
        "triton_version": triton.__version__,
        "source_reference": "JD SGLang 7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6 musa_decode_quant_gemm.py",
        "kernels": records,
        "status": "EXPORTED_NOT_YET_NUMERICALLY_VALIDATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2))
    logger.info("Exported native Dense block32 kernels: %s", args.output)


if __name__ == "__main__":
    main()
