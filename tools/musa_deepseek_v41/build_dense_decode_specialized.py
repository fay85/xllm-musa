# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the delivery's exact-shape M1 block32 decode GEMM for comparison.

Keep the seven-argument native probe ABI. M/N/K are compile-time constants,
and the repeated input rows follow musa_decode_quant_gemm.py directly.
This experiment does not enable a model serving path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.build_dense_block32_aot import (
    _export,
    dense_block32_reduce,
)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def specialized_decode_split(
    inputs: tl.tensor,
    weight: tl.tensor,
    weight_scale: tl.tensor,
    partials: tl.tensor,
    runtime_rows: tl.tensor,
    runtime_columns: tl.tensor,
    runtime_hidden: tl.tensor,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    block_column = tl.program_id(0)
    split = tl.program_id(1)
    input_rows = tl.arange(0, 32) % M
    output_columns = block_column * 32 + tl.arange(0, 32)
    contraction = tl.arange(0, 32)
    accumulator = tl.zeros((32, 32), tl.float32)
    for local_tile in range(K // 32 // SPLITS):
        tile = split * (K // 32 // SPLITS) + local_tile
        offsets = tile * 32 + contraction
        values = tl.load(inputs + input_rows[:, None] * K + offsets[None, :]).to(
            tl.float32
        )
        maximum = tl.maximum(tl.max(tl.abs(values), 1), 1e-10)
        raw_scale = maximum * (1.0 / 448.0)
        bits = raw_scale.to(tl.int32, bitcast=True)
        scale = ((bits + 0x7FFFFF) & -0x800000).to(tl.float32, bitcast=True)
        quantized = tl.clamp(values / scale[:, None], -448.0, 448.0).to(
            weight.dtype.element_ty
        )
        weights = tl.load(weight + output_columns[None, :] * K + offsets[:, None])
        scales = tl.load(weight_scale + (output_columns // 32) * (K // 32) + tile)
        accumulator += tl.dot(quantized, weights) * scale[:, None] * scales[None, :]
    rows = tl.arange(0, 32)
    tl.store(
        partials + split * M * N + rows[:, None] * N + output_columns[None, :],
        accumulator,
        rows[:, None] < M,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes",
        type=Path,
        required=True,
        help="JSON list of [N,K] pairs from the real TP8 fixture",
    )
    args = parser.parse_args()
    target = GPUTarget("musa", 31, 32)
    for columns, hidden in json.loads(args.shapes.read_text()):
        splits = 16 if hidden % 512 == 0 else 4
        folder = args.output / f"m1_n{columns}_k{hidden}"
        folder.mkdir(parents=True, exist_ok=True)
        signature = {
            "inputs": "*bf16",
            "weight": "*fp8e4nv",
            "weight_scale": "*fp32",
            "partials": "*fp32",
            "runtime_rows": "i32",
            "runtime_columns": "i32",
            "runtime_hidden": "i32",
        }
        compiled = triton.compile(
            ASTSource(
                specialized_decode_split,
                signature=signature,
                constants={"M": 1, "N": columns, "K": hidden, "SPLITS": splits},
            ),
            target=target,
            options={"num_warps": 4, "num_stages": 1},
        )
        split = _export(compiled, folder, "split" + str(splits))
        reduction = triton.compile(
            ASTSource(
                dense_block32_reduce,
                signature={"partials": "*fp32", "output": "*bf16", "elements": "i32"},
                constants={"SPLITS": splits},
            ),
            target=target,
            options={"num_warps": 4, "num_stages": 1},
        )
        reduce = _export(reduction, folder, "reduce" + str(splits))
        manifest = {
            "target": {"backend": "musa", "arch": 31, "warp_size": 32},
            "static_shape": [1, columns, hidden],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_reference": "JD SGLang7c97bb0 musa_decode_quant_gemm.py",
            "kernels": {"split" + str(splits): split, "reduce" + str(splits): reduce},
            "status": "EXPORTED_NOT_YET_VALIDATED_NOT_ENABLED_IN_SERVING",
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(
            "Exported exact decode M1 N%d K%d splits%d", columns, hidden, splits
        )


if __name__ == "__main__":
    main()
