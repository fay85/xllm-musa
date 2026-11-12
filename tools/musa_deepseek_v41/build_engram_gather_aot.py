# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export JD SGLang's fused FP8/UE8M0 Engram gather for native C++.

The current xLLM table is GPU resident. Typed U8 pointers replace the donor's
raw-address ABI; bounded ownership, FP32 decode and final BF16 rounding stay
unchanged. Explicit NaN codes retain the original FP8 formats' semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.build_dense_block32_aot import _export
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def engram_gather(
    weight: tl.tensor,
    scales: tl.tensor,
    ids: tl.tensor,
    output: tl.tensor,
    row_lo: tl.tensor,
    row_hi: tl.tensor,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids + row)
    owned = (index >= row_lo) & (index < row_hi)
    local = tl.where(owned, index - row_lo, 0)
    columns = tl.arange(0, 256)
    bits = tl.load(weight + local * 256 + columns, owned, 0).to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * ((exponent + 120) << 23).to(
        tl.float32, bitcast=True
    )
    values = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)
    values = tl.where((bits & 127) == 127, float("nan"), values)
    values = tl.where((bits >> 7) != 0, -values, values)
    exponents = tl.load(scales + local * 8 + columns // 32, owned, 0).to(tl.int32)
    scale_bits = tl.where(exponents == 0, 0x00400000, exponents << 23)
    scale_bits = tl.where(exponents == 255, 0x7FC00000, scale_bits)
    decoded_scale = scale_bits.to(tl.float32, bitcast=True)
    result = tl.where(owned, values * decoded_scale, 0.0)
    tl.store(output + row * 256 + columns, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kernel = triton.compile(
        ASTSource(
            engram_gather,
            signature={
                "weight": "*u8",
                "scales": "*u8",
                "ids": "*i64",
                "output": "*bf16",
                "row_lo": "i64",
                "row_hi": "i64",
            },
        ),
        target=GPUTarget("musa", 31, 32),
        options={"num_warps": 4, "num_stages": 1},
    )
    record = _export(kernel, args.output, "engram_gather")
    manifest = {
        "target": {"backend": "musa", "arch": 31, "warp_size": 32},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_reference": "JD SGLang7c97bb0 kernels/ops/embeddings/engram_gather.py",
        "head_dim": 256,
        "scale_group": 32,
        "kernels": {"engram_gather": record},
        "status": "EXPORTED_NOT_YET_VALIDATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Exported fused native Engram gather to %s", args.output)


if __name__ == "__main__":
    main()
