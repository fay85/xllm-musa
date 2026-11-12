# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export delivery radix top-k with V4.1 variable K and exact FP32 ties.

Derived from JD SGLang indexer_radix_topk.py. Extend the original K512/width
4096,8192 implementation to runtime K (the current model uses512), visible lengths, padding and
candidate masks while retaining original-index stable tie breaking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.build_dense_block32_aot import _export
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def indexer_radix_select(
    scores: tl.tensor,
    candidates: tl.tensor,
    visible: tl.tensor,
    output: tl.tensor,
    width: tl.tensor,
    topk: tl.tensor,
    PADDED: tl.constexpr,
    HAS_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, PADDED)
    in_range = columns < width
    values = tl.load(scores + row * width + columns, in_range, -float("inf"))
    if HAS_CANDIDATES:
        candidate_mask = tl.load(candidates + row * width + columns, in_range, False)
        values = tl.where(candidate_mask, values, -float("inf"))
    # PyTorch's stable argsort treats both signed zeros as equal.
    values = tl.where(values == 0.0, 0.0, values)
    bits = values.to(tl.uint32, bitcast=True)
    signed = bits.to(tl.int32, bitcast=True)
    ordered = tl.where(signed < 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    ordered = tl.where(values != values, 0xFFFFFFFF, ordered)
    ordered = tl.where(in_range, ordered, 0)
    threshold = tl.full((), 0, tl.uint32)
    for step in tl.static_range(32):
        candidate = threshold | (1 << (31 - step))
        count = tl.sum(((ordered >= candidate) & in_range).to(tl.int32), 0)
        threshold = tl.where(count >= topk, candidate, threshold)
    higher = (ordered > threshold) & in_range
    equal = (ordered == threshold) & in_range
    higher_count = tl.sum(higher.to(tl.int32), 0)
    equal_rank = tl.cumsum(equal.to(tl.int32), 0)
    selected = higher | (equal & (equal_rank <= topk - higher_count))
    positions = tl.cumsum(selected.to(tl.int32), 0) - 1
    limit = tl.load(visible + row)
    valid = (columns < limit) & (tl.abs(values) < float("inf"))
    tl.store(output + row * topk + positions, tl.where(valid, columns, -1), selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.score_directory / "manifest.json").read_text())
    for record in manifest["kernels"].values():
        shutil.copyfile(
            args.score_directory / record["file"], args.output / record["file"]
        )
    for width in (128, 256, 512, 1024, 2048, 4096, 8192):
        for masked in (False, True):
            compiled = triton.compile(
                ASTSource(
                    indexer_radix_select,
                    signature={
                        "scores": "*fp32",
                        "candidates": "*i1",
                        "visible": "*i64",
                        "output": "*i32",
                        "width": "i32",
                        "topk": "i32",
                    },
                    constants={"PADDED": width, "HAS_CANDIDATES": masked},
                ),
                target=GPUTarget("musa", 31, 32),
                options={"num_warps": 8 if width <= 4096 else 16, "num_stages": 1},
            )
            key = f"select_{width}_{int(masked)}"
            manifest["kernels"][key] = _export(compiled, args.output, key)
            logger.info(
                "Exported stable radix top-k width%d candidates=%s", width, masked
            )
    manifest["topk_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest["topk_source_reference"] = (
        "JD SGLang7c97bb0 indexer_radix_topk.py, exact FP32 radix"
    )
    manifest["status"] = "EXPORTED_SCORE_VALIDATED_TOPK_NOT_YET_VALIDATED"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
