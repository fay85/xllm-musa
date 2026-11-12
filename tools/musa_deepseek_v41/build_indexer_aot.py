# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the delivery's BF16-rounded indexer scorer for native C++ serving.

Ported from JD SGLang7c97bb0's indexer_full_scorer.py. Runtime row counts and
an explicit committed-row bound let native paged-cache views retain NaN
padding without reading it. Dot, weighted product and head sum each keep
the reference BF16 round; the output is F32 for masking and top-k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl


@triton.jit
def indexer_score(
    query: tl.tensor,
    keys: tl.tensor,
    weights: tl.tensor,
    output: tl.tensor,
    visible_lengths: tl.tensor,
    number_keys: tl.tensor,
    number_queries: tl.tensor,
    committed: tl.tensor,
):
    query_heads = tl.program_id(0) * 64 + tl.arange(0, 64)
    key_columns = tl.program_id(1) * 128 + tl.arange(0, 128)
    inner = tl.arange(0, 128)
    valid_query_heads = query_heads < number_queries * 32
    valid_keys = (key_columns < number_keys) & (key_columns < committed)
    query_tile = tl.load(
        query + query_heads[:, None] * 128 + inner[None, :],
        valid_query_heads[:, None],
        0.0,
    )
    key_tile = tl.load(
        keys + key_columns[None, :] * 128 + inner[:, None],
        valid_keys[None, :],
        0.0,
    )
    scores = tl.dot(query_tile, key_tile).to(tl.bfloat16).to(tl.float32)
    head_weights = tl.load(weights + query_heads, valid_query_heads, 0.0).to(tl.float32)
    weighted = (
        (tl.maximum(scores, 0.0) * head_weights[:, None]).to(tl.bfloat16).to(tl.float32)
    )
    reduced = (
        tl.sum(tl.reshape(weighted, (2, 32, 128)), axis=1)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    rows = tl.program_id(0) * 2 + tl.arange(0, 2)
    valid_rows = rows < number_queries
    visible = tl.load(visible_lengths + rows, valid_rows, 0)
    reduced = tl.where(key_columns[None, :] < visible[:, None], reduced, float("-inf"))
    tl.store(
        output + rows[:, None] * number_keys + key_columns[None, :],
        reduced,
        valid_rows[:, None] & (key_columns[None, :] < number_keys),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {name: "*bf16" for name in ("query", "keys", "weights")}
    signature.update(
        output="*fp32",
        visible_lengths="*i32",
        number_keys="i32",
        number_queries="i32",
        committed="i32",
    )
    compiled = triton.compile(
        ASTSource(indexer_score, signature=signature),
        target=GPUTarget("musa", 31, 32),
        options={"num_warps": 8, "num_stages": 1},
    )
    binary = compiled.asm["mubin"]
    if isinstance(binary, tuple):
        binary = Path(binary[1]).read_bytes()
    if isinstance(binary, str):
        binary = Path(binary).read_bytes()
    output = args.output / "indexer_score.mubin"
    output.write_bytes(binary)
    metadata = compiled.metadata
    record = {
        "target": {"backend": "musa", "arch": 31, "warp_size": 32},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_reference": "JD SGLang7c97bb0 indexer_full_scorer.py",
        "kernels": {
            "indexer_score": {
                "file": output.name,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "bytes": len(binary),
                "symbol": metadata.name,
                "threads": metadata.num_warps * 32,
                "shared": metadata.shared,
                "num_ctas": metadata.num_ctas,
                "cluster_dims": metadata.cluster_dims,
            }
        },
        "status": "EXPORTED_NOT_YET_NUMERICALLY_VALIDATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2))
    logger.info("Exported native Indexer scorer to %s", output)


if __name__ == "__main__":
    main()
