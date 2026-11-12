# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract the actual TP8 rank-0 wo_a BF16 carrier without modifying weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_deepseek_v41.validate_dense_block32_aot import _tensor
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "layers.0.attn.wo_a"
    raw = (
        _tensor(args.model, index, prefix + ".weight")
        .reshape(8192, 4096)
        .chunk(8, 0)[0]
        .contiguous()
    )
    exponent = (
        _tensor(args.model, index, prefix + ".scale")
        .reshape(256, 128)
        .chunk(8, 0)[0]
        .to(torch.int32)
        .contiguous()
    )
    if exponent.eq(255).any():
        raise ValueError("Actual wo_a contains NaN UE8M0 scale")
    scale = (
        torch.where(exponent.eq(0), 0x00400000, exponent << 23)
        .contiguous()
        .view(torch.float32)
    )
    weight = (
        (raw.float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1))
        .to(torch.bfloat16)
        .reshape(1, 1024, 4096)
    )
    path = args.output / "wo_a_rank0.pt"
    torch.save({"weight": weight}, path)
    metadata = {
        "model": str(args.model),
        "weight_name": prefix + ".weight",
        "scale_name": prefix + ".scale",
        "source_shard": index[prefix + ".weight"],
        "tp": 8,
        "rank": 0,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "scale_contract": "original FP8 block32 UE8M0, then same BF16 load carrier as production; no model asset rewritten",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Extracted actual wo_a fixture %s", path)


if __name__ == "__main__":
    main()
