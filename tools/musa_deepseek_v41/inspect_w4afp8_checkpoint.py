# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Inspect the converted checkpoint contract without loading tensor payloads.

This is an offline loader gate, not a GEMM or model-correctness test. File SHA
verification is performed by the asset downloader, independently of this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Incomplete Safetensors prefix: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 1 < length < min(128 * 1024**2, path.stat().st_size):
            raise ValueError(f"Invalid Safetensors header size: {path.name}")
        return json.loads(stream.read(length))


def inspect_checkpoint(model: Path) -> dict[str, Any]:
    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    quant = config["quantization_config"]
    text = config["text_config"]
    if quant.get("quant_method") != "w4afp8" or quant.get("moe_group_size") != 128:
        raise ValueError("Requires the converted W4AFP8 expert group128 asset")
    hidden = int(text["hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    counts = {
        "layers": (int(text["num_hidden_layers"]), int(text["n_routed_experts"])),
        "mtp": (
            int(text["num_nextn_predict_layers"]),
            int(text["dspark_n_routed_experts"]),
        ),
    }
    if hidden % 128 or intermediate % 128:
        raise ValueError("Expert matrix K dimensions must be divisible by 128")
    index_bytes = (model / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)["weight_map"]
    entries: dict[str, dict[str, Any]] = {}
    dtype_counts: Counter[str] = Counter()
    files = sorted(set(index.values()))
    for filename in files:
        if Path(filename).name != filename:
            raise ValueError("Expected basename-only shard references")
        for name, metadata in _read_header(model / filename).items():
            if name == "__metadata__":
                continue
            if name in entries or index.get(name) != filename:
                raise ValueError(f"Duplicate or misindexed tensor: {name}")
            entries[name] = metadata
            dtype_counts[metadata["dtype"]] += 1
    if set(entries) != set(index):
        raise ValueError("Shard tensor names and index do not match")

    expert_pattern = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.")
    observed: dict[str, set[tuple[int, int]]] = {"layers": set(), "mtp": set()}
    expert_tensors = 0
    for name in entries:
        match = expert_pattern.match(name)
        if match:
            kind, layer, expert = match.groups()
            observed[kind].add((int(layer), int(expert)))
            expert_tensors += 1
    for kind, (layer_count, expert_count) in counts.items():
        expected = {
            (layer, expert)
            for layer in range(layer_count)
            for expert in range(expert_count)
        }
        if observed[kind] != expected:
            raise ValueError(f"Missing or unexpected expert identities: {kind}")
        for layer, expert in sorted(expected):
            prefix = f"{kind}.{layer}.ffn.experts.{expert}"
            for projection in ("w1", "w3", "w2"):
                n, k = (
                    (hidden, intermediate)
                    if projection == "w2"
                    else (intermediate, hidden)
                )
                required = {
                    "weight": ("I8", [n, k // 2]),
                    "weight_scale_inv": ("F32", [n, k // 128]),
                    "input_scale": ("BF16", [1]),
                }
                for suffix, (dtype, shape) in required.items():
                    key = f"{prefix}.{projection}.{suffix}"
                    metadata = entries.get(key)
                    if (
                        metadata is None
                        or metadata["dtype"] != dtype
                        or metadata["shape"] != shape
                    ):
                        raise ValueError(f"Unexpected converted expert layout: {key}")
    expected_expert_tensors = sum(
        layers * experts * 9 for layers, experts in counts.values()
    )
    if expert_tensors != expected_expert_tensors:
        raise ValueError("Unexpected extra converted expert tensors")
    logger.info("Validated W4AFP8 expert header layouts in %d shards", len(files))
    return {
        "status": "PASS_CHECKPOINT_LAYOUT_ONLY",
        "model": str(model),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "indexed_shards": len(files),
        "indexed_tensors": len(entries),
        "dtype_counts": dict(dtype_counts),
        "main_layers": counts["layers"][0],
        "main_experts_per_layer": counts["layers"][1],
        "draft_layers": counts["mtp"][0],
        "draft_experts_per_layer": counts["mtp"][1],
        "expert_tensors_checked": expert_tensors,
        "nonexpert_tensors": len(entries) - expert_tensors,
        "hidden": hidden,
        "intermediate": intermediate,
        "expert_group_size": 128,
        "expert_encoding": "signed INT4 packed in INT8; not original E2M1 FP4",
        "scope": "headers/index/layout; no tensor-payload SHA, quantization, GEMM or model forward",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
