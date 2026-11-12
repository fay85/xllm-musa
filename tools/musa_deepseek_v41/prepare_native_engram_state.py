# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Export tokenizer-derived metadata for the native reader, not model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from scripts.logger import logger
from xllm.python.models.deepseek_v41_engram import (
    EngramLayout,
    compute_hash_multipliers,
)


def _sha(path: Path) -> str:
    with path.open("rb") as file:
        return (
            hashlib.file_digest(file, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _sha_stream(file)
        )


def _sha_stream(file: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.output.name != "engram_hash_state.native":
        raise ValueError("Use .native to keep this sidecar outside weight discovery")
    model = args.model_view
    raw = json.loads((model / "config.json").read_text())
    cfg = raw.get("text_config", raw)
    layout = EngramLayout(
        cfg["engram_layer_ids"],
        cfg["engram_num_embeddings"],
        cfg["engram_max_ngram_size"],
        cfg["engram_n_heads"],
        cfg["engram_head_dim"],
        cfg["engram_vocab_size"],
        cfg["engram_compressed_vocab_size"],
        cfg["engram_pad_token_id"],
    )
    source = model / "engram_hash_state.bin"
    state = torch.load(source, map_location="cpu", weights_only=True)
    token_map = state["token_map"].to(torch.int64)
    multipliers = state["multipliers"].to(torch.int64)
    if token_map.ndim != 1 or token_map.numel() != cfg["vocab_size"]:
        raise ValueError(
            "Existing tokenizer-derived map does not cover the frozen vocabulary"
        )
    if (
        int(token_map.max()) + 1 != layout.compressed_vocab_size
        or int(token_map.min()) < 0
    ):
        raise ValueError(
            "Existing tokenizer map does not match the configured compressed vocabulary"
        )
    expected = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
    )
    if not torch.equal(multipliers, expected):
        raise ValueError(
            "Hash multipliers differ from the configured reference algorithm"
        )
    values = {
        "token_map": token_map,
        "multipliers": multipliers,
        "primes": torch.tensor(layout.primes, dtype=torch.int64),
        "offsets": layout.offsets,
        "layer_ids": torch.tensor(layout.layer_ids, dtype=torch.int64),
    }
    metadata = {
        "model_revision": args.revision,
        "config_sha256": _sha(model / "config.json"),
        "source_engram_state_sha256": _sha(source),
        "tokenizer_sha256": _sha(model / "tokenizer.json"),
        "scope": "tokenizer-derived metadata only; no original weights modified",
    }
    header: dict[str, Any] = {"__metadata__": metadata}
    payloads, offset = [], 0
    for name, tensor in values.items():
        payload = tensor.contiguous().numpy().astype("<i8", copy=False).tobytes()
        header[name] = {
            "dtype": "I64",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((-len(head)) % 8)
    with args.output.open("xb") as file:
        file.write(struct.pack("<Q", len(head)))
        file.write(head)
        for payload in payloads:
            file.write(payload)
    report = {
        **metadata,
        "path": str(args.output),
        "sha256": _sha(args.output),
        "bytes": args.output.stat().st_size,
        "layers": layout.layer_ids,
        "columns_per_layer": layout.n_hash_cols,
        "token_map_entries": token_map.numel(),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2))
    logger.info("Prepared native Engram metadata: %s", json.dumps(report))


if __name__ == "__main__":
    main()
