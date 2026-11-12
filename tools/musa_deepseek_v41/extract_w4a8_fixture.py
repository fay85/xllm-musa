# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract two exact expert payloads for a bounded backend numerical test.

The result is a test fixture, not a deployment checkpoint. No quantization,
dequantization, GPU execution or changes to the source checkpoint occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=("experts", "mhc"), default="experts")
    args = parser.parse_args()
    index_path = args.model / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)["weight_map"]
    prefixes = (
        ("layers.0.ffn.experts.0.", "layers.0.ffn.experts.1.")
        if args.kind == "experts"
        else ("layers.0.hc_attn_", "layers.0.hc_ffn_")
    )
    names = sorted(name for name in index if name.startswith(prefixes))
    expected = 18 if args.kind == "experts" else 6
    if len(names) != expected:
        raise ValueError(f"Expected {expected} exact fixture tensors, got {len(names)}")
    headers: dict[str, tuple[int, dict]] = {}
    payloads: list[bytes] = []
    output_header: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    offset = 0
    for name in names:
        shard = index[name]
        if shard not in headers:
            with (args.model / shard).open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                if not 0 < length < 100 * 1024**2:
                    raise ValueError("Invalid source Safetensors header length")
                headers[shard] = (8 + length, json.loads(source.read(length)))
        base, header = headers[shard]
        tensor = header[name]
        start, end = tensor["data_offsets"]
        with (args.model / shard).open("rb") as source:
            source.seek(base + start)
            payload = source.read(end - start)
        if len(payload) != end - start:
            raise EOFError(f"Incomplete source tensor {name}")
        output_header[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
        provenance[name] = {
            "source_shard": shard,
            "source_data_offsets": [start, end],
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    encoded = json.dumps(output_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for payload in payloads:
            output.write(payload)
    record = {
        "scope": f"exact copied layer0/{args.kind} fixture, not a deployment model",
        "source_model": str(args.model),
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "source_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "fixture": str(args.output),
        "fixture_bytes": args.output.stat().st_size,
        "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "tensors": provenance,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    logger.info("Extracted %d exact fixture tensors to %s", len(names), args.output)


if __name__ == "__main__":
    main()
