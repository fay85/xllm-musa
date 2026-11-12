# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Port delivery Engram hashing to xLLM's paged history and PD lookback.

Hash arithmetic follows JD SGLang7c97bb0 engram_hash.py. History is a paged
compressed-ID plane here, not SGLang's raw-token request-slot history. Finish
all hash reads before a second kernel commits each slot's last logical row.
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


class _OrderedASTSource(ASTSource):
    """Include ABI argument order omitted by this toolchain's cache key."""

    def hash(self) -> str:
        ordered = json.dumps(list(self.signature.items()))
        return hashlib.sha256((super().hash() + ordered).encode()).hexdigest()


@triton.jit
def _map_token(raw: tl.tensor, token_map: tl.tensor, map_size: tl.tensor) -> tl.tensor:
    mapped = tl.load(token_map + tl.minimum(tl.maximum(raw, 0), map_size - 1))
    return tl.where((raw == 129264) | (raw == 129265), -1, mapped)


@triton.jit
def paged_engram_hash(
    ids: tl.tensor,
    positions: tl.tensor,
    table: tl.tensor,
    history: tl.tensor,
    token_map: tl.tensor,
    multipliers: tl.tensor,
    primes: tl.tensor,
    offsets: tl.tensor,
    lookback: tl.tensor,
    output: tl.tensor,
    active: tl.tensor,
    slots: tl.tensor,
    mapped_tokens: tl.tensor,
    count: tl.tensor,
    map_size: tl.tensor,
    history_size: tl.tensor,
    table_size: tl.tensor,
    block_size: tl.tensor,
    start: tl.tensor,
    lookback_size: tl.tensor,
    pad_id: tl.tensor,
) -> None:
    row = tl.program_id(0)
    raw = tl.load(ids + row)
    position = tl.load(positions + row)
    current = _map_token(raw, token_map, map_size)
    # Publish four explicit bool bytes per aligned u32 word. The C++ caller
    # pads the final partial word and returns only the logical token rows.
    # This is our explicit packed ABI, not a hardware bool-store limitation.
    if row % 4 == 0:
        word = tl.full((), 0, tl.uint32)
        for byte in tl.static_range(4):
            token = tl.load(ids + row + byte, row + byte < count, 129264)
            valid_token = ((token != 129264) & (token != 129265)).to(tl.uint32)
            word = word | (valid_token << (8 * byte))
        tl.store(active + row // 4, word)
    tl.store(mapped_tokens + row, current)
    block = tl.load(table + (position // block_size) % table_size)
    tl.store(slots + row, (block * block_size + position % block_size).to(tl.int32))
    columns = tl.arange(0, 64)
    valid = columns < 48
    layer = columns // 24
    gram = (columns % 24) // 8 + 2
    rolling = tl.full((64,), 0, tl.int64)
    blocked = tl.full((), False, tl.int1)
    for shift in tl.static_range(4):
        previous = position - shift
        batch_token = tl.load(ids + tl.minimum(tl.maximum(row - shift, 0), count - 1))
        from_batch = _map_token(batch_token, token_map, map_size)
        clamped = tl.maximum(previous, 0)
        old_block = tl.load(table + (clamped // block_size) % table_size)
        old_slot = old_block * block_size + clamped % block_size
        cached = tl.load(
            history + tl.minimum(tl.maximum(old_slot, 0), history_size - 1)
        ).to(tl.int64)
        fallback = tl.where(old_slot >= 0, cached, -1)
        column = start - 1 - previous
        known = (lookback_size > 0) & (column >= 0) & (column < lookback_size)
        old_token = tl.load(
            lookback
            + tl.minimum(tl.maximum(column, 0), tl.maximum(lookback_size - 1, 0)),
            mask=lookback_size > 0,
            other=-1,
        )
        fallback = tl.where(
            known & (old_token >= 0),
            _map_token(old_token, token_map, map_size),
            fallback,
        )
        source = tl.where(previous >= start, from_batch, fallback)
        blocked = blocked | (previous < 0) | (source == -1)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * 4 + shift, valid, 0)
        rolling = rolling ^ tl.where(shift < gram, value * multiplier, 0)
    prime = tl.load(primes + columns, valid, 1)
    offset = tl.load(offsets + columns, valid, 0)
    remainder = rolling % prime
    remainder = tl.where(remainder < 0, remainder + prime, remainder)
    tl.store(output + row * 48 + columns, remainder + offset, valid)


@triton.jit
def paged_engram_commit(
    slots: tl.tensor,
    mapped_tokens: tl.tensor,
    history: tl.tensor,
    count: tl.tensor,
    history_size: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    others = tl.arange(0, BLOCK)
    other_slots = tl.load(slots + others, others < count, -1)
    later = (others > row) & (others < count) & (other_slots == slot)
    last = tl.sum(later.to(tl.int32), 0) == 0
    value = tl.load(mapped_tokens + row)
    tl.store(history + slot, value, last & (slot >= 0) & (slot < history_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # ASTSource consumes signature values in insertion order on this MUSA
    # toolchain. Keep mixed pointer dtypes in the exact function order.
    signature = {
        "ids": "*i64",
        "positions": "*i64",
        "table": "*i64",
        "history": "*i32",
        "token_map": "*i64",
        "multipliers": "*i64",
        "primes": "*i64",
        "offsets": "*i64",
        "lookback": "*i64",
        "output": "*i32",
        "active": "*u32",
        "slots": "*i32",
        "mapped_tokens": "*i32",
        "count": "i32",
        "map_size": "i64",
        "history_size": "i64",
        "table_size": "i64",
        "block_size": "i64",
        "start": "i64",
        "lookback_size": "i64",
        "pad_id": "i64",
    }
    if list(signature) != paged_engram_hash.arg_names:
        raise ValueError("AOT signature order differs from kernel arguments")
    target = GPUTarget("musa", 31, 32)
    compiled = triton.compile(
        _OrderedASTSource(paged_engram_hash, signature=signature),
        target=target,
        options={"num_warps": 4, "num_stages": 1},
    )
    records = {"hash": _export(compiled, args.output, "hash")}
    for block in (32, 512):
        compiled = triton.compile(
            ASTSource(
                paged_engram_commit,
                signature={
                    "slots": "*i32",
                    "mapped_tokens": "*i32",
                    "history": "*i32",
                    "count": "i32",
                    "history_size": "i64",
                },
                constants={"BLOCK": block},
            ),
            target=target,
            options={"num_warps": 4, "num_stages": 1},
        )
        records["commit" + str(block)] = _export(
            compiled, args.output, "commit" + str(block)
        )
    manifest = {
        "target": {"backend": "musa", "arch": 31, "warp_size": 32},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_reference": "JD SGLang7c97bb0 embeddings/engram_hash.py, adapted paged history",
        "layers": 2,
        "max_ngram": 4,
        "heads_per_ngram": 8,
        "max_chunk": 512,
        "history_slot_dtype": "int32",
        "mask_store_packed_bytes": 4,
        "argument_signature": list(signature.items()),
        "kernels": records,
        "status": "EXPORTED_NOT_YET_VALIDATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Exported fused hash and ordered paged-history commit: %s", args.output)


if __name__ == "__main__":
    main()
