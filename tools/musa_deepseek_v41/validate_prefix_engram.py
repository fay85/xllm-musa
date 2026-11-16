# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual AOT Engram hash and history across selective speculative commits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import torch
import torch_musa
import v41_prefix_checkpoint_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--library-directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111636)
    raw = args.metadata.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    assets = []
    for name in ("token_map", "multipliers", "primes", "offsets"):
        item = header[name]
        first, last = item["data_offsets"]
        assets.append(torch.frombuffer(bytearray(raw[8 + length + first:8 + length + last]), dtype=torch.int64).reshape(item["shape"]).to("musa:0"))
    pad = int(assets[0][2].cpu())
    table = torch.tensor([[3, 1]], device="musa:0", dtype=torch.int32)
    records = []
    with torch.inference_mode():
        for nondefault in (False, True):
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            stream.wait_stream(torch.musa.default_stream())
            with torch.musa.stream(stream):
                for begin in (0, 31, 127, 128, 255, 257):
                    for accepted in range(7):
                        reference = native.NativeV41EngramHash(*assets, pad, args.library_directory)
                        target = native.NativeV41EngramHash(*assets, pad, args.library_directory)
                        original = torch.randint(-1, 50000, (5 * 128,), device="musa:0", dtype=torch.int32)
                        expected, actual = original.clone(), original.clone()
                        reference.bind_history(expected)
                        target.bind_history(actual)
                        ids = torch.randint(0, assets[0].numel(), (7,), device="musa:0")
                        positions = torch.arange(begin, begin + 7, device="musa:0")
                        checkpoint = native.PrefixCheckpoint(begin, 6, 4096)
                        checkpoint.add_paged(actual.view(5, 128, 1), table, 1, True)
                        out, mask = target.forward(ids[:6], positions[:6], table, 128, begin, None)
                        checkpoint.restore_suffix(accepted)
                        if accepted:
                            wanted, wanted_mask = reference.forward(ids[:accepted], positions[:accepted], table, 128, begin, None)
                            assert torch.equal(out[:accepted].cpu(), wanted.cpu())
                            assert torch.equal(mask[:accepted].cpu(), wanted_mask.cpu())
                        assert torch.equal(actual.cpu(), expected.cpu()), (begin, accepted)
                        next_expected = reference.forward(ids[accepted:accepted + 1], positions[accepted:accepted + 1], table, 128, begin + accepted, None)
                        next_actual = target.forward(ids[accepted:accepted + 1], positions[accepted:accepted + 1], table, 128, begin + accepted, None)
                        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(next_actual, next_expected))
                        assert torch.equal(actual.cpu(), expected.cpu())
                        records.append({"begin": begin, "accepted_inputs": accepted, "nondefault": nondefault, "hash_mask_history_next_token_exact": True, "saved_bytes": checkpoint.saved_bytes()})
            stream.synchronize()
    args.output.write_text(json.dumps({"status": "PASS_ACTUAL_AOT_ENGRAM_PREFIX_RESTORE", "cases": records, "accuracy_scope": "仅通过基础功能检查，未完成模型精度验收"}, indent=2))
    logger.info("Actual Engram prefix transaction cases: %d", len(records))


if __name__ == "__main__":
    main()
