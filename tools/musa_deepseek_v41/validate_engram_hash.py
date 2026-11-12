# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require exact hash IDs/masks/history for paged C1 and PD lookback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _read_metadata(path: Path) -> dict[str, torch.Tensor]:
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    values = {}
    for name in ("token_map", "multipliers", "primes", "offsets"):
        spec = header[name]
        begin, end = spec["data_offsets"]
        values[name] = (
            torch.frombuffer(
                bytearray(raw[8 + length + begin : 8 + length + end]), dtype=torch.int64
            )
            .reshape(spec["shape"])
            .clone()
        )
    return values


def _reference(
    assets: dict[str, torch.Tensor],
    ids: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    history: torch.Tensor,
    start: int,
    lookback: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mapping, mult, primes, offsets = (
        assets[n] for n in ("token_map", "multipliers", "primes", "offsets")
    )
    pad = int(mapping[2])

    def _map(raw: int) -> int:
        return (
            -1
            if raw in (129264, 129265)
            else int(mapping[min(max(raw, 0), mapping.numel() - 1)])
        )

    def _slot(position: int) -> int:
        return int(table[0, (position // 32) % table.shape[1]]) * 32 + position % 32

    tokens = [_map(int(raw)) for raw in ids]
    output = torch.empty((ids.numel(), 2, 24), dtype=torch.int32)
    for row in range(ids.numel()):
        for layer in range(2):
            rolling, blocked = 0, False
            for shift in range(4):
                previous = int(positions[row]) - shift
                if previous >= start:
                    value = tokens[min(max(row - shift, 0), len(tokens) - 1)]
                else:
                    slot = _slot(max(previous, 0))
                    value = (
                        int(history[min(max(slot, 0), history.numel() - 1)])
                        if slot >= 0
                        else -1
                    )
                    column = start - 1 - previous
                    if (
                        lookback is not None
                        and 0 <= column < lookback.shape[1]
                        and int(lookback[0, column]) >= 0
                    ):
                        value = _map(int(lookback[0, column]))
                blocked = blocked or previous < 0 or value == -1
                value = pad if blocked else value
                rolling ^= (value * int(mult[layer, shift])) & ((1 << 64) - 1)
                signed = rolling if rolling < (1 << 63) else rolling - (1 << 64)
                if shift:
                    for head in range(8):
                        output[row, layer, (shift - 1) * 8 + head] = signed % int(
                            primes[layer, shift - 1, head]
                        ) + int(offsets[layer, (shift - 1) * 8 + head])
    committed = history.clone()
    for row, token in enumerate(tokens):
        slot = _slot(int(positions[row]))
        if slot >= 0:
            committed[slot] = token
    active = (ids != 129264) & (ids != 129265)
    return output, active, committed


def _time(function: Callable[[], Any]) -> float:
    for _ in range(2):
        function()
    torch.musa.synchronize()
    start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(10):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(111212)
    assets = _read_metadata(args.metadata)
    device = torch.device("musa:0")
    gpu = {k: v.to(device) for k, v in assets.items()}
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for dtype in (torch.int32, torch.int64):
            for pages in ([2, 0, 3, 1], [3, -1, 0, 3]):
                table = torch.tensor([pages], dtype=dtype)
                for start, chunks, lookback in (
                    (0, [512], None),
                    (0, [1, 1, 15, 128, 1, 33, 256, 77], None),
                    (
                        159,
                        [1, 17, 128, 256],
                        torch.tensor([[17, -1, 129264]], dtype=dtype),
                    ),
                    (63, [0, 1, 17, 33], torch.tensor([[31]], dtype=dtype)),
                ):
                    ids = torch.randint(
                        0, assets["token_map"].numel(), (sum(chunks),), dtype=dtype
                    )
                    if ids.numel() >= 4:
                        ids[:4] = torch.tensor(
                            [-7, 129264, 129265, 129303], dtype=dtype
                        )
                    original = torch.randint(-1, 99092, (128,), dtype=torch.int32)
                    expected_history = original.clone()
                    h_reference, h_target = original.to(device), original.to(device)
                    params = [
                        gpu[n]
                        for n in ("token_map", "multipliers", "primes", "offsets")
                    ]
                    reference = native.NativeV41EngramHash(
                        *params, int(assets["token_map"][2])
                    )
                    target = native.NativeV41EngramHash(
                        *params,
                        int(assets["token_map"][2]),
                        str(args.library_directory),
                    )
                    reference.bind_history(h_reference)
                    target.bind_history(h_target)
                    offset = 0
                    for size in chunks:
                        batch = ids[offset : offset + size]
                        positions = torch.arange(
                            start + offset, start + offset + size, dtype=dtype
                        )
                        previous_history = expected_history.clone()
                        expected, active, expected_history = _reference(
                            assets,
                            batch,
                            positions,
                            table,
                            expected_history,
                            start + offset,
                            lookback,
                        )
                        arguments = (
                            batch.to(device),
                            positions.to(device),
                            table.to(device),
                            32,
                            start + offset,
                            lookback.to(device) if lookback is not None else None,
                        )
                        cpp = reference.forward(*arguments)
                        actual = target.forward(*arguments)
                        for name, values, history in (
                            ("reference", cpp, h_reference),
                            ("aot", actual, h_target),
                        ):
                            observed_hash = values[0].cpu()
                            if not torch.equal(observed_hash, expected):
                                mismatch = torch.nonzero(observed_hash != expected)
                                args.output.write_text(
                                    json.dumps(
                                        {
                                            "status": "FAIL_HASH",
                                            "implementation": name,
                                            "dtype": str(dtype),
                                            "pages": pages,
                                            "start": start + offset,
                                            "rows": size,
                                            "mismatch_indices": mismatch[:100].tolist(),
                                            "expected_first_rows": expected[
                                                :4
                                            ].tolist(),
                                            "observed_first_rows": observed_hash[
                                                :4
                                            ].tolist(),
                                        },
                                        indent=2,
                                    )
                                )
                                torch.save(
                                    {
                                        "assets": assets,
                                        "ids": batch,
                                        "positions": positions,
                                        "table": table,
                                        "history": previous_history,
                                        "start": start + offset,
                                        "lookback": lookback,
                                    },
                                    args.output.with_suffix(".fixture.pt"),
                                )
                                logger.error(
                                    "%s hash differs, M%d start%d",
                                    name,
                                    size,
                                    start + offset,
                                )
                            observed_history = history.cpu()
                            if not torch.equal(observed_history, expected_history):
                                args.output.write_text(
                                    json.dumps(
                                        {
                                            "status": "FAIL_HISTORY",
                                            "implementation": name,
                                            "dtype": str(dtype),
                                            "pages": pages,
                                            "start": start + offset,
                                            "rows": size,
                                            "input_ids": batch.tolist(),
                                            "expected_history": expected_history.tolist(),
                                            "observed_history": observed_history.tolist(),
                                            "original_history": original.tolist(),
                                        },
                                        indent=2,
                                    )
                                )
                                logger.error(
                                    "%s history differs, M%d start%d",
                                    name,
                                    size,
                                    start + offset,
                                )
                            observed_mask = values[1].cpu()
                            if not torch.equal(observed_mask, active):
                                mismatch = torch.nonzero(
                                    observed_mask != active
                                ).flatten()
                                diagnostic = {
                                    "status": "FAIL_ACTIVE_MASK",
                                    "implementation": name,
                                    "dtype": str(dtype),
                                    "pages": pages,
                                    "start": start + offset,
                                    "rows": size,
                                    "input_ids": batch.tolist(),
                                    "expected_active": active.tolist(),
                                    "observed_active": observed_mask.tolist(),
                                    "mismatch_indices": mismatch.tolist(),
                                }
                                args.output.write_text(json.dumps(diagnostic, indent=2))
                                logger.error(
                                    "%s mask differs at %s",
                                    name,
                                    mismatch[:32].tolist(),
                                )
                            torch.testing.assert_close(
                                observed_hash, expected, atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                observed_mask, active, atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                observed_history, expected_history, atol=0, rtol=0
                            )
                        records.append(
                            {
                                "dtype": str(dtype),
                                "pages": pages,
                                "start": start + offset,
                                "rows": size,
                                "lookback": (
                                    lookback.tolist() if lookback is not None else None
                                ),
                                "hash_mask_history_exact": True,
                            }
                        )
                        offset += size
                    logger.info(
                        "Paged hash dtype%s pages%s start%d chunks%s exact PASS",
                        dtype,
                        pages,
                        start,
                        chunks,
                    )
                    args.output.write_text(
                        json.dumps(
                            {"status": "IN_PROGRESS", "cases": records}, indent=2
                        )
                    )
        timings = []
        table = torch.tensor([[2, 0, 3, 1]], device=device, dtype=torch.int64)
        for rows in (1, 17, 128, 256, 512):
            ids = torch.randint(0, 129264, (rows,), device=device)
            pos = torch.arange(64, 64 + rows, device=device)
            ref = native.NativeV41EngramHash(*params, int(assets["token_map"][2]))
            target = native.NativeV41EngramHash(
                *params, int(assets["token_map"][2]), str(args.library_directory)
            )
            ref.bind_history(torch.zeros(128, device=device, dtype=torch.int32))
            target.bind_history(torch.zeros(128, device=device, dtype=torch.int32))
            baseline = _time(lambda: ref.forward(ids, pos, table, 32, 64, None))
            fused = _time(lambda: target.forward(ids, pos, table, 32, 64, None))
            timings.append(
                {
                    "rows": rows,
                    "reference_gpu_ms": baseline,
                    "native_gpu_ms": fused,
                    "speedup": baseline / fused,
                    "warmups": 2,
                    "repeats": 10,
                }
            )
            logger.info("Paged hash M%d %.5f -> %.5f ms", rows, baseline, fused)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_PAGED_ENGRAM_HASH",
                "metadata_sha256": hashlib.sha256(
                    args.metadata.read_bytes()
                ).hexdigest(),
                "cases": records,
                "timings": timings,
                "scope": "hash+mask+history with real tokenizer metadata and independent CPU integer oracle; not full-model accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
