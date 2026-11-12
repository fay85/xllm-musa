# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify C2 pool rounding, paged history and complete compressor timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _oracle(
    values: torch.Tensor,
    scores: torch.Tensor,
    table: torch.Tensor,
    value_state: torch.Tensor,
    score_state: torch.Tensor,
    start: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def _slot(position: int) -> int:
        page = max(int(table[0, (position // block_size) % table.shape[1]]), 0)
        return page * 32 + position % 32

    dimension = values.shape[1]
    parked_v = value_state.reshape(-1, dimension)
    parked_s = score_state.reshape(-1, dimension)
    outputs, positions = [], []
    for position in range(start // 2 * 2, (start + values.shape[0]) // 2 * 2, 2):
        members_v, members_s = [], []
        for member in (position, position + 1):
            row = member - start
            members_v.append(values[row] if row >= 0 else parked_v[_slot(member)])
            members_s.append(scores[row] if row >= 0 else parked_s[_slot(member)])
        pair_v, pair_s = torch.stack(members_v), torch.stack(members_s)
        outputs.append((pair_v * pair_s.softmax(0)).sum(0).to(torch.bfloat16))
        positions.append(position)
    out_v, out_s = value_state.clone(), score_state.clone()
    for row in range(max(0, values.shape[0] - 32), values.shape[0]):
        position = start + row
        if position % 2 == 0:
            out_v.view(-1, dimension)[_slot(position)] = values[row]
            out_s.view(-1, dimension)[_slot(position)] = scores[row]
    latent = (
        torch.stack(outputs)
        if outputs
        else torch.empty((0, dimension), dtype=torch.bfloat16)
    )
    return latent, torch.tensor(positions, dtype=torch.int64), out_v, out_s


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


def _rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float(
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-10)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(111227)
    device = torch.device("musa:0")
    records, timings = [], []
    with torch.inference_mode():
        for dimension in (128, 512):
            for block_size in (32, 128):
                for start, count in (
                    (0, 0),
                    (0, 1),
                    (0, 17),
                    (1, 1),
                    (31, 33),
                    (159, 128),
                    (1, 256),
                    (159, 512),
                ):
                    for amplitude in (1.0, 100.0):
                        table = torch.tensor([[3, -1, 0, 3]], dtype=torch.int32)
                        values = torch.randn(count, dimension)
                        scores = torch.randn(count, dimension) * amplitude
                        states = [torch.randn(4, 32, dimension) for _ in range(2)]
                        expected, positions, ev, es = _oracle(
                            values, scores, table, *states, start, block_size
                        )
                        actual_states = [state.to(device) for state in states]
                        actual, actual_pos = native.v41_pair_pool(
                            values.to(device),
                            scores.to(device),
                            table.to(device),
                            *actual_states,
                            start,
                            block_size,
                        )
                        observed = actual.cpu()
                        error = _rms(observed, expected)
                        if error > 0.001 or not torch.isfinite(observed).all():
                            raise AssertionError(
                                f"Pool D{dimension} start{start} M{count}: RMS {error}"
                            )
                        torch.testing.assert_close(
                            actual_pos.cpu(), positions, atol=0, rtol=0
                        )
                        for got, want in zip(actual_states, (ev, es)):
                            torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)
                        records.append(
                            {
                                "dimension": dimension,
                                "block_size": block_size,
                                "start": start,
                                "rows": count,
                                "score_amplitude": amplitude,
                                "relative_rms": error,
                                "state_and_positions_exact": True,
                            }
                        )
            # Complete production class: includes FP32 projections and RMSNorm.
            weight = torch.randn(dimension, 128, device=device) / 16
            gate = torch.randn_like(weight) / 8
            norm = torch.randn(dimension, device=device, dtype=torch.bfloat16)
            table = torch.tensor([[3, 1, 0, 2]], device=device, dtype=torch.int64)
            for count in (1, 17, 128, 256):
                value = torch.randn(count, 128, device=device, dtype=torch.bfloat16)
                initial_v = torch.randn(4, 32, dimension, device=device)
                initial_s = torch.randn_like(initial_v)
                reference = native.NativeV41Compressor(
                    weight,
                    norm,
                    gate,
                    initial_v.clone(),
                    initial_s.clone(),
                    2,
                    128,
                    1e-6,
                    False,
                )
                target = native.NativeV41Compressor(
                    weight,
                    norm,
                    gate,
                    initial_v.clone(),
                    initial_s.clone(),
                    2,
                    128,
                    1e-6,
                    True,
                )
                expected, expected_pos = reference.forward(value, 159, table)
                actual, actual_pos = target.forward(value, 159, table)
                error = _rms(actual.cpu(), expected.cpu())
                if error > 0.003:
                    raise AssertionError(
                        f"Complete C2 D{dimension} M{count}: RMS {error}"
                    )
                torch.testing.assert_close(
                    actual_pos.cpu(), expected_pos.cpu(), atol=0, rtol=0
                )
                baseline = _time(lambda: reference.forward(value, 159, table))
                fused = _time(lambda: target.forward(value, 159, table))
                timings.append(
                    {
                        "dimension": dimension,
                        "rows": count,
                        "relative_rms": error,
                        "reference_gpu_ms": baseline,
                        "native_gpu_ms": fused,
                        "warmups": 2,
                        "repeats": 10,
                        "scope": "complete C2 projections+pool+norm+state; fixed numerical fixture, not full-model speedup",
                    }
                )
                logger.info(
                    "C2 D%d M%d RMS%.6g %.5f -> %.5f ms",
                    dimension,
                    count,
                    error,
                    baseline,
                    fused,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"status": "PASS_PAGED_PAIR_POOL", "cases": records, "timings": timings},
            indent=2,
        )
    )
    logger.info(
        "C2 pool %d paged cases and %d complete-class cases PASS",
        len(records),
        len(timings),
    )


if __name__ == "__main__":
    main()
