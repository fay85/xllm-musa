# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare full H64 Attention projection with eight native H8 TP slices."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
import torch_musa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt()
    relative = rms / expected.float().square().mean().sqrt().clamp_min(1e-12)
    if not torch.isfinite(actual).all() or relative.item() > 0.02:
        raise AssertionError(f"Projection layout mismatch: relative RMS {relative}")
    return {"relative_rms": relative.item(), "max_abs": delta.abs().max().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="v41_attention_layout_test_module")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    native = importlib.import_module(args.module)
    torch.manual_seed(111604)
    torch.set_num_threads(1)
    device, dtype = "musa:0", torch.bfloat16
    shapes = [(1280, 5120), (1280,), (32768, 1280), (512, 5120),
              (512,), (8192, 4096), (5120, 8192)]
    weights = [torch.ones(s, device=device, dtype=dtype) if len(s) == 1 else
               (torch.randn(s, device=device, dtype=dtype) * 0.02).contiguous()
               for s in shapes]
    whole = native.NativeV41AttentionProjection(weights, 64, 8, 1024, 1e-6, True, False)
    angle = torch.randn(512, 32, device=device)
    rope = torch.cat([angle.cos(), angle.sin()], -1).float().contiguous()
    cases = []
    with torch.inference_mode():
        for rows in (1, 3):
            hidden = torch.randn(rows, 5120, device=device, dtype=dtype)
            positions = torch.arange(rows, device=device, dtype=torch.int64) + 257
            q_full, kv_full, qr_full = whole.prepare(hidden, positions, rope)
            attention = torch.randn(rows, 64, 512, device=device, dtype=dtype)
            full_output = whole.project_output(attention, positions, rope)
            queries, outputs, duplicate_errors = [], [], []
            for rank in range(8):
                sliced = list(weights)
                sliced[2] = weights[2].narrow(0, rank * 4096, 4096).contiguous()
                sliced[5] = weights[5].narrow(0, rank * 1024, 1024).contiguous()
                sliced[6] = weights[6].narrow(1, rank * 1024, 1024).contiguous()
                part = native.NativeV41AttentionProjection(
                    sliced, 8, 1, 1024, 1e-6, True, False
                )
                q, kv, qr = part.prepare(hidden, positions, rope)
                queries.append(q)
                duplicate_errors.append({"kv": _error(kv, kv_full), "qr": _error(qr, qr_full)})
                outputs.append(part.project_output(
                    attention.narrow(1, rank * 8, 8).contiguous(), positions, rope
                ))
            torch.musa.synchronize()
            cases.append({
                "rows": rows,
                "query": _error(torch.cat(queries, 1), q_full),
                "output_sum": _error(torch.stack(outputs).float().sum(0).to(dtype), full_output),
                "replicated_projection_checks": duplicate_errors,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "PASS_ATTENTION_REPLICATION_PROJECTIONS", "cases": cases,
        "scope": "Actual native projection code, FP8 block32 QDQ retained; H8 outputs summed locally on GPU. Does not substitute for 8-rank communication or full-model tests."
    }, indent=2))
    logger.info("H64 versus eight H8 projection cases passed: %d", len(cases))


if __name__ == "__main__":
    main()
