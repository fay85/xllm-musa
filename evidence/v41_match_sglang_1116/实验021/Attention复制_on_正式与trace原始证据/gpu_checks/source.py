# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check H64 target and H8 draft sparse libraries in one process and graph."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys

import torch
import torch_musa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _check(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    relative = (
        difference.square().mean()
        / expected.float().square().mean().clamp_min(1e-20)
    ).sqrt().item()
    if not torch.isfinite(actual).all() or relative > 0.01:
        raise AssertionError(f"Sparse library pair mismatch: RMS {relative}")
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="v41_attention_layout_test_module")
    parser.add_argument("--h8", type=Path, required=True)
    parser.add_argument("--h64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    native = importlib.import_module(args.module)
    torch.manual_seed(1121)
    torch.set_num_threads(1)
    scale = 1 / math.sqrt(512)
    reference = native.NativeV41SparseAttention(scale, 32 * 1024**2, "")
    target = native.NativeV41SparseAttention(scale, 32 * 1024**2, str(args.h64))
    draft = native.NativeV41SparseAttention(scale, 32 * 1024**2, str(args.h8))
    stream = torch.musa.Stream()
    cases = []
    with torch.inference_mode(), torch.musa.stream(stream):
        for rows in (4, 6):
            query = torch.randn(rows, 64, 512, device="musa", dtype=torch.bfloat16)
            draft_query = query[:, :8].contiguous()
            kv = torch.randn(512, 512, device="musa", dtype=torch.bfloat16)
            picks = torch.randint(1, 512, (rows, 384), device="musa")
            sink = torch.randn(64, device="musa")
            draft_sink = sink[:8].contiguous()
            for _ in range(2):
                target.forward(query, kv, picks, sink)
                draft.forward(draft_query, kv, picks, draft_sink)
            stream.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=stream):
                target_output = target.forward(query, kv, picks, sink)
                draft_output = draft.forward(draft_query, kv, picks, draft_sink)
            for trial in range(4):
                query.copy_(torch.randn_like(query))
                draft_query.copy_(query[:, :8])
                kv.copy_(torch.randn_like(kv))
                kv[0].fill_(float("nan"))
                picks.copy_(torch.randint(1, 512, picks.shape, device="musa"))
                picks[:, ::7] = -1
                picks[:, -1] = 2**40
                if trial == 3:
                    picks.fill_(-1)
                expected = reference.forward(query, kv, picks, sink)
                eager_target = target.forward(query, kv, picks, sink)
                eager_draft = draft.forward(draft_query, kv, picks, draft_sink)
                graph.replay()
                stream.synchronize()
                errors = {
                    "target_eager": _check(eager_target, expected),
                    "draft_eager": _check(eager_draft, expected[:, :8]),
                    "target_graph": _check(target_output, expected),
                    "draft_graph": _check(draft_output, expected[:, :8]),
                }
                cases.append({"rows": rows, "trial": trial, "rms": errors})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "PASS_H64_H8_PAIR",
        "cases": cases,
        "h8": str(args.h8),
        "h64": str(args.h64),
        "scope": "Native sparse API, q4/q6, changed inputs, invalid masks, "
                 "same-process dual libraries and Graph; not model precision.",
    }, indent=2))
    logger.info("H64/H8 sparse pair passed %d cases", len(cases))


if __name__ == "__main__":
    main()
