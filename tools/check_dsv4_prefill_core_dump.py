#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Independent CPU FP64 causal SWA+compressed attention for real input dumps."""

import argparse
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _check(path: Path) -> dict:
    state = torch.jit.load(str(path), map_location="cpu").state_dict()
    q, kv = state["q"].double(), state["kv"].reshape(-1, state["q"].shape[-1]).double()
    actual = state["output"].double()
    start, ratio = map(int, state["metadata"].tolist())
    scale = state["softmax_scale"].item()
    positions = torch.arange(q.shape[0]) + start
    keys = torch.arange(kv.shape[0])
    swa_mask = ((keys[None, :] <= positions[:, None]) &
                (keys[None, :] >= positions[:, None] - 127))
    scores = torch.einsum("thd,sd->hts", q, kv) * scale
    scores.masked_fill_(~swa_mask[None, :, :], -torch.inf)
    values = kv
    compressed = state.get("compressed")
    if compressed is not None and compressed.numel():
        compressed = compressed.reshape(-1, q.shape[-1]).double()
        compressed_mask = torch.arange(compressed.shape[0])[None, :] < (
            (positions[:, None] + 1) // ratio)
        indices = state.get("indices")
        if indices is not None and indices.numel():
            chosen = torch.zeros_like(compressed_mask)
            for row in range(q.shape[0]):
                valid = indices[row][(indices[row] >= 0) & (indices[row] < compressed.shape[0])]
                chosen[row, valid.long()] = True
            compressed_mask &= chosen
        cscores = torch.einsum("thd,sd->hts", q, compressed) * scale
        cscores.masked_fill_(~compressed_mask[None, :, :], -torch.inf)
        scores = torch.cat((scores, cscores), dim=-1)
        values = torch.cat((values, compressed), dim=0)
    sink = state["sink"].double()[:, None, None].expand(q.shape[1], q.shape[0], 1)
    probabilities = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    expected = torch.einsum("hts,sd->thd", probabilities, values)
    difference = actual - expected
    relative = difference.norm().item() / max(expected.norm().item(), 1e-30)
    result = {"file": str(path), "query_shape": list(q.shape), "kv_rows": kv.shape[0],
              "query_start": start, "ratio": ratio, "max_abs": difference.abs().max().item(),
              "mean_abs": difference.abs().mean().item(), "rel_l2": relative,
              "reference_absmax": expected.abs().max().item(),
              "finite": bool(torch.isfinite(actual).all())}
    logger.info("%s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error("output must be a new artifact below this checkout")
    torch.set_num_threads(4)
    paths = sorted((p for p in args.directory.glob("layer_*_rank7.pt")
                    if re.fullmatch(r"layer_\d+_rank7\.pt", p.name)),
                   key=lambda p: int(p.name.split("_")[1]))
    if not paths:
        parser.error("no real prefill dumps found")
    results = [_check(path) for path in paths]
    args.output.write_text(json.dumps({"reference": "independent CPU FP64", "layers": results,
                                     "scope": "Attention core only; not projection/state correctness"},
                                    indent=2) + "\n")


if __name__ == "__main__":
    main()
