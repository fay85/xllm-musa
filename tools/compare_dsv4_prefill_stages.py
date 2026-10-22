#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Align xLLM SP rank7 rows with SGLang TP0 heads for the same real prompt."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _cpp(path: Path) -> dict:
    return torch.jit.load(str(path), map_location="cpu").state_dict()


def _python(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    if actual.shape != reference.shape:
        raise ValueError(f"unaligned shapes {actual.shape} vs {reference.shape}")
    a, b = actual.float(), reference.float()
    delta = a - b
    return {"shape": list(a.shape), "relative_l2": (delta.norm() / b.norm().clamp_min(1e-30)).item(),
            "max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xllm", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error("output must be new and inside checkout")
    torch.set_num_threads(4)
    rows = []
    for layer in range(43):
        x = _cpp(args.xllm / f"layer_{layer}_rank7.pt")
        s = _python(args.sglang / f"layer_{layer}_prepare_rank0.pt")
        start = int(x["metadata"][0])
        end = start + x["q"].shape[0]
        heads = s["q"].shape[1]
        record = {"layer": layer, "query_start": start, "query_end": end,
                  "q": _error(x["q"][:, :heads], s["q"][start:end]),
                  "core": _error(x["output"][:, :heads],
                                 _python(args.sglang / f"layer_{layer}_core_rank0.pt")["output"][start:end])}
        record["input"] = _error(_cpp(args.xllm / f"layer_{layer}_input_rank7.pt")["value"],
                                 s["input"][start:end])
        record["attention_output"] = _error(
            _cpp(args.xllm / f"layer_{layer}_attention_rank7.pt")["value"],
            _python(args.sglang / f"layer_{layer}_attention_rank0.pt")["output"][start:end])
        rows.append(record)
        logger.info("layer=%d input=%.6f q=%.6f core=%.6f attention_out=%.6f",
                    layer, record["input"]["relative_l2"], record["q"]["relative_l2"],
                    record["core"]["relative_l2"], record["attention_output"]["relative_l2"])
    args.output.write_text(json.dumps({"xllm": str(args.xllm), "sglang": str(args.sglang),
        "alignment": "SP rank7 valid absolute query rows; TP0 heads 0..7; prefill only",
        "layers": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
