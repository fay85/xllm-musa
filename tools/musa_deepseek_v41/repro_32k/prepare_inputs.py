# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recreate the exact first 48 ShareGPT prompts without publishing the dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads((HERE / "workload.json").read_text())
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["dataset_sha256"]:
        raise ValueError("ShareGPT dataset SHA-256 differs from the measured input")
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != spec["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA-256 differs from the measured input")
    dataset = json.loads(raw)
    del raw
    conversations = (row.get("conversations", row.get("conversation", [])) for row in dataset)
    prompts = [conversation[0]["value"] for conversation in conversations if len(conversation) >= 2]
    del dataset
    random.Random(spec["seed"]).shuffle(prompts)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    width = 32000 - tokenizer.num_special_tokens_to_add(False)
    prepared = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        if not ids:
            continue
        text = tokenizer.decode((ids * ((width + len(ids) - 1) // len(ids)))[:width],
                                skip_special_tokens=False)
        expected = spec["rows"][len(prepared)]
        if (hashlib.sha256(text.encode()).hexdigest() != expected["sha"]
                or len(tokenizer.encode(text).ids) != expected["input_cpu"]):
            raise ValueError(f"Prompt {expected['i']} differs from the measured workload")
        prepared.append(text)
        if len(prepared) == len(spec["rows"]):
            break
    if len(prepared) != len(spec["rows"]):
        raise ValueError("Dataset did not provide all measured prompts")
    args.output.mkdir()
    for i, prompt in enumerate(prepared):
        (args.output / f"{i:03}.txt").write_bytes(prompt.encode())


if __name__ == "__main__":
    main()
