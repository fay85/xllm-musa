#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Compare full xLLM prompt/token IDs against SGLang's actual CPU encoder."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xllm-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        parser.error("output must be a new file below this xLLM checkout")
    from sglang.srt.entrypoints.openai import encoding_dsv4
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model, tokenizer_backend="fastokens",
                              trust_remote_code=True)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    xllm = {row["id"]: row for row in map(json.loads,
            args.xllm_prompts.read_text().splitlines())}
    template = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{problem}"
    mismatches = []
    prompt_dump = args.output.with_suffix(".prompts.jsonl")
    if prompt_dump.exists():
        parser.error("refusing to overwrite prompt snapshot")
    with prompt_dump.open("w") as output:
        for index, row in enumerate(rows):
            messages = [{"role": "system", "content": ""},
                        {"role": "user", "content": template.format(problem=row["problem"])}]
            rendered = encoding_dsv4.encode_messages(messages, thinking_mode="chat",
                                                     reasoning_effort=None)
            token_ids = tokenizer.encode(rendered)
            identity = f"gsm8k-{index}"
            observed = xllm[identity]
            match = (rendered == observed["rendered_prompt"] and
                     token_ids == observed["token_ids"])
            if not match:
                mismatches.append(identity)
            output.write(json.dumps({"id": identity, "rendered_prompt": rendered,
                                     "token_ids": token_ids, "matches_xllm": match},
                                    ensure_ascii=False) + "\n")
    encoder_path = Path(inspect.getfile(encoding_dsv4))
    summary = {"examples": len(rows), "matched": len(rows) - len(mismatches),
               "mismatch_ids": mismatches, "thinking_mode": "chat",
               "tokenizer_backend": "fastokens", "reference_encoder": str(encoder_path),
               "reference_encoder_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
               "xllm_dump_sha256": hashlib.sha256(args.xllm_prompts.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Prompt/token audit: %s", summary)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
