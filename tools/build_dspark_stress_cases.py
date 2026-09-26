#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Build unique, exact native-chat token lengths before measuring the server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger

MODEL = Path("/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, default=30000)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--kind", choices=("counting", "gsm8k"), default="counting")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "logs") or out.exists():
        parser.error("output must be a new dataset inside logs")
    if args.count <= 0 or args.output_tokens < 2 or args.output_tokens % 2:
        parser.error("positive count and even output token limit required")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))

    def token_count(content: str) -> int:
        return len(tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>" + content +
                                    "<｜Assistant｜></think>", add_special_tokens=False).ids)

    gsm = []
    if args.kind == "gsm8k":
        source = Path("/data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/dataset_test.jsonl")
        gsm = [json.loads(line) for line in source.read_text().splitlines()]
        if args.count > len(gsm):
            parser.error("not enough unique GSM8K rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    hashes = set()
    with out.open("x") as stream:
        for index in range(args.count):
            row = {"id": f"stress-{index:04d}", "kind": args.kind}
            if gsm:
                example = gsm[index]
                content = ("Solve the following math problem. Make sure to put the answer "
                           "(and only answer) inside \\boxed{}.\n\n" + example["problem"])
                row.update(id=f"gsm8k-{index}", problem=example["problem"],
                           expected_answer=example["expected_answer"])
            else:
                # A distinct marker at the START prevents long shared-prefix
                # hits. Vary the counting range so cross-request mixing is visible.
                start = 1 + index % max(1, 999 - args.output_tokens // 2)
                end = start + args.output_tokens // 2 - 1
                header = f"Request {index:04d}, context marker {hashlib.sha256(str(index).encode()).hexdigest()[:16]}.\n"
                instruction = (f"\nIgnore the background. Output only consecutive integers from {start} "
                               f"through {end}, separated by one ASCII space. No explanations or punctuation.")
                filler = max(0, args.input_tokens - token_count(header + instruction))
                for _ in range(12):
                    content = header + "x " * filler + instruction
                    delta = args.input_tokens - token_count(content)
                    if delta == 0:
                        break
                    filler = max(0, filler + delta)
                if token_count(content) != args.input_tokens:
                    raise ValueError(f"cannot build exact prompt {index}")
                row.update(count_from=start, count_to=end)
            payload = {"model": "DeepSeek-V4-Flash-FP8", "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"thinking": False}, "temperature": 0,
                       "max_tokens": args.output_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}
            if gsm:
                payload["top_p"] = 0.95
            else:
                payload["ignore_eos"] = True
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in hashes:
                raise ValueError("duplicate input")
            hashes.add(digest)
            row.update(request_payload=payload, prompt_tokens_built=token_count(content),
                       prompt_sha256=digest, output_tokens_requested=args.output_tokens)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0:
                logger.info("Built %d/%d exact cases", index + 1, args.count)
    meta = {"kind": args.kind, "count": args.count, "unique_prompts": len(hashes),
            "input_tokens": args.input_tokens if not gsm else "variable GSM8K",
            "output_token_limit": args.output_tokens, "tokenizer": str(MODEL / "tokenizer.json"),
            "dataset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "scope": "Synthetic counting stability, not long-context reasoning accuracy" if not gsm else "GSM8K"}
    out.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Dataset ready: %s", meta)


if __name__ == "__main__":
    main()
