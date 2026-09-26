#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Locate first divergence with reference-free prefill/continuation probes."""

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error("output must be a new checkout artifact")
    prompts = {r["id"]: r for r in map(json.loads, args.prompts.read_text().splitlines())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", buffering=1) as output:
        for index in map(int, args.indices.split(",")):
            identity = f"gsm8k-{index}"
            row = prompts[identity]
            prefix = []
            # Bypass chat rendering once using the audited exact token IDs.
            stages = (("prefill_1", 1),) if args.prefill_only else (
                ("prefill_1", 1), ("continuation_32", 32))
            for stage, limit in stages:
                payload = {"model": args.model, "prompt": row["token_ids"],
                           "temperature": 0, "top_p": 0.95, "max_tokens": limit,
                           "echo": False, "logprobs": 5, "skip_special_tokens": False}
                started = time.time()
                response = _post(args.url, payload)
                output.write(json.dumps({"id": identity, "stage": stage, "payload": payload,
                                         "response": response, "latency": time.time() - started},
                                        ensure_ascii=False) + "\n")
                logger.info("%s %s %s", identity, stage, response.get("choices"))
                if stage == "continuation_32":
                    prefix = response["choices"][0].get("logprobs", {}).get("token_ids", [])
            # Re-evaluate each prefix as a prefill. Never feed a reference answer.
            for offset in range(min(5, len(prefix))):
                payload = {"model": args.model,
                           "prompt": row["token_ids"] + prefix[:offset],
                           "temperature": 0, "top_p": 0.95, "max_tokens": 1,
                           "echo": False, "logprobs": 5, "skip_special_tokens": False}
                response = _post(args.url, payload)
                output.write(json.dumps({"id": identity, "stage": "forced_own_prefix",
                                         "offset": offset, "continuation_token": prefix[offset],
                                         "payload": payload, "response": response}, ensure_ascii=False) + "\n")
                logger.info("%s offset=%s expected-from-own-decode=%s response=%s",
                            identity, offset, prefix[offset], response.get("choices"))


if __name__ == "__main__":
    main()
