#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Replay selected raw benchmark payloads for controlled diagnostics only."""

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    source = {row["id"]: row for row in
              (json.loads(line) for line in args.source.read_text().splitlines())}
    indices = [int(value) for value in args.indices.split(",")]
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", buffering=1) as output:
        for index in indices:
            item = source[f"gsm8k-{index}"]
            payload = item["request_payload"]
            started = time.time()
            result = {"id": item["id"], "request_payload": payload,
                      "expected_answer": item["expected_answer"], "problem": item["problem"],
                      "source": str(args.source), "diagnostic_only": True}
            try:
                request = urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    body = json.loads(response.read())
                choice = body["choices"][0]
                result.update(response=body, generation=choice["message"].get("content", ""),
                              finish_reason=choice.get("finish_reason"), usage=body.get("usage", {}))
            except Exception as error:
                result.update(error=f"{type(error).__name__}: {error}", generation="", finish_reason="error")
            result["latency_seconds"] = time.time() - started
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info("%s finish=%s seconds=%.3f output=%r", result["id"], result["finish_reason"],
                        result["latency_seconds"], result["generation"][:180])
            if "error" in result:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
