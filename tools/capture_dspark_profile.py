#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Capture one warmed DSpark request; profiled timings are never benchmark gates."""

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from replay_dspark_perf import stream_request


def _control(url: str) -> int:
    request = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
        return response.status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="server base URL without /v1")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "logs"):
        parser.error("output must be new and below this checkout's logs")
    if args.max_tokens <= 0 or args.max_tokens % 2:
        parser.error("counting probe needs a positive even token budget")
    source = json.loads(args.record.read_text())
    payload = dict(source["request_payload"])
    payload["max_tokens"] = args.max_tokens
    result = {"diagnostic_only": True, "profiled_timings_not_benchmark": True,
              "request_payload": payload, "source_record": str(args.record)}
    start_attempted = False
    try:
        start_attempted = True
        result["start_status"] = _control(args.url.rstrip("/") + "/start_profile")
        logger.info("Profiler started; issuing one %d-token decode probe", args.max_tokens)
        result["request"] = stream_request(args.url.rstrip("/") + "/v1/chat/completions",
                                            payload, args.max_tokens // 2)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, urllib.error.HTTPError):
            result["error_body"] = error.read().decode(errors="replace")
        logger.exception("Profile capture failed")
    finally:
        # Start can succeed on some TP ranks before another rank fails.
        # Always try to stop that partial collection as well.
        if start_attempted:
            try:
                result["stop_status"] = _control(args.url.rstrip("/") + "/stop_profile")
            except Exception as error:
                result["stop_error"] = f"{type(error).__name__}: {error}"
                logger.exception("Profiler stop failed; inspect existing service")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    logger.info("Profile control record: %s", args.output)
    raise SystemExit(1 if "error" in result or "stop_error" in result else 0)


if __name__ == "__main__":
    main()
