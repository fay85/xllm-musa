#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Warm HTTP threads, replay canonical perf, then diagnose the first20 GSM8K rows."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from replay_dspark_perf import stream_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / "logs") or not (run / "launch_manifest.txt").is_file():
        parser.error("run must identify an existing launched service inside this checkout")
    if (run / "http_warmup.json").exists():
        parser.error("refusing to overwrite an existing diagnostic run")
    base_url = args.url.rstrip("/")
    for attempt in range(300):
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if attempt % 15 == 0:
            logger.info("Waiting on existing service, attempt=%d", attempt)
        time.sleep(2)
    else:
        raise SystemExit("readiness deadline; inspect existing ranks before any restart")
    warmups = []
    for _ in range(4):
        payload = {"model": "DeepSeek-V4-Flash-FP8",
                   "messages": [{"role": "user", "content": "Reply with the word Ready."}],
                   "chat_template_kwargs": {"thinking": False}, "temperature": 0,
                   "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True}}
        warmups.append(stream_request(base_url + "/v1/chat/completions", payload, 0))
    (run / "http_warmup.json").write_text(json.dumps(
        {"diagnostic_only": True, "requests": warmups}, indent=2) + "\n")
    base = [sys.executable, "tools/replay_dspark_perf.py", "--url",
            base_url + "/v1/chat/completions", "--input-dir",
            "logs/goal_repro_q6_legacy_perf_20260923"]
    for name, first, last in [("warmup", "0", "1"), ("formal", "2", "7")]:
        subprocess.run(base + ["--output-dir", str(run / name), "--first", first, "--last", last],
                       cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/replay_gsm8k_sgl_cases.py", "--source",
                    "logs/goal_hc_post_fix_full_live_20260925/responses.raw.jsonl",
                    "--indices", ",".join(str(i) for i in range(20)), "--output",
                    str(run / "first20.jsonl"), "--url", base_url + "/v1/chat/completions",
                    "--timeout", "120"], cwd=ROOT, check=True)
    logger.info("Perf and diagnostic first20 complete; full GSM8K acceptance is still separate")


if __name__ == "__main__":
    main()
