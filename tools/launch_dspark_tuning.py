#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Launch an isolated experiment from the recorded, accepted q6 configuration."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/gsm8k_candidates/short_c4_causal_model_20260926/xllm")
    parser.add_argument("--base-run", type=Path, default=ROOT / "logs/goal_gsm95_tpot10_short_c4_20260926")
    parser.add_argument("--port", type=int, default=19466)
    parser.add_argument("--master-port", type=int, default=31010)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--draft-tokens", type=int, default=5)
    parser.add_argument("--set-env", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.draft_tokens <= 16:
        parser.error("draft tokens must be in [1,16]")
    run = args.run_dir.resolve()
    binary = args.binary.resolve()
    if not run.is_relative_to(ROOT / "logs") or run.exists():
        parser.error("run must be a NEW directory inside this checkout's logs")
    if not binary.is_relative_to(ROOT / "build") or not binary.is_file():
        parser.error("binary must exist in this checkout's build directory")
    fields = dict(line.split("=", 1) for line in
                  (args.base_run / "launch_manifest.txt").read_text().splitlines() if "=" in line)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("XLLM_DSV4_", "XLLM_GRAPH_", "XLLM_MUSA_GRAPH_", "XLLM2_EXTERNAL_TP8_"))
           and key != "XLLM_KEEP_HOST_MIRROR_REFRESH"}
    env.update({key: value for key, value in fields.items() if key.startswith("XLLM_DSV4_")})
    env.update(DSPARK_BIN=str(binary), MODEL_PATH=fields["model_path"],
               XLLM2_EXTERNAL_TP8_HTTP_PORT=str(args.port),
               XLLM2_EXTERNAL_TP8_MASTER_ADDR=f"127.0.0.1:{args.master_port}",
               XLLM2_EXTERNAL_TP8_OPS_PATH=fields["ops_path"],
               XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO=fields["custom_ar_so"],
               XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS=str(args.draft_tokens),
               XLLM2_EXTERNAL_TP8_ONLINE_PROFILE="true" if args.profile else "false",
               PYTHONDONTWRITEBYTECODE="1")
    for override in args.set_env:
        key, separator, value = override.partition("=")
        if not separator or not key.startswith(("XLLM_DSV4_", "XLLM_GRAPH_", "XLLM_MUSA_")):
            parser.error("overrides must be named XLLM execution controls with KEY=VALUE")
        env[key] = value
    logger.info("Launching binary=%s run=%s profile=%s", binary, run, args.profile)
    os.chdir(ROOT)
    os.execve("/bin/bash", ["bash", "tools/start_dspark_custom.sh", str(run), "10240", "0.8"], env)


if __name__ == "__main__":
    main()
