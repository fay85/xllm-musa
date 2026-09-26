#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Test batch/long-context limits without editing the pinned B1 launcher."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--port", type=int, default=19467)
    parser.add_argument("--master-port", type=int, default=31250)
    parser.add_argument("--prefill-chunk", type=int, default=10240)
    parser.add_argument("--prefix-cache", choices=("true", "false"), default="true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists() or not run.is_relative_to(ROOT / "logs"):
        parser.error("use a new run under this checkout's logs")
    if not 1 <= args.batch <= 16 or not 1024 <= args.prefill_chunk <= 32768:
        parser.error("invalid batch or prefill chunk")
    baseline = ROOT / "logs/goal_gsm96_tpot9_n6_groupedstate_20260926/launch_manifest.txt"
    fields = dict(line.split("=", 1) for line in baseline.read_text().splitlines() if "=" in line)
    binary = Path(fields["xllm_bin"])
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != "8228891e3c9a4994c38131a869738a477a6fdef8a581c846ee693a26f77b5c87":
        parser.error("accepted binary changed")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("XLLM_DSV4_", "XLLM_GRAPH_", "XLLM_MUSA_GRAPH_", "XLLM2_EXTERNAL_TP8_"))
           and k not in {"XLLM_KEEP_HOST_MIRROR_REFRESH", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}}
    env.update({k: v for k, v in fields.items() if k.startswith("XLLM_DSV4_")})
    env.update(DSPARK_BIN=str(binary), MODEL_PATH=fields["model_path"],
               PYTHONDONTWRITEBYTECODE="1", XLLM2_EXTERNAL_TP8_HTTP_PORT=str(args.port),
               XLLM2_EXTERNAL_TP8_MASTER_ADDR=f"127.0.0.1:{args.master_port}",
               XLLM2_EXTERNAL_TP8_OPS_PATH=fields["ops_path"],
               XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO=fields["custom_ar_so"],
               XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS="6",
               XLLM2_EXTERNAL_TP8_MAX_SEQS_PER_BATCH=str(args.batch),
               XLLM2_EXTERNAL_TP8_MAX_CONCURRENT_REQUESTS=str(args.batch),
               XLLM2_EXTERNAL_TP8_PREFILL_CHUNK=str(args.prefill_chunk),
               XLLM2_EXTERNAL_TP8_PREFIX_CACHE=args.prefix_cache,
               XLLM2_EXTERNAL_TP8_ONLINE_PROFILE="false")
    logger.info("Pinned binary %s batch=%d prefill_chunk=%d run=%s",
                digest, args.batch, args.prefill_chunk, run)
    os.chdir(ROOT)
    os.execve("/bin/bash", ["bash", "tools/start_dspark_stress.sh", str(run),
                           str(args.prefill_chunk), "0.8"], env)


if __name__ == "__main__":
    main()
