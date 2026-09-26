#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Run inside wjm_sg_dspark; preserve package/model/cache donors read-only."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "logs/goal_sgl_case55_same_host_20260925")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / "logs"):
        parser.error("run must remain in checkout logs")
    cache = ROOT / "build/sgl_prefill_probe_cache_20260925"
    if run.exists():
        raise ValueError("refusing to reuse an experiment directory")
    run.mkdir(parents=True)
    (run / "dump").mkdir()
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    donors = {"/root/.cache/mate": cache / ".cache/mate",
              "/root/.cache/tvm-ffi": cache / "tvm-ffi",
              "/root/.tilelang/cache": cache / "tilelang",
              "/root/.triton": cache / "triton"}
    for source, dest in donors.items():
        if Path(source).is_dir() and not dest.exists():
            shutil.copytree(source, dest, symlinks=True)
    env = dict(os.environ)
    reference = Path("/data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/start_server_tp8.sh")
    for line in reference.read_text().splitlines():
        match = re.fullmatch(r"export (SGLANG_[A-Z0-9_]+)=([a-zA-Z0-9_.]+)", line)
        if match:
            env[match[1]] = match[2]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", MATE_MUSA_ARCH_LIST="3.1",
               MATE_WORKSPACE_BASE=str(cache), TVM_FFI_CACHE_DIR=str(cache / "tvm-ffi"),
               TILELANG_CACHE_DIR=str(cache / "tilelang"), TILELANG_TMP_DIR=str(cache / "tmp"),
               TRITON_CACHE_DIR=str(cache / "triton/cache"), TMPDIR=str(cache / "tmp"),
               XDG_CACHE_HOME=str(cache / ".cache"), SGLANG_CACHE_DIR=str(cache / "sglang"),
               TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
               HF_HOME=str(cache / "huggingface"), XLLM_SGL_PREFILL_PROBE_DIR=str(run / "dump"))
    command = [sys.executable, str(ROOT / "tools/sgl_dsv4_prefill_probe_server.py"),
               "--trust-remote-code", "--model-path", "/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt",
               "--tp", "8", "--mem-fraction-static", "0.8", "--host", "127.0.0.1", "--port", "24450",
               "--chunked-prefill-size", "8192", "--disable-overlap-schedule",
               "--tokenizer-backend", "fastokens", "--watchdog-timeout", "6000",
               "--cuda-graph-max-bs", "1", "--max-running-requests", "1"]
    if args.eager:
        command.append("--disable-cuda-graph")
    (run / "launch.json").write_text(json.dumps({"argv": command, "pid": os.getpid(),
        "environment": {k: v for k, v in env.items() if k.startswith(("SGLANG_", "MATE_", "TILELANG_", "TVM_FFI_", "XLLM_"))},
        "purpose": "same-host Prefill diagnostic, not an accuracy or performance benchmark"}, indent=2) + "\n")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
