# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Run sparse-attention C++ tests and a real two-device MCCL layer comparison."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.logger import logger


def _single(binary: Path, arguments: list[str], log: Path,
            environment: dict[str, str]) -> None:
    with log.open("w") as output:
        result = subprocess.run(
            [str(binary), *arguments], stdout=output, stderr=subprocess.STDOUT,
            env=environment, timeout=240, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{binary.name} failed with {result.returncode}; see {log}")
    logger.info("Passed %s; log: %s", binary.name, log)


def _tensor_parallel(binary: Path, output_dir: Path,
                     environment: dict[str, str]) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes: list[subprocess.Popen] = []
    with ExitStack() as stack:
        try:
            for rank in range(2):
                child_env = dict(environment)
                child_env.update(XLLM_GLM_TP_RANK=str(rank),
                                 XLLM_GLM_TP_PORT=str(port))
                log = output_dir / f"glm_sparse_mccl_rank{rank}.log"
                output = stack.enter_context(log.open("w"))
                processes.append(subprocess.Popen(
                    [str(binary), "--gtest_filter=GlmSparseTest.McclTensorParallelWorker"],
                    stdout=output, stderr=subprocess.STDOUT, env=child_env,
                ))
            deadline = time.monotonic() + 180
            statuses = [process.wait(timeout=max(1, deadline-time.monotonic()))
                        for process in processes]
            if any(status != 0 for status in statuses):
                raise RuntimeError(f"MCCL workers failed: {statuses}; see {output_dir}")
            logger.info("Passed two-device MCCL TP; logs: %s", output_dir)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-path", type=Path, default=REPO_ROOT / "build/mate_sparse_ops")
    parser.add_argument("--bin-dir", type=Path,
                        default=REPO_ROOT / "build/lib.linux-x86_64-cpython-310/xllm")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("sparse_attention_validation"))
    parser.add_argument("--tp-only", action="store_true")
    args = parser.parse_args()
    binary = args.bin_dir.resolve() / "glm5_next_sparse_attention_test"
    if not binary.is_file():
        parser.error(f"build glm5_next_sparse_attention_test first: {binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, FLASHINFER_OPS_PATH=str(args.ops_path.resolve()))
    if not args.tp_only:
        _single(args.bin_dir.resolve() / "sparse_mla_test", [],
                args.output_dir / "xllm_cpp_sparse_mla_test.log", environment)
        _single(binary, ["--gtest_filter=-GlmSparseTest.McclTensorParallelWorker"],
                args.output_dir / "glm_sparse_layer_test.log", environment)
    _tensor_parallel(binary, args.output_dir.resolve(), environment)


if __name__ == "__main__":
    main()
