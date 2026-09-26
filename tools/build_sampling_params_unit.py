#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Build actual SamplingParameters regression tests with read-only dependencies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    candidate = args.candidate.resolve()
    if not dest.is_relative_to(ROOT / "build") or dest.exists():
        parser.error("output must be a new checkout build directory")
    dest.mkdir(parents=True)
    (dest / "tmp").mkdir()
    source = ROOT / "tests/core/framework/sampling/sampling_params_test.cpp"
    template = candidate / "sampling_params.cpp.command.txt"
    command = shlex.split(template.read_text())
    command[command.index("-o") + 1] = str(dest / "test.o")
    command[command.index("-c") + 1] = str(source)
    command.insert(1, "-I" + str(ROOT / "xllm/core/framework/sampling"))
    base = ROOT / "build/gsm8k_candidates/c128_shared_address_final_unit_20260925"
    link = shlex.split((base / "link.sh").read_text())
    link[1] = str(dest / "test.o")
    link[link.index("-o") + 1] = str(dest / "sampling_params_test")
    old_kernel = next(token for token in link if token.endswith("/libcuda_kernels.a"))
    link[link.index(old_kernel)] = str(candidate / "libsampler.a")
    env = dict(os.environ, TMPDIR=str(dest / "tmp"))
    for name, cmd in [("compile", command), ("link", link)]:
        logger.info("sampling unit %s", name)
        (dest / f"{name}.sh").write_text(shlex.join(cmd) + "\n")
        with (dest / f"{name}.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (dest / "manifest.json").write_text(json.dumps({
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "archive": str(candidate / "libsampler.a"),
        "archive_sha256": hashlib.sha256((candidate / "libsampler.a").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
