#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Link a CPU prompt audit against the exact existing server dependencies."""

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
    env = dict(os.environ, TMPDIR=str(dest / "tmp"))
    source = ROOT / "tools/dump_dsv4_prompt_tokens.cpp"
    command = shlex.split((candidate / "sampling_params.cpp.command.txt").read_text())
    obj = dest / "prompt_probe.o"
    command[command.index("-o") + 1] = str(obj)
    command[command.index("-c") + 1] = str(source)
    link = (candidate / "link.sh").read_text()
    link = link.replace("xllm/CMakeFiles/xllm.dir/xllm.cpp.o", str(obj))
    link = link.replace(str(candidate / "xllm"), str(dest / "prompt_probe"))
    (dest / "link.sh").write_text(link)
    (dest / "compile.sh").write_text(shlex.join(command) + "\n")
    for name, cmd in (("compile", command), ("link", ["bash", str(dest / "link.sh")])):
        logger.info("Prompt audit %s", name)
        with (dest / f"{name}.log").open("w") as output:
            subprocess.run(cmd, cwd=ROOT / "build/dspark-musa-local", env=env,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
    if not (dest / "prompt_probe").is_file():
        raise RuntimeError("Link wrapper failed; inspect link.log")
    (dest / "manifest.json").write_text(json.dumps({
        "candidate": str(candidate), "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256((dest / "prompt_probe").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
