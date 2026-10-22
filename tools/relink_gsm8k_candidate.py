#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Relink unchanged model objects with a private kernel archive; never rebuild donors."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--kernel-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, kernel, output = args.base.resolve(), args.kernel_archive.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "build") or output.exists():
        parser.error("output must be a new directory inside this checkout's build")
    manifest = json.loads((base / "manifest.json").read_text())
    base_sha = _sha(base / "xllm")
    if base_sha != manifest["binary_sha256"] or not kernel.is_file():
        parser.error("base binary changed or replacement kernel missing")
    if not manifest.get("required_dspark_sources"):
        parser.error("base must be a source-complete DSpark model candidate")
    for record in manifest["sources"] + manifest.get("local_policy_headers", []):
        if _sha(ROOT / record["path"]) != record["sha256"]:
            parser.error(f"model source/header changed; recompile it: {record['path']}")
    link = (base / "link.sh").read_text()
    old_kernel = manifest["kernel_archive"]
    if old_kernel not in link or str(base / "xllm") not in link:
        parser.error("base link script does not explicitly identify kernel/output")
    link = link.replace(old_kernel, str(kernel)).replace(str(base / "xllm"), str(output / "xllm"))
    for relative, record in manifest["archives"].items():
        archive_text = next((token for token in link.split() if token.endswith(Path(relative).name)), "")
        archive = Path(archive_text.strip("'\"")) if archive_text else Path(record["base_path"])
        if not archive.is_file() or _sha(archive) != record["candidate_sha256"]:
            parser.error(f"base archive changed or unavailable: {archive}")
    output.mkdir(parents=True)
    (output / "tmp").mkdir()
    (output / "link.sh").write_text(link)
    for record in manifest["sources"]:
        shutil.copyfile(ROOT / record["path"], output / Path(record["path"]).name)
    (output / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    with (output / "link.log").open("w") as log:
        subprocess.run(["bash", str(output / "link.sh")], cwd=ROOT / "build/dspark-musa-local",
                       env=dict(os.environ, TMPDIR=str(output / "tmp")),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest.update(relinked_from=str(base), relinked_from_binary_sha256=base_sha,
                    kernel_archive=str(kernel), kernel_archive_sha256=_sha(kernel),
                    binary_sha256=_sha(output / "xllm"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Relinked candidate %s sha256=%s", output, manifest["binary_sha256"])


if __name__ == "__main__":
    main()
