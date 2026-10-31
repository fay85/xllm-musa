# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Explicitly seed empty submodule directories from the verified image worktrees."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-source", type=Path, default=Path("/workspace/xllm-mt-main"))
    parser.add_argument("--source", type=Path, default=REPO)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    image = args.image_source.resolve(strict=True)
    if source == image or args.report.exists():
        raise ValueError("Use an independent checkout and a new report path")
    config = configparser.ConfigParser()
    config.read(source / ".gitmodules")
    pairs = []
    for section in config.sections():
        relative = Path(config[section]["path"])
        origin, destination = image / relative, source / relative
        if not destination.resolve().is_relative_to(source) or not origin.resolve().is_relative_to(image):
            raise ValueError("Submodule path escapes source root")
        if not origin.is_dir() or not any(origin.iterdir()):
            raise FileNotFoundError(origin)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to replace existing dependency: {destination}")
        pairs.append((relative, origin, destination))
    records = []
    for relative, origin, destination in pairs:
        revision = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(origin), "diff", "--binary", "HEAD"])
        shutil.copytree(
            origin,
            destination,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__"),
        )
        records.append(
            dict(
                path=str(relative),
                image_revision=revision,
                working_diff_sha256=hashlib.sha256(diff).hexdigest(),
                includes_working_changes=bool(diff),
                copied_from=str(origin),
            )
        )
        logger.info("Seeded image dependency: %s", relative)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(
            dict(
                policy="Image dependency worktrees, not necessarily remote gitlink revisions",
                source=str(source),
                image_source=str(image),
                dependencies=records,
            ),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
