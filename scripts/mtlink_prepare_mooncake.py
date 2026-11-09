#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply or verify the exact Mooncake overlay for this MTLink branch."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.logger import logger

PACKAGE_ROOT = REPO_ROOT / "third_party" / "patches" / "mooncake_musa_ipc"


def _git(root: Path, arguments: List[str]) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular source file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid manifest path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Manifest path leaves Mooncake: {name}")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Source path contains a symbolic link: {name}")
    return path


def _load_manifest() -> Dict[str, Any]:
    manifest = json.loads((PACKAGE_ROOT / "manifest.json").read_text("utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported Mooncake manifest schema")
    patch = PACKAGE_ROOT / manifest["patch_file"]
    if patch.parent != PACKAGE_ROOT or _sha256(patch) != manifest["patch_sha256"]:
        raise ValueError("Mooncake patch SHA256 does not match the manifest")
    return manifest


def _prepare(root: Path, apply: bool) -> None:
    manifest = _load_manifest()
    actual_root = Path(_git(root, ["rev-parse", "--show-toplevel"])).resolve()
    if actual_root != root:
        raise ValueError("Mooncake is not initialized as a Git working tree")
    head = _git(root, ["rev-parse", "HEAD"])
    if head != manifest["base_commit"]:
        raise ValueError(
            f"Mooncake HEAD must be {manifest['base_commit']}; found {head}"
        )

    files = manifest["files"]
    hashes = {
        entry["path"]: _sha256(_source_path(root, entry["path"]))
        for entry in files
    }
    final_matches = all(
        hashes[entry["path"]] == entry["after_sha256"] for entry in files
    )
    if final_matches:
        logger.info("Mooncake MTLink overlay verified; no changes needed")
        return

    mismatches = [
        entry["path"]
        for entry in files
        if hashes[entry["path"]] != entry["before_sha256"]
    ]
    if mismatches:
        raise ValueError(
            "Affected source files differ from the pinned base or complete "
            "overlay; preserve local changes before retrying: "
            + ", ".join(mismatches)
        )
    if not apply:
        raise ValueError("Mooncake overlay is not applied; rerun with --apply")

    patch = str(PACKAGE_ROOT / manifest["patch_file"])
    _git(root, ["apply", "--check", "--whitespace=nowarn", patch])
    _git(root, ["apply", "--whitespace=nowarn", patch])
    for entry in files:
        if _sha256(_source_path(root, entry["path"])) != entry["after_sha256"]:
            raise ValueError(f"Post-apply verification failed: {entry['path']}")
    logger.info("Applied and verified the complete pinned Mooncake MTLink overlay")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true", help="apply, or verify if applied")
    action.add_argument("--check", action="store_true", help="verify without changing files")
    parser.add_argument(
        "--mooncake-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "Mooncake",
        help="initialized Mooncake checkout (defaults to this repository's submodule)",
    )
    args = parser.parse_args()
    try:
        _prepare(args.mooncake_root.resolve(), args.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
