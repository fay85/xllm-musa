# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage verified source snapshots and rebuild with the original PyTorch toolchain."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = json.loads((HERE / "build_manifest.json").read_text())


def _check_hash(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _stage(name: str, output: Path) -> None:
    entry = MANIFEST["variants"][name]
    patch = HERE / (name + ".patch")
    worker = HERE / "wire_build_worker.py"
    helper = HERE / "pytorch_musa_paths.py"
    _check_hash(patch, entry["patch_sha256"])
    _check_hash(worker, entry["worker_sha256"])
    _check_hash(helper, entry["pytorch_helper_sha256"])
    # Populate a disposable Git index from the measured base, regardless of the
    # caller's branch or worktree. Nothing is applied to the caller's source.
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git = ["git", "-C", str(REPO)]
        subprocess.run(git + ["read-tree", MANIFEST["base_commit"]], env=env, check=True)
        subprocess.run(git + ["apply", "--cached", "--unidiff-zero", str(patch)],
                       env=env, check=True)
        for name, expected in entry["spec"]["sources_sha256"].items():
            data = subprocess.check_output(git + ["show", ":" + name], env=env)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Reconstructed source SHA-256 mismatch: {name}")
            target = output / "src" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    shutil.copyfile(worker, output / "worker.py")
    shutil.copyfile(helper, output / helper.name)
    (output / "spec.json").write_text(json.dumps(entry["spec"], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=(*MANIFEST["variants"], "G32RAGLIB1"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path, help="Host artifact root to mount read-only at /expt")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.variant == "G32RAGLIB1" and not args.assets:
        parser.error("G32RAGLIB1 requires --assets")
    assets = args.assets.resolve(strict=True) if args.assets else None
    output.mkdir()
    if args.variant == "G32RAGLIB1":
        shutil.copyfile(HERE / "ragged_launcher_build.py", output / "worker.py")
        entry = MANIFEST["variants"]["G32PM2561"]
    else:
        _stage(args.variant, output)
        entry = MANIFEST["variants"][args.variant]
    if assets is None:
        return
    image = entry["builder_image"]
    subprocess.run(["docker", "image", "inspect", image],
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run([
        "docker", "run", "--rm", "--init", "--runtime=runc", "--network=none",
        "--mount", f"type=bind,source={output},target=/guide",
        "--mount", f"type=bind,source={assets},target=/expt,readonly",
        "--entrypoint", "/opt/venv/bin/python", image, "-u", "/guide/worker.py",
    ], check=True)
    if args.variant != "G32RAGLIB1":
        dry = json.loads((output / "dry-run.json").read_text())
        if dry["graph"] != entry["ninja_sha256"]:
            raise ValueError("Build graph differs from the recorded Ninja graph")
    print((output / "artifact.json").read_text(), end="")


if __name__ == "__main__":
    main()
