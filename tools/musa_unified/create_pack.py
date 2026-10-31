# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Create a frozen engine/service/AOT pack on top of the fixed d1029-opt3 image."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.logger import logger
from tools.musa_unified.artifacts import (
    BASE_IMAGE,
    BASE_IMAGE_ID,
    check_packages,
    digest,
    inventory,
)


def components(args: argparse.Namespace) -> list[dict]:
    base = args.bundle.resolve(strict=True)
    entries = [
        ("prefill", base / "prefill", False),
        ("decode", base / "decode", False),
        ("cached_ops", base / "cached_ops", False),
        ("qwen_cached_ops", base / "qwen_cached_ops", True),
        ("mate_mubin", base / "mate_mubin", True),
        ("service/deepseek/xllm_master_serving", args.deepseek_service, False),
        ("service/deepseek/librust_tokenizers.so", args.deepseek_tokenizers, False),
        ("service/qwen/xllm_master_serving", args.qwen_service, True),
        ("service/qwen/librust_tokenizers.so", args.qwen_tokenizers, True),
        ("bootstrap/libmusa_backend_init.so", args.bootstrap, True),
        ("etcd/etcd", args.etcd, True),
        ("tools/musa_qwen3_8_vl_bench", REPO / "tools/musa_qwen3_8_vl_bench", False),
        ("tools/musa_unified", REPO / "tools/musa_unified", False),
        ("scripts/logger.py", REPO / "scripts/logger.py", False),
    ]
    rows = []
    for destination, source, provided in entries:
        source = source.resolve(strict=True)
        files = (
            inventory(source)
            if source.is_dir()
            else {source.name: {"size": source.stat().st_size, "sha256": digest(source)}}
        )
        excluded_dirs = (
            {"__pycache__", "results", "runs", "build"} if source.is_relative_to(REPO / "tools") else {"__pycache__"}
        )
        files = {
            name: info
            for name, info in files.items()
            if not set(Path(name).parts).intersection(excluded_dirs) and not name.endswith(".pyc")
        }
        rows.append(
            dict(
                destination=destination,
                source=str(source),
                files=files,
                bytes=sum(row["size"] for row in files.values()),
                image_provided=provided,
            )
        )
    return rows


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        excluded = ["__pycache__", "*.pyc"]
        if source.is_relative_to(REPO / "tools"):
            excluded.extend(["results", "runs", "build"])
        shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns(*excluded))
    else:
        shutil.copy2(source, target, follow_symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engine-manifest", type=Path)
    parser.add_argument("--deepseek-service", type=Path, required=True)
    parser.add_argument("--deepseek-tokenizers", type=Path, required=True)
    parser.add_argument(
        "--qwen-service",
        type=Path,
        default=Path("/opt/xllm_service/bin/xllm_master_serving.rr_probe_fix_20261024"),
    )
    parser.add_argument(
        "--qwen-tokenizers",
        type=Path,
        default=Path("/opt/xllm_service/lib/librust_tokenizers.so"),
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path("/workspace/artifacts/libmusa_backend_init.so"),
    )
    parser.add_argument("--etcd", type=Path, default=Path("/opt/xllm_service/bin/etcd"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; never overwrite an installed pack",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tar", action="store_true", help="Also create sibling .tar.gz")
    args = parser.parse_args()
    packages = check_packages()
    manifest_path = args.engine_manifest or args.bundle / "engine_manifest.json"
    if not manifest_path.exists() and args.engine_manifest is None:
        manifest_path = args.bundle.parent / "bundle_manifest.json"
    engine = json.loads(manifest_path.read_text())
    for role in ["prefill", "decode"]:
        for name in ["xllm", "libasio.so"]:
            key = role + "/" + name
            if digest(args.bundle / key) != engine["artifacts"][key]["sha256"]:
                raise ValueError(f"Engine/Asio pair differs from manifest: {key}")
    rows = components(args)
    report = dict(
        base_image=BASE_IMAGE,
        base_image_id=BASE_IMAGE_ID,
        components=rows,
        frozen_bytes=sum(row["bytes"] for row in rows),
        non_image_component_bytes=sum(row["bytes"] for row in rows if not row["image_provided"]),
        packages=packages,
        excluded=[
            "models",
            "build cache",
            "vcpkg",
            "Rust/Cargo",
            "Torch/MUSA/TVM-FFI wheels",
            "driver",
        ],
    )
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return
    output = args.output.resolve()
    if output.exists() or (args.tar and Path(str(output) + ".tar.gz").exists()):
        raise FileExistsError("Choose a fresh pack directory/archive; existing packages are preserved")
    output.mkdir(parents=True)
    for row in rows:
        _copy(Path(row["source"]), output / row["destination"])
    shutil.copy2(manifest_path, output / "engine_manifest.json")
    shutil.copy2(REPO / "tools/musa_unified/README.md", output / "README.md")
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    report.update(
        schema=1,
        source_commit=git.stdout.strip() if git.returncode == 0 else None,
        artifacts=inventory(output),
        acceptance="Packaging/link verification is not model performance or accuracy acceptance",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(output / "tools/musa_unified/check_pack.py"),
            "--pack",
            str(output),
            "--verify-hashes",
        ],
        check=True,
    )
    if args.tar:
        archive = Path(str(output) + ".tar.gz")
        with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
            tar.add(output, arcname="xllm_pack")
        (Path(str(archive) + ".sha256")).write_text(digest(archive) + "  " + archive.name + "\n")
        logger.info("Created %s (%d bytes)", archive, archive.stat().st_size)
    logger.info("Frozen pack ready for relocation checks: %s", output)


if __name__ == "__main__":
    main()
