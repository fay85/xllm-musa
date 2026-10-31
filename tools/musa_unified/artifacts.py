# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Inspect and assemble role-paired engine/AOT artifacts for d1029-opt3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.logger import logger

BASE_IMAGE = "registry.mthreads.com/presale/devtech/xllm:d1029-opt3"
BASE_IMAGE_ID = "sha256:f19c55daf710fc2899b057a3faf62c82c1a72cf4f0c67faf420520efbd3a8407"
PACKAGES = {
    "torch": "2.9.1",
    "torch-musa": "2.9.1",
    "mate": "0.2.7",
    "mate-mubin": "0.2.7",
    "apache-tvm-ffi": "0.1.11.post1+musa.1",
    "tilelang-musa": "0.1.12+musa.2",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inventory(root: Path, exclude: frozenset[str] = frozenset()) -> dict[str, dict]:
    result = {}
    for directory, _, files in os.walk(root, followlinks=True):
        if len(Path(directory).relative_to(root).parts) > 32:
            raise ValueError(f"Unexpected directory depth: {directory}")
        for name in sorted(files):
            path = Path(directory) / name
            if path.relative_to(root).as_posix() in exclude:
                continue
            if not path.is_file():
                raise FileNotFoundError(path)
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
    return result


def require_idle(bundle: Path) -> None:
    """Refuse an in-place rebuild while this bundle's binaries/libraries are mapped."""
    bundle = bundle.resolve()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            executable = Path(os.readlink(proc / "exe").removesuffix(" (deleted)")).resolve()
            maps = (proc / "maps").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if executable.is_relative_to(bundle) or str(bundle) + "/" in maps:
            raise RuntimeError(f"Bundle still in use by PID {proc.name}: {executable}")


def check_packages() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    for name, expected in PACKAGES.items():
        # Torch may carry a wheel-local build suffix; the ABI baseline is 2.9.1.
        observed = actual[name].split("+", 1)[0] if name in {"torch", "torch-musa"} else actual[name]
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual[name]}")
    return actual


def _copy_profile(source: Path | None, destination: Path) -> None:
    if source is not None:
        source = source.resolve(strict=True)
        if source == destination.resolve():
            return
        if not destination.exists():
            shutil.copytree(source, destination, symlinks=False)
        elif inventory(source) != inventory(destination):
            raise RuntimeError(f"Existing profile differs; choose a fresh bundle: {destination}")
    if not destination.is_dir():
        raise FileNotFoundError(f"Missing operator profile: {destination}")
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise RuntimeError(f"Operator profile must be materialized: {destination}")


def assemble(args: argparse.Namespace) -> None:
    require_idle(args.bundle)
    packages = check_packages()
    profiles = {
        "cached_ops": Path(args.deepseek_ops) if args.deepseek_ops else None,
        "qwen_cached_ops": args.qwen_ops,
        "mate_mubin": args.mubin,
    }
    for name, source in profiles.items():
        _copy_profile(source, args.bundle / name)
    for name in [
        "gemm_ops",
        "mla_ops",
        "dsa_model1_mate027_h8_topk512_extra128_bs128",
        "dsa_model1_mate027_prefill_h64",
        "mhc_pre_big_fuse_ops",
    ]:
        if not (args.bundle / "cached_ops" / name).is_dir():
            raise FileNotFoundError(f"DeepSeek AOT family missing: {name}")
    for pattern in ["mate_gdn_*", "fmha_fwd_*"]:
        if not any((args.bundle / "qwen_cached_ops").glob(pattern)):
            raise FileNotFoundError(f"Qwen AOT family missing: {pattern}")
    for role in ["prefill", "decode"]:
        for name in ["xllm", "libasio.so"]:
            if not (args.bundle / role / name).is_file():
                raise FileNotFoundError(args.bundle / role / name)
    source = args.source.resolve()
    git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    files = inventory(args.bundle, frozenset({"engine_manifest.json"}))
    source_hashes = {
        path.relative_to(source).as_posix(): digest(path)
        for folder in ["xllm", "tools/pd_guard_20261006", "cmake"]
        for path in sorted((source / folder).rglob("*"))
        if path.is_file() and path.suffix in {".h", ".cpp", ".cu", ".cmake", ".txt"}
    }
    manifest = dict(
        schema=1,
        base_image=BASE_IMAGE,
        base_image_id=BASE_IMAGE_ID,
        source_commit=git.stdout.strip() if git.returncode == 0 else None,
        source_path=str(source),
        source_sha256=source_hashes,
        packages=packages,
        artifacts=files,
        profiles={"deepseek": "cached_ops", "qwen": "qwen_cached_ops"},
        acceptance="Build verification only; model acceptance is separate",
    )
    (args.manifest or args.bundle / "engine_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Built role-paired bundle: %s", args.bundle)


def help_xml(binary: Path, log: Path) -> None:
    result = subprocess.run([str(binary), "--helpxml"], capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Help/loader check failed: {log}")
    text = result.stdout
    root = ET.fromstring(text[text.index("<?xml") : text.index("</AllFlags>") + len("</AllFlags>")])
    flags = {node.findtext("name") for node in root.findall("flag")}
    required = {
        "model",
        "enable_graph",
        "enable_native_service_adapter",
        "num_speculative_tokens",
    }
    if not required <= flags:
        raise ValueError(f"Engine flags missing: {required - flags}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    idle = subs.add_parser("idle")
    idle.add_argument("--bundle", type=Path, required=True)
    help_parser = subs.add_parser("help-xml")
    help_parser.add_argument("--binary", type=Path, required=True)
    help_parser.add_argument("--log", type=Path, required=True)
    assemble_parser = subs.add_parser("assemble")
    assemble_parser.add_argument("--bundle", type=Path, required=True)
    assemble_parser.add_argument("--source", type=Path, required=True)
    assemble_parser.add_argument("--deepseek-ops", default="")
    assemble_parser.add_argument("--qwen-ops", type=Path, required=True)
    assemble_parser.add_argument("--mubin", type=Path, required=True)
    assemble_parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "idle":
        require_idle(args.bundle)
    elif args.command == "help-xml":
        help_xml(args.binary, args.log)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
