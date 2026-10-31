# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Check image package versions, artifact integrity and paired shared-library loading."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tools.musa_unified.artifacts import check_packages, digest
from tools.musa_unified.deepseek.run import runtime_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    pack = args.pack.resolve(strict=True)
    manifest = json.loads((pack / "manifest.json").read_text())
    packages = check_packages()
    if args.verify_hashes:
        for name, expected in manifest["artifacts"].items():
            path = pack / name
            if not path.is_file() or path.is_symlink() or digest(path) != expected["sha256"]:
                raise ValueError(f"Missing, linked or changed pack artifact: {name}")
    checks = []
    for role, binary, library in [
        ("prefill", pack / "prefill/xllm", pack / "prefill"),
        ("decode", pack / "decode/xllm", pack / "decode"),
        (
            "deepseek",
            pack / "service/deepseek/xllm_master_serving",
            pack / "service/deepseek",
        ),
        ("qwen", pack / "service/qwen/xllm_master_serving", pack / "service/qwen"),
    ]:
        env = runtime_environment(pack, role if role in {"prefill", "decode", "deepseek"} else "deepseek")
        env["LD_LIBRARY_PATH"] = str(library) + ":" + env["LD_LIBRARY_PATH"]
        env.pop("LD_PRELOAD", None)
        env.update(
            MUSA_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="",
            MTHREADS_VISIBLE_DEVICES="",
        )
        result = subprocess.run(["ldd", "-r", str(binary)], env=env, capture_output=True, text=True)
        text = result.stdout + result.stderr
        if result.returncode or "not found" in text or "undefined symbol" in text:
            raise RuntimeError(f"{role} loader mismatch:\n{text}")
        help_result = subprocess.run([str(binary), "--helpxml"], env=env, capture_output=True, text=True)
        if help_result.returncode not in (0, 1) or "</AllFlags>" not in help_result.stdout:
            raise RuntimeError(f"{role} help/ABI check failed: {help_result.stderr}")
        checks.append(
            dict(
                role=role,
                executable=str(binary),
                sha256=digest(binary),
                ldd=text,
                help_xml_complete=True,
            )
        )
    report = dict(
        passed=True,
        packages=packages,
        loader_checks=checks,
        gpu_inference_executed=False,
        image_contract=manifest["base_image"],
    )
    if args.report:
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
    logger.info("Pack integrity and four role/service loader checks passed; no GPU inference executed")


if __name__ == "__main__":
    main()
