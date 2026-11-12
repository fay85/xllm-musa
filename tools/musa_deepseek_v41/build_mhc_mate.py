# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile delivery MATE mHC prenorm GEMM/square-sum without GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("MATE_MUSA_ARCH_LIST") != "3.1":
        raise ValueError("Set MATE_MUSA_ARCH_LIST=3.1 for the verified MP31 target")
    if not os.environ.get("MATE_WORKSPACE_BASE"):
        raise ValueError("Use a task-owned MATE_WORKSPACE_BASE")
    import mate
    from mate.jit.gemm.deep_gemm.hyperconnection import (
        HcPrenormGemmJitConfig,
        gen_hyperconnection_spec,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    libraries = {}
    for tile_m in (32, 64):
        config = HcPrenormGemmJitConfig(tile_m, 32, 32, 3, 2)
        spec = gen_hyperconnection_spec(config)
        spec.aot_path = None
        path = spec.build(verbose=True)
        target = args.output / f"libdeepseek_v41_mhc_m{tile_m}.so"
        shutil.copy2(path, target)
        libraries[str(tile_m)] = {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "config": config.__dict__,
            "symbol": "tf32_hc_prenorm_gemm",
            "compile_commands": spec.get_compile_commands(),
        }
        logger.info("Compiled MATE mHC prenorm tile_m=%d: %s", tile_m, target)
    source = Path(mate.__file__).resolve().parent
    record = {
        "status": "COMPILED_NOT_GPU_VALIDATED",
        "mate_source": str(source),
        "implementation_sha256": hashlib.sha256(
            (source / "jit/gemm/deep_gemm/hyperconnection.py").read_bytes()
        ).hexdigest(),
        "libraries": libraries,
        "scope": "Offline build; V4.1 shifted-pre semantics and numerical/performance validation remain required",
    }
    (args.output / "mhc_mate_build.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
