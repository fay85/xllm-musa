# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze SGLang's corrected MATE S4/FP8 GEMM for native C++ callers.

Run with the pinned delivery MATE source on PYTHONPATH, an explicit
MATE_MUSA_ARCH_LIST, and a task-owned MATE cache. This is CPU compilation:
it does not initialize a GPU or load a model. The generated library uses
the same typed TVM-FFI export as the delivery implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patched-include", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MATE_MUSA_ARCH_LIST") != "3.1":
        raise ValueError("Explicit MATE_MUSA_ARCH_LIST=3.1 is required")
    if not os.environ.get("MATE_WORKSPACE_BASE"):
        raise ValueError("Use a task-owned MATE_WORKSPACE_BASE")

    import torch
    import mate
    from mate.jit.gemm.masked_moe_gemm_mixed_dtype import (
        gen_masked_moe_gemm_mixed_dtype_spec,
        masked_moe_gemm_mixed_dtype_dispatch_name,
    )

    include = args.patched_include.resolve()
    header = include / "mate/gemm/deep_gemm/mp31_masked_moe_gemm_mixed_dtype.hpp"
    digest = _sha256(header)
    # Match SGLang's _mixed_dtype_module, not MATE's generic -O3 recipe.
    # The kernel relies on shared-memory/vector aliasing and explicit pipeline
    # barriers; preserve the delivery compiler conditions during validation.
    delivery_flags = [
        "-Od3",
        "-O2",
        "-DNDEBUG",
        "-fno-strict-aliasing",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-load-cluster-mutation=1",
        "-mllvm",
        "--num-dwords-of-load-in-mutation=64",
    ]
    flags_digest = hashlib.sha256(json.dumps(delivery_flags).encode()).hexdigest()[:12]
    recipe = (1, 128)
    spec = gen_masked_moe_gemm_mixed_dtype_spec(torch.bfloat16, recipe, "K")
    spec.name += "_sglang_" + digest[:16] + "_flags_" + flags_digest
    spec.extra_cuda_cflags = delivery_flags
    spec.extra_cflags = ["-O2", "-fno-strict-aliasing"]
    spec.extra_include_dirs = [include, *(spec.extra_include_dirs or [])]
    # Do not reuse an installed AOT binary predating the pipeline correction.
    spec.aot_path = None
    library = spec.build(verbose=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "libdeepseek_v41_w4a8_group128.so"
    shutil.copy2(library, target)
    source_root = Path(mate.__file__).resolve().parent
    source_files = [source_root / "jit/gemm/masked_moe_gemm_mixed_dtype.py", header]
    source_files.extend(sorted(include.rglob("*.hpp")))
    record = {
        "status": "COMPILED_NOT_GPU_VALIDATED",
        "library": str(target),
        "library_sha256": _sha256(target),
        "symbol": masked_moe_gemm_mixed_dtype_dispatch_name(
            torch.bfloat16, recipe, "K"
        ),
        "activation_recipe": list(recipe),
        "weight_recipe": [1, 128],
        "activation_dtype": "E4M3",
        "weight_dtype": "packed signed INT4",
        "activation_scale_dtype": "FP32",
        "weight_scale_dtype": "BF16",
        "output_dtype": "BF16",
        "mate_source": str(source_root),
        "delivery_compiler_flags": delivery_flags,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch_musa", "mate", "apache-tvm-ffi")
        },
        "sources": {str(path): _sha256(path) for path in source_files},
        "compile_commands": spec.get_compile_commands(),
        "scope": "Offline build only; numerical and runtime tests remain required",
    }
    (output / "w4a8_mate_build.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    logger.info("Compiled corrected S4FP8 group128 GEMM: %s", target)


if __name__ == "__main__":
    main()
