# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned delivery's H8/D512 MATE sparse kernel without GPUs.

Serve through native C++ TVM-FFI; Python is strictly an offline build tool.
The input is already the V4.1 QDQ BF16 cache carrier, without another cast to
V4's packed FP8 KV format. The current wrapper intentionally uses explicit
sparse indices for both single-token decode and prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.donor_root))
    os.environ["TILELANG_CACHE_DIR"] = str(args.output / "cache")
    # The delivery kernel remaps invalid indices to row zero. Masking its
    # probability cannot prevent 0*NaN in PV when that row is unused. Preserve
    # the original source and use the existing bounded robust load's zero-fill
    # for invalid addresses instead; this avoids copying the whole KV cache.
    donor_source = (
        args.donor_root / "mate/sparse_mla/tilelang/sparse_mla_model1_fwd_pipelined.py"
    )
    patched_source, count = re.subn(
        r"kperm_mask_local\[r\], kperm_indices_local\[r\], 0",
        "kperm_mask_local[r], kperm_indices_local[r], seq_len_kv",
        donor_source.read_text(),
    )
    if count != 2:
        raise ValueError(f"Pinned robust-load patch expected two sites, found {count}")
    patched_path = args.output / "sparse_mla_model1_safe.py"
    patched_path.write_text(patched_source)
    module_name = "mate.sparse_mla.tilelang.xllm_v41_safe_sparse"
    spec = importlib.util.spec_from_file_location(module_name, patched_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the isolated sparse kernel source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    factory = module.sparse_attention_fwd_kernel_model1

    factory.target = {"kind": "musa", "arch": "mp_31"}
    factory.execution_backend = "tvm_ffi"
    parameters = {
        "num_heads": 8,
        "dim": 512,
        "has_attn_sink": True,
        "has_topk_length": False,
        "is_persistence": False,
    }
    compiled = factory(**parameters)
    # This pinned TileLang build publishes the standalone executable to its
    # on-disk cache; use those bytes, not a JIT Python closure, for serving.
    source = Path(compiled._tilelang_cache_path) / "executable.so"
    if not source.is_file():
        raise FileNotFoundError(f"No standalone native executable at {source}")
    library = args.output / "libdeepseek_v41_sparse_h8_bf16.so"
    shutil.copyfile(source, library)
    source_files = [
        "mate/sparse_mla/tilelang/sparse_mla_model1_fwd_pipelined.py",
        "mate/sparse_mla/tilelang/sparse_mla_prefill_common.py",
        "tilelang/jit/adapter/tvm_ffi.py",
    ]
    record = {
        "library": str(library),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "bytes": library.stat().st_size,
        "target": factory.target,
        "parameters": parameters,
        "source_reference": "JD delivery SGLang7c97bb0/MATE source overlay",
        "invalid_index_fix": "bounded robust load zero-fills seq_len_kv instead of reading row0",
        "patched_source_sha256": hashlib.sha256(patched_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((args.donor_root / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "compiler_flags": [str(value) for value in factory.compile_flags],
        "status": "BUILT_NOT_YET_NUMERICALLY_VALIDATED",
        "entry": "main(q,kv,indices,out,max_logits,lse,sink,sm_scale_log2,persistent_blocks)",
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2))
    (args.output / "device_source.mu").write_text(compiled.get_kernel_source())
    logger.info("Exported native MATE sparse attention: %s", library)


if __name__ == "__main__":
    main()
