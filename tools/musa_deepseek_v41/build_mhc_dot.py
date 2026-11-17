# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline-export the pinned SGLang mHC dot160 and fused RMS/Sinkhorn.

Compilation does not establish GPU numerical or Graph correctness. The native
caller supports M=1..8, K=20480, hc=4, RMS eps=1e-20, HC eps=1e-6,
and 20 Sinkhorn iterations, matching the W4AFP8-mt1017 text configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

_SOURCE_SHA = {
    "mhc_dot.py": "732cbbc2aa9bb6429d205d6a30ed54a2ebefaaa180e6b64345c4e2ddd68e2ff7",
    "mhc_stats_projection.py": "8c7eef615fcc301addd81ddd148992498fc0bbc21af879ab7ce5b4f99725258f",
}


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sg-kernel-directory", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_config = json.loads(args.model_config.read_text())
    config = raw_config.get("text_config", raw_config)
    if (config["rms_norm_eps"] != 1e-20 or config["hc_eps"] != 1e-6
            or config["hc_sinkhorn_iters"] != 20 or config["hc_mult"] != 4):
        raise ValueError("Unsupported mHC numerical contract in model configuration")
    for filename, expected in _SOURCE_SHA.items():
        actual = hashlib.sha256((args.sg_kernel_directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unverified SGLang source: {filename} SHA={actual}")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TILELANG_CACHE_DIR"] = str(args.output / "cache")
    os.environ["TRITON_CACHE_DIR"] = str(args.output / "triton_cache")
    dot = _load("xllm_mhc_dot_source", args.sg_kernel_directory / "mhc_dot.py")
    factory = dot._get_kernel(160)
    factory.target = {"kind": "musa", "arch": "mp_31"}
    factory.execution_backend = "tvm_ffi"
    compiled = factory.compile()
    library = Path(compiled._tilelang_cache_path) / "executable.so"
    shutil.copy2(library, args.output / "libsg_mhc_dot160.so")
    (args.output / "device_source.mu").write_text(compiled.get_kernel_source())
    (args.output / "tir.txt").write_text(factory.get_tir().script())

    import triton
    from triton.backends.compiler import AttrsDescriptor, GPUTarget
    from triton.compiler import ASTSource

    stats = _load("xllm_mhc_stats_source",
                  args.sg_kernel_directory / "mhc_stats_projection.py")
    constants = {"NUM_SLICES": 160, "NUM_N_BLOCKS": 1, "SQ_STRIDE": 1,
                 "EPS": config["rms_norm_eps"], "HC_EPS": config["hc_eps"],
                 "SINKHORN_ITERS": config["hc_sinkhorn_iters"],
                 "VECTORIZE_SLICES": True, "CONTIGUOUS_ROW": True}
    signature = {name: "*fp32" for name in
                 ("PART_MIX", "PART_SQ", "HC_SCALE", "HC_BASE", "MIXES", "PRE", "POST", "COMB")}
    signature["M"] = "i32"
    kernel = triton.compile(ASTSource(
        stats._mhc_mix_stats_sinkhorn_reduce_kernel, signature=signature,
        constants=constants, attrs=AttrsDescriptor.from_hints({i: 16 for i in range(8)})),
        target=GPUTarget("musa", 31, 32), options={"num_warps": 1, "num_stages": 1})
    binary = kernel.asm["mubin"]
    if isinstance(binary, tuple):
        binary = Path(binary[1]).read_bytes()
    if isinstance(binary, str):
        binary = Path(binary).read_bytes()
    (args.output / "reduce.mubin").write_bytes(binary)
    metadata = kernel.metadata
    manifest = {
        "status": "CPU_EXPORTED_REQUIRES_GPU_NUMERIC_AND_GRAPH_CHECKS",
        "target": {"kind": "musa", "arch": 31},
        "sg_commit": "7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6",
        "source_sha256": _SOURCE_SHA,
        "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
        "dot_library_sha256": hashlib.sha256(
            (args.output / "libsg_mhc_dot160.so").read_bytes()).hexdigest(),
        "input_contract": "FP32[M,20480] and FP32[24,20480], M1..8; carried_pre is separate",
        "reduce_constants": constants,
        "kernels": {"reduce": {
            "file": "reduce.mubin", "sha256": hashlib.sha256(binary).hexdigest(),
            "bytes": len(binary), "symbol": metadata.name,
            "threads": metadata.num_warps * 32, "shared": metadata.shared,
            "num_ctas": metadata.num_ctas, "cluster_dims": metadata.cluster_dims,
        }},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Exported mHC candidate to %s; GPU validation is still required", args.output)


if __name__ == "__main__":
    main()
