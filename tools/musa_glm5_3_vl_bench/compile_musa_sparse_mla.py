# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Export independent-candidate MP31 BF16 sparse MLA for the C++ runtime."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

if TYPE_CHECKING:
    from tilelang.jit.kernel import JITKernel


def _export(kernel: "JITKernel", output: Path) -> None:
    import tvm_ffi

    temporary = output.with_suffix(f".{os.getpid()}.tmp.so")
    kernel.export_library(str(temporary))
    tvm_ffi.load_module(str(temporary)).get_function("main")
    temporary.replace(output)


def _export_indexer(ops_path: Path) -> None:
    import torch
    import tvm_ffi
    from mate.jit.gemm.deep_gemm.paged_mqa_logits import (
        PagedMqaLogitsJitConfig,
        gen_paged_mqa_logits_metadata_spec,
        gen_paged_mqa_logits_spec,
    )

    config = PagedMqaLogitsJitConfig(1, 32, 128, 64, False, torch.float32)
    for name, function, spec in (
        ("glm_kpool_metadata_v1", "get_paged_mqa_logits_metadata",
         gen_paged_mqa_logits_metadata_spec(128)),
        ("glm_kpool_score_v1", "fp8_paged_mqa_logits",
         gen_paged_mqa_logits_spec(config)),
    ):
        library = spec.build()
        tvm_ffi.load_module(str(library)).get_function(function)
        directory = ops_path / name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.so"
        temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
        shutil.copy2(library, temporary)
        temporary.replace(target)
        target.with_suffix(".json").write_text(json.dumps({
            "abi": 1, "mate": importlib.metadata.version("mate"),
            "spec": spec.name, "function": function,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        logger.info("Exported %s", target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-path", type=Path,
                        default=os.environ.get("FLASHINFER_OPS_PATH"))
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--indexer-only", action="store_true")
    args = parser.parse_args()
    if args.ops_path is None:
        parser.error("pass --ops-path or set FLASHINFER_OPS_PATH")
    if any(h not in (8, 16, 32, 64) for h in args.heads):
        parser.error("MP31 heads must be 8,16,32,64")
    import torch
    import torch_musa  # noqa: F401
    import tilelang
    # Disk-cached JITKernel objects omit the runtime module needed for export.
    tilelang.disable_cache()
    from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_attention_fwd_kernel_model1 as factory,
    )

    if torch.musa.get_device_capability() != (3, 1):
        raise RuntimeError("This export ABI requires MP31")
    if importlib.metadata.version("mate") != "0.2.8.post1":
        raise RuntimeError("Sparse MLA v1 export requires verified MATE 0.2.8.post1")
    _export_indexer(args.ops_path)
    if args.indexer_only:
        return
    for heads in args.heads:
        name = f"sparse_mla_bf16_v1_mp31_h{heads}"
        prim = factory.get_tir(
            heads, 512, has_topk_length=True, has_attn_sink=False,
            is_persistence=False,
        )
        kernel = tilelang.compile(
            prim, target="musa", execution_backend="tvm_ffi",
            pass_configs=factory.pass_configs, compile_flags=factory.compile_flags,
        )
        directory = args.ops_path / name
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{name}.so"
        _export(kernel, output)
        manifest = {
            "abi": 1, "mate": importlib.metadata.version("mate"),
            "heads": heads, "head_dim": 512, "dtype": "bfloat16",
            "factory_sha256": hashlib.sha256(factory.func_source.encode()).hexdigest(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info("Exported %s", output)

    from mate.jit.mla_ops import gen_mla_ops_spec
    from mate.sparse_mla.tilelang.sparse_mla_v32_decode_fwd_scheduled import (
        sparse_attention_fwd_kernel as decode_factory,
    )
    spec = gen_mla_ops_spec()
    library = spec.build()
    directory = args.ops_path / "mla_ops"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"mla_ops.{os.getpid()}.tmp.so"
    shutil.copy2(library, temporary)
    temporary.replace(directory / "mla_ops.so")
    parts = torch.musa.get_device_properties(0).multi_processor_count
    if not 1 < parts <= 64:
        raise RuntimeError("Sparse MLA v1 decode requires 2..64 MP parts")
    for heads in args.heads:
        max_splits = 32 if parts <= 32 else 64
        name = f"sparse_mla_nope_fp8_v1_mp31_h{heads}_splits{max_splits}"
        prim = decode_factory.get_tir(
            heads, 512, 0, has_topk_length=True, has_attn_sink=False,
            support_split=True, max_nums_splits=max_splits,
        )
        kernel = tilelang.compile(
            prim, target="musa", execution_backend="tvm_ffi",
            pass_configs=decode_factory.pass_configs,
            compile_flags=decode_factory.compile_flags,
        )
        directory = args.ops_path / name
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{name}.so"
        _export(kernel, output)
        output.with_suffix(".json").write_text(json.dumps({
            "abi": 1, "mate": importlib.metadata.version("mate"),
            "heads": heads, "max_splits": max_splits,
            "cache_record_bytes": 528, "dtype": "bfloat16",
            "factory_sha256": hashlib.sha256(decode_factory.func_source.encode()).hexdigest(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        logger.info("Exported %s", name)


if __name__ == "__main__":
    main()
