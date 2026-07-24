#!/usr/bin/env python3
# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate MP31 AOT artifacts from SGLang's Apache-licensed MoE kernel.

The runtime consumes a deliberately frozen 22-argument ABI. This tool imports
the upstream SGLang Triton source without importing the full SGLang package,
compiles TorchAda's Qwen3.5 decode buckets, and rejects source revisions that
change the optimized kernel ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl

from scripts.logger import logger


_BATCH_SIZES = tuple(range(1, 9))
_STAGES = ("gate", "down")


def _module_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load_sglang_kernel(source: Path) -> types.ModuleType:
    """Load only the Triton definitions needed for fused_moe_kernel."""
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
    ):
        _module_stub(name)

    batch_ops = _module_stub("sglang.srt.batch_invariant_ops")
    batch_ops.is_batch_invariant_mode_enabled = lambda: False  # type: ignore[attr-defined]

    fp8 = _module_stub("sglang.srt.layers.quantization.fp8_kernel")
    fp8.per_token_group_quant_fp8 = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fp8.scaled_fp8_quant = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fp8.sglang_per_token_group_quant_fp8 = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    int8 = _module_stub("sglang.srt.layers.quantization.int8_kernel")
    int8.per_token_group_quant_int8 = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    int8.per_token_quant_int8 = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    int8.sglang_per_token_group_quant_int8 = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    utils = _module_stub("sglang.srt.utils")
    utils.cpu_has_amx_support = lambda: False  # type: ignore[attr-defined]
    utils.get_bool_env_var = lambda name: False  # type: ignore[attr-defined]
    utils.is_cpu = lambda: False  # type: ignore[attr-defined]
    utils.is_cuda = lambda: False  # type: ignore[attr-defined]
    utils.is_hip = lambda: False  # type: ignore[attr-defined]
    utils.is_musa = lambda: True  # type: ignore[attr-defined]
    utils.is_sm90_supported = lambda: False  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(
        "xllm_sglang_fused_moe_triton_kernels", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load SGLang kernel source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_for(batch_size: int, dtype: str = "fp8") -> dict[str, int]:
    if dtype == "bf16":
        # TorchAda's S5000 BF16 E=256,N=512 table. SGLang selects the nearest
        # recorded token bucket; ties keep the lower bucket because JSON
        # insertion order is preserved.
        if batch_size == 1:
            return {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
            }
        if 2 <= batch_size <= 3:
            return {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
            }
        if 4 <= batch_size <= 6:
            return {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
            }
        if 7 <= batch_size <= 8:
            return {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
            }
        raise ValueError(f"Unsupported batch size: {batch_size}")
    if batch_size == 1:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
        }
    if 2 <= batch_size <= 4:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
        }
    if 5 <= batch_size <= 8:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
        }
    raise ValueError(f"Unsupported batch size: {batch_size}")


def _runtime_argument_count(ttir: str) -> int:
    signature = ttir.split("attributes {", maxsplit=1)[0]
    return len(re.findall(r"%arg\d+:", signature))


def _validate_runtime_abi(ttir: str, expected_argument_count: int) -> None:
    argument_count = _runtime_argument_count(ttir)
    if argument_count != expected_argument_count:
        raise RuntimeError(
            "SGLang fused_moe_kernel optimized ABI changed: "
            f"expected {expected_argument_count} arguments, found {argument_count}"
        )


def _compile_variant(
    module: types.ModuleType, batch_size: int, stage: str, dtype: str = "fp8"
) -> tuple[bytes, dict[str, Any]]:
    # Triton's direct JITFunction.run cache can alias variants whose runtime
    # arguments share the same specialization properties even when their
    # constexpr MoE stage differs. Compile each artifact from an empty cache;
    # otherwise a down-projection kernel can be written as a gate artifact.
    module.fused_moe_kernel.cache.clear()
    config = _config_for(batch_size, dtype)
    num_warps = config.pop("num_warps", 4)
    router_top_k = 8
    assignments = batch_size * router_top_k
    experts = 256
    if stage == "gate":
        rows = batch_size
        input_size = 2048
        output_size = 1024
        kernel_top_k = router_top_k
        mul_routed_weight = False
    else:
        rows = assignments
        input_size = 512
        output_size = 2048
        kernel_top_k = 1
        mul_routed_weight = True

    tensor_dtype = torch.float8_e4m3fn if dtype == "fp8" else torch.bfloat16
    input_tensor = torch.empty((rows, input_size), dtype=tensor_dtype, device="musa")
    weights = torch.empty(
        (experts, output_size, input_size),
        dtype=tensor_dtype,
        device="musa",
    )
    output = torch.empty(
        (assignments, output_size), dtype=torch.bfloat16, device="musa"
    )
    input_scale = torch.empty(
        (rows, input_size // 128), dtype=torch.float32, device="musa"
    )
    weight_scale = torch.empty(
        (experts, output_size // 128, input_size // 128),
        dtype=torch.float32,
        device="musa",
    )
    topk_weights = torch.empty(
        (batch_size, router_top_k), dtype=torch.float32, device="musa"
    )
    sorted_length = assignments * config["BLOCK_SIZE_M"]
    sorted_token_ids = torch.empty((sorted_length,), dtype=torch.int32, device="musa")
    expert_ids = torch.empty((assignments,), dtype=torch.int32, device="musa")
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device="musa")
    grid = (
        triton.cdiv(sorted_length, config["BLOCK_SIZE_M"])
        * triton.cdiv(output_size, config["BLOCK_SIZE_N"]),
    )

    kernel = module.fused_moe_kernel.run(
        input_tensor,
        None,
        weights,
        None,
        None,
        output,
        input_scale,
        weight_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output_size,
        input_size,
        sorted_length,
        assignments,
        input_tensor.stride(0),
        input_tensor.stride(1),
        weights.stride(0),
        weights.stride(2),
        weights.stride(1),
        0,
        0,
        output.stride(0),
        output.stride(1),
        input_scale.stride(0),
        input_scale.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(2),
        weight_scale.stride(1),
        128,
        128,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=dtype == "fp8",
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=True,
        c_sorted=False,
        filter_expert=False,
        swap_ab=False,
        FUSE_SUM_ALL_REDUCE=False,
        ROUTER_TOPK=router_top_k,
        num_warps=num_warps,
        num_stages=1,
        grid=grid,
        warmup=True,
        **config,
    )
    expected_argument_count = 22
    _validate_runtime_abi(kernel.asm["ttir"], expected_argument_count)
    binary = kernel.asm.get("mubin")
    if not isinstance(binary, bytes):
        raise RuntimeError("Triton did not produce a MUSA MUBIN artifact")
    manifest: dict[str, Any] = {
        "batch_size": batch_size,
        "stage": stage,
        "input_size": input_size,
        "output_size": output_size,
        "top_k": kernel_top_k,
        "dtype": dtype,
        "runtime_argument_count": expected_argument_count,
        "shared_memory": kernel.metadata.shared,
        "threads_per_block": num_warps * 32,
        "sha256": hashlib.sha256(binary).hexdigest(),
        **config,
    }
    return binary, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sglang-kernel-source",
        type=Path,
        required=True,
        help="Path to SGLang fused_moe_triton_kernels.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(_BATCH_SIZES)
    )
    parser.add_argument("--stages", choices=_STAGES, nargs="+", default=list(_STAGES))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    invalid_batches = sorted(set(args.batch_sizes) - set(_BATCH_SIZES))
    if invalid_batches:
        raise ValueError(f"Unsupported batch sizes: {invalid_batches}")

    torch.musa.set_device(0)
    module = _load_sglang_kernel(args.sglang_kernel_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    variant_hashes: dict[tuple[int, str], str] = {}
    for batch_size in args.batch_sizes:
        for stage in args.stages:
            binary, manifest = _compile_variant(
                module, batch_size, stage, args.dtype
            )
            variant_hashes[(batch_size, stage)] = manifest["sha256"]
            gate_hash = variant_hashes.get((batch_size, "gate"))
            down_hash = variant_hashes.get((batch_size, "down"))
            if gate_hash is not None and gate_hash == down_hash:
                raise RuntimeError(
                    f"Batch {batch_size} gate and down artifacts are identical; "
                    "the Triton specialization cache was reused incorrectly"
                )
            output = args.output_dir / f"b{batch_size}_{stage}.mubin"
            output.write_bytes(binary)
            output.with_suffix(".json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            variants.append(manifest)
            logger.info("Generated %s", output)

    root_manifest = {
        "target": "mp31",
        "kernel": "fused_moe_kernel",
        "runtime_argument_count": 22,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "sglang_kernel_source": str(args.sglang_kernel_source.resolve()),
        "variants": variants,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
