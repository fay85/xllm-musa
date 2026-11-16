"""MATE S4FP8 masked GEMM with SGLang's corrected pipeline header."""
from functools import lru_cache
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _module(out_dtype, a_quant_recipe, scale_a_major):
    from mate.jit.gemm.masked_moe_gemm_mixed_dtype import (
        gen_masked_moe_gemm_mixed_dtype_spec,
        masked_moe_gemm_mixed_dtype_dispatch_name,
    )

    include = Path(__file__).resolve().parent / "include"
    header = include / "mate/gemm/deep_gemm/mp31_masked_moe_gemm_mixed_dtype.hpp"
    digest = hashlib.sha256(header.read_bytes()).hexdigest()[:16]
    spec = gen_masked_moe_gemm_mixed_dtype_spec(
        out_dtype, a_quant_recipe, scale_a_major
    )
    # The public MATE loader otherwise resolves its installed header and may
    # load an AOT binary which predates the shared-storage/pipeline fixes.
    spec.name += "_sglang_" + digest
    spec.extra_include_dirs = [include, *(spec.extra_include_dirs or [])]
    spec.aot_path = None
    module = spec.build_and_load()
    logger.info("MUSA W4A8 masked GEMM header=%s sha256=%s library=%s",
                header, digest, spec.jit_library_path)
    return masked_moe_gemm_mixed_dtype_dispatch_name(
        out_dtype, a_quant_recipe, scale_a_major
    ), module


def masked_moe_gemm_w4a8(
    input_a, input_b, masked_tokens_info, out, expect_tokens, a_quant_recipe
):
    from mate.gemm import check_w4a8_mutlass
    from mate.mate_runtime import resolve_num_mps

    if not check_w4a8_mutlass(
        input_a, input_b, masked_tokens_info, out, a_quant_recipe, False
    ):
        raise ValueError("Unsupported non-overlap MATE S4FP8 masked GEMM layout")
    a, scale_a = input_a
    b, scale_b = input_b
    if scale_b.shape[-1] % 2:
        padded = torch.zeros(
            (*scale_b.shape[:-1], scale_b.shape[-1] + 1),
            device=scale_b.device, dtype=scale_b.dtype,
        )
        padded[..., :-1].copy_(scale_b)
        scale_b = padded
    major = "M" if a_quant_recipe == (1, 128) and scale_a.stride(1) == 1 else "K"
    name, module = _module(out.dtype, a_quant_recipe, major)
    module.get_function(name)(
        a, scale_a, b, scale_b, out, masked_tokens_info,
        int(expect_tokens), resolve_num_mps(a.device, None),
    )
    return out
