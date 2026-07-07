"""TileLang MUSA kernel for fused Qwen3.5 GDN projection split (contiguous layout).

Ported from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.gdn_fused_proj.
The C++ runtime uses a native MUSA kernel with identical indexing; this module is
the TileLang reference and can be AOT-compiled into FLASHINFER_OPS_PATH when
tilelang is available in the MUSA build environment.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import tilelang
import tilelang.language as T

_PASS_CONFIGS: dict = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False
for _key, _value in (
    ("TL_DISABLE_THREAD_STORAGE_SYNC", True),
    ("TL_ENABLE_MUSA_BURST", True),
    ("TL_ENABLE_REDUCE_BURST", True),
    ("TL_DISABLE_SAFE_MEMORY_ACCESS", True),
    ("TL_DISABLE_INDEX_TYPE_PROMOTION", True),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

MUSA_COMPILE_FLAGS = [
    "-Od3",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-enable-postra-sched=0",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def tilelang_dtype(dtype_name: str) -> str:
    if dtype_name in ("float16", "half"):
        return "float16"
    if dtype_name in ("bfloat16", "bf16"):
        return "bfloat16"
    if dtype_name in ("float32", "float"):
        return "float32"
    raise TypeError(f"Unsupported dtype for TileLang MUSA kernel: {dtype_name}")


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fused_qkvzba_split_reshape_cat_contiguous_kernel(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str,
    ba_dtype: str,
):
    m = T.dynamic("m")
    v_per_group = num_heads_v // num_heads_qk
    total_q = num_heads_qk * head_qk
    total_k = total_q
    total_v = num_heads_v * head_v
    qkv_dim = total_q + total_k + total_v
    total_qkvz = qkv_dim + total_v
    total_ba = num_heads_v * 2
    v_group_dim = v_per_group * head_v

    @T.prim_func
    def qkvzba_contiguous(
        mixed_qkv: T.Tensor((m, qkv_dim), input_dtype),
        z: T.Tensor((m, num_heads_v, head_v), input_dtype),
        b: T.Tensor((m, num_heads_v), ba_dtype),
        a: T.Tensor((m, num_heads_v), ba_dtype),
        mixed_qkvz: T.Tensor((m, total_qkvz), input_dtype),
        mixed_ba: T.Tensor((m, total_ba), ba_dtype),
    ):
        with T.Kernel(m, num_heads_qk, threads=128) as (row, hq):
            for d in T.Parallel(head_qk):
                mixed_qkv[row, hq * head_qk + d] = mixed_qkvz[row, hq * head_qk + d]
                mixed_qkv[row, total_q + hq * head_qk + d] = mixed_qkvz[
                    row, total_q + hq * head_qk + d
                ]

            for d in T.Parallel(v_group_dim):
                v_offset = hq * v_group_dim + d
                mixed_qkv[row, total_q + total_k + v_offset] = mixed_qkvz[
                    row, total_q + total_k + v_offset
                ]
                z[row, hq * v_per_group + d // head_v, d % head_v] = mixed_qkvz[
                    row, qkv_dim + v_offset
                ]

            for d in T.Parallel(v_per_group):
                v_head = hq * v_per_group + d
                b[row, v_head] = mixed_ba[row, v_head]
                a[row, v_head] = mixed_ba[row, num_heads_v + v_head]

    return qkvzba_contiguous


def compile_specialization(
    output_root: str | Path,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str = "bfloat16",
    ba_dtype: str = "bfloat16",
) -> Path:
    """Compile one specialization and write the TVM-FFI .so under output_root."""
    output_root = Path(output_root)
    uri = (
        f"gdn_fused_qkvzba_contiguous_hqk{num_heads_qk}_hv{num_heads_v}_"
        f"hk{head_qk}_hv{head_v}_{input_dtype}"
    )
    out_dir = output_root / uri
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / f"{uri}.so"
    if so_path.exists() and os.environ.get("XLLM_TILELANG_FORCE", "0") != "1":
        return so_path

    kernel = _fused_qkvzba_split_reshape_cat_contiguous_kernel(
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        tilelang_dtype(input_dtype),
        tilelang_dtype(ba_dtype),
    )
    if hasattr(kernel, "export_library"):
        kernel.export_library(str(so_path))
    else:
        artifact = kernel.get_kernel_source(wrap_return_func=True)
        raise RuntimeError(
            "tilelang kernel object does not support export_library; "
            f"generated source:\n{artifact}"
        )
    return so_path
