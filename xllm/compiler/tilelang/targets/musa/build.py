"""MUSA TileLang AOT build entry (scaffold).

Native kernels in core/kernels/musa/gdn_decode.cu mirror the TileLang reference
in kernels/gdn_fused_proj.py. Run compile_gdn_fused_proj.py from a MUSA dev
container when tilelang is installed to produce TVM-FFI artifacts.
"""

from __future__ import annotations

from pathlib import Path

from ...common.manifest import KernelFamilyManifest


def build_kernels(
    output_root: str | Path,
    kernel_names: list[str] | None = None,
    force: bool = False,
) -> list[KernelFamilyManifest]:
    if kernel_names and "gdn_fused_proj" not in kernel_names:
        return []
    from .kernels.gdn_fused_proj import compile_specialization

    # Qwen3.5-27B local TP=1 defaults (override via env as needed).
    import os

    compile_specialization(
        output_root,
        num_heads_qk=int(os.environ.get("XLLM_GDN_HQK", "4")),
        num_heads_v=int(os.environ.get("XLLM_GDN_HV", "8")),
        head_qk=int(os.environ.get("XLLM_GDN_HEAD_QK", "128")),
        head_v=int(os.environ.get("XLLM_GDN_HEAD_V", "128")),
        input_dtype=os.environ.get("XLLM_GDN_DTYPE", "bfloat16"),
        ba_dtype=os.environ.get("XLLM_GDN_BA_DTYPE", "bfloat16"),
    )
    return []
