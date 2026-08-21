# Copyright 2026 The xLLM Authors.
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

"""MUSA kernel semantic API for the Python model executor.

``xllm.python.initialize_runtime()`` binds this package as
``xllm.python.kernels`` on MUSA. Ordinary package import is build-safe and does
not inspect native operators until ``_initialize_runtime`` runs.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "normalization": (
        "rms_norm",
        "fused_add_rms_norm",
        "gemma_rms_norm",
        "fused_add_gemma_rms_norm",
        "rms_norm_gated",
    ),
    "activation": (
        "silu_and_mul",
        "mul_sigmoid_gate_inplace",
        "fused_swiglu_quant_fp8",
    ),
    "rotary_embedding": ("fused_qk_norm_rope",),
    "attention": (
        "reshape_paged_cache",
        "update_decode_graph_metadata",
        "update_fa3_graph_metadata",
        "fa3_decode_scheduler_metadata",
        "fa3_decode",
        "fa3_prefill",
    ),
    "collectives": ("all_gather", "all_reduce_", "init_tp_group"),
    "quantization": (
        "block_fp8_linear",
        "block_fp8_linear_quantized",
        "per_token_group_quant_fp8",
    ),
    "causal_conv1d": (
        "causal_conv1d_prefill",
        "causal_conv1d_decode",
        "causal_conv1d_mtp_verify",
    ),
    "gated_delta_net": (
        "resolve_gdn_prefill_backend",
        "resolve_gdn_decode_backend",
        "fused_gdn_prefill_post_conv",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "mate_gated_delta_rule_decode",
        "fused_gdn_mtp_checkpoint",
        "chunk_gated_delta_rule",
    ),
    "prefill_piecewise": (
        "python_prefill_piecewise_begin",
        "python_prefill_piecewise_end",
        "python_prefill_piecewise_replay",
    ),
}

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "gemma_rms_norm",
    "fused_add_gemma_rms_norm",
    "rms_norm_gated",
    "silu_and_mul",
    "mul_sigmoid_gate_inplace",
    "fused_swiglu_quant_fp8",
    "fused_qk_norm_rope",
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "update_fa3_graph_metadata",
    "fa3_decode_scheduler_metadata",
    "fa3_decode",
    "fa3_prefill",
    "all_gather",
    "all_reduce_",
    "init_tp_group",
    "block_fp8_linear",
    "block_fp8_linear_quantized",
    "per_token_group_quant_fp8",
    "causal_conv1d_prefill",
    "causal_conv1d_decode",
    "causal_conv1d_mtp_verify",
    "resolve_gdn_prefill_backend",
    "resolve_gdn_decode_backend",
    "fused_gdn_prefill_post_conv",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "mate_gated_delta_rule_decode",
    "fused_gdn_mtp_checkpoint",
    "chunk_gated_delta_rule",
    "python_prefill_piecewise_begin",
    "python_prefill_piecewise_end",
    "python_prefill_piecewise_replay",
]
_runtime_initialized = False


def _initialize_runtime() -> None:
    """Load native-op bindings and publish the MUSA semantic API once."""

    global _runtime_initialized
    if _runtime_initialized:
        return

    importlib.import_module(f"{__name__}._custom_op")
    exported: dict[str, Any] = {}
    for module_name, names in _EXPORTS.items():
        module = importlib.import_module(f"{__name__}.{module_name}")
        exported.update((name, getattr(module, name)) for name in names)

    globals().update(exported)
    _runtime_initialized = True


def __getattr__(name: str) -> Any:
    if name in __all__:
        raise RuntimeError(
            "xllm.python.kernels_musa runtime is not initialized; call "
            "xllm.python.initialize_runtime() after registering native "
            "torch operators"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
