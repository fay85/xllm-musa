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

"""MUSA gated-delta-network kernels."""

from __future__ import annotations

import os
from typing import Literal

import torch

GdnPrefillBackend = Literal["mate"]
GdnDecodeBackend = Literal["mate", "fused"]

fused_gdn_prefill_post_conv = torch.ops.xllm_ops.fused_gdn_prefill_post_conv
chunk_gated_delta_rule = torch.ops.xllm_ops.chunk_gated_delta_rule
fused_recurrent_gated_delta_rule_packed_decode = (
    torch.ops.xllm_ops.fused_recurrent_gated_delta_rule_packed_decode
)
mate_gated_delta_rule_decode = torch.ops.xllm_ops.mate_gated_delta_rule_decode
fused_gdn_mtp_checkpoint = torch.ops.xllm_ops.fused_gdn_mtp_checkpoint


def resolve_gdn_prefill_backend(
    capability: tuple[int, int] | None = None,
) -> GdnPrefillBackend:
    del capability
    return "mate"


def resolve_gdn_decode_backend() -> GdnDecodeBackend:
    """Match C++ ``XLLM_GDN_DECODE_BACKEND``; default is mate."""
    backend = os.environ.get("XLLM_GDN_DECODE_BACKEND", "mate")
    if backend not in ("mate", "fused"):
        raise RuntimeError(
            f"unsupported XLLM_GDN_DECODE_BACKEND={backend}; "
            "expected mate or fused"
        )
    return backend


__all__ = [
    "GdnPrefillBackend",
    "GdnDecodeBackend",
    "resolve_gdn_prefill_backend",
    "resolve_gdn_decode_backend",
    "fused_gdn_prefill_post_conv",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "mate_gated_delta_rule_decode",
    "fused_gdn_mtp_checkpoint",
    "chunk_gated_delta_rule",
]
