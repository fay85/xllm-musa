# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""MUSA GLM operators: MATE KDA with explicit GPU reference primitives."""

from xllm.python.kernels_musa.glm_reference import (
    batch_matmul_transpose,
    hc_post,
    rms_norm_sigmoid_gated,
)
from xllm.python.kernels_musa.kda import recurrent_kda
from xllm.python.kernels_musa.mate_ops import hc_pre_fused, rms_norm

__all__ = [
    "batch_matmul_transpose", "hc_post", "hc_pre_fused", "rms_norm",
    "rms_norm_sigmoid_gated", "recurrent_kda",
]
