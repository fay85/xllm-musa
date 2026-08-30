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

"""Tensor-parallel gated MLP shared by Python model implementations."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_executor.forward_context import acquire_graph_activation

_FUSED_SWIGLU_GROUP_SIZE = 128


def _fused_dense_swiglu_fp8_enabled() -> bool:
    value = os.environ.get("XLLM_FUSED_DENSE_SWIGLU_FP8")
    if value is None:
        return True
    return value in {"1", "true", "TRUE", "True"}


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if intermediate_size % tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")
        local_intermediate_size = intermediate_size // tp_size
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * local_intermediate_size,
            tp_size,
            dtype=dtype,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            local_intermediate_size,
            hidden_size,
            tp_size,
            dtype=dtype,
            device=device,
            reduce_results=reduce_results,
        )
        self._silu_output: torch.Tensor | None = None

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("GatedMLP graph reserve requires max_tokens > 0")
        self.gate_up_proj.reserve_graph_workspace(max_tokens)
        self.down_proj.reserve_graph_workspace(max_tokens)
        fp8_weight = getattr(self.down_proj, "_fp8_weight", None)
        hidden = int(
            fp8_weight.size(1)
            if fp8_weight is not None
            else self.down_proj.weight.size(1)
        )
        need_silu = self._silu_output is None or self._silu_output.size(0) < max_tokens
        if need_silu:
            self._silu_output = torch.empty(
                (max_tokens, hidden),
                dtype=self.down_proj.weight.dtype,
                device=self.down_proj.weight.device,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._can_fused_fp8_mlp(hidden_states):
            # Keep gate_up as one C++ op (quant + GEMM). Splitting them
            # across Python added per-layer host bubbles and made eager
            # mlp_gate_up slower than the fused block_fp8_linear path.
            gate_up = self.gate_up_proj(hidden_states)
            quantized, scale = kernels.fused_swiglu_quant_fp8(
                gate_up, _FUSED_SWIGLU_GROUP_SIZE
            )
            return self.down_proj.forward_quantized(quantized, scale)
        gate_up = self.gate_up_proj(hidden_states)
        hidden = int(gate_up.size(-1) // 2)
        silu_output = acquire_graph_activation(
            (int(gate_up.size(0)), hidden),
            gate_up.dtype,
            gate_up.device,
        )
        if (
            silu_output is None
            and self._silu_output is not None
            and gate_up.size(0) <= self._silu_output.size(0)
        ):
            silu_output = self._silu_output[: gate_up.size(0)]
        activated = kernels.silu_and_mul(gate_up, silu_output)
        return self.down_proj(activated)

    def _can_fused_fp8_mlp(self, hidden_states: torch.Tensor) -> bool:
        if not _fused_dense_swiglu_fp8_enabled():
            return False
        gate_up_weight = getattr(self.gate_up_proj, "_fp8_weight", None)
        down_weight = getattr(self.down_proj, "_fp8_weight", None)
        if gate_up_weight is None or down_weight is None:
            return False
        if hidden_states.dim() != 2 or hidden_states.size(0) <= 0:
            return False
        if hidden_states.dtype != torch.bfloat16:
            return False
        if int(hidden_states.size(-1)) % _FUSED_SWIGLU_GROUP_SIZE != 0:
            return False
        return int(gate_up_weight.size(0)) % (2 * _FUSED_SWIGLU_GROUP_SIZE) == 0
