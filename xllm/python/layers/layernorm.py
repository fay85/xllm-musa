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

"""RMSNorm layer (with optional fused residual-add), matching xLLM's
``apply_norm``. Depends on the op dispatch layer (:mod:`python.ops`)."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.model_executor.forward_context import acquire_graph_activation
from xllm.python.platform import current_platform


class RMSNorm(nn.Module):
    """RMSNorm with optional fused residual-add, matching xLLM's apply_norm.

    - ``forward(x)`` -> normed x
    - ``forward(x, residual)`` -> (normed(x + residual), x + residual)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        self._graph_output: torch.Tensor | None = None

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("RMSNorm graph reserve requires max_tokens > 0")
        dim = int(self.weight.size(0))
        need_output = (
            self._graph_output is None or self._graph_output.size(0) < max_tokens
        )
        if need_output:
            self._graph_output = torch.empty(
                (max_tokens, dim),
                dtype=self.weight.dtype,
                device=self.weight.device,
            )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            graph_output = acquire_graph_activation(
                (int(x.size(0)), int(x.size(-1))), x.dtype, x.device
            )
            if graph_output is None:
                reserved = self._graph_output
                if reserved is not None and x.size(0) <= reserved.size(0):
                    graph_output = reserved[: x.size(0)]
            if graph_output is not None:
                return kernels.rms_norm(x, self.weight, self.eps, graph_output)
            return kernels.rms_norm(x, self.weight, self.eps)
        return kernels.fused_add_rms_norm(x, residual, self.weight, self.eps)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm used by Qwen3.5.

    Checkpoints store the offset from one, so the effective scale is
    ``1 + weight`` rather than ``weight``.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=device))
        self._activation_dtype = dtype
        self._kernel_weight: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("GemmaRMSNorm graph reserve requires max_tokens > 0")
        if self._activation_dtype is None:
            raise RuntimeError("GemmaRMSNorm reserve requires an activation dtype")
        dim = int(self.weight.size(0))
        need_output = (
            self._graph_output is None or self._graph_output.size(0) < max_tokens
        )
        if need_output:
            self._graph_output = torch.empty(
                (max_tokens, dim),
                dtype=self._activation_dtype,
                device=self.weight.device,
            )

    def _activation_weight(self, dtype: torch.dtype) -> torch.Tensor:
        cached = self._kernel_weight
        if cached is not None and cached.dtype == dtype and cached.is_contiguous():
            return cached
        self._kernel_weight = self.weight.detach().to(dtype=dtype).contiguous()
        return self._kernel_weight

    def _forward_musa(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self._activation_weight(x.dtype)
        if residual is None:
            output = acquire_graph_activation(
                (int(x.size(0)), int(x.size(-1))), x.dtype, x.device
            )
            if output is None:
                reserved = self._graph_output
                if reserved is not None and x.size(0) <= reserved.size(0):
                    output = reserved[: x.size(0)]
                else:
                    output = torch.empty_like(x)
            return kernels.gemma_rms_norm(x, weight, self.eps, output)
        if x.data_ptr() == residual.data_ptr():
            raise RuntimeError(
                "fused Gemma RMSNorm requires distinct residual storage"
            )
        if not x.is_contiguous() or not residual.is_contiguous():
            raise RuntimeError("fused Gemma RMSNorm requires contiguous tensors")
        return kernels.fused_add_gemma_rms_norm(x, residual, weight, self.eps)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if current_platform.is_musa():
            return self._forward_musa(x, residual)
        original_dtype = x.dtype
        normalized = x.float()
        if residual is not None:
            normalized = normalized + residual.float()
            residual = normalized.to(original_dtype)

        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = normalized * (self.weight + 1.0)
        output = normalized.to(original_dtype)
        if residual is None:
            return output
        return output, residual
