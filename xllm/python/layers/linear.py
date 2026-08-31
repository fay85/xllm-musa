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

"""Tensor-parallel linear layers.

At ``tp_size==1`` these hold full-size weights and skip all collectives, so they
are numerically identical to plain ``nn.Linear`` and preserve the single-card
byte parity. At ``tp_size>1`` each rank holds a per-partition shard and inserts
the same all-reduce / all-gather the native C++ parallel layers use (via the op
dispatch layer :mod:`python.ops`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.model_executor.forward_context import acquire_graph_activation


def _reserve_linear_workspace(module: nn.Module, max_tokens: int) -> None:
    if max_tokens <= 0:
        raise RuntimeError("linear graph reserve requires max_tokens > 0")
    fp8_weight = getattr(module, "_fp8_weight", None)
    effective_weight = fp8_weight if fp8_weight is not None else module.weight
    graph_output = getattr(module, "_graph_output", None)
    out_features = int(effective_weight.size(0))
    need_output = graph_output is None or graph_output.size(0) < max_tokens
    if need_output:
        module._graph_output = torch.empty(
            (max_tokens, out_features),
            dtype=getattr(module, "_output_dtype", effective_weight.dtype),
            device=effective_weight.device,
        )


def _load_fp8_weight(
    module: nn.Module,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> None:
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError("block-fp8 weight and scale must be 2D")
    target_device = module.weight.device
    output_dtype = module.weight.dtype
    # The safetensors loader exposes CPU mmap tensors. Move both operands to
    # MUSA before dispatching the block-FP8 op, and release the BF16
    # constructor storage that is not used by the FP8 path.
    module.weight.data = torch.empty(
        0,
        dtype=output_dtype,
        device=target_device,
    )
    module._fp8_weight = weight.to(device=target_device).contiguous()
    module._weight_scale_inv = scale.to(
        device=target_device, dtype=torch.float32
    ).contiguous()
    module._output_dtype = output_dtype
    module._fp8_block_n = block_n
    module._fp8_block_k = block_k


def _acquire_linear_output(
    x: torch.Tensor,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    graph_output: torch.Tensor | None,
) -> torch.Tensor | None:
    if x.dim() != 2:
        return None
    pooled = acquire_graph_activation(
        (int(x.size(0)), int(out_features)), dtype, device
    )
    if pooled is not None:
        return pooled
    if (
        graph_output is not None
        and x.size(0) <= graph_output.size(0)
        and graph_output.size(1) == out_features
    ):
        return graph_output[: x.size(0)]
    return None


def _fp8_quantized_linear(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    module: nn.Module,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    fp8_weight = getattr(module, "_fp8_weight", None)
    if fp8_weight is None:
        raise RuntimeError("FP8 quantized linear requires loaded FP8 weights")
    output_dtype = getattr(module, "_output_dtype", torch.bfloat16)
    output = _acquire_linear_output(
        quantized,
        int(fp8_weight.size(0)),
        output_dtype,
        quantized.device,
        getattr(module, "_graph_output", None),
    )
    result = kernels.block_fp8_linear_quantized(
        quantized,
        scale,
        fp8_weight,
        module._weight_scale_inv,
        module._fp8_block_n,
        module._fp8_block_k,
        output,
        output_dtype,
    )
    if bias is not None:
        result = result + bias
    return result


def _maybe_fp8_linear(
    x: torch.Tensor,
    module: nn.Module,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    fp8_weight = getattr(module, "_fp8_weight", None)
    if fp8_weight is None:
        return None
    output = _acquire_linear_output(
        x,
        int(fp8_weight.size(0)),
        getattr(module, "_output_dtype", x.dtype),
        x.device,
        getattr(module, "_graph_output", None),
    )
    result = kernels.block_fp8_linear(
        x,
        fp8_weight,
        module._weight_scale_inv,
        module._fp8_block_n,
        module._fp8_block_k,
        output,
    )
    if bias is not None:
        result = result + bias
    return result


def _linear_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    graph_output: torch.Tensor | None,
) -> torch.Tensor:
    # Match C++ musa::matmul(output_buf): mm_out / addmm_out into a persistent
    # buffer so capture does not record aten empty for every GEMM.
    output = _acquire_linear_output(
        x, int(weight.size(0)), x.dtype, x.device, graph_output
    )
    if output is None:
        return torch.nn.functional.linear(x, weight, bias)
    # Prefer a contiguous [K, N] operand. ColumnParallelLinear.prefer_nn_gemm_layout
    # rewrites [N, K] storage so weight.t() is already contiguous (NN GEMM).
    weight_t = weight.t()
    if bias is not None:
        torch.addmm(bias, x, weight_t, out=output)
    else:
        torch.mm(x, weight_t, out=output)
    return output


class ColumnParallelLinear(nn.Module):
    """Linear sharded on the output dim (dim 0): each rank owns
    ``[out_per_partition, in]`` and computes its slice of the output. No
    communication unless ``gather_output`` (then an all-gather along the last
    dim reconstructs the full output — used by lm_head). An optional bias is
    sharded on the output dim like the weight and applied per partition (before
    any gather). Mirrors native ColumnParallelLinear / QKVParallelLinear (which
    set gather_output=False so the following RowParallel all-reduce combines the
    partial outputs).
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_partition: int,
        tp_size: int,
        gather_output: bool = False,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(
                out_features_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features_per_partition, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)
        self._out_features = out_features_per_partition
        self._graph_output: torch.Tensor | None = None
        self._fp8_weight: torch.Tensor | None = None
        self._weight_scale_inv: torch.Tensor | None = None
        self._fp8_block_n = 128
        self._fp8_block_k = 128

    def load_fp8(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        block_n: int = 128,
        block_k: int = 128,
    ) -> None:
        _load_fp8_weight(self, weight, scale, block_n, block_k)

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        _reserve_linear_workspace(self, max_tokens)

    def prefer_nn_gemm_layout(self) -> None:
        """Store ``[N, K]`` so ``weight.t()`` is contiguous ``[K, N]``.

        ``_linear_out`` runs ``torch.mm(x, weight.t())``. Row-major ``[N, K]``
        makes that a non-contiguous NT GEMM. Rewriting storage to column-major
        ``[N, K]`` (physical ``[K, N]``) keeps the Parameter shape and values
        and lets M=1 vocab GEMM use the faster NN kernel. No extra resident
        copy: the old storage is released after the rewrite.
        """
        weight = self.weight
        if weight.dim() != 2:
            raise ValueError("prefer_nn_gemm_layout requires a 2D weight")
        out_features = int(weight.size(0))
        if weight.stride(0) == 1 and weight.stride(1) == out_features:
            return
        weight.data = weight.detach().t().contiguous().t()

    def forward_quantized(
        self, quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        out = _fp8_quantized_linear(quantized, scale, self, self.bias)
        if self.gather_output and self.tp_size > 1:
            out = kernels.all_gather(out, dim=-1, world_size=self.tp_size)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _maybe_fp8_linear(x, self, self.bias)
        if out is None:
            out = _linear_out(x, self.weight, self.bias, self._graph_output)
        if self.gather_output and self.tp_size > 1:
            out = kernels.all_gather(out, dim=-1, world_size=self.tp_size)
        return out


class RowParallelLinear(nn.Module):
    """Linear sharded on the input dim (dim 1): each rank owns
    ``[out, in_per_partition]`` and consumes its slice of an already-partitioned
    input, producing a partial output that is SUM all-reduced across the TP
    group. An optional bias is replicated (full ``out``) and added once AFTER
    the all-reduce, so it is not summed ``tp_size`` times. Mirrors native
    RowParallelLinear (o_proj / down_proj with enable_result_reduction=true).
    """

    def __init__(
        self,
        in_features_per_partition: int,
        out_features: int,
        tp_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features_per_partition,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)
        self._out_features = out_features
        self._graph_output: torch.Tensor | None = None
        self._fp8_weight: torch.Tensor | None = None
        self._weight_scale_inv: torch.Tensor | None = None
        self._fp8_block_n = 128
        self._fp8_block_k = 128

    def load_fp8(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        block_n: int = 128,
        block_k: int = 128,
    ) -> None:
        _load_fp8_weight(self, weight, scale, block_n, block_k)

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        _reserve_linear_workspace(self, max_tokens)

    def forward_quantized(
        self, quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        out = _fp8_quantized_linear(quantized, scale, self, None)
        if self.reduce_results and self.tp_size > 1:
            kernels.all_reduce_(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bias is applied after the TP reduce so it is not summed tp_size times.
        out = _maybe_fp8_linear(x, self, None)
        if out is None:
            out = _linear_out(x, self.weight, None, self._graph_output)
        if self.reduce_results and self.tp_size > 1:
            kernels.all_reduce_(out)
        if self.bias is not None:
            out = out + self.bias
        return out
