# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit MUSA primitives for the eager DeepSeek-V4.1 text model.

The FP32 RMSNorm contract used by V4.1 is evaluated by TorchMUSA on GPU:
MATE 0.2.7 RMSNorm accepts only FP16/BF16 and cannot receive that input.
Half-precision callers use MATE directly. No operation copies inputs to CPU.
"""

import torch
import torch.nn.functional as F


def rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    if value.device.type != "musa" or weight.device != value.device:
        raise ValueError("MUSA RMSNorm requires input and weight on the same MUSA device")
    if value.dtype == torch.float32 and weight.dtype == torch.float32:
        variance = value.square().mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + epsilon) * weight
    if value.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != value.dtype:
        raise ValueError("MATE RMSNorm requires matching FP16/BF16 input and weight")
    from mate.norm import rmsnorm

    return rmsnorm(value.reshape(-1, value.shape[-1]), weight, eps=epsilon).reshape_as(value)


def prepare_row_parallel_weight(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    return weight.contiguous(), False


def silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    if value.device.type != "musa":
        raise ValueError("MUSA SwiGLU requires a MUSA tensor")
    gate, up = value.chunk(2, dim=-1)
    return F.silu(gate) * up
