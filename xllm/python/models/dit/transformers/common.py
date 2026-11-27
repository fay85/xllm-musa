# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Building blocks shared by DiT transformers.

Python counterparts of the ``qwenimage::`` blocks in
``transformer_qwen_image.h`` (Timesteps / TimestepEmbedding / FeedForward),
kept diffusers-checkpoint compatible: the FeedForward loads ``net.0.proj``
into ``linear1`` and ``net.2`` into ``linear2``, matching the C++ loaders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xllm.python.models.dit.dit_component_base import require_loaded

_kernels_module = None
_kernels_checked = False


def _kernels():
    """Return the active platform kernel package, or ``None`` when the runtime
    has not been initialized (build tooling, offline unit tests)."""
    global _kernels_module, _kernels_checked
    if not _kernels_checked:
        _kernels_checked = True
        try:
            from xllm.python import kernels as _k

            _kernels_module = _k
        except RuntimeError:
            _kernels_module = None
    return _kernels_module


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (diffusers ``get_timestep_embedding``)."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            raise ValueError("Timesteps should be a 1d-array")

        half_dim = self.embedding_dim // 2
        exponent = (
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            / (half_dim - self.downscale_freq_shift)
        )
        emb = torch.exp(exponent)
        emb = timesteps.unsqueeze(1).to(torch.float32) * emb.unsqueeze(0)
        emb = self.scale * emb

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        if self.flip_sin_to_cos:
            emb = torch.cat([cos_emb, sin_emb], dim=-1)
        else:
            emb = torch.cat([sin_emb, cos_emb], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    """linear_1 -> SiLU -> linear_2 over the timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = -1,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act_fn = nn.SiLU()
        time_embed_dim_out = out_dim if out_dim > 0 else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(sample)
        x = self.act_fn(x)
        return self.linear_2(x)

    def load_weights(self, state_dicts: list) -> None:
        loaded = {
            "linear_1.weight": False,
            "linear_1.bias": False,
            "linear_2.weight": False,
            "linear_2.bias": False,
        }
        for state_dict in state_dicts:
            for name, param in (
                ("linear_1.weight", self.linear_1.weight),
                ("linear_1.bias", self.linear_1.bias),
                ("linear_2.weight", self.linear_2.weight),
                ("linear_2.bias", self.linear_2.bias),
            ):
                if not loaded[name] and state_dict.has(name):
                    param.data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)


class FeedForward(nn.Module):
    """GELU-tanh feed-forward with diffusers ``net.0.proj``/``net.2`` layout."""

    def __init__(self, dim: int, dim_out: int, mult: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, inner_dim, bias=True)
        self.linear2 = nn.Linear(inner_dim, dim_out, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.nn.functional.gelu(self.linear1(hidden_states), approximate="tanh"))


class RMSNorm(nn.Module):
    """Weight-only RMSNorm over the last dimension, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused ``npu_rms_norm`` path (bit-exact with the fp32 reference below).
        kernels = _kernels()
        if kernels is not None:
            return kernels.rms_norm_fp32(x, self.weight, self.eps)
        x_float = x.to(torch.float32)
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(torch.float32)).to(x.dtype)


def fp32_layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Affine-free LayerNorm computed in float32 (matches the C++ fp32_norm)."""
    x_float = x.to(torch.float32)
    return torch.nn.functional.layer_norm(x_float, (x_float.size(-1),), None, None, eps).to(x.dtype)
