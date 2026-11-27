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

"""JoyOV2 audio VAE (AutoencoderKLXDAC) — decoder-only.

Port of ``xllm/models/dit/autoencoders/autoencoder_kl_xdac.h`` (itself a
decoder-only port of the SGLang / diffusers XDAC codec). Latent 128ch, output
stereo waveform, hop length = prod(encoder_rates) = 2048 by default.

The checkpoint stores torch parametrized weight-norm keys
(``*.parametrizations.weight.original0`` / ``original1``); the legacy
``weight_g`` / ``weight_v`` names are accepted as fallback. All weights and
activations stay fp32 — the C++ original forces float32 regardless of the
pipeline-wide DiT dtype, and ``keep_fp32_modules`` restores that after the
executor's device/dtype sweep.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.models.dit.dit_component_base import DiTComponentBase, prefixed, require_loaded

# Match the native XDAC decoder: add the division guard AFTER sqrt.
# torch.nn.utils.weight_norm itself does not add this guard.
_WEIGHT_NORM_EPS = 1e-8


def _load_wn_weight(module: nn.Module, state_dicts: list, loaded: dict) -> None:
    """Load weight-norm factors; prefer parametrizations keys, fall back."""
    g_key = "parametrizations.weight.original0"
    v_key = "parametrizations.weight.original1"
    for state_dict in state_dicts:
        if not loaded[g_key]:
            for key in (g_key, "weight_g"):
                if state_dict.has(key):
                    module.weight_g.data.copy_(state_dict.get_tensor(key))
                    loaded[g_key] = True
                    break
        if not loaded[v_key]:
            for key in (v_key, "weight_v"):
                if state_dict.has(key):
                    module.weight_v.data.copy_(state_dict.get_tensor(key))
                    loaded[v_key] = True
                    break


class WeightNormConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight_g = nn.Parameter(torch.empty(out_channels, 1, 1), requires_grad=False)
        self.weight_v = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.weight_v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt() + _WEIGHT_NORM_EPS
        weight = self.weight_v * self.weight_g / norm
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation)

    def load_weights(self, state_dicts: list) -> None:
        loaded = {
            "parametrizations.weight.original0": False,
            "parametrizations.weight.original1": False,
        }
        if self.bias is not None:
            loaded["bias"] = False
        _load_wn_weight(self, state_dicts, loaded)
        if self.bias is not None:
            for state_dict in state_dicts:
                if not loaded["bias"] and state_dict.has("bias"):
                    self.bias.data.copy_(state_dict.get_tensor("bias"))
                    loaded["bias"] = True
                    break
        require_loaded(loaded, type(self).__name__)


class WeightNormConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight_g = nn.Parameter(torch.empty(in_channels, 1, 1), requires_grad=False)
        self.weight_v = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.weight_v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt() + _WEIGHT_NORM_EPS
        weight = self.weight_v * self.weight_g / norm
        return F.conv_transpose1d(x, weight, self.bias, self.stride, self.padding, self.output_padding)

    def load_weights(self, state_dicts: list) -> None:
        loaded = {
            "parametrizations.weight.original0": False,
            "parametrizations.weight.original1": False,
            "bias": False,
        }
        _load_wn_weight(self, state_dicts, loaded)
        for state_dict in state_dicts:
            if not loaded["bias"] and state_dict.has("bias"):
                self.bias.data.copy_(state_dict.get_tensor("bias"))
                loaded["bias"] = True
        require_loaded(loaded, type(self).__name__)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.reciprocal(self.alpha + 1e-9) * torch.sin(self.alpha * x).pow(2)

    def load_weights(self, state_dicts: list) -> None:
        for state_dict in state_dicts:
            if state_dict.has("alpha"):
                self.alpha.data.copy_(state_dict.get_tensor("alpha"))
                return
        raise RuntimeError(f"weights are not loaded for {type(self).__name__}: ['alpha']")


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = WeightNormConv1d(dim, dim, 7, 1, pad, dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = WeightNormConv1d(dim, dim, 1, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        pad = (x.size(-1) - y.size(-1)) // 2
        if pad > 0:
            x = x[..., pad : x.size(-1) - pad]
        return x + y

    def load_weights(self, state_dicts: list) -> None:
        # Checkpoint layout: block.0/1/2/3 = snake, wnconv, snake, wnconv.
        self.snake1.load_weights(prefixed(state_dicts, "block.0."))
        self.conv1.load_weights(prefixed(state_dicts, "block.1."))
        self.snake2.load_weights(prefixed(state_dicts, "block.2."))
        self.conv2.load_weights(prefixed(state_dicts, "block.3."))


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        pad = int(math.ceil(stride / 2.0))
        out_pad = stride % 2
        self.snake = Snake1d(input_dim)
        self.upsample = WeightNormConvTranspose1d(input_dim, output_dim, 2 * stride, stride, pad, out_pad)
        self.res1 = ResidualUnit(output_dim, 1)
        self.res3 = ResidualUnit(output_dim, 3)
        self.res9 = ResidualUnit(output_dim, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.upsample(self.snake(x))
        h = self.res1(h)
        h = self.res3(h)
        return self.res9(h)

    def load_weights(self, state_dicts: list) -> None:
        # Checkpoint layout: block.0=snake, block.1=transpose, block.2/3/4=res.
        self.snake.load_weights(prefixed(state_dicts, "block.0."))
        self.upsample.load_weights(prefixed(state_dicts, "block.1."))
        self.res1.load_weights(prefixed(state_dicts, "block.2."))
        self.res3.load_weights(prefixed(state_dicts, "block.3."))
        self.res9.load_weights(prefixed(state_dicts, "block.4."))


class XDACDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        channels: int,
        rates: list[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.first_conv = WeightNormConv1d(latent_channels, channels, 7, 1, 3, 1)
        layers = []
        for i, rate in enumerate(rates):
            in_dim = channels // (1 << i)
            out_dim = channels // (1 << (i + 1))
            layers.append(DecoderBlock(in_dim, out_dim, rate))
        self.layers = nn.ModuleList(layers)
        final_dim = channels // (1 << len(rates))
        self.out_snake1 = Snake1d(final_dim)
        self.out_conv1 = WeightNormConv1d(final_dim, final_dim, 7, 1, 3, 1)
        self.out_snake2 = Snake1d(final_dim)
        self.out_conv2 = WeightNormConv1d(final_dim, out_channels, 7, 1, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(z)
        for layer in self.layers:
            x = layer(x)
        x = self.out_conv1(self.out_snake1(x))
        x = self.out_conv2(self.out_snake2(x))
        return torch.tanh(x)

    def load_weights(self, state_dicts: list) -> None:
        self.first_conv.load_weights(prefixed(state_dicts, "first_conv."))
        for i, layer in enumerate(self.layers):
            layer.load_weights(prefixed(state_dicts, f"layers.{i}."))
        self.out_snake1.load_weights(prefixed(state_dicts, "out.0."))
        self.out_conv1.load_weights(prefixed(state_dicts, "out.1."))
        self.out_snake2.load_weights(prefixed(state_dicts, "out.2."))
        self.out_conv2.load_weights(prefixed(state_dicts, "out.3."))


class AutoencoderKLXDAC(DiTComponentBase):
    """JoyOV2 audio VAE: decode [B, C, T] latents to [B, 2, samples] waveform."""

    def __init__(
        self,
        config: dict,
        vae_group: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Audio codec weights are fp32; ignore the pipeline-wide DiT dtype.
        # The XDAC decoder has no spatial-parallel path, so the VAE group is
        # accepted and unused (the executor reuses the image-VAE group slot).
        del vae_group, dtype
        DiTComponentBase.__init__(self)

        self.latent_channels = int(config.get("latent_channels", 128))
        self.out_channels = int(config.get("out_channels", 2))
        self.sampling_rate = int(config.get("sampling_rate", 44100))
        self.decoder_dim = int(config.get("decoder_dim", 2048))
        self.encoder_rates = [int(v) for v in (config.get("encoder_rates") or [2, 4, 4, 8, 8])]
        self.decoder_rates = [int(v) for v in (config.get("decoder_rates") or [8, 8, 4, 4, 2])]
        self.latents_mean = [float(v) for v in (config.get("latents_mean") or [])]
        self.latents_std = [float(v) for v in (config.get("latents_std") or [])]

        self.post_quant_conv = nn.Conv1d(self.latent_channels, self.latent_channels, kernel_size=1)
        self.decoder = XDACDecoder(self.latent_channels, self.decoder_dim, self.decoder_rates, self.out_channels)
        self.register_buffer("latents_mean_buf", None)
        self.register_buffer("latents_inv_std_buf", None)

    @property
    def hop_length(self) -> int:
        hop = 1
        for rate in self.encoder_rates:
            hop *= rate
        return hop if hop > 0 else 2048

    def keep_fp32_modules(self) -> None:
        self.float()

    def load_weights(self, state_dicts: list) -> None:
        loaded = {
            "post_quant_conv.weight": False,
            "post_quant_conv.bias": False,
        }
        for state_dict in state_dicts:
            for key in ("post_quant_conv.weight", "post_quant_conv.bias"):
                if not loaded[key] and state_dict.has(key):
                    param = getattr(self.post_quant_conv, key.split(".")[1])
                    param.data.copy_(state_dict.get_tensor(key))
                    loaded[key] = True
        require_loaded(loaded, type(self).__name__)
        self.decoder.load_weights(prefixed(state_dicts, "decoder."))
        for state_dict in state_dicts:
            if state_dict.has("latents_mean_buf"):
                self.latents_mean_buf = state_dict.get_tensor("latents_mean_buf").to(torch.float32).cpu()
            if state_dict.has("latents_inv_std_buf"):
                self.latents_inv_std_buf = state_dict.get_tensor("latents_inv_std_buf").to(torch.float32).cpu()

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.latents_mean_buf is not None and self.latents_inv_std_buf is not None:
            return latents.to(torch.float32) / self.latents_inv_std_buf.to(latents.device) + self.latents_mean_buf.to(
                latents.device
            )
        if not self.latents_mean or not self.latents_std:
            return latents
        if len(self.latents_mean) != latents.size(1) or len(self.latents_std) != latents.size(1):
            raise ValueError("latents_mean/std length must match latent channels")
        mean = torch.tensor(self.latents_mean, dtype=torch.float32, device=latents.device).view(1, -1, 1)
        std = torch.tensor(self.latents_std, dtype=torch.float32, device=latents.device).view(1, -1, 1)
        return latents.to(torch.float32) * std + mean

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 3:
            raise ValueError(f"XDAC decode expects [B,C,T], got {latent.shape}")
        if latent.size(1) != self.latent_channels:
            raise ValueError(f"XDAC decode channel mismatch: {latent.size(1)} vs {self.latent_channels}")
        # fp32 to match the weight-norm conv dtype.
        z = self.post_quant_conv(latent.to(torch.float32))
        return self.decoder(z)
