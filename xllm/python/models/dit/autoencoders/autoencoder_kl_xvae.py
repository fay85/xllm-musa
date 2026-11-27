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

"""JoyOV2 video VAE (AutoencoderKLXVAE) — decoder-only.

Port of ``xllm/models/dit/autoencoders/autoencoder_kl_xvae.h`` (itself a
decoder-only port of the SGLang / diffusers XVAE). Spatial compression is
patch_size * 2^4 = 32x, temporal 4x, z_dim 128. Full-sequence decode only.

All weights and activations stay fp32 (the C++ original forces float32
regardless of the pipeline-wide DiT dtype); ``keep_fp32_modules`` restores
that after the executor's device/dtype sweep.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.models.dit.dit_component_base import DiTComponentBase, prefixed, require_loaded
from xllm.python.parallel import VaeParallelMixin


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# Temporal-upsample blocks slice over the input T above this size (exact
# with a 2-frame causal context; see UpsampleBlock._forward_temporal_sliced).
# The exactness test raises both to build a full-sequence reference.
TEMPORAL_SLICE_FRAMES = 16
# Frame-local (spatial-only) upsample blocks always slice; this caps the
# fp32 transients of the deepest scales.
SPATIAL_SLICE_FRAMES = 8


class CausalConv3d(nn.Module, VaeParallelMixin):
    """Causal 3D conv: temporal front pad 2*pad_t, H/W pad inside the conv.

    Unlike the C++ original (which picks between F.pad-everything and
    conv-side spatial padding based on VAE-parallel state at construction),
    the spatial padding always lives in the Conv3d here; halo exchange is
    decided per-forward from the tensor's device. Both layouts zero-pad the
    same amounts, so results are identical in the single-rank case.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        vae_group: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.padding = list(padding)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, padding[1], padding[2]),
            bias=True,
        )
        # F.pad layout {W_l, W_r, H_l, H_r, T_front, T_back}: temporal only.
        self._temporal_pad = (0, 0, 0, 0, 2 * padding[0], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_halo = self.vae_parallel_enabled(x.device) and self.padding[1] == 1 and self.padding[2] == 1
        if use_halo:
            x = self.exchange(x, pad=True)
        x = F.pad(x, self._temporal_pad)
        out = self.conv(x)
        if use_halo:
            out = out[..., 1 : out.size(-1) - 1]
        return out

    def load_weights(self, state_dicts: list) -> None:
        loaded = {"weight": False, "bias": False}
        for state_dict in state_dicts:
            if not loaded["weight"] and state_dict.has("weight"):
                self.conv.weight.data.copy_(state_dict.get_tensor("weight"))
                loaded["weight"] = True
            if not loaded["bias"] and state_dict.has("bias"):
                self.conv.bias.data.copy_(state_dict.get_tensor("bias"))
                loaded["bias"] = True
        require_loaded(loaded, type(self).__name__)


class XVAERMSNorm(nn.Module):
    """Channel-dim RMSNorm with gamma [C, 1, 1, 1], fp32 internal."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
        norm_input = x.to(torch.float32) if needs_fp32 else x
        normed = F.normalize(norm_input, dim=1, eps=1e-12).to(x.dtype)
        return normed * self.scale * self.gamma

    def load_weights(self, state_dicts: list) -> None:
        for state_dict in state_dicts:
            if state_dict.has("gamma"):
                self.gamma.data.copy_(state_dict.get_tensor("gamma"))
                return
        raise RuntimeError(f"weights are not loaded for {type(self).__name__}: ['gamma']")


class FP32Upsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.to(torch.float32),
            scale_factor=(1.0, 2.0, 2.0),
            mode="nearest-exact",
        ).to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vae_group: str | None,
    ) -> None:
        super().__init__()
        self.norm1 = XVAERMSNorm(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group)
        self.norm2 = XVAERMSNorm(out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), vae_group)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(_swish(self.norm2(h)))
        if self.nin_shortcut is not None:
            shortcut = self.nin_shortcut(shortcut)
        return h + shortcut

    def load_weights(self, state_dicts: list) -> None:
        self.norm1.load_weights(prefixed(state_dicts, "norm1."))
        self.conv1.load_weights(prefixed(state_dicts, "conv1."))
        self.norm2.load_weights(prefixed(state_dicts, "norm2."))
        self.conv2.load_weights(prefixed(state_dicts, "conv2."))
        if self.nin_shortcut is not None:
            self.nin_shortcut.load_weights(prefixed(state_dicts, "nin_shortcut."))


class AttnBlock(nn.Module, VaeParallelMixin):
    """Per-(b,t) self-attention over the (h*w) axis; K/V gathered globally."""

    def __init__(self, in_channels: int, vae_group: str | None) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.in_channels = in_channels
        self.norm = XVAERMSNorm(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        self.k = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        self.v = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1, bias=True)

    def _to_bt_hw_c(self, y: torch.Tensor, width: int) -> torch.Tensor:
        b, c, t, h = y.size(0), y.size(1), y.size(2), y.size(3)
        # b c t h w -> (b t) 1 (h w) c
        return y.permute(0, 2, 3, 4, 1).contiguous().view(b * t, h * width, c).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"AttnBlock channel mismatch: {c} vs {self.in_channels}")
        residual = x
        h_norm = self.norm(x)

        q = self._to_bt_hw_c(self.q(h_norm), w)
        k_sp = self.k(h_norm)
        v_sp = self.v(h_norm)
        if self.vae_parallel_enabled(x.device):
            k_sp = self.merge(k_sp)
            v_sp = self.merge(v_sp)
        w_k = k_sp.size(-1)
        k = self._to_bt_hw_c(k_sp, w_k)
        v = self._to_bt_hw_c(v_sp, w_k)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.squeeze(1).view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        attn = self.proj_out(attn)
        return residual + attn

    def load_weights(self, state_dicts: list) -> None:
        self.norm.load_weights(prefixed(state_dicts, "norm."))
        loaded = {f"{name}.{part}": False for name in ("q", "k", "v", "proj_out") for part in ("weight", "bias")}
        for state_dict in state_dicts:
            for name in ("q", "k", "v", "proj_out"):
                module = getattr(self, name)
                for part in ("weight", "bias"):
                    key = f"{name}.{part}"
                    if not loaded[key] and state_dict.has(key):
                        getattr(module, part).data.copy_(state_dict.get_tensor(key))
                        loaded[key] = True
        require_loaded(loaded, type(self).__name__)


class UpsampleBlock(nn.Module, VaeParallelMixin):
    """2x spatial (optionally 2x temporal) upsample with factor-major shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_upsample: bool,
        vae_group: str | None,
    ) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.temporal = CausalConv3d(
                in_channels,
                in_channels * 2,
                (3, 1, 1),
                (1, 1, 1),
                (1, 0, 0),
                vae_group,
            )
        else:
            self.temporal = None
        self.spatial = nn.Sequential(
            FP32Upsample(),
            nn.Conv3d(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1), bias=True),
        )
        # (8 if temporal else 4) * out / in
        factor = 8 if temporal_upsample else 4
        if (factor * out_channels) % in_channels:
            raise ValueError(f"invalid upsample channel ratio in={in_channels} out={out_channels}")
        self.repeats = (factor * out_channels) // in_channels

    def shortcut(self, x: torch.Tensor, first_chunk: bool = True) -> torch.Tensor:
        r1 = 2 if self.temporal_upsample else 1
        skip = 1 if (self.temporal_upsample and first_chunk) else 0
        b, _, f, h, w = x.shape
        y = x.repeat_interleave(self.repeats, dim=1)
        # JoyO XVAE shortcut layout is factor-major (unpatchify family):
        # b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3). Channel-major
        # (Wan DupUp3D) scrambling produces grid garbage.
        c = y.size(1) // (r1 * 2 * 2)
        y = y.view(b, r1, 2, 2, c, f, h, w)
        y = y.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
        y = y.view(b, c, f * r1, h * 2, w * 2)
        if skip > 0:
            if y.size(2) <= skip:
                raise ValueError(
                    f"XVAE UpsampleBlock shortcut: temporal dim {y.size(2)} cannot drop first {skip} frame(s)"
                )
            y = y[:, :, skip:]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal is None:
            return self._forward_spatial_sliced(x)
        if x.size(2) > TEMPORAL_SLICE_FRAMES:
            return self._forward_temporal_sliced(x)
        return self._forward_temporal_full(x)

    def _forward_temporal_full(self, x: torch.Tensor) -> torch.Tensor:
        # Full-sequence path: shortcut first_chunk=True always.
        sc = self.shortcut(x, first_chunk=True)

        hidden = x
        b, c, t, height, width = hidden.shape
        x0 = hidden[:, :, 0:1]
        if t > 1:
            x_t = self.temporal(hidden)[:, :, 1:t]
            # b (r c) t h w -> b c (t r) h w
            t2 = x_t.size(2)
            x_t = x_t.view(b, 2, c, t2, height, width)
            x_t = x_t.permute(0, 2, 3, 1, 4, 5).contiguous()
            x_t = x_t.view(b, c, t2 * 2, height, width)
            hidden = torch.cat([x0, x_t], dim=2)
        else:
            dummy = self.temporal(hidden)
            t_dummy = dummy.size(2)
            dummy = dummy.view(b, 2, c, t_dummy, height, width)
            dummy = dummy.permute(0, 2, 3, 1, 4, 5).contiguous()
            dummy = dummy.view(b, c, t_dummy * 2, height, width)
            dummy = dummy[:, :, 0:1]
            hidden = x0 + 0.0 * dummy

        hidden = self.exchange(hidden, pad=False)
        hidden = self.spatial(hidden)
        hidden = self.trim_halo(hidden, per_side=2)
        return hidden + sc

    def _forward_temporal_sliced(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal-upsample path sliced over the INPUT T with a 2-frame
        # causal context: the temporal conv is k3 causal (kept frames never
        # touch the zero pad), and the spatial section plus the shortcut are
        # frame-local, so this is exact vs _forward_temporal_full — the CPU
        # exactness test pins it. Caps the fp32 transients of the deepest
        # head scales (L2's interpolate output alone is ~3.5 GiB full-video).
        b, c, t, height, width = x.shape
        out_channels = self.spatial[1].out_channels
        out = torch.empty(b, out_channels, 2 * t - 1, height * 2, width * 2, dtype=x.dtype, device=x.device)
        ctx = 2
        p = 0
        while p < t:
            q = min(p + TEMPORAL_SLICE_FRAMES, t)
            lo = max(0, p - ctx) if p > 0 else 0
            seg = x[:, :, lo:q]
            if lo == 0:
                hidden = self._temporal_first(seg)
                sc = self.shortcut(seg, first_chunk=True)
                hid = self.exchange(hidden, pad=False)
                hid = self.spatial(hid)
                hid = self.trim_halo(hid, per_side=2)
                out[:, :, 0 : 2 * q - 1] = hid + sc
            else:
                # Context frames are real history (not the sequence start),
                # so the conv runs over the whole slice — no x0 special case —
                # and every slice frame yields two output frames, matching the
                # shortcut's uniform interleave (first_chunk=False).
                x_t = self.temporal(seg)
                ln = x_t.size(2)
                x_t = x_t.view(b, 2, c, ln, height, width)
                x_t = x_t.permute(0, 2, 3, 1, 4, 5).contiguous()
                hidden = x_t.view(b, c, ln * 2, height, width)
                sc = self.shortcut(seg, first_chunk=False)
                hid = self.exchange(hidden, pad=False)
                hid = self.spatial(hid)
                hid = self.trim_halo(hid, per_side=2)
                seg_out = hid + sc
                drop = 2 * (p - lo)
                out[:, :, 2 * p - 1 : 2 * q - 1] = seg_out[:, :, drop:]
            p = q
        return out

    def _temporal_first(self, seg: torch.Tensor) -> torch.Tensor:
        # The x0-special-cased temporal upsample for a slice that starts at
        # the true sequence head (verbatim from the full-sequence path).
        b, c, t, height, width = seg.shape
        x0 = seg[:, :, 0:1]
        if t > 1:
            x_t = self.temporal(seg)[:, :, 1:t]
            t2 = x_t.size(2)
            x_t = x_t.view(b, 2, c, t2, height, width)
            x_t = x_t.permute(0, 2, 3, 1, 4, 5).contiguous()
            x_t = x_t.view(b, c, t2 * 2, height, width)
            return torch.cat([x0, x_t], dim=2)
        dummy = self.temporal(seg)
        t_dummy = dummy.size(2)
        dummy = dummy.view(b, 2, c, t_dummy, height, width)
        dummy = dummy.permute(0, 2, 3, 1, 4, 5).contiguous()
        dummy = dummy.view(b, c, t_dummy * 2, height, width)
        dummy = dummy[:, :, 0:1]
        return x0 + 0.0 * dummy

    def _forward_spatial_sliced(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial-only upsample (temporal_upsample=False): every op here is
        # frame-local — interpolate (1,2,2), conv (1,3,3), shortcut repeat/
        # reshape — so slice over T to cap the fp32 transients. The
        # full-video staging of the shortcut alone is ~4x the block output
        # (repeat_interleave on channels), which is what OOMs the decode.
        b, _, t, h, w = x.shape
        out_channels = self.spatial[1].out_channels
        out = torch.empty(b, out_channels, t, h * 2, w * 2, dtype=x.dtype, device=x.device)
        for s0 in range(0, t, SPATIAL_SLICE_FRAMES):
            s1 = min(s0 + SPATIAL_SLICE_FRAMES, t)
            seg = x[:, :, s0:s1]
            sc = self.shortcut(seg, first_chunk=True)
            hid = self.exchange(seg, pad=False)
            hid = self.spatial(hid)
            hid = self.trim_halo(hid, per_side=2)
            out[:, :, s0:s1] = hid + sc
        return out

    def load_weights(self, state_dicts: list) -> None:
        if self.temporal is not None:
            self.temporal.load_weights(prefixed(state_dicts, "temporal."))
        loaded = {"spatial.1.weight": False, "spatial.1.bias": False}
        for state_dict in state_dicts:
            for key in ("spatial.1.weight", "spatial.1.bias"):
                if not loaded[key] and state_dict.has(key):
                    param = self.spatial[1].weight if key.endswith("weight") else self.spatial[1].bias
                    param.data.copy_(state_dict.get_tensor(key))
                    loaded[key] = True
        require_loaded(loaded, type(self).__name__)


class XVAEDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_in_channels: list[int],
        temporal_upsample: list[bool],
        vae_group: str | None,
    ) -> None:
        super().__init__()
        self.block_in_channels = block_in_channels
        self.temporal_upsample = temporal_upsample
        if not block_in_channels:
            raise ValueError("XVAEDecoder needs block_in_channels")
        if len(temporal_upsample) != len(block_in_channels):
            raise ValueError("temporal_upsample length mismatch")

        block_in0 = block_in_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in0, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group)

        blocks: list[nn.Module] = [
            ResidualBlock(block_in0, block_in0, vae_group),
            AttnBlock(block_in0, vae_group),
            ResidualBlock(block_in0, block_in0, vae_group),
        ]
        self.mid_blocks = nn.ModuleList(blocks)  # [res, attn, res]

        up: list[nn.Module] = []
        up_is_upsample: list[bool] = []
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(num_res_blocks + 1):
                up.append(ResidualBlock(block_in, block_in, vae_group))
                up_is_upsample.append(False)
            if i_level + 1 < len(block_in_channels):
                out_ch = block_in_channels[i_level + 1]
                up.append(UpsampleBlock(block_in, out_ch, temporal_upsample[i_level], vae_group))
                up_is_upsample.append(True)
        self.up_blocks = nn.ModuleList(up)
        if len(up_is_upsample) != len(up):
            raise RuntimeError("upsample bookkeeping mismatch")

        # Head/tail split for sliced decoding: everything AFTER the last
        # temporal-upsample block is T-preserving with a bounded causal
        # receptive field, so it can run in temporal slices with a small
        # context prefix (frames whose outputs are recomputed and dropped).
        self.tail_start = None
        for i, block in enumerate(up):
            if isinstance(block, UpsampleBlock) and block.temporal_upsample:
                self.tail_start = i + 1
        if self.tail_start is not None:
            tail_rf = 0
            for block in up[self.tail_start :]:
                if isinstance(block, ResidualBlock):
                    tail_rf += 4  # conv1 + conv2, k3 causal each
            tail_rf += 2  # conv_out (k3 causal)
            self.tail_receptive_field = tail_rf

        block_in_last = block_in_channels[-1]
        self.norm_out = XVAERMSNorm(block_in_last)
        self.conv_out = CausalConv3d(block_in_last, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        # ResidualBlock / AttnBlock / UpsampleBlock all take (x) and return
        # the same-shape-family tensor, so no per-block type dispatch needed.
        for block in self.mid_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)
        x = _swish(self.norm_out(x))
        return self.conv_out(x)

    def forward_head(self, z: torch.Tensor) -> torch.Tensor:
        """conv_in → mid → up_blocks[:tail_start] (ends at the last temporal
        upsample). Must run full-sequence: the head's causal receptive field
        spans the whole latent sequence."""
        x = self.conv_in(z)
        for block in self.mid_blocks:
            x = block(x)
        for block in self.up_blocks[: self.tail_start]:
            x = block(x)
        return x

    def forward_tail(self, h: torch.Tensor) -> torch.Tensor:
        """up_blocks[tail_start:] → norm → conv_out. T-preserving; exact for a
        slice whose first tail_receptive_field frames are context (outputs
        for the context frames are recomputed and dropped by the caller)."""
        x = h
        for block in self.up_blocks[self.tail_start :]:
            x = block(x)
        x = _swish(self.norm_out(x))
        return self.conv_out(x)

    def load_weights(self, state_dicts: list) -> None:
        self.conv_in.load_weights(prefixed(state_dicts, "conv_in."))
        for i, block in enumerate(self.mid_blocks):
            block.load_weights(prefixed(state_dicts, f"mid_blocks.{i}."))
        for i, block in enumerate(self.up_blocks):
            block.load_weights(prefixed(state_dicts, f"up_blocks.{i}."))
        self.norm_out.load_weights(prefixed(state_dicts, "norm_out."))
        self.conv_out.load_weights(prefixed(state_dicts, "conv_out."))


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    b, c_in, t, h, w = x.shape
    r1 = r2 = patch_size
    if c_in % (r1 * r2):
        raise ValueError(f"unpatchify channels {c_in} not divisible by {r1 * r2}")
    c = c_in // (r1 * r2)
    # b (r1 r2 c) t h w -> b c t (h r1) (w r2)
    y = x.view(b, r1, r2, c, t, h, w)
    y = y.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
    return y.view(b, c, t, h * r1, w * r2)


class AutoencoderKLXVAE(DiTComponentBase, VaeParallelMixin):
    """JoyOV2 video VAE: decode [B, C, T, H, W] latents to [B, 3, T', H', W']."""

    def __init__(
        self,
        config: dict,
        vae_group: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # VAE checkpoint and Conv3d weights are fp32; the pipeline-wide DiT
        # dtype is deliberately ignored (C++ parity: options_.dtype(kFloat)).
        del dtype
        DiTComponentBase.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)

        self.latent_channels = int(config.get("latent_channels", 128))
        self.out_channels = int(config.get("out_channels", 3))
        self.scale_factor_spatial = int(
            config.get("scale_factor_spatial", 0) or config.get("vae_scale_factor_spatial", 0) or 32
        )
        self.scale_factor_temporal = int(
            config.get("scale_factor_temporal", 0) or config.get("vae_scale_factor_temporal", 0) or 4
        )
        self.patch_size = int(config.get("patch_size", 0) or config.get("vae_patch_size", 0) or 2)
        self.num_res_blocks = int(config.get("num_res_blocks", 2))
        # The checkpoint stores null for unset optionals (diffusers
        # convention); ``or []`` mirrors the C++ empty-vector fallback.
        block_in_channels = [int(v) for v in (config.get("block_in_channels") or [])]
        if not block_in_channels:
            # Checkpoint stores null → SGLang/diffusers defaults.
            block_in_channels = [160, 320, 640, 1280, 1280]
        temporal_upsample = [bool(v) for v in (config.get("temporal_downsample") or [])]
        if not temporal_upsample:
            temporal_upsample = [False, True, True, False, False]
        if len(temporal_upsample) != len(block_in_channels):
            raise ValueError("temporal_downsample length mismatch")
        self.latents_mean = [float(v) for v in (config.get("latents_mean") or [])]
        self.latents_std = [float(v) for v in (config.get("latents_std") or [])]

        # Decoder consumes reversed block channels (config order = encoder).
        decoder_block_in_channels = list(reversed(block_in_channels))
        decoder_out_channels = self.out_channels * self.patch_size * self.patch_size
        self.decoder = XVAEDecoder(
            self.latent_channels,
            decoder_out_channels,
            self.num_res_blocks,
            decoder_block_in_channels,
            temporal_upsample,
            vae_group,
        )
        self.register_buffer("latents_mean_buf", None)
        self.register_buffer("latents_inv_std_buf", None)

    def keep_fp32_modules(self) -> None:
        self.float()

    def load_weights(self, state_dicts: list) -> None:
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
        mean = torch.tensor(self.latents_mean, dtype=torch.float32, device=latents.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.latents_std, dtype=torch.float32, device=latents.device).view(1, -1, 1, 1, 1)
        return latents.to(torch.float32) * std + mean

    def decode(self, latent: torch.Tensor, slice_frames: int = 16) -> torch.Tensor:
        if latent.dim() != 5:
            raise ValueError(f"XVAE decode expects [B,C,T,H,W], got {latent.shape}")
        if latent.size(1) != self.latent_channels:
            raise ValueError(f"XVAE decode channel mismatch: {latent.size(1)} vs {self.latent_channels}")
        # fp32 to match Conv3d weight dtype; DiT bf16 latents would fall back
        # to CPU aten kernels.
        z = latent.to(torch.float32)
        z = self.split(z)
        decoder = self.decoder
        if decoder.tail_start is None:
            decoded = decoder(z)
        else:
            # The head (through the last temporal upsample) must run
            # full-sequence; the tail is T-preserving with a bounded causal
            # receptive field, so it runs in temporal slices with a context
            # prefix (context outputs are recomputed and dropped). This keeps
            # the deepest-scale fp32 transients to a slice instead of the
            # whole video.
            h = decoder.forward_head(z)
            t = h.size(2)
            ctx = decoder.tail_receptive_field
            if t <= slice_frames + ctx:
                decoded = decoder.forward_tail(h)
            else:
                outs = []
                p = 0
                while p < t:
                    q = min(p + slice_frames, t)
                    lo = max(0, p - ctx) if p > 0 else 0
                    seg = decoder.forward_tail(h[:, :, lo:q])
                    outs.append(seg if p == 0 else seg[:, :, p - lo :])
                    p = q
                decoded = torch.cat(outs, dim=2)
        decoded = self.merge(decoded)
        return unpatchify(decoded, self.patch_size)
