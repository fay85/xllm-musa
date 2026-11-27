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

"""JoyOV2Pipeline — Text-to-Video+Audio (T2VA) DiT pipeline.

Port of ``xllm/models/dit/pipelines/pipeline_joyo_v2.h``. One request runs
three refiner streams (text / pixel noise / audio) and the packed mix MoE
layers per denoise step, then decodes video through XVAE and audio through
XDAC.

Text path: requests supply ``prompt_embeds`` (``[T, D]`` or ``[1, T, D]``)
or plain ``prompts`` — when the pipeline has a ``text_encoder`` component,
the Python-owned Qwen3-VL text backbone encodes them via
``encode_joyo_prompts``
(bare user-turn chat wrap, full pre-norm residual, unpadded). Without
``negative_prompt_embeds`` or ``negative_prompts``, a CFG request splits
the packed ``prompt_embeds`` into ``n_cfg`` equal parts (H3 convention).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xllm_dit_runtime import DiTForwardInput, DiTForwardOutput, DiTGenerationParams

import math

import torch

from xllm.python.models.dit.pipelines.diffusion_pipeline_base import DiffusionPipeline
from xllm.python.models.dit.transformers.joyo_v2 import PackedRopeMeta, RopeMeta

# CFG Zero-Star defaults, matching the C++ pipeline.
_ZERO_CFG_STAR_STEP = 0
_ZERO_INIT_STEPS = 0


def sd3_timeshift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """SD3-style time-shift: sigma(t) = 1 - (shift*t) / (1 + (shift-1)*t)."""
    return 1.0 - (shift * t) / (1.0 + (shift - 1.0) * t)


def compute_audio_latent_length(sampling_rate: int, hop_length: int, num_frames: int, fps: float) -> int:
    audio_num_samples = sampling_rate * num_frames / fps
    return int(math.ceil(audio_num_samples / hop_length))


def build_position_ids(
    t: int,
    h: int,
    w: int,
    device: torch.device,
    pre_len: int = 0,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """[3, t*h*w] MRoPE ids: T rows offset by ``pre_len``, H/W from 0."""
    opts = {"dtype": dtype, "device": device}
    t_ids = torch.arange(pre_len, pre_len + t, **opts).unsqueeze(1).unsqueeze(2).expand(t, h, w)
    h_ids = torch.arange(h, **opts).unsqueeze(0).unsqueeze(2).expand(t, h, w)
    w_ids = torch.arange(w, **opts).unsqueeze(0).unsqueeze(1).expand(t, h, w)
    return torch.stack([t_ids.flatten(), h_ids.flatten(), w_ids.flatten()], dim=0)


def build_rope_meta(
    text_seqlens: torch.Tensor,  # [n_samples]
    latent_t: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    audio_seqlens: torch.Tensor | None = None,
    num_real_samples: int | None = None,
) -> PackedRopeMeta:
    """Rope metadata for the packed T2VA forward (C++ build_rope_meta twin)."""
    n_samples = int(text_seqlens.size(0))
    n_real = num_real_samples if num_real_samples is not None else n_samples
    pixel_seqlen = latent_t * latent_h * latent_w
    has_audio = audio_seqlens is not None
    text_lens = [int(v) for v in text_seqlens.tolist()]

    # Text: per-sample arange over [3, t_len].
    text_position_ids = torch.cat(
        [torch.arange(t_len, dtype=torch.int32, device=device).unsqueeze(0).expand(3, t_len) for t_len in text_lens],
        dim=1,
    )
    text_meta = RopeMeta(
        seqlens=text_seqlens.to(torch.int32).to(device),
        position_id=text_position_ids,
    )

    # Pixel position ids have NO text offset (Joytron PreCalRopeMeta /
    # SGLang); only the pixel slice inside mix is shifted by text_len.
    pixel_position_ids = torch.cat(
        [build_position_ids(latent_t, latent_h, latent_w, device) for _ in range(n_real)],
        dim=1,
    )
    pixel_meta = RopeMeta(
        seqlens=torch.full((n_real,), pixel_seqlen, dtype=torch.int32, device=device),
        position_id=pixel_position_ids,
    )

    audio_meta = None
    if has_audio:
        audio_lens = [int(v) for v in audio_seqlens.tolist()]
        audio_position_ids = torch.cat(
            [
                torch.arange(a_len, dtype=torch.int32, device=device).unsqueeze(0).expand(3, a_len)
                for a_len in audio_lens
            ],
            dim=1,
        )
        audio_meta = RopeMeta(
            seqlens=audio_seqlens.to(torch.int32).to(device),
            position_id=audio_position_ids,
        )

    # Mix: per real sample [text | pixel(pre_len=t_len) | audio]. Audio uses
    # fractional positions overlapping the pixel T-axis (Joytron
    # audio-fractional-rope), so the mix ids are float when audio is present.
    mix_pos_list = []
    mix_lens = []
    for i in range(n_real):
        t_len = text_lens[i]
        t_pos = torch.arange(t_len, dtype=torch.int32, device=device).unsqueeze(0).expand(3, t_len)
        p_pos = build_position_ids(latent_t, latent_h, latent_w, device, t_len)
        if has_audio:
            a_len = int(audio_seqlens[i])
            a_pos = (
                torch.linspace(
                    float(t_len),
                    float(t_len + latent_t - 1),
                    a_len,
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
                .expand(3, a_len)
            )
            mix_pos_list.append(
                torch.cat(
                    [
                        t_pos.to(torch.float32),
                        p_pos.to(torch.float32),
                        a_pos,
                    ],
                    dim=1,
                )
            )
            mix_lens.append(t_len + pixel_seqlen + a_len)
        else:
            mix_pos_list.append(torch.cat([t_pos, p_pos], dim=1))
            mix_lens.append(t_len + pixel_seqlen)
    mix_meta = RopeMeta(
        seqlens=torch.tensor(mix_lens, dtype=torch.int32, device=device),
        position_id=torch.cat(mix_pos_list, dim=1),
    )

    return PackedRopeMeta(text=text_meta, pixel=pixel_meta, audio=audio_meta, mix=mix_meta)


def denoise_update(
    hidden_states: torch.Tensor,
    noise_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_idx: int,
    guidance_scale: float,
    zero_cfg_star_step: int,
    zero_init_steps: int,
    do_cfg: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Joytron GaussianDenoiser step — CFG Zero-Star + Euler.

    The C++ original returns bf16 unconditionally; the Python port casts back
    to the pipeline dtype (bf16 in every real deployment, so parity holds).
    """
    dt = sigmas[step_idx + 1] - sigmas[step_idx]
    hidden_states = hidden_states.to(torch.float32)
    noise_pred = noise_pred.to(torch.float32)

    if do_cfg:
        hs_pos, _hs_neg = hidden_states.chunk(2, dim=0)
        np_pos, np_neg = noise_pred.chunk(2, dim=0)
        if zero_cfg_star_step >= 0:
            dot = torch.sum(np_neg * np_pos)
            sq = torch.sum(np_neg * np_neg) + 1e-8
            st_star = dot / sq
        else:
            st_star = torch.ones((), dtype=np_pos.dtype, device=np_pos.device)
        combined = np_neg * st_star + guidance_scale * (np_pos - np_neg * st_star)
        if step_idx >= zero_init_steps:
            hs_pos = hs_pos + combined * dt
        hidden_states = torch.cat([hs_pos, hs_pos], dim=0)
    elif step_idx >= zero_init_steps:
        hidden_states = hidden_states + noise_pred * dt
    return hidden_states.to(out_dtype)


class JoyOV2Pipeline(DiffusionPipeline):
    """T2VA pipeline: triple-stream DiT denoise + XVAE/XDAC decode."""

    def __init__(self, components: dict, config: dict) -> None:
        DiffusionPipeline.__init__(self)
        self.register_modules(**components)

        required = ("transformer", "vae")
        missing = [name for name in required if name not in self.components]
        if missing:
            raise KeyError(f"JoyOV2Pipeline missing components: {missing}")
        self.audio_vae = self.components.get("audio_vae")
        self.generate_audio = self.audio_vae is not None
        # Python-owned text backbone, absent when requests supply embeddings.
        self.text_encoder = self.components.get("text_encoder")

        self.latent_channels = int(self.transformer.latent_channels)
        self.text_dim = int(self.transformer.text_dim)
        if int(self.vae.latent_channels) != self.latent_channels:
            raise ValueError(
                f"JoyOV2 XVAE latent_channels ({self.vae.latent_channels}) "
                f"must match transformer ({self.latent_channels})"
            )
        if self.generate_audio and (int(self.audio_vae.latent_channels) != self.latent_channels):
            raise ValueError(
                f"JoyOV2 audio VAE latent_channels "
                f"({self.audio_vae.latent_channels}) must match transformer "
                f"({self.latent_channels})"
            )

        # nn.Module already exposes device/dtype properties; the configured
        # targets live under underscore names (executor resolves both).
        self._device = torch.device(config["device"])
        self._dtype = config["dtype"]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ---- entry point ---------------------------------------------------------

    def __call__(self, request: DiTForwardInput) -> DiTForwardOutput:
        gp = request.generation_params
        height = int(gp.height or 0) or 288
        width = int(gp.width or 0) or 512
        num_frames = int(gp.num_frames or 0) or 161
        num_steps = int(gp.num_inference_steps or 0) or 50
        guidance = self._resolve_guidance(gp)
        timeshift = float(gp.flow_shift or 0) or 8.0
        fps = float(gp.video_fps or 0) or 16.0
        seed = int(gp.seed or 0)
        if seed < 0:
            seed = 42
        do_cfg = guidance > 1.0
        n_cfg = 2 if do_cfg else 1
        batch_size = 1  # C++ parity: single-sample requests only.

        scale_s = int(self.vae.scale_factor_spatial or 32)
        scale_t = int(self.vae.scale_factor_temporal or 4)
        latent_h = height // scale_s
        latent_w = width // scale_s
        latent_t = (num_frames - 1) // scale_t + 1

        device = self._device
        dtype = self._dtype

        # ---- text embeds ----
        text_embeddings, text_seqlens = self._resolve_prompt_embeds(request, do_cfg, n_cfg, device)

        # ---- initial noise (private CPU generators; see _randn_tensor) ----
        latents = self._randn_tensor((batch_size, self.latent_channels, latent_t, latent_h, latent_w), seed)
        if do_cfg:
            latents = torch.cat([latents, latents], dim=0)
        num_samples = latents.size(0)
        hidden_states = latents.permute(0, 2, 3, 4, 1).reshape(-1, self.latent_channels).to(dtype)

        audio_hidden_states = None
        audio_latent_t = 0
        if self.generate_audio:
            sr = int(self.audio_vae.sampling_rate)
            gp_sr = int(gp.audio_sampling_rate or 0)
            if gp_sr > 0 and gp_sr != sr:
                raise ValueError(f"JoyOV2 audio_sampling_rate ({gp_sr}) must match the audio VAE config ({sr})")
            hop = int(self.audio_vae.hop_length)
            audio_latent_t = compute_audio_latent_length(sr, hop, num_frames, fps)
            # seed+1 keeps the audio stream independent of the video noise.
            audio_noise = self._randn_tensor((batch_size, audio_latent_t, self.latent_channels), seed + 1)
            if do_cfg:
                audio_noise = torch.cat([audio_noise, audio_noise], dim=0)
            audio_hidden_states = audio_noise.reshape(-1, self.latent_channels).to(dtype)
            audio_seqlens = torch.full((num_samples,), audio_latent_t, dtype=torch.int32, device=device)
        else:
            audio_seqlens = None

        rope_meta = build_rope_meta(
            text_seqlens,
            latent_t,
            latent_h,
            latent_w,
            device,
            audio_seqlens,
            num_real_samples=num_samples,
        )

        t_lin = torch.linspace(1, 0, num_steps + 1, device=device)
        sigmas = sd3_timeshift(t_lin, timeshift)
        # Match the native T2VA scheduler: audio uses an independent shift.
        audio_sigmas = sd3_timeshift(t_lin, 3.0) if self.generate_audio else None

        # ---- denoise ----
        for step_idx in range(num_steps):
            timestep = sigmas[step_idx].reshape(1).expand(num_samples)
            audio_timestep = audio_sigmas[step_idx].reshape(1).expand(num_samples) if audio_sigmas is not None else None
            noise_pred, audio_pred = self.transformer(
                hidden_states,
                text_embeddings,
                timestep,
                rope_meta,
                audio_hidden_states,
                audio_timestep=audio_timestep,
            )
            hidden_states = denoise_update(
                hidden_states,
                noise_pred,
                sigmas,
                step_idx,
                guidance,
                _ZERO_CFG_STAR_STEP,
                _ZERO_INIT_STEPS,
                do_cfg,
                dtype,
            )
            if audio_hidden_states is not None:
                audio_hidden_states = denoise_update(
                    audio_hidden_states,
                    audio_pred,
                    audio_sigmas,
                    step_idx,
                    guidance,
                    _ZERO_CFG_STAR_STEP,
                    _ZERO_INIT_STEPS,
                    do_cfg,
                    dtype,
                )

        if do_cfg:
            s_per = latent_t * latent_h * latent_w
            hidden_states = hidden_states[0 : batch_size * s_per]
            if audio_hidden_states is not None:
                audio_hidden_states = audio_hidden_states[0 : batch_size * audio_latent_t]

        # ---- decode ----
        # Keep the Transformer resident; only offload the audio VAE for video decode.
        if self.generate_audio:
            self.audio_vae.to("cpu")
        if device.type == "npu":
            torch.npu.empty_cache()
        video_latents = (
            hidden_states.to(torch.float32)
            .view(batch_size, latent_t, latent_h, latent_w, self.latent_channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        video_latents = self.vae.denormalize_latents(video_latents)
        video = self.vae.decode(video_latents).to(torch.float32)
        video = (video / 2.0 + 0.5).clamp(0.0, 1.0)
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        from xllm_dit_runtime import DiTForwardOutput

        result = DiTForwardOutput()
        result.tensors = [video.cpu()]
        if audio_hidden_states is not None:
            self.audio_vae.to(device)
            audio_latents = (
                audio_hidden_states.to(torch.float32)
                .view(batch_size, audio_latent_t, self.latent_channels)
                .permute(0, 2, 1)
                .contiguous()
            )
            audio_latents = self.audio_vae.denormalize_latents(audio_latents)
            audio = self.audio_vae.decode(audio_latents).to(torch.float32)
            result.audio_tensors = [audio.cpu()]
        return result

    # ---- text embeds ---------------------------------------------------------

    def _resolve_guidance(self, gp: DiTGenerationParams) -> float:
        # Match the native JoyOV2 pipeline; true_cfg_scale belongs to other pipelines.
        guidance = float(gp.guidance_scale or 0)
        return guidance if guidance > 0 else 4.0

    def _normalize_text_emb(self, t: torch.Tensor) -> torch.Tensor:
        # H3 uses [1, T, D]; the JoyO DiT consumes packed [T, D].
        if t.dim() == 3 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.dim() != 2:
            raise ValueError(f"JoyOV2 prompt_embed must be [T,D] or [1,T,D], got {tuple(t.shape)}")
        if t.size(1) != self.text_dim:
            raise ValueError(f"JoyOV2 prompt_embed last dim must be text_dim={self.text_dim}, got {t.size(1)}")
        if not t.is_floating_point():
            raise ValueError("JoyOV2 prompt_embed must be floating point")
        t = t.to(self._device, self._dtype)
        if not torch.isfinite(t).all():
            raise ValueError("JoyOV2 prompt_embed must not contain NaN or Inf")
        return t

    def _encode_prompts_in_process(
        self, request: DiTForwardInput, do_cfg: bool, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """In-process encoding via the Python-owned Qwen3-VL text backbone.

        Mirrors the C++ ``resolve_prompt_embeds`` in-process branch: one
        unpadded [T, D] pre-norm residual per prompt, seqlens derived per
        prompt (no padding, no prefix slice).
        """
        if self.text_encoder is None:
            raise ValueError(
                "JoyOV2 requires `prompt_embeds` (client-side encoded) or "
                "the in-process text encoder (--dit_enable_joyo_text_encoder "
                "+ text_encoder component)"
            )
        prompts = request.prompts or []
        if not prompts:
            raise ValueError("JoyOV2 in-process encode requires input prompts")
        # Mirror JoyOV2Pipeline: the text encoder rides the device only for
        # the encode window; offloading it right after (plus an allocator
        # flush) frees ~0.5 GiB per rank so the fp32 XVAE decode fits the
        # device budget.
        encoder = self.text_encoder
        encoder.set_on_device(True)
        try:
            residuals = encoder.encode_joyo_prompts([str(prompts[0])])
            if do_cfg:
                negative_prompts = request.negative_prompts or []
                if not negative_prompts:
                    raise ValueError("JoyOV2 CFG with in-process encoder requires negative_prompt")
                residuals.extend(encoder.encode_joyo_prompts([str(negative_prompts[0])]))
        finally:
            encoder.set_on_device(False)
        if device.type == "npu":
            torch.npu.empty_cache()
        text_embeddings = self._normalize_text_emb(residuals[0])
        if do_cfg:
            neg_emb = self._normalize_text_emb(residuals[1])
            text_seqlens = torch.tensor(
                [text_embeddings.size(0), neg_emb.size(0)],
                dtype=torch.int32,
                device=device,
            )
            text_embeddings = torch.cat([text_embeddings, neg_emb], dim=0)
        else:
            text_seqlens = torch.tensor([text_embeddings.size(0)], dtype=torch.int32, device=device)
        return text_embeddings, text_seqlens

    def _resolve_prompt_embeds(
        self, request: DiTForwardInput, do_cfg: bool, n_cfg: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = request.prompt_embeds
        if prompt_embeds is None:
            return self._encode_prompts_in_process(request, do_cfg, device)
        text_embeddings = self._normalize_text_emb(prompt_embeds)

        negative_prompt_embeds = request.negative_prompt_embeds
        if do_cfg and negative_prompt_embeds is not None:
            neg_emb = self._normalize_text_emb(negative_prompt_embeds)
            text_seqlens = torch.tensor(
                [text_embeddings.size(0), neg_emb.size(0)],
                dtype=torch.int32,
                device=device,
            )
            text_embeddings = torch.cat([text_embeddings, neg_emb], dim=0)
        else:
            # No negative embed and no seqlens side-channel (C++-only):
            # H3-style equal split of the packed prompt_embed.
            if text_embeddings.size(0) % n_cfg != 0:
                raise ValueError(
                    f"packed prompt_embed T ({text_embeddings.size(0)}) must "
                    f"divide n_cfg ({n_cfg}) without negative_prompt_embeds"
                )
            per = text_embeddings.size(0) // n_cfg
            text_seqlens = torch.full((n_cfg,), per, dtype=torch.int32, device=device)
        return text_embeddings, text_seqlens

    # ---- noise ---------------------------------------------------------------

    def _randn_tensor(self, shape: tuple[int, ...], seed: int) -> torch.Tensor:
        """Fresh CPU generator, randn in fp32, then move (C++ randn_tensor)."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(self._device)
