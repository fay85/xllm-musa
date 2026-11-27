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

"""Python pipeline owner driven directly by the C++ PyDiTExecutor."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xllm_dit_runtime import DiTForwardInput, DiTForwardOutput

import torch

from xllm.python.models.dit import get_component_class
from xllm.python.models.dit.pipelines import get_pipeline_class

# Components constructed from their config dict; the executor passes the
# component dtype and the matching parallel group name.
_PARALLEL_GROUP_KEY = {
    "transformer": "dit_sp_group",
    "vae": "dit_vae_group",
    # The audio VAE reuses the image-VAE group slot; the XDAC decoder has no
    # spatial-parallel path, so it accepts and ignores the group.
    "audio_vae": "dit_vae_group",
}
_NEEDS_DTYPE = frozenset(("transformer", "vae", "audio_vae", "text_encoder"))
# The JoyOV2 transformer resolves its TP shard shapes at construction time
# (200B weights only fit sharded), so it needs the device to look up the
# ``dit_tp`` named group before any tensor exists.
_NEEDS_DEVICE = frozenset(("transformer",))


class DiTPipelineExecutor:
    """Assemble, load and run a Python DiT pipeline."""

    def __init__(self, config: dict) -> None:
        if config.get("dit_cache_policy") not in (None, "", "None"):
            raise ValueError("The Python Joyo V2 backend does not support dit_cache_policy; use None.")
        self.config = config
        self.device = torch.device(config["device"])
        self.dtype = self._resolve_dtype(config.get("dtype"))
        # Publish the resolved values so components and the pipeline read the
        # same device/dtype without re-parsing the C++ strings.
        config["device"] = self.device
        config["dtype"] = self.dtype

        specs = config["components"]
        self.component_dtypes = {
            name: self._resolve_component_dtype(spec) for name, spec in specs.items() if name in _NEEDS_DTYPE
        }
        # Latents and timestep inputs follow the transformer, not the VAE.
        self.dtype = self.component_dtypes.get("transformer", self.dtype)
        config["dtype"] = self.dtype
        # Pipeline-level views of the component configs (latent stats, patch
        # size, ...) — the pipeline reads these instead of reaching into
        # component instances.
        config.setdefault("transformer_config", specs["transformer"]["config"])
        config.setdefault("vae_config", specs["vae"]["config"])
        if "audio_vae" in specs:
            config.setdefault("audio_vae_config", specs["audio_vae"]["config"])

        components: dict[str, object] = {}
        for name, spec in specs.items():
            components[name] = self._build_component(name, spec)
        self._build_image_processor(components, specs)

        self.pipeline = get_pipeline_class(config["pipeline"])(components, config)
        self.pipeline.to(
            device=self.device,
            dtype=self.dtype,
            component_dtypes=self.component_dtypes,
        )
        # Component conventions applied after the move, mirroring the C++
        # pipelines' load_model: fp32-sensitive modules are re-cast.
        for component in components.values():
            keep_fp32 = getattr(component, "keep_fp32_modules", None)
            if callable(keep_fp32):
                keep_fp32()
        # inference_mode disables autograd, but does not switch module behavior.
        self.pipeline.eval()
        if "text_encoder" in components:
            components["text_encoder"].set_on_device(False)

    @classmethod
    def create(cls, config: dict) -> DiTPipelineExecutor:
        return cls(config)

    @torch.inference_mode()
    def execute(self, request: DiTForwardInput) -> DiTForwardOutput:
        """Pass one bound C++ request to the pipeline without a dict conversion."""
        return self.pipeline(request)

    # ---- internals ----------------------------------------------------------

    def _resolve_component_dtype(self, spec: dict) -> torch.dtype:
        """Resolve component precision before construction and weight loading."""
        class_name = spec["class_name"]
        # Both decoders require FP32 weights throughout loading and execution.
        # Casting back after a pipeline-wide BF16 move loses weight precision.
        if class_name in ("AutoencoderKLXVAE", "AutoencoderKLXDAC"):
            return torch.float32
        component_config = spec["config"]
        if class_name == "JoyoTextEncoder":
            text = component_config.get("text_config", component_config)
            configured = (
                text.get("dtype")
                or text.get("torch_dtype")
                or component_config.get("dtype")
                or component_config.get("torch_dtype")
            )
            return self._resolve_dtype(configured or "bfloat16")
        configured = component_config.get("dtype") or component_config.get("torch_dtype")
        if configured:
            return self._resolve_dtype(configured)
        if class_name == "JoyOV2Transformer3DModel":
            # Match the native component default when config.json omits dtype.
            return torch.bfloat16
        return self.dtype

    def _build_component(self, name: str, spec: dict) -> object:
        if name == "text_encoder":
            from xllm.python.models.dit.encoders.joyo import JoyoTextEncoder

            encoder = JoyoTextEncoder(
                spec["config"],
                spec["tokenizer"],
                self.device,
                self.component_dtypes[name],
                self.config["dit_text_encoder_tp_group"],
            )
            encoder.load_weights(spec["state_dicts"])
            return encoder

        component_cls = get_component_class(spec["class_name"])
        kwargs: dict = {}
        if name in _NEEDS_DTYPE:
            kwargs["dtype"] = self.component_dtypes[name]
        if name in _NEEDS_DEVICE and "device" in inspect.signature(component_cls.__init__).parameters:
            # Components that shard at construction time look up their named
            # group by device before any component tensor exists.
            kwargs["device"] = self.device
        group_key = _PARALLEL_GROUP_KEY.get(name)
        if group_key is not None and group_key in self.config:
            kwargs["sp_group" if name == "transformer" else "vae_group"] = self.config[group_key]
        component = component_cls(spec["config"], **kwargs)

        state_dicts = spec.get("state_dicts")
        if state_dicts is not None:
            # Call even on an empty list: components verify required weights
            # and fail loudly instead of silently running on random ones.
            component.load_weights(state_dicts)
        return component

    def _build_image_processor(self, components: dict[str, object], specs: dict) -> None:
        """Build the VAE image processor from the VAE component's config.

        The processor is not a model folder, so the C++ shell does not pass a
        component spec for it; the pipelines still require it for VAE reference
        pre-processing and output post-processing.
        """
        if "vae_image_processor" in components or "vae" not in specs:
            return
        vae_config = specs["vae"]["config"]
        processor_cls = get_component_class("VAEImageProcessor")
        components["vae_image_processor"] = processor_cls(
            vae_config,
            latent_channels=int(vae_config.get("z_dim", 4)),
            scale_factor=int(vae_config.get("scale_factor_spatial", 8)),
        )

    @staticmethod
    def _resolve_dtype(dtype: object) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        name = dtype if dtype else "bfloat16"
        resolved = getattr(torch, str(name), None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return resolved
