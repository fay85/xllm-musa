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

"""DiT component layer: transformers, VAEs, encoders, schedulers, processors.

Components are organized like diffusers (``transformers/``, ``autoencoders/``,
``schedulers/``, ...). The registry below maps a component config's
``_class_name`` to its Python class so the pipeline executor can assemble a
pipeline from per-component config dicts, mirroring diffusers'
``DiffusionPipeline.from_pretrained`` component resolution.

Dependency rule: this package may import ``xllm.python.layers``,
``xllm.python.parallel`` and ``xllm.python.distributed.collectives`` but
never drives communication itself.
"""

from __future__ import annotations

from importlib import import_module

_COMPONENTS: dict[str, tuple[str, str]] = {
    "JoyOV2Transformer3DModel": (
        "xllm.python.models.dit.transformers.joyo_v2",
        "JoyOV2Transformer3DModel",
    ),
    "AutoencoderKLXVAE": (
        "xllm.python.models.dit.autoencoders.autoencoder_kl_xvae",
        "AutoencoderKLXVAE",
    ),
    "AutoencoderKLXDAC": (
        "xllm.python.models.dit.autoencoders.autoencoder_kl_xdac",
        "AutoencoderKLXDAC",
    ),
    "FlowMatchEulerDiscreteScheduler": (
        "xllm.python.models.dit.schedulers.flowmatch_euler",
        "FlowMatchEulerDiscreteScheduler",
    ),
    "VAEImageProcessor": (
        "xllm.python.models.dit.processors.vae_image_processor",
        "VAEImageProcessor",
    ),
}


def get_component_class(name: str) -> type:
    """Resolve a DiT component class by its config ``_class_name``."""
    if name not in _COMPONENTS:
        raise KeyError(f"DiT component '{name}' not registered; available: {sorted(_COMPONENTS)}")
    module_name, class_name = _COMPONENTS[name]
    return getattr(import_module(module_name), class_name)
