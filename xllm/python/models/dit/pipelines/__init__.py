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

"""Pipeline orchestration layer of the DiT component package.

Pipelines compose the sibling DiT components (``transformers/``,
``autoencoders/``, ``schedulers/``, ...) into a full denoising loop. The
registry maps the C++-resolved pipeline type (the ``model_type`` string the
C++ loader reports, e.g. ``JoyOV2Pipeline``) to its Python class, so
``DiTPipelineExecutor`` can build a pipeline from the config dict the C++
shell passes across the boundary.
"""

from __future__ import annotations

from importlib import import_module

_PIPELINES: dict[str, tuple[str, str]] = {
    "JoyOV2Pipeline": (
        "xllm.python.models.dit.pipelines.joyo_v2",
        "JoyOV2Pipeline",
    ),
}


def get_pipeline_class(name: str) -> type:
    """Resolve a pipeline class by the C++ pipeline (model) type string."""
    if name not in _PIPELINES:
        raise KeyError(f"DiT pipeline '{name}' not registered; available: {sorted(_PIPELINES)}")
    module_name, class_name = _PIPELINES[name]
    return getattr(import_module(module_name), class_name)
