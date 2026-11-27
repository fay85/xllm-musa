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

"""Diffusion pipeline base.

A pipeline OWNS and composes components (transformer, vae, scheduler, text
encoder, image processor) assembled by the pipeline executor; dependency flow
is strictly ``models/dit/pipelines -> models/dit components -> parallel``. The
base provides the component bookkeeping (registration, device/dtype movement)
that mirrors diffusers' ``DiffusionPipeline``, while the denoising loop lives
entirely in each concrete pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xllm_dit_runtime import DiTForwardInput, DiTForwardOutput

import torch
import torch.nn as nn


class DiffusionPipeline(nn.Module):
    """Base class for Python DiT pipelines driven by ``DiTPipelineExecutor``.

    Subclass contract: accept the bound ``DiTForwardInput`` and return a
    ``DiTForwardOutput`` containing video/image and optional audio tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.components: dict[str, Any] = {}

    def register_modules(self, **components: Any) -> None:
        """Store components by name and as submodules (or plain attributes).

        Non-``nn.Module`` components (schedulers and image processors) become plain attributes so they are not swept into
        ``parameters()``/``to()``.
        """
        for name, component in components.items():
            self.components[name] = component
            if isinstance(component, nn.Module):
                self.add_module(name, component)
            else:
                setattr(self, name, component)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        component_dtypes: dict[str, torch.dtype] | None = None,
    ) -> DiffusionPipeline:
        """Move every registered module to ``device``/``dtype``.

        Mirrors the C++ pipelines' explicit ``to(options_)`` calls after
        construction. Per-component dtypes override the shared dtype without
        an intermediate cast. Non-module components are left alone; they receive the
        device/dtype through their own configuration.
        """
        component_dtypes = component_dtypes or {}
        for name, component in self.components.items():
            if isinstance(component, nn.Module):
                component.to(device=device, dtype=component_dtypes.get(name, dtype))
        return self

    @property
    def device(self) -> torch.device:
        for component in self.components.values():
            if isinstance(component, nn.Module):
                return next(component.parameters()).device
        raise RuntimeError("pipeline has no nn.Module components; cannot infer device")

    @property
    def dtype(self) -> torch.dtype:
        for component in self.components.values():
            if isinstance(component, nn.Module):
                return next(component.parameters()).dtype
        raise RuntimeError("pipeline has no nn.Module components; cannot infer dtype")

    def __call__(self, request: DiTForwardInput) -> DiTForwardOutput:
        raise NotImplementedError
