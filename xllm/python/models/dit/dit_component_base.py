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

"""Shared base for DiT pipeline components.

A component builds its module graph from its diffusers-style config dict and
fills parameters in ``load_weights`` from the C++ ``StateDict`` wrappers
(``xllm_weight_loader.StateDict`` — the same objects LLM models consume).
Components that need per-component state not expressible in config.json take
it through constructor arguments from the pipeline executor instead of
reaching into the C++ layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class DiTComponentBase(nn.Module):
    """Base class for DiT components the Python pipeline executor drives."""

    def load_weights(self, state_dicts: list[StateDict]) -> None:
        """Load checkpoint weights into this component's parameters.

        The component owns ALL weight transform logic: key mapping, fused-
        weight splitting, dtype conversion, etc. Missing keys are an error
        (the C++ ports fail loudly via ``verify_loaded_weights``).
        """
        raise NotImplementedError


class PrefixedStateDict:
    """A StateDict view narrowed to ``prefix`` (get_dict_with_prefix twin)."""

    def __init__(self, inner: StateDict, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix

    def has(self, name: str) -> bool:
        return self._inner.has(self._prefix + name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._inner.get_tensor(self._prefix + name)


def prefixed(state_dicts: list[StateDict], prefix: str) -> list[PrefixedStateDict]:
    """Narrow every state dict to ``prefix`` before handing it to a submodule."""
    return [PrefixedStateDict(state_dict, prefix) for state_dict in state_dicts]


def require_loaded(loaded: dict[str, bool], owner: str) -> None:
    """Raise when a tracked weight never arrived (mirrors C++ CHECK)."""
    missing = sorted(name for name, is_loaded in loaded.items() if not is_loaded)
    if missing:
        raise RuntimeError(f"weights are not loaded for {owner}: {missing}")
