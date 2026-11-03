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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.model_executor.forward_context import EplbRuntimeState, LayerLoadContext, LayerSynchronizer
    from xllm.python.model_executor.input_batch import InputBatch
    from xllm.python.models.multimodal import MultimodalInputs


@dataclass
class ModelInput:
    """One stage's inputs, shared by orchestration and graph admission."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: AttentionMetadata
    input_embedding: torch.Tensor | None = None
    layer_synchronizer: LayerSynchronizer | None = None
    eplb: EplbRuntimeState | None = None
    input_batch: InputBatch | None = None
    layer_load_context: LayerLoadContext | None = None
    multimodal_inputs: MultimodalInputs | None = None
