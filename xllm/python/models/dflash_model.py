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
from typing import TYPE_CHECKING, Any, cast
from weakref import ref

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.execution_context import can_execute_graph
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.sampling_metadata import (
    ModelSamplingMetadata,
    ModelSamplingMetadataBuilder,
    SamplingMode,
    prepare_sampling_input,
    sample_logits,
)

if TYPE_CHECKING:
    from xllm.python.models.qwen3_dflash import DFlashQwen3ForCausalLM


@dataclass(frozen=True)
class DFlashDecoderMetadata:
    num_requests: int


class DFlashDecoderMetadataBuilder:
    """Capture complete query blocks even when warmup does not request sampling."""

    metadata_type: type[object] = DFlashDecoderMetadata

    def __init__(self, query_width: int) -> None:
        if query_width <= 1:
            raise ValueError("DFlash query width must include an anchor and proposal tokens")
        self._query_width = query_width

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DFlashDecoderMetadata:
        requests = 0 if input_batch.is_dummy or input_batch.sampling is None else input_batch.num_reqs
        return DFlashDecoderMetadata(requests)

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DFlashDecoderMetadata:
        requests, remainder = divmod(input_batch.num_tokens_after_padding, self._query_width)
        return DFlashDecoderMetadata(0 if remainder else requests)

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: AttentionMetadata
    ) -> None:
        return None


class DFlashDecoderModel(nn.Module):
    def __init__(self, model: DFlashQwen3ForCausalLM, query_width: int) -> None:
        super().__init__()
        self._model_ref = ref(model)
        self._query_width = query_width

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(cast("DFlashQwen3ForCausalLM", self._model_ref()).model, name)

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        model = cast("DFlashQwen3ForCausalLM", self._model_ref())
        return (
            bool(getattr(inputs.metadata, "use_dflash_proposal_attention", False))
            and inputs.input_batch is not None
            and model.lm_head is not None
            and can_execute_graph(model.model, inputs)
            and can_execute_graph(model.lm_head, inputs)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = cast("DFlashQwen3ForCausalLM", self._model_ref())
        hidden = model.model(input_ids, positions, input_embedding)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("DFlash backbone must return hidden states")
        decoder = get_execution_context(DFlashDecoderMetadata)
        if decoder is None or decoder.num_requests == 0:
            return hidden.new_empty((0, model.cfg.vocab_size)), hidden
        proposal_hidden = hidden.view(decoder.num_requests, self._query_width, -1)[:, 1:].reshape(
            decoder.num_requests * (self._query_width - 1), -1
        )
        return model.compute_logits(proposal_hidden, None), hidden


class DFlashProposalModel(nn.Module):
    def __init__(self, builder: ModelSamplingMetadataBuilder, num_steps: int) -> None:
        super().__init__()
        self._builder = builder
        self._num_steps = num_steps

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        return inputs.input_batch is not None and self._builder.can_execute_graph(inputs.input_batch.sampling)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> ModelOutput:
        metadata = get_execution_context(ModelSamplingMetadata)
        if metadata is None or metadata.sampling is None or input_embedding is None:
            raise RuntimeError("DFlash proposal requires logits and sampling metadata")
        if metadata.sampling.parameters.sampling_mode != SamplingMode.GREEDY or metadata.sampling.return_draft_logits:
            raise ValueError("DFlash draft sampling must remain greedy without draft probabilities")
        sampling = prepare_sampling_input(metadata.sampling, input_embedding)
        tokens, _ = sample_logits(input_embedding, sampling, tp_size=metadata.tp_size)
        return ModelOutput(draft_token_ids=tokens.view(-1, self._num_steps))
