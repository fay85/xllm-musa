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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from weakref import ref

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder, can_execute_graph
from xllm.python.model_executor.execution_pipeline import ExecutionPipeline
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import DraftProposal, ModelOutput
from xllm.python.model_executor.runner import RunnerConfig
from xllm.python.model_executor.sampling_metadata import (
    ModelSamplingMetadata,
    ModelSamplingMetadataBuilder,
    prepare_sampling_input,
    sample_logits,
)
from xllm.python.models.base import PyModelBase

if TYPE_CHECKING:
    from xllm.python.models.dspark import DSparkConfidenceHead, DSparkMarkovHead


class DSparkModel(PyModelBase):
    """Own DSpark modules and expose their execution pipeline.

    Execution belongs to DSparkExecutionPipeline, not this weight container.
    Module ownership preserves checkpoint and shared-weight paths.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        lm_head: nn.Module | None,
        markov_head: DSparkMarkovHead,
        confidence_head: DSparkConfidenceHead | None,
        query_width: int,
        target_vocab_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lm_head = lm_head
        self.markov_head = markov_head
        self.confidence_head = confidence_head
        self._query_width = query_width
        self.target_vocab_size = target_vocab_size
        self.register_buffer("hot_token_ids", None, persistent=False)

    @property
    def query_width(self) -> int:
        return self._query_width

    def create_execution_pipeline(
        self,
        builders: Sequence[ExecutionMetadataBuilder],
        runner_config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
    ) -> ExecutionPipeline:
        from xllm.python.pipelines.dspark_pipeline import DSparkExecutionPipeline

        return DSparkExecutionPipeline(
            self, builders, runner_config, attention_backend=attention_backend, restore_history=restore_history
        )

    @property
    def cfg(self) -> Any:
        return self.backbone.cfg

    @property
    def dtype(self) -> torch.dtype:
        return self.markov_head.markov_w1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.markov_head.markov_w1.weight.device

    def get_token_embedding(self) -> nn.Module | None:
        return self.backbone.embed_tokens

    def share_weights_from(self, source: PyModelBase) -> None:
        if self.lm_head is None:
            self.lm_head = source.get_lm_head()
        if self.backbone.embed_tokens is None:
            self.backbone.embed_tokens = source.get_token_embedding()

    def confidence_probs(
        self,
        hidden: torch.Tensor,
        previous_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.confidence_head is None:
            raise RuntimeError("DSpark confidence head is not enabled")
        markov_embedding = self.markov_head.embed(previous_tokens) if previous_tokens is not None else None
        return self.confidence_head(hidden, markov_embedding)


@dataclass
class ProposalMetadata:
    """Optional backbone features consumed by the proposal confidence head."""

    hidden_states: torch.Tensor | None


class ProposalMetadataBuilder:
    metadata_type = ProposalMetadata

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> ProposalMetadata:
        return ProposalMetadata(input_batch.hidden_states)

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> ProposalMetadata:
        hidden = input_batch.hidden_states
        if hidden is None:
            return ProposalMetadata(None)
        return ProposalMetadata(hidden.new_zeros((input_batch.num_tokens_after_padding, *hidden.shape[1:])))

    def update_persistent(
        self, persistent_metadata: ProposalMetadata, input_batch: InputBatch, metadata: AttentionMetadata
    ) -> None:
        hidden = input_batch.hidden_states
        if (persistent_metadata.hidden_states is None) != (hidden is None):
            raise ValueError("Proposal hidden-state inputs changed after capture")
        if hidden is not None:
            persistent_metadata.hidden_states[: input_batch.num_tokens].copy_(hidden)
            persistent_metadata.hidden_states[input_batch.num_tokens :].zero_()


class DSparkDecoderModel(nn.Module):
    """Execute backbone and LM head together, returning logits and hidden states."""

    def __init__(self, model: DSparkModel) -> None:
        super().__init__()
        self._model_ref = ref(model)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(cast(DSparkModel, self._model_ref()).backbone, name)

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        model = cast(DSparkModel, self._model_ref())
        return (
            model.lm_head is not None
            and can_execute_graph(model.backbone, inputs)
            and can_execute_graph(model.lm_head, inputs)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = cast(DSparkModel, self._model_ref())
        model_output = model.backbone(input_ids, positions, input_embedding)
        hidden = model_output[0] if isinstance(model_output, tuple) else model_output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("DSpark backbone must return hidden states")
        return model.compute_logits(hidden, None), hidden


class DSparkProposalModel(nn.Module):
    """Sample input_embedding logits using DSpark-owned Markov/confidence heads."""

    def __init__(self, model: DSparkModel, sampling_builder: ModelSamplingMetadataBuilder) -> None:
        super().__init__()
        self._model_ref = ref(model)
        self._sampling_builder = sampling_builder

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        model = cast(DSparkModel, self._model_ref())
        return (
            inputs.input_batch is not None
            and self._sampling_builder.can_execute_graph(inputs.input_batch.sampling)
            and can_execute_graph(model.markov_head, inputs)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> ModelOutput:
        inputs = get_execution_context(ProposalMetadata)
        sampling_metadata = get_execution_context(ModelSamplingMetadata)
        if inputs is None or sampling_metadata is None or sampling_metadata.sampling is None:
            raise RuntimeError("DSpark proposal model requires proposal and sampling metadata")
        if input_embedding is None:
            raise RuntimeError("DSpark proposal model requires logits")
        model = cast(DSparkModel, self._model_ref())
        query_width = model.query_width
        num_sequences = input_ids.numel() // query_width
        logits = input_embedding.reshape(num_sequences, query_width, input_embedding.shape[-1])
        anchor_tokens = input_ids.reshape(num_sequences, query_width)[:, 0].long()
        prepared_sampling = prepare_sampling_input(sampling_metadata.sampling, logits)
        previous_tokens = anchor_tokens
        sampled_steps: list[torch.Tensor] = []
        probability_steps: list[torch.Tensor] = []
        for token_index in range(query_width):
            markov_bias = model.markov_head.bias(previous_tokens)
            step_logits = logits[:, token_index, :] + markov_bias
            sampled_tokens, probabilities = sample_logits(
                step_logits,
                prepared_sampling,
                tp_size=sampling_metadata.tp_size,
                use_native_tp=sampling_metadata.use_native_tp,
            )
            if model.hot_token_ids is not None:
                sampled_tokens = model.hot_token_ids.index_select(0, sampled_tokens)
                if probabilities is not None:
                    full_probs = probabilities.new_zeros((num_sequences, model.target_vocab_size))
                    full_probs[:, model.hot_token_ids] = probabilities
                    probabilities = full_probs
            sampled_steps.append(sampled_tokens)
            if prepared_sampling.return_probs:
                if probabilities is None:
                    raise RuntimeError("Draft sampling did not return the required probabilities")
                probability_steps.append(probabilities)
            previous_tokens = sampled_tokens
        token_ids = torch.stack(sampled_steps, dim=1)
        confidence = None
        if prepared_sampling.return_confidence:
            if inputs.hidden_states is None:
                raise ValueError("Confidence requires the backbone hidden states")
            previous_matrix = torch.cat((anchor_tokens[:, None], token_ids[:, :-1]), dim=1)
            hidden = inputs.hidden_states.reshape(num_sequences, query_width, inputs.hidden_states.shape[-1])
            confidence = model.confidence_probs(hidden, previous_matrix)
        return ModelOutput(
            draft_proposal=DraftProposal(
                token_ids=token_ids,
                draft_probs=torch.stack(probability_steps, dim=1) if prepared_sampling.return_probs else None,
            ),
            draft_confidence_probs=confidence,
        )
