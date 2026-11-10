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
from dataclasses import replace
from typing import TYPE_CHECKING

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.execution_pipeline import ExecutionPipeline
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import DraftCandidates, DraftDecoderOutput
from xllm.python.model_executor.runner import RunnerConfig
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.models.dflash2_model import (
    DFlash2DecoderMetadataBuilder,
    DFlash2DecoderModel,
    DFlash2ProposalModel,
    DFlash2SamplingMetadataBuilder,
)

if TYPE_CHECKING:
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM


class DFlash2ExecutionPipeline(ExecutionPipeline):
    def __init__(
        self,
        model: DFlash2Qwen3ForCausalLM,
        builders: Sequence[ExecutionMetadataBuilder],
        runner_config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
    ) -> None:
        sampling_builders = tuple(
            builder for builder in builders if isinstance(builder, DFlash2SamplingMetadataBuilder)
        )
        if len(sampling_builders) != 1:
            raise ValueError("DFlash2 execution requires exactly one sampling metadata builder")
        self._sampling_builder = sampling_builders[0]
        super().__init__(
            DFlash2DecoderModel(model, self._sampling_builder.query_width),
            (
                *(builder for builder in builders if builder not in sampling_builders),
                DFlash2DecoderMetadataBuilder(self._sampling_builder.query_width),
            ),
            runner_config,
            attention_backend=attention_backend,
            restore_history=restore_history,
        )
        self._model = model
        self.proposal_runner = self.register_model(
            DFlash2ProposalModel(self._sampling_builder, model.cfg.vocab_size, model.dtype),
            sampling_builders,
            coordinate_graph_execution=True,
        )

    def execute(self, inputs: ModelInput) -> ModelExecutionOutput:
        batch = inputs.input_batch
        has_proposal = batch is not None and not batch.is_dummy and batch.sampling is not None
        width = self._sampling_builder.query_width
        if has_proposal and (
            batch.num_tokens != batch.num_reqs * width or any(count != width for count in batch.num_scheduled_tokens)
        ):
            raise ValueError("DFlash2 input must contain complete query blocks")
        output = self.entry_runner.execute(inputs)
        if not isinstance(output, DraftDecoderOutput):
            raise TypeError("DFlash2 decoder must return hidden states and optional candidates")
        if not has_proposal:
            self.proposal_runner.execute(inputs, skip_execution=True)
            return output.hidden_states
        if output.candidates is None:
            raise RuntimeError("DFlash2 decoder must return candidates for proposal execution")
        return self.proposal_runner.execute(self._prepare_proposal_inputs(inputs, output.candidates))

    def _prepare_proposal_inputs(self, inputs: ModelInput, candidates: DraftCandidates) -> ModelInput:
        batch = inputs.input_batch
        width = self._sampling_builder.query_width
        steps = self._sampling_builder.num_steps
        requests = batch.num_reqs
        input_ids = inputs.input_ids.view(requests, width)[:, :-1].reshape(-1)
        positions = inputs.positions.view(requests, width)[:, 1:].reshape(-1)
        proposal_batch = replace(
            batch,
            input_ids=input_ids,
            positions=positions,
            num_tokens=requests * steps,
            num_tokens_after_padding=requests * steps,
            num_scheduled_tokens=(steps,) * requests,
            num_computed_tokens=tuple(count + 1 for count in batch.num_computed_tokens),
            query_start_loc=tuple(index * steps for index in range(requests + 1)),
            is_padding=None,
            hidden_states=candidates.token_ids.flatten(0, 1),
        )
        return replace(
            inputs,
            input_ids=input_ids,
            positions=positions,
            input_embedding=candidates.edge_logits.flatten(0, 1),
            input_batch=proposal_batch,
            layer_synchronizer=None,
            layer_load_context=None,
            eplb=None,
            multimodal_inputs=None,
        )
