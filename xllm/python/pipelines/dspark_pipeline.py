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
from typing import cast

import torch

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.execution_pipeline import ExecutionPipeline
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.runner import RunnerConfig
from xllm.python.model_executor.sampling_metadata import ModelSamplingMetadataBuilder
from xllm.python.models.dspark_model import (
    DSparkDecoderModel,
    DSparkModel,
    DSparkProposalModel,
    ProposalMetadataBuilder,
)


class DSparkExecutionPipeline(ExecutionPipeline):
    def __init__(
        self,
        model: DSparkModel,
        builders: Sequence[ExecutionMetadataBuilder],
        runner_config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
    ) -> None:
        sampling_builders = tuple(builder for builder in builders if isinstance(builder, ModelSamplingMetadataBuilder))
        if len(sampling_builders) != 1:
            raise ValueError("DSpark execution requires exactly one sampling metadata builder")
        super().__init__(
            DSparkDecoderModel(model),
            tuple(builder for builder in builders if builder not in sampling_builders),
            runner_config,
            attention_backend=attention_backend,
            restore_history=restore_history,
        )
        self._model = model
        self.proposal_runner = self.register_model(
            DSparkProposalModel(model, sampling_builders[0]), (ProposalMetadataBuilder(), sampling_builders[0])
        )

    def execute(self, inputs: ModelInput) -> ModelOutput:
        if inputs.input_batch is None or inputs.input_batch.sampling is None:
            raise RuntimeError("DSpark pipeline requires sampling metadata to produce a draft proposal")
        width = self._model.query_width
        num_tokens = inputs.input_ids.numel()
        if width <= 0 or num_tokens % width:
            raise ValueError("DSpark input must contain complete query blocks")
        logits, hidden = cast(tuple[torch.Tensor, torch.Tensor], self.entry_runner.execute(inputs))
        proposal_batch = replace(
            inputs.input_batch, hidden_states=hidden if inputs.input_batch.sampling.return_confidence else None
        )
        proposal_inputs = replace(
            inputs,
            input_embedding=logits,
            input_batch=proposal_batch,
            layer_synchronizer=None,
            layer_load_context=None,
            eplb=None,
            multimodal_inputs=None,
        )
        return cast(ModelOutput, self.proposal_runner.execute(proposal_inputs))
