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

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.runner import Runner, RunnerConfig
from xllm.python.model_executor.runners.base import ModelExecutionOutput


class ExecutionPipeline(ABC):
    """Orchestrate model execution without choosing execution backends.

    Concrete pipelines return ordinary model outputs and establish any stream
    dependencies needed by their caller. Threads, futures and intermediate
    events remain implementation details of the concrete pipeline.
    """

    def __init__(
        self,
        entry_model: nn.Module,
        metadata_builders: Sequence[ExecutionMetadataBuilder],
        runner_config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
    ) -> None:
        self._runner_config = runner_config
        self.runners: list[Runner] = []
        self.entry_runner = self.register_model(
            entry_model, metadata_builders, attention_backend=attention_backend, restore_history=restore_history
        )

    def register_model(
        self,
        model: nn.Module,
        builders: Sequence[ExecutionMetadataBuilder],
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
        coordinate_graph_execution: bool = False,
    ) -> Runner:
        runner = Runner(
            model,
            builders,
            self._runner_config,
            attention_backend=attention_backend,
            restore_history=restore_history,
            coordinate_graph_execution=coordinate_graph_execution,
        )
        self.runners.append(runner)
        return runner

    @abstractmethod
    def execute(self, inputs: ModelInput) -> ModelExecutionOutput: ...


class SingleModelExecutionPipeline(ExecutionPipeline):
    def execute(self, inputs: ModelInput) -> ModelExecutionOutput:
        return self.entry_runner.execute(inputs)


@runtime_checkable
class ExecutionPipelineProvider(Protocol):
    def create_execution_pipeline(
        self,
        builders: Sequence[ExecutionMetadataBuilder],
        runner_config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
    ) -> ExecutionPipeline: ...


def create_execution_pipeline(
    model: nn.Module,
    builders: Sequence[ExecutionMetadataBuilder],
    runner_config: RunnerConfig,
    *,
    attention_backend: AttentionBackend | None = None,
    restore_history: Callable[[AttentionMetadata], None] | None = None,
) -> ExecutionPipeline:
    if isinstance(model, ExecutionPipelineProvider):
        return model.create_execution_pipeline(
            builders, runner_config, attention_backend=attention_backend, restore_history=restore_history
        )
    return SingleModelExecutionPipeline(
        model, builders, runner_config, attention_backend=attention_backend, restore_history=restore_history
    )
