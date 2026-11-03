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

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.execution_context import can_execute_graph
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
    EplbRuntimeState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.runners.acl_graph_capture import capture_acl_graph
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput


@dataclass
class _GraphEntry:
    input_batch: InputBatch
    input_embedding: torch.Tensor | None
    context: ForwardContext
    graph: Any
    output: ModelExecutionOutput
    tasks: list[AclGraphTask]


class AclGraphRunner(BaseRunner):
    """Run a Model with persistent forward arguments and auxiliary metadata.

    Models keep the standard forward signature. The runner owns argument
    storage; builders own typed ForwardContext resources. DecodeAclGraphRunner
    retains the Attention/KV-specific preparation protocol.
    """

    def __init__(self, model: nn.Module, device: torch.device, *, max_requests: int) -> None:
        super().__init__(model, None, device)
        if max_requests < 1:
            raise ValueError("ACL graph request capacity must be positive")
        self.max_requests = max_requests
        self._graphs: dict[tuple[object, ...], _GraphEntry] = {}
        self._stream: Any = None
        self._graph_pool: Any | None = None
        self.last_padded_batch_size: int | None = None

    def can_execute(self, inputs: ModelInput) -> bool:
        if inputs.multimodal_inputs is not None:
            return False
        input_ids = inputs.input_ids
        input_embedding = inputs.input_embedding
        input_batch = inputs.input_batch
        if input_batch is None or input_batch.is_dummy:
            return False
        if not 0 < input_batch.num_reqs <= self.max_requests:
            return False
        width, remainder = divmod(input_batch.num_tokens, input_batch.num_reqs)
        if width <= 0 or remainder or input_ids.ndim != 1 or input_ids.numel() != input_batch.num_tokens:
            return False
        if any(count != width for count in input_batch.num_scheduled_tokens):
            return False
        if inputs.positions.ndim != 1 or inputs.positions.numel() != input_ids.numel():
            return False
        for tensor in (input_embedding, input_batch.hidden_states):
            if tensor is not None and (tensor.ndim < 1 or tensor.shape[0] != input_batch.num_tokens):
                return False
        tensors = (input_ids, inputs.positions, input_embedding, input_batch.hidden_states)
        if any(tensor is not None and tensor.device != self.device for tensor in tensors):
            return False
        return can_execute_graph(self.model, inputs)

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        """Execute inputs already admitted by can_execute."""
        if input_batch is None:
            raise ValueError("Model graph execution requires an input batch")
        width = input_batch.num_tokens // input_batch.num_reqs
        requests = min(1 << (input_batch.num_reqs - 1).bit_length(), self.max_requests)
        rows = requests * width
        tensors = (input_ids, positions, input_embedding, input_batch.hidden_states)
        key = (
            rows,
            width,
            tuple(None if tensor is None else (tuple(tensor.shape[1:]), tensor.dtype) for tensor in tensors),
        )
        caller = torch.npu.current_stream(self.device)
        if self._stream is None:
            self._stream = torch.npu.Stream(device=self.device)
        self._stream.wait_stream(caller)
        with torch.npu.stream(self._stream):
            for tensor in tensors:
                if tensor is not None:
                    tensor.record_stream(self._stream)
            entry = self._graphs.get(key)
            if entry is None:
                static_batch = input_batch.bind_graph_inputs(
                    input_ids.new_zeros(rows),
                    positions.new_zeros(rows),
                    torch.zeros(rows, dtype=torch.bool, device=self.device),
                )
                static_embedding = (
                    input_embedding.new_zeros((rows, *input_embedding.shape[1:]))
                    if input_embedding is not None
                    else None
                )
                contexts = {}
                for builder in self.execution_metadata_builders:
                    if builder.metadata_type in contexts:
                        raise RuntimeError(f"duplicate execution metadata builder for {builder.metadata_type.__name__}")
                    contexts[builder.metadata_type] = builder.allocate_persistent(static_batch, metadata)
                context = ForwardContext(
                    None,
                    self.device,
                    metadata,
                    [],
                    execution_contexts=contexts,
                    execution_state=AclGraphExecutionState({}),
                    eplb=eplb,
                )
                self._update_inputs(static_batch, context, input_batch, metadata, static_embedding, input_embedding)
                for _ in range(3):
                    self._forward(static_batch, context, static_embedding)
                self._stream.synchronize()
                if self._graph_pool is None:
                    self._graph_pool = torch.npu.graph_pool_handle()
                tasks: list[AclGraphTask] = []
                capture_context = replace(context, acl_graph=AclGraphCaptureContext(self._stream, tasks))
                graph, output = capture_acl_graph(
                    lambda: self._forward(static_batch, capture_context, static_embedding),
                    self._stream,
                    pool=self._graph_pool,
                )
                static_batch = replace(static_batch, hidden_states=None)
                entry = _GraphEntry(static_batch, static_embedding, context, graph, output, tasks)
                self._graphs[key] = entry
            else:
                self._update_inputs(
                    entry.input_batch, entry.context, input_batch, metadata, entry.input_embedding, input_embedding
                )
            for task in entry.tasks:
                torch.npu.graph_task_update_begin(self._stream, task.handle)
                task.update()
                torch.npu.graph_task_update_end(self._stream)
                task.event.record(self._stream)
            entry.graph.replay()
            output = self._slice_output(entry.output, input_batch.num_tokens, num_reqs=input_batch.num_reqs)
        caller.wait_stream(self._stream)
        self.last_padded_batch_size = rows
        return output

    def _forward(
        self, input_batch: InputBatch, context: ForwardContext, input_embedding: torch.Tensor | None
    ) -> ModelExecutionOutput:
        with forward_context(context):
            if input_embedding is None:
                return self.model(input_batch.input_ids, input_batch.positions)
            return self.model(input_batch.input_ids, input_batch.positions, input_embedding)

    def _update_inputs(
        self,
        destination: InputBatch,
        context: ForwardContext,
        source: InputBatch,
        metadata: AttentionMetadata,
        destination_embedding: torch.Tensor | None,
        source_embedding: torch.Tensor | None,
    ) -> None:
        for target, value in ((destination.input_ids, source.input_ids), (destination.positions, source.positions)):
            target[: source.num_tokens].copy_(value)
            target[source.num_tokens :].zero_()
        destination.is_padding[: source.num_tokens].fill_(False)
        destination.is_padding[source.num_tokens :].fill_(True)
        if destination_embedding is not None:
            destination_embedding[: source.num_tokens].copy_(source_embedding)
            destination_embedding[source.num_tokens :].zero_()
        for builder in self.execution_metadata_builders:
            builder.update_persistent(context.execution_contexts[builder.metadata_type], source, metadata)
