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
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform


@dataclass(frozen=True)
class RunnerConfig:
    device: torch.device
    model_config: dict = field(default_factory=dict)
    enable_graph: bool = False
    max_seqs_per_batch: int = 1
    num_decoding_tokens: int = 1
    acl_graph_decode_batch_size_limit: int | None = None
    is_spec_draft: bool = False


class Runner:
    """Select and invoke an execution backend for one model."""

    def __init__(
        self,
        model: nn.Module,
        metadata_builders: Sequence[ExecutionMetadataBuilder],
        config: RunnerConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        restore_history: Callable[[AttentionMetadata], None] | None = None,
        coordinate_graph_execution: bool = False,
    ) -> None:
        self.model = model
        self.metadata_builders = tuple(metadata_builders)
        self.eager_runner = EagerRunner(model, attention_backend, config.device)
        if attention_backend is not None:
            self.eager_runner.cp_size = int(config.model_config.get("cp_size", 1))
            self.eager_runner.cp_rank = int(config.model_config.get("cp_rank", 0))
        self.graph_runner = self._create_graph_runner(config, attention_backend)
        for backend in (self.eager_runner, self.graph_runner):
            if backend is not None:
                backend.bind_execution_metadata_builders(self.metadata_builders)
        self.execution_mode: str | None = None
        self.last_graph_rows: int | None = None
        self.restore_history = restore_history
        self._graph_dp_size = int(config.model_config.get("dp_size", 1)) if coordinate_graph_execution else 1

    def _create_graph_runner(
        self, options: RunnerConfig, attention_backend: AttentionBackend | None
    ) -> BaseRunner | None:
        if not options.enable_graph:
            return None
        device = options.device
        if attention_backend is None:
            if not current_platform.is_npu():
                return None
            from xllm.python.model_executor.runners.acl_graph import AclGraphRunner

            max_requests = options.max_seqs_per_batch
            graph_limit = options.acl_graph_decode_batch_size_limit
            if graph_limit is not None and graph_limit > 0:
                max_requests = min(max_requests, graph_limit)
            return AclGraphRunner(self.model, device, max_requests=max_requests)

        config = options.model_config
        layerwise_split_size = int(config.get("layerwise_split_size", 1))
        if layerwise_split_size > 1:
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; graph execution is not supported."
            )
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        if current_platform.is_cuda():
            from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner

            graph_runner = DecodeCudaGraphRunner(
                self.model,
                attention_backend,
                device,
                options.max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
            )
            logger.warning(
                "enable_graph=true selected the CUDA graph runner, but its "
                "warmup/capture path is not wired into ModelExecutor; "
                "decode will run eagerly without CUDA graph capture/replay."
            )
            return graph_runner
        if current_platform.is_npu():
            from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

            eplv2_policy = getattr(self.model, "_comm_policy", None)
            eplv2_graph_kwargs: dict[str, int] = {}
            if eplv2_policy is not None and int(config.get("expert_parallel_degree", 0)) == 2:
                tp_size = int(config.get("tp_size", 1))
                eplv2_graph_kwargs["eplv2_graph_token_limit"] = eplv2_policy.graph_token_limit * tp_size
            glm_eplv2 = (
                config.get("model_type") in ("glm5_next", "glm5_next_text")
                and int(config.get("expert_parallel_degree", 0)) == 2
                and int(config.get("ep_size", 1)) > 1
            )
            return DecodeAclGraphRunner(
                self.model,
                attention_backend,
                device,
                options.max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
                options.acl_graph_decode_batch_size_limit,
                num_decoding_tokens=options.num_decoding_tokens,
                enable_mega_moe_token_mask=bool(config.get("enable_mega_moe", False) and not glm_eplv2),
                is_spec_draft=options.is_spec_draft,
                draft_query_width=int(config.get("draft_query_width", 1) or 1),
                capture_sizes=config.get("graph_capture_sizes", ()),
                **eplv2_graph_kwargs,
            )
        return None

    def execute(self, inputs: ModelInput, *, skip_execution: bool = False) -> ModelExecutionOutput:
        """Run a stage, or only join its admission when the pipeline has no local work."""
        self.last_graph_rows = None
        graph = self.graph_runner
        use_graph = graph is not None and (skip_execution or graph.can_execute(inputs))
        if graph is not None and self._graph_dp_size > 1:
            use_graph = distributed.dp_all_true(use_graph, graph.device, self._graph_dp_size)
        if skip_execution:
            self.execution_mode = None
            return inputs.input_ids
        if use_graph:
            if inputs.layer_load_context is not None:
                inputs.layer_load_context.synchronize_all_layers()
            graph.warmup(inputs.input_ids, inputs.positions, inputs.metadata, inputs.input_embedding)
            if not getattr(graph, "supports_draft_to_target_overlap", False):
                self._restore_history(inputs.metadata)
            self.execution_mode = "graph"
            output = graph.execute(
                inputs.input_ids,
                inputs.positions,
                inputs.metadata,
                inputs.input_embedding,
                eplb=inputs.eplb,
                input_batch=inputs.input_batch,
            )
            self.last_graph_rows = getattr(graph, "last_padded_batch_size", None)
            return output
        self._restore_history(inputs.metadata)
        self.execution_mode = "eager"
        return self.eager_runner.execute(
            inputs.input_ids,
            inputs.positions,
            inputs.metadata,
            inputs.input_embedding,
            layer_synchronizer=inputs.layer_synchronizer,
            eplb=inputs.eplb,
            input_batch=inputs.input_batch,
            layer_load_context=inputs.layer_load_context,
            multimodal_inputs=inputs.multimodal_inputs,
        )

    def _restore_history(self, metadata: AttentionMetadata) -> None:
        if self.restore_history is not None:
            self.restore_history(metadata)
