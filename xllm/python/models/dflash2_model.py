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

from collections.abc import Mapping
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
from xllm.python.model_executor.model_output import DraftCandidates, DraftDecoderOutput, ModelOutput
from xllm.python.model_executor.sampling_metadata import ModelSamplingInput, SamplingMode
from xllm.python.models.dflash_model import DFlashDecoderMetadata

if TYPE_CHECKING:
    from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM


@dataclass
class DFlash2SamplingMetadata:
    candidate_ids: torch.Tensor
    noise: torch.Tensor
    temperatures: torch.Tensor | None
    do_sample: torch.Tensor | None
    sampling_mode: SamplingMode


class DFlash2SamplingMetadataBuilder:
    """Own candidate and sampling inputs; noise is already agreed across TP."""

    metadata_type: type[object] = DFlash2SamplingMetadata

    def __init__(self, num_speculative_tokens: int, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("DFlash2 requires a positive selector top_k")
        self.num_speculative_tokens = num_speculative_tokens
        self.top_k = top_k

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> DFlash2SamplingMetadataBuilder:
        return cls(cast("int", config["num_speculative_tokens"]), cast("int", config["dflash2_selector_top_k"]))

    def can_execute_graph(self, input_batch: InputBatch) -> bool:
        # TODO: Support random/mixed and temperature-scaled graphs with mode-aware
        # graph keys and persistent temperature/do_sample buffers. Keep eager
        # fallback until capture variants and metadata updates are implemented.
        sampling = input_batch.sampling
        return (
            sampling is not None
            and self._has_valid_inputs(input_batch)
            and sampling.parameters.sampling_mode == SamplingMode.GREEDY
            and sampling.parameters.temperatures is None
        )

    def _has_valid_inputs(self, input_batch: InputBatch) -> bool:
        sampling = input_batch.sampling
        candidates = input_batch.hidden_states
        if sampling is None or sampling.gumbel_noise is None or candidates is None:
            return False
        noise = sampling.gumbel_noise
        requests = input_batch.num_reqs
        if input_batch.num_tokens != requests * self.num_speculative_tokens:
            return False
        if any(count != self.num_speculative_tokens for count in input_batch.num_scheduled_tokens):
            return False
        if candidates.shape != (input_batch.num_tokens, self.top_k):
            return False
        if noise.shape != (requests, self.num_speculative_tokens, self.top_k):
            return False
        if any(tensor.device != input_batch.input_ids.device for tensor in (noise, candidates)):
            return False
        parameters = sampling.parameters
        if parameters.sampling_mode == SamplingMode.MIXED and parameters.do_sample is None:
            return False
        return all(
            tensor is None or tensor.shape == (requests,) for tensor in (parameters.temperatures, parameters.do_sample)
        )

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DFlash2SamplingMetadata:
        if not self._has_valid_inputs(input_batch):
            raise ValueError("DFlash2 sampling requires candidate rows and per-request selector inputs")
        sampling = input_batch.sampling
        parameters = sampling.parameters
        device = input_batch.input_ids.device
        temperatures = (
            None if parameters.temperatures is None else parameters.temperatures.to(device=device, dtype=torch.float32)
        )
        do_sample = (
            None
            if parameters.sampling_mode != SamplingMode.MIXED
            else parameters.do_sample.to(device=device, dtype=torch.bool)
        )
        return DFlash2SamplingMetadata(
            input_batch.hidden_states,
            sampling.gumbel_noise,
            temperatures,
            do_sample,
            parameters.sampling_mode,
        )

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DFlash2SamplingMetadata:
        requests, remainder = divmod(input_batch.num_tokens_after_padding, self.num_speculative_tokens)
        if remainder:
            raise ValueError("DFlash2 graph padding must preserve complete candidate blocks")
        device = input_batch.input_ids.device
        return DFlash2SamplingMetadata(
            input_batch.hidden_states.new_zeros((input_batch.num_tokens_after_padding, self.top_k)),
            torch.zeros((requests, self.num_speculative_tokens, self.top_k), device=device, dtype=torch.float32),
            None,
            None,
            SamplingMode.GREEDY,
        )

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: AttentionMetadata
    ) -> None:
        if not self.can_execute_graph(input_batch):
            raise ValueError("DFlash2 sampling graph requires greedy sampling without temperature scaling")
        stored = cast("DFlash2SamplingMetadata", persistent_metadata)
        sampling = cast("ModelSamplingInput", input_batch.sampling)
        for destination, value, count in (
            (stored.candidate_ids, input_batch.hidden_states, input_batch.num_tokens),
            (stored.noise, sampling.gumbel_noise, input_batch.num_reqs),
        ):
            if value.device.type == "npu":
                value.record_stream(torch.npu.current_stream(value.device))
            destination[:count].copy_(value)
            destination[count:].zero_()


class DFlash2DecoderModel(nn.Module):
    """Execute backbone, proposal-position LM head and candidate selector together."""

    def __init__(self, model: DFlash2Qwen3ForCausalLM, query_width: int) -> None:
        super().__init__()
        self._model_ref = ref(model)
        self._query_width = query_width

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(cast("DFlash2Qwen3ForCausalLM", self._model_ref()).model, name)

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        model = cast("DFlash2Qwen3ForCausalLM", self._model_ref())
        return (
            bool(getattr(inputs.metadata, "use_dflash_proposal_attention", False))
            and inputs.input_batch is not None
            and model.lm_head is not None
            and can_execute_graph(model.model, inputs)
            and can_execute_graph(model.lm_head, inputs)
            and can_execute_graph(model.model.candidate_selector, inputs)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> DraftDecoderOutput:
        model = cast("DFlash2Qwen3ForCausalLM", self._model_ref())
        hidden = model.model(input_ids, positions, input_embedding)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("DFlash2 backbone must return hidden states")
        decoder = get_execution_context(DFlashDecoderMetadata)
        if decoder is None or decoder.num_requests == 0:
            return DraftDecoderOutput(hidden_states=hidden)
        requests = decoder.num_requests
        steps = self._query_width - 1
        proposal_hidden = hidden.view(requests, self._query_width, -1)[:, 1:].reshape(requests * steps, -1)
        logits = model.compute_logits(proposal_hidden, None).view(requests, steps, -1)
        candidate_ids, edge_logits = model.dflash2_candidates(
            proposal_hidden.view(requests, steps, -1),
            logits,
            input_ids.view(requests, self._query_width)[:, 0],
        )
        return DraftDecoderOutput(
            hidden_states=hidden,
            candidates=DraftCandidates(candidate_ids, edge_logits),
        )


class DFlash2ProposalModel(nn.Module):
    """Sample selector paths, retaining native greedy/random/mixed fast paths.

    Penalties, filters and grammar remain target-verification responsibilities.
    Only greedy sampling without temperature scaling is captured; other request
    modes use the same forward through eager execution.
    """

    def __init__(self, builder: DFlash2SamplingMetadataBuilder, vocab_size: int, logits_dtype: torch.dtype) -> None:
        super().__init__()
        self._builder = builder
        self._vocab_size = vocab_size
        self._logits_dtype = logits_dtype

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        return inputs.input_batch is not None and self._builder.can_execute_graph(inputs.input_batch)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> ModelOutput:
        sampling = get_execution_context(DFlash2SamplingMetadata)
        if sampling is None or input_embedding is None:
            raise RuntimeError("DFlash2 proposal requires selector edges and sampling metadata")
        steps = self._builder.num_speculative_tokens
        requests = input_ids.numel() // steps
        top_k = self._builder.top_k
        candidate_ids = sampling.candidate_ids.view(requests, steps, top_k)
        edge_logits = input_embedding.view(requests, steps, top_k, top_k).float()
        if sampling.temperatures is not None:
            temperatures = sampling.temperatures.to(self._logits_dtype).float()
            temperatures = temperatures.masked_fill(temperatures == 0, 1).view(requests, 1, 1, 1)
            edge_logits = edge_logits / temperatures
        edge_rows = edge_logits.view(-1, top_k)
        previous_indices = torch.zeros(requests, dtype=torch.long, device=input_ids.device)
        batch_offsets = torch.arange(requests, device=input_ids.device) * (steps * top_k)
        tokens = torch.empty((requests, steps), dtype=candidate_ids.dtype, device=input_ids.device)
        return_draft_logits = sampling.sampling_mode != SamplingMode.GREEDY
        candidate_logits = (
            torch.empty((requests, steps, top_k), dtype=torch.float32, device=input_ids.device)
            if return_draft_logits
            else None
        )
        for step in range(steps):
            transition_logits = edge_rows.index_select(0, previous_indices + batch_offsets + step * top_k)
            sampled = (transition_logits + sampling.noise[:, step]).argmax(dim=-1)
            tokens[:, step] = candidate_ids[:, step].gather(1, sampled[:, None]).squeeze(1)
            if candidate_logits is not None:
                if sampling.do_sample is not None:
                    greedy_logits = torch.full_like(transition_logits, float("-inf")).scatter_(1, sampled[:, None], 0.0)
                    transition_logits = torch.where(sampling.do_sample[:, None], transition_logits, greedy_logits)
                candidate_logits[:, step] = transition_logits
            previous_indices = sampled
        draft_logits = None
        if candidate_logits is not None:
            draft_logits = torch.full(
                (requests, steps, self._vocab_size), float("-inf"), dtype=torch.float32, device=input_ids.device
            ).scatter_(-1, candidate_ids, candidate_logits)
        return ModelOutput(draft_token_ids=tokens, draft_logits=draft_logits)
