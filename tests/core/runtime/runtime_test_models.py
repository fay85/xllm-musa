# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Model/executor dependencies for public C++ speculative-worker tests."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from xllm.python.model_executor.model_output import ModelOutput


@dataclass
class _Trace:
    draft_tokens: list[torch.Tensor] = field(default_factory=list)
    context_slots: list[torch.Tensor] = field(default_factory=list)
    context_cache: torch.Tensor | None = None


_TRACE = _Trace()


class RuntimeTestModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.is_draft = bool(config["is_draft_engine"])
        self.device = torch.device(config["device"])
        if self.is_draft:
            _TRACE.context_cache = torch.zeros(128, device=self.device)
            size = 2 if self.device.type == "cpu" else 4096
            self._delay_lhs = torch.ones((size, size), device=self.device, dtype=torch.float16)
            self._delay_rhs = torch.ones_like(self._delay_lhs)
            self._delay_scratch = torch.empty_like(self._delay_lhs)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> ModelOutput:
        if not self.is_draft:
            return ModelOutput(
                hidden_states=tokens.float().unsqueeze(1).expand(-1, 4).contiguous(),
                aux_hidden_states=positions.float().unsqueeze(1).expand(-1, 4).contiguous(),
            )
        _TRACE.draft_tokens.append(tokens.clone())
        num_speculative_tokens = int(self.config["num_speculative_tokens"])
        tokens_per_sequence = num_speculative_tokens + (self.config["model_type"] != "DSparkDraftModel")
        anchors = tokens.view(-1, tokens_per_sequence)[:, 0]
        proposal = torch.stack((anchors + 1, anchors + 2), dim=1).long()
        if self.config.get("model_type") == "DFlashDraftModel":
            return ModelOutput(draft_token_ids=proposal)
        draft_logits = torch.full((*proposal.shape, 64), float("-inf"), device=tokens.device)
        draft_logits.scatter_(2, proposal.unsqueeze(2), 0.0)
        return ModelOutput(draft_token_ids=proposal, draft_logits=draft_logits)

    def compute_logits(self, hidden: torch.Tensor, selected: torch.Tensor | None = None) -> torch.Tensor:
        if selected is not None:
            hidden = hidden.index_select(0, selected.long())
        tokens = hidden[:, 0].long() + 1
        logits = torch.zeros((tokens.numel(), 64), device=hidden.device)
        return logits.scatter_(1, tokens.unsqueeze(1), 100.0)

    def write_context_kv(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        caches: Any,
        synchronizer: Any,
    ) -> torch.Tensor:
        del positions, caches, synchronizer
        _TRACE.context_slots.append(slots.clone())
        iterations = 1 if self.device.type == "cpu" else 32
        for _ in range(iterations):
            torch.mm(self._delay_lhs, self._delay_rhs, out=self._delay_scratch)
        assert _TRACE.context_cache is not None
        _TRACE.context_cache.index_fill_(0, slots.flatten().long(), 9)
        return hidden

    def load_weights(self, state_dicts: Any, rank: int, size: int) -> None:
        del state_dicts, rank, size

    def adapt_weights_for_reference_model(self, path: str) -> None:
        del path

    def share_lm_head(self, target_model: nn.Module) -> None:
        del target_model

    def share_embed_tokens(self, target_model: nn.Module) -> None:
        del target_model

    def has_dspark_confidence_head(self) -> bool:
        return False


class RuntimeTestExecutor:
    def __init__(self, model: RuntimeTestModel, config: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.model = model
        self.config = config

    def supports_device_only_decode_inputs(self, kv_page_size: int) -> bool:
        del kv_page_size
        return True

    def set_device_only_decode_inputs(self, enabled: bool) -> None:
        del enabled

    def bind_kv_caches(self, caches: Any) -> None:
        del caches

    def execute(self, tokens: torch.Tensor, positions: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        del args, kwargs
        return self.model(tokens, positions)


class _ModelDependencies:
    def __init__(self) -> None:
        from xllm.python import registry
        from xllm.python.model_executor import executor

        resolve_model = registry.get_model_class
        aliases = {"qwen3", "DSparkDraftModel", "DFlashDraftModel", "DFlash2DraftModel"}

        def resolve(name: str) -> type[nn.Module]:
            if name in aliases:
                return RuntimeTestModel
            return resolve_model(name)

        self._stack = ExitStack()
        # Substitute the public loader boundary, not a worker's installed leaf.
        # The registry itself (including execution metadata builders) stays intact.
        self._stack.enter_context(patch.object(registry, "get_model_class", resolve))
        self._stack.enter_context(patch.object(executor, "ModelExecutor", RuntimeTestExecutor))
        global _TRACE
        _TRACE = _Trace()
        self.trace = _TRACE

    def close(self) -> None:
        self._stack.close()
        self.trace.draft_tokens.clear()
        self.trace.context_slots.clear()
        self.trace.context_cache = None


def _create_fixture() -> _ModelDependencies:
    return _ModelDependencies()
