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
from typing import TYPE_CHECKING

import torch

from xllm.python.attention.backend import (
    build_speculative_ssm_state_indices,
    linear_state_checkpoint_stride,
    resolve_linear_state_io_indices,
)

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache
    from xllm.python.model_executor.input_batch import InputBatch


@dataclass(frozen=True, slots=True)
class KdaSpecVerifyMetadata:
    """Request-scoped operator inputs shared by every KDA layer in a step."""

    verify_width: int
    tail_padding: int
    query_start_loc: torch.Tensor
    slot_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    ssm_state_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class KdaMetadata:
    """KDA inputs for the active phase."""

    spec_verify: KdaSpecVerifyMetadata | None = None


class KdaMetadataBuilder:
    """Prepare KDA verify inputs for eager execution and ACL Graph buckets."""

    metadata_type = KdaMetadata

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> KdaMetadataBuilder | None:
        if torch.device(str(config.get("device", "cuda"))).type not in ("npu", "privateuseone"):
            return None
        return cls()

    def __init__(self) -> None:
        self._checkpoint_stride: int | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self._checkpoint_stride = next(
            (
                linear_state_checkpoint_stride(cache.conv, cache.ssm)
                for cache in layer_caches
                if cache.conv is not None and cache.ssm is not None
            ),
            None,
        )

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> KdaMetadata:
        stride = self._checkpoint_stride
        if stride is None or input_batch.is_dummy or input_batch.num_reqs == 0:
            return KdaMetadata()
        if not getattr(metadata, "is_spec_verify", False) and not self._is_legacy_verify(input_batch, metadata):
            return KdaMetadata()
        width = input_batch.num_scheduled_tokens[0]
        slots, accepted = self._request_state(input_batch, metadata, width)
        device = input_batch.input_ids.device
        slots = slots.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
        accepted = accepted.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
        verify = KdaSpecVerifyMetadata(
            verify_width=width,
            tail_padding=0,
            query_start_loc=torch.arange(slots.numel() + 1, dtype=torch.int32, device=device) * width,
            slot_indices=slots,
            num_accepted_tokens=accepted,
            ssm_state_indices=build_speculative_ssm_state_indices(slots, stride),
        )
        return KdaMetadata(verify)

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> KdaMetadata:
        stride = self._checkpoint_stride
        if stride is None or input_batch.num_speculative_tokens is None:
            return KdaMetadata()
        width = input_batch.num_speculative_tokens + 1
        num_slots, tail_padding = divmod(input_batch.num_tokens_after_padding, width)
        if num_slots <= 0:
            raise ValueError("KDA spec verify requires room for one complete token block")
        device = input_batch.input_ids.device
        slots = torch.zeros(num_slots, dtype=torch.int32, device=device)
        return KdaMetadata(
            KdaSpecVerifyMetadata(
                verify_width=width,
                tail_padding=tail_padding,
                query_start_loc=torch.arange(num_slots + 1, dtype=torch.int32, device=device) * width,
                slot_indices=slots,
                num_accepted_tokens=torch.ones_like(slots),
                ssm_state_indices=torch.empty((num_slots, stride), dtype=torch.int32, device=device),
            )
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, KdaMetadata):
            raise TypeError("Expected persistent KDA metadata")
        verify = persistent_metadata.spec_verify
        if verify is None:
            return
        num_reqs = 0 if input_batch.is_dummy else input_batch.num_reqs
        if num_reqs:
            if num_reqs > verify.slot_indices.numel():
                raise RuntimeError("KDA spec verify requests exceed the graph capacity")
            slots, accepted = self._request_state(input_batch, metadata, verify.verify_width)
            verify.slot_indices[:num_reqs].copy_(slots, non_blocking=True)
            verify.num_accepted_tokens[:num_reqs].copy_(accepted, non_blocking=True)
        verify.slot_indices[num_reqs:].zero_()
        verify.num_accepted_tokens[num_reqs:].fill_(1)

        stride = verify.ssm_state_indices.shape[1]
        verify.ssm_state_indices.copy_(build_speculative_ssm_state_indices(verify.slot_indices, stride))

    @staticmethod
    def _is_legacy_verify(input_batch: InputBatch, metadata: AttentionMetadata) -> bool:
        """Recognize older per-token verify layouts from their shapes."""
        if input_batch.num_tokens == input_batch.num_reqs:
            return False
        slots = getattr(metadata, "linear_state_write_indices", None)
        if slots is None:
            slots = getattr(metadata, "linear_state_indices", None)
        if slots is None:
            return False
        if (
            not (metadata.is_prefill or metadata.is_chunked_prefill)
            and getattr(metadata, "num_accepted_tokens", None) is not None
        ):
            return True
        lengths = metadata.q_cu_seq_lens
        return lengths is not None and (
            lengths.numel() == input_batch.num_tokens + 1 or slots.numel() == input_batch.num_tokens
        )

    @staticmethod
    def _request_state(
        input_batch: InputBatch,
        metadata: AttentionMetadata,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = input_batch.num_reqs
        if any(count != width for count in input_batch.num_scheduled_tokens):
            raise RuntimeError("KDA spec verify requires uniform rows per sequence")
        _, slots = resolve_linear_state_io_indices(metadata)
        if slots is None or slots.ndim != 1:
            raise RuntimeError("KDA spec verify requires linear-state slots")
        if slots.numel() == input_batch.num_tokens and num_reqs != input_batch.num_tokens:
            slots = slots.reshape(num_reqs, width)[:, 0]
        elif slots.numel() != num_reqs:
            raise RuntimeError("KDA spec verify requires one linear-state slot per logical sequence")
        accepted = getattr(metadata, "num_accepted_tokens", None)
        if accepted is None or accepted.shape != (num_reqs,):
            raise RuntimeError("KDA spec verify requires one accepted-token count per logical sequence")
        return slots, accepted
