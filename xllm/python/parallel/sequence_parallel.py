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

"""Sequence parallelism for DiT components.

Port of ``dit::SequenceParallelMixin``: scatter sequence-sharded inputs, run
the wrapped forward, gather outputs back. Padding bookkeeping is INSTANCE
state here (``self._padding_lengths``); the C++ original kept it in an
``inline static`` map, which leaked across model instances.

Submodules that ``pad_tensor``/``unpad_tensor`` inside the wrapped forward
(e.g. the attention resharding after ``all_to_all_4D``) must observe the SAME
bookkeeping as the wrapper: pass the owning instance's ``_padding_lengths``
dict into their own mixin constructor instead of letting them allocate one.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from xllm.python.distributed import collectives


class SequenceParallelMixin:
    """Mixin providing sequence parallelism around a forward function."""

    def __init__(
        self,
        sp_group: str | None = None,
        padding_lengths: dict[str, int] | None = None,
    ) -> None:
        self.sp_group = sp_group
        self._padding_lengths = padding_lengths if padding_lengths is not None else {}

    def pad_tensor(self, x: torch.Tensor, tensor_name: str, dim: int) -> torch.Tensor:
        """Right-pad ``dim`` back to the length recorded at scatter time."""
        padding = self._padding_length(tensor_name)
        if padding == 0:
            return x
        normalized_dim = self._normalize_dim(x, dim)
        if x.size(normalized_dim) < padding:
            raise ValueError(f"Padding length {padding} exceeds tensor size {x.size(normalized_dim)}")
        padding_config = [0] * (2 * x.dim())
        padding_index = 2 * (x.dim() - normalized_dim - 1) + 1
        padding_config[padding_index] = padding
        return F.pad(x, padding_config, mode="constant", value=0)

    def unpad_tensor(self, x: torch.Tensor, tensor_name: str, dim: int) -> torch.Tensor:
        """Remove the padding recorded for ``tensor_name`` from ``dim``."""
        padding = self._padding_length(tensor_name)
        if padding == 0:
            return x
        normalized_dim = self._normalize_dim(x, dim)
        if x.size(normalized_dim) < padding:
            raise ValueError(f"Padding length {padding} exceeds tensor size {x.size(normalized_dim)}")
        return x.narrow(normalized_dim, 0, x.size(normalized_dim) - padding)

    def sequence_parallel_forward(
        self,
        inputs: dict[str, tuple[torch.Tensor, int]],
        forward_fn: Callable[[dict[str, torch.Tensor]], dict[str, tuple[torch.Tensor, int]]],
    ) -> dict[str, tuple[torch.Tensor, int]]:
        """Scatter named inputs, run ``forward_fn`` on the local shards, gather.

        ``inputs``/return map names to ``(tensor, sequence_dim)`` pairs, the
        same shape as the C++ ``SequenceParallelTensorMap``.
        """
        self._padding_lengths.clear()
        local_inputs = {
            name: self._scatter(tensor, name, sequence_dim) for name, (tensor, sequence_dim) in inputs.items()
        }
        outputs = forward_fn(local_inputs)
        return {
            name: (self._gather(tensor, name, sequence_dim), sequence_dim)
            for name, (tensor, sequence_dim) in outputs.items()
        }

    def _world_size(self, device: torch.device) -> int:
        return collectives.named_group_size(self.sp_group, device)

    def _rank(self, device: torch.device) -> int:
        return collectives.named_group_rank(self.sp_group, device)

    def _enabled(self, device: torch.device) -> bool:
        return self._world_size(device) > 1

    def _scatter(self, x: torch.Tensor, tensor_name: str, sequence_dim: int) -> torch.Tensor:
        if x is None:
            return x
        world_size = self._world_size(x.device)
        padding = (world_size - x.size(sequence_dim) % world_size) % world_size
        self._padding_lengths[tensor_name] = padding
        if not self._enabled(x.device):
            return x
        padded = self.pad_tensor(x, tensor_name, sequence_dim)
        return collectives.scatter(padded, sequence_dim, world_size, self.sp_group)

    def _gather(
        self,
        x: torch.Tensor,
        tensor_name: str,
        sequence_dim: int,
    ) -> torch.Tensor:
        if x is None or not self._enabled(x.device):
            return x
        world_size = self._world_size(x.device)
        gathered = collectives.gather(x.contiguous(), sequence_dim, world_size, self.sp_group)
        return self.unpad_tensor(gathered, tensor_name, sequence_dim)

    @staticmethod
    def _normalize_dim(x: torch.Tensor, dim: int) -> int:
        normalized_dim = dim if dim >= 0 else x.dim() + dim
        if not 0 <= normalized_dim < x.dim():
            raise ValueError(f"Invalid tensor dimension: {dim}")
        return normalized_dim

    def _padding_length(self, tensor_name: str) -> int:
        try:
            return self._padding_lengths[tensor_name]
        except KeyError:
            raise KeyError(f"Missing sequence-parallel padding metadata: {tensor_name}") from None
