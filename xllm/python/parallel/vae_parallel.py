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

"""1D spatial (W-dimension) parallelism for the Wan VAE.

Port of ``dit::VaeParallelMixin``. Splits the global W dimension across the
VAE group; each rank owns a contiguous W-local slice. Conv layers obtain
neighbour columns through :meth:`VaeParallelMixin.exchange`; ops needing the
full global tensor use :meth:`VaeParallelMixin.merge` then
:meth:`VaeParallelMixin.split` to restore the local view. When the group is
absent or has a single rank, every operation is an identity.
"""

from __future__ import annotations

import torch

from xllm.python.distributed import collectives


class VaeParallelMixin:
    """Mixin providing 1D spatial-parallel VAE encode/decode.

    ``vae_group`` is the registered group name (e.g. ``"dit_vae"``); ``None``
    or an unregistered name makes every operation an identity.
    """

    def __init__(self, vae_group: str | None = None) -> None:
        self.vae_group = vae_group

    # ---- queries -----------------------------------------------------------

    def vae_parallel_enabled(self, device: torch.device | None = None) -> bool:
        return self._size(self._resolve_device(device)) > 1

    # ---- operations (identity when parallelism is disabled) -----------------

    def split(self, x: torch.Tensor) -> torch.Tensor:
        """Split the global tensor along the last (W) dim into this rank's slice."""
        if not self.vae_parallel_enabled(x.device):
            return x
        width = x.size(-1)
        start, local_w = self._w_extent(width, x.device)
        return x.narrow(-1, start, local_w).contiguous()

    def merge(self, local_patch: torch.Tensor) -> torch.Tensor:
        """All-gather local slices along W and concatenate back to the global tensor."""
        if not self.vae_parallel_enabled(local_patch.device):
            return local_patch

        rank = self._rank(local_patch.device)
        prefix_numel = 1
        for dim in range(local_patch.dim() - 1):
            prefix_numel *= local_patch.size(dim)

        widths = self._allgather_widths(local_patch)
        flat_sizes = [prefix_numel * width for width in widths]

        local_flat = local_patch.reshape(-1)
        gathered = collectives.all_gather_variable(local_flat.contiguous(), flat_sizes, rank, self.vae_group)

        orig_sizes = list(local_patch.shape)
        gathered_tensors = []
        offset = 0
        for width, flat_size in zip(widths, flat_sizes):
            target_shape = orig_sizes.copy()
            target_shape[-1] = width
            gathered_tensors.append(gathered[offset : offset + flat_size].reshape(target_shape))
            offset += flat_size
        return torch.cat(gathered_tensors, dim=-1)

    def exchange(self, local_patch: torch.Tensor, pad: bool) -> torch.Tensor:
        """Halo exchange over the W dimension.

        If ``pad`` is true, missing neighbours are zero-padded; otherwise the
        received columns are concatenated without padding (upsample trim path).
        The even/odd send/recv ordering is required to avoid deadlock in the
        blocking point-to-point ring and must not be reordered.
        """
        if not self.vae_parallel_enabled(local_patch.device):
            return local_patch

        left_col = local_patch.narrow(-1, 0, 1).contiguous()
        right_col = local_patch.narrow(-1, local_patch.size(-1) - 1, 1).contiguous()

        device = local_patch.device
        has_left = self._has_left(device)
        has_right = self._has_right(device)
        rank = self._rank(device)

        left_recv = None
        right_recv = None

        if rank % 2 == 0:
            if has_right:
                collectives.send(right_col, rank + 1, self.vae_group)
                right_recv = collectives.recv(right_col, rank + 1, self.vae_group)
            if has_left:
                collectives.send(left_col, rank - 1, self.vae_group)
                left_recv = collectives.recv(left_col, rank - 1, self.vae_group)
        else:
            if has_left:
                left_recv = collectives.recv(left_col, rank - 1, self.vae_group)
                collectives.send(left_col, rank - 1, self.vae_group)
            if has_right:
                right_recv = collectives.recv(right_col, rank + 1, self.vae_group)
                collectives.send(right_col, rank + 1, self.vae_group)
        if pad:
            left_pad = left_recv if has_left else torch.zeros_like(left_col)
            right_pad = right_recv if has_right else torch.zeros_like(right_col)
            return torch.cat([left_pad, local_patch, right_pad], dim=-1)
        if not has_left:
            return torch.cat([local_patch, right_recv], dim=-1)
        if not has_right:
            return torch.cat([left_recv, local_patch], dim=-1)
        return torch.cat([left_recv, local_patch, right_recv], dim=-1)

    def trim_halo(self, x: torch.Tensor, per_side: int) -> torch.Tensor:
        """Trim halo columns from W after an upsample.

        Removes ``per_side`` columns on each side that received a neighbour's
        halo: both sides for a middle rank, right-only for the first rank,
        left-only for the last rank.
        """
        if not self.vae_parallel_enabled(x.device):
            return x
        current_w = x.size(-1)
        device = x.device
        has_left = self._has_left(device)
        has_right = self._has_right(device)
        if has_left and has_right:
            return x.narrow(-1, per_side, current_w - 2 * per_side)
        if not has_left:
            return x.narrow(-1, 0, current_w - per_side)
        return x.narrow(-1, per_side, current_w - per_side)

    # ---- internals ----------------------------------------------------------

    def _resolve_device(self, device: torch.device | None) -> torch.device:
        if device is not None:
            return device
        device_attr = getattr(self, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr
        raise RuntimeError(
            "VaeParallelMixin requires a `device` attribute (torch.device) to "
            "resolve its process group when no tensor is available"
        )

    def _size(self, device: torch.device) -> int:
        return collectives.named_group_size(self.vae_group, device)

    def _rank(self, device: torch.device) -> int:
        return collectives.named_group_rank(self.vae_group, device)

    def _has_left(self, device: torch.device) -> bool:
        return self._rank(device) > 0

    def _has_right(self, device: torch.device) -> bool:
        return self._rank(device) < self._size(device) - 1

    def _w_extent(self, width: int, device: torch.device) -> tuple[int, int]:
        """(start, local_width) of this rank's slice of a global W of ``width``."""
        size = self._size(device)
        rank = self._rank(device)
        base_w = width // size
        remainder = width % size
        start = rank * base_w + min(rank, remainder)
        local_w = base_w + (1 if rank < remainder else 0)
        return start, local_w

    def _allgather_widths(self, local_patch: torch.Tensor) -> list[int]:
        size = self._size(local_patch.device)
        local_width = torch.tensor([[local_patch.size(-1)]], dtype=torch.int64, device=local_patch.device)
        widths = collectives.all_gather(local_width, 0, size, self.vae_group)
        return [int(width) for width in widths[:, 0]]
