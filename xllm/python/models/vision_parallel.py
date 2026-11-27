# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scheduling and gather primitives shared by vision-encoder parallel modes.

Multimodal towers run in one of two parallel modes inside the language-model TP
group:

* encoder DP — every rank keeps a replicated tower and processes a disjoint
  subset of the batch's media items, then the per-item outputs are AllGathered
  across the group;
* encoder TP — the tower's weights are sharded across that same group and every
  rank processes all items.

This module owns the mode-independent mechanics: the largest-processing-time
item assignment used by encoder DP and the variable-row AllGather that stitches
per-rank outputs back together.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from xllm.python import distributed


def lpt_item_assignment(
    sizes: Sequence[int],
    num_ranks: int,
) -> tuple[list[int], list[int], list[int]]:
    """Assign whole items to ranks: largest first, to the least-loaded rank.

    Returns ``(item_to_rank, item_counts, loads)``. ``item_to_rank`` is a
    rank-major permutation of original item indices, and ``item_counts[r]`` is
    how many of its leading entries rank ``r`` owns.
    """
    if num_ranks <= 0:
        raise ValueError("vision encoder DP rank count must be positive")
    if any(size < 0 for size in sizes):
        raise ValueError("vision encoder DP item sizes must not be negative")
    assignments: list[list[int]] = [[] for _ in range(num_ranks)]
    loads = [0] * num_ranks
    for item_index in sorted(
        range(len(sizes)),
        key=lambda index: sizes[index],
        reverse=True,
    ):
        # Largest item first, assigned to the least-loaded rank by size.
        target_rank = min(range(num_ranks), key=lambda rank: loads[rank])
        assignments[target_rank].append(item_index)
        loads[target_rank] += sizes[item_index]

    # A rank-major permutation of original item indices, not a map from item
    # index to rank: each rank owns the next item_counts[rank] items.
    item_to_rank: list[int] = []
    item_counts: list[int] = []
    for rank_assignment in assignments:
        item_to_rank.extend(rank_assignment)
        item_counts.append(len(rank_assignment))
    return item_to_rank, item_counts, loads


def all_gather_variable_rows(
    local_rows: torch.Tensor,
    rows_per_rank: Sequence[int],
    *,
    rank: int,
    world_size: int,
) -> list[torch.Tensor]:
    """AllGather per-rank row blocks whose real lengths differ.

    AllGather needs equal shapes, so every rank pads its block to the busiest
    rank's row count, gathers along dim 0, and drops the padding on the way
    out. Returns one tensor of real rows per source rank, in rank order.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("vision encoder DP rank and size are invalid")
    if len(rows_per_rank) != world_size:
        raise ValueError("vision encoder DP row counts must cover every rank")
    if local_rows.shape[0] != rows_per_rank[rank]:
        raise ValueError(
            f"vision encoder rows {local_rows.shape[0]} do not match the scheduled "
            f"{rows_per_rank[rank]} rows for rank {rank}"
        )
    if world_size == 1:
        return [local_rows]
    # V4.1 need not initialize a Python TP group. Use the native-first bridge,
    # which falls back to the Python group only outside a native runtime.
    bucket_rows = max(max(rows_per_rank, default=0), 1)
    padding_rows = bucket_rows - local_rows.shape[0]
    if padding_rows:
        padding = local_rows.new_empty((padding_rows, *local_rows.shape[1:]))
        local_rows = torch.cat((local_rows, padding), dim=0)
    gathered = distributed.tp_all_gather(local_rows.contiguous(), dim=0, world_size=world_size)
    return [
        gathered[source * bucket_rows : source * bucket_rows + rows_per_rank[source]] for source in range(world_size)
    ]
