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
"""NPU Triton helpers for fused KPool state preparation and storage."""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _store_fused_pool(
    valid_ptr,
    positions_ptr,
    pooled_ptr,
    cache_ptr,
    tail_ptr,
    table_ptr,
    request,
    read_id,
    read_slot,
    first_position,
    pool,
    NUM_POOL_BLOCKS: tl.constexpr,
    POOL_BLOCK_SIZE: tl.constexpr,
    TABLE_WIDTH,
    TABLE_REQUEST_STRIDE,
    TAIL_CAPACITY: tl.constexpr,
    POOLED_ROWS,
    QUERY_LEN,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    first_row = request * QUERY_LEN
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    completion_start = (RATE - 1 - first_position % RATE) % RATE
    offset = completion_start + pool * RATE
    completion = tl.load(positions_ptr + first_row + offset, offset < QUERY_LEN, other=-1).to(tl.int64)
    complete = (read_id > 0) & (offset < QUERY_LEN) & (completion >= RATE - 1) & (completion == first_position + offset)
    for member in tl.static_range(RATE):
        position = completion - RATE + 1 + member
        source = position - first_position
        safe_source = tl.minimum(tl.maximum(source, 0), QUERY_LEN - 1)
        current_position = tl.load(positions_ptr + first_row + safe_source).to(tl.int64)
        from_current = (source >= 0) & (source < QUERY_LEN) & (current_position == position)
        current_valid = tl.load(valid_ptr + first_row + safe_source).to(tl.int1)
        old_gate = tl.load(
            tail_ptr + ((read_slot * 2 + 1) * TAIL_CAPACITY + position % TAIL_CAPACITY) * HEAD_DIM,
            mask=position >= 0,
            other=float("-inf"),
        ).to(tl.float32)
        # Scalar abs fails to lower in Triton Ascend 3.2.1; comparisons
        # preserve the finite-value check without changing its semantics.
        history_valid = (
            (position >= 0) & (old_gate != float("inf")) & (old_gate != float("-inf")) & (old_gate == old_gate)
        )
        complete = complete & tl.where(from_current, current_valid, history_valid)
    pool_id = tl.maximum(completion, 0) // RATE
    logical_block = pool_id // POOL_BLOCK_SIZE
    physical_block = tl.load(
        table_ptr + request * TABLE_REQUEST_STRIDE + logical_block,
        mask=complete & (logical_block < TABLE_WIDTH),
        other=-1,
    ).to(tl.int64)
    store_pool = complete & (physical_block >= 0) & (physical_block < NUM_POOL_BLOCKS)
    pooled = tl.load(
        pooled_ptr + (request * POOLED_ROWS + pool) * HEAD_DIM + dims,
        mask=store_pool & dim_valid,
        other=0,
    )
    tl.store(
        cache_ptr + (physical_block * POOL_BLOCK_SIZE + pool_id % POOL_BLOCK_SIZE) * HEAD_DIM + dims,
        pooled,
        mask=store_pool & dim_valid,
    )


@triton.jit(
    do_not_specialize=[
        "valid_ptr",
        "positions_ptr",
        "pooled_ptr",
        "cache_ptr",
        "tail_ptr",
        "read_ids_ptr",
        "table_ptr",
        "POOLED_ROWS",
        "QUERY_LEN",
        "TABLE_WIDTH",
        "TABLE_REQUEST_STRIDE",
    ]
)
def _store_fused_pools_kernel(
    valid_ptr,
    positions_ptr,
    pooled_ptr,
    cache_ptr,
    tail_ptr,
    table_ptr,
    read_ids_ptr,
    NUM_TAIL_SLOTS: tl.constexpr,
    SNAPSHOT_TAIL_IO: tl.constexpr,
    NUM_POOL_BLOCKS: tl.constexpr,
    POOL_BLOCK_SIZE: tl.constexpr,
    TABLE_WIDTH,
    TABLE_REQUEST_STRIDE,
    TAIL_CAPACITY: tl.constexpr,
    POOLED_ROWS,
    QUERY_LEN,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    pool = tl.program_id(1)
    read_id = tl.load(read_ids_ptr + request).to(tl.int64)
    read_id = tl.minimum(tl.maximum(read_id, 0), NUM_TAIL_SLOTS - 1)
    first_position = tl.load(positions_ptr + request * QUERY_LEN).to(tl.int64)
    _store_fused_pool(
        valid_ptr,
        positions_ptr,
        pooled_ptr,
        cache_ptr,
        tail_ptr,
        table_ptr,
        request,
        read_id,
        request * 2 if SNAPSHOT_TAIL_IO else read_id,
        first_position,
        pool,
        NUM_POOL_BLOCKS,
        POOL_BLOCK_SIZE,
        TABLE_WIDTH,
        TABLE_REQUEST_STRIDE,
        TAIL_CAPACITY,
        POOLED_ROWS,
        QUERY_LEN,
        RATE,
        HEAD_DIM,
        BLOCK_D,
    )


@triton.jit
def _update_small_fused_tail(
    keys_ptr,
    gates_ptr,
    positions_ptr,
    tail_valid_ptr,
    first_row,
    query_len,
    write_id,
    dims,
    dim_valid,
    tail_rows,
    keys,
    gates,
    HEAD_DIM: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    stash_rows: tl.constexpr = BLOCK_Q if BLOCK_Q <= TAIL_CAPACITY else TAIL_CAPACITY
    stash_start = tl.maximum(0, query_len - TAIL_CAPACITY) if BLOCK_Q > TAIL_CAPACITY else 0
    last_position = tl.full((), -1, tl.int64)
    for offset in tl.static_range(stash_rows):
        row = stash_start + offset
        if BLOCK_Q == 4 and offset == 3:
            row = tl.minimum(row, query_len - 1)
        position = tl.load(positions_ptr + first_row + row).to(tl.int64)
        valid = tl.load(tail_valid_ptr + first_row + row).to(tl.int1)
        if TAIL_CAPACITY >= BLOCK_Q:
            last_position = tl.maximum(last_position, tl.where(valid, position, -1))
        current_keys = tl.load(keys_ptr + (first_row + row) * HEAD_DIM + dims, dim_valid, other=0)
        current_gates = tl.load(gates_ptr + (first_row + row) * HEAD_DIM + dims, dim_valid, other=0)
        replace = (write_id > 0) & (position >= 0) & (tail_rows[:, None] == position % TAIL_CAPACITY)
        keys = tl.where(replace, tl.where(valid, current_keys, 0)[None, :], keys)
        gates = tl.where(replace, tl.where(valid, current_gates, float("-inf"))[None, :], gates)
    return keys, gates, last_position


@triton.jit(
    do_not_specialize=[
        "keys_ptr",
        "gates_ptr",
        "valid_ptr",
        "tail_valid_ptr",
        "positions_ptr",
        "pooled_ptr",
        "states_ptr",
        "cache_ptr",
        "tail_ptr",
        "read_tail_ptr",
        "read_ids_ptr",
        "write_ids_ptr",
        "table_ptr",
        "POOLED_ROWS",
        "QUERY_LEN",
        "TABLE_WIDTH",
        "TABLE_REQUEST_STRIDE",
    ]
)
def _store_fused_kpool_kernel(
    keys_ptr,
    gates_ptr,
    valid_ptr,
    tail_valid_ptr,
    positions_ptr,
    pooled_ptr,
    states_ptr,
    cache_ptr,
    tail_ptr,
    read_tail_ptr,
    read_ids_ptr,
    write_ids_ptr,
    table_ptr,
    NUM_TAIL_SLOTS: tl.constexpr,
    SNAPSHOT_TAIL_IO: tl.constexpr,
    NUM_POOL_BLOCKS: tl.constexpr,
    POOL_BLOCK_SIZE: tl.constexpr,
    TABLE_WIDTH,
    TABLE_REQUEST_STRIDE,
    TAIL_CAPACITY: tl.constexpr,
    POOLED_ROWS,
    QUERY_LEN,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    SMALL_QUERY: tl.constexpr,
    DISTINCT_TAIL_IO: tl.constexpr,
):
    """Store CANN results without recomputing pooling or serial tensor writes."""
    request = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    query_len = BLOCK_Q if BLOCK_Q <= 2 else QUERY_LEN
    first_row = request * query_len
    read_id = tl.load(read_ids_ptr + request).to(tl.int64)
    write_id = tl.load(write_ids_ptr + request).to(tl.int64)
    read_id = tl.minimum(tl.maximum(read_id, 0), NUM_TAIL_SLOTS - 1)
    write_id = tl.minimum(tl.maximum(write_id, 0), NUM_TAIL_SLOTS - 1)
    first_position = tl.load(positions_ptr + first_row).to(tl.int64)

    # Write slots belong to one request. Read slots may alias other requests'
    # write slots; out-of-place restores read the batch snapshot instead.
    if SMALL_QUERY:
        for pool in tl.static_range((BLOCK_Q + RATE - 1) // RATE):
            _store_fused_pool(
                valid_ptr,
                positions_ptr,
                pooled_ptr,
                cache_ptr,
                read_tail_ptr,
                table_ptr,
                request,
                read_id,
                request * 2 if SNAPSHOT_TAIL_IO else read_id,
                first_position,
                pool,
                NUM_POOL_BLOCKS,
                POOL_BLOCK_SIZE,
                TABLE_WIDTH,
                TABLE_REQUEST_STRIDE,
                TAIL_CAPACITY,
                POOLED_ROWS,
                query_len,
                RATE,
                HEAD_DIM,
                BLOCK_D,
            )

    tail_rows = tl.arange(0, BLOCK_TAIL)
    tail_mask = (tail_rows[:, None] < TAIL_CAPACITY) & dim_valid[None, :]
    base_id = write_id
    if DISTINCT_TAIL_IO:
        base_id = tl.where((read_id > 0) & (write_id > 0), read_id, write_id)
    if SNAPSHOT_TAIL_IO:
        base_id = request * 2 + tl.where((read_id > 0) & (write_id > 0), 0, 1)
    base = (base_id * 2 * TAIL_CAPACITY + tail_rows[:, None]) * HEAD_DIM + dims[None, :]
    keys = tl.load(read_tail_ptr + base, tail_mask, other=0).to(tl.float32)
    gates = tl.load(read_tail_ptr + base + TAIL_CAPACITY * HEAD_DIM, tail_mask, other=float("-inf")).to(tl.float32)
    # Select the last current token for each ring row in registers. Padded
    # rows never write, even when their clamped position aliases a live row.
    if BLOCK_Q <= 4:
        keys, gates, last_position = _update_small_fused_tail(
            keys_ptr,
            gates_ptr,
            positions_ptr,
            tail_valid_ptr,
            first_row,
            query_len,
            write_id,
            dims,
            dim_valid,
            tail_rows,
            keys,
            gates,
            HEAD_DIM,
            TAIL_CAPACITY,
            BLOCK_Q,
        )
    else:
        for offset in range(tl.maximum(0, query_len - TAIL_CAPACITY), query_len):
            position = tl.load(positions_ptr + first_row + offset).to(tl.int64)
            valid = tl.load(tail_valid_ptr + first_row + offset).to(tl.int1)
            current_keys = tl.load(keys_ptr + (first_row + offset) * HEAD_DIM + dims, dim_valid, other=0)
            current_gates = tl.load(gates_ptr + (first_row + offset) * HEAD_DIM + dims, dim_valid, other=0)
            replace = (write_id > 0) & (position >= 0) & (tail_rows[:, None] == position % TAIL_CAPACITY)
            keys = tl.where(replace, tl.where(valid, current_keys, 0)[None, :], keys)
            gates = tl.where(replace, tl.where(valid, current_gates, float("-inf"))[None, :], gates)

    offsets = tl.arange(0, BLOCK_Q)
    # Like va, inspect the query end first: request positions are ordered and
    # the last valid tile contains the final position. Only padding requires
    # another tile. Keep UB bounded instead of reducing the whole query.
    if BLOCK_Q > 4 or TAIL_CAPACITY < BLOCK_Q:
        last_position = tl.full((), -1, tl.int64)
        tile_end = query_len
        while (tile_end > 0) & (last_position < 0):
            rows = tl.maximum(0, tile_end - BLOCK_Q) + offsets
            positions = tl.load(positions_ptr + first_row + rows, rows < tile_end, other=-1).to(tl.int64)
            tile_valid = tl.load(tail_valid_ptr + first_row + rows, rows < tile_end, other=0).to(tl.int1)
            last_position = tl.max(tl.where(tile_valid & (positions >= 0), positions, -1), 0)
            tile_end = tl.maximum(0, tile_end - BLOCK_Q)
    end = last_position + 1
    remainder = end % RATE
    pool_start = end - remainder
    member = (tail_rows - pool_start % TAIL_CAPACITY + TAIL_CAPACITY) % TAIL_CAPACITY
    live = (member[:, None] < remainder) & (write_id > 0)
    state_base = (request * 2 * RATE + member[:, None]) * HEAD_DIM + dims[None, :]
    state_keys = tl.load(states_ptr + state_base, live & tail_mask, other=0)
    state_gates = tl.load(states_ptr + state_base + RATE * HEAD_DIM, live & tail_mask, other=0)
    keys = tl.where(live, state_keys, keys)
    gates = tl.where(live, state_gates, gates)
    destination = (write_id * 2 * TAIL_CAPACITY + tail_rows[:, None]) * HEAD_DIM + dims[None, :]
    tl.store(tail_ptr + destination, keys, tail_mask & (write_id > 0))
    tl.store(tail_ptr + destination + TAIL_CAPACITY * HEAD_DIM, gates, tail_mask & (write_id > 0))


def store_fused_kpool(
    raw_k: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_rows: torch.Tensor,
    tail_valid_rows: torch.Tensor,
    positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_read_ids: torch.Tensor,
    tail_write_ids: torch.Tensor,
    block_table: torch.Tensor,
    query_len: int,
    rate: int,
    fused_states: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    read_tail_cache: torch.Tensor | None = None,
) -> None:
    """Persist uniform queries, retaining CANN's exact FP32 tail.

    Prefill query lengths share a binary per power-of-two scan tile, capped at 256.
    Keep the smaller scan tiles for decode.
    """
    if len(fused_states) == 1:
        pooled = fused_states[0][0].unsqueeze(0).contiguous()
        states = fused_states[0][1].unsqueeze(0).contiguous()
    else:
        pooled = torch.stack([state[0] for state in fused_states])
        states = torch.stack([state[1] for state in fused_states])
    head_dim = raw_k.shape[-1]
    snapshot_tail_io = read_tail_cache is not None
    read_tail_cache = tail_cache if read_tail_cache is None else read_tail_cache
    requests = len(fused_states)
    block_q = min(256, triton.next_power_of_2(query_len))
    if block_table.ndim != 2 or block_table.shape[0] not in (requests, requests * query_len):
        raise ValueError("kPool block table must have one row per request or query token")
    table_stride = block_table.stride(0) * (query_len if block_table.shape[0] != requests else 1)
    if query_len > 8:
        _store_fused_pools_kernel[(requests, triton.cdiv(query_len, rate))](
            valid_rows.reshape(-1).contiguous(),
            positions.reshape(-1).contiguous(),
            pooled,
            compressed_cache,
            read_tail_cache,
            block_table,
            tail_read_ids.reshape(-1).contiguous(),
            NUM_TAIL_SLOTS=tail_cache.shape[0],
            SNAPSHOT_TAIL_IO=snapshot_tail_io,
            NUM_POOL_BLOCKS=compressed_cache.shape[0],
            POOL_BLOCK_SIZE=compressed_cache.shape[1],
            TABLE_WIDTH=block_table.shape[1],
            TABLE_REQUEST_STRIDE=table_stride,
            TAIL_CAPACITY=tail_cache.shape[2],
            POOLED_ROWS=pooled.shape[1],
            QUERY_LEN=query_len,
            RATE=rate,
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
    _store_fused_kpool_kernel[(requests,)](
        raw_k.reshape(-1, head_dim).contiguous(),
        gate_scores.reshape(-1, head_dim).contiguous(),
        valid_rows.reshape(-1).contiguous(),
        tail_valid_rows.reshape(-1).contiguous(),
        positions.reshape(-1).contiguous(),
        pooled,
        states,
        compressed_cache,
        tail_cache,
        read_tail_cache,
        tail_read_ids.reshape(-1).contiguous(),
        tail_write_ids.reshape(-1).contiguous(),
        block_table,
        NUM_TAIL_SLOTS=tail_cache.shape[0],
        SNAPSHOT_TAIL_IO=snapshot_tail_io,
        NUM_POOL_BLOCKS=compressed_cache.shape[0],
        POOL_BLOCK_SIZE=compressed_cache.shape[1],
        TABLE_WIDTH=block_table.shape[1],
        TABLE_REQUEST_STRIDE=table_stride,
        TAIL_CAPACITY=tail_cache.shape[2],
        POOLED_ROWS=pooled.shape[1],
        QUERY_LEN=query_len,
        RATE=rate,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_TAIL=triton.next_power_of_2(tail_cache.shape[2]),
        BLOCK_Q=block_q,
        SMALL_QUERY=query_len <= 8,
        DISTINCT_TAIL_IO=tail_read_ids is not tail_write_ids,
    )


@triton.jit
def _prepare_fused_kpool_history_kernel(
    positions_ptr,
    valid_ptr,
    ids_ptr,
    tail_ptr,
    states_ptr,
    starts_ptr,
    lengths_ptr,
    QUERY_LEN: tl.constexpr,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TAIL_CAPACITY: tl.constexpr,
    TAIL_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    request = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    members = tl.arange(0, RATE)
    offsets = tl.arange(0, BLOCK_Q)
    position = tl.maximum(tl.load(positions_ptr + request * QUERY_LEN), 0).to(tl.int64)
    start = position % RATE
    length = tl.sum(tl.load(valid_ptr + request * QUERY_LEN + offsets, offsets < QUERY_LEN, other=0).to(tl.int32), 0)
    tail_id = tl.load(ids_ptr + request).to(tl.int64)
    tail_id = tl.minimum(tl.maximum(tail_id, 0), TAIL_SLOTS - 1)
    rows = (position - start + members) % TAIL_CAPACITY
    address = (tail_id * 2 * TAIL_CAPACITY + rows[:, None]) * HEAD_DIM + dims[None, :]
    mask = dims[None, :] < HEAD_DIM
    keys = tl.load(tail_ptr + address, mask, other=0)
    gates = tl.load(tail_ptr + address + TAIL_CAPACITY * HEAD_DIM, mask, other=0)
    # CANN rows concatenate K and gate rather than interleaving features.
    # Initialize request scratch together with its history. Only the middle
    # block holds the old tail; CANN may read the other blocks before writing.
    for block in tl.static_range(3):
        destination = ((request * 3 + block) * RATE + members[:, None]) * (2 * HEAD_DIM) + dims[None, :]
        tl.store(states_ptr + destination, keys if block == 1 else tl.full((RATE, BLOCK_D), 0, tl.float32), mask)
        tl.store(
            states_ptr + destination + HEAD_DIM, gates if block == 1 else tl.full((RATE, BLOCK_D), 0, tl.float32), mask
        )
    tl.store(starts_ptr + request, start.to(tl.int32))
    tl.store(lengths_ptr + 2 * request, 0)
    tl.store(lengths_ptr + 2 * request + 1, tl.maximum(length, 1))


@triton.jit
def _extract_fused_kpool_tail_kernel(
    states_ptr,
    starts_ptr,
    lengths_ptr,
    tails_ptr,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    members = tl.arange(0, RATE)
    start = tl.load(starts_ptr + request)
    length = tl.load(lengths_ptr + 2 * request + 1)
    block = tl.where(start + length <= RATE, 1, 2)
    source = ((request * 3 + block) * RATE + members[:, None]) * 2 * HEAD_DIM + dims[None, :]
    mask = dims[None, :] < HEAD_DIM
    keys = tl.load(states_ptr + source, mask, other=0)
    gates = tl.load(states_ptr + source + HEAD_DIM, mask, other=0)
    destination = (request * 2 * RATE + members[:, None]) * HEAD_DIM + dims[None, :]
    tl.store(tails_ptr + destination, keys, mask)
    tl.store(tails_ptr + destination + RATE * HEAD_DIM, gates, mask)


@triton.jit
def _compact_fused_kpool_queries_kernel(
    hidden,
    lengths,
    compact,
    cu_seqlens,
    REQUESTS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    request = tl.program_id(0)
    feature = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    requests = tl.arange(0, BLOCK_B)
    all_lengths = tl.load(lengths + requests * 2 + 1, requests < REQUESTS, other=0)
    start = tl.sum(tl.where(requests < request, all_lengths, 0), 0)
    length = tl.load(lengths + request * 2 + 1)
    rows = tl.arange(0, BLOCK_Q)
    values = tl.load(
        hidden + (request * QUERY_LEN + rows[:, None]) * HIDDEN + feature[None, :],
        (rows[:, None] < length) & (feature[None, :] < HIDDEN),
        other=0,
    )
    tl.store(
        compact + (start + rows[:, None]) * HIDDEN + feature[None, :],
        values,
        (rows[:, None] < length) & (feature[None, :] < HIDDEN),
    )
    if request == REQUESTS - 1:
        # Compaction leaves an unused suffix. Keep it zero without a separate
        # fill launch, using only the last request so valid copies cannot race.
        suffix = start + length + tl.arange(0, BLOCK_B * BLOCK_Q)
        tl.store(
            compact + suffix[:, None] * HIDDEN + feature[None, :],
            0,
            (suffix[:, None] < REQUESTS * QUERY_LEN) & (feature[None, :] < HIDDEN),
        )
    # Finish vector copies before scalar length stores reuse local buffers.
    tl.debug_barrier()
    if tl.program_id(1) == 0:
        tl.store(cu_seqlens + request, start)
        if request == REQUESTS - 1:
            tl.store(cu_seqlens + REQUESTS, start + length)


@triton.jit
def _unpack_fused_kpool_kernel(
    pooled,
    starts,
    lengths,
    states,
    output,
    tails,
    REQUESTS: tl.constexpr,
    POOLED_ROWS: tl.constexpr,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    requests = tl.arange(0, BLOCK_B)
    all_starts = tl.load(starts + requests, requests < REQUESTS, other=0)
    all_lengths = tl.load(lengths + requests * 2 + 1, requests < REQUESTS, other=0)
    counts = (all_starts + all_lengths) // RATE
    offset = tl.sum(tl.where(requests < request, counts, 0), 0)
    start = tl.load(starts + request)
    length = tl.load(lengths + request * 2 + 1)
    count = (start + length) // RATE
    pools = tl.arange(0, BLOCK_P)
    dims = tl.arange(0, BLOCK_D)
    values = tl.load(
        pooled + (offset + pools[:, None]) * HEAD_DIM + dims[None, :],
        (pools[:, None] < count) & (dims[None, :] < HEAD_DIM),
        other=0,
    )
    tl.store(
        output + (request * POOLED_ROWS + pools[:, None]) * HEAD_DIM + dims[None, :],
        values,
        (pools[:, None] < POOLED_ROWS) & (dims[None, :] < HEAD_DIM),
    )
    members = tl.arange(0, RATE)
    block = tl.where(start + length <= RATE, 1, 2)
    source = ((request * 3 + block) * RATE + members[:, None]) * 2 * HEAD_DIM + dims[None, :]
    keys = tl.load(states + source, dims[None, :] < HEAD_DIM, other=0)
    gates = tl.load(states + source + HEAD_DIM, dims[None, :] < HEAD_DIM, other=0)
    destination = (request * 2 * RATE + members[:, None]) * HEAD_DIM + dims[None, :]
    tl.store(tails + destination, keys, dims[None, :] < HEAD_DIM)
    tl.store(tails + destination + RATE * HEAD_DIM, gates, dims[None, :] < HEAD_DIM)


@lru_cache(maxsize=128)
def _fused_kpool_batch_table(requests: int, width: int, device: torch.device) -> torch.Tensor:
    table = torch.full((requests, width), 2, device=device, dtype=torch.int32)
    table[:, 0] = 1
    return table + torch.arange(requests, device=device, dtype=torch.int32)[:, None] * 3


def prepare_fused_kpool_graph(
    hidden: torch.Tensor,
    key_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_read_ids: torch.Tensor,
    positions: torch.Tensor,
    valid_rows: torch.Tensor,
    query_len: int,
    rate: int,
    norm_epsilon: float,
    table: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batch fixed-shape request metadata while preserving CANN arithmetic."""
    requests = hidden.shape[0] // query_len
    head_dim = key_weight.shape[0]
    states = torch.empty((requests, 3, rate, 2 * head_dim), dtype=torch.float32, device=hidden.device)
    starts = torch.empty(requests, dtype=torch.int32, device=hidden.device)
    lengths = torch.empty((requests, 2), dtype=torch.int32, device=hidden.device)
    tails = torch.empty((requests, 2, rate, head_dim), dtype=torch.float32, device=hidden.device)
    _prepare_fused_kpool_history_kernel[(requests,)](
        positions.reshape(-1).contiguous(),
        valid_rows.reshape(-1).contiguous(),
        tail_read_ids.reshape(-1).contiguous(),
        tail_cache,
        states,
        starts,
        lengths,
        QUERY_LEN=query_len,
        RATE=rate,
        HEAD_DIM=head_dim,
        TAIL_CAPACITY=tail_cache.shape[2],
        TAIL_SLOTS=tail_cache.shape[0],
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_Q=triton.next_power_of_2(query_len),
    )
    if requests > 1:
        compact = torch.empty_like(hidden)
        cu_seqlens = torch.empty(requests + 1, dtype=torch.int32, device=hidden.device)
        _compact_fused_kpool_queries_kernel[(requests, triton.cdiv(hidden.shape[1], 256))](
            hidden,
            lengths,
            compact,
            cu_seqlens,
            REQUESTS=requests,
            QUERY_LEN=query_len,
            HIDDEN=hidden.shape[1],
            BLOCK_B=triton.next_power_of_2(requests),
            BLOCK_Q=triton.next_power_of_2(query_len),
            BLOCK_H=256,
        )
        batched = torch.ops.xllm_ops.key_pool(
            compact,
            key_weight,
            gate_weight,
            ape,
            states.view(requests * 3, rate, 2 * head_dim),
            _fused_kpool_batch_table(requests, table.shape[1], hidden.device),
            starts,
            norm_weight,
            norm_bias,
            cu_seqlens,
            rate,
            norm_epsilon,
        )
        pooled_rows = min(query_len, query_len // rate + 1)
        pooled = hidden.new_empty((requests, pooled_rows, head_dim))
        _unpack_fused_kpool_kernel[(requests,)](
            batched,
            starts,
            lengths,
            states,
            pooled,
            tails,
            REQUESTS=requests,
            POOLED_ROWS=pooled_rows,
            RATE=rate,
            HEAD_DIM=head_dim,
            BLOCK_B=triton.next_power_of_2(requests),
            BLOCK_P=triton.next_power_of_2(pooled_rows),
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
        return [(pooled[request], tails[request]) for request in range(requests)]
    pooled = [
        torch.ops.xllm_ops.key_pool(
            hidden[request * query_len : (request + 1) * query_len],
            key_weight,
            gate_weight,
            ape,
            states[request],
            table,
            starts[request : request + 1],
            norm_weight,
            norm_bias,
            lengths[request],
            rate,
            norm_epsilon,
        )
        for request in range(requests)
    ]
    _extract_fused_kpool_tail_kernel[(requests,)](
        states,
        starts,
        lengths,
        tails,
        RATE=rate,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return [(value, tails[request]) for request, value in enumerate(pooled)]
