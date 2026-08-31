# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FakeTensor implementations for the MUSA ``xllm_ops`` operators."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _is_registered(qualname: str) -> bool:
    namespace, op_name = qualname.split("::", 1)
    library = getattr(torch.ops, namespace, None)
    return library is not None and hasattr(library, op_name)


def _register_fake(qualname: str, fake_impl: Callable) -> None:
    if not _is_registered(qualname):
        raise RuntimeError(
            f"operator '{qualname}' is not registered; "
            "xllm/core/kernels/musa/musa_ops_library.cpp must define it before "
            "its fake implementation can be attached"
        )
    torch.library.register_fake(qualname)(fake_impl)


def _rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor,
) -> torch.Tensor:
    del input, weight, eps
    return output


def _fused_add_rms_norm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    return input, residual


def _gemma_rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor,
) -> torch.Tensor:
    del input, weight, eps
    return output


def _fused_add_gemma_rms_norm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    return input, residual


def _mul_sigmoid_gate_inplace_fake(
    out: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    del gate
    return out


def _silu_and_mul_fake(
    input: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    del input
    return output


def _fused_qk_norm_rope_fake(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    interleaved: bool,
    position_ids: torch.Tensor,
    k_head_offset: int = 0,
) -> torch.Tensor:
    del (
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        interleaved,
        position_ids,
        k_head_offset,
    )
    return qkv


def _reshape_paged_cache_fake(
    slot_mapping: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    del slot_mapping, keys, values, value_cache
    return key_cache


def _update_decode_graph_metadata_fake(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    dst_tokens: torch.Tensor,
    dst_positions: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    dst_kv_seq_lens: torch.Tensor,
    dst_kv_seq_lens_delta: torch.Tensor,
    dst_paged_kv_indptr: torch.Tensor,
    dst_paged_kv_indices: torch.Tensor,
    dst_paged_kv_last_page_len: torch.Tensor,
    padded_num_tokens: int,
) -> torch.Tensor:
    del (
        tokens,
        positions,
        slot_mapping,
        kv_seq_lens,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        dst_positions,
        dst_slot_mapping,
        dst_kv_seq_lens,
        dst_kv_seq_lens_delta,
        dst_paged_kv_indptr,
        dst_paged_kv_indices,
        dst_paged_kv_last_page_len,
        padded_num_tokens,
    )
    return dst_tokens


def _update_fa3_graph_metadata_fake(
    kv_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    dst_kv_seq_lens: torch.Tensor,
    dst_block_table: torch.Tensor,
    actual_batch_size: int,
) -> torch.Tensor:
    del kv_seq_lens, block_table, dst_block_table, actual_batch_size
    return dst_kv_seq_lens


_register_fake("xllm_ops::rms_norm", _rms_norm_fake)
_register_fake("xllm_ops::fused_add_rms_norm", _fused_add_rms_norm_fake)
_register_fake("xllm_ops::silu_and_mul", _silu_and_mul_fake)
_register_fake("xllm_ops::fused_qk_norm_rope", _fused_qk_norm_rope_fake)


def _fa3_decode_scheduler_metadata_fake(
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim_qk: int,
    head_dim_vo: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    num_splits: int,
    scheduler_metadata: torch.Tensor,
) -> torch.Tensor:
    del (
        cu_seqlens_q,
        seqused_k,
        batch_size,
        num_heads_q,
        num_heads_kv,
        head_dim_qk,
        head_dim_vo,
        max_seqlen_q,
        max_seqlen_k,
        window_size_left,
        window_size_right,
        num_splits,
    )
    return scheduler_metadata


def _fa3_decode_fake(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    max_seqlen_q: int,
    window_left: int,
    window_right: int,
    sm_scale: float,
    num_splits: int,
    output: torch.Tensor,
    output_lse: torch.Tensor,
) -> torch.Tensor:
    del (
        k_cache,
        v_cache,
        cu_seqlens_q,
        seqused_k,
        page_table,
        scheduler_metadata,
        max_seqlen_q,
        window_left,
        window_right,
        sm_scale,
        num_splits,
        output_lse,
    )
    del query
    return output


def _fa3_prefill_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_left: int,
    window_right: int,
    sm_scale: float,
    output: torch.Tensor,
    output_lse: torch.Tensor,
) -> torch.Tensor:
    del (
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_left,
        window_right,
        sm_scale,
        output_lse,
    )
    return output


_register_fake("xllm_ops::reshape_paged_cache", _reshape_paged_cache_fake)
_register_fake(
    "xllm_ops::update_decode_graph_metadata", _update_decode_graph_metadata_fake
)
_register_fake(
    "xllm_ops::update_fa3_graph_metadata", _update_fa3_graph_metadata_fake
)
_register_fake(
    "xllm_ops::fa3_decode_scheduler_metadata",
    _fa3_decode_scheduler_metadata_fake,
)
_register_fake("xllm_ops::fa3_decode", _fa3_decode_fake)
_register_fake("xllm_ops::fa3_prefill", _fa3_prefill_fake)


def _block_fp8_linear_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
    output: torch.Tensor,
) -> torch.Tensor:
    del input, weight, weight_scale_inv, block_n, block_k
    return output


def _block_fp8_linear_quantized_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
    output: torch.Tensor,
) -> torch.Tensor:
    del input, input_scale, weight, weight_scale_inv, block_n, block_k
    return output


def _per_token_group_quant_fp8_fake(
    input: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = input.size(0)
    hidden = input.size(-1)
    output_q = input.new_empty((num_rows, hidden), dtype=torch.float8_e4m3fn)
    output_s = input.new_empty(
        (num_rows, hidden // group_size), dtype=torch.float32
    )
    return output_q, output_s


def _fused_swiglu_quant_fp8_fake(
    input: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = input.size(0)
    intermediate = input.size(-1) // 2
    output_q = input.new_empty(
        (num_rows, intermediate), dtype=torch.float8_e4m3fn
    )
    output_s = input.new_empty(
        (num_rows, intermediate // group_size), dtype=torch.float32
    )
    return output_q, output_s


def _causal_conv1d_prefill_fake(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    del weight, conv_state, state_indices, has_initial_state, query_start_loc
    return torch.empty_like(value)


def _causal_conv1d_decode_fake(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    del weight, conv_state, state_indices
    return torch.empty_like(value)


def _causal_conv1d_mtp_verify_fake(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_key_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    del (
        weight,
        conv_state,
        cache_indices,
        num_accepted_tokens,
        num_key_heads,
        key_head_dim,
        value_head_dim,
    )
    return torch.empty_like(mixed_qkv)


def _fused_gdn_prefill_post_conv_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_key_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    del b, dt_bias
    num_tokens = mixed_qkv.shape[0]
    num_value_heads = a_log.numel()
    query = mixed_qkv.new_empty(num_tokens, num_key_heads, key_head_dim)
    key = torch.empty_like(query)
    value = mixed_qkv.new_empty(num_tokens, num_value_heads, value_head_dim)
    gate = a.new_empty(num_tokens, num_value_heads, dtype=torch.float32)
    beta = torch.empty_like(gate)
    return query, key, value, gate, beta


def _chunk_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q, k, g, beta, cu_seqlens, backend
    return torch.empty_like(v), torch.empty_like(initial_state)


def _mate_gated_delta_rule_decode_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    num_k_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    del a, b, a_log, dt_bias, state_indices, scale, num_k_heads, key_head_dim
    return mixed_qkv.new_empty(
        mixed_qkv.shape[0], initial_state.shape[1], value_head_dim
    )


def _fused_recurrent_gated_delta_rule_packed_decode_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    num_k_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    del a, b, a_log, dt_bias, state_indices, scale, num_k_heads, key_head_dim
    return mixed_qkv.new_empty(
        mixed_qkv.shape[0], initial_state.shape[1], value_head_dim
    )


def _fused_gdn_mtp_checkpoint_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    logical_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    checkpoint_stride: int,
    scale: float,
    num_key_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    del (
        a,
        b,
        dt_bias,
        ssm_state,
        logical_state_indices,
        num_accepted_tokens,
        checkpoint_stride,
        scale,
        num_key_heads,
        key_head_dim,
    )
    return mixed_qkv.new_empty(mixed_qkv.shape[0], a_log.numel(), value_head_dim)


def _rms_norm_gated_fake(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor,
) -> torch.Tensor:
    del value, gate, weight, eps
    return output


def _python_prefill_piecewise_begin_fake(tokens: torch.Tensor) -> None:
    del tokens
    return None


def _python_prefill_piecewise_end_fake(
    tokens: torch.Tensor,
) -> tuple[int, int, int]:
    del tokens
    return 0, 0, 0


def _python_prefill_piecewise_replay_fake(
    tokens: torch.Tensor,
    handle: int,
    gdn_cu_seq_lens: torch.Tensor,
    gdn_kkt_cu_seq_lens: torch.Tensor,
    actual_num_tokens: int,
    gdn_kkt_num_tokens: int,
) -> None:
    del tokens, handle, gdn_cu_seq_lens, gdn_kkt_cu_seq_lens
    del actual_num_tokens, gdn_kkt_num_tokens
    return None


_register_fake("xllm_ops::gemma_rms_norm", _gemma_rms_norm_fake)
_register_fake(
    "xllm_ops::fused_add_gemma_rms_norm", _fused_add_gemma_rms_norm_fake
)
_register_fake(
    "xllm_ops::mul_sigmoid_gate_inplace", _mul_sigmoid_gate_inplace_fake
)
_register_fake("xllm_ops::block_fp8_linear", _block_fp8_linear_fake)
_register_fake(
    "xllm_ops::block_fp8_linear_quantized", _block_fp8_linear_quantized_fake
)
_register_fake("xllm_ops::fused_swiglu_quant_fp8", _fused_swiglu_quant_fp8_fake)
_register_fake(
    "xllm_ops::per_token_group_quant_fp8", _per_token_group_quant_fp8_fake
)
_register_fake("xllm_ops::causal_conv1d_prefill", _causal_conv1d_prefill_fake)
_register_fake("xllm_ops::causal_conv1d_decode", _causal_conv1d_decode_fake)
_register_fake(
    "xllm_ops::causal_conv1d_mtp_verify", _causal_conv1d_mtp_verify_fake
)
_register_fake(
    "xllm_ops::fused_gdn_prefill_post_conv", _fused_gdn_prefill_post_conv_fake
)
_register_fake("xllm_ops::chunk_gated_delta_rule", _chunk_gated_delta_rule_fake)
_register_fake(
    "xllm_ops::fused_recurrent_gated_delta_rule_packed_decode",
    _fused_recurrent_gated_delta_rule_packed_decode_fake,
)
_register_fake(
    "xllm_ops::mate_gated_delta_rule_decode",
    _mate_gated_delta_rule_decode_fake,
)
_register_fake(
    "xllm_ops::fused_gdn_mtp_checkpoint", _fused_gdn_mtp_checkpoint_fake
)
_register_fake("xllm_ops::rms_norm_gated", _rms_norm_gated_fake)
_register_fake(
    "xllm_ops::python_prefill_piecewise_begin",
    _python_prefill_piecewise_begin_fake,
)
_register_fake(
    "xllm_ops::python_prefill_piecewise_end",
    _python_prefill_piecewise_end_fake,
)
_register_fake(
    "xllm_ops::python_prefill_piecewise_replay",
    _python_prefill_piecewise_replay_fake,
)
