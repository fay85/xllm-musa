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

"""Source-built MP31 routed GEMM for the native Qwen3.5 decode path.

Adapted from SGLang's fused_moe_kernel at commit
1519acf37c23f2189adb93f57ca9cd2db1bebf18 (Apache-2.0), itself adapted
from vLLM. Only contiguous BF16 and block-scaled FP8 with N/K multiples of
128 are supported here.
The 22 runtime arguments match the native MUSA launch contract.
"""

import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    a_scale_ptr: tl.tensor,
    b_scale_ptr: tl.tensor,
    topk_weights_ptr: tl.tensor,
    sorted_token_ids_ptr: tl.tensor,
    expert_ids_ptr: tl.tensor,
    num_tokens_post_padded_ptr: tl.tensor,
    output_size: tl.tensor,
    input_size: tl.tensor,
    max_padded_tokens: tl.tensor,
    num_valid_tokens: tl.tensor,
    stride_am: tl.tensor,
    stride_be: tl.tensor,
    stride_bn: tl.tensor,
    stride_bias_e: tl.tensor,
    stride_bias_n: tl.tensor,
    stride_cm: tl.tensor,
    stride_asm: tl.tensor,
    stride_bse: tl.tensor,
    stride_bsn: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_FP8: tl.constexpr,
) -> None:
    tl.static_assert(128 % BLOCK_SIZE_N == 0 and 128 % BLOCK_SIZE_K == 0)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_padded_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    padded_tokens = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= padded_tokens:
        return

    sorted_rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.load(sorted_token_ids_ptr + sorted_rows).to(tl.int64)
    valid_rows = (rows >= 0) & (rows < num_valid_tokens)
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    columns = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    weight_columns = tl.max_contiguous(tl.multiple_of(columns % output_size, BLOCK_SIZE_N), BLOCK_SIZE_N)
    reduction = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] // TOP_K * stride_am + reduction[None, :]
    b_ptrs = b_ptr + expert * stride_be + reduction[:, None] + weight_columns[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for k_start in range(0, input_size, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=valid_rows[:, None], other=0.0)
        b = tl.load(b_ptrs)
        if USE_FP8:
            group = k_start // 128
            a_scale = tl.load(a_scale_ptr + rows // TOP_K * stride_asm + group, mask=valid_rows, other=0.0)
            # Each N tile lies inside one 128-column quantization block.
            b_scale = tl.load(
                b_scale_ptr + expert * stride_bse + pid_n * BLOCK_SIZE_N // 128 * stride_bsn + group,
            )
            accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + rows, mask=valid_rows, other=0.0)
        accumulator *= weights[:, None]

    output_ptrs = c_ptr + rows[:, None] * stride_cm + columns[None, :]
    tl.store(
        output_ptrs,
        accumulator.to(tl.bfloat16),
        mask=valid_rows[:, None] & (columns[None, :] < output_size),
    )
