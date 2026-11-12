/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <glog/logging.h>
#include <torch/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/kernels/musa/deepseek_v41_pair_pool.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

__device__ int64_t state_slot(int64_t position,
                              const int64_t* table,
                              int64_t table_size,
                              int64_t block_size) {
  const int64_t block = table[(position / block_size) % table_size];
  return (block < 0 ? 0 : block) * 32 + position % 32;
}

// Adapt JD SGLang pair_pool_decode.py to xLLM's physical-page history.
// Separate products/addition preserve torch pool_pairs rounding boundaries.
__global__ void pair_pool_kernel(const float* values,
                                 const float* scores,
                                 const int64_t* table,
                                 const float* value_state,
                                 const float* score_state,
                                 c10::BFloat16* output,
                                 int64_t* positions,
                                 int64_t start,
                                 int64_t dimension,
                                 int64_t table_size,
                                 int64_t block_size) {
  const int64_t group = blockIdx.x;
  const int64_t position = (start / 2 + group) * 2;
  const int64_t first_row = position - start;
  if (threadIdx.x == 0) {
    positions[group] = position;
  }
  for (int64_t col = threadIdx.x; col < dimension; col += blockDim.x) {
    const int64_t old_slot =
        state_slot(position, table, table_size, block_size);
    const float v0 = first_row >= 0 ? values[first_row * dimension + col]
                                    : value_state[old_slot * dimension + col];
    const float s0 = first_row >= 0 ? scores[first_row * dimension + col]
                                    : score_state[old_slot * dimension + col];
    const float v1 = values[(first_row + 1) * dimension + col];
    const float s1 = scores[(first_row + 1) * dimension + col];
    const float maximum = fmaxf(s0, s1);
    const float e0 = expf(s0 - maximum);
    const float e1 = expf(s1 - maximum);
    const float denominator = __fadd_rn(e0, e1);
    const float p0 = __fmul_rn(v0, __fdiv_rn(e0, denominator));
    const float p1 = __fmul_rn(v1, __fdiv_rn(e1, denominator));
    output[group * dimension + col] =
        static_cast<c10::BFloat16>(__fadd_rn(p0, p1));
  }
}

__global__ void park_pair_tail_kernel(const float* values,
                                      const float* scores,
                                      const int64_t* table,
                                      float* value_state,
                                      float* score_state,
                                      int64_t start,
                                      int64_t count,
                                      int64_t dimension,
                                      int64_t table_size,
                                      int64_t block_size) {
  const int64_t first = count > 32 ? count - 32 : 0;
  const int64_t row = first + blockIdx.x;
  const int64_t position = start + row;
  if (position % 2 != 0) {
    return;
  }
  const int64_t slot = state_slot(position, table, table_size, block_size);
  for (int64_t col = threadIdx.x; col < dimension; col += blockDim.x) {
    value_state[slot * dimension + col] = values[row * dimension + col];
    score_state[slot * dimension + col] = scores[row * dimension + col];
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_pair_pool(
    const torch::Tensor& values,
    const torch::Tensor& scores,
    const torch::Tensor& table,
    torch::Tensor& value_state,
    torch::Tensor& score_state,
    int64_t start,
    int64_t block_size) {
  CHECK(values.defined() && scores.defined() && table.defined());
  CHECK(values.device().type() == torch::kPrivateUse1);
  CHECK_EQ(values.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(values.sizes(), scores.sizes());
  CHECK_EQ(values.dim(), 2);
  CHECK(values.size(1) == 128 || values.size(1) == 512);
  CHECK(values.is_contiguous() && scores.is_contiguous());
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), 1);
  CHECK_GT(table.size(1), 0);
  CHECK_GE(start, 0);
  CHECK_GT(block_size, 0);
  CHECK_EQ(block_size % 32, 0);
  for (const torch::Tensor& state : {value_state, score_state}) {
    CHECK_EQ(state.device(), values.device());
    CHECK_EQ(state.scalar_type(), torch::kFloat32);
    CHECK_EQ(state.dim(), 3);
    CHECK_GT(state.size(0), 0);
    CHECK_EQ(state.size(1), 32);
    CHECK_EQ(state.size(2), values.size(1));
    CHECK(state.is_contiguous());
  }
  CHECK_EQ(value_state.sizes(), score_state.sizes());
  CHECK_EQ(scores.device(), values.device());
  CHECK_EQ(table.device(), values.device());
  CHECK(table.scalar_type() == torch::kInt32 ||
        table.scalar_type() == torch::kInt64);
  torch::DeviceGuard guard(values.device());
  const torch::Tensor pages = table.to(torch::kInt64).contiguous();
  const int64_t count = values.size(0);
  const int64_t groups = (start + count) / 2 - start / 2;
  torch::Tensor pooled = torch::empty({groups, values.size(1)},
                                      values.options().dtype(torch::kBFloat16));
  torch::Tensor positions =
      torch::empty({groups}, table.options().dtype(torch::kInt64));
  if (groups > 0) {
    pair_pool_kernel<<<static_cast<uint32_t>(groups),
                       256,
                       0,
                       current_musa_stream(values.device())>>>(
        values.data_ptr<float>(),
        scores.data_ptr<float>(),
        pages.data_ptr<int64_t>(),
        value_state.data_ptr<float>(),
        score_state.data_ptr<float>(),
        pooled.data_ptr<c10::BFloat16>(),
        positions.data_ptr<int64_t>(),
        start,
        values.size(1),
        pages.numel(),
        block_size);
    check_musa_kernel_launch("V4.1 ratio-2 pair pool");
  }
  if (count > 0) {
    // Same-stream second launch keeps every old-state read before writes.
    park_pair_tail_kernel<<<static_cast<uint32_t>(std::min<int64_t>(32, count)),
                            256,
                            0,
                            current_musa_stream(values.device())>>>(
        values.data_ptr<float>(),
        scores.data_ptr<float>(),
        pages.data_ptr<int64_t>(),
        value_state.data_ptr<float>(),
        score_state.data_ptr<float>(),
        start,
        count,
        values.size(1),
        pages.numel(),
        block_size);
    check_musa_kernel_launch("V4.1 ratio-2 park tail");
  }
  return {pooled, positions};
}

}  // namespace xllm::kernel::musa
