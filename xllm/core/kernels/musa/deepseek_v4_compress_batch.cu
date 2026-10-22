/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kCompressThreads = 256;

// Keep the scalar reference TU untouched. Only request pointer offsets differ
// from its per-dimension recurrence; token/row/coefficient accumulation order
// and the unused-output zeroing are intentionally identical.
template <typename Scalar, typename Position, int32_t Ratio>
__global__ void __launch_bounds__(kCompressThreads, 1)
    compress_batch_graph_kernel(float* __restrict__ kv_state,
                                float* __restrict__ score_state,
                                const Scalar* __restrict__ kv_steps,
                                int64_t kv_stride,
                                const Scalar* __restrict__ score_steps,
                                int64_t score_stride,
                                const float* __restrict__ ape,
                                const Position* __restrict__ positions,
                                Scalar* __restrict__ output,
                                int64_t tokens,
                                int64_t head_dim,
                                int64_t capacity) {
  constexpr int32_t kCoeff = Ratio == 4 ? 2 : 1;
  constexpr int32_t kRows = Ratio * kCoeff;
  const int64_t width = kCoeff * head_dim;
  const int64_t sequence = blockIdx.y;
  kv_state += sequence * kRows * width;
  score_state += sequence * kRows * width;
  kv_steps += sequence * tokens * kv_stride;
  score_steps += sequence * tokens * score_stride;
  output += sequence * capacity * head_dim;
  const int64_t start = static_cast<int64_t>(positions[sequence * tokens]);
  for (int64_t dim = blockIdx.x * blockDim.x + threadIdx.x; dim < head_dim;
       dim += gridDim.x * blockDim.x) {
    int64_t output_row = 0;
    for (int64_t token = 0; token < tokens && start >= 0; ++token) {
      const int64_t position = (start + token) % Ratio;
      const int64_t row = (kCoeff - 1) * Ratio + position;
      for (int32_t coeff = 0; coeff < kCoeff; ++coeff) {
        const int64_t col = coeff * head_dim + dim;
        kv_state[row * width + col] =
            musa_convert_to_float(kv_steps[token * kv_stride + col]);
        score_state[row * width + col] =
            musa_convert_to_float(score_steps[token * score_stride + col]) +
            ape[position * width + col];
      }
      if (position != Ratio - 1) {
        continue;
      }
      float max_score = -INFINITY;
      for (int32_t candidate = 0; candidate < kRows; ++candidate) {
        const int64_t col = (candidate / Ratio) * head_dim + dim;
        max_score = fmaxf(max_score, score_state[candidate * width + col]);
      }
      float denominator = 0.0f;
      float weighted_sum = 0.0f;
      for (int32_t candidate = 0; candidate < kRows; ++candidate) {
        const int64_t col = (candidate / Ratio) * head_dim + dim;
        const int64_t offset = candidate * width + col;
        const float weight = expf(score_state[offset] - max_score);
        denominator += weight;
        weighted_sum += weight * kv_state[offset];
      }
      if (output_row < capacity) {
        output[output_row * head_dim + dim] =
            static_cast<Scalar>(weighted_sum / denominator);
      }
      ++output_row;
      if constexpr (Ratio == 4) {
        for (int32_t history_row = 0; history_row < Ratio; ++history_row) {
          for (int32_t coeff = 0; coeff < kCoeff; ++coeff) {
            const int64_t col = coeff * head_dim + dim;
            const int64_t dst = history_row * width + col;
            const int64_t src = (Ratio + history_row) * width + col;
            kv_state[dst] = kv_state[src];
            score_state[dst] = score_state[src];
          }
        }
      }
    }
    for (; output_row < capacity; ++output_row) {
      output[output_row * head_dim + dim] = static_cast<Scalar>(0.0f);
    }
  }
}

}  // namespace

torch::Tensor deepseek_v4_compress_decode_batch_graph(
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const torch::Tensor& kv_steps,
    const torch::Tensor& score_steps,
    const torch::Tensor& ape,
    int64_t compress_ratio,
    int64_t head_dim,
    int64_t query_width,
    const torch::Tensor& positions,
    const torch::Tensor& output) {
  CHECK(compress_ratio == 4 || compress_ratio == 128);
  CHECK(head_dim > 0 && query_width > 0);
  CHECK(kv_state.dim() == 3 && kv_state.size(0) > 1 && kv_state.size(0) <= 16);
  const int64_t sequences = kv_state.size(0);
  const int64_t coeff = compress_ratio == 4 ? 2 : 1;
  const int64_t width = coeff * head_dim;
  const int64_t capacity = (query_width + compress_ratio - 1) / compress_ratio;
  CHECK(kv_state.sizes() ==
        torch::IntArrayRef({sequences, compress_ratio * coeff, width}));
  CHECK(score_state.sizes() == kv_state.sizes());
  CHECK(kv_steps.sizes() ==
        torch::IntArrayRef({sequences * query_width, width}));
  CHECK(score_steps.sizes() == kv_steps.sizes());
  CHECK(kv_steps.stride(1) == 1 && score_steps.stride(1) == 1);
  CHECK(score_steps.scalar_type() == kv_steps.scalar_type());
  CHECK(kv_steps.scalar_type() == torch::kFloat32 ||
        kv_steps.scalar_type() == torch::kFloat16 ||
        kv_steps.scalar_type() == torch::kBFloat16);
  CHECK(ape.sizes() == torch::IntArrayRef({compress_ratio, width}));
  CHECK(kv_state.scalar_type() == torch::kFloat32 &&
        score_state.scalar_type() == torch::kFloat32 &&
        ape.scalar_type() == torch::kFloat32);
  CHECK(positions.dim() == 1 && positions.numel() == sequences * query_width);
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  CHECK(output.sizes() == torch::IntArrayRef({sequences * capacity, head_dim}));
  CHECK(output.scalar_type() == kv_steps.scalar_type());
  CHECK(!kv_steps.device().is_cpu());
  for (const torch::Tensor& tensor :
       {kv_state, score_state, ape, positions, output}) {
    CHECK(tensor.is_contiguous());
    CHECK(tensor.device() == kv_steps.device());
  }
  CHECK(score_steps.device() == kv_steps.device());
  const XLLMOptionalDeviceGuard guard(kv_steps.device());
  // Match the frozen scalar cache route. During native MUSA Graph capture,
  // the CUDA-compat accessor can still select the mapped caller stream and
  // let a kernel escape capture; eager keeps its established mapped ordering.
  const musaStream_t stream = is_stream_capturing()
                                  ? c10::musa::getCurrentMUSAStream().stream()
                                  : current_musa_stream(kv_steps.device());
  const dim3 grid((head_dim + kCompressThreads - 1) / kCompressThreads,
                  sequences);
  DISPATCH_FLOATING_TYPES(
      kv_steps.scalar_type(), "dsv4_compress_batch_graph", [&] {
        const auto launch = [&](auto position_tag, auto ratio_tag) {
          using Position = decltype(position_tag);
          constexpr int32_t kRatio = decltype(ratio_tag)::value;
          compress_batch_graph_kernel<scalar_t, Position, kRatio>
              <<<grid, kCompressThreads, 0, stream>>>(
                  kv_state.data_ptr<float>(),
                  score_state.data_ptr<float>(),
                  kv_steps.data_ptr<scalar_t>(),
                  kv_steps.stride(0),
                  score_steps.data_ptr<scalar_t>(),
                  score_steps.stride(0),
                  ape.data_ptr<float>(),
                  positions.data_ptr<Position>(),
                  output.data_ptr<scalar_t>(),
                  query_width,
                  head_dim,
                  capacity);
        };
        const auto dispatch_ratio = [&](auto position_tag) {
          if (compress_ratio == 4) {
            launch(position_tag, std::integral_constant<int32_t, 4>{});
          } else {
            launch(position_tag, std::integral_constant<int32_t, 128>{});
          }
        };
        if (positions.scalar_type() == torch::kInt32) {
          dispatch_ratio(int32_t{0});
        } else {
          dispatch_ratio(int64_t{0});
        }
      });
  check_musa_kernel_launch("dsv4 batched compression");
  return output;
}

}  // namespace xllm::kernel::musa
