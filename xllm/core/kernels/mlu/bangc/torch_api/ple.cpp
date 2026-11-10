/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernels/mlu/bangc/include/ple.h"

#include <ATen/DeviceGuard.h>
#include <ATen/MemoryOverlap.h>
#include <glog/logging.h>

#include <cstdint>
#include <initializer_list>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/ple/ple_launchers.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {
namespace {
constexpr int64_t kH = 2560;
constexpr int64_t kD = 10240;
constexpr int64_t kN = 12800;
constexpr int64_t kQueries = 4;

struct TensorArgument {
  const torch::Tensor* tensor;
  const char* name;
};

int32_t checked_rows(const torch::Tensor& tensor,
                     int64_t columns,
                     const char* name) {
  CHECK(detail::check_tensor(tensor, torch::kBFloat16))
      << name << " must be contiguous BF16 on MLU";
  CHECK_EQ(tensor.dim(), 2) << name << " must be a matrix";
  const int64_t rows = tensor.size(0);
  CHECK(rows == 4 || rows == 8 || rows == 16 || rows == 32 || rows == 64 ||
        rows == 128)
      << "PLE requires Q4 and batch in {1,2,4,8,16,32}";
  CHECK(tensor.sizes() == torch::IntArrayRef({rows, columns}))
      << name << " has an unexpected shape";
  return static_cast<int32_t>(rows);
}

void check_disjoint(std::initializer_list<TensorArgument> mutated,
                    std::initializer_list<TensorArgument> inputs) {
  for (auto left = mutated.begin(); left != mutated.end(); ++left) {
    for (auto right = left + 1; right != mutated.end(); ++right) {
      CHECK(torch::get_overlap_status(*left->tensor, *right->tensor) ==
            torch::MemOverlapStatus::No)
          << left->name << " must not overlap " << right->name;
    }
    for (const TensorArgument& input : inputs) {
      CHECK(torch::get_overlap_status(*left->tensor, *input.tensor) ==
            torch::MemOverlapStatus::No)
          << left->name << " must not overlap " << input.name;
    }
  }
}

void check_projection_weights(const torch::Tensor& weight,
                              const torch::Tensor& packed_weight) {
  CHECK(detail::check_tensor(weight, torch::kBFloat16))
      << "weight must have the expected dtype/device and be contiguous";
  CHECK(weight.sizes() == torch::IntArrayRef({kN, kH}))
      << "weight has an unexpected shape";
  const torch::Device device = weight.device();
  CHECK(detail::check_tensor(packed_weight, torch::kBFloat16, device))
      << "packed_weight must have the expected dtype/device and be contiguous";
  CHECK(packed_weight.sizes() == torch::IntArrayRef({kN, kH}))
      << "packed_weight has an unexpected shape";
  check_disjoint({{&packed_weight, "packed_weight"}}, {{&weight, "weight"}});
}

void check_head_weights(const torch::Tensor& key_scale,
                        const torch::Tensor& query_scale,
                        const torch::Tensor& conv_scale,
                        const torch::Tensor& affine_pair) {
  CHECK(detail::check_tensor(key_scale, torch::kFloat32))
      << "key_scale must have the expected dtype/device and be contiguous";
  CHECK(key_scale.sizes() == torch::IntArrayRef({kD}))
      << "key_scale has an unexpected shape";
  const torch::Device device = key_scale.device();
  CHECK(detail::check_tensor(query_scale, torch::kFloat32, device))
      << "query_scale must have the expected dtype/device and be contiguous";
  CHECK(query_scale.sizes() == torch::IntArrayRef({kD}))
      << "query_scale has an unexpected shape";
  CHECK(detail::check_tensor(conv_scale, torch::kFloat32, device))
      << "conv_scale must have the expected dtype/device and be contiguous";
  CHECK(conv_scale.sizes() == torch::IntArrayRef({kD}))
      << "conv_scale has an unexpected shape";
  CHECK(detail::check_tensor(affine_pair, torch::kBFloat16, device))
      << "affine_pair must have the expected dtype/device and be contiguous";
  CHECK(affine_pair.sizes() == torch::IntArrayRef({4, 2, kH}))
      << "affine_pair has an unexpected shape";
  check_disjoint({{&affine_pair, "affine_pair"}},
                 {{&key_scale, "key_scale"},
                  {&query_scale, "query_scale"},
                  {&conv_scale, "conv_scale"}});
}

int32_t check_projection(const torch::Tensor& embedding,
                         const torch::Tensor& packed_weight,
                         const torch::Tensor& kv) {
  const int32_t rows = checked_rows(embedding, kH, "embedding");
  const torch::Device device = embedding.device();
  CHECK(detail::check_tensor(packed_weight, torch::kBFloat16, device))
      << "packed_weight must have the expected dtype/device and be contiguous";
  CHECK(packed_weight.sizes() == torch::IntArrayRef({kN, kH}))
      << "packed_weight has an unexpected shape";
  CHECK(detail::check_tensor(kv, torch::kBFloat16, device))
      << "kv must have the expected dtype/device and be contiguous";
  CHECK(kv.sizes() == torch::IntArrayRef({rows, kN}))
      << "kv has an unexpected shape";
  check_disjoint(
      {{&kv, "kv"}},
      {{&embedding, "embedding"}, {&packed_weight, "packed_weight"}});
  return rows;
}

int32_t check_head(const torch::Tensor& kv,
                   const torch::Tensor& hidden,
                   const torch::Tensor& affine_pair,
                   const torch::Tensor& values,
                   const torch::Tensor& normalized) {
  const int32_t rows = checked_rows(kv, kN, "kv");
  const torch::Device device = kv.device();
  CHECK(detail::check_tensor(hidden, torch::kBFloat16, device))
      << "hidden must have the expected dtype/device and be contiguous";
  CHECK(hidden.sizes() == torch::IntArrayRef({rows, kD}))
      << "hidden has an unexpected shape";
  CHECK(detail::check_tensor(affine_pair, torch::kBFloat16, device))
      << "affine_pair must have the expected dtype/device and be contiguous";
  CHECK(affine_pair.sizes() == torch::IntArrayRef({4, 2, kH}))
      << "affine_pair has an unexpected shape";
  CHECK(detail::check_tensor(values, torch::kBFloat16, device))
      << "values must have the expected dtype/device and be contiguous";
  CHECK(values.sizes() == torch::IntArrayRef({rows, kD}))
      << "values has an unexpected shape";
  CHECK(detail::check_tensor(normalized, torch::kBFloat16, device))
      << "normalized must have the expected dtype/device and be contiguous";
  CHECK(normalized.sizes() == torch::IntArrayRef({rows, kD}))
      << "normalized has an unexpected shape";
  check_disjoint(
      {{&values, "values"}, {&normalized, "normalized"}},
      {{&kv, "kv"}, {&hidden, "hidden"}, {&affine_pair, "affine_pair"}});
  return rows;
}

int32_t check_tail(const torch::Tensor& values,
                   const torch::Tensor& normalized,
                   const torch::Tensor& history,
                   const torch::Tensor& taps,
                   const torch::Tensor& hidden,
                   const torch::Tensor& output) {
  const int32_t rows = checked_rows(values, kD, "values");
  const torch::Device device = values.device();
  CHECK(detail::check_tensor(normalized, torch::kBFloat16, device))
      << "normalized must have the expected dtype/device and be contiguous";
  CHECK(normalized.sizes() == torch::IntArrayRef({rows, kD}))
      << "normalized has an unexpected shape";
  CHECK(detail::check_tensor(history, torch::kBFloat16, device))
      << "history must have the expected dtype/device and be contiguous";
  CHECK(history.sizes() == torch::IntArrayRef({rows / kQueries, 13, kD}))
      << "history has an unexpected shape";
  CHECK(detail::check_tensor(taps, torch::kFloat32, device))
      << "taps must have the expected dtype/device and be contiguous";
  CHECK(taps.sizes() == torch::IntArrayRef({4, kD}))
      << "taps has an unexpected shape";
  CHECK(detail::check_tensor(hidden, torch::kBFloat16, device))
      << "hidden must have the expected dtype/device and be contiguous";
  CHECK(hidden.sizes() == torch::IntArrayRef({rows, kD}))
      << "hidden has an unexpected shape";
  CHECK(detail::check_tensor(output, torch::kBFloat16, device))
      << "output must have the expected dtype/device and be contiguous";
  CHECK(output.sizes() == torch::IntArrayRef({rows, kD}))
      << "output has an unexpected shape";
  check_disjoint({{&history, "history"}, {&output, "output"}},
                 {{&values, "values"},
                  {&normalized, "normalized"},
                  {&taps, "taps"},
                  {&hidden, "hidden"}});
  return rows;
}
}  // namespace

void ple_pack_projection_out(const torch::Tensor& weight,
                             torch::Tensor& packed_weight) {
  check_projection_weights(weight, packed_weight);
  const torch::DeviceGuard guard(weight.device());
  launch_pack_projection_bf16(torch_mlu::getCurMLUStream(),
                              weight.data_ptr(),
                              packed_weight.data_ptr());
}

void ple_prepare_head_out(const torch::Tensor& key_scale,
                          const torch::Tensor& query_scale,
                          const torch::Tensor& conv_scale,
                          torch::Tensor& affine_pair) {
  check_head_weights(key_scale, query_scale, conv_scale, affine_pair);
  const torch::DeviceGuard guard(key_scale.device());
  launch_prepare_head_scale_packed_hc(torch_mlu::getCurMLUStream(),
                                      key_scale.data_ptr(),
                                      query_scale.data_ptr(),
                                      conv_scale.data_ptr(),
                                      affine_pair.data_ptr(),
                                      static_cast<int32_t>(kD));
}

void ple_prepare_weights_out(const torch::Tensor& weight,
                             const torch::Tensor& key_scale,
                             const torch::Tensor& query_scale,
                             const torch::Tensor& conv_scale,
                             torch::Tensor& packed_weight,
                             torch::Tensor& affine_pair) {
  check_projection_weights(weight, packed_weight);
  check_head_weights(key_scale, query_scale, conv_scale, affine_pair);
  CHECK(weight.device() == key_scale.device())
      << "All PLE weights must share one MLU device";
  check_disjoint(
      {{&packed_weight, "packed_weight"}, {&affine_pair, "affine_pair"}},
      {{&weight, "weight"},
       {&key_scale, "key_scale"},
       {&query_scale, "query_scale"},
       {&conv_scale, "conv_scale"}});
  const torch::DeviceGuard guard(weight.device());
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  launch_pack_projection_bf16(
      queue, weight.data_ptr(), packed_weight.data_ptr());
  launch_prepare_head_scale_packed_hc(queue,
                                      key_scale.data_ptr(),
                                      query_scale.data_ptr(),
                                      conv_scale.data_ptr(),
                                      affine_pair.data_ptr(),
                                      static_cast<int32_t>(kD));
}

void ple_projection_out(const torch::Tensor& embedding,
                        const torch::Tensor& packed_weight,
                        torch::Tensor& kv) {
  const int32_t rows = check_projection(embedding, packed_weight, kv);
  const torch::DeviceGuard guard(embedding.device());
  launch_projection_packed_bf16(torch_mlu::getCurMLUStream(),
                                embedding.data_ptr(),
                                packed_weight.data_ptr(),
                                kv.data_ptr(),
                                rows);
}

void ple_head_out(const torch::Tensor& kv,
                  const torch::Tensor& hidden,
                  const torch::Tensor& affine_pair,
                  torch::Tensor& values,
                  torch::Tensor& normalized) {
  const int32_t rows = check_head(kv, hidden, affine_pair, values, normalized);
  const torch::DeviceGuard guard(kv.device());
  launch_head_prepared_packed_hc(torch_mlu::getCurMLUStream(),
                                 kv.data_ptr(),
                                 hidden.data_ptr(),
                                 affine_pair.data_ptr(),
                                 values.data_ptr(),
                                 normalized.data_ptr(),
                                 rows);
}

void ple_tail_out(const torch::Tensor& values,
                  const torch::Tensor& normalized,
                  torch::Tensor& history,
                  const torch::Tensor& taps,
                  const torch::Tensor& hidden,
                  torch::Tensor& output) {
  const int32_t rows =
      check_tail(values, normalized, history, taps, hidden, output);
  const torch::DeviceGuard guard(values.device());
  launch_tail(torch_mlu::getCurMLUStream(),
              values.data_ptr(),
              normalized.data_ptr(),
              history.data_ptr(),
              taps.data_ptr(),
              hidden.data_ptr(),
              output.data_ptr(),
              rows / static_cast<int32_t>(kQueries));
}

void ple_forward_out(const torch::Tensor& embedding,
                     const torch::Tensor& hidden,
                     const torch::Tensor& packed_weight,
                     const torch::Tensor& affine_pair,
                     const torch::Tensor& taps,
                     torch::Tensor& history,
                     torch::Tensor& kv,
                     torch::Tensor& values,
                     torch::Tensor& normalized,
                     torch::Tensor& output) {
  const int32_t rows = check_projection(embedding, packed_weight, kv);
  check_head(kv, hidden, affine_pair, values, normalized);
  check_tail(values, normalized, history, taps, hidden, output);
  check_disjoint({{&history, "history"},
                  {&kv, "kv"},
                  {&values, "values"},
                  {&normalized, "normalized"},
                  {&output, "output"}},
                 {{&embedding, "embedding"},
                  {&hidden, "hidden"},
                  {&packed_weight, "packed_weight"},
                  {&affine_pair, "affine_pair"},
                  {&taps, "taps"}});
  const torch::DeviceGuard guard(embedding.device());
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  launch_projection_packed_bf16(queue,
                                embedding.data_ptr(),
                                packed_weight.data_ptr(),
                                kv.data_ptr(),
                                rows);
  launch_head_prepared_packed_hc(queue,
                                 kv.data_ptr(),
                                 hidden.data_ptr(),
                                 affine_pair.data_ptr(),
                                 values.data_ptr(),
                                 normalized.data_ptr(),
                                 rows);
  launch_tail(queue,
              values.data_ptr(),
              normalized.data_ptr(),
              history.data_ptr(),
              taps.data_ptr(),
              hidden.data_ptr(),
              output.data_ptr(),
              rows / static_cast<int32_t>(kQueries));
}

}  // namespace xllm::kernel::mlu::bangc
