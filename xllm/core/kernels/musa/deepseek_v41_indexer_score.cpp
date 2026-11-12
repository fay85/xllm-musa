/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include "core/kernels/musa/deepseek_v41_indexer_score.h"

#include <glog/logging.h>
#include <torch/utils.h>

#include <algorithm>
#include <array>
#include <limits>

#include "core/kernels/musa/musa_aot_kernel.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_indexer_score(const torch::Tensor& query,
                                         const torch::Tensor& keys,
                                         const torch::Tensor& weights,
                                         const torch::Tensor& positions,
                                         int64_t committed,
                                         int64_t ratio,
                                         const std::string& aot_directory) {
  CHECK(query.defined() && keys.defined() && weights.defined() &&
        positions.defined());
  CHECK(query.device().type() == torch::kPrivateUse1);
  CHECK_EQ(query.scalar_type(), torch::kBFloat16);
  CHECK_EQ(keys.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK(positions.scalar_type() == torch::kInt64 ||
        positions.scalar_type() == torch::kInt32);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(1), 32);
  CHECK_EQ(query.size(2), 128);
  CHECK_EQ(keys.dim(), 2);
  CHECK_EQ(keys.size(1), 128);
  CHECK_EQ(weights.sizes(), torch::IntArrayRef({query.size(0), 32}));
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({query.size(0)}));
  for (const torch::Tensor& value : {query, keys, weights, positions}) {
    CHECK_EQ(value.device(), query.device());
    CHECK(value.is_contiguous());
  }
  CHECK_GE(committed, 0);
  CHECK_LE(committed, keys.size(0));
  CHECK_LE(keys.size(0), std::numeric_limits<int32_t>::max());
  CHECK_LE(query.size(0), 512);
  CHECK(ratio == 1 || ratio == 2);
  CHECK(!aot_directory.empty());
  torch::DeviceGuard guard(query.device());
  torch::Tensor output = torch::empty({query.size(0), keys.size(0)},
                                      query.options().dtype(torch::kFloat32));
  if (output.numel() == 0) {
    return output;
  }
  if (committed == 0) {
    output.fill_(-std::numeric_limits<float>::infinity());
    return output;
  }
  const torch::Tensor visible = torch::floor_divide(positions + 1, ratio)
                                    .clamp(0, committed)
                                    .to(torch::kInt32)
                                    .contiguous();
  int32_t number_keys = static_cast<int32_t>(keys.size(0));
  int32_t number_queries = static_cast<int32_t>(query.size(0));
  int32_t committed_rows = static_cast<int32_t>(committed);
  void* query_pointer = query.data_ptr();
  void* keys_pointer = keys.data_ptr();
  void* weights_pointer = weights.data_ptr();
  void* output_pointer = output.data_ptr();
  void* visible_pointer = visible.data_ptr();
  std::array<void*, 8> arguments = {&query_pointer,
                                    &keys_pointer,
                                    &weights_pointer,
                                    &output_pointer,
                                    &visible_pointer,
                                    &number_keys,
                                    &number_queries,
                                    &committed_rows};
  launch_musa_aot_kernel(query.device(),
                         aot_directory,
                         "indexer_score",
                         {static_cast<uint32_t>((number_queries + 1) / 2),
                          static_cast<uint32_t>((number_keys + 127) / 128),
                          1},
                         arguments.data());
  return output;
}

torch::Tensor deepseek_v41_indexer_select(
    const torch::Tensor& scores,
    const torch::Tensor& visible,
    int64_t topk,
    const std::optional<torch::Tensor>& candidates,
    const std::string& aot_directory) {
  CHECK(scores.defined() && visible.defined());
  CHECK(scores.device().type() == torch::kPrivateUse1);
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.dim(), 2);
  CHECK_EQ(visible.sizes(), torch::IntArrayRef({scores.size(0)}));
  CHECK_EQ(visible.device(), scores.device());
  CHECK(visible.scalar_type() == torch::kInt64 ||
        visible.scalar_type() == torch::kInt32);
  CHECK_GT(scores.size(1), 0);
  CHECK_LE(scores.size(1), 8192);
  CHECK_GT(topk, 0);
  CHECK(!aot_directory.empty());
  torch::DeviceGuard guard(scores.device());
  const torch::Tensor values = scores.contiguous();
  const torch::Tensor limits = visible.to(torch::kInt64).contiguous();
  torch::Tensor mask;
  if (candidates.has_value()) {
    CHECK(candidates->defined());
    CHECK_EQ(candidates->device(), scores.device());
    CHECK_EQ(candidates->scalar_type(), torch::kBool);
    CHECK_EQ(candidates->sizes(), scores.sizes());
    mask = candidates->contiguous();
  }
  int32_t width = static_cast<int32_t>(scores.size(1));
  int32_t count = static_cast<int32_t>(std::min(topk, scores.size(1)));
  int32_t padded = 128;
  while (padded < width) {
    padded *= 2;
  }
  torch::Tensor output = torch::empty({scores.size(0), count},
                                      scores.options().dtype(torch::kInt32));
  if (scores.size(0) == 0) {
    return output;
  }
  void* scores_pointer = values.data_ptr();
  void* limits_pointer = limits.data_ptr();
  void* candidates_pointer = mask.defined() ? mask.data_ptr() : nullptr;
  void* output_pointer = output.data_ptr();
  std::array<void*, 6> arguments = {&scores_pointer,
                                    &candidates_pointer,
                                    &limits_pointer,
                                    &output_pointer,
                                    &width,
                                    &count};
  const std::string key = "select_" + std::to_string(padded) + "_" +
                          (candidates.has_value() ? "1" : "0");
  launch_musa_aot_kernel(scores.device(),
                         aot_directory,
                         key,
                         {static_cast<uint32_t>(scores.size(0)), 1, 1},
                         arguments.data());
  return output;
}

}  // namespace xllm::kernel::musa
