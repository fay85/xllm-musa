/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace {

TEST(StateDictUtilsTest, LoadsMoeWeightsIntoPhysicalExpertPrefix) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("0.up_proj.weight",
                  torch::tensor({{1, 2}, {3, 4}}, torch::kInt32));
  tensors.emplace("1.up_proj.weight",
                  torch::tensor({{5, 6}, {7, 8}}, torch::kInt32));
  StateDict state_dict(std::move(tensors));

  torch::Tensor physical_weight = torch::full({3, 2, 2}, -1, torch::kInt32);
  std::vector<torch::Tensor> accumulated_tensors;
  bool weight_is_loaded = false;

  weight::load_moe_weight(state_dict,
                          "up_proj.",
                          "weight",
                          /*dim=*/-1,
                          /*rank=*/0,
                          /*world_size=*/1,
                          /*start_expert_id=*/0,
                          /*num_experts_per_rank=*/2,
                          accumulated_tensors,
                          physical_weight,
                          weight_is_loaded);

  EXPECT_TRUE(weight_is_loaded);
  EXPECT_TRUE(torch::equal(physical_weight[0],
                           torch::tensor({{1, 2}, {3, 4}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(physical_weight[1],
                           torch::tensor({{5, 6}, {7, 8}}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(physical_weight[2], torch::full({2, 2}, -1, torch::kInt32)));
}

torch::Tensor scale_with_marker(int64_t rows, int64_t marker) {
  return torch::arange(rows, torch::kFloat32).unsqueeze(1) + marker;
}

TEST(StateDictUtilsTest, ShardsSeparateQkvBlockScalesPerProjection) {
  constexpr int64_t kBlock = 128;
  const std::vector<int64_t> partitions = {512, 256, 256};
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("q_proj.weight_scale_inv", scale_with_marker(4, 10));
  tensors.emplace("k_proj.weight_scale_inv", scale_with_marker(2, 20));
  tensors.emplace("v_proj.weight_scale_inv", scale_with_marker(2, 30));
  StateDict state_dict(std::move(tensors));

  torch::Tensor scale = torch::empty({4, 1}, torch::kFloat32);
  std::vector<torch::Tensor> accumulated;
  bool loaded = false;
  weight::load_fused_weight(state_dict,
                            {"q_proj.", "k_proj.", "v_proj."},
                            "weight_scale_inv",
                            /*dim=*/0,
                            /*rank=*/1,
                            /*world_size=*/2,
                            accumulated,
                            scale,
                            loaded,
                            /*num_kv_head_replicas=*/1,
                            &partitions,
                            /*block_fp8_scales_are_separate=*/true);

  EXPECT_TRUE(loaded);
  const torch::Tensor expected = torch::cat(
      {scale_with_marker(4, 10).slice(0, 2, 4),
       scale_with_marker(2, 20).slice(0, 1, 2),
       scale_with_marker(2, 30).slice(0, 1, 2)},
      /*dim=*/0);
  EXPECT_TRUE(torch::equal(scale, expected));
  EXPECT_EQ(kBlock * scale.size(0), 512);
}

TEST(StateDictUtilsTest, ShardsFusedQkvBlockScaleAcrossPartitions) {
  const std::vector<int64_t> partitions = {512, 256, 256};
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("weight_scale_inv", scale_with_marker(8, 100));
  StateDict state_dict(std::move(tensors));

  torch::Tensor scale = torch::empty({4, 1}, torch::kFloat32);
  std::vector<torch::Tensor> accumulated;
  bool loaded = false;
  weight::load_fused_weight(state_dict,
                            {""},
                            "weight_scale_inv",
                            /*dim=*/0,
                            /*rank=*/0,
                            /*world_size=*/2,
                            accumulated,
                            scale,
                            loaded,
                            /*num_kv_head_replicas=*/1,
                            &partitions,
                            /*block_fp8_scales_are_separate=*/false);

  EXPECT_TRUE(loaded);
  const torch::Tensor full = scale_with_marker(8, 100);
  const torch::Tensor expected = torch::cat(
      {full.slice(0, 0, 2), full.slice(0, 4, 5), full.slice(0, 6, 7)},
      /*dim=*/0);
  EXPECT_TRUE(torch::equal(scale, expected));
}

TEST(StateDictUtilsTest, ShardsMergedBlockScaleByProjectionBoundaries) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("weight_scale_inv", scale_with_marker(8, 200));
  StateDict state_dict(std::move(tensors));
  const std::vector<int64_t> full_rows = {512, 256, 256};
  const std::vector<int64_t> rank_rows = {256, 128, 128};
  const std::vector<int64_t> scale_rows =
      weight::block_fp8_scale_shard_sizes(rank_rows);
  torch::Tensor scale = torch::empty({4, 1}, torch::kFloat32);
  bool loaded = false;
  weight::load_merged_block_fp8_scale(state_dict,
                                      "weight_scale_inv",
                                      /*dim=*/0,
                                      /*rank=*/1,
                                      /*world_size=*/2,
                                      /*shard_tensor_count=*/3,
                                      full_rows,
                                      scale_rows,
                                      scale,
                                      loaded);
  EXPECT_TRUE(loaded);
  const torch::Tensor full = scale_with_marker(8, 200);
  const torch::Tensor expected = torch::cat(
      {full.slice(0, 2, 4), full.slice(0, 5, 6), full.slice(0, 7, 8)},
      /*dim=*/0);
  EXPECT_TRUE(torch::equal(scale, expected));
}

TEST(StateDictUtilsTest, ReshapesMoeScaleInsidePhysicalExpertPrefix) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("0.up_proj.weight_scale",
                  torch::tensor({{1}, {2}}, torch::kFloat32));
  tensors.emplace("1.up_proj.weight_scale",
                  torch::tensor({{3}, {4}}, torch::kFloat32));
  StateDict state_dict(std::move(tensors));

  torch::Tensor physical_scale = torch::full({3, 2}, -1, torch::kFloat32);
  std::vector<torch::Tensor> accumulated_tensors;
  bool scale_is_loaded = false;

  weight::load_moe_weight(state_dict,
                          "up_proj.",
                          "weight_scale",
                          /*dim=*/-1,
                          /*rank=*/0,
                          /*world_size=*/1,
                          /*start_expert_id=*/0,
                          /*num_experts_per_rank=*/2,
                          accumulated_tensors,
                          physical_scale,
                          scale_is_loaded);

  EXPECT_TRUE(scale_is_loaded);
  EXPECT_TRUE(
      torch::equal(physical_scale,
                   torch::tensor({{1, 2}, {3, 4}, {-1, -1}}, torch::kFloat32)));
}

}  // namespace
}  // namespace xllm
