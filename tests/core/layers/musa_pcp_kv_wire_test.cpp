/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/layers/musa/pcp_context.h"

namespace xllm::layer::musa_pcp {
namespace {

class CompletedBroadcast final : public c10d::Work {
 public:
  CompletedBroadcast() : Work(/*rank=*/1, c10d::OpType::BROADCAST) { finish(); }
};

struct BroadcastRecord final {
  torch::Tensor rows;
  int32_t root_rank = 0;
};

class RecordingProcessGroup final : public ProcessGroup {
 public:
  explicit RecordingProcessGroup(int32_t rank)
      : ProcessGroup(rank, /*world_size=*/2, torch::Device(torch::kCPU)) {}

  c10::intrusive_ptr<c10d::Work> broadcast_async(torch::Tensor& input,
                                                 int32_t root_rank) override {
    input.fill_(static_cast<int64_t>(broadcasts.size()) + 1);
    broadcasts.emplace_back(BroadcastRecord{input, root_rank});
    return c10::make_intrusive<CompletedBroadcast>();
  }

  std::vector<BroadcastRecord> broadcasts;
};

Context receive_context(ProcessGroup* group) {
  Context context;
  context.cp_group = group;
  context.cp_rank = group->rank();
  context.row_offsets = {0, 3, 8};
  return context;
}

TEST(MusaPcpKvWireTest, ReceivesProjectedBf16RowsWithFp8CacheStorage) {
  RecordingProcessGroup group(/*rank=*/1);
  Context context = receive_context(&group);
  const torch::Tensor projection_weight =
      torch::zeros({4}, torch::TensorOptions().dtype(torch::kBFloat16));
  const torch::Tensor cache = torch::empty(
      {2, 64, 1, 4}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn));
  ASSERT_NE(projection_weight.scalar_type(), cache.scalar_type());

  expect_prefix_kv(/*num_kv_heads=*/1,
                   /*head_dim=*/4,
                   projection_weight.options(),
                   context);

  ASSERT_EQ(group.broadcasts.size(), 1u);
  EXPECT_EQ(group.broadcasts[0].root_rank, 0);
  const torch::Tensor rows = take_prefix_rows(context);
  EXPECT_EQ(rows.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(rows.sizes().vec(), std::vector<int64_t>({3, 8}));
  EXPECT_EQ(rows.nbytes(), 3u * 8u * 2u);
  EXPECT_TRUE(torch::equal(rows, torch::ones_like(rows)));
  EXPECT_TRUE(context.pending_receives.empty());
}

TEST(MusaPcpKvWireTest, KeepsLayerOrderAndProjectionDtypes) {
  RecordingProcessGroup group(/*rank=*/1);
  Context context = receive_context(&group);
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat16, torch::kBFloat16, torch::kFloat32};
  for (const torch::ScalarType dtype : dtypes) {
    expect_prefix_kv(/*num_kv_heads=*/1,
                     /*head_dim=*/4,
                     torch::TensorOptions().dtype(dtype),
                     context);
  }
  ASSERT_EQ(group.broadcasts.size(), dtypes.size());
  for (int64_t layer = 0; layer < static_cast<int64_t>(dtypes.size());
       ++layer) {
    const torch::Tensor rows = take_prefix_rows(context);
    EXPECT_EQ(rows.scalar_type(), dtypes[layer]);
    EXPECT_TRUE(torch::equal(rows, torch::full_like(rows, layer + 1)));
  }
  EXPECT_TRUE(context.pending_receives.empty());
}

TEST(MusaPcpKvWireTest, SenderDoesNotPostAnUnusedReceive) {
  RecordingProcessGroup group(/*rank=*/0);
  Context context = receive_context(&group);
  expect_prefix_kv(/*num_kv_heads=*/1,
                   /*head_dim=*/4,
                   torch::TensorOptions().dtype(torch::kBFloat16),
                   context);
  EXPECT_TRUE(group.broadcasts.empty());
  EXPECT_TRUE(context.pending_receives.empty());
}

TEST(MusaPcpKvWireTest, RejectsFp8StorageOptionsBeforePostingCollective) {
  RecordingProcessGroup group(/*rank=*/1);
  Context context = receive_context(&group);
  EXPECT_DEATH(expect_prefix_kv(
                   /*num_kv_heads=*/1,
                   /*head_dim=*/4,
                   torch::TensorOptions().dtype(torch::kFloat8_e4m3fn),
                   context),
               "projection compute dtype");
  EXPECT_TRUE(group.broadcasts.empty());
  EXPECT_TRUE(context.pending_receives.empty());
}

}  // namespace
}  // namespace xllm::layer::musa_pcp
