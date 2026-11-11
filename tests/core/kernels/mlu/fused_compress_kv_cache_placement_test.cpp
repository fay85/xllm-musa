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

#include <framework/core/device.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {

using PlacementCase = std::tuple<int32_t, bool, int32_t, bool>;

std::vector<PlacementCase> placement_cases() {
  std::vector<PlacementCase> cases;
  cases.reserve(32);
  for (int32_t ratio : {1, 2}) {
    for (bool reversed : {false, true}) {
      for (int32_t rows : {32768, 49151, 49152, 49153, 65536, 98305, 131072}) {
        cases.emplace_back(rows, reversed, ratio, false);
      }
      cases.emplace_back(65536, reversed, ratio, true);
    }
  }
  return cases;
}

class FusedCompressKvCachePlacementTest
    : public ::testing::TestWithParam<PlacementCase> {};

TEST_P(FusedCompressKvCachePlacementTest, WritesEveryRowToItsPhysicalSlot) {
  const auto [length, reversed_pages, ratio, split_requests] = GetParam();
  constexpr int32_t kHeadDim = 512;
  constexpr int32_t kPageSize = 128;
  constexpr int32_t kPadding = 3;
  // A request ending at 1493 makes the rollover window end off the usual
  // tile boundary, with short tiles among the pending stores.
  const std::vector<int32_t> lengths =
      split_requests ? std::vector<int32_t>{1493, length - 1493}
                     : std::vector<int32_t>{length};
  const int32_t batch = static_cast<int32_t>(lengths.size());
  const int32_t columns = (length + kPageSize - 1) / kPageSize;
  const int32_t pages = batch * columns;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto ints = torch::TensorOptions().dtype(torch::kInt32);
  const auto floats = torch::TensorOptions().dtype(torch::kFloat32);
  const auto bf16 = floats.dtype(torch::kBFloat16);

  // Each row encodes its ID in +/-1 values. RMSNorm with unit gamma and
  // identity RoPE preserve these exactly in BF16, giving an independent
  // oracle for both arithmetic and placement without numerical tolerances.
  // Seventeen bits distinguish every row in the largest case (131072).
  auto expected_rows = torch::empty({length, kHeadDim}, floats);
  float* values = expected_rows.data_ptr<float>();
  for (int32_t row = 0; row < length; ++row) {
    for (int32_t column = 0; column < kHeadDim; ++column) {
      values[static_cast<int64_t>(row) * kHeadDim + column] =
          ((row >> (column % 17)) & 1) == 0 ? -1.0f : 1.0f;
    }
  }
  expected_rows = expected_rows.to(torch::kBFloat16);
  const auto raw = ratio == 1 ? expected_rows.to(device)
                              : expected_rows.repeat_interleave(ratio, 0).to(
                                    device, torch::kFloat32);
  std::optional<CompressKvPoolingInputs> pooling;
  if (ratio == 2) {
    // Identical members with zero gates preserve each row's marker under
    // pooling, isolating cache placement from gate numerical accuracy.
    pooling.emplace();
    pooling->scores = torch::zeros_like(raw);
    pooling->state_block_table =
        torch::arange(batch, ints).view({batch, 1}).to(device);
    pooling->state_raw =
        torch::zeros({batch, 32, kHeadDim}, floats.device(device));
    pooling->state_score = torch::zeros_like(pooling->state_raw);
  }
  const auto gamma = torch::ones({kHeadDim}, floats.device(device));
  auto rope = torch::zeros({ratio * length, 128}, floats.device(device));
  rope.slice(1, 0, 64).fill_(1);
  std::vector<int32_t> offsets{0};
  offsets.reserve(batch + 1);
  for (int32_t count : lengths) {
    offsets.emplace_back(offsets.back() + count);
  }
  const auto prefix = torch::tensor(offsets, ints).to(device);
  const auto query = prefix * ratio;
  const auto begin = torch::zeros({batch}, ints.device(device));
  auto page_ids = (torch::arange(pages, ints) + 1).view({batch, columns});
  if (reversed_pages) {
    page_ids = page_ids.flip({1});
  }
  const auto table = page_ids.to(device);
  auto expected_positions = torch::empty({length}, ints);
  auto expected_slots = torch::empty({length}, ints);
  for (int32_t seq = 0; seq < batch; ++seq) {
    const auto logical_rows = torch::arange(lengths[seq], ints);
    expected_positions.narrow(0, offsets[seq], lengths[seq])
        .copy_(logical_rows * ratio);
    expected_slots.narrow(0, offsets[seq], lengths[seq])
        .copy_(page_ids[seq].index_select(
                   0,
                   torch::floor_divide(logical_rows, kPageSize)
                       .to(torch::kInt64)) *
                   kPageSize +
               logical_rows.remainder(kPageSize));
  }

  // Keep guard pages both inside and outside the cache view. Also check
  // unused rows in a partial last page, not only the requested destinations.
  auto cache_storage =
      torch::full({pages + 4, kPageSize, kHeadDim}, -7, bf16.device(device));
  const auto cache = cache_storage.slice(0, 1, pages + 3);
  auto expected_cache = torch::full({pages + 4, kPageSize, kHeadDim}, -7, bf16);
  expected_cache.slice(0, 1, pages + 3)
      .view({-1, kHeadDim})
      .index_copy_(0, expected_slots.to(torch::kInt64), expected_rows);
  const auto latent =
      torch::full({length + kPadding, kHeadDim}, -9, bf16.device(device));
  const auto positions =
      torch::full({length + kPadding}, -9, ints.device(device));
  const auto slots = torch::full_like(positions, -9);
  const auto valid_rows = torch::full({1}, -9, ints.device(device));

  CompressKvMetadata metadata;
  metadata.query_start_loc = query;
  metadata.chunk_begins = begin;
  metadata.compressed_start_loc = prefix;
  metadata.compressed_block_table = table;
  CompressKvOutputs outputs;
  outputs.latent = latent;
  outputs.positions = positions;
  outputs.slots = slots;
  outputs.num_valid_rows = valid_rows;
  CompressKvOptions options;
  options.ratio = ratio;
  options.rms_norm_eps = 1e-20;
  fused_compress_kv_cache(
      raw, gamma, rope, metadata, pooling, cache, outputs, options);
  torch_mlu::synchronize();

  EXPECT_EQ(valid_rows.item<int32_t>(), length);
  EXPECT_TRUE(
      torch::equal(positions.cpu().narrow(0, 0, length), expected_positions));
  EXPECT_TRUE(torch::equal(slots.cpu().narrow(0, 0, length), expected_slots));
  EXPECT_TRUE(torch::equal(latent.cpu().narrow(0, 0, length), expected_rows));
  EXPECT_TRUE(positions.narrow(0, length, kPadding).eq(0).all().item<bool>());
  EXPECT_TRUE(slots.narrow(0, length, kPadding).eq(-1).all().item<bool>());
  EXPECT_TRUE(latent.narrow(0, length, kPadding).eq(0).all().item<bool>());

  const auto actual_cache = cache_storage.cpu();
  const auto written = actual_cache.slice(0, 1, pages + 3)
                           .view({-1, kHeadDim})
                           .index_select(0, expected_slots.to(torch::kInt64));
  const auto bad_rows = written.ne(expected_rows).any(1).nonzero().flatten();
  const int64_t bad_row_count = bad_rows.numel();
  const int64_t first_bad_row =
      bad_row_count == 0 ? -1 : bad_rows[0].item<int64_t>();
  RecordProperty("bad_cache_rows", std::to_string(bad_row_count));
  RecordProperty("first_bad_logical_row", std::to_string(first_bad_row));
  EXPECT_EQ(bad_row_count, 0)
      << "length=" << length << ", reversed_pages=" << reversed_pages
      << ", first_bad_logical_row=" << first_bad_row;
  EXPECT_TRUE(torch::equal(actual_cache, expected_cache))
      << "Cache contents, unused rows, and guard pages must all match";
}

INSTANTIATE_TEST_SUITE_P(
    LargePrefill,
    FusedCompressKvCachePlacementTest,
    // On MLU590 (32 cores), 49152 rows fill 1536 metadata entries per core.
    // The adjacent lengths bracket the first rollover; larger cases also
    // exercise repeated rollovers. These remain valid on other topologies.
    ::testing::ValuesIn(placement_cases()),
    [](const ::testing::TestParamInfo<PlacementCase>& info) {
      return "Rows" + std::to_string(std::get<0>(info.param)) +
             (std::get<1>(info.param) ? "ReversedPages" : "OrderedPages") +
             "Ratio" + std::to_string(std::get<2>(info.param)) +
             (std::get<3>(info.param) ? "SplitRequests" : "SingleRequest");
    });

}  // namespace
}  // namespace xllm::kernel::mlu::bangc
