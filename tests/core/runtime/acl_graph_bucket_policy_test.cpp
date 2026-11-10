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

#include "core/runtime/acl_graph_bucket_policy.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "core/runtime/decode_graph_bucket.h"
#include "core/scheduler/profile/decode_graph_warmup_plan.h"

namespace xllm {
namespace {

TEST(DecodeGraphCaptureSizesTest, MatchesVllmUniformQueryRounding) {
  EXPECT_EQ(
      runtime::normalize_decode_graph_capture_sizes({10, 3, 7, 3}, 1, 128, 16),
      (std::vector<int32_t>{3, 7, 10}));
  EXPECT_EQ(runtime::normalize_decode_graph_capture_sizes(
                {1, 2, 4, 8, 16}, 3, 128, 16),
            (std::vector<int32_t>{3, 6, 9}));
  EXPECT_EQ(
      runtime::normalize_decode_graph_capture_sizes({1, 4, 8, 16}, 3, 8, 16),
      (std::vector<int32_t>{3, 6}));
  EXPECT_EQ(
      runtime::normalize_decode_graph_capture_sizes({1, 4, 8, 16}, 3, 128, 2),
      (std::vector<int32_t>{3, 6}));
  EXPECT_TRUE(runtime::normalize_decode_graph_capture_sizes({1, 2}, 3, 128, 16)
                  .empty());
}

TEST(DecodeGraphCaptureSizesTest, HandlesInvalidWidthsAndIntegerBoundaries) {
  EXPECT_TRUE(runtime::normalize_decode_graph_capture_sizes(
                  {1}, std::numeric_limits<int64_t>::max(), 128, 16)
                  .empty());
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(runtime::normalize_decode_graph_capture_sizes(
                {kMax - 1, kMax}, 2, kMax, kMax),
            (std::vector<int32_t>{kMax - 1}));
}

TEST(DecodeGraphCaptureSizesTest, WarmupUsesLocalRowsAndPartialDpBatches) {
  runtime::DecodeGraphWarmupConfig config;
  config.graph_capture_sizes = {1, 3, 7, 10};
  config.max_graph_batch_size = 16;
  EXPECT_EQ(build_decode_graph_warmup_plan(config, 30, 4).batch_sizes,
            (std::vector<int32_t>{4, 12, 28, 30}));
  EXPECT_EQ(build_decode_graph_warmup_plan(config, 10, 1).batch_sizes,
            (std::vector<int32_t>{1, 3, 7, 10}));
  config.num_decoding_tokens = 3;
  config.graph_capture_sizes = runtime::normalize_decode_graph_capture_sizes(
      {1, 3, 5, 7, 10, 12}, config.num_decoding_tokens, 128, 4);
  EXPECT_EQ(build_decode_graph_warmup_plan(config, 13, 4).batch_sizes,
            (std::vector<int32_t>{4, 8, 12, 13}));
  EXPECT_EQ(build_decode_graph_warmup_plan(config, 65, 32).batch_sizes,
            (std::vector<int32_t>{32, 64, 65}));
  EXPECT_EQ(build_decode_graph_warmup_plan(config, 4, 4).batch_sizes,
            (std::vector<int32_t>{4}));
}

TEST(AclGraphBucketPolicyUnitTest, ConvertsLocalLimitToGlobalDpCapacity) {
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(3000, 64, 32), 2048);
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(64, 64, 32), 64);
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(3000, 16, 32), 512);
}

TEST(AclGraphBucketPolicyUnitTest, MapsGlobalWarmupBucketsToLocalDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(1, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(2, 16, 4));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(3, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(4, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(12, 64, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(10, 40, 4));
}

TEST(AclGraphBucketPolicyUnitTest, RejectsNonCanonicalSingleDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(16, 20, 1));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(20, 20, 1));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(3, 20, 1));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(12, 20, 1));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNoPaddingMtpTokenBuckets) {
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(
      /*batch_size=*/12,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*num_decoding_tokens=*/4));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(
      /*batch_size=*/11,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*num_decoding_tokens=*/4));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/12,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*num_decoding_tokens=*/4));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNonDivisibleDpMtpTailBucket) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/3,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      /*num_decoding_tokens=*/4));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/4,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      /*num_decoding_tokens=*/4));
}

TEST(AclGraphBucketPolicyUnitTest, AppliesWarmupCapacityInLocalDpUnits) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(4, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(5, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, false));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, true));
}

}  // namespace
}  // namespace xllm
