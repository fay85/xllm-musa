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

#include "core/runtime/musa/prefill_graph_policy.h"

namespace xllm {
namespace {

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

TEST(MusaPrefillGraphPolicyUnitTest, EnforcesMemoryHeadroomAndOomLatch) {
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  // Minimum floor is 1536 MiB even when reserve_mb is smaller.
  EXPECT_EQ(runtime::musa::prefill_capture_required_free_bytes(0),
            1536ULL * kMiB);
  EXPECT_EQ(runtime::musa::prefill_capture_required_free_bytes(2048),
            2048ULL * kMiB);

  // Cache hits always replay even when OOM is latched or free memory is 0.
  EXPECT_TRUE(runtime::musa::can_admit_prefill_graph_with_memory(
      /*cached_entries=*/128,
      /*max_entries=*/128,
      /*cache_hit=*/true,
      /*oom_latched=*/true,
      /*physical_free_bytes=*/0,
      /*reserve_mb=*/2048));

  // Unseen keys are rejected when OOM is latched.
  EXPECT_FALSE(runtime::musa::can_admit_prefill_graph_with_memory(
      /*cached_entries=*/10,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/true,
      /*physical_free_bytes=*/4096ULL * kMiB,
      /*reserve_mb=*/2048));

  // Unseen keys are rejected when physical free memory is below reserve.
  EXPECT_FALSE(runtime::musa::can_admit_prefill_graph_with_memory(
      /*cached_entries=*/10,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/2047ULL * kMiB,
      /*reserve_mb=*/2048));

  // Unseen keys are admitted when within max_entries and free memory >= reserve.
  EXPECT_TRUE(runtime::musa::can_admit_prefill_graph_with_memory(
      /*cached_entries=*/10,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/2048ULL * kMiB,
      /*reserve_mb=*/2048));
}

}  // namespace
}  // namespace xllm
