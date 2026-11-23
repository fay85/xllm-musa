#include <gtest/gtest.h>

#include "core/runtime/musa/prefill_graph_policy.h"

namespace xllm::runtime::musa {
namespace {

TEST(MusaPrefillGraphPolicyUnitTest, EnforcesMemoryHeadroomAndOomLatch) {
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  EXPECT_EQ(prefill_capture_required_free_bytes(2048, false), 2048ULL * kMiB);
  EXPECT_EQ(prefill_capture_required_free_bytes(0, false), 1536ULL * kMiB);

  EXPECT_TRUE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/128,
      /*max_entries=*/128,
      /*cache_hit=*/true,
      /*oom_latched=*/true,
      /*physical_free_bytes=*/0,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/false));

  EXPECT_FALSE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/1,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/true,
      /*physical_free_bytes=*/8192ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/true));

  EXPECT_FALSE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/128,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/8192ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/true));

  EXPECT_FALSE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/4,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/1024ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/false));
}

TEST(MusaPrefillGraphPolicyUnitTest, PoolReuseChangesOnlyTheNewKeyMemoryBudget) {
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  EXPECT_EQ(prefill_capture_required_free_bytes(2048, false), 2048ULL * kMiB);
  EXPECT_EQ(prefill_capture_required_free_bytes(2048, true), 768ULL * kMiB);
  EXPECT_EQ(prefill_capture_required_free_bytes(128, true), 768ULL * kMiB);

  EXPECT_FALSE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/16,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/1024ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/false));
  EXPECT_TRUE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/16,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/1024ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/true));
  EXPECT_FALSE(can_admit_prefill_graph_with_memory(
      /*cached_entries=*/16,
      /*max_entries=*/128,
      /*cache_hit=*/false,
      /*oom_latched=*/false,
      /*physical_free_bytes=*/512ULL * kMiB,
      /*reserve_mb=*/2048,
      /*vmm_pool_reusable=*/true));
}

}  // namespace
}  // namespace xllm::runtime::musa
