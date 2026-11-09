/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "scheduler/linear_prefix_admission_policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <latch>
#include <memory>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/block/block_manager_pool_test_peer.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/sequences_group.h"

namespace xllm {

namespace {

using Clock = LinearPrefixAdmissionPolicy::Clock;

std::vector<XXH3Key> prompt_hashes(const std::vector<int32_t>& tokens,
                                   size_t block_size = 4) {
  const size_t blocks = tokens.size() / block_size;
  std::vector<XXH3Key> hashes;
  hashes.reserve(blocks);
  extend_prefix_hashes(BlockHasherType::TEXT,
                       MMData(),
                       Slice<int32_t>(tokens),
                       block_size,
                       blocks,
                       hashes);
  return hashes;
}

std::vector<XXH3Key> branch_hashes(int32_t suffix,
                                   size_t shared_tokens = 8,
                                   size_t prompt_tokens = 40) {
  std::vector<int32_t> tokens(prompt_tokens, suffix);
  std::fill_n(tokens.begin(), shared_tokens, 1);
  return prompt_hashes(tokens);
}

LinearPrefixAdmissionPolicy::Options admission_options(
    size_t max_entries = 64,
    size_t min_distinct_successors = 4) {
  return {.chunk_size = 16,
          .max_entries = max_entries,
          .min_distinct_successors = min_distinct_successors};
}

}  // namespace

TEST(LinearPrefixAdmissionPolicyTest,
     AdmitsHotFrontPrefixBetweenChunkBoundaries) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  const std::vector<XXH3Key> probe = branch_hashes(6);
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  for (int32_t suffix = 2; suffix < 5; ++suffix) {
    policy.observe(branch_hashes(suffix));
    EXPECT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  }
  policy.observe(branch_hashes(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     DefaultThresholdRequiresFourDistinctSuccessors) {
  LinearPrefixAdmissionPolicy policy(4, {.chunk_size = 16, .max_entries = 64});
  const std::vector<XXH3Key> probe = branch_hashes(6);
  for (int32_t suffix = 2; suffix < 5; ++suffix) {
    policy.observe(branch_hashes(suffix));
    ASSERT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  }
  for (int32_t observation = 0; observation < 8; ++observation) {
    for (int32_t suffix = 2; suffix < 5; ++suffix) {
      policy.observe(branch_hashes(suffix));
    }
    EXPECT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  }
  policy.observe(branch_hashes(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     RepeatedIdenticalPromptsDoNotCreateCheckpoint) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  const std::vector<XXH3Key> hashes =
      prompt_hashes(std::vector<int32_t>(40, 1));
  for (int32_t observation = 0; observation < 8; ++observation) {
    policy.observe(hashes);
    EXPECT_EQ(policy.find_checkpoint_tokens(hashes), 0u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest, SingleLongPromptDoesNotCreateCheckpoint) {
  LinearPrefixAdmissionPolicy policy(128,
                                     {.chunk_size = 10240, .max_entries = 128});
  const std::vector<XXH3Key> hashes =
      prompt_hashes(std::vector<int32_t>(1000000, 1), 128);
  policy.observe(hashes);
  policy.observe(hashes);
  EXPECT_EQ(policy.find_checkpoint_tokens(hashes), 0u);
}

TEST(LinearPrefixAdmissionPolicyTest, PartialSuffixUsesExistingTailCheckpoint) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    policy.observe(branch_hashes(suffix));
  }
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6, 8, 10)), 0u);
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6, 8, 8)), 0u);
}

TEST(LinearPrefixAdmissionPolicyTest, ExcludesRegularCheckpoint) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    policy.observe(branch_hashes(suffix, 16));
  }
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6, 16)), 0u);
}

TEST(LinearPrefixAdmissionPolicyTest, SelectsDeepestHotBranch) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(64, 2));
  std::vector<int32_t> tokens(40, 2);
  std::fill_n(tokens.begin(), 8, 1);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 12, tokens.end(), 3);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.end(), 4);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.begin() + 12, 2);
  std::fill(tokens.begin() + 12, tokens.end(), 5);
  EXPECT_EQ(policy.find_checkpoint_tokens(prompt_hashes(tokens)), 12u);
}

TEST(LinearPrefixAdmissionPolicyTest, ColdDeeperBranchFallsBackToHotAncestor) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(64, 3));
  std::vector<int32_t> tokens(40, 2);
  std::fill_n(tokens.begin(), 8, 1);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 12, tokens.end(), 3);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.end(), 4);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.end(), 5);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.begin() + 12, 2);
  std::fill(tokens.begin() + 12, tokens.end(), 6);
  EXPECT_EQ(policy.find_checkpoint_tokens(prompt_hashes(tokens)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     RegularDeeperBranchDoesNotHideEligibleAncestor) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(64, 2));
  std::vector<int32_t> tokens(40, 2);
  std::fill_n(tokens.begin(), 8, 1);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 16, tokens.end(), 3);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.end(), 4);
  policy.observe(prompt_hashes(tokens));
  std::fill(tokens.begin() + 8, tokens.begin() + 16, 2);
  std::fill(tokens.begin() + 16, tokens.end(), 5);
  EXPECT_EQ(policy.find_checkpoint_tokens(prompt_hashes(tokens)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     RepeatedObservationsDoNotAddDistinctSuccessors) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(64, 4));
  const std::vector<XXH3Key> first = branch_hashes(2);
  const std::vector<XXH3Key> second = branch_hashes(3);
  for (int32_t observation = 0; observation < 100; ++observation) {
    policy.observe(first);
    policy.observe(second);
    EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  }
  policy.observe(branch_hashes(4));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  policy.observe(branch_hashes(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest, PromptEndIsNotADifferentSuccessor) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  const std::vector<XXH3Key> first = branch_hashes(2);
  policy.observe(first);
  policy.observe(Slice<XXH3Key>(first).slice(0, 2));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  policy.observe(branch_hashes(3));
  policy.observe(branch_hashes(4));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  policy.observe(branch_hashes(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest, IgnoresObservationAtDifferentDepth) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  const std::vector<XXH3Key> first = branch_hashes(2);
  policy.observe(first);
  policy.observe(std::vector<XXH3Key>{first[1]});
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  policy.observe(branch_hashes(3));
  policy.observe(branch_hashes(4));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  policy.observe(branch_hashes(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     LongColdSuffixPreservesSharedBranchWithinCapacity) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(4, 2));
  const std::vector<XXH3Key> first = branch_hashes(2, 8, 400);
  const std::vector<XXH3Key> second = branch_hashes(3, 8, 400);
  policy.observe(first);
  policy.observe(second);
  for (int32_t observation = 0; observation < 4; ++observation) {
    policy.observe(first);
    policy.observe(second);
    EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4, 8, 400)), 8u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest,
     FullCapacityLeafExtensionLearnsFourSuccessors) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(8));
  const Clock::time_point start{};
  const std::vector<XXH3Key> probe = branch_hashes(6, 8, 12);
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 1)), start);
  policy.observe(prompt_hashes(std::vector<int32_t>(24, 9)),
                 start + std::chrono::seconds(1));
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    const Clock::time_point now = start + std::chrono::seconds(suffix);
    policy.observe(branch_hashes(suffix, 8, 12), now);
    EXPECT_EQ(policy.find_checkpoint_tokens(probe, now), suffix == 5 ? 8u : 0u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest,
     FullCapacityLeafExtensionEvictsOtherColdLeaves) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(6));
  const Clock::time_point start{};
  const std::vector<XXH3Key> probe = branch_hashes(6, 8, 12);
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 1)), start);
  for (int32_t suffix = 10; suffix < 14; ++suffix) {
    policy.observe(prompt_hashes(std::vector<int32_t>(4, suffix)),
                   start + std::chrono::seconds(suffix - 9));
  }
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    const Clock::time_point now = start + std::chrono::seconds(suffix + 3);
    policy.observe(branch_hashes(suffix, 8, 12), now);
    EXPECT_EQ(policy.find_checkpoint_tokens(probe, now), suffix == 5 ? 8u : 0u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest,
     CapacityOneDoesNotKeepDisconnectedPrefixes) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(1, 2));
  for (int32_t suffix = 2; suffix < 16; ++suffix) {
    policy.observe(branch_hashes(suffix));
    EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(suffix + 1)), 0u);
    policy.observe(prompt_hashes(std::vector<int32_t>(16, suffix)));
    EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(suffix)), 0u);
  }
  policy.reset();
  policy.observe(branch_hashes(2));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(3)), 0u);
}

TEST(LinearPrefixAdmissionPolicyTest, EvictsOldBranchesWithoutBreakingTheTree) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(3, 2));
  policy.observe(branch_hashes(2, 4, 8));
  policy.observe(branch_hashes(3, 4, 8));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4, 4, 8)), 4u);
  policy.observe(prompt_hashes(std::vector<int32_t>(12, 5)));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4, 4, 8)), 0u);
  policy.observe(branch_hashes(3, 4, 8));
  policy.observe(branch_hashes(6, 4, 8));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(7, 4, 8)), 4u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     BoundedTreeSurvivesRepeatedBranchEviction) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(4, 2));
  for (int32_t suffix = 2; suffix < 202; ++suffix) {
    policy.observe(branch_hashes(suffix, 4));
    EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(999, 4)),
              suffix == 2 ? 0u : 4u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest, RepeatedObservationsRefreshLru) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(4, 2));
  const Clock::time_point start{};
  const std::vector<XXH3Key> first = branch_hashes(2, 4, 8);
  const std::vector<XXH3Key> second = branch_hashes(3, 4, 8);
  policy.observe(first, start);
  policy.observe(second, start + std::chrono::seconds(1));
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 4)),
                 start + std::chrono::seconds(2));
  policy.observe(first, start + std::chrono::seconds(3));
  policy.observe(second, start + std::chrono::seconds(4));
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 5)),
                 start + std::chrono::seconds(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6, 4, 8),
                                          start + std::chrono::seconds(6)),
            4u);
}

TEST(LinearPrefixAdmissionPolicyTest, MatchingRefreshesLruWithoutObservation) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(4, 2));
  const Clock::time_point start{};
  const std::vector<XXH3Key> first = branch_hashes(2, 4, 8);
  const std::vector<XXH3Key> second = branch_hashes(3, 4, 8);
  policy.observe(first, start);
  policy.observe(second, start + std::chrono::seconds(1));
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 4)),
                 start + std::chrono::seconds(2));
  EXPECT_EQ(
      policy.find_checkpoint_tokens(first, start + std::chrono::seconds(3)),
      4u);
  EXPECT_EQ(
      policy.find_checkpoint_tokens(second, start + std::chrono::seconds(4)),
      4u);
  policy.observe(prompt_hashes(std::vector<int32_t>(8, 5)),
                 start + std::chrono::seconds(5));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6, 4, 8),
                                          start + std::chrono::seconds(6)),
            4u);
}

TEST(LinearPrefixAdmissionPolicyTest, OldBranchesExpireAndCanBeRelearned) {
  LinearPrefixAdmissionPolicy::Options options = admission_options(64, 2);
  options.observation_ttl = std::chrono::seconds(10);
  LinearPrefixAdmissionPolicy policy(4, options);
  const Clock::time_point start{};
  policy.observe(branch_hashes(2), start);
  policy.observe(branch_hashes(3), start + std::chrono::seconds(1));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4),
                                          start + std::chrono::seconds(2)),
            8u);
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4),
                                          start + std::chrono::seconds(12)),
            0u);
  policy.observe(branch_hashes(2), start + std::chrono::seconds(13));
  policy.observe(branch_hashes(3), start + std::chrono::seconds(14));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4),
                                          start + std::chrono::seconds(15)),
            8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     MatchingRefreshesExpiryWithoutObservation) {
  LinearPrefixAdmissionPolicy::Options options = admission_options(64, 2);
  options.observation_ttl = std::chrono::seconds(10);
  LinearPrefixAdmissionPolicy policy(4, options);
  const Clock::time_point start{};
  const std::vector<XXH3Key> first = branch_hashes(2);
  const std::vector<XXH3Key> second = branch_hashes(3);
  policy.observe(first, start);
  policy.observe(second, start + std::chrono::seconds(1));
  EXPECT_EQ(
      policy.find_checkpoint_tokens(first, start + std::chrono::seconds(5)),
      8u);
  EXPECT_EQ(
      policy.find_checkpoint_tokens(second, start + std::chrono::seconds(6)),
      8u);
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(4),
                                          start + std::chrono::seconds(11)),
            8u);
}

TEST(LinearPrefixAdmissionPolicyTest,
     EmptyObservationsDoNotAddDistinctSuccessors) {
  LinearPrefixAdmissionPolicy policy(4, admission_options(64, 3));
  policy.observe(branch_hashes(2));
  policy.observe(branch_hashes(3));
  policy.observe({});
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(5)), 0u);
  policy.observe(branch_hashes(2));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(5)), 0u);
  policy.observe(branch_hashes(4));
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(5)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest, ResetClearsTreeAndCanBeReused) {
  LinearPrefixAdmissionPolicy policy(4, admission_options());
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    policy.observe(branch_hashes(suffix));
  }
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 8u);
  policy.reset();
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 0u);
  EXPECT_EQ(policy.find_checkpoint_tokens({}), 0u);
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    policy.observe(branch_hashes(suffix));
  }
  EXPECT_EQ(policy.find_checkpoint_tokens(branch_hashes(6)), 8u);
}

TEST(LinearPrefixAdmissionPolicyTest, HotFrontAdmissionUsesProductionGeometry) {
  LinearPrefixAdmissionPolicy policy(128,
                                     {.chunk_size = 10240, .max_entries = 128});
  std::vector<int32_t> tokens(1000000, 2);
  std::fill_n(tokens.begin(), 8192, 1);
  const std::vector<XXH3Key> first = prompt_hashes(tokens, 128);
  std::fill(tokens.begin() + 8192, tokens.end(), 3);
  const std::vector<XXH3Key> second = prompt_hashes(tokens, 128);
  std::fill(tokens.begin() + 8192, tokens.end(), 4);
  const std::vector<XXH3Key> third = prompt_hashes(tokens, 128);
  std::fill(tokens.begin() + 8192, tokens.end(), 5);
  const std::vector<XXH3Key> fourth = prompt_hashes(tokens, 128);
  std::fill(tokens.begin() + 8192, tokens.end(), 6);
  const std::vector<XXH3Key> probe = prompt_hashes(tokens, 128);
  policy.observe(first);
  policy.observe(second);
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  policy.observe(first);
  policy.observe(third);
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), 0u);
  policy.observe(first);
  policy.observe(second);
  policy.observe(fourth);
  for (int32_t observation = 0; observation < 4; ++observation) {
    policy.observe(first);
    policy.observe(second);
    policy.observe(third);
    policy.observe(fourth);
    EXPECT_EQ(policy.find_checkpoint_tokens(probe), 8192u);
  }
}

TEST(LinearPrefixAdmissionPolicyTest, RejectsInvalidParameters) {
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(0, admission_options()),
               "block_size_");
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(4, {}), "chunk_size");
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(4, admission_options(0)),
               "max_entries");
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(4, admission_options(64, 0)),
               "min_distinct_successors");
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(4, admission_options(64, 1)),
               "min_distinct_successors");
  LinearPrefixAdmissionPolicy::Options options = admission_options();
  options.observation_ttl = std::chrono::seconds::zero();
  EXPECT_DEATH(LinearPrefixAdmissionPolicy(4, options), "observation_ttl");
}

TEST(LinearPrefixAdmissionPolicyTest, MatchesReferenceBranchStatistics) {
  using SuccessorSet =
      std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>;
  const LinearPrefixAdmissionPolicy::Options options = admission_options(4096);
  LinearPrefixAdmissionPolicy policy(4, options);
  std::unordered_map<XXH3Key,
                     SuccessorSet,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      observations;
  std::mt19937 generator(20261103);
  for (int32_t request_index = 0; request_index < 500; ++request_index) {
    const size_t block_count = static_cast<size_t>(generator() % 7 + 2);
    std::vector<int32_t> tokens(block_count * 4, 1);
    for (size_t block_index = 1; block_index < block_count; ++block_index) {
      const int32_t token = static_cast<int32_t>(generator() % 5 + 1);
      std::fill(tokens.begin() + block_index * 4,
                tokens.begin() + (block_index + 1) * 4,
                token);
    }
    const std::vector<XXH3Key> hashes = prompt_hashes(tokens);
    size_t expected_checkpoint = 0;
    for (size_t hash_index = 0; hash_index + 1 < hashes.size(); ++hash_index) {
      const size_t token_count = (hash_index + 1) * 4;
      if (token_count % static_cast<size_t>(options.chunk_size) == 0) {
        continue;
      }
      const auto found = observations.find(hashes[hash_index]);
      if (found == observations.end() ||
          found->second.size() < options.min_distinct_successors) {
        continue;
      }
      expected_checkpoint = token_count;
    }
    EXPECT_EQ(policy.find_checkpoint_tokens(hashes), expected_checkpoint)
        << "request_index=" << request_index;
    policy.observe(hashes);
    for (size_t hash_index = 0; hash_index + 1 < hashes.size(); ++hash_index) {
      observations[hashes[hash_index]].insert(hashes[hash_index + 1]);
    }
  }
}

TEST(LinearPrefixAdmissionPolicyTest, ConcurrentObservePlanAndReset) {
  constexpr size_t kBlockSize = 4;
  constexpr size_t kNumObservers = 4;
  constexpr int32_t kIterations = 64;
  constexpr size_t kCheckpointTokens = 8;
  LinearPrefixAdmissionPolicy policy(kBlockSize, admission_options());
  const std::vector<XXH3Key> probe = branch_hashes(kNumObservers + 2);
  std::atomic<bool> invalid_checkpoint{false};
  std::promise<void> start;
  const std::shared_future<void> ready = start.get_future().share();
  std::vector<std::thread> observers;
  observers.reserve(kNumObservers + 1);
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    std::vector<XXH3Key> hashes = branch_hashes(suffix);
    observers.emplace_back([&policy,
                            &probe,
                            &invalid_checkpoint,
                            ready,
                            hashes = std::move(hashes)] {
      ready.wait();
      for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        policy.observe(hashes);
        const size_t checkpoint = policy.find_checkpoint_tokens(probe);
        if (checkpoint != 0 && checkpoint != kCheckpointTokens) {
          invalid_checkpoint.store(true, std::memory_order_relaxed);
        }
      }
    });
  }
  observers.emplace_back([&policy, ready] {
    ready.wait();
    for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
      policy.reset();
    }
  });
  start.set_value();
  for (auto& observer : observers) {
    observer.join();
  }
  EXPECT_FALSE(invalid_checkpoint.load(std::memory_order_relaxed));
  for (int32_t suffix = 2; suffix < 6; ++suffix) {
    policy.observe(branch_hashes(suffix));
  }
  EXPECT_EQ(policy.find_checkpoint_tokens(probe), kCheckpointTokens);
}

TEST(LinearPrefixAdmissionPolicyTest, RejectsRegularChunkGeometry) {
  for (const size_t block_size : {4u, 8u}) {
    LinearPrefixAdmissionPolicy::Options options = admission_options();
    options.chunk_size = 4;
    EXPECT_DEATH(LinearPrefixAdmissionPolicy(block_size, options),
                 "block_size_ %");
  }
}

namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

BlockManagerPool::Options make_linear_state_pool_options(
    int32_t linear_state_num_slots) {
  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // The EMBEDDING pool is sized directly by num_embedding_blocks and is only
  // built under spec decode, so opt in here.
  options.max_seqs_per_batch(0)
      .num_embedding_blocks(4)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

std::vector<int32_t> make_hot_front_prompt(int32_t suffix,
                                           size_t prompt_tokens = 12,
                                           size_t shared_tokens = 8) {
  std::vector<int32_t> tokens(prompt_tokens, suffix);
  std::fill_n(tokens.begin(), shared_tokens, 1);
  return tokens;
}

class LinearPrefixCheckpointTest : public ::testing::Test {
 protected:
  LinearPrefixCheckpointTest()
      : policy_(std::make_unique<LinearPrefixAdmissionPolicy>(
            4,
            admission_options(65536))) {
    sampling_param_.beam_width = 0;
    sampling_param_.is_embeddings = false;
  }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_tokens,
                         bool is_graph_warmup = false) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.is_graph_warmup = is_graph_warmup;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "linear_prefix_checkpoint_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"prompt",
        /*num_prompt_tokens=*/prompt_tokens.size(),
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);

    return Sequence(index,
                    prompt_tokens,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    params);
  }

  bool prefill_prompt(BlockManagerPool& pool, Sequence& sequence) {
    const size_t prompt_tokens = sequence.num_prompt_tokens();
    policy_->plan_checkpoint(&sequence);
    if (!pool.allocate(&sequence, prompt_tokens)) {
      return false;
    }
    sequence.kv_state().set_kv_cache_tokens_num(prompt_tokens);
    pool.deallocate(&sequence);
    return true;
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  std::unique_ptr<LinearPrefixAdmissionPolicy> policy_;
};

}  // namespace

TEST_F(LinearPrefixCheckpointTest, IgnoresNullAndWarmupRequests) {
  EXPECT_EQ(policy_->plan_checkpoint(nullptr), 0u);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence warmup =
        make_sequence(suffix, make_hot_front_prompt(suffix), true);
    EXPECT_EQ(policy_->plan_checkpoint(&warmup), 0u);
    EXPECT_FALSE(warmup.linear_prefix_checkpoint_tokens().has_value());
  }
  Sequence probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest, DoesNotPlanCheckpointDuringDecode) {
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence completed = make_sequence(suffix, make_hot_front_prompt(suffix));
    policy_->plan_checkpoint(&completed);
  }
  Sequence decode = make_sequence(6, make_hot_front_prompt(6));
  decode.kv_state().set_kv_cache_tokens_num(decode.num_prompt_tokens());
  ASSERT_EQ(decode.stage(), SequenceStage::DECODE);
  EXPECT_EQ(policy_->plan_checkpoint(&decode), 0u);
  EXPECT_FALSE(decode.linear_prefix_checkpoint_tokens().has_value());
}

TEST_F(LinearPrefixCheckpointTest, PlansAndObservesBeforePromptExecution) {
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence sequence = make_sequence(suffix, make_hot_front_prompt(suffix));
    EXPECT_EQ(policy_->plan_checkpoint(&sequence), 0u);
    ASSERT_TRUE(sequence.linear_prefix_checkpoint_tokens().has_value());
    EXPECT_EQ(sequence.linear_prefix_checkpoint_tokens().value(), 0u);
    EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);
    EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  }
  Sequence probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  EXPECT_EQ(probe.kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(probe.kv_state().has_any_blocks());
}

TEST_F(LinearPrefixCheckpointTest, DpRankChangesPreserveCheckpointDecision) {
  for (const size_t checkpoint : {0u, 8u}) {
    Sequence sequence = make_sequence(0, make_hot_front_prompt(2));
    sequence.set_dp_rank(0);
    sequence.set_linear_prefix_checkpoint_tokens(checkpoint);
    sequence.set_dp_rank(1);
    ASSERT_TRUE(sequence.linear_prefix_checkpoint_tokens().has_value());
    EXPECT_EQ(sequence.linear_prefix_checkpoint_tokens().value(), checkpoint);
  }
}

TEST_F(LinearPrefixCheckpointTest, SequenceResetPreservesCheckpointDecision) {
  for (const size_t checkpoint : {0u, 8u}) {
    Sequence sequence = make_sequence(0, make_hot_front_prompt(2));
    sequence.set_linear_prefix_checkpoint_tokens(checkpoint);
    sequence.reset();
    ASSERT_TRUE(sequence.linear_prefix_checkpoint_tokens().has_value());
    EXPECT_EQ(sequence.linear_prefix_checkpoint_tokens().value(), checkpoint);
  }
}

TEST_F(LinearPrefixCheckpointTest, SequenceCopyPreservesCheckpointDecision) {
  for (const size_t checkpoint : {0u, 8u}) {
    Sequence sequence = make_sequence(0, make_hot_front_prompt(2));
    sequence.set_linear_prefix_checkpoint_tokens(checkpoint);
    Sequence sibling(sequence, 1);
    ASSERT_TRUE(sibling.linear_prefix_checkpoint_tokens().has_value());
    EXPECT_EQ(sibling.linear_prefix_checkpoint_tokens().value(), checkpoint);
  }
}

TEST_F(LinearPrefixCheckpointTest,
       PublishesHotFrontCheckpointBeforeDivergentTail) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool::Options options = make_linear_state_pool_options(32);
  options.num_blocks(64);
  BlockManagerPool pool(options, 1);
  auto make_prompt = [](int32_t suffix) {
    return make_hot_front_prompt(suffix, /*prompt_tokens=*/43);
  };

  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence finished = make_sequence(suffix, make_prompt(suffix));
    policy_->plan_checkpoint(&finished);
    for (const size_t target_tokens : {16u, 32u, 40u, 43u}) {
      ASSERT_TRUE(pool.allocate(&finished, target_tokens));
      finished.kv_state().set_kv_cache_tokens_num(target_tokens);
    }
    pool.deallocate(&finished);
  }

  Sequence publishing = make_sequence(6, make_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&publishing), 8u);
  for (const size_t target_tokens : {8u, 16u, 32u, 40u, 43u}) {
    ASSERT_TRUE(pool.allocate(&publishing, target_tokens));
    publishing.kv_state().set_kv_cache_tokens_num(target_tokens);
  }
  pool.deallocate(&publishing);

  Sequence hit = make_sequence(7, make_prompt(7));
  pool.allocate_shared(&hit);
  EXPECT_EQ(hit.kv_cache_tokens_num(), 8u);
  EXPECT_EQ(hit.kv_state().shared_blocks_num(BlockType::KV), 2u);
  EXPECT_EQ(hit.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(hit.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(hit.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_NE(hit.kv_cache_tokens_num(), 16u);
  EXPECT_NE(hit.kv_cache_tokens_num(), 40u);
  hit.reset();
}

TEST_F(LinearPrefixCheckpointTest, HotFrontIncludesBlockAlignedShortSuffix) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
    Sequence candidate = make_sequence(suffix + 6, make_hot_front_prompt(2));
    EXPECT_EQ(policy_->plan_checkpoint(&candidate), suffix < 5 ? 0u : 8u);
  }
  Sequence probe = make_sequence(6, make_hot_front_prompt(/*suffix=*/6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       IdenticalCompletedPromptsDoNotCreateCheckpoint) {
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  for (int32_t request_index = 0; request_index < 2; ++request_index) {
    Sequence observed = make_sequence(request_index, make_hot_front_prompt(2));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  Sequence probe = make_sequence(2, make_hot_front_prompt(3));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest,
       ResidentLinearCheckpointDoesNotChangePrediction) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      /*value=*/16);
  BlockManagerPool pool(
      make_linear_state_pool_options(/*linear_state_num_slots=*/32),
      /*dp_size=*/1);
  for (int32_t suffix : {2, 3, 4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }

  Sequence publishing =
      make_sequence(/*index=*/6, make_hot_front_prompt(/*suffix=*/6));
  ASSERT_EQ(policy_->plan_checkpoint(&publishing), 8u);
  for (size_t target_tokens : {8u, 12u}) {
    ASSERT_TRUE(pool.allocate(&publishing, target_tokens));
    publishing.kv_state().set_kv_cache_tokens_num(target_tokens);
  }
  policy_->plan_checkpoint(&publishing);
  pool.deallocate(&publishing);

  Sequence probe =
      make_sequence(/*index=*/7, make_hot_front_prompt(/*suffix=*/7));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(probe.linear_prefix_checkpoint_tokens().value(), 8u);
  pool.allocate_shared(&probe);
  EXPECT_EQ(probe.kv_cache_tokens_num(), 8u);
  BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0)
      ->reset_prefix_cache();
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  probe.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest, HotFrontPredictionSurvivesKvEviction) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  Sequence before_eviction = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&before_eviction), 8u);

  BlockManager* kv_leaf =
      BlockManagerPoolTestPeer::leaf_of(pool, 0, BlockType::KV);
  ASSERT_NE(kv_leaf, nullptr);
  kv_leaf->reset_prefix_cache();

  Sequence after_eviction = make_sequence(7, make_hot_front_prompt(7));
  EXPECT_EQ(policy_->plan_checkpoint(&after_eviction), 8u);
  EXPECT_EQ(policy_->plan_checkpoint(&before_eviction), 8u);
  EXPECT_FALSE(after_eviction.kv_state().has_any_blocks());
  EXPECT_FALSE(after_eviction.kv_state().prefix_cache_matched());
  pool.allocate_shared(&after_eviction);
  EXPECT_EQ(after_eviction.kv_cache_tokens_num(), 0u);
}

TEST_F(LinearPrefixCheckpointTest,
       HotFrontUsesInjectedOptionsInsteadOfSchedulerFlags) {
  ScopedValue<bool> chunked_guard(
      &SchedulerConfig::get_instance().enable_chunked_prefill(), false);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  Sequence probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       PredictionRequiresHotBranchNotResidentBlocks) {
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  Sequence producer = make_sequence(0, make_hot_front_prompt(2));
  ASSERT_TRUE(prefill_prompt(pool, producer));
  EXPECT_TRUE(producer.linear_prefix_checkpoint_tokens().has_value());
  Sequence consumer = make_sequence(1, make_hot_front_prompt(3));
  EXPECT_EQ(policy_->plan_checkpoint(&consumer), 0u);
  ASSERT_TRUE(prefill_prompt(pool, consumer));
  for (int32_t suffix : {4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  Sequence hot_probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
  EXPECT_FALSE(hot_probe.kv_state().has_any_blocks());
  EXPECT_FALSE(hot_probe.kv_state().prefix_cache_matched());
  EXPECT_EQ(hot_probe.kv_cache_tokens_num(), 0u);

  BlockManagerPoolTestPeer::leaf_of(pool, 0, BlockType::KV)
      ->reset_prefix_cache();
  Sequence missing = make_sequence(7, make_hot_front_prompt(7));
  EXPECT_EQ(policy_->plan_checkpoint(&missing), 8u);
}

TEST_F(LinearPrefixCheckpointTest, HotFilterCapacityAndThresholdAreInjected) {
  for (const bool limit_capacity : {false, true}) {
    SCOPED_TRACE(limit_capacity);
    BlockManagerPool::Options options = make_linear_state_pool_options(32);
    policy_ = std::make_unique<LinearPrefixAdmissionPolicy>(
        4,
        LinearPrefixAdmissionPolicy::Options{
            .chunk_size = 16,
            .max_entries = limit_capacity ? 1u : 64u,
            .min_distinct_successors = limit_capacity ? 2u : 3u});
    BlockManagerPool pool(options, 1);
    for (int32_t suffix = 2; suffix <= 3; ++suffix) {
      Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
      ASSERT_TRUE(prefill_prompt(pool, observed));
    }
    Sequence cold = make_sequence(4, make_hot_front_prompt(4));
    EXPECT_EQ(policy_->plan_checkpoint(&cold), 0u);
    Sequence third = make_sequence(5, make_hot_front_prompt(5));
    ASSERT_TRUE(prefill_prompt(pool, third));
    Sequence probe = make_sequence(6, make_hot_front_prompt(6));
    EXPECT_EQ(policy_->plan_checkpoint(&probe), limit_capacity ? 0u : 8u);
  }
}

TEST_F(LinearPrefixCheckpointTest, RepeatedPlanningDoesNotAdmitOnePromptTwice) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool::Options options = make_linear_state_pool_options(32);
  options.num_blocks(64);
  BlockManagerPool pool(options, 1);
  std::vector<int32_t> tokens =
      make_hot_front_prompt(/*suffix=*/2, /*prompt_tokens=*/40);
  Sequence observed = make_sequence(0, tokens);
  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    EXPECT_EQ(policy_->plan_checkpoint(&observed), 0u);
    ASSERT_TRUE(pool.allocate(&observed, tokens.size()));
    observed.kv_state().set_kv_cache_tokens_num(tokens.size());
    pool.deallocate(&observed);
  }
  std::fill(tokens.begin() + 8, tokens.end(), 3);
  Sequence probe = make_sequence(1, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest, StatisticsResetPreservesHotFrontDecision) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  for (int32_t suffix : {2, 3, 4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  std::fill(tokens.begin() + 8, tokens.end(), 6);
  Sequence probe = make_sequence(6, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  pool.reset_prefix_cache();
  policy_->reset();
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  Sequence fresh_probe = make_sequence(/*index=*/7, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&fresh_probe), 0u);
  probe.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       StatisticsResetDoesNotReobserveExistingRequests) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      /*value=*/16);
  BlockManagerPool pool(
      make_linear_state_pool_options(/*linear_state_num_slots=*/32),
      /*dp_size=*/1);
  const std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  Sequence first = make_sequence(/*index=*/0, tokens);
  Sequence second = make_sequence(1, make_hot_front_prompt(3));
  Sequence third = make_sequence(2, make_hot_front_prompt(4));
  Sequence fourth = make_sequence(3, make_hot_front_prompt(5));
  for (Sequence* observed : {&first, &second, &third, &fourth}) {
    ASSERT_TRUE(prefill_prompt(pool, *observed));
  }
  pool.reset_prefix_cache();
  policy_->reset();
  for (Sequence* observed : {&first, &second, &third, &fourth}) {
    ASSERT_TRUE(prefill_prompt(pool, *observed));
  }

  Sequence probe = make_sequence(4, make_hot_front_prompt(2));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  const std::vector<XXH3Key> hashes = branch_hashes(6, 8, 12);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence fresh = make_sequence(suffix + 3, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, fresh));
    EXPECT_EQ(policy_->find_checkpoint_tokens(hashes), suffix < 5 ? 0u : 8u);
  }
  probe.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  Sequence fresh_probe = make_sequence(9, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&fresh_probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest, HotFrontObservationsCombineAcrossDpRanks) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), /*dp_size=*/2);
  for (size_t round = 0; round < 4; ++round) {
    for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
      const std::vector<int32_t> tokens =
          make_hot_front_prompt(static_cast<int32_t>(round * 2) + dp_rank + 2);
      Sequence observed = make_sequence(round * 2 + dp_rank, tokens);
      observed.set_dp_rank(dp_rank);
      ASSERT_TRUE(prefill_prompt(pool, observed));
    }
    const std::vector<int32_t> probe_tokens = make_hot_front_prompt(2);
    for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
      Sequence probe = make_sequence(8 + round * 2 + dp_rank, probe_tokens);
      probe.set_dp_rank(dp_rank);
      EXPECT_EQ(policy_->plan_checkpoint(&probe), round < 1 ? 0u : 8u);
    }
  }
}

TEST_F(LinearPrefixCheckpointTest,
       HotFrontStatisticsAreSharedAcrossDpByDefault) {
  constexpr int32_t kChunkSize = 16;
  constexpr int32_t kLinearSlots = 32;
  constexpr int32_t kDpSize = 2;
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkSize);
  BlockManagerPool::Options options =
      make_linear_state_pool_options(kLinearSlots);
  BlockManagerPool pool(options, kDpSize);
  for (int32_t observation = 0; observation < 4; ++observation) {
    Sequence observed =
        make_sequence(observation, make_hot_front_prompt(observation + 2));
    observed.set_dp_rank(observation % kDpSize);
    ASSERT_TRUE(prefill_prompt(pool, observed));
    for (int32_t dp_rank = 0; dp_rank < kDpSize; ++dp_rank) {
      Sequence probe = make_sequence(4 + dp_rank, make_hot_front_prompt(2));
      probe.set_dp_rank(dp_rank);
      EXPECT_EQ(policy_->plan_checkpoint(&probe), observation < 3 ? 0u : 8u);
    }
  }
}

TEST_F(LinearPrefixCheckpointTest,
       SharedHotFrontStatisticsDoNotShareCachedBlocks) {
  constexpr int32_t kChunkSize = 16;
  constexpr int32_t kLinearSlots = 32;
  constexpr int32_t kDpSize = 2;
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkSize);
  BlockManagerPool::Options options =
      make_linear_state_pool_options(kLinearSlots);
  BlockManagerPool pool(options, kDpSize);
  for (int32_t suffix : {2, 3, 4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    observed.set_dp_rank(0);
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }

  Sequence cold = make_sequence(6, make_hot_front_prompt(6));
  cold.set_dp_rank(1);
  ASSERT_EQ(policy_->plan_checkpoint(&cold), 8u);
  pool.allocate_shared(&cold);
  EXPECT_EQ(cold.kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(cold.kv_state().has_any_blocks());
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[1], 0u);

  const size_t checkpoint = policy_->plan_checkpoint(&cold);
  ASSERT_TRUE(pool.allocate(&cold, checkpoint));
  cold.kv_state().set_kv_cache_tokens_num(checkpoint);
  ASSERT_TRUE(pool.allocate(&cold, cold.num_prompt_tokens()));
  cold.kv_state().set_kv_cache_tokens_num(cold.num_prompt_tokens());
  pool.deallocate(&cold);

  Sequence local_hit = make_sequence(7, make_hot_front_prompt(7));
  local_hit.set_dp_rank(1);
  pool.allocate_shared(&local_hit);
  EXPECT_EQ(local_hit.kv_cache_tokens_num(), checkpoint);
  pool.deallocate(&local_hit);
}

TEST_F(LinearPrefixCheckpointTest, SharedHotFrontDecisionSurvivesDpRankChange) {
  constexpr int32_t kChunkSize = 16;
  constexpr int32_t kLinearSlots = 32;
  constexpr int32_t kDpSize = 2;
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkSize);
  BlockManagerPool::Options options =
      make_linear_state_pool_options(kLinearSlots);
  BlockManagerPool pool(options, kDpSize);
  Sequence probe = make_sequence(4, make_hot_front_prompt(6));
  probe.set_dp_rank(0);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  for (int32_t observation = 0; observation < 4; ++observation) {
    Sequence observed =
        make_sequence(observation, make_hot_front_prompt(observation + 2));
    observed.set_dp_rank(observation % kDpSize);
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  probe.set_dp_rank(1);
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(probe.linear_prefix_checkpoint_tokens().value(), 0u);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  Sequence fresh_probe = make_sequence(5, make_hot_front_prompt(7));
  fresh_probe.set_dp_rank(1);
  EXPECT_EQ(policy_->plan_checkpoint(&fresh_probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       HotFrontPlanUsesDeepestSharedPrefixAcrossDpRanks) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), /*dp_size=*/2);
  for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
    const size_t shared_tokens = dp_rank == 0 ? 8 : 4;
    for (int32_t suffix : {2, 3, 4, 5}) {
      const std::vector<int32_t> tokens =
          make_hot_front_prompt(suffix, /*prompt_tokens=*/12, shared_tokens);
      Sequence observed = make_sequence(suffix, tokens);
      observed.set_dp_rank(dp_rank);
      ASSERT_TRUE(prefill_prompt(pool, observed));
    }
  }

  Sequence probe = make_sequence(0, make_hot_front_prompt(/*suffix=*/6));
  probe.set_dp_rank(0);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  probe.set_dp_rank(1);
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  probe.set_dp_rank(1);
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
  probe.set_dp_rank(0);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest, RepeatedPlanningObservesPromptOnlyOnce) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  Sequence observed = make_sequence(0, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&observed), 0u);
  ASSERT_TRUE(pool.allocate(&observed, tokens.size()));
  observed.kv_state().set_kv_cache_tokens_num(tokens.size());
  policy_->plan_checkpoint(&observed);
  pool.cache(&observed);
  policy_->plan_checkpoint(&observed);
  pool.cache(&observed);
  policy_->plan_checkpoint(&observed);
  pool.deallocate(&observed);
  std::fill(tokens.begin() + 8, tokens.end(), 3);
  Sequence probe = make_sequence(1, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest,
       PlannedSiblingRetainsDecisionAfterStatisticsReset) {
  Sequence observed =
      make_sequence(/*index=*/0, make_hot_front_prompt(/*suffix=*/2));
  EXPECT_EQ(policy_->plan_checkpoint(&observed), 0u);
  Sequence sibling(observed, /*index=*/1);
  policy_->reset();
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence fresh = make_sequence(suffix, make_hot_front_prompt(suffix));
    EXPECT_EQ(policy_->plan_checkpoint(&fresh), 0u);
  }
  EXPECT_EQ(policy_->plan_checkpoint(&sibling), 0u);
  Sequence probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       ConcurrentPlannedSiblingsDoNotReobservePrompt) {
  policy_ = std::make_unique<LinearPrefixAdmissionPolicy>(
      4,
      LinearPrefixAdmissionPolicy::Options{.chunk_size = 16,
                                           .min_distinct_successors = 3});
  Sequence original =
      make_sequence(/*index=*/1, make_hot_front_prompt(/*suffix=*/3));
  EXPECT_EQ(policy_->plan_checkpoint(&original), 0u);
  policy_->reset();
  constexpr size_t kNumSiblings = 8;
  std::vector<std::unique_ptr<Sequence>> siblings;
  siblings.reserve(kNumSiblings);
  for (size_t index = 0; index < kNumSiblings; ++index) {
    siblings.emplace_back(std::make_unique<Sequence>(original, index + 2));
  }

  std::latch ready(kNumSiblings);
  std::latch start(/*expected=*/1);
  std::vector<std::thread> workers;
  workers.reserve(kNumSiblings);
  for (const auto& sibling : siblings) {
    workers.emplace_back([this, &ready, &start, sequence = sibling.get()] {
      ready.count_down();
      start.wait();
      EXPECT_EQ(policy_->plan_checkpoint(sequence), 0u);
    });
  }
  ready.wait();
  start.count_down();
  for (auto& worker : workers) {
    worker.join();
  }

  Sequence probe =
      make_sequence(kNumSiblings + 2, make_hot_front_prompt(/*suffix=*/4));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  Sequence independent =
      make_sequence(kNumSiblings + 3, make_hot_front_prompt(/*suffix=*/5));
  EXPECT_EQ(policy_->plan_checkpoint(&independent), 0u);
  Sequence third = make_sequence(kNumSiblings + 4, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&third), 0u);
  Sequence hot_probe =
      make_sequence(kNumSiblings + 5, make_hot_front_prompt(7));
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       UnfinishedPromptsContributeSuccessorsDuringPlanning) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  for (int32_t suffix = 2; suffix <= 5; ++suffix) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    EXPECT_EQ(policy_->plan_checkpoint(&observed), 0u);
    ASSERT_TRUE(pool.allocate(&observed, 4));
    observed.kv_state().set_kv_cache_tokens_num(4);
    policy_->plan_checkpoint(&observed);
    pool.cache(&observed);
    policy_->plan_checkpoint(&observed);
    pool.deallocate(&observed);
  }
  Sequence probe = make_sequence(6, make_hot_front_prompt(6));
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
}

TEST_F(LinearPrefixCheckpointTest,
       HotFrontDecisionStaysStableAcrossSequenceReset) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  std::fill(tokens.begin() + 8, tokens.end(), 6);
  Sequence probe = make_sequence(6, tokens);
  EXPECT_FALSE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  pool.reset_prefix_cache();
  policy_->reset();
  for (int32_t suffix : {2, 3, 4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
    EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  }
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(probe.linear_prefix_checkpoint_tokens().value(), 0u);
  probe.reset();
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  Sequence hot_probe = make_sequence(7, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
  pool.reset_prefix_cache();
  policy_->reset();
  hot_probe.reset();
  ASSERT_TRUE(hot_probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
  EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
  ASSERT_TRUE(probe.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(probe.linear_prefix_checkpoint_tokens().value(), 0u);
}

TEST_F(LinearPrefixCheckpointTest, BeamSiblingsDoNotAddDuplicateSuccessors) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  Sequence first = make_sequence(0, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&first), 0u);
  Sequence sibling(first, 1);
  for (Sequence* observed : {&first, &sibling}) {
    EXPECT_EQ(policy_->plan_checkpoint(observed), 0u);
    ASSERT_TRUE(pool.allocate(observed, tokens.size()));
    observed->kv_state().set_kv_cache_tokens_num(tokens.size());
    pool.deallocate(observed);
  }
  std::fill(tokens.begin() + 8, tokens.end(), 3);
  Sequence cold_probe = make_sequence(2, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&cold_probe), 0u);
  Sequence independent = make_sequence(3, tokens);
  ASSERT_TRUE(prefill_prompt(pool, independent));
  for (int32_t suffix : {4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  std::fill(tokens.begin() + 8, tokens.end(), 6);
  Sequence hot_probe = make_sequence(6, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);

  pool.reset_prefix_cache();
  policy_->reset();
  for (Sequence* observed : {&first, &sibling}) {
    ASSERT_TRUE(prefill_prompt(pool, *observed));
  }
  Sequence fresh = make_sequence(/*index=*/7, tokens);
  ASSERT_TRUE(prefill_prompt(pool, fresh));
  for (int32_t suffix : {4, 5}) {
    Sequence observed =
        make_sequence(suffix + 4, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  hot_probe.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
  Sequence fresh_probe = make_sequence(10, make_hot_front_prompt(7));
  EXPECT_EQ(policy_->plan_checkpoint(&fresh_probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest, ParallelSamplesDoNotAddDuplicateSuccessors) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  BlockManagerPool pool(make_linear_state_pool_options(32), 1);
  std::vector<int32_t> tokens = make_hot_front_prompt(/*suffix=*/2);
  SequenceParams params;
  params.seq_capacity = 20;
  params.best_of = 2;
  params.sampling_param = &sampling_param_;
  params.stopping_checker = &stopping_checker_;
  SequencesGroup group("prompt", tokens, torch::Tensor(), MMData(), params);
  ASSERT_TRUE(group.expand_sequences(false));
  ASSERT_EQ(group.sequences().size(), 2u);
  for (const auto& observed : group.sequences()) {
    EXPECT_EQ(policy_->plan_checkpoint(observed.get()), 0u);
    ASSERT_TRUE(pool.allocate(observed.get(), tokens.size()));
    observed->kv_state().set_kv_cache_tokens_num(tokens.size());
    pool.deallocate(observed.get());
  }
  std::fill(tokens.begin() + 8, tokens.end(), 3);
  Sequence cold_probe = make_sequence(2, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&cold_probe), 0u);
  Sequence independent = make_sequence(3, tokens);
  ASSERT_TRUE(prefill_prompt(pool, independent));
  for (int32_t suffix : {4, 5}) {
    Sequence observed = make_sequence(suffix, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  std::fill(tokens.begin() + 8, tokens.end(), 6);
  Sequence hot_probe = make_sequence(6, tokens);
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);

  pool.reset_prefix_cache();
  policy_->reset();
  for (const auto& observed : group.sequences()) {
    ASSERT_TRUE(prefill_prompt(pool, *observed));
  }
  Sequence fresh = make_sequence(/*index=*/7, tokens);
  ASSERT_TRUE(prefill_prompt(pool, fresh));
  for (int32_t suffix : {4, 5}) {
    Sequence observed =
        make_sequence(suffix + 4, make_hot_front_prompt(suffix));
    ASSERT_TRUE(prefill_prompt(pool, observed));
  }
  hot_probe.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&hot_probe), 8u);
  Sequence fresh_probe = make_sequence(10, make_hot_front_prompt(7));
  EXPECT_EQ(policy_->plan_checkpoint(&fresh_probe), 0u);
}

TEST_F(LinearPrefixCheckpointTest,
       HotFrontUsesInjectedHasherWithoutClearingOtherStrides) {
  for (BlockHasherType hasher_type : {BlockHasherType::TEXT,
                                      BlockHasherType::MM,
                                      BlockHasherType::MTP_TEXT,
                                      BlockHasherType::MTP_MM}) {
    policy_ = std::make_unique<LinearPrefixAdmissionPolicy>(
        4,
        LinearPrefixAdmissionPolicy::Options{.chunk_size = 16,
                                             .hasher_type = hasher_type});
    std::vector<int32_t> tokens(20, 2);
    std::fill_n(tokens.begin(), 9, 1);
    Sequence first = make_sequence(0, tokens);
    std::fill(tokens.begin() + 9, tokens.end(), 3);
    Sequence second = make_sequence(1, tokens);
    std::fill(tokens.begin() + 9, tokens.end(), 4);
    Sequence third = make_sequence(2, tokens);
    std::fill(tokens.begin() + 9, tokens.end(), 5);
    Sequence fourth = make_sequence(3, tokens);
    first.update_block_hashes(8, hasher_type);
    ASSERT_EQ(first.block_hashes(8).size(), 2u);
    const XXH3Key original_hash = first.block_hashes(8)[0];
    for (Sequence* observed : {&first, &second, &third, &fourth}) {
      policy_->plan_checkpoint(observed);
      observed->kv_state().set_kv_cache_tokens_num(tokens.size());
    }
    ASSERT_EQ(first.block_hashes(8).size(), 2u);
    EXPECT_TRUE(FixedStringKeyEqual{}(first.block_hashes(8)[0], original_hash));
    std::fill(tokens.begin() + 9, tokens.end(), 6);
    Sequence probe = make_sequence(4, tokens);
    probe.update_block_hashes(8, hasher_type);
    EXPECT_EQ(policy_->plan_checkpoint(&probe), 8u);
    ASSERT_EQ(probe.block_hashes(8).size(), 2u);
    EXPECT_TRUE(FixedStringKeyEqual{}(probe.block_hashes(8)[0], original_hash));
  }
}

TEST_F(LinearPrefixCheckpointTest, MtpHotFrontExcludesUnavailableLookahead) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 16);
  for (BlockHasherType hasher_type :
       {BlockHasherType::MTP_TEXT, BlockHasherType::MTP_MM}) {
    BlockManagerPool::Options options = make_linear_state_pool_options(32);
    options.hasher_type(hasher_type);
    BlockManagerPool pool(options, 1);
    policy_ = std::make_unique<LinearPrefixAdmissionPolicy>(
        4,
        LinearPrefixAdmissionPolicy::Options{.chunk_size = 16,
                                             .hasher_type = hasher_type});
    Sequence first = make_sequence(0, std::vector<int32_t>(8, 1));
    Sequence second = make_sequence(1, std::vector<int32_t>(8, 1));
    for (Sequence* observed : {&first, &second}) {
      policy_->plan_checkpoint(observed);
      ASSERT_TRUE(pool.allocate(observed, 8));
      observed->kv_state().set_kv_cache_tokens_num(8);
      pool.cache(observed);
      EXPECT_EQ(observed->block_hashes(4).size(), 1u);
    }
    Sequence probe = make_sequence(2, std::vector<int32_t>(8, 1));
    EXPECT_EQ(policy_->plan_checkpoint(&probe), 0u);
    EXPECT_EQ(probe.block_hashes(4).size(), 1u);
    policy_->plan_checkpoint(&first);
    pool.deallocate(&first);
    policy_->plan_checkpoint(&second);
    pool.deallocate(&second);
  }
}

TEST_F(LinearPrefixCheckpointTest,
       GeneratedTokenUpdatesPreserveCheckpointDecision) {
  Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 1));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(2);
  sequence.set_linear_prefix_checkpoint_tokens(/*token_count=*/8);
  sequence.update_token(sequence.num_prompt_tokens(), Token(3));
  ASSERT_TRUE(sequence.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(sequence.linear_prefix_checkpoint_tokens().value(), 8u);
}

TEST_F(LinearPrefixCheckpointTest, CachedNoStopPlanSurvivesResetAndDpChanges) {
  Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 1));
  EXPECT_FALSE(sequence.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&sequence), 0u);
  ASSERT_TRUE(sequence.linear_prefix_checkpoint_tokens().has_value());
  sequence.set_dp_rank(0);
  EXPECT_EQ(policy_->plan_checkpoint(&sequence), 0u);
  sequence.set_dp_rank(1);
  EXPECT_EQ(policy_->plan_checkpoint(&sequence), 0u);
  sequence.reset();
  EXPECT_EQ(policy_->plan_checkpoint(&sequence), 0u);
  Sequence independent = make_sequence(
      /*index=*/1, std::vector<int32_t>(/*count=*/12, /*value=*/1));
  EXPECT_FALSE(independent.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(policy_->plan_checkpoint(&independent), 0u);
  ASSERT_TRUE(independent.linear_prefix_checkpoint_tokens().has_value());
  EXPECT_EQ(independent.linear_prefix_checkpoint_tokens().value(), 0u);
}
}  // namespace xllm
