/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/block/block_manager_pool.h"
#include "core/runtime/native_provider_runtime.h"

namespace xllm {
namespace {

TEST(NativeProviderCacheGroups, DistinguishesV41CompressedGroups) {
  BlockManagerPool::Options options;
  options.model_type("deepseek_v41")
      .num_blocks(64)
      .block_size(32)
      .enable_prefix_cache(false)
      .sliding_window_size(128)
      .swa_blocks_per_seq(4)
      .swa_num_blocks(32)
      .max_seqs_per_batch(2)
      .max_tokens_per_batch(64)
      .manager_types({1, 0, 0})
      .compress_ratios({0, 2, 1});
  const BlockManagerPool pool(options);
  EXPECT_EQ(native_provider_cache_groups(pool),
            (std::vector<std::string>{"group:1", "group:6", "group:7"}));
}

TEST(NativeProviderCacheGroups, IncludesOrdinaryKvAndSpeculativeEmbedding) {
  BlockManagerPool::Options options;
  options.num_blocks(32)
      .block_size(32)
      .enable_prefix_cache(false)
      .num_speculative_tokens(1)
      .num_embedding_blocks(8);
  const BlockManagerPool pool(options);
  EXPECT_EQ(native_provider_cache_groups(pool),
            (std::vector<std::string>{"group:0", "group:4"}));
}

TEST(NativeProviderCacheGroups, OrdinaryKvHasOneGroupAcrossDpRanks) {
  BlockManagerPool::Options options;
  options.num_blocks(32).block_size(32).enable_prefix_cache(false);
  const BlockManagerPool pool(options, /*dp_size=*/2);
  EXPECT_EQ(native_provider_cache_groups(pool),
            (std::vector<std::string>{"group:0"}));
}

}  // namespace
}  // namespace xllm
