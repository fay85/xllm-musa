/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/runtime/deepseek_v41_execution_contract.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_estimation.h"
#include "framework/kv_cache/kv_cache_shape.h"

namespace xllm {
namespace {

class ScopedMode final {
 public:
  ScopedMode(std::string name, const char* value) : name_(std::move(name)) {
    const char* previous = std::getenv(name_.c_str());
    if (previous != nullptr) {
      previous_ = previous;
    }
    setenv(name_.c_str(), value, 1);
  }
  ~ScopedMode() {
    if (previous_.has_value()) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

TEST(DeepseekV41ExecutionContract, DenseOriginalMustNotShareLegacyNamespace) {
  ScopedMode expert("XLLM_DSV41_MUSA_FP4_LINEAR", "reference");
  ScopedMode engram("XLLM_DSV41_MUSA_ENGRAM_GATE", "reference");
  ScopedMode dense("XLLM_DSV41_MUSA_DENSE_QUANT", "reference");
  const std::string legacy = deepseek_v41_execution_contract(/*is_v41=*/true);
  {
    ScopedMode original("XLLM_DSV41_MUSA_DENSE_QUANT", "original_qdq");
    EXPECT_NE(deepseek_v41_execution_contract(/*is_v41=*/true), legacy);
    EXPECT_TRUE(deepseek_v41_execution_contract(/*is_v41=*/false).empty());
  }
  EXPECT_EQ(deepseek_v41_execution_contract(/*is_v41=*/true), legacy);
}

TEST(DeepseekV41ExecutionContract,
     EquivalentImplementationsKeepSemanticIdentity) {
  ScopedMode dense("XLLM_DSV41_MUSA_DENSE_QUANT", "original_qdq");
  ScopedMode expert("XLLM_DSV41_MUSA_FP4_LINEAR", "block32");
  ScopedMode engram("XLLM_DSV41_MUSA_ENGRAM_GATE", "original");
  const std::string original = deepseek_v41_execution_contract(/*is_v41=*/true);
  ScopedMode matrix("XLLM_DSV41_MUSA_FP4_LINEAR", "partition32");
  ScopedMode fused("XLLM_DSV41_MUSA_ENGRAM_GATE", "sglang_fused");
  EXPECT_EQ(deepseek_v41_execution_contract(/*is_v41=*/true), original);
}

TEST(DeepseekV41ExecutionContract, InvalidModeFailsBeforeServing) {
  ScopedMode invalid("XLLM_DSV41_MUSA_ENGRAM_GATE", "typo");
  EXPECT_DEATH(deepseek_v41_execution_contract(/*is_v41=*/true),
               "Check failed");
  EXPECT_TRUE(deepseek_v41_execution_contract(/*is_v41=*/false).empty());
}

class ScopedAttentionQuant final {
 public:
  explicit ScopedAttentionQuant(const char* mode) {
    const char* previous = std::getenv("XLLM_DSV41_MUSA_ATTENTION_QUANT");
    if (previous != nullptr) {
      previous_ = previous;
    }
    setenv("XLLM_DSV41_MUSA_ATTENTION_QUANT", mode, 1);
  }
  ~ScopedAttentionQuant() {
    if (previous_.has_value()) {
      setenv("XLLM_DSV41_MUSA_ATTENTION_QUANT", previous_->c_str(), 1);
    } else {
      unsetenv("XLLM_DSV41_MUSA_ATTENTION_QUANT");
    }
  }

 private:
  std::optional<std::string> previous_;
};

ModelArgs model_args() {
  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(64)
      .index_head_dim(32)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});
  return args;
}

KVCacheCreateOptions create_options() {
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(6)
      .model_type("deepseek_v41")
      .block_size(128)
      .head_dim(64)
      .index_head_dim(32)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});
  return options;
}

TEST(DeepseekV41OriginalCache, ModeDoesNotChangeV4) {
  ScopedAttentionQuant mode("original_qdq");
  const auto legacy = get_dsv4_cache_policy(torch::kBFloat16);
  EXPECT_EQ(legacy.index_dtype, torch::kInt8);
  EXPECT_TRUE(legacy.has_indexer_cache_scale);
  const auto original =
      get_dsv4_cache_policy(torch::kBFloat16, /*is_v41=*/true);
  EXPECT_EQ(original.index_dtype, torch::kBFloat16);
  EXPECT_EQ(original.index_dtype_size, 2);
  EXPECT_FALSE(original.has_indexer_cache_scale);
  EXPECT_EQ(original.scale_dtype_size, 0);
}

TEST(DeepseekV41OriginalCache, ActualPoolsAndSwapKeepCarrier) {
  ScopedAttentionQuant mode("original_qdq");
  KVCacheCapacity capacity;
  capacity.block_size(128).swa_count(5).c2_count(3).c1_count(6);
  const KVCacheShape shape(capacity, model_args(), /*world_size=*/1);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, create_options());
  ASSERT_EQ(caches.size(), 6);
  for (const int64_t layer : {2, 4}) {
    EXPECT_EQ(caches[layer].get_index_cache().scalar_type(), torch::kBFloat16);
    EXPECT_FALSE(caches[layer].get_indexer_cache_scale().has_value());
    const BlockType group = layer == 2 ? BlockType::C2 : BlockType::C1;
    const auto tensors = caches[layer].get_block_type_tensors(group);
    ASSERT_EQ(tensors.count(KVCacheTensorRole::INDEX), 1);
    EXPECT_EQ(tensors.count(KVCacheTensorRole::INDEX_SCALE), 0);
    EXPECT_EQ(tensors.at(KVCacheTensorRole::INDEX).element_size(), 2);
  }
  auto index = caches[2].get_index_cache();
  index.zero_();
  index.select(/*dim=*/0, /*index=*/0).fill_(-75);
  index.select(/*dim=*/0, /*index=*/1).fill_(60);
  torch::Tensor source = torch::tensor({0}, torch::kInt64);
  torch::Tensor target = torch::tensor({1}, torch::kInt64);
  caches[2].swap_blocks(source, target);
  const torch::Tensor restored = caches[2].get_index_cache();
  EXPECT_TRUE(restored.select(0, 1).eq(-75).all().item<bool>());
  EXPECT_TRUE(restored.select(0, 0).eq(-75).all().item<bool>());
}

TEST(DeepseekV41OriginalCache, EstimatorCoversActualTensorBytes) {
  ScopedAttentionQuant mode("original_qdq");
  KVCacheEstimateOptions estimate;
  estimate.dtype = torch::kBFloat16;
  estimate.block_size = 128;
  estimate.max_seqs_per_batch = 2;
  estimate.max_concurrent_requests = 2;
  estimate.max_tokens_per_batch = 256;
  estimate.max_tokens_per_chunk_for_prefill = 256;
  estimate.cache_size_in_bytes = 16 * 1024 * 1024;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args(), estimate);
  ASSERT_GT(capacity.c2_count(), 0);
  ASSERT_EQ(capacity.c1_count(), capacity.c2_count() * 2);
  const KVCacheShape shape(capacity, model_args(), /*world_size=*/1);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, create_options());
  int64_t actual_bytes = 0;
  for (const auto& cache : caches) {
    for (const auto& descriptor : cache.get_cache_tensors()) {
      actual_bytes +=
          descriptor.tensor.numel() * descriptor.tensor.element_size();
    }
  }
  EXPECT_LE(actual_bytes, estimate.cache_size_in_bytes);
  // A substantial unused budget would expose mismatched estimator/storage.
  EXPECT_GT(actual_bytes, estimate.cache_size_in_bytes * 9 / 10);
}

TEST(DeepseekV41OriginalCache, HostTransferRolesUseSameDtype) {
  ScopedAttentionQuant mode("original_qdq");
  KVCacheCapacity capacity;
  capacity.block_size(128).swa_count(5).c2_count(3).c1_count(6);
  const KVCacheShape shape(capacity, model_args(), /*world_size=*/1);
  auto options = create_options();
  options.host_blocks_factor(2.0);
  KVCache host(shape, options, BlockType::C2, /*layer_count=*/1);
  const auto roles = host.get_block_type_tensors(BlockType::C2);
  ASSERT_EQ(roles.count(KVCacheTensorRole::INDEX), 1);
  EXPECT_EQ(roles.at(KVCacheTensorRole::INDEX).scalar_type(), torch::kBFloat16);
  EXPECT_EQ(roles.count(KVCacheTensorRole::INDEX_SCALE), 0);
  EXPECT_FALSE(host.get_indexer_cache_scale().has_value());
}

}  // namespace
}  // namespace xllm
