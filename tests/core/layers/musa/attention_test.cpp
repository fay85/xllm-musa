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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/layers/musa/flashinfer_attention.h"

namespace xllm::layer {
namespace {

TEST(MusaAttentionTest, DispatchHonorsCompiledContractAndOverrides) {
  struct Settings {
    const char* attention;
    const char* decode;
    bool prefill_enabled;
    bool decode_enabled;
  };
  const std::array<Settings, 5> settings{{
      {nullptr, nullptr, true, true},
      {"1", nullptr, true, true},
      {"0", nullptr, false, false},
      {"1", "0", true, false},
      {"0", "1", false, true},
  }};
  for (const Settings& setting : settings) {
    // The dispatch caches environment settings; isolate each configuration.
    EXPECT_EXIT(
        ([&] {
          const auto set_or_unset = [](const char* name, const char* value) {
            if (value == nullptr) {
              CHECK_EQ(unsetenv(name), 0);
            } else {
              CHECK_EQ(setenv(name, value, /*overwrite=*/1), 0);
            }
          };
          set_or_unset("XLLM_USE_FA3", setting.attention);
          set_or_unset("XLLM_USE_FA3_DECODE", setting.decode);
          for (const bool decode : {false, true}) {
            const bool enabled =
                decode ? setting.decode_enabled : setting.prefill_enabled;
            for (const int64_t ratio : {6, 8}) {
              CHECK_EQ(use_fa3_attention(256,
                                         ratio * 2,
                                         2,
                                         torch::kBFloat16,
                                         decode,
                                         /*window_left=*/-1),
                       enabled);
              if (enabled) {
                check_fa3_attention(256,
                                    ratio * 2,
                                    2,
                                    torch::kBFloat16,
                                    decode,
                                    /*window_left=*/-1);
              }
              for (const int64_t window_left : {0, 1, 1023}) {
                CHECK(!use_fa3_attention(
                    256, ratio * 2, 2, torch::kBFloat16, decode, window_left));
              }
            }
            CHECK(!use_fa3_attention(
                256, 12, 2, torch::kFloat16, decode, /*window_left=*/-1));
            CHECK(!use_fa3_attention(
                128, 12, 2, torch::kBFloat16, decode, /*window_left=*/-1));
            CHECK(!use_fa3_attention(
                256, 8, 2, torch::kBFloat16, decode, /*window_left=*/-1));
            CHECK(!use_fa3_attention(
                256, 12, 0, torch::kBFloat16, decode, /*window_left=*/-1));
          }
          std::_Exit(0);
        }()),
        ::testing::ExitedWithCode(0),
        "");
  }
  // These fail before allocating workspace or loading any operator URI.
  EXPECT_DEATH(([] {
                 CHECK_EQ(setenv("XLLM_USE_FA3_DECODE", "0", /*overwrite=*/1),
                          0);
                 FlashInferAttentionImpl(12, 256, 0.1f, 2, -1);
               }()),
               "Unsupported MUSA attention configuration");
  EXPECT_DEATH(FlashInferAttentionImpl(12, 128, 0.1f, 2, -1),
               "Unsupported MUSA attention configuration");
  EXPECT_DEATH(FlashInferAttentionImpl(12, 256, 0.1f, 2, 1),
               "Unsupported MUSA attention configuration");
  EXPECT_DEATH(check_fa3_attention(256,
                                   12,
                                   2,
                                   torch::kFloat16,
                                   /*is_decode=*/true,
                                   /*window_left=*/-1),
               "Unsupported MUSA attention configuration");
  EXPECT_DEATH(
      ([] {
        AttentionMetadata metadata;
        metadata.is_prefill = true;
        metadata.max_query_len = 3;
        metadata.max_seq_len = 5;
        metadata.q_seq_lens_vec = {2, 3};
        metadata.kv_seq_lens_vec = {5, 2};
        metadata.q_cu_seq_lens = torch::tensor({0, 2, 5}, torch::kInt32);
        torch::Tensor query = torch::zeros({5, 12, 256}, torch::kBFloat16);
        torch::Tensor key = torch::zeros({5, 2, 256}, torch::kBFloat16);
        torch::Tensor value = key.clone();
        torch::Tensor output = torch::empty_like(query);
        KVCache cache;
        FlashInferAttentionImpl attention(12, 256, 0.0625f, 2, -1);
        attention.forward(metadata, query, key, value, output, cache);
      }()),
      "KV length must include every current query token");
}

}  // namespace
}  // namespace xllm::layer
