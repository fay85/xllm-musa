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
#include <torch/torch.h>

#include <algorithm>
#include <vector>

#include "layers/mlu/deepseek_v41/deepseek_v41_attention.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV41CompressorTest, RoundsBeforeNormAndPreservesFp32State) {
  torch::InferenceMode inference;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  for (int32_t ratio : {1, 2}) {
    SCOPED_TRACE(ratio);
    ModelArgs args;
    args.hidden_size(32)
        .n_heads(8)
        .head_dim(512)
        .q_lora_rank(16)
        .qk_rope_head_dim(64)
        .o_groups(1)
        .o_lora_rank(8)
        .window_size(16)
        .rms_norm_eps(1e-20)
        .compress_ratios({ratio})
        .kv_source_layer_ids({0})
        .index_head_dim(128)
        .index_topk(0)
        .num_speculative_tokens(5);
    auto channels = torch::arange(512, cpu);
    auto weight = torch::zeros({512, 32}, cpu);
    weight.select(1, 0).copy_(0.5 + channels.remainder(64) / 64);
    weight.select(1, 1).fill_(1);
    auto gate_weight = torch::zeros_like(weight);
    gate_weight.select(1, 0).copy_(channels.remainder(8) / 8);
    gate_weight.select(1, 1).fill_(1);
    auto index_weight = torch::zeros({128, 512}, cpu);
    for (int64_t channel = 0; channel < 128; ++channel) {
      index_weight[channel][4 * channel].fill_(1);
    }
    DeepseekV41Attention attention(
        ModelContext(parallel, args, QuantArgs(), options), /*layer_id=*/0);
    attention->load_state_dict(StateDict({
        {"wq_a.weight", torch::zeros({16, 32}, options)},
        {"wkv.weight", torch::zeros({512, 32}, options)},
        {"q_norm.weight", torch::ones({16}, options)},
        {"kv_norm.weight", torch::ones({512}, options)},
        {"wq_b.weight", torch::zeros({4096, 16}, options)},
        {"wo_a.weight", torch::zeros({1, 8, 4096}, options)},
        {"wo_b.weight", torch::zeros({32, 8}, options)},
        {"attn_sink", torch::zeros({8}, options)},
        {"compressor.wkv.weight", weight.to(options)},
        {"compressor.wgate.weight", gate_weight.to(options)},
        {"compressor.norm.weight", torch::ones({512}, options)},
        {"indexer.wk.weight", index_weight.to(options)},
        {"indexer.k_norm.weight", torch::ones({128}, options)},
    }));
    attention->verify_loaded_weights();
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros({2, 128, 1, 512}, options);
    tensors.key_cache = torch::full({4, 128, 1, 512}, -7, options);
    tensors.index_cache = torch::full({4, 128, 1, 128}, -7, options);
    if (ratio == 2) {
      tensors.compress_kv_state =
          torch::full({2, 32, 512}, -7, options.dtype(torch::kFloat32));
      tensors.compress_score_state =
          torch::full_like(tensors.compress_kv_state, -7);
    }
    std::vector<KVCache> caches;
    caches.emplace_back(tensors);
    auto expected_main = tensors.key_cache.cpu().view({512, 512});
    auto expected_index = tensors.index_cache.cpu().view({512, 128});
    auto expected_state = torch::full({64, 512}, -7, cpu);
    auto expected_scores = torch::full_like(expected_state, -7);
    auto hidden = torch::zeros({2, 132, 32}, cpu);
    hidden.select(2, 0).copy_(
        (1 + torch::arange(264, cpu).remainder(5) / 8).view({2, 132}));
    hidden.select(2, 1).fill_(1.0 / 512);
    // Only two nonzero products: the CPU double reference is exact here.
    auto raw = torch::matmul(hidden.to(torch::kFloat64),
                             weight.to(torch::kFloat64).t())
                   .to(torch::kFloat32);
    auto scores = torch::matmul(hidden.to(torch::kFloat64),
                                gate_weight.to(torch::kFloat64).t())
                      .to(torch::kFloat32);
    ASSERT_EQ(raw[0][0][32].item<float>(), 1.001953125f);
    auto rope =
        torch::cat({torch::zeros({256, 64}, options.dtype(torch::kFloat32)),
                    torch::ones({256, 64}, options.dtype(torch::kFloat32))},
                   1);
    auto normalize = [](const torch::Tensor& value) {
      auto fp32 = value.to(torch::kFloat32);
      return (fp32 * torch::rsqrt(fp32.square().mean(-1, true) + 1e-20))
          .to(torch::kBFloat16);
    };
    auto rotate = [](torch::Tensor value) {
      auto pairs = value.slice(-1, value.size(-1) - 64).reshape({-1, 32, 2});
      auto rotated =
          torch::stack({-pairs.select(-1, 1), pairs.select(-1, 0)}, -1);
      value.slice(-1, value.size(-1) - 64).copy_(rotated.flatten(1));
      return value;
    };
    int64_t begin = 0;
    bool distinguishes_old_rounding = false;
    for (int32_t count : {1, 2, 3, 123, 1, 2}) {
      SCOPED_TRACE(::testing::Message()
                   << "begin=" << begin << " count=" << count);
      const int64_t end = begin + count;
      ModelInputParams input;
      input.meta.num_sequences = 2;
      input.attention.host.q_seq_lens = {0, count, 2 * count};
      auto swa_table = torch::tensor({1, 0}, torch::kInt32).view({2, 1});
      auto main_table = torch::tensor({3, 1, 2, 0}, torch::kInt32).view({2, 2});
      input.multi_block_tables = {swa_table, main_table, main_table};
      Csa2Context shared;
      shared.topk.resize(2);
      for (auto& topk : shared.topk) {
        topk = torch::empty({count, 0}, options.dtype(torch::kInt64));
      }
      auto positions =
          torch::arange(begin, end, options.dtype(torch::kInt64)).repeat({2});
      input = input.to(device);
      ASSERT_TRUE(input.multi_block_tables[0].is_cpu());
      deepseek_v41::prepare_metadata(
          shared, input, positions, input.meta.num_sequences);
      attention->forward(
          hidden.slice(1, begin, end).reshape({2 * count, 32}).to(options),
          positions,
          rope,
          caches,
          input,
          shared);
      for (int64_t seq = 0; seq < 2; ++seq) {
        auto pooled =
            raw[seq].slice(0, (begin / ratio) * ratio, (end / ratio) * ratio);
        if (ratio == 2) {
          auto gates = scores[seq].slice(0, (begin / 2) * 2, (end / 2) * 2);
          pooled =
              (pooled.view({-1, 2, 512}) * gates.view({-1, 2, 512}).softmax(1))
                  .sum(1);
          const int64_t parked = std::min<int64_t>(count, 32);
          for (int64_t pos = end - parked; pos < end; ++pos) {
            const int64_t slot = (1 - seq) * 32 + pos % 32;
            expected_state[slot].copy_(raw[seq][pos]);
            expected_scores[slot].copy_(scores[seq][pos]);
          }
        }
        auto latent = normalize(pooled.to(torch::kBFloat16));
        distinguishes_old_rounding |= !torch::equal(latent, normalize(pooled));
        auto keys = rotate(normalize(
            torch::matmul(latent.to(torch::kFloat32), index_weight.t())
                .to(torch::kBFloat16)));
        auto main = rotate(latent.clone());
        for (int64_t logical = begin / ratio; logical < end / ratio;
             ++logical) {
          const int64_t page = logical / 128 == 0 ? 3 - seq : 1 - seq;
          const int64_t slot = page * 128 + logical % 128;
          expected_main[slot].copy_(main[logical - begin / ratio]);
          expected_index[slot].copy_(keys[logical - begin / ratio]);
        }
      }
      EXPECT_TRUE(torch::equal(tensors.key_cache.cpu().view_as(expected_main),
                               expected_main));
      EXPECT_TRUE(torch::equal(
          tensors.index_cache.cpu().view_as(expected_index), expected_index));
      if (ratio == 2) {
        EXPECT_EQ(tensors.compress_kv_state.scalar_type(), torch::kFloat32);
        EXPECT_TRUE(torch::equal(
            tensors.compress_kv_state.cpu().view_as(expected_state),
            expected_state));
        EXPECT_TRUE(torch::equal(
            tensors.compress_score_state.cpu().view_as(expected_scores),
            expected_scores));
      }
      begin = end;
    }
    EXPECT_TRUE(distinguishes_old_rounding);
  }
}
}  // namespace
}  // namespace xllm::layer
