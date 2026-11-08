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
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/musa/glm5_next/glm5_next_linear_attention.h"

namespace xllm::layer {
namespace {
TEST(GlmKdaLayerTest, PackedChunkDecodeAndSlotResetMatchFullPrefill) {
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard guard;
  torch::manual_seed(53);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto opts =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ProcessGroup group(0, 1, device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.hidden_size(128)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .rms_norm_eps(1e-5);
  Glm5NextKDA layer(args, QuantArgs(), parallel, opts);
  for (auto& parameter : layer->named_parameters()) {
    parameter.value().normal_(0.0, 0.1);
  }
  auto make_cache = [&]() {
    return KVCache(LinearAttentionKVCacheTensors{
        torch::randn({4, 768, 3}, opts),
        torch::randn({4, 2, 128, 128}, opts.dtype(torch::kFloat32))});
  };
  auto full = make_cache();
  auto chunk = make_cache();
  auto run = [&](const torch::Tensor& x,
                 KVCache& cache,
                 const std::vector<int32_t>& cu,
                 bool initial,
                 bool prefill) {
    AttentionMetadata metadata{};
    metadata.is_prefill = prefill;
    metadata.is_chunked_prefill = prefill && initial;
    metadata.max_query_len = 32;
    metadata.q_cu_seq_lens = torch::tensor(cu, opts.dtype(torch::kInt32));
    metadata.has_initial_states =
        torch::full({2}, initial, opts.dtype(torch::kBool));
    ModelInputParams params;
    params.embedding.linear_state_ids = {2, 1};
    return layer->forward(x, metadata, cache, params);
  };
  auto x = torch::randn({40, 128}, opts);
  auto expected = run(x, full, {0, 19, 40}, false, true);
  // Independent full-layer reference: separate projections, scalar-time
  // convolution, V-first delta recurrence, and FP32 sigmoid RMSNorm.
  const auto weights = layer->named_parameters();
  auto linear = [&](const torch::Tensor& input, const std::string& name) {
    return torch::matmul(input.to(torch::kFloat32),
                         weights[name].to(torch::kFloat32).t())
        .to(torch::kBFloat16);
  };
  auto projected = linear(x, "in_proj_qkvbfg_a.weight");
  auto parts = projected.split_with_sizes({768, 2, 128, 128}, -1);
  auto forget = linear(parts[2], "f_b_proj.weight")
                    .view({40, 2, 128})
                    .to(torch::kFloat32);
  auto output_gate = linear(parts[3], "g_b_proj.weight")
                         .view({40, 2, 128})
                         .to(torch::kFloat32);
  auto conv_weight = torch::cat({weights["q_conv1d.weight"],
                                 weights["k_conv1d.weight"],
                                 weights["v_conv1d.weight"]})
                         .to(torch::kFloat32);
  std::vector<torch::Tensor> rows;
  rows.reserve(40);
  for (const auto& bounds : {std::pair<int64_t, int64_t>{0, 19}, {19, 40}}) {
    auto state = torch::zeros({2, 128, 128}, opts.dtype(torch::kFloat32));
    for (int64_t t = bounds.first; t < bounds.second; ++t) {
      auto conv = torch::zeros({768}, opts.dtype(torch::kFloat32));
      for (int64_t tap = 0; tap < 4; ++tap) {
        const int64_t src = t + tap - 3;
        if (src >= bounds.first) {
          conv.add_(parts[0][src].to(torch::kFloat32) *
                    conv_weight.select(1, tap));
        }
      }
      conv = (conv * conv.sigmoid()).to(torch::kBFloat16).to(torch::kFloat32);
      auto qkv = conv.view({3, 2, 128});
      auto q = qkv[0] / (qkv[0].square().sum(-1, true) + 1e-6).sqrt();
      auto k = qkv[1] / (qkv[1].square().sum(-1, true) + 1e-6).sqrt();
      auto decay = (-5.0 * ((forget[t] + weights["dt_bias"].view({2, 128})) *
                            weights["A_log"].exp().unsqueeze(-1))
                               .sigmoid())
                       .exp();
      state.mul_(decay.unsqueeze(-2));
      auto delta = (qkv[2] - (state * k.unsqueeze(-2)).sum(-1)) *
                   parts[1][t].to(torch::kFloat32).sigmoid().unsqueeze(-1);
      state.add_(delta.unsqueeze(-1) * k.unsqueeze(-2));
      auto output = ((state * q.unsqueeze(-2)).sum(-1) / std::sqrt(128.0))
                        .to(torch::kBFloat16)
                        .to(torch::kFloat32);
      output = output * (output.square().mean(-1, true) + 1e-5).rsqrt() *
               weights["o_norm.weight"].to(torch::kFloat32) *
               output_gate[t].sigmoid();
      rows.emplace_back(output.to(torch::kBFloat16).flatten());
    }
  }
  auto reference = linear(torch::stack(rows), "o_proj.weight");
  EXPECT_TRUE(torch::allclose(expected.to(torch::kFloat32),
                              reference.to(torch::kFloat32),
                              0.04,
                              0.005));

  auto first = run(torch::cat({x.slice(0, 0, 2), x.slice(0, 19, 22)}),
                   chunk,
                   {0, 2, 5},
                   false,
                   true);
  auto second = run(torch::cat({x.slice(0, 2, 18), x.slice(0, 22, 39)}),
                    chunk,
                    {0, 16, 33},
                    true,
                    true);
  auto last = run(torch::cat({x.slice(0, 18, 19), x.slice(0, 39, 40)}),
                  chunk,
                  {0, 1, 2},
                  true,
                  false);
  auto actual = torch::cat({first.slice(0, 0, 2),
                            second.slice(0, 0, 16),
                            last.slice(0, 0, 1),
                            first.slice(0, 2, 5),
                            second.slice(0, 16, 33),
                            last.slice(0, 1, 2)});
  EXPECT_TRUE(torch::allclose(
      actual.to(torch::kFloat32), expected.to(torch::kFloat32), 0.03, 0.003));
  auto ids = torch::tensor({1, 2}, opts.dtype(torch::kLong));
  EXPECT_TRUE(torch::allclose(chunk.get_ssm_cache().index_select(0, ids),
                              full.get_ssm_cache().index_select(0, ids),
                              0.03,
                              0.003));
  EXPECT_TRUE(torch::equal(chunk.get_conv_cache().index_select(0, ids),
                           full.get_conv_cache().index_select(0, ids)));
  auto reset = run(x, chunk, {0, 19, 40}, false, true);
  EXPECT_TRUE(torch::allclose(
      reset.to(torch::kFloat32), expected.to(torch::kFloat32), 0.0, 0.0));
}
}  // namespace
}  // namespace xllm::layer
