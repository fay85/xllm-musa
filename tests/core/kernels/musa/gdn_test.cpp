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

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/kernels/musa/gdn_ops.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::kernel::musa {
namespace {

TEST(MusaGdnTest, PrefillMatchesReferenceForBothHeadCountsAndSequenceLayouts) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::set_num_threads(1);
  torch::manual_seed(23);
  const torch::Device target(torch::kPrivateUse1, /*index=*/0);
  const auto on_device = [target](const torch::Tensor& input) {
    return torch::empty_strided(
               input.sizes(), input.strides(), input.options().device(target))
        .copy_(input);
  };
  constexpr int64_t kKeyHeads = 16;
  constexpr int64_t kHeadDim = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  const std::vector<std::pair<int64_t, std::vector<int32_t>>> layouts{
      {1, {0, 1}},
      {1, {0, 63}},
      {1, {0, 64}},
      {1, {0, 65}},
      {1, {0, 1, 66}},
      {1, {0, 31, 62}},
      {2, {0, 31, 62}}};
  for (const int64_t value_heads : {32, 48}) {
    for (const auto& [input_batch, cu] : layouts) {
      const int64_t tokens = cu.back();
      const int64_t batch_tokens = tokens / input_batch;
      const int64_t sequences = static_cast<int64_t>(cu.size()) - 1;
      SCOPED_TRACE(::testing::Message()
                   << "Hv=" << value_heads << " tokens=" << tokens
                   << " sequences=" << sequences << " batch=" << input_batch);
      // A wider backing tensor exercises the strided Q/K/V launch ABI.
      const torch::Tensor query =
          torch::randn({1, tokens, kKeyHeads + 1, kHeadDim})
              .to(torch::kBFloat16)
              .narrow(2, 0, kKeyHeads);
      const torch::Tensor key =
          torch::randn({1, tokens, kKeyHeads + 1, kHeadDim})
              .to(torch::kBFloat16)
              .narrow(2, 0, kKeyHeads);
      const torch::Tensor value =
          (torch::randn({1, tokens, value_heads + 1, kHeadDim}) * 0.1)
              .to(torch::kBFloat16)
              .narrow(2, 0, value_heads);
      const torch::Tensor beta =
          torch::rand({1, tokens, value_heads}) * 0.5 + 0.1;
      const torch::Tensor decay =
          -torch::rand({1, tokens, value_heads}) * 0.05 - 0.01;
      // The public recurrent state layout is [sequence, head, value, key].
      const torch::Tensor initial =
          torch::randn({sequences, value_heads, kHeadDim, kHeadDim}) * 0.01;
      MateGatedDeltaRulePrefillParams params;
      params.q = on_device(query).view(
          {input_batch, batch_tokens, kKeyHeads, kHeadDim});
      params.k =
          on_device(key).view({input_batch, batch_tokens, kKeyHeads, kHeadDim});
      params.v = on_device(value).view(
          {input_batch, batch_tokens, value_heads, kHeadDim});
      params.beta =
          beta.to(target).view({input_batch, batch_tokens, value_heads});
      params.g =
          decay.to(target).view({input_batch, batch_tokens, value_heads});
      params.initial_state = initial.to(target);
      params.cu_seqlens_host = cu;
      params.cu_seqlens = torch::tensor(cu, torch::kInt32).to(target);
      params.scale = scale;
      auto [output, final_state] = mate_gated_delta_rule_prefill(params);

      const auto normalize = [](const torch::Tensor& input) {
        const torch::Tensor fp32 = input.to(torch::kFloat32);
        return (fp32 / torch::sqrt((fp32 * fp32).sum(-1, true) + 1e-6))
            .to(torch::kBFloat16)
            .to(torch::kFloat32);
      };
      const torch::Tensor q =
          normalize(query).repeat_interleave(value_heads / kKeyHeads, 2);
      const torch::Tensor k =
          normalize(key).repeat_interleave(value_heads / kKeyHeads, 2);
      torch::Tensor expected = torch::empty({1, tokens, value_heads, kHeadDim});
      torch::Tensor expected_state = initial.clone();
      for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        torch::Tensor state = expected_state[sequence];
        for (int64_t token = cu[sequence]; token < cu[sequence + 1]; ++token) {
          state.mul_(decay[0][token].exp().view({value_heads, 1, 1}));
          const torch::Tensor key_row = k[0][token];
          const torch::Tensor residual =
              (value[0][token].to(torch::kFloat32) -
               (state * key_row.unsqueeze(-2)).sum(-1)) *
              beta[0][token].unsqueeze(-1);
          state.add_(residual.unsqueeze(-1) * key_row.unsqueeze(-2));
          expected[0][token] =
              (state * q[0][token].unsqueeze(-2)).sum(-1) * scale;
        }
      }
      const torch::Tensor actual =
          output.cpu()
              .to(torch::kFloat32)
              .reshape({1, tokens, value_heads, kHeadDim});
      const torch::Tensor actual_state = final_state.cpu();
      EXPECT_TRUE(torch::allclose(actual, expected, 0.04, 0.001))
          << "output max_abs=" << (actual - expected).abs().max().item<float>();
      EXPECT_TRUE(torch::allclose(actual_state, expected_state, 0.04, 0.002))
          << "state max_abs="
          << (actual_state - expected_state).abs().max().item<float>()
          << " actual_max=" << actual_state.abs().max().item<float>()
          << " expected_max=" << expected_state.abs().max().item<float>();
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::musa
