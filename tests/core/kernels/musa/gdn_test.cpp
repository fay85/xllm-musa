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

#include <c10/core/StreamGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/kernels/musa/gdn_ops.h"
#include "core/kernels/param.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

TEST(MusaGdnTest, DecodeProjectsTheUpdatedFp32StateNearRoundingBoundaries) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  constexpr int64_t kBatch = 64;
  constexpr int64_t kHeadDim = 4;
  const torch::Device target = device.unwrap();
  // Power-of-two norms make the normalized query exactly one in each lane.
  const torch::Tensor query = torch::full({kBatch, kHeadDim}, 1024.0f);
  const torch::Tensor key =
      torch::tensor({1024.0f, 0.0f, 0.0f, 0.0f}).repeat({kBatch, 1});
  const torch::Tensor value = torch::full({kBatch, kHeadDim}, 0.25f);
  torch::Tensor initial = torch::zeros({kBatch, 1, kHeadDim, kHeadDim});
  const torch::Tensor first =
      (torch::arange(kBatch).to(torch::kFloat32) * 0.12345 + 2048.123)
          .view({kBatch, 1, 1})
          .expand({kBatch, 1, kHeadDim});
  initial.select(-1, 0).copy_(first);
  // Cancellation exposes reordering near BF16 output rounding boundaries.
  initial.select(-1, 1).copy_(-first * 0.5 + 0.0013);
  MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv =
      torch::cat({query, key, value}, 1).to(torch::kBFloat16).to(target);
  params.state = initial.to(target);
  params.A_log = torch::zeros({1}).to(target);
  params.dt_bias = torch::zeros({1}).to(target);
  params.a = torch::full({kBatch, 1}, 0.5f, torch::kBFloat16).to(target);
  params.b = torch::zeros_like(params.a);
  params.state_indices = torch::arange(kBatch, torch::kInt32).to(target);
  params.num_k_heads = 1;
  params.num_v_heads = 1;
  params.head_k_dim = kHeadDim;
  params.head_v_dim = kHeadDim;
  params.scale = 2.0;
  const torch::Tensor output = fused_gated_delta_rule_decode(params).cpu();
  const torch::Tensor projected =
      params.state.cpu().sum(-1).to(torch::kBFloat16);
  EXPECT_TRUE(torch::equal(output, projected))
      << "Output must reflect the state stored by the same recurrence";
}

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

TEST(MusaGdnTest, ConvolutionUpdatesSharedHistoryMajorStorageThroughView) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::set_num_threads(1);
  torch::manual_seed(29);
  const torch::Device target(torch::kPrivateUse1, /*index=*/0);
  constexpr int64_t kDim = 3;
  constexpr int64_t kHistory = 3;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  // Square axes and asymmetric data prevent shape-only transpose checks.
  const torch::Tensor initial =
      torch::arange(3 * kHistory * kDim, options).reshape({3, kHistory, kDim}) /
      10;
  const torch::Tensor weight = torch::randn({kDim, kHistory + 1}, options);
  const std::vector<int32_t> cu = {0, 1, 1026};
  const std::vector<int32_t> indices = {2, 0};
  const torch::Tensor has_initial =
      torch::tensor({true, false}, torch::kBool).to(target);
  const torch::Tensor device_cu = torch::tensor(cu, torch::kInt32).to(target);
  const torch::Tensor device_indices =
      torch::tensor(indices, torch::kInt32).to(target);
  const torch::Tensor device_weight = weight.to(target);
  const torch::Tensor input = torch::randn({cu.back(), kDim}, options);
  torch::Tensor expected_state = initial.clone();
  torch::Tensor expected_output = torch::empty_like(input);
  for (int64_t sequence = 0; sequence < 2; ++sequence) {
    torch::Tensor history = expected_state[indices[sequence]];
    if (sequence == 1) {
      history.zero_();
    }
    for (int64_t token = cu[sequence]; token < cu[sequence + 1]; ++token) {
      const torch::Tensor window =
          torch::cat({history, input[token].unsqueeze(0)}, /*dim=*/0);
      expected_output[token] = (window.transpose(0, 1) * weight).sum(-1);
      history.copy_(window.narrow(/*dim=*/0, /*start=*/1, kHistory));
    }
  }

  // Exercise both prefill kernels directly, independent of the runtime toggle.
  for (const bool token_major : {false, true}) {
    SCOPED_TRACE(token_major);
    torch::Tensor storage = initial.to(target);
    torch::Tensor view = storage.transpose(-1, -2);
    ASSERT_EQ(storage.data_ptr(), view.data_ptr());
    ASSERT_FALSE(view.is_contiguous());
    torch::Tensor x =
        token_major ? input.to(target) : input.t().contiguous().to(target);
    torch::Tensor output = torch::empty_like(x);
    if (token_major) {
      causal_conv1d_fwd_token_major(x,
                                    device_weight,
                                    output,
                                    std::nullopt,
                                    view,
                                    device_cu,
                                    device_indices,
                                    has_initial,
                                    /*silu_activation=*/false,
                                    /*pad_slot_id=*/-1);
    } else {
      causal_conv1d_fwd(x,
                        device_weight,
                        output,
                        std::nullopt,
                        view,
                        device_cu,
                        device_indices,
                        has_initial,
                        /*silu_activation=*/false,
                        /*pad_slot_id=*/-1);
    }
    const torch::Tensor actual = token_major ? output.cpu() : output.cpu().t();
    EXPECT_TRUE(torch::allclose(actual, expected_output, 1e-5, 1e-5));
    EXPECT_TRUE(torch::equal(storage.cpu(), expected_state));

    // Decode consumes the same view and must update the backing storage.
    const torch::Tensor decode_input = torch::randn({2, kDim}, options);
    torch::Tensor expected_decode = torch::empty_like(decode_input);
    torch::Tensor decode_state = expected_state.clone();
    for (int64_t sequence = 0; sequence < 2; ++sequence) {
      torch::Tensor history = decode_state[indices[sequence]];
      const torch::Tensor window =
          torch::cat({history, decode_input[sequence].unsqueeze(0)}, /*dim=*/0);
      expected_decode[sequence] = (window.transpose(0, 1) * weight).sum(-1);
      history.copy_(window.narrow(/*dim=*/0, /*start=*/1, kHistory));
    }
    CausalConv1dUpdateParams params;
    params.x = decode_input.to(target);
    params.weight = device_weight;
    params.conv_state = view;
    params.conv_state_indices = device_indices;
    params.query_start_loc = torch::tensor({0, 1, 2}, torch::kInt32).to(target);
    params.activation = false;
    const torch::Tensor decode_output = torch::empty_like(params.x);
    const torch::Tensor actual_decode =
        causal_conv1d_update(params, decode_output).cpu();
    EXPECT_TRUE(torch::allclose(actual_decode, expected_decode, 1e-5, 1e-5));
    EXPECT_TRUE(torch::equal(storage.cpu(), decode_state));
  }
}

TEST(MusaGdnTest, SpecVerifyMatchesCheckpointRecurrenceAndConvHistory) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::NoGradGuard no_grad;
  torch::manual_seed(53);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto transfer = [&](const torch::Tensor& tensor) {
    return tensor.to(device.unwrap());
  };
  constexpr int64_t kBatch = 2;
  constexpr int64_t kQueryHeads = 8;
  constexpr int64_t kValueHeads = 24;
  constexpr int64_t kDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  // Cover VK/KV layouts and a non-specialized runtime verification length.
  for (int64_t seq_len : {3, 4}) {
    SCOPED_TRACE(seq_len);
    const torch::Tensor conv_input = torch::randn({kBatch, 8, seq_len}, bf16);
    const torch::Tensor weight = torch::randn({8, 4}, bf16);
    const torch::Tensor state = torch::randn({3, 8, seq_len + 2}, bf16);
    const torch::Tensor indices = torch::tensor({2, 0}, torch::kInt64);
    const torch::Tensor accepted =
        torch::tensor({int64_t{1}, seq_len}, torch::kInt64);
    const torch::Tensor initial = state.to(torch::kFloat32);
    torch::Tensor expected_state = state.clone();
    torch::Tensor expected_conv = torch::empty_like(conv_input);
    for (int64_t row = 0; row < kBatch; ++row) {
      const int64_t slot = indices[row].item<int64_t>();
      const int64_t start = accepted[row].item<int64_t>() - 1;
      const torch::Tensor history = initial[slot].narrow(1, start, 3);
      const torch::Tensor joined =
          torch::cat({history, conv_input[row].to(torch::kFloat32)}, 1);
      expected_conv[row].copy_(torch::silu(
          (joined.unfold(1, 4, 1) * weight.to(torch::kFloat32).unsqueeze(1))
              .sum(-1)));
      expected_state[slot].copy_(joined.narrow(1, 1, seq_len + 2));
    }
    // Verify the shared history-major storage is updated through its view.
    torch::Tensor device_storage = transfer(state.transpose(1, 2).contiguous());
    torch::Tensor device_state =
        seq_len == 3 ? transfer(state) : device_storage.transpose(1, 2);
    if (seq_len == 4) {
      ASSERT_FALSE(device_state.is_contiguous());
      ASSERT_EQ(device_state.data_ptr(), device_storage.data_ptr());
    }
    const torch::Tensor conv_output =
        causal_conv1d_spec_verify(transfer(conv_input),
                                  transfer(weight),
                                  device_state,
                                  transfer(indices),
                                  transfer(accepted));
    EXPECT_TRUE(torch::allclose(conv_output.cpu().to(torch::kFloat32),
                                expected_conv.to(torch::kFloat32),
                                1e-2,
                                1e-3));
    EXPECT_TRUE(torch::equal(device_state.cpu(), expected_state));
    if (seq_len == 4) {
      EXPECT_TRUE(
          torch::equal(device_storage.cpu(), expected_state.transpose(1, 2)));
    }

    const torch::Tensor q =
        torch::randn({kBatch, kQueryHeads, kDim, seq_len}, bf16)
            .permute({0, 3, 1, 2});
    const torch::Tensor k = torch::randn_like(q);
    const torch::Tensor v =
        (torch::randn({kBatch, kValueHeads, kDim, seq_len}, bf16) * 0.1)
            .permute({0, 3, 1, 2});
    const torch::Tensor a = torch::randn({kBatch, seq_len, kValueHeads}, bf16);
    const torch::Tensor b = torch::randn_like(a);
    const torch::Tensor a_log =
        torch::randn({kValueHeads}, torch::kFloat32) * 0.1;
    const torch::Tensor dt_bias = torch::randn_like(a_log) * 0.1;
    const torch::Tensor initial_ssm =
        torch::randn({kBatch, kValueHeads, kDim, kDim}) * 0.01;
    torch::Tensor recurrent = initial_ssm.transpose(-1, -2).contiguous();
    const torch::Tensor device_q = transfer(q);
    const torch::Tensor device_k = transfer(k);
    const torch::Tensor device_v = transfer(v);
    const bool state_layout_vk = seq_len == 3;
    const torch::Tensor checkpoint_indices = transfer(
        indices.unsqueeze(1) * seq_len + torch::arange(seq_len, torch::kInt64));
    const torch::Tensor accepted_device = transfer(accepted);
    torch::Tensor device_state_pool =
        transfer(torch::randn({3 * seq_len, kValueHeads, kDim, kDim}) * 0.01);
    const torch::Tensor initial_indices =
        checkpoint_indices.gather(1, (accepted_device - 1).unsqueeze(1))
            .squeeze(1);
    device_state_pool.index_copy_(
        0,
        initial_indices,
        transfer(state_layout_vk ? initial_ssm
                                 : initial_ssm.transpose(-1, -2).contiguous()));
    const torch::Tensor untouched =
        device_state_pool.narrow(0, seq_len, seq_len).clone();
    const torch::Tensor device_a_log = transfer(a_log);
    const torch::Tensor device_a = transfer(a);
    const torch::Tensor device_bias = transfer(dt_bias);
    const torch::Tensor device_b = transfer(b);
    const auto forward = [&]() {
      return gated_delta_rule_spec_verify(device_q,
                                          device_k,
                                          device_v,
                                          device_state_pool,
                                          checkpoint_indices,
                                          accepted_device,
                                          device_a_log,
                                          device_a,
                                          device_bias,
                                          device_b,
                                          scale,
                                          state_layout_vk);
    };
    const auto result = forward();
    EXPECT_TRUE(result.is_contiguous());
    const torch::Tensor verify_output = result.cpu();
    torch::Tensor verify_checkpoints =
        device_state_pool.index_select(0, checkpoint_indices.flatten())
            .view({kBatch, seq_len, kValueHeads, kDim, kDim})
            .cpu();
    if (!state_layout_vk) {
      verify_checkpoints = verify_checkpoints.transpose(-1, -2);
    }
    EXPECT_TRUE(
        torch::equal(device_state_pool.narrow(0, seq_len, seq_len), untouched));
    MateGatedDeltaRuleDecodeParams decode;
    decode.state = transfer(initial_ssm);
    decode.A_log = device_a_log;
    decode.dt_bias = device_bias;
    decode.state_indices = transfer(torch::arange(kBatch, torch::kInt32));
    decode.num_k_heads = kQueryHeads;
    decode.num_v_heads = kValueHeads;
    decode.head_k_dim = kDim;
    decode.head_v_dim = kDim;
    decode.scale = scale;
    for (int64_t token = 0; token < seq_len; ++token) {
      const auto normalize = [&](const torch::Tensor& tensor) {
        const torch::Tensor x = tensor.select(1, token).to(torch::kFloat32);
        return (x * torch::rsqrt(x.square().sum(-1, true) + 1e-6))
            .repeat_interleave(3, 1);
      };
      const torch::Tensor qt = normalize(q) * scale;
      const torch::Tensor kt = normalize(k);
      const torch::Tensor gate =
          -a_log.exp() * torch::nn::functional::softplus(
                             a.select(1, token).to(torch::kFloat32) + dt_bias);
      const torch::Tensor beta =
          torch::sigmoid(b.select(1, token).to(torch::kFloat32));
      recurrent = recurrent * gate.exp().unsqueeze(-1).unsqueeze(-1);
      const torch::Tensor memory = (recurrent * kt.unsqueeze(-1)).sum(-2);
      const torch::Tensor delta =
          (v.select(1, token).to(torch::kFloat32) - memory) *
          beta.unsqueeze(-1);
      recurrent = recurrent + kt.unsqueeze(-1) * delta.unsqueeze(-2);
      const torch::Tensor expected_output =
          (recurrent * qt.unsqueeze(-1)).sum(-2).to(torch::kBFloat16);
      decode.mixed_qkv =
          transfer(torch::cat({q.select(1, token).reshape({kBatch, -1}),
                               k.select(1, token).reshape({kBatch, -1}),
                               v.select(1, token).reshape({kBatch, -1})},
                              1));
      decode.a = device_a.select(1, token).contiguous();
      decode.b = device_b.select(1, token).contiguous();
      EXPECT_TRUE(torch::equal(verify_output.select(1, token),
                               fused_gated_delta_rule_decode(decode).cpu()));
      EXPECT_TRUE(torch::equal(verify_checkpoints.select(1, token),
                               decode.state.cpu()));
      EXPECT_TRUE(
          torch::allclose(verify_output.select(1, token).to(torch::kFloat32),
                          expected_output.to(torch::kFloat32),
                          1e-2,
                          1e-3));
      EXPECT_TRUE(torch::allclose(verify_checkpoints.select(1, token),
                                  recurrent.transpose(-1, -2),
                                  1e-3,
                                  1e-4));
    }

    if (seq_len == 3) {
      device.synchronize_default_stream();
      at::musa::MUSAGraph graph;
      const c10::musa::MUSAStream stream = c10::musa::getStreamFromPool();
      torch::Tensor captured;
      {
        c10::StreamGuard guard(stream.unwrap());
        graph.capture_begin({0, 0}, musaStreamCaptureModeThreadLocal);
        captured = forward();
        graph.capture_end();
      }
      stream.synchronize();
      device_state_pool.mul_(0.5);
      device_q.mul_(0.5);
      const torch::Tensor before_replay = device_state_pool.clone();
      const auto expected = forward();
      const torch::Tensor expected_state = device_state_pool.clone();
      device_state_pool.copy_(before_replay);
      device.synchronize_default_stream();
      graph.replay();
      EXPECT_TRUE(torch::equal(captured.cpu(), expected.cpu()));
      EXPECT_TRUE(torch::equal(device_state_pool.cpu(), expected_state.cpu()));
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::musa
