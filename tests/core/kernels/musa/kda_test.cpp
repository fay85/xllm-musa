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

#include "core/kernels/musa/kda.h"

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/aten/musa/MUSAGraph.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

namespace xllm::kernel::musa {
namespace {

// Independent token recurrence, using the V-first state convention of vLLM
// FLA KDA. The no-parameter MATE prepare path consumes base-2 log gates.
std::pair<torch::Tensor, torch::Tensor> reference(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& initial,
    const torch::Tensor& cu,
    const std::optional<torch::Tensor>& a_log,
    const std::optional<torch::Tensor>& dt_bias,
    double scale,
    double lower_bound) {
  auto output = torch::empty_like(v, v.options().dtype(torch::kFloat32));
  auto final = initial.to(torch::kFloat32).clone();
  int64_t group = v.size(2) / q.size(2);
  for (int64_t seq = 0; seq < cu.numel() - 1; ++seq) {
    auto state = final[seq];
    for (int64_t t = cu[seq].item<int64_t>(); t < cu[seq + 1].item<int64_t>();
         ++t) {
      auto qt = q[0][t].to(torch::kFloat32).repeat_interleave(group, 0);
      auto kt = k[0][t].to(torch::kFloat32).repeat_interleave(group, 0);
      qt = qt / (qt.square().sum(-1, true) + 1e-6).sqrt();
      kt = kt / (kt.square().sum(-1, true) + 1e-6).sqrt();
      auto gate = g[0][t].to(torch::kFloat32);
      if (a_log.has_value()) {
        gate = (lower_bound *
                ((gate + *dt_bias) * a_log->exp().unsqueeze(-1)).sigmoid())
                   .exp();
      } else {
        gate = gate.exp2();
      }
      state.mul_(gate.unsqueeze(1));
      auto delta =
          (v[0][t].to(torch::kFloat32) - (state * kt.unsqueeze(1)).sum(-1)) *
          beta[0][t].to(torch::kFloat32).sigmoid().unsqueeze(-1);
      state.add_(delta.unsqueeze(-1) * kt.unsqueeze(1));
      output[0][t].copy_((state * qt.unsqueeze(1)).sum(-1) * scale);
    }
  }
  return {output, final};
}

TEST(KdaTest, FullFlashKdaSurfaceMatchesTokenRecurrence) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  torch::manual_seed(42);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  EXPECT_EQ(flash_kda_workspace_size(34, 4, 2), 0);
  for (torch::ScalarType dtype : {torch::kHalf, torch::kBFloat16}) {
    for (bool fp32_state : {false, true}) {
      for (bool varlen : {false, true}) {
        for (bool gate_params : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << dtype << " fp32=" << fp32_state
                       << " varlen=" << varlen << " gate=" << gate_params);
          auto opts = torch::TensorOptions().dtype(dtype);
          auto state_opts = opts.dtype(fp32_state ? torch::kFloat32 : dtype);
          // 34 tokens cross chunk boundaries and cover GVA (2 QK heads, 4 V
          // heads).
          auto packed = torch::randn({1, 34, 8, 128}, opts);
          auto q_cpu = packed.narrow(2, 0, 2);
          auto k_cpu = packed.narrow(2, 2, 2);
          auto v_cpu = packed.narrow(2, 4, 4);
          auto g_cpu = -torch::rand({1, 34, 4, 128}, opts);
          auto beta_cpu = torch::randn({1, 34, 4}, opts);
          auto cu_cpu = varlen ? torch::tensor({0, 0, 1, 18, 34}, torch::kInt64)
                               : torch::tensor({0, 17, 34}, torch::kInt64);
          int64_t nseq = cu_cpu.numel() - 1;
          auto initial_cpu =
              torch::randn({nseq, 4, 128, 128}, state_opts) * 0.01;
          std::optional<torch::Tensor> a_cpu;
          std::optional<torch::Tensor> bias_cpu;
          if (gate_params) {
            a_cpu = torch::randn({4}, torch::kFloat32) * 0.1;
            bias_cpu = torch::randn({4, 128}, torch::kFloat32) * 0.1;
          }
          auto expected = reference(q_cpu,
                                    k_cpu,
                                    v_cpu,
                                    g_cpu,
                                    beta_cpu,
                                    initial_cpu,
                                    cu_cpu,
                                    a_cpu,
                                    bias_cpu,
                                    /*scale=*/0.2,
                                    /*lower_bound=*/-0.7);
          auto packed_device = packed.to(device);
          auto q = packed_device.narrow(2, 0, 2);
          auto k = packed_device.narrow(2, 2, 2);
          auto v = packed_device.narrow(2, 4, 4);
          auto g = g_cpu.to(device);
          auto beta = beta_cpu.to(device);
          auto out_storage =
              torch::full({1, 34, 8, 128}, -99, opts.device(device));
          auto out = out_storage.narrow(2, 0, 4);
          std::optional<torch::Tensor> cu;
          if (varlen) {
            // Exercise both accepted index dtypes and rank-3 packed inputs.
            cu = cu_cpu.to(device, fp32_state ? torch::kInt64 : torch::kInt32);
            q = q.squeeze(0);
            k = k.squeeze(0);
            v = v.squeeze(0);
            g = g.squeeze(0);
            beta = beta.squeeze(0);
            out = out.squeeze(0);
          } else {
            q = q.reshape({2, 17, 2, 128});
            k = k.reshape_as(q);
            v = v.reshape({2, 17, 4, 128});
            g = g.reshape_as(v);
            beta = beta.reshape({2, 17, 4});
            out = out.reshape({2, 17, 4, 128});
          }
          auto initial = initial_cpu.to(device);
          auto final = torch::empty_like(initial);
          std::optional<torch::Tensor> a;
          std::optional<torch::Tensor> bias;
          if (gate_params) {
            a = a_cpu->to(device);
            bias = bias_cpu->to(device);
          }
          flash_kda(q,
                    k,
                    v,
                    g,
                    beta,
                    /*scale=*/0.2,
                    out,
                    a,
                    bias,
                    /*lower_bound=*/-0.7,
                    initial,
                    final,
                    cu);
          stream.synchronize();
          EXPECT_TRUE(torch::allclose(out.to(torch::kCPU)
                                          .reshape_as(expected.first)
                                          .to(torch::kFloat32),
                                      expected.first,
                                      /*rtol=*/0.04,
                                      /*atol=*/0.005));
          EXPECT_TRUE(torch::allclose(final.to(torch::kCPU).to(torch::kFloat32),
                                      expected.second,
                                      /*rtol=*/0.04,
                                      /*atol=*/0.02));
          EXPECT_TRUE(torch::equal(initial.to(torch::kCPU), initial_cpu));
          EXPECT_TRUE(torch::equal(
              out_storage.narrow(2, 4, 4).to(torch::kCPU),
              torch::full_like(out_storage.narrow(2, 4, 4).to(torch::kCPU),
                               -99)));
          // All optional buffers/metadata omitted, default scale, T=1 decode.
          auto q1 = packed_device.narrow(1, 0, 1).narrow(2, 0, 2);
          auto k1 = packed_device.narrow(1, 0, 1).narrow(2, 2, 2);
          auto v1 = packed_device.narrow(1, 0, 1).narrow(2, 4, 4);
          auto g1 = g_cpu.narrow(1, 0, 1).to(device);
          auto beta1 = beta_cpu.narrow(1, 0, 1).to(device);
          auto out1 = torch::empty_like(v1);
          flash_kda(q1,
                    k1,
                    v1,
                    g1,
                    beta1,
                    std::nullopt,
                    out1,
                    std::nullopt,
                    std::nullopt,
                    /*lower_bound=*/-5.0);
          auto zero = torch::zeros({1, 4, 128, 128}, opts);
          auto one_expected = reference(q_cpu.narrow(1, 0, 1),
                                        k_cpu.narrow(1, 0, 1),
                                        v_cpu.narrow(1, 0, 1),
                                        g_cpu.narrow(1, 0, 1),
                                        beta_cpu.narrow(1, 0, 1),
                                        zero,
                                        torch::tensor({0, 1}, torch::kInt64),
                                        std::nullopt,
                                        std::nullopt,
                                        /*scale=*/1.0 / std::sqrt(128.0),
                                        /*lower_bound=*/-5.0);
          stream.synchronize();
          EXPECT_TRUE(torch::allclose(out1.to(torch::kCPU).to(torch::kFloat32),
                                      one_expected.first,
                                      0.04,
                                      0.005));
        }
      }
    }
  }
}

TEST(KdaTest, GraphReplayReadsUpdatedLengthsAndState) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto opts = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto q = torch::randn({1, 33, 3, 128}, opts);
  auto k = torch::randn_like(q);
  auto v = torch::randn({1, 33, 6, 128}, opts);
  auto g = -torch::rand_like(v);
  auto beta = torch::randn({1, 33, 6}, opts);
  auto out = torch::empty_like(v);
  auto expected = torch::empty_like(v);
  auto initial = torch::zeros({2, 6, 128, 128}, opts.dtype(torch::kFloat32));
  auto final = torch::empty_like(initial);
  auto expected_final = torch::empty_like(initial);
  auto cu = torch::tensor({0, 16, 33}, torch::kInt32).to(device);
  auto run = [&](torch::Tensor& output, const torch::Tensor& state_out) {
    flash_kda(q,
              k,
              v,
              g,
              beta,
              std::nullopt,
              output,
              std::nullopt,
              std::nullopt,
              /*lower_bound=*/-5.0,
              initial,
              state_out,
              cu);
  };
  run(out, final);
  stream.synchronize();
  at::musa::MUSAGraph graph;
  graph.capture_begin();
  run(out, final);
  graph.capture_end();
  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    q.copy_(torch::randn_like(q));
    initial.copy_(torch::randn_like(initial) * 0.01);
    cu.copy_(
        torch::tensor({0, 1 + iteration * 15, 33}, torch::kInt32).to(device));
    run(expected, expected_final);
    graph.replay();
    stream.synchronize();
    EXPECT_TRUE(torch::equal(out.to(torch::kCPU), expected.to(torch::kCPU)));
    EXPECT_TRUE(
        torch::equal(final.to(torch::kCPU), expected_final.to(torch::kCPU)));
  }
}

TEST(KdaTest, DefaultStreamOrdersFfiInputsOutputsAndScratch) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto opts = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto q = torch::randn({2, 17, 2, 128}, opts);
  auto k = torch::randn_like(q);
  auto v = torch::randn({2, 17, 4, 128}, opts);
  auto g = -torch::rand_like(v);
  auto beta = torch::randn({2, 17, 4}, opts);
  auto out = torch::empty_like(v);
  auto initial = torch::zeros({2, 4, 128, 128}, opts.dtype(torch::kFloat32));
  auto final = torch::empty_like(initial);
  auto cu = torch::tensor({0, 17, 34}, torch::kInt64);
  auto expected = reference(q.to(torch::kCPU).reshape({1, 34, 2, 128}),
                            k.to(torch::kCPU).reshape({1, 34, 2, 128}),
                            v.to(torch::kCPU).reshape({1, 34, 4, 128}),
                            g.to(torch::kCPU).reshape({1, 34, 4, 128}),
                            beta.to(torch::kCPU).reshape({1, 34, 4}),
                            initial.to(torch::kCPU),
                            cu,
                            std::nullopt,
                            std::nullopt,
                            /*scale=*/0.2,
                            /*lower_bound=*/-5.0);
  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    // The dense path copies the FFI result back using Torch on the default
    // stream. Repeated calls also exercise release and reuse of scratch.
    flash_kda(q,
              k,
              v,
              g,
              beta,
              /*scale=*/0.2,
              out,
              std::nullopt,
              std::nullopt,
              /*lower_bound=*/-5.0,
              initial,
              final);
    auto consumed = out * 2;
    EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    EXPECT_TRUE(torch::allclose(
        consumed.to(torch::kCPU).reshape_as(expected.first).to(torch::kFloat32),
        expected.first * 2,
        /*rtol=*/0.04,
        /*atol=*/0.01));
    EXPECT_TRUE(torch::allclose(final.to(torch::kCPU),
                                expected.second,
                                /*rtol=*/0.04,
                                /*atol=*/0.02));
  }
}
}  // namespace
}  // namespace xllm::kernel::musa
