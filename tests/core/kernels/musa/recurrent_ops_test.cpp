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

#include <Python.h>
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/kernels/musa/causal_conv1d_state.h"
#include "core/kernels/musa/kda_ops.h"
#include "core/kernels/musa/rms_norm_sigmoid_gated.h"

namespace xllm::kernel::musa {
namespace {

// The oracle below uses scalar double arithmetic only. It neither initializes
// Python nor calls a GPU kernel, CPU BLAS, MATE or an attention implementation.
torch::Tensor pattern(torch::IntArrayRef shape,
                      double phase,
                      double amplitude = 1.0,
                      torch::ScalarType dtype = torch::kBFloat16) {
  torch::Tensor result = torch::empty(shape, torch::dtype(torch::kDouble));
  double* values = result.data_ptr<double>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    values[i] =
        amplitude * (std::sin(static_cast<double>(i) * 0.037 + phase) +
                     0.3 * std::cos(static_cast<double>(i) * 0.113 + phase));
  }
  return result.to(dtype);
}

KdaInputs make_inputs(int64_t batch, int64_t tokens, int64_t heads) {
  const std::vector<int64_t> shape{batch, tokens, heads, 128};
  return {pattern(shape, 0.7),
          pattern(shape, 1.8),
          pattern(shape, 3.1),
          (pattern(shape, 4.4).to(torch::kFloat32) * 0.7 - 1.0)
              .to(torch::kBFloat16),
          pattern({batch, tokens, heads}, 5.3),
          torch::linspace(-2.0, 0.5, heads, torch::dtype(torch::kFloat32)),
          pattern({heads, 128}, 6.2, 0.3, torch::kFloat32),
          pattern({batch, heads, 128, 128}, 7.1, 0.03, torch::kFloat32),
          {/*gate_lower_bound=*/-5.0f, /*scale=*/0.08838834764831845f}};
}

KdaInputs to_device(const KdaInputs& cpu, const torch::Device& device) {
  return {cpu.query.to(device),
          cpu.key.to(device),
          cpu.value.to(device),
          cpu.raw_forget.to(device),
          cpu.raw_beta.to(device),
          cpu.a_log.to(device),
          cpu.dt_bias.to(device),
          cpu.initial_state.to(device),
          cpu.options};
}

KdaInputs token_slice(const KdaInputs& inputs,
                      int64_t begin,
                      int64_t length,
                      const torch::Tensor& state) {
  return {inputs.query.narrow(1, begin, length).contiguous(),
          inputs.key.narrow(1, begin, length).contiguous(),
          inputs.value.narrow(1, begin, length).contiguous(),
          inputs.raw_forget.narrow(1, begin, length).contiguous(),
          inputs.raw_beta.narrow(1, begin, length).contiguous(),
          inputs.a_log,
          inputs.dt_bias,
          state,
          inputs.options};
}

std::pair<torch::Tensor, torch::Tensor> reference(const KdaInputs& cpu) {
  const int64_t batch = cpu.query.size(0);
  const int64_t tokens = cpu.query.size(1);
  const int64_t heads = cpu.query.size(2);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(8);
  for (const torch::Tensor* tensor : {&cpu.query,
                                      &cpu.key,
                                      &cpu.value,
                                      &cpu.raw_forget,
                                      &cpu.raw_beta,
                                      &cpu.a_log,
                                      &cpu.dt_bias,
                                      &cpu.initial_state}) {
    tensors.push_back(tensor->to(torch::kCPU).to(torch::kDouble).contiguous());
  }
  const double* q = tensors[0].data_ptr<double>();
  const double* k = tensors[1].data_ptr<double>();
  const double* v = tensors[2].data_ptr<double>();
  const double* f = tensors[3].data_ptr<double>();
  const double* beta_raw = tensors[4].data_ptr<double>();
  const double* a_log = tensors[5].data_ptr<double>();
  const double* bias = tensors[6].data_ptr<double>();
  torch::Tensor final_state = tensors[7].clone();
  double* state = final_state.data_ptr<double>();
  torch::Tensor output =
      torch::empty(cpu.query.sizes(), torch::dtype(torch::kDouble));
  double* out = output.data_ptr<double>();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < tokens; ++t) {
      for (int64_t h = 0; h < heads; ++h) {
        const int64_t offset = ((b * tokens + t) * heads + h) * 128;
        double norm_q = 1e-6;
        double norm_k = 1e-6;
        for (int64_t d = 0; d < 128; ++d) {
          norm_q += q[offset + d] * q[offset + d];
          norm_k += k[offset + d] * k[offset + d];
        }
        double key[128];
        double query[128];
        double decay[128];
        for (int64_t d = 0; d < 128; ++d) {
          key[d] = k[offset + d] / std::sqrt(norm_k);
          query[d] = q[offset + d] / std::sqrt(norm_q) * cpu.options.scale;
          const double gate =
              std::exp(a_log[h]) * (f[offset + d] + bias[h * 128 + d]);
          decay[d] =
              std::exp(cpu.options.gate_lower_bound / (1.0 + std::exp(-gate)));
        }
        const double beta =
            1.0 / (1.0 + std::exp(-beta_raw[(b * tokens + t) * heads + h]));
        for (int64_t row = 0; row < 128; ++row) {
          const int64_t start = ((b * heads + h) * 128 + row) * 128;
          double predicted = 0.0;
          for (int64_t d = 0; d < 128; ++d) {
            state[start + d] *= decay[d];
            predicted += state[start + d] * key[d];
          }
          const double delta = (v[offset + row] - predicted) * beta;
          double result = 0.0;
          for (int64_t d = 0; d < 128; ++d) {
            state[start + d] += delta * key[d];
            result += state[start + d] * query[d];
          }
          out[offset + row] = result;
        }
      }
    }
  }
  return {output, final_state};
}

void expect_close(const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  double atol,
                  double rtol,
                  double relative_l2_limit) {
  ASSERT_TRUE(actual.sizes() == expected.sizes());
  torch::Tensor a = actual.to(torch::kCPU).to(torch::kDouble).contiguous();
  torch::Tensor e = expected.to(torch::kCPU).to(torch::kDouble).contiguous();
  const double* av = a.data_ptr<double>();
  const double* ev = e.data_ptr<double>();
  double sum_error = 0.0;
  double sum_reference = 0.0;
  double max_error = 0.0;
  int64_t mismatched = 0;
  for (int64_t i = 0; i < a.numel(); ++i) {
    const double error = std::abs(av[i] - ev[i]);
    sum_error += error * error;
    sum_reference += ev[i] * ev[i];
    max_error = std::max(max_error, error);
    mismatched += static_cast<int64_t>(!std::isfinite(av[i]) ||
                                       error > atol + rtol * std::abs(ev[i]));
  }
  const double relative_l2 =
      std::sqrt(sum_error / std::max(sum_reference, 1e-30));
  EXPECT_EQ(mismatched, 0) << "max_abs=" << max_error;
  EXPECT_LE(relative_l2, relative_l2_limit);
  LOG(INFO) << "max_abs=" << max_error << " relative_l2=" << relative_l2;
}

class MusaRecurrentOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int32_t count = 0;
    ASSERT_EQ(musaGetDeviceCount(&count), musaSuccess);
    ASSERT_GT(count, 0) << "MUSA device required; no skipped acceptance";
    ASSERT_FALSE(Py_IsInitialized());
    at::detail::getMUSAHooks().initMUSA();
    no_grad_.emplace();
    stream_.emplace(c10::musa::getStreamFromPool(false, 0));
    stream_guard_.emplace(stream_->unwrap());
  }
  void TearDown() override {
    if (stream_.has_value()) {
      stream_->synchronize();
    }
    EXPECT_FALSE(Py_IsInitialized());
  }
  const torch::Device device_{torch::kPrivateUse1, 0};
  std::optional<torch::NoGradGuard> no_grad_;
  std::optional<c10::musa::MUSAStream> stream_;
  std::optional<c10::StreamGuard> stream_guard_;
};

TEST_F(MusaRecurrentOpsTest, PrefillAndDecodeMatchIndependentScalarReference) {
  std::vector<std::pair<int64_t, int64_t>> shapes;
  shapes.reserve(12);
  for (const int64_t t : {1, 3, 4, 5, 15, 16, 17, 63, 64, 65}) {
    shapes.emplace_back(1, t);
  }
  shapes.emplace_back(2, 129);
  shapes.emplace_back(2, 17);
  for (const auto& [batch, tokens] : shapes) {
    const int64_t heads = (batch == 2 && tokens == 17) ? 8 : 64;
    SCOPED_TRACE(::testing::Message()
                 << "B=" << batch << " T=" << tokens << " H=" << heads);
    KdaInputs cpu = make_inputs(batch, tokens, heads);
    if (tokens == 1) {
      cpu.initial_state.zero_();
    }
    const auto golden = reference(cpu);
    KdaInputs inputs = to_device(cpu, device_);
    KdaWorkspace workspace(batch, tokens, heads, device_);
    torch::Tensor output = torch::empty_like(inputs.query);
    torch::Tensor final_state = torch::empty_like(inputs.initial_state);
    kda_prefill(inputs, output, final_state, workspace);
    LOG(INFO) << "prefill B=" << batch << " T=" << tokens << " H=" << heads;
    expect_close(output, golden.first, 0.002, 0.03, 0.025);
    expect_close(final_state, golden.second, 0.02, 0.05, 0.03);
    EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
    torch::Tensor state = inputs.initial_state;
    std::vector<torch::Tensor> parts;
    parts.reserve(tokens);
    for (int64_t t = 0; t < tokens; ++t) {
      KdaInputs step = token_slice(inputs, t, 1, state);
      torch::Tensor part = torch::empty_like(step.query);
      torch::Tensor updated = torch::empty_like(state);
      kda_decode(step, part, updated);
      parts.push_back(std::move(part));
      state = std::move(updated);
    }
    LOG(INFO) << "decode B=" << batch << " T=" << tokens << " H=" << heads;
    expect_close(torch::cat(parts, 1), golden.first, 0.002, 0.03, 0.025);
    expect_close(state, golden.second, 2e-6, 5e-5, 5e-6);
    EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
  }
}

TEST_F(MusaRecurrentOpsTest, BoundedGateAndScaleAreCallerControlled) {
  const std::vector<KdaOptions> configurations{
      {/*gate_lower_bound=*/-0.25f, /*scale=*/0.03f},
      {/*gate_lower_bound=*/-1.0f, /*scale=*/0.125f},
      {/*gate_lower_bound=*/-3.0f, /*scale=*/0.2f},
      {/*gate_lower_bound=*/-8.0f, /*scale=*/0.07f}};
  KdaWorkspace workspace(
      /*max_batch=*/2, /*max_tokens=*/17, /*heads=*/4, device_);
  for (const KdaOptions& options : configurations) {
    SCOPED_TRACE(::testing::Message()
                 << "gate_lower_bound=" << options.gate_lower_bound
                 << " scale=" << options.scale);
    KdaInputs cpu = make_inputs(/*batch=*/2, /*tokens=*/17, /*heads=*/4);
    cpu.options = options;
    const auto golden = reference(cpu);
    KdaInputs inputs = to_device(cpu, device_);
    torch::Tensor output = torch::empty_like(inputs.query);
    torch::Tensor final_state = torch::empty_like(inputs.initial_state);
    kda_prefill(inputs, output, final_state, workspace);
    expect_close(output, golden.first, 0.002, 0.03, 0.025);
    expect_close(final_state, golden.second, 0.02, 0.05, 0.03);
    torch::Tensor state = inputs.initial_state;
    std::vector<torch::Tensor> parts;
    parts.reserve(17);
    for (int64_t token = 0; token < 17; ++token) {
      KdaInputs step = token_slice(inputs, token, 1, state);
      torch::Tensor part = torch::empty_like(step.query);
      torch::Tensor next = torch::empty_like(state);
      kda_decode(step, part, next);
      parts.emplace_back(std::move(part));
      state = std::move(next);
    }
    expect_close(torch::cat(parts, 1), golden.first, 0.002, 0.03, 0.025);
    expect_close(state, golden.second, 2e-6, 5e-5, 5e-6);
    EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
  }
}

TEST_F(MusaRecurrentOpsTest, RejectsUnsetAndNonfiniteKdaOptions) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  KdaInputs inputs =
      to_device(make_inputs(/*batch=*/1, /*tokens=*/1, /*heads=*/4), device_);
  torch::Tensor output = torch::empty_like(inputs.query);
  torch::Tensor state = torch::empty_like(inputs.initial_state);
  KdaWorkspace workspace(
      /*max_batch=*/1, /*max_tokens=*/1, /*heads=*/4, device_);
  KdaInputs invalid = inputs;
  invalid.options = {};
  EXPECT_DEATH(kda_prefill(invalid, output, state, workspace),
               "gate_lower_bound");
  EXPECT_DEATH(kda_decode(invalid, output, state), "gate_lower_bound");
  invalid = inputs;
  invalid.options.gate_lower_bound = std::numeric_limits<float>::quiet_NaN();
  EXPECT_DEATH(kda_decode(invalid, output, state), "gate_lower_bound");
  invalid = inputs;
  invalid.options.scale = 0.0f;
  EXPECT_DEATH(kda_prefill(invalid, output, state, workspace), "scale");
  invalid.options.scale = std::numeric_limits<float>::infinity();
  EXPECT_DEATH(kda_decode(invalid, output, state), "scale");
}

TEST_F(MusaRecurrentOpsTest, ChunkContinuationAndBatchReorderingPreserveState) {
  const KdaInputs cpu = make_inputs(2, 65, 8);
  const auto golden = reference(cpu);
  KdaInputs inputs = to_device(cpu, device_);
  KdaWorkspace workspace(2, 65, 8, device_);
  const void* scratch_address = workspace.storage().data_ptr();
  torch::Tensor state = inputs.initial_state;
  std::vector<torch::Tensor> parts;
  parts.reserve(3);
  const std::vector<int64_t> cuts{0, 3, 17, 65};
  for (size_t i = 1; i < cuts.size(); ++i) {
    KdaInputs chunk =
        token_slice(inputs, cuts[i - 1], cuts[i] - cuts[i - 1], state);
    torch::Tensor output = torch::empty_like(chunk.query);
    torch::Tensor updated = torch::empty_like(state);
    kda_prefill(chunk, output, updated, workspace);
    parts.push_back(std::move(output));
    state = std::move(updated);
  }
  expect_close(torch::cat(parts, 1), golden.first, 0.002, 0.03, 0.025);
  expect_close(state, golden.second, 0.02, 0.05, 0.03);
  KdaInputs reordered{inputs.query.flip({0}),
                      inputs.key.flip({0}),
                      inputs.value.flip({0}),
                      inputs.raw_forget.flip({0}),
                      inputs.raw_beta.flip({0}),
                      inputs.a_log,
                      inputs.dt_bias,
                      inputs.initial_state.flip({0}),
                      inputs.options};
  torch::Tensor output = torch::empty_like(reordered.query);
  torch::Tensor updated = torch::empty_like(reordered.initial_state);
  kda_prefill(reordered, output, updated, workspace);
  expect_close(output.flip({0}), golden.first, 0.002, 0.03, 0.025);
  expect_close(updated.flip({0}), golden.second, 0.02, 0.05, 0.03);
  EXPECT_EQ(workspace.storage().data_ptr(), scratch_address);
  EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
}

TEST_F(MusaRecurrentOpsTest,
       DecodeReplaysOnNondefaultStreamWithNewStateAndGates) {
  KdaInputs cpu = make_inputs(2, 1, 8);
  KdaInputs inputs = to_device(cpu, device_);
  torch::Tensor output = torch::empty_like(inputs.query);
  torch::Tensor state = torch::empty_like(inputs.initial_state);
  kda_decode(inputs, output, state);
  stream_->synchronize();
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(
      musaStreamBeginCapture(stream_->stream(), musaStreamCaptureModeGlobal),
      musaSuccess);
  kda_decode(inputs, output, state);
  ASSERT_EQ(musaStreamEndCapture(stream_->stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
            musaSuccess);
  for (int64_t iteration = 0; iteration < 3; ++iteration) {
    cpu.initial_state =
        pattern({2, 8, 128, 128}, 8.0 + iteration, 0.07, torch::kFloat32);
    cpu.raw_forget = pattern({2, 1, 8, 128}, 9.0 + iteration, 4.0);
    inputs.initial_state.copy_(cpu.initial_state);
    inputs.raw_forget.copy_(cpu.raw_forget);
    const auto golden = reference(cpu);
    ASSERT_EQ(musaGraphLaunch(executable, stream_->stream()), musaSuccess);
    stream_->synchronize();
    expect_close(output, golden.first, 0.002, 0.03, 0.025);
    expect_close(state, golden.second, 2e-6, 5e-5, 5e-6);
    EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST_F(MusaRecurrentOpsTest, PrefillGraphReplaysWithNewStateAndRawGates) {
  KdaInputs cpu = make_inputs(2, 17, 8);
  KdaInputs inputs = to_device(cpu, device_);
  KdaWorkspace workspace(
      /*max_batch=*/2, /*max_tokens=*/17, /*heads=*/8, device_);
  torch::Tensor output = torch::empty_like(inputs.query);
  torch::Tensor state = torch::empty_like(inputs.initial_state);
  kda_prefill(inputs, output, state, workspace);
  stream_->synchronize();
  musaGraph_t graph = nullptr;
  musaGraphExec_t executable = nullptr;
  ASSERT_EQ(
      musaStreamBeginCapture(stream_->stream(), musaStreamCaptureModeGlobal),
      musaSuccess);
  kda_prefill(inputs, output, state, workspace);
  ASSERT_EQ(musaStreamEndCapture(stream_->stream(), &graph), musaSuccess);
  ASSERT_EQ(musaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
            musaSuccess);
  for (int64_t iteration = 0; iteration < 3; ++iteration) {
    cpu.initial_state =
        pattern({2, 8, 128, 128}, 11.0 + iteration, 0.07, torch::kFloat32);
    cpu.raw_forget = pattern({2, 17, 8, 128}, 12.0 + iteration, 4.0);
    inputs.initial_state.copy_(cpu.initial_state);
    inputs.raw_forget.copy_(cpu.raw_forget);
    const auto golden = reference(cpu);
    ASSERT_EQ(musaGraphLaunch(executable, stream_->stream()), musaSuccess);
    stream_->synchronize();
    expect_close(output, golden.first, 0.002, 0.03, 0.025);
    expect_close(state, golden.second, 0.02, 0.05, 0.03);
    EXPECT_TRUE(torch::equal(inputs.initial_state.cpu(), cpu.initial_state));
  }
  ASSERT_EQ(musaGraphExecDestroy(executable), musaSuccess);
  ASSERT_EQ(musaGraphDestroy(graph), musaSuccess);
}

TEST_F(MusaRecurrentOpsTest, ZeroQueryKeyAndSaturatedRawGatesRemainFinite) {
  KdaInputs cpu = make_inputs(/*batch=*/1, /*tokens=*/17, /*heads=*/8);
  cpu.query.select(2, 0).zero_();
  cpu.key.select(2, 1).zero_();
  cpu.raw_forget.narrow(2, 0, 4).fill_(100.0);
  cpu.raw_forget.narrow(2, 4, 4).fill_(-100.0);
  cpu.raw_beta = pattern({1, 17, 8}, /*phase=*/3.5, /*amplitude=*/100.0);
  cpu.a_log.zero_();
  cpu.dt_bias.zero_();
  const auto golden = reference(cpu);
  KdaInputs inputs = to_device(cpu, device_);
  KdaWorkspace workspace(
      /*max_batch=*/1, /*max_tokens=*/17, /*heads=*/8, device_);
  torch::Tensor output = torch::empty_like(inputs.query);
  torch::Tensor state = torch::empty_like(inputs.initial_state);
  kda_prefill(inputs, output, state, workspace);
  expect_close(output, golden.first, 0.002, 0.03, 0.025);
  expect_close(state, golden.second, 0.02, 0.05, 0.03);
  torch::Tensor carried = inputs.initial_state;
  std::vector<torch::Tensor> parts;
  parts.reserve(17);
  for (int64_t t = 0; t < 17; ++t) {
    KdaInputs step = token_slice(inputs, t, 1, carried);
    torch::Tensor part = torch::empty_like(step.query);
    torch::Tensor next = torch::empty_like(carried);
    kda_decode(step, part, next);
    parts.emplace_back(std::move(part));
    carried = std::move(next);
  }
  expect_close(torch::cat(parts, 1), golden.first, 0.002, 0.03, 0.025);
  expect_close(carried, golden.second, 2e-6, 5e-5, 5e-6);
}

TEST_F(MusaRecurrentOpsTest,
       CausalConvMatchesScalarReferenceAndContinuesHistory) {
  const int64_t batch = 2;
  const int64_t tokens = 5;
  const int64_t channels = 8192;
  torch::Tensor cpu_x = pattern({batch, tokens, channels}, 0.3);
  torch::Tensor cpu_weight = pattern({channels, 4}, 1.7, 0.3);
  torch::Tensor cpu_history = pattern({batch, 3, channels}, 2.8);
  torch::Tensor golden = torch::empty_like(cpu_x, torch::dtype(torch::kDouble));
  torch::Tensor history_golden = torch::empty_like(cpu_history);
  const torch::Tensor x64 = cpu_x.to(torch::kDouble);
  const torch::Tensor w64 = cpu_weight.to(torch::kDouble);
  const torch::Tensor h64 = cpu_history.to(torch::kDouble);
  const double* x = x64.data_ptr<double>();
  const double* w = w64.data_ptr<double>();
  const double* h = h64.data_ptr<double>();
  double* y = golden.data_ptr<double>();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < channels; ++c) {
      double history[3]{h[(b * 3) * channels + c],
                        h[(b * 3 + 1) * channels + c],
                        h[(b * 3 + 2) * channels + c]};
      for (int64_t t = 0; t < tokens; ++t) {
        const int64_t position = (b * tokens + t) * channels + c;
        const double current = x[position];
        const double sum = history[0] * w[c * 4] + history[1] * w[c * 4 + 1] +
                           history[2] * w[c * 4 + 2] + current * w[c * 4 + 3];
        y[position] = sum / (1.0 + std::exp(-sum));
        history[0] = history[1];
        history[1] = history[2];
        history[2] = current;
      }
      for (int64_t d = 0; d < 3; ++d) {
        history_golden.data_ptr<torch::BFloat16>()[(b * 3 + d) * channels + c] =
            static_cast<torch::BFloat16>(history[d]);
      }
    }
  }
  torch::Tensor input = cpu_x.to(device_);
  torch::Tensor weight = cpu_weight.to(device_);
  torch::Tensor initial = cpu_history.to(device_);
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor final = torch::empty_like(initial);
  causal_conv1d_with_state(input, weight, initial, output, final);
  expect_close(output, golden, 0.004, 0.01, 0.004);
  EXPECT_TRUE(torch::equal(final.cpu(), history_golden));
  EXPECT_TRUE(torch::equal(initial.cpu(), cpu_history));
  std::vector<torch::Tensor> pieces;
  pieces.reserve(tokens);
  torch::Tensor history = initial;
  for (int64_t t = 0; t < tokens; ++t) {
    torch::Tensor one = input.narrow(1, t, 1).contiguous();
    torch::Tensor result = torch::empty_like(one);
    torch::Tensor next = torch::empty_like(history);
    causal_conv1d_with_state(one, weight, history, result, next);
    pieces.push_back(std::move(result));
    history = std::move(next);
  }
  EXPECT_TRUE(torch::equal(torch::cat(pieces, 1).cpu(), output.cpu()));
  EXPECT_TRUE(torch::equal(history.cpu(), final.cpu()));
}

TEST_F(MusaRecurrentOpsTest,
       OutputNormUsesSigmoidAndPreservesPerHeadNormalization) {
  const torch::Tensor cpu_value = pattern({3, 64, 128}, 0.8);
  const torch::Tensor cpu_gate = pattern({3, 64, 128}, 2.2, 8.0);
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat32}) {
    const torch::Tensor cpu_weight = pattern({128}, 4.1, 1.0, dtype);
    const torch::Tensor v64 = cpu_value.to(torch::kDouble);
    const torch::Tensor g64 = cpu_gate.to(torch::kDouble);
    const torch::Tensor w64 = cpu_weight.to(torch::kDouble);
    torch::Tensor golden =
        torch::empty(cpu_value.sizes(), torch::dtype(torch::kDouble));
    for (int64_t row = 0; row < cpu_value.numel() / 128; ++row) {
      double squares = 0.0;
      for (int64_t d = 0; d < 128; ++d) {
        const double x = v64.data_ptr<double>()[row * 128 + d];
        squares += x * x;
      }
      for (int64_t d = 0; d < 128; ++d) {
        const int64_t offset = row * 128 + d;
        golden.data_ptr<double>()[offset] =
            v64.data_ptr<double>()[offset] / std::sqrt(squares / 128.0 + 1e-6) *
            w64.data_ptr<double>()[d] /
            (1.0 + std::exp(-g64.data_ptr<double>()[offset]));
      }
    }
    torch::Tensor value = cpu_value.to(device_);
    torch::Tensor gate = cpu_gate.to(device_);
    torch::Tensor weight = cpu_weight.to(device_);
    torch::Tensor output = torch::empty_like(value);
    rms_norm_sigmoid_gated(value, gate, weight, 1e-6, output);
    expect_close(output, golden, 0.02, 0.01, 0.004);
    EXPECT_TRUE(torch::equal(value.cpu(), cpu_value));
  }
}

TEST_F(MusaRecurrentOpsTest, RejectsAliasingInvalidShapesAndWorkspaceOverflow) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  KdaInputs inputs = to_device(make_inputs(1, 1, 8), device_);
  torch::Tensor output = torch::empty_like(inputs.query);
  torch::Tensor state = torch::empty_like(inputs.initial_state);
  KdaWorkspace workspace(1, 1, 8, device_);
  EXPECT_DEATH(kda_decode(inputs, output, inputs.initial_state), "is_alias_of");
  EXPECT_DEATH(kda_decode(inputs, inputs.query, state), "is_alias_of");
  KdaInputs invalid = inputs;
  invalid.initial_state =
      torch::empty({1, 8, 128, 127}, inputs.initial_state.options());
  EXPECT_DEATH(kda_decode(invalid, output, state), "sizes");
  invalid = inputs;
  invalid.initial_state = inputs.initial_state.to(torch::kBFloat16);
  EXPECT_DEATH(kda_decode(invalid, output, state), "scalar_type");
  KdaInputs two = to_device(make_inputs(1, 2, 8), device_);
  torch::Tensor two_output = torch::empty_like(two.query);
  EXPECT_DEATH(kda_decode(two, two_output, state), "one token");
  EXPECT_DEATH(kda_prefill(two, two_output, state, workspace), "max_tokens");
}

}  // namespace
}  // namespace xllm::kernel::musa
