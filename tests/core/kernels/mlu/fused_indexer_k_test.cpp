/* Copyright 2026 The xLLM Authors.

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

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
constexpr float kSentinel = -7.0f;

struct Inputs {
  torch::Tensor x, latent, wp, wk, gamma, rope, positions, slots;
  torch::Tensor cache, head, cache_storage, head_storage;
};

Inputs make_inputs(int64_t mx, int64_t mk) {
  torch::manual_seed(2026);
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  Inputs data;
  data.x = torch::randn({mx, 5120}).to(bf16).to(device);
  torch::Tensor latent = torch::randn({mk, 512});
  latent *=
      torch::rsqrt(latent.square().mean(/*dim=*/-1, /*keepdim=*/true) + 1e-20);
  data.latent = latent.to(bf16).to(device);
  data.wp = (torch::randn({32, 5120}) / std::sqrt(5120.0)).to(bf16).to(device);
  data.wk = (torch::randn({128, 512}) / std::sqrt(512.0)).to(bf16).to(device);
  data.gamma = (1 + torch::randn({128}) * 0.05).to(bf16).to(device);
  const int64_t table_rows = std::max<int64_t>(mk * 2 + 17, 512);
  const torch::Tensor angles =
      torch::arange(table_rows, torch::kFloat32).unsqueeze(1) *
      (torch::arange(32, torch::kFloat32) + 1)
          .repeat_interleave(2)
          .unsqueeze(0) *
      0.003;
  data.rope = torch::cat({angles.cos(), angles.sin()}, /*dim=*/1).to(device);
  const torch::Tensor logical = torch::arange(mk, torch::kInt64);
  data.positions =
      ((logical * 3 + 1).remainder(table_rows)).to(torch::kInt32).to(device);
  if (mk > 0) {
    data.positions[mk - 1] = table_rows - 1;
  }
  const int64_t blocks = (mk + 127) / 128 + 2;
  data.slots =
      ((torch::floor_divide(logical, 128) + 1).remainder(blocks) * 128 +
       logical.remainder(128))
          .to(torch::kInt32)
          .to(device);
  data.cache_storage =
      torch::full({blocks * 128 + 256, 128}, kSentinel, bf16.device(device));
  data.cache =
      data.cache_storage.slice(/*dim=*/0, /*start=*/128, blocks * 128 + 128)
          .view({blocks, 1, 128, 128});
  data.head_storage = torch::full(
      {mx + 2, 32}, -23.0, bf16.device(device).dtype(torch::kFloat32));
  data.head = data.head_storage.slice(/*dim=*/0, /*start=*/1, mx + 1);
  return data;
}

torch::Tensor run(
    const Inputs& data,
    const std::optional<torch::Tensor>& num_valid_input_rows = std::nullopt,
    const std::optional<torch::Tensor>& num_valid_compressed_rows =
        std::nullopt) {
  return fused_indexer_k(data.x,
                         data.latent,
                         data.wp,
                         data.wk,
                         data.gamma,
                         data.rope,
                         data.positions,
                         data.slots,
                         data.cache,
                         data.head,
                         num_valid_input_rows,
                         num_valid_compressed_rows);
}

torch::Tensor rounded(const torch::Tensor& value) {
  return value.to(torch::kBFloat16).to(torch::kFloat64);
}

std::pair<torch::Tensor, torch::Tensor> reference(const Inputs& data,
                                                  bool staged,
                                                  int64_t lx,
                                                  int64_t lk) {
  torch::Tensor head =
      torch::matmul(data.x.cpu().slice(0, 0, lx).to(torch::kFloat64),
                    data.wp.cpu().to(torch::kFloat64).t());
  torch::Tensor key =
      torch::matmul(data.latent.cpu().slice(0, 0, lk).to(torch::kFloat64),
                    data.wk.cpu().to(torch::kFloat64).t());
  torch::Tensor gamma = data.gamma.cpu().to(torch::kFloat64);
  if (staged) {
    head = rounded(head);
    key = rounded(key).to(torch::kFloat32);
    gamma = gamma.to(torch::kFloat32);
  }
  head /= 64;
  key = key * torch::rsqrt(key.square().mean(-1, true) + 1e-20) * gamma;
  if (staged) {
    key = rounded(key);
  }
  const torch::Tensor valid = data.slots.cpu().slice(0, 0, lk).ge(0);
  const torch::Tensor positions = torch::where(
      valid, data.positions.cpu().slice(0, 0, lk).to(torch::kInt64), 0);
  torch::Tensor table =
      data.rope.cpu().index_select(0, positions).to(torch::kFloat64);
  torch::Tensor tail = key.slice(/*dim=*/1, /*start=*/64).clone();
  if (staged) {
    tail = tail.to(torch::kFloat32);
    table = table.to(torch::kFloat32);
  }
  // Compute pair components separately from the production rotate helper.
  const torch::Tensor even = torch::arange(0, 64, 2, torch::kInt64);
  const torch::Tensor odd = even + 1;
  torch::Tensor rotated = torch::empty_like(tail);
  rotated.index_copy_(
      1,
      even,
      tail.index_select(1, even) * table.index_select(1, even) -
          tail.index_select(1, odd) * table.index_select(1, even + 64));
  rotated.index_copy_(
      1,
      odd,
      tail.index_select(1, odd) * table.index_select(1, odd) +
          tail.index_select(1, even) * table.index_select(1, odd + 64));
  key.slice(/*dim=*/1, /*start=*/64).copy_(rotated);
  return {head, staged ? rounded(key) : key};
}

std::pair<torch::Tensor, torch::Tensor> torch_reference(const Inputs& data,
                                                        int64_t lx,
                                                        int64_t lk) {
  const torch::Tensor head =
      torch::matmul(data.x.slice(0, 0, lx), data.wp.t()).to(torch::kFloat32) /
      64;
  torch::Tensor key = torch::matmul(data.latent.slice(0, 0, lk), data.wk.t())
                          .to(torch::kFloat32);
  key = (key * torch::rsqrt(key.square().mean(-1, true) + 1e-20) *
         data.gamma.to(torch::kFloat32))
            .to(torch::kBFloat16);
  const torch::Tensor valid = data.slots.slice(0, 0, lk).ge(0);
  const torch::Tensor positions =
      torch::where(valid, data.positions.slice(0, 0, lk).to(torch::kInt64), 0);
  const torch::Tensor table = data.rope.index_select(0, positions);
  const torch::Tensor tail = key.slice(1, 64).to(torch::kFloat32);
  const torch::Tensor pairs = tail.reshape({lk, 32, 2});
  const torch::Tensor swapped =
      torch::stack({-pairs.select(2, 1), pairs.select(2, 0)}, -1)
          .reshape({lk, 64});
  key.slice(1, 64).copy_(
      (tail * table.slice(1, 0, 64) + swapped * table.slice(1, 64))
          .to(torch::kBFloat16));
  return {head.cpu(), key.cpu()};
}

void expect_precision(const torch::Tensor& actual,
                      const torch::Tensor& baseline,
                      const torch::Tensor& golden,
                      double floor) {
  ASSERT_TRUE(actual.sizes() == golden.sizes());
  if (actual.numel() == 0) {
    return;
  }
  const torch::Tensor result = actual.cpu().to(torch::kFloat64);
  const torch::Tensor base = baseline.cpu().to(torch::kFloat64);
  const torch::Tensor ideal = golden.to(torch::kFloat64);
  for (const auto& value : {result, base}) {
    ASSERT_TRUE(torch::equal(value.isnan(), ideal.isnan()));
    ASSERT_TRUE(torch::equal(value.isposinf(), ideal.isposinf()));
    ASSERT_TRUE(torch::equal(value.isneginf(), ideal.isneginf()));
  }
  const torch::Tensor finite = ideal.isfinite();
  const torch::Tensor g = ideal.masked_select(finite);
  if (g.numel() == 0) {
    return;
  }
  const torch::Tensor r = result.masked_select(finite);
  const torch::Tensor b = base.masked_select(finite);
  const torch::Tensor denom = torch::where(g.eq(0), 1e-9, g.abs());
  const double l1 = denom.sum().item<double>();
  const double l2 = denom.square().sum().sqrt().item<double>();
  EXPECT_LE((r - g).abs().sum().item<double>() / l1,
            std::max((b - g).abs().sum().item<double>() / l1, floor) * 10);
  EXPECT_LE(
      (r - g).square().sum().sqrt().item<double>() / l2,
      std::max((b - g).square().sum().sqrt().item<double>() / l2, floor) * 10);
  const int64_t gt = r.gt(g).sum().item<int64_t>();
  const int64_t lt = r.lt(g).sum().item<int64_t>();
  const int64_t base_gt = b.gt(g).sum().item<int64_t>();
  const int64_t base_lt = b.lt(g).sum().item<int64_t>();
  EXPECT_TRUE(gt + lt < 100 || (gt > 0 && lt > 0) ||
              (base_gt > 0 && base_lt == 0) || (base_lt > 0 && base_gt == 0));
}

void check_result(const Inputs& data,
                  int64_t lx,
                  int64_t lk,
                  const torch::Tensor& before) {
  const auto [gold_head, gold_key] = reference(data, /*staged=*/false, lx, lk);
  const auto [staged_head, staged_key] =
      reference(data, /*staged=*/true, lx, lk);
  const auto [base_head, base_key] = torch_reference(data, lx, lk);
  const torch::Tensor actual_head = data.head.cpu();
  const torch::Tensor actual_cache = data.cache.cpu().view({-1, 128});
  const torch::Tensor valid =
      torch::nonzero(data.slots.cpu().slice(0, 0, lk).ge(0)).flatten();
  const torch::Tensor slots =
      data.slots.cpu().slice(0, 0, lk).index_select(0, valid).to(torch::kInt64);
  const torch::Tensor actual_key = actual_cache.index_select(0, slots);
  expect_precision(
      actual_head.slice(0, 0, lx), base_head, gold_head, /*floor=*/1e-6);
  expect_precision(
      actual_head.slice(0, 0, lx), staged_head, gold_head, /*floor=*/1e-6);
  expect_precision(actual_key,
                   base_key.index_select(0, valid),
                   gold_key.index_select(0, valid),
                   /*floor=*/1e-4);
  expect_precision(actual_key,
                   staged_key.index_select(0, valid),
                   gold_key.index_select(0, valid),
                   /*floor=*/1e-4);
  EXPECT_TRUE(actual_head.slice(0, lx).eq(0).all().item<bool>());
  torch::Tensor unchanged = before.cpu().view({-1, 128}).clone();
  unchanged.index_copy_(0, slots, actual_key);
  EXPECT_TRUE(torch::allclose(
      actual_cache, unchanged, /*rtol=*/0, /*atol=*/0, /*equal_nan=*/true));
  EXPECT_TRUE(
      data.cache_storage.slice(0, 0, 128).eq(kSentinel).all().item<bool>());
  EXPECT_TRUE(data.cache_storage.slice(0, data.cache_storage.size(0) - 128)
                  .eq(kSentinel)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(data.head_storage[0].eq(-23).all().item<bool>());
  EXPECT_TRUE(data.head_storage[-1].eq(-23).all().item<bool>());
}

TEST(FusedIndexerKTest, MatchesReferencesAcrossDispatchAndUnbalancedRows) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  const std::vector<std::pair<int64_t, int64_t>> shapes = {
      {1, 1},      {17, 9},      {31, 31},    {32, 32},    {33, 33},
      {63, 63},    {64, 64},     {65, 65},    {96, 96},    {127, 127},
      {128, 128},  {129, 129},   {255, 255},  {256, 256},  {257, 257},
      {128, 129},  {129, 128},   {129, 256},  {256, 129},  {128, 257},
      {257, 128},  {511, 0},     {512, 0},    {513, 0},    {0, 511},
      {0, 512},    {0, 513},     {257, 1009}, {104, 1025}, {115, 1025},
      {128, 1793}, {2048, 8192}, {17, 0},     {0, 65},     {0, 0}};
  for (const auto& [mx, mk] : shapes) {
    SCOPED_TRACE(::testing::Message() << "x=" << mx << " k=" << mk);
    Inputs data = make_inputs(mx, mk);
    const torch::Tensor before = data.cache.cpu();
    const torch::Tensor result = run(data);
    EXPECT_EQ(result.data_ptr(), data.head.data_ptr());
    check_result(data, mx, mk, before);
  }
}

TEST(FusedIndexerKTest, EmptyBranchesAllowAbsentInputs) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] :
       std::vector<std::pair<int64_t, int64_t>>{{17, 0}, {0, 65}, {0, 0}}) {
    Inputs data = make_inputs(mx, mk);
    const torch::Tensor before = data.cache.cpu();
    fused_indexer_k(mx ? std::optional(data.x) : std::nullopt,
                    mk ? std::optional(data.latent) : std::nullopt,
                    mx ? std::optional(data.wp) : std::nullopt,
                    mk ? std::optional(data.wk) : std::nullopt,
                    mk ? std::optional(data.gamma) : std::nullopt,
                    mk ? std::optional(data.rope) : std::nullopt,
                    mk ? std::optional(data.positions) : std::nullopt,
                    mk ? std::optional(data.slots) : std::nullopt,
                    data.cache,
                    data.head);
    check_result(data, mx, mk, before);
  }
}

TEST(FusedIndexerKTest,
     MasksSlotsBeforeReadingPositionsAndHonorsIndependentLive) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] : std::vector<std::pair<int64_t, int64_t>>{
           {17, 9}, {129, 65}, {256, 256}, {257, 1009}, {2048, 8192}}) {
    Inputs data = make_inputs(mx, mk);
    const torch::Tensor num_valid_input_rows =
        torch::zeros({1}, data.positions.options());
    const torch::Tensor num_valid_compressed_rows =
        torch::zeros_like(num_valid_input_rows);
    for (const auto& [lx, lk] : std::vector<std::pair<int64_t, int64_t>>{
             {mx, mk}, {0, 0}, {1, mk - 1}, {mx - 1, 1}}) {
      data.cache.fill_(kSentinel);
      const torch::Tensor before = data.cache.cpu();
      data.head.fill_(std::numeric_limits<float>::quiet_NaN());
      data.slots.slice(0, 0, mk, 3).fill_(-1);
      data.positions.masked_fill_(data.slots.lt(0),
                                  std::numeric_limits<int32_t>::max());
      num_valid_input_rows.fill_(lx);
      num_valid_compressed_rows.fill_(lk);
      const torch::Tensor positions = data.positions.clone();
      data.positions.slice(0, lk).fill_(std::numeric_limits<int32_t>::max());
      run(data, num_valid_input_rows, num_valid_compressed_rows);
      check_result(data, lx, lk, before);
      data.positions.copy_(positions);
    }
    data.slots.fill_(-1);
    data.positions.fill_(std::numeric_limits<int32_t>::max());
    data.cache.fill_(kSentinel);
    run(data);
    check_result(data, mx, mk, data.cache.cpu());
  }
}

TEST(FusedIndexerKTest, PreservesStridedInputsOffsetsAndBothGammaTypes) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] : std::vector<std::pair<int64_t, int64_t>>{
           {17, 9}, {129, 65}, {256, 256}, {257, 1009}}) {
    Inputs data = make_inputs(mx, mk);
    std::vector<std::pair<torch::Tensor, torch::Tensor>> saved;
    for (torch::Tensor* tensor : {&data.x, &data.latent, &data.rope}) {
      torch::Tensor storage = torch::full(
          {tensor->size(0) + 2, tensor->size(1) + 13}, -19, tensor->options());
      torch::Tensor view =
          storage.slice(0, 1, storage.size(0) - 1).slice(1, 0, tensor->size(1));
      view.copy_(*tensor);
      *tensor = view;
      saved.emplace_back(storage, storage.clone());
    }
    for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat32}) {
      data.gamma = data.gamma.to(dtype);
      data.cache.fill_(kSentinel);
      const torch::Tensor before = data.cache.cpu();
      // Framework storage uses [blocks,128,1,128], with the same physical rows.
      data.cache = data.cache.view({data.cache.size(0), 128, 1, 128});
      run(data);
      check_result(data, mx, mk, before);
      for (const auto& [storage, original] : saved) {
        EXPECT_TRUE(torch::equal(storage.cpu(), original.cpu()));
      }
    }
  }
}

TEST(FusedIndexerKTest, MatchesZeroNearZeroRoundingAndNonfiniteValues) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] :
       std::vector<std::pair<int64_t, int64_t>>{{17, 9}, {129, 65}}) {
    for (const float value : {0.0f,
                              1e-15f,
                              0.99609375f,
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
      Inputs data = make_inputs(mx, mk);
      if (std::isfinite(value)) {
        data.latent.fill_(value);
        data.x.fill_(value);
      } else {
        data.latent[0][0] = value;
        data.x[0][0] = value;
      }
      const torch::Tensor before = data.cache.cpu();
      run(data);
      check_result(data, mx, mk, before);
    }
  }
}

TEST(FusedIndexerKTest, ReplaysChangingLiveMetadataAndAccumulatesCache) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] :
       std::vector<std::pair<int64_t, int64_t>>{{96, 96},
                                                {129, 65},
                                                {129, 129},
                                                {256, 256},
                                                {257, 1009},
                                                {257, 0},
                                                {512, 0},
                                                {0, 512},
                                                {0, 2017}}) {
    Inputs data = make_inputs(mx, mk);
    const torch::Tensor num_valid_input_rows =
        torch::full({1}, mx, data.positions.options());
    const torch::Tensor num_valid_compressed_rows =
        torch::full({1}, mk, data.positions.options());
    const torch::Tensor x = data.x.clone();
    const torch::Tensor latent = data.latent.clone();
    const torch::Tensor positions = data.positions.clone();
    const torch::Tensor slots = data.slots.clone();
    struct State {
      int64_t lx, lk;
      torch::Tensor head, cache;
    };
    std::vector<State> states;
    for (const auto& [lx, lk] : std::vector<std::pair<int64_t, int64_t>>{
             {mx, mk},
             {0, 0},
             {std::min<int64_t>(1, mx), mk},
             {mx, std::min<int64_t>(1, mk)},
             {mx / 2, mk / 2},
             {mx, mk}}) {
      const int64_t id = static_cast<int64_t>(states.size());
      data.x.copy_(x + id / 32.0);
      data.latent.copy_(latent * (1 - id / 1024.0));
      data.positions.copy_((positions + id).remainder(data.rope.size(0)));
      data.positions.slice(0, lk).fill_(std::numeric_limits<int32_t>::max());
      data.slots.copy_(slots.roll(id));
      num_valid_input_rows.fill_(lx);
      num_valid_compressed_rows.fill_(lk);
      data.cache.fill_(kSentinel);
      const torch::Tensor before = data.cache.cpu();
      run(data, num_valid_input_rows, num_valid_compressed_rows);
      check_result(data, lx, lk, before);
      states.emplace_back(
          State{lx, lk, data.head.cpu().clone(), data.cache.cpu().clone()});
    }
    run(data, num_valid_input_rows, num_valid_compressed_rows);
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(
          /*isHighPriority=*/false, data.x.device().index()));
      graph.capture_begin();
      run(data, num_valid_input_rows, num_valid_compressed_rows);
      graph.capture_end();
    }
    const void* head_ptr = data.head.data_ptr();
    const void* cache_ptr = data.cache.data_ptr();
    torch::Tensor expected =
        torch::full_like(data.cache.cpu(), kSentinel).view({-1, 128});
    data.cache.fill_(kSentinel);
    for (int64_t step = 0; step < 1000; ++step) {
      const int64_t id = step % static_cast<int64_t>(states.size());
      const State& state = states[id];
      data.x.copy_(x + id / 32.0);
      data.latent.copy_(latent * (1 - id / 1024.0));
      data.positions.copy_((positions + id).remainder(data.rope.size(0)));
      data.positions.slice(0, state.lk)
          .fill_(std::numeric_limits<int32_t>::max());
      data.slots.copy_(slots.roll(id));
      num_valid_input_rows.fill_(state.lx);
      num_valid_compressed_rows.fill_(state.lk);
      graph.replay();
      const torch::Tensor active_slots =
          data.slots.cpu().slice(0, 0, state.lk).to(torch::kInt64);
      expected.index_copy_(
          0,
          active_slots,
          state.cache.view({-1, 128}).index_select(0, active_slots));
      ASSERT_TRUE(torch::equal(data.head.cpu(), state.head)) << step;
      ASSERT_TRUE(torch::equal(data.cache.cpu().view({-1, 128}), expected))
          << step;
      ASSERT_EQ(data.head.data_ptr(), head_ptr);
      ASSERT_EQ(data.cache.data_ptr(), cache_ptr);
    }
  }
}

TEST(FusedIndexerKTest, PreservesReadOnlyInputs) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  Inputs data = make_inputs(/*mx=*/129, /*mk=*/1025);
  std::vector<std::pair<torch::Tensor, torch::Tensor>> saved;
  saved.reserve(8);
  for (const auto& tensor : {data.x,
                             data.latent,
                             data.wp,
                             data.wk,
                             data.gamma,
                             data.rope,
                             data.positions,
                             data.slots}) {
    saved.emplace_back(tensor, tensor.cpu().clone());
  }
  run(data);
  for (const auto& [tensor, original] : saved) {
    EXPECT_TRUE(torch::equal(tensor.cpu(), original));
  }
}

TEST(FusedIndexerKTest, RejectsInvalidTensorContracts) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  Inputs data = make_inputs(/*mx=*/17, /*mk=*/9);
  EXPECT_DEATH(
      {
        auto bad = data;
        bad.x = data.x.to(torch::kFloat32);
        run(bad);
      },
      "scalar_type");
  EXPECT_DEATH(
      {
        auto bad = data;
        bad.positions = data.positions.to(torch::kInt64);
        run(bad);
      },
      "scalar_type");
  EXPECT_DEATH(
      {
        auto bad = data;
        bad.wp = data.wp.t();
        run(bad);
      },
      "sizes");
  EXPECT_DEATH(
      {
        auto bad = data;
        bad.head = data.head.cpu();
        run(bad);
      },
      "device");
}

TEST(FusedIndexerKTest, DISABLED_Uses64BitCacheAndRopeOffsets) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const int64_t boundary : {int64_t{1} << 31, int64_t{1} << 32}) {
    for (const bool test_cache : {false, true}) {
      Inputs data = make_inputs(/*mx=*/257, /*mk=*/65);
      if (test_cache) {
        const int64_t blocks = (boundary / 256 + 65 + 127) / 128;
        data.cache =
            torch::full({blocks, 1, 128, 128}, kSentinel, data.cache.options());
        data.slots.copy_(torch::arange(65, data.slots.options()) +
                         boundary / 256 - 1);
      } else {
        // A sparse row stride reaches high byte offsets with just 65 reads.
        const int64_t rows = boundary / 1024 + 66;
        torch::Tensor table =
            torch::empty_strided({rows, 128}, {256, 1}, data.rope.options());
        data.positions.copy_(torch::arange(65, data.positions.options()) +
                             rows - 65);
        const torch::Tensor values = data.rope.slice(0, 0, 65).clone();
        table.index_copy_(0, data.positions.to(torch::kInt64), values);
        data.rope = table;
      }
      run(data);
      const torch::Tensor result =
          data.cache.view({-1, 128})
              .index_select(0, data.slots.to(torch::kInt64))
              .cpu();
      const auto [head, key] = torch_reference(data, /*lx=*/257, /*lk=*/65);
      Inputs compact = data;
      compact.rope =
          data.rope.index_select(0, data.positions.to(torch::kInt64));
      compact.positions = torch::arange(65, data.positions.options());
      const auto [gold_head, gold_key] =
          reference(compact, /*staged=*/false, /*lx=*/257, /*lk=*/65);
      expect_precision(result, key, gold_key, /*floor=*/1e-4);
      expect_precision(data.head.cpu(), head, gold_head, /*floor=*/1e-6);
      const int64_t first_slot = data.slots[0].cpu().item<int32_t>();
      const int64_t last_slot = data.slots[-1].cpu().item<int32_t>();
      EXPECT_TRUE(data.cache.view({-1, 128})[first_slot - 1]
                      .cpu()
                      .eq(kSentinel)
                      .all()
                      .item<bool>());
      EXPECT_TRUE(data.cache.view({-1, 128})[last_slot + 1]
                      .cpu()
                      .eq(kSentinel)
                      .all()
                      .item<bool>());
    }
  }
}

struct Timing {
  double event_us;
  double host_us;
};

Timing measure(const std::function<void()>& call) {
  for (int32_t i = 0; i < 10; ++i) {
    call();
  }
  torch_mlu::synchronize();
  cnrtNotifier_t start, end;
  CHECK_EQ(cnrtNotifierCreate(&start), cnrtSuccess);
  CHECK_EQ(cnrtNotifierCreate(&end), cnrtSuccess);
  Timing total{};
  for (int32_t group = 0; group < 3; ++group) {
    const auto begin = std::chrono::steady_clock::now();
    CHECK_EQ(cnrtPlaceNotifier(start, torch_mlu::getCurMLUStream()),
             cnrtSuccess);
    for (int32_t i = 0; i < 100; ++i) {
      call();
    }
    CHECK_EQ(cnrtPlaceNotifier(end, torch_mlu::getCurMLUStream()), cnrtSuccess);
    torch_mlu::synchronize();
    float elapsed_ms = 0;
    CHECK_EQ(cnrtNotifierElapsedTime(start, end, &elapsed_ms), cnrtSuccess);
    total.event_us += elapsed_ms * 1000.0 / 100.0;
    total.host_us += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - begin)
                         .count() /
                     100.0;
  }
  CHECK_EQ(cnrtNotifierDestroy(start), cnrtSuccess);
  CHECK_EQ(cnrtNotifierDestroy(end), cnrtSuccess);
  return {total.event_us / 3, total.host_us / 3};
}

TEST(FusedIndexerKTest, DISABLED_ReportsPairedLatency) {
  const torch::DeviceGuard guard(
      torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  for (const auto& [mx, mk] :
       std::vector<std::pair<int64_t, int64_t>>{{16, 16},
                                                {96, 96},
                                                {127, 127},
                                                {128, 128},
                                                {129, 129},
                                                {255, 255},
                                                {256, 256},
                                                {257, 257},
                                                {128, 129},
                                                {129, 128},
                                                {256, 129},
                                                {129, 256},
                                                {511, 0},
                                                {512, 0},
                                                {513, 0},
                                                {0, 511},
                                                {0, 512},
                                                {0, 513},
                                                {2048, 16384}}) {
    Inputs data = make_inputs(mx, mk);
    const int64_t requests = mx == 16 || mx == 96 ? 16 : 1;
    const int64_t rows = mk / requests;
    torch::Tensor base_head;
    const auto baseline = [&]() {
      base_head = torch::matmul(data.x, data.wp.t()).to(torch::kFloat32) / 64;
      for (int64_t begin = 0; begin < mk; begin += rows) {
        torch::Tensor key =
            torch::matmul(data.latent.slice(0, begin, begin + rows),
                          data.wk.t())
                .to(torch::kFloat32);
        key = (key * torch::rsqrt(key.square().mean(-1, true) + 1e-20) *
               data.gamma.to(torch::kFloat32))
                  .to(torch::kBFloat16);
        const torch::Tensor table = data.rope.index_select(
            0, data.positions.slice(0, begin, begin + rows).to(torch::kInt64));
        const torch::Tensor tail = key.slice(1, 64).to(torch::kFloat32);
        const torch::Tensor pairs = tail.reshape({rows, 32, 2});
        const torch::Tensor swapped =
            torch::stack({-pairs.select(2, 1), pairs.select(2, 0)}, -1)
                .reshape({rows, 64});
        key.slice(1, 64).copy_(
            (tail * table.slice(1, 0, 64) + swapped * table.slice(1, 64))
                .to(torch::kBFloat16));
        data.cache.view({-1, 128}).index_copy_(
            0, data.slots.slice(0, begin, begin + rows).to(torch::kInt64), key);
      }
    };
    const Timing prepared = measure([&]() {
      Inputs merged = data;
      std::vector<torch::Tensor> latent, positions, slots;
      latent.reserve(requests);
      positions.reserve(requests);
      slots.reserve(requests);
      for (int64_t begin = 0; begin < mk; begin += rows) {
        latent.emplace_back(data.latent.slice(0, begin, begin + rows));
        positions.emplace_back(data.positions.slice(0, begin, begin + rows));
        slots.emplace_back(data.slots.slice(0, begin, begin + rows));
      }
      if (requests > 1) {
        merged.latent = torch::cat(latent, 0);
        merged.positions = torch::cat(positions, 0);
        merged.slots = torch::cat(slots, 0);
      }
      run(merged);
    });
    LOG(INFO) << "indexer_k x=" << mx << " k=" << mk << " requests=" << requests
              << " prepared_fused_event_us=" << prepared.event_us
              << " prepared_fused_host_us=" << prepared.host_us;
    for (const bool capture : {false, true}) {
      torch_mlu::MLUGraph base_graph, fused_graph;
      if (capture) {
        baseline();
        run(data);
        torch_mlu::synchronize();
        torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(
            /*isHighPriority=*/false, data.x.device().index()));
        base_graph.capture_begin();
        baseline();
        base_graph.capture_end();
        fused_graph.capture_begin();
        run(data);
        fused_graph.capture_end();
      }
      const Timing base = measure([&]() {
        if (capture) {
          base_graph.replay();
        } else {
          baseline();
        }
      });
      const Timing fused = measure([&]() {
        if (capture) {
          fused_graph.replay();
        } else {
          run(data);
        }
      });
      LOG(INFO) << "indexer_k x=" << mx << " k=" << mk << " graph=" << capture
                << " baseline_event_us=" << base.event_us
                << " fused_event_us=" << fused.event_us
                << " baseline_host_us=" << base.host_us
                << " fused_host_us=" << fused.host_us;
    }
  }
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
