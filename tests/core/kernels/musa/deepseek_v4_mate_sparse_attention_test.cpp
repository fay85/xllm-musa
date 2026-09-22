/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace xllm::util {
std::string get_string_env(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  return value == nullptr ? std::string() : std::string(value);
}
int ceil_pow2(int value) {
  if (value <= 1) return 1;
  int result = 1;
  while (result < value) result <<= 1;
  return result;
}
}  // namespace xllm::util

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kHeads = 8;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kPackedBytes = 584;
constexpr int64_t kPayloadBytes = 576;
constexpr int64_t kNopeDim = 448;
constexpr int64_t kQuantGroup = 64;
constexpr double kSoftmaxScale = 0.044194173824159216;

torch::Tensor decode_packed_rows(const torch::Tensor& cache,
                                 const torch::Tensor& physical) {
  TORCH_CHECK(cache.defined());
  TORCH_CHECK(physical.defined());
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cache.dim() == 4);
  TORCH_CHECK(cache.size(2) == 1);
  TORCH_CHECK(cache.size(3) == kPackedBytes);

  auto flat = physical.to(torch::kCPU).reshape({-1}).to(torch::kLong).contiguous();
  auto valid = flat.ge(0).logical_and(
      flat.lt(cache.size(0) * cache.size(1)));
  flat = torch::masked_select(flat, valid).contiguous();
  if (flat.numel() == 0) {
    return torch::empty({0, kHeadDim},
                        torch::TensorOptions().dtype(torch::kFloat32));
  }
  const int64_t page_stride = cache.stride(0);
  auto pages = torch::floor_divide(flat, kBlockSize);
  auto offsets = torch::remainder(flat, kBlockSize);
  auto storage = cache.as_strided({cache.size(0) * page_stride}, {1}).to(torch::kCPU);
  auto payload_base = pages * page_stride + offsets * kPayloadBytes;
  auto payload_offsets = torch::arange(
      kPayloadBytes,
      torch::TensorOptions().dtype(torch::kLong));
  auto payload = storage
                     .index_select(
                         0,
                         (payload_base.unsqueeze(1) +
                          payload_offsets.unsqueeze(0))
                             .reshape({-1}))
                     .view({flat.numel(), kPayloadBytes});

  auto scale_base = pages * page_stride + kBlockSize * kPayloadBytes +
                    offsets * 8;
  auto scale_offsets = torch::arange(
      7, torch::TensorOptions().dtype(torch::kLong));
  auto scale_exponents = storage
                             .index_select(
                                 0,
                                 (scale_base.unsqueeze(1) +
                                  scale_offsets.unsqueeze(0))
                                     .reshape({-1}))
                             .view({flat.numel(), 7})
                             .to(torch::kFloat32) -
                         127.0;
  auto scales = torch::pow(torch::full_like(scale_exponents, 2.0),
                           scale_exponents)
                    .repeat_interleave(kQuantGroup, /*dim=*/1);
  auto nope = payload.slice(1, 0, kNopeDim)
                  .view(torch::kFloat8_e4m3fn)
                  .to(torch::kFloat32) *
              scales;
  auto rope = payload.slice(1, kNopeDim, kPayloadBytes)
                  .view(torch::kBFloat16)
                  .to(torch::kFloat32);
  return torch::cat({nope, rope}, /*dim=*/1).contiguous();
}

torch::Tensor dense_reference(const torch::Tensor& q,
                              const torch::Tensor& cache,
                              const torch::Tensor& main_physical,
                              int64_t topk_length,
                              const torch::Tensor& extra_physical,
                              int64_t extra_topk_length,
                              const torch::Tensor& sink) {
  TORCH_CHECK(q.dim() == 3);
  const int64_t batch = q.size(0);
  std::vector<torch::Tensor> rows;
  rows.reserve(static_cast<size_t>(batch));
  for (int64_t row = 0; row < batch; ++row) {
    auto main_row = main_physical.select(0, row).slice(0, 0, topk_length);
    auto extra_row =
        extra_physical.select(0, row).slice(0, 0, extra_topk_length);
    auto main_kv = decode_packed_rows(cache, main_row);
    auto extra_kv = decode_packed_rows(cache, extra_row);
    auto selected = torch::cat({main_kv, extra_kv}, /*dim=*/0);
    auto q_row = q.select(0, row).to(torch::kCPU).to(torch::kFloat32);
    auto scores = torch::matmul(q_row, selected.transpose(0, 1)) *
                  kSoftmaxScale;
    auto max_score = std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/true));
    auto exp_scores = torch::exp(scores - max_score);
    auto denom = exp_scores.sum(/*dim=*/-1, /*keepdim=*/true);
    if (sink.defined()) {
      denom = denom + torch::exp(sink.to(torch::kCPU).to(torch::kFloat32).view({kHeads, 1}) -
                                 max_score);
    }
    rows.push_back(torch::matmul(exp_scores / denom, selected)
                       .to(torch::kBFloat16));
  }
  return torch::stack(rows, /*dim=*/0).contiguous();
}

struct MateCaseResult {
  torch::Tensor actual;
  torch::Tensor expected;
};

MateCaseResult run_mate_case(const torch::Tensor& q,
                             const torch::Tensor& values,
                             const torch::Tensor& main_physical,
                             int64_t topk_length,
                             const torch::Tensor& extra_physical,
                             int64_t extra_topk_length) {
  const torch::Device device = q.device();
  const int64_t cache_rows = values.size(0);
  const int64_t blocks = (cache_rows + kBlockSize - 1) / kBlockSize;
  const int64_t page_stride =
      ((kBlockSize * kPackedBytes + 576 - 1) / 576) * 576;
  auto storage = torch::zeros({blocks * page_stride},
                              torch::TensorOptions()
                                  .dtype(torch::kUInt8)
                                  .device(device));
  auto cache = storage.as_strided({blocks, kBlockSize, 1, kPackedBytes},
                                  {page_stride, kPackedBytes, kPackedBytes, 1});
  auto slots = torch::arange(cache_rows,
                             torch::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(device));
  deepseek_v4_pack_fp8_cache(cache, slots, values);
  auto sink = torch::full({kHeads},
                          -std::numeric_limits<float>::infinity(),
                          torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(device));
  auto actual = deepseek_v4_mate_sparse_attention_decode(
      q,
      cache,
      main_physical,
      topk_length,
      cache,
      extra_physical,
      extra_topk_length,
      sink,
      kSoftmaxScale,
      std::nullopt);
  if (actual.defined()) {
    actual = actual.contiguous();
  }
  auto expected = dense_reference(q,
                                 cache,
                                 main_physical,
                                 topk_length,
                                 extra_physical,
                                 extra_topk_length,
                                 sink);
  return {actual, expected};
}

torch::Tensor deterministic_values(int64_t rows, const torch::Device& device) {
  auto cpu = torch::arange(rows * kHeadDim,
                           torch::TensorOptions().dtype(torch::kFloat32))
                 .reshape({rows, kHeadDim});
  // Keep the values bounded so the FP8 payload and the BF16 rope payload both
  // exercise nontrivial scales without overflowing the MODEL1 format.
  return ((cpu * 0.0017).sin() * 1.7).to(device, torch::kBFloat16);
}

void expect_close(const MateCaseResult& result, const char* name) {
  ASSERT_TRUE(result.actual.defined())
      << name << ": MATE artifact/kernel returned undefined output";
  ASSERT_EQ(result.actual.sizes(), result.expected.sizes()) << name;
  auto diff = (result.actual.to(torch::kCPU).to(torch::kFloat32) -
               result.expected.to(torch::kFloat32))
                  .abs();
  const float max_abs = diff.max().item<float>();
  const float mean_abs = diff.mean().item<float>();
  LOG(INFO) << "[MATE_UNIT] case=" << name << " max_abs=" << max_abs
            << " mean_abs=" << mean_abs;
  EXPECT_LT(max_abs, 0.08f) << name;
  EXPECT_LT(mean_abs, 0.01f) << name;
}

TEST(DeepseekV4MateSparseAttentionTest, Q1AndQ2MatchDenseReference) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib(
          "dsa_model1_h8_topk512_extra128_bs128"))
           .good()) {
    GTEST_SKIP() << "MATE MODEL1 h8 decode artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAStream stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());

  const int64_t rows = 512;
  auto values = deterministic_values(rows, device);
  auto q1 = deterministic_values(kHeads, device).view({1, kHeads, kHeadDim});
  auto main1 = torch::arange(
                   8, torch::TensorOptions().dtype(torch::kInt32).device(device))
                   .view({1, 8});
  auto extra1 = torch::arange(
                     128,
                     torch::TensorOptions().dtype(torch::kInt32).device(device))
                     .view({1, 128});
  auto q1_result = run_mate_case(
      q1, values, main1, /*topk_length=*/8, extra1, /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(q1_result, "q1_scalar");

  auto q2 = deterministic_values(2 * kHeads, device).view({2, kHeads, kHeadDim});
  auto main2 = torch::stack({
      torch::arange(8, torch::TensorOptions().dtype(torch::kInt32).device(device)),
      torch::arange(8, 16, torch::TensorOptions().dtype(torch::kInt32).device(device))});
  auto extra2 = torch::stack({
      torch::arange(128, torch::TensorOptions().dtype(torch::kInt32).device(device)),
      torch::arange(128, 256,
                    torch::TensorOptions().dtype(torch::kInt32).device(device))});
  auto q2_result = run_mate_case(
      q2, values, main2, /*topk_length=*/8, extra2, /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(q2_result, "q2_batched_rows");
}

TEST(DeepseekV4MateSparseAttentionTest, RollingSwaBoundaryMappingMatchesDense) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib(
          "dsa_model1_h8_topk512_extra128_bs128"))
           .good()) {
    GTEST_SKIP() << "MATE MODEL1 h8 decode artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  c10::musa::MUSAStream stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());

  auto values = deterministic_values(4 * kBlockSize, device);
  auto q2 = deterministic_values(2 * kHeads, device).view({2, kHeads, kHeadDim});
  auto logical_main = torch::tensor(
      {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto compressed_table = torch::tensor(
      {{1, 2}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto rolling_swa_table = torch::tensor(
      {{-1, 2, 3}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto mapped = deepseek_v4_prepare_mate_dsa_indices(logical_main,
                                                     compressed_table,
                                                     rolling_swa_table,
                                                     /*sequence_index=*/0,
                                                     /*kv_len=*/258,
                                                     /*window_size=*/128,
                                                     /*block_size=*/128);
  auto result = run_mate_case(q2,
                              values,
                              std::get<0>(mapped),
                              /*topk_length=*/8,
                              std::get<1>(mapped),
                              /*extra_topk_length=*/128);
  stream.synchronize();
  expect_close(result, "q2_swa_boundary_minus1_2_3");
}

}  // namespace
}  // namespace xllm::kernel::musa
