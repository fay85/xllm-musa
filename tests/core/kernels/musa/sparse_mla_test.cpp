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

#include "core/kernels/musa/sparse_mla.h"

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/layers/musa/glm5_next_sparse_attention.h"

namespace xllm::kernel::musa {
namespace {
// Scalar reference deliberately avoids the process's mixed MKL LP64/ILP64
// link environment. Expand K/V explicitly for the absorbed-projection case.
torch::Tensor reference(const torch::Tensor& query,
                        const torch::Tensor& cache,
                        const std::vector<std::vector<int64_t>>& selected,
                        const torch::Tensor& wk = {},
                        const torch::Tensor& wv = {}) {
  auto q = query.to(torch::kFloat32).contiguous();
  auto c = cache.to(torch::kFloat32).contiguous();
  auto k_weight =
      wk.defined() ? wk.to(torch::kFloat32).contiguous() : torch::Tensor();
  auto v_weight =
      wv.defined() ? wv.to(torch::kFloat32).contiguous() : torch::Tensor();
  const int64_t dim = q.size(2);
  const int64_t heads = q.size(1);
  auto result = torch::zeros({q.size(0), heads, dim}, torch::kFloat32);
  const float* qp = q.data_ptr<float>();
  const float* cp = c.data_ptr<float>();
  float* out = result.data_ptr<float>();
  for (int64_t row = 0; row < q.size(0); ++row) {
    const auto& ids = selected[row];
    if (ids.empty()) {
      continue;
    }
    for (int64_t head = 0; head < heads; ++head) {
      std::vector<double> logits(ids.size(), 0);
      std::vector<double> values(ids.size() * dim, 0);
      for (int64_t j = 0; j < static_cast<int64_t>(ids.size()); ++j) {
        for (int64_t d = 0; d < dim; ++d) {
          double key = cp[ids[j] * 512 + d];
          double value = key;
          if (wk.defined()) {
            key = 0;
            value = 0;
            for (int64_t latent = 0; latent < 512; ++latent) {
              const int64_t offset = (head * 512 + latent) * dim + d;
              key += cp[ids[j] * 512 + latent] *
                     k_weight.data_ptr<float>()[offset];
              value += cp[ids[j] * 512 + latent] *
                       v_weight.data_ptr<float>()[offset];
            }
          }
          logits[j] += qp[(row * heads + head) * dim + d] * key * 0.0625;
          values[j * dim + d] = value;
        }
      }
      const double maximum = *std::max_element(logits.begin(), logits.end());
      double denominator = 0;
      for (double& logit : logits) {
        logit = std::exp(logit - maximum);
        denominator += logit;
      }
      for (int64_t d = 0; d < dim; ++d) {
        double value = 0;
        for (int64_t j = 0; j < static_cast<int64_t>(ids.size()); ++j) {
          value += logits[j] * values[j * dim + d] / denominator;
        }
        out[(row * heads + head) * dim + d] = static_cast<float>(value);
      }
    }
  }
  return result;
}

TEST(SparseMlaTest, IndependentCandidatesHolesAndEmptyRows) {
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  torch::manual_seed(53);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  auto query = torch::randn({4, 64, 512}, options) * 0.1;
  auto cache = torch::randn({17, 1, 512}, options) * 0.1;
  auto indices = torch::full({4, 1, 2176}, -1, torch::kInt32);
  indices[0][0][0] = 3;
  indices[0][0][2050] = 9;
  indices[1][0][0] = 7;
  indices[1][0][1] = 12;
  indices[1][0][2] = 1000;
  indices[2][0][0] = 2;
  indices[2][0][2050] = 8;  // Outside this row's prefix.
  auto lengths = torch::tensor({2051, 3, 1, 2176}, torch::kInt32);
  auto expected =
      reference(query, cache.view({17, 512}), {{3, 9}, {7, 12}, {2}, {}});
  for (int64_t heads : {8, 16, 32, 64}) {
    for (int64_t repeat = 0; repeat < 2; ++repeat) {
      auto result =
          sparse_mla_bf16(query.narrow(1, 0, heads).contiguous().to(device),
                          cache.to(device),
                          indices.to(device),
                          lengths.to(device),
                          0.0625);
      ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
      EXPECT_TRUE(torch::allclose(result.cpu().to(torch::kFloat32),
                                  expected.narrow(1, 0, heads),
                                  /*rtol=*/0.02,
                                  /*atol=*/0.01));
      EXPECT_EQ(result[3].cpu().abs().sum().item<float>(), 0);
    }
  }
}

TEST(SparseMlaTest, PagedNopeAbsorptionMatchesExplicitKeysAndValues) {
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  torch::manual_seed(53);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16);
  auto cache = torch::randn({4, 4, 1, 512}, opts) * 0.1;
  auto q = torch::randn({3, 8, 256}, opts) * 0.1;
  auto latent = torch::randn({3, 512}, opts) * 0.1;
  auto wk = torch::randn({8, 512, 256}, opts) * 0.01;
  auto wv = torch::randn({8, 512, 256}, opts) * 0.01;
  auto ids = torch::full({3, 64}, -1, torch::kInt32);
  ids[0][0] = 0;
  ids[0][1] = 4;
  ids[0][2] = 5;  // Future position is removed.
  ids[1][0] = 1;
  ids[1][1] = 4;
  ids[2][0] = 0;  // Padded row is removed.
  auto rows = torch::tensor({0, 1, -1}, torch::kInt32);
  auto positions = torch::tensor({4, 4, -1}, torch::kInt64);
  auto table = torch::tensor({{2, 0}, {1, 3}}, torch::kInt32);
  auto slots = torch::tensor({0, 12, -1}, torch::kInt64);
  auto lengths = torch::tensor({3, 2, 1}, torch::kInt32);
  cache[0][0][0].copy_(latent[0]);
  cache[3][0][0].copy_(latent[1]);
  auto expected =
      reference(q, cache.view({16, 512}), {{8, 0}, {5, 12}, {}}, wk, wv);
  auto device_cache = cache.to(device);
  // Cache writer must install the current latent, and ignore the padded row.
  device_cache[0][0].zero_();
  device_cache[3][0].zero_();
  auto result = layer::glm5_next_sparse_attention(q.to(device),
                                                  latent.to(device),
                                                  wk.to(device),
                                                  wv.to(device),
                                                  ids.to(device),
                                                  lengths.to(device),
                                                  rows.to(device),
                                                  positions.to(device),
                                                  table.to(device),
                                                  slots.to(device),
                                                  device_cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(result.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/0.03,
                              /*atol=*/0.001));
  EXPECT_TRUE(torch::equal(device_cache[2].cpu(), cache[2]));
  auto fp8_cache =
      torch::empty({4, 4, 1, 528},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  auto all_slots = torch::arange(
      16, torch::TensorOptions().dtype(torch::kInt64).device(device));
  sparse_mla_fp8_cache_write(
      cache.view({16, 512}).to(device), all_slots, fp8_cache);
  auto fp8_result = layer::glm5_next_sparse_attention(q.to(device),
                                                      latent.to(device),
                                                      wk.to(device),
                                                      wv.to(device),
                                                      ids.to(device),
                                                      lengths.to(device),
                                                      rows.to(device),
                                                      positions.to(device),
                                                      table.to(device),
                                                      slots.to(device),
                                                      fp8_cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto records = fp8_cache.cpu().view({16, 528});
  auto quantized = records.narrow(1, 0, 512)
                       .contiguous()
                       .view(torch::kFloat8_e4m3fn)
                       .to(torch::kFloat32);
  auto scales = records.narrow(1, 512, 16).contiguous().view(torch::kFloat32);
  auto decoded =
      (quantized.view({16, 4, 128}) * scales.unsqueeze(-1)).view({16, 512});
  auto fp8_expected = reference(q, decoded, {{8, 0}, {5, 12}, {}}, wk, wv);
  EXPECT_TRUE(torch::allclose(fp8_result.cpu().to(torch::kFloat32),
                              fp8_expected,
                              /*rtol=*/0.03,
                              /*atol=*/0.001));
}

TEST(SparseMlaTest, InlineFp8CacheDecodeMatchesDequantizedReference) {
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  torch::manual_seed(53);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto stream = c10::musa::getStreamFromPool(false, 0);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto latent = (torch::randn({16, 512}, torch::kBFloat16) * 0.1).to(device);
  auto cache =
      torch::zeros({2, 8, 1, 528},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  auto slots = torch::arange(
      16, torch::TensorOptions().dtype(torch::kInt64).device(device));
  sparse_mla_fp8_cache_write(latent, slots, cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto bytes = cache.cpu().view({16, 528});
  auto values = bytes.narrow(1, 0, 512)
                    .contiguous()
                    .view(torch::kFloat8_e4m3fn)
                    .to(torch::kFloat32);
  auto scales = bytes.narrow(1, 512, 16).contiguous().view(torch::kFloat32);
  auto dequant =
      (values.view({16, 4, 128}) * scales.unsqueeze(-1)).view({16, 512});
  auto groups = latent.cpu().to(torch::kFloat32).view({16, 4, 128});
  auto expected_scales = (groups.abs().amax(-1) / 448.0).clamp_min(1.0e-4);
  // Test rounding against the actual stored scale; scale selection is checked
  // independently below, so its final FP32 bit does not perturb tie cases.
  auto expected_values = (groups / scales.unsqueeze(-1))
                             .to(torch::kFloat8_e4m3fn)
                             .to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(scales, expected_scales, 1.0e-6, 1.0e-8));
  const auto difference = values.view({16, 4, 128}) - expected_values;
  EXPECT_TRUE(torch::equal(values.view({16, 4, 128}), expected_values))
      << "max_quant_difference=" << difference.abs().max().item<float>()
      << "different_values=" << difference.ne(0).sum().item<int64_t>();
  auto query = torch::randn({2, 1, 64, 512}, torch::kBFloat16) * 0.1;
  auto ids = torch::full({2, 1, 1, 2176}, -1, torch::kInt32);
  ids[0][0][0][0] = 1;
  ids[0][0][0][2050] = 9;
  auto lengths = torch::tensor({2051, 2176}, torch::kInt32);
  auto expected = reference(query.squeeze(1), dequant, {{1, 9}, {}})[0];
  for (int64_t heads : {8, 16, 32, 64}) {
    for (int64_t repeat = 0; repeat < 2; ++repeat) {
      auto out =
          sparse_mla_nope_fp8(query.narrow(2, 0, heads).contiguous().to(device),
                              cache,
                              ids.to(device),
                              lengths.to(device),
                              0.0625);
      ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
      EXPECT_TRUE(torch::allclose(out[0][0].cpu().to(torch::kFloat32),
                                  expected.narrow(0, 0, heads),
                                  0.02,
                                  0.01));
      EXPECT_EQ(out[1].cpu().abs().sum().item<float>(), 0);
    }
  }
}
}  // namespace
}  // namespace xllm::kernel::musa
