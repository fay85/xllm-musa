/* Copyright 2025-2026 The xLLM Authors.

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
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangChunkGatedDeltaRuleFwdHTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct ChunkGdrTestCase {
  std::string name;
  int64_t batch_size;
  int64_t seq_len;
  int64_t num_heads;
  int64_t num_key_heads;
  int64_t key_dim;
  int64_t value_dim;
  int64_t chunk_size;
  int64_t seed;
};

torch::Tensor make_chunk_local_cumsum(int64_t batch_size,
                                      int64_t seq_len,
                                      int64_t num_heads,
                                      int64_t chunk_size,
                                      int64_t seed) {
  torch::manual_seed(seed);
  constexpr float kGateScale = 0.002F;
  const int64_t chunks_per_sequence = (seq_len + chunk_size - 1) / chunk_size;
  const torch::Tensor gates =
      torch::randn({batch_size, chunks_per_sequence, chunk_size, num_heads},
                   torch::TensorOptions().dtype(torch::kFloat32)) *
      kGateScale;
  const auto chunk_local_gates = gates.cumsum(/*dim=*/2);
  std::vector<torch::Tensor> sequences;
  sequences.reserve(batch_size);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    sequences.push_back(
        chunk_local_gates[batch]
            .reshape({chunks_per_sequence * chunk_size, num_heads})
            .slice(/*dim=*/0, /*start=*/0, /*end=*/seq_len));
  }
  return torch::stack(sequences).reshape({batch_size * seq_len, num_heads});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ref_chunk_gated_delta_rule(const torch::Tensor& k,
                           const torch::Tensor& w,
                           const torch::Tensor& u,
                           const torch::Tensor& g,
                           const torch::Tensor& initial_state,
                           const torch::Tensor& cu_seqlens,
                           int64_t chunk_size) {
  const int64_t batch_size = initial_state.size(0);
  const int64_t num_heads = u.size(1);
  const int64_t num_key_heads = k.size(1);
  const int64_t key_dim = k.size(2);
  const int64_t value_dim = u.size(2);
  const int64_t hg_ratio = num_heads / num_key_heads;

  auto k_f = k.to(torch::kFloat32);
  auto w_f = w.to(torch::kFloat32);
  auto u_f = u.to(torch::kFloat32);
  auto g_f = g.to(torch::kFloat32);
  auto h0_f = initial_state.to(torch::kFloat32);

  auto cu_cpu = cu_seqlens.cpu().contiguous();
  auto cu_ptr = cu_cpu.data_ptr<int32_t>();

  int64_t nt_max = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t seq_len = cu_ptr[i + 1] - cu_ptr[i];
    int64_t nt_i = (seq_len + chunk_size - 1) / chunk_size;
    nt_max = std::max(nt_max, nt_i);
  }

  auto h_out = torch::empty(
      {1, batch_size * nt_max, num_heads, key_dim, value_dim}, k.options());
  auto v_new = torch::empty_like(u);
  auto ht = torch::empty({batch_size, num_heads, key_dim, value_dim},
                         torch::kFloat32);

  for (int64_t n = 0; n < batch_size; ++n) {
    int64_t bos = static_cast<int64_t>(cu_ptr[n]);
    int64_t eos = static_cast<int64_t>(cu_ptr[n + 1]);
    for (int64_t head = 0; head < num_heads; ++head) {
      auto state = h0_f[n][head].clone();
      int64_t k_head_idx = head / hg_ratio;

      int64_t chunk_id = 0;
      for (int64_t start = bos; start < eos; start += chunk_size) {
        int64_t end = std::min(start + chunk_size, eos);

        // h[t] ← state (before this chunk's update)
        h_out[0][n * nt_max + chunk_id][head].copy_(state.to(k.dtype()));

        // residual = u - w @ h
        auto u_chunk =
            u_f.slice(0, start, end).select(1, head);  // [L, value_dim]
        auto w_chunk =
            w_f.slice(0, start, end).select(1, head);  // [L, key_dim]
        auto residual =
            u_chunk - torch::matmul(w_chunk, state);  // [L, value_dim]

        // Gating: state_v = residual * exp(where(g_diff<=0, g_diff, -inf))
        //          state   = state   * exp(last_g)
        auto g_chunk = g_f.slice(0, start, end).select(1, head);  // [L]
        auto last_g = g_chunk[-1].clone();
        auto g_diff = last_g - g_chunk;
        auto g_diff_masked = torch::where(
            g_diff <= 0.0, g_diff, torch::full_like(g_diff, -INFINITY));
        auto state_v = residual * g_diff_masked.exp().unsqueeze(/*dim=*/-1);
        state = state * std::exp(last_g.item<float>());

        // v_new[t] ← residual (before gating)
        v_new.slice(0, start, end)
            .select(1, head)
            .copy_(residual.to(k.dtype()));

        // h += k^seq_len @ v_new
        auto k_chunk =
            k_f.slice(0, start, end).select(1, k_head_idx);  // [L, key_dim]
        state = state + torch::matmul(k_chunk.t(), state_v);

        ++chunk_id;
      }
      ht[n][head].copy_(state);
    }
  }

  return {h_out, v_new, ht};
}

void run_chunk_gated_delta_rule_fwd_h_accuracy_case(
    const ChunkGdrTestCase& test_case) {
  ASSERT_GT(test_case.batch_size, 0);
  ASSERT_GT(test_case.seq_len, 0);
  ASSERT_GT(test_case.num_heads, 0);
  ASSERT_GT(test_case.num_key_heads, 0);
  ASSERT_GE(test_case.num_heads, test_case.num_key_heads);
  ASSERT_GT(test_case.key_dim, 0);
  ASSERT_GT(test_case.value_dim, 0);
  ASSERT_EQ(test_case.chunk_size, 64);

  const auto npu_device = torch::Device("npu:0");
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);
  const auto f32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(npu_device);

  torch::manual_seed(test_case.seed);

  const int64_t batch_size = test_case.batch_size;
  const int64_t seq_len = test_case.seq_len;
  const int64_t num_heads = test_case.num_heads;
  const int64_t num_key_heads = test_case.num_key_heads;
  const int64_t key_dim = test_case.key_dim;
  const int64_t value_dim = test_case.value_dim;

  auto k_npu =
      torch::randn({batch_size * seq_len, num_key_heads, key_dim}, bf16_opts) *
      0.01F;
  auto w_npu =
      torch::randn({batch_size * seq_len, num_heads, key_dim}, bf16_opts) *
      0.01F;
  auto u_npu =
      torch::randn({batch_size * seq_len, num_heads, value_dim}, bf16_opts) *
      0.01F;
  auto h0_npu =
      torch::randn({batch_size, num_heads, key_dim, value_dim}, f32_opts) *
      0.01F;

  auto g_cpu = make_chunk_local_cumsum(
      batch_size, seq_len, num_heads, /*chunk_size=*/64, test_case.seed);
  auto g_npu = g_cpu.to(npu_device);

  auto k_cpu = k_npu.cpu();
  auto w_cpu = w_npu.cpu();
  auto u_cpu = u_npu.cpu();
  auto h0_cpu = h0_npu.cpu();

  std::vector<int32_t> cu_seqlens_vec(batch_size + 1);
  for (int64_t i = 0; i <= batch_size; ++i) {
    cu_seqlens_vec[i] = static_cast<int32_t>(i * seq_len);
  }
  auto cu_cpu =
      torch::from_blob(
          cu_seqlens_vec.data(), {batch_size + 1}, torch::dtype(torch::kInt32))
          .clone();
  auto cu_npu = cu_cpu.to(npu_device);

  auto [h_ref, v_new_ref, ht_ref] = ref_chunk_gated_delta_rule(
      k_cpu, w_cpu, u_cpu, g_cpu, h0_cpu, cu_cpu, /*chunk_size=*/64);

  auto [h_npu, v_new_npu, ht_npu] =
      chunk_gated_delta_rule_fwd_h(k_npu,
                                   w_npu,
                                   u_npu,
                                   g_npu,
                                   h0_npu,
                                   /*output_final_state=*/true,
                                   /*chunk_size=*/64,
                                   /*save_new_value=*/true,
                                   cu_npu,
                                   /*chunk_offsets=*/std::nullopt);

  constexpr float kRtol = 5e-2F;
  // Inputs are scaled to 0.01; an absolute tolerance of 0.5 accepts zero
  // outputs. Keep BF16 rounding slack without masking a missing computation.
  constexpr float kAtol = 5e-4F;

  ASSERT_EQ(h_npu.sizes(), h_ref.sizes())
      << "h_out shape mismatch for case " << test_case.name;
  ASSERT_EQ(v_new_npu.sizes(), v_new_ref.sizes())
      << "v_new shape mismatch for case " << test_case.name;
  ASSERT_EQ(ht_npu.sizes(), ht_ref.sizes())
      << "ht shape mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(h_npu.cpu(), h_ref, kRtol, kAtol))
      << "h_out mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(v_new_npu.cpu(), v_new_ref, kRtol, kAtol))
      << "v_new mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(ht_npu.cpu(), ht_ref, kRtol, kAtol))
      << "ht mismatch for case " << test_case.name;
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHTest,
       ChunkGatedDeltaRuleFwdHMatchesReference) {
  const std::vector<ChunkGdrTestCase> accuracy_cases = {
      // t65 is the primary regression; t32 and t127 cover different tails.
      {.name = "accuracy_partial_tail_n1_t32_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 32,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260518},
      {.name = "accuracy_partial_tail_n2_t65_h16_hg8_k128_v128",
       .batch_size = 2,
       .seq_len = 65,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260517},
      {.name = "accuracy_partial_tail_n1_t127_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 127,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260519},
      // Tails in chunks 3 and 4 reuse a full even/odd L1 tile, respectively.
      {.name = "accuracy_reused_tail_n1_t129_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 129,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260520},
      {.name = "accuracy_reused_tail_n1_t193_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 193,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260521},
      {.name = "accuracy_n1_t64_h16_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 64,
       .num_heads = 16,
       .num_key_heads = 16,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260513},
      {.name = "accuracy_tp2_n1_t128_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .num_heads = 16,
       .num_key_heads = 8,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260516},
      {.name = "accuracy_n2_t128_h16_hg16_k128_v128",
       .batch_size = 2,
       .seq_len = 128,
       .num_heads = 16,
       .num_key_heads = 16,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260512},
      {.name = "accuracy_n1_t128_h32_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .num_heads = 32,
       .num_key_heads = 16,
       .key_dim = 128,
       .value_dim = 128,
       .chunk_size = 64,
       .seed = 20260514},
  };

  for (const auto& test_case : accuracy_cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_accuracy_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
