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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kernels/musa/gdn_ops.h"
#include "core/layers/musa/pcp_context.h"

namespace at {
struct MUSAHooksInterface;
namespace detail {
const MUSAHooksInterface& getMUSAHooks();
}  // namespace detail
}  // namespace at

namespace xllm::layer::musa_pcp {
namespace {

void ensure_musa_backend() {
  if (!c10::is_privateuse1_backend_registered()) {
    c10::register_privateuse1_backend("musa");
  }
  (void)at::detail::getMUSAHooks();
}

class CompletedWork final : public c10d::Work {
 public:
  explicit CompletedWork(c10d::OpType op) : Work(/*rank=*/0, op) { finish(); }
};

struct SharedGatherState final {
  torch::Tensor rank0_send;
  torch::Tensor rank1_send;
};

class LoopbackCpGroup final : public ProcessGroup {
 public:
  LoopbackCpGroup(int32_t rank, std::shared_ptr<SharedGatherState> shared)
      : ProcessGroup(rank, /*world_size=*/2, torch::Device(torch::kCPU)),
        shared_(std::move(shared)) {}

  c10::intrusive_ptr<c10d::Work> broadcast_async(torch::Tensor& input,
                                                 int32_t root_rank) override {
    broadcast_shapes.push_back(input.sizes().vec());
    broadcast_roots.push_back(root_rank);
    if (rank() != root_rank) {
      input.fill_(1);
    }
    return c10::make_intrusive<CompletedWork>(c10d::OpType::BROADCAST);
  }

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    CHECK(shared_ != nullptr);
    if (rank() == 0) {
      shared_->rank0_send = input.clone();
    } else {
      shared_->rank1_send = input.clone();
    }
    CHECK(shared_->rank0_send.defined() && shared_->rank1_send.defined());
    return torch::stack({shared_->rank0_send, shared_->rank1_send}, /*dim=*/0);
  }

  std::vector<std::vector<int64_t>> broadcast_shapes;
  std::vector<int32_t> broadcast_roots;

 private:
  std::shared_ptr<SharedGatherState> shared_;
};

AttentionMetadata make_global_metadata(const std::vector<int32_t>& q_lens,
                                       const std::vector<int32_t>& kv_lens,
                                       const std::vector<bool>& init_states) {
  CHECK_EQ(q_lens.size(), kv_lens.size());
  CHECK_EQ(q_lens.size(), init_states.size());
  const size_t b = q_lens.size();
  std::vector<int32_t> q_cu(b + 1, 0);
  std::vector<int32_t> kv_cu(b + 1, 0);
  bool any_hist = false;
  int32_t max_q = 0;
  int32_t max_kv = 0;
  for (size_t i = 0; i < b; ++i) {
    q_cu[i + 1] = q_cu[i] + q_lens[i];
    kv_cu[i + 1] = kv_cu[i] + kv_lens[i];
    any_hist = any_hist || (kv_lens[i] > q_lens[i]);
    max_q = std::max(max_q, q_lens[i]);
    max_kv = std::max(max_kv, kv_lens[i]);
  }
  const int32_t total_q = q_cu.back();

  AttentionMetadata meta;
  meta.q_seq_lens_vec = q_lens;
  meta.kv_seq_lens_vec = kv_lens;
  meta.q_cu_seq_lens =
      torch::tensor(q_cu, torch::TensorOptions().dtype(torch::kInt32));
  meta.kv_cu_seq_lens =
      torch::tensor(kv_cu, torch::TensorOptions().dtype(torch::kInt32));
  meta.q_seq_lens =
      torch::tensor(q_lens, torch::TensorOptions().dtype(torch::kInt32));
  meta.kv_seq_lens =
      torch::tensor(kv_lens, torch::TensorOptions().dtype(torch::kInt32));
  meta.slot_mapping = torch::arange(
      1000, 1000 + total_q, torch::TensorOptions().dtype(torch::kInt32));
  meta.block_table = torch::zeros(
      {static_cast<int64_t>(b), 32},
      torch::TensorOptions().dtype(torch::kInt32));
  std::vector<uint8_t> init_u8(b, 0);
  for (size_t i = 0; i < b; ++i) {
    init_u8[i] = init_states[i] ? 1 : 0;
  }
  meta.has_initial_states =
      torch::tensor(init_u8, torch::TensorOptions().dtype(torch::kBool));
  meta.max_query_len = max_q;
  meta.max_seq_len = max_kv;
  meta.total_kv_len = kv_cu.back();
  meta.is_prefill = !any_hist;
  meta.is_chunked_prefill = any_hist;
  return meta;
}

TEST(MusaPcpMultiBatchTest, BuildsVariableLengthColdAndWarmContext) {
  auto shared = std::make_shared<SharedGatherState>();
  LoopbackCpGroup group0(/*rank=*/0, shared);
  LoopbackCpGroup group1(/*rank=*/1, shared);
  ParallelArgs args0(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  ParallelArgs args1(/*rank=*/1, /*world_size=*/2, /*process_group=*/nullptr);
  args0.cp_group_ = &group0;
  args1.cp_group_ = &group1;

  // 3 sequences: cold 256, warm 320 (history 512), warm non-64-multiple 191
  // (history 128).
  const std::vector<int32_t> q_lens = {256, 320, 191};
  const std::vector<int32_t> kv_lens = {256, 832, 319};
  const std::vector<bool> init_states = {false, true, true};
  const int32_t total_rows = 256 + 320 + 191;
  AttentionMetadata global_meta =
      make_global_metadata(q_lens, kv_lens, init_states);

  std::optional<Context> ctx0;
  std::optional<Context> ctx1;
  build_context(global_meta, args0, total_rows, &ctx0);
  build_context(global_meta, args1, total_rows, &ctx1);

  ASSERT_TRUE(ctx0.has_value());
  ASSERT_TRUE(ctx1.has_value());
  EXPECT_TRUE(ctx0->is_batch);
  EXPECT_TRUE(ctx1->is_batch);
  EXPECT_EQ(ctx0->batch_size, 3);
  EXPECT_EQ(ctx1->batch_size, 3);
  EXPECT_EQ(ctx0->row_offsets, ctx1->row_offsets);
  EXPECT_EQ(ctx0->row_offsets.back(), total_rows);

  const auto& f = ctx0->local_metadata.q_seq_lens_vec;
  const auto& r = ctx1->local_metadata.q_seq_lens_vec;
  ASSERT_EQ(f.size(), 3u);
  ASSERT_EQ(r.size(), 3u);
  for (size_t b = 0; b < 3; ++b) {
    EXPECT_EQ(f[b] % 64, 0) << "front split must align to 64-token GDN chunk";
    EXPECT_GE(f[b], 64);
    EXPECT_GE(r[b], 32);
    EXPECT_EQ(f[b] + r[b], q_lens[b]);
    EXPECT_EQ(ctx0->local_metadata.kv_seq_lens_vec[b],
              (kv_lens[b] - q_lens[b]) + f[b]);
    EXPECT_EQ(ctx1->local_metadata.kv_seq_lens_vec[b], kv_lens[b]);
  }
  EXPECT_TRUE(ctx0->local_metadata.is_chunked_prefill);
  EXPECT_TRUE(ctx1->local_metadata.is_chunked_prefill);
  EXPECT_TRUE(torch::equal(ctx0->local_metadata.has_initial_states,
                           global_meta.has_initial_states));
  EXPECT_TRUE(torch::equal(
      ctx1->local_metadata.has_initial_states,
      torch::tensor({true, true, true},
                    torch::TensorOptions().dtype(torch::kBool))));
}

TEST(MusaPcpMultiBatchTest, ShardAndGatherRoundTripRestoresSequenceMajorOrder) {
  auto shared = std::make_shared<SharedGatherState>();
  LoopbackCpGroup group0(/*rank=*/0, shared);
  LoopbackCpGroup group1(/*rank=*/1, shared);
  ParallelArgs args0(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  ParallelArgs args1(/*rank=*/1, /*world_size=*/2, /*process_group=*/nullptr);
  args0.cp_group_ = &group0;
  args1.cp_group_ = &group1;

  const std::vector<int32_t> q_lens = {192, 320, 145};
  const std::vector<int32_t> kv_lens = {192, 320, 145};
  const std::vector<bool> init_states = {false, false, false};
  const int32_t total_rows = 192 + 320 + 145;
  AttentionMetadata global_meta =
      make_global_metadata(q_lens, kv_lens, init_states);

  std::optional<Context> ctx0;
  std::optional<Context> ctx1;
  build_context(global_meta, args0, total_rows, &ctx0);
  build_context(global_meta, args1, total_rows, &ctx1);
  ASSERT_TRUE(ctx0.has_value());
  ASSERT_TRUE(ctx1.has_value());

  // 2D hidden states [T, 8]
  const torch::Tensor global_hidden =
      torch::arange(total_rows * 8,
                    torch::TensorOptions().dtype(torch::kFloat32))
          .view({total_rows, 8});
  const torch::Tensor local0 = shard_rows(global_hidden, *ctx0, /*dim=*/0);
  const torch::Tensor local1 = shard_rows(global_hidden, *ctx1, /*dim=*/0);

  // Populate shared allgather state from rank 0 then rank 1.
  shared->rank1_send =
      torch::zeros({ctx0->max_rows, 8}, global_hidden.options());
  (void)gather_rows(local0, *ctx0);
  const torch::Tensor gathered1 = gather_rows(local1, *ctx1);
  const torch::Tensor gathered0 = gather_rows(local0, *ctx0);
  EXPECT_TRUE(torch::equal(gathered0, global_hidden));
  EXPECT_TRUE(torch::equal(gathered1, global_hidden));

  // mRoPE [3, T] sharding along dim 1
  const torch::Tensor mrope_pos =
      torch::arange(3 * total_rows, torch::TensorOptions().dtype(torch::kInt32))
          .view({3, total_rows});
  const torch::Tensor mrope0 = shard_rows(mrope_pos, *ctx0, /*dim=*/1);
  const torch::Tensor mrope1 = shard_rows(mrope_pos, *ctx1, /*dim=*/1);
  EXPECT_EQ(mrope0.size(0), 3);
  EXPECT_EQ(mrope1.size(0), 3);
  EXPECT_EQ(mrope0.size(1) + mrope1.size(1), total_rows);

  // Slot mapping union covers exact original slots via unpermute_indices.
  const torch::Tensor cat_slots =
      torch::cat({ctx0->local_metadata.slot_mapping,
                  ctx1->local_metadata.slot_mapping},
                 /*dim=*/0);
  const torch::Tensor restored_slots =
      cat_slots.index_select(/*dim=*/0, ctx0->unpermute_indices);
  EXPECT_TRUE(torch::equal(restored_slots, global_meta.slot_mapping));
}

TEST(MusaPcpMultiBatchTest, ExtractsAndReceivesMultiBatchConvAndSsmPrefixes) {
  auto shared = std::make_shared<SharedGatherState>();
  LoopbackCpGroup group0(/*rank=*/0, shared);
  LoopbackCpGroup group1(/*rank=*/1, shared);
  ParallelArgs args0(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  ParallelArgs args1(/*rank=*/1, /*world_size=*/2, /*process_group=*/nullptr);
  args0.cp_group_ = &group0;
  args1.cp_group_ = &group1;

  const std::vector<int32_t> q_lens = {256, 192};
  const int32_t total_rows = 256 + 192;
  AttentionMetadata global_meta =
      make_global_metadata(q_lens, q_lens, {false, false});

  std::optional<Context> ctx0;
  std::optional<Context> ctx1;
  build_context(global_meta, args0, total_rows, &ctx0);
  build_context(global_meta, args1, total_rows, &ctx1);
  ASSERT_TRUE(ctx0.has_value());
  ASSERT_TRUE(ctx1.has_value());

  const int32_t f0 = ctx0->local_metadata.q_seq_lens_vec[0];
  const int32_t f1 = ctx0->local_metadata.q_seq_lens_vec[1];
  const int64_t qkv_dim = 6;
  const int64_t qkvz_dim = 10;
  const torch::Tensor local_qkvz =
      torch::arange((f0 + f1) * qkvz_dim,
                    torch::TensorOptions().dtype(torch::kBFloat16))
          .view({f0 + f1, qkvz_dim});

  const torch::Tensor prefix =
      extract_conv_prefix(local_qkvz, /*conv_state_len=*/3, qkv_dim, *ctx0);
  EXPECT_EQ(prefix.sizes().vec(), std::vector<int64_t>({2, 3, qkv_dim}));
  EXPECT_TRUE(torch::equal(
      prefix[0], local_qkvz.narrow(0, f0 - 3, 3).narrow(1, 0, qkv_dim)));
  EXPECT_TRUE(torch::equal(
      prefix[1], local_qkvz.narrow(0, f0 + f1 - 3, 3).narrow(1, 0, qkv_dim)));

  const torch::Tensor ssm_cache = torch::zeros(
      {16, 4, 8, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
  expect_conv_prefix(/*conv_state_len=*/3, qkv_dim, local_qkvz.options(), *ctx1);
  expect_prefix_cache_row(ssm_cache, *ctx1);
  ASSERT_EQ(group1.broadcast_shapes.size(), 2u);
  EXPECT_EQ(group1.broadcast_shapes[0], std::vector<int64_t>({2, 3, qkv_dim}));
  EXPECT_EQ(group1.broadcast_shapes[1], std::vector<int64_t>({2, 4, 8, 8}));
  (void)take_prefix_rows(*ctx1);
  (void)take_prefix_rows(*ctx1);
}

TEST(MusaPcpMultiBatchTest, RejectsSubMinimumSequenceCleanly) {
  auto shared = std::make_shared<SharedGatherState>();
  LoopbackCpGroup group0(/*rank=*/0, shared);
  ParallelArgs args0(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  args0.cp_group_ = &group0;

  const std::vector<int32_t> q_lens = {256, 64};
  AttentionMetadata global_meta =
      make_global_metadata(q_lens, q_lens, {false, false});
  std::optional<Context> ctx0;
  build_context(global_meta, args0, /*num_rows=*/320, &ctx0);
  EXPECT_FALSE(ctx0.has_value());
}

TEST(MusaPcpMultiBatchGpuTest, CausalConv1dAndMateGdnSplitMatchUnsplitBatch) {
  ensure_musa_backend();
  const torch::Device dev(c10::DeviceType::PrivateUse1, 0);
  torch::manual_seed(42);

  // 3 variable-length sequences: [192, 256, 150], split on 64-token boundaries
  // front = [128, 128, 64], back = [64, 128, 86].
  const std::vector<int32_t> q_full = {192, 256, 150};
  const std::vector<int32_t> f_lens = {128, 128, 64};
  const std::vector<int32_t> r_lens = {64, 128, 86};
  const int32_t total_full = 192 + 256 + 150;
  const int32_t total_f = 128 + 128 + 64;
  const int32_t total_r = 64 + 128 + 86;
  const int64_t dim = 256;
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  const auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(dev);
  const auto bool_opts =
      torch::TensorOptions().dtype(torch::kBool).device(dev);

  const torch::Tensor x_full = torch::randn({total_full, dim}, bf16_opts) * 0.2f;
  const torch::Tensor weight = torch::randn({dim, 4}, bf16_opts) * 0.2f;
  const torch::Tensor init_conv = torch::randn({8, dim, 3}, bf16_opts) * 0.2f;
  torch::Tensor conv_ref = init_conv.clone();
  torch::Tensor conv_r0 = init_conv.clone();
  torch::Tensor conv_r1 = init_conv.clone();

  const torch::Tensor cu_full =
      torch::tensor({0, 192, 448, 598}, i32_opts);
  const torch::Tensor cu_f =
      torch::tensor({0, 128, 256, 320}, i32_opts);
  const torch::Tensor cu_r =
      torch::tensor({0, 64, 192, 278}, i32_opts);
  const torch::Tensor cache_idx = torch::tensor({1, 3, 5}, i32_opts);
  const torch::Tensor has_init_r0 =
      torch::tensor({false, true, false}, bool_opts);
  const torch::Tensor has_init_r1 =
      torch::tensor({true, true, true}, bool_opts);

  const torch::Tensor out_ref = xllm::kernel::musa::causal_conv1d_prefill(
      x_full,
      weight,
      conv_ref,
      std::nullopt,
      cu_full,
      cache_idx,
      has_init_r0,
      /*silu_activation=*/true,
      std::nullopt,
      cache_idx);

  const torch::Tensor x_f = torch::cat(
      {x_full.narrow(0, 0, 128),
       x_full.narrow(0, 192, 128),
       x_full.narrow(0, 448, 64)},
      /*dim=*/0);
  const torch::Tensor x_r = torch::cat(
      {x_full.narrow(0, 128, 64),
       x_full.narrow(0, 320, 128),
       x_full.narrow(0, 512, 86)},
      /*dim=*/0);
  const torch::Tensor prefix_x = torch::stack(
      {x_f.narrow(0, 128 - 3, 3),
       x_f.narrow(0, 256 - 3, 3),
       x_f.narrow(0, 320 - 3, 3)},
      /*dim=*/0);

  const torch::Tensor out_f = xllm::kernel::musa::causal_conv1d_prefill(
      x_f,
      weight,
      conv_r0,
      std::nullopt,
      cu_f,
      cache_idx,
      has_init_r0,
      /*silu_activation=*/true,
      std::nullopt,
      cache_idx);
  const torch::Tensor out_r = xllm::kernel::musa::causal_conv1d_prefill(
      x_r,
      weight,
      conv_r1,
      std::nullopt,
      cu_r,
      cache_idx,
      has_init_r1,
      /*silu_activation=*/true,
      prefix_x,
      cache_idx);

  const torch::Tensor out_reconstructed = torch::cat(
      {out_f.narrow(0, 0, 128),
       out_r.narrow(0, 0, 64),
       out_f.narrow(0, 128, 128),
       out_r.narrow(0, 64, 128),
       out_f.narrow(0, 256, 64),
       out_r.narrow(0, 192, 86)},
      /*dim=*/0);

  EXPECT_TRUE(torch::allclose(out_reconstructed, out_ref, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(conv_r1.index_select(0, cache_idx),
                              conv_ref.index_select(0, cache_idx),
                              /*rtol=*/0.0,
                              /*atol=*/0.0));
  (void)total_f;
  (void)total_r;
  (void)f_lens;
  (void)r_lens;
}

TEST(MusaPcpMultiBatchGpuTest, MateGdnPackedMultiSeqSplitMatchesUnsplitBatch) {
  ensure_musa_backend();
  const torch::Device dev(c10::DeviceType::PrivateUse1, 0);
  torch::manual_seed(123);

  const int32_t total_full = 192 + 256 + 150;
  const int64_t nk = 16;
  const int64_t nv = 48;
  const int64_t dk = 128;
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  const auto f32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(dev);

  const torch::Tensor q_full =
      torch::nn::functional::normalize(
          torch::randn({1, total_full, nk, dk}, f32_opts),
          torch::nn::functional::NormalizeFuncOptions().dim(-1))
          .to(torch::kBFloat16);
  const torch::Tensor k_full =
      torch::nn::functional::normalize(
          torch::randn({1, total_full, nk, dk}, f32_opts),
          torch::nn::functional::NormalizeFuncOptions().dim(-1))
          .to(torch::kBFloat16);
  const torch::Tensor v_full =
      (torch::randn({1, total_full, nv, dk}, bf16_opts) * 0.1f);
  const torch::Tensor g_full =
      (-torch::rand({1, total_full, nv}, f32_opts) * 0.05f);
  const torch::Tensor beta_full =
      torch::sigmoid(torch::randn({1, total_full, nv}, bf16_opts));
  torch::Tensor init_state = torch::randn({3, nv, dk, dk}, bf16_opts) * 0.05f;
  init_state[0].zero_();  // seq 0 cold, seq 1 & 2 warm

  const double scale = 1.0 / std::sqrt(static_cast<double>(dk));

  xllm::kernel::musa::MateGatedDeltaRulePrefillParams ref_params;
  ref_params.q = q_full;
  ref_params.k = k_full;
  ref_params.v = v_full;
  ref_params.g = g_full;
  ref_params.beta = beta_full;
  ref_params.scale = scale;
  ref_params.initial_state = init_state;
  ref_params.cu_seqlens_host = std::vector<int32_t>{0, 192, 448, 598};
  auto [out_ref, state_ref] =
      xllm::kernel::musa::mate_gated_delta_rule_prefill(ref_params);

  auto slice_r0 = [](const torch::Tensor& t) {
    return torch::cat(
        {t.narrow(1, 0, 128), t.narrow(1, 192, 128), t.narrow(1, 448, 64)},
        /*dim=*/1);
  };
  auto slice_r1 = [](const torch::Tensor& t) {
    return torch::cat(
        {t.narrow(1, 128, 64), t.narrow(1, 320, 128), t.narrow(1, 512, 86)},
        /*dim=*/1);
  };

  xllm::kernel::musa::MateGatedDeltaRulePrefillParams r0_params;
  r0_params.q = slice_r0(q_full);
  r0_params.k = slice_r0(k_full);
  r0_params.v = slice_r0(v_full);
  r0_params.g = slice_r0(g_full);
  r0_params.beta = slice_r0(beta_full);
  r0_params.scale = scale;
  r0_params.initial_state = init_state;
  r0_params.cu_seqlens_host = std::vector<int32_t>{0, 128, 256, 320};
  auto [out_r0, state_r0] =
      xllm::kernel::musa::mate_gated_delta_rule_prefill(r0_params);

  xllm::kernel::musa::MateGatedDeltaRulePrefillParams r1_params;
  r1_params.q = slice_r1(q_full);
  r1_params.k = slice_r1(k_full);
  r1_params.v = slice_r1(v_full);
  r1_params.g = slice_r1(g_full);
  r1_params.beta = slice_r1(beta_full);
  r1_params.scale = scale;
  r1_params.initial_state = state_r0.to(torch::kBFloat16);
  r1_params.cu_seqlens_host = std::vector<int32_t>{0, 64, 192, 278};
  auto [out_r1, state_r1] =
      xllm::kernel::musa::mate_gated_delta_rule_prefill(r1_params);

  const torch::Tensor out_reconstructed = torch::cat(
      {out_r0.narrow(1, 0, 128),
       out_r1.narrow(1, 0, 64),
       out_r0.narrow(1, 128, 128),
       out_r1.narrow(1, 64, 128),
       out_r0.narrow(1, 256, 64),
       out_r1.narrow(1, 192, 86)},
      /*dim=*/1);

  const double max_out_diff =
      (out_reconstructed.to(torch::kFloat32) - out_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<double>();
  const double max_state_diff =
      (state_r1.to(torch::kFloat32) - state_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<double>();
  EXPECT_LT(max_out_diff, 2e-2);
  EXPECT_LT(max_state_diff, 2e-2);
}

}  // namespace
}  // namespace xllm::layer::musa_pcp
