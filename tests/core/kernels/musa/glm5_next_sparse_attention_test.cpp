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

#include "core/layers/musa/glm5_next_sparse_attention.h"

#include <c10/core/StreamGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/layers/musa/glm5_next/glm5_next_decoder_layer.h"

namespace xllm::layer {
namespace {
using Weights = std::unordered_map<std::string, torch::Tensor>;
using Selection = kernel::musa::GlmKPoolSelection;

class GlmSparseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    at::detail::getMUSAHooks().initMUSA();
    torch::manual_seed(53);
    stream_ = c10::musa::getStreamFromPool(false, 0);
    guard_ = std::make_unique<c10::StreamGuard>(stream_.unwrap());
  }
  torch::NoGradGuard no_grad_;
  torch::Device device_{torch::kPrivateUse1, 0};
  c10::musa::MUSAStream stream_ = c10::musa::getDefaultMUSAStream();
  std::unique_ptr<c10::StreamGuard> guard_;
};

// CPU reference uses scalar Hadamard/scoring and never calls CPU BLAS.
std::array<float, 128> rotate(std::array<float, 128> input) {
  for (int32_t stride = 1; stride < 128; stride *= 2) {
    const auto previous = input;
    for (int32_t lane = 0; lane < 128; ++lane) {
      input[lane] = (lane & stride) ? previous[lane ^ stride] - previous[lane]
                                    : previous[lane] + previous[lane ^ stride];
    }
  }
  for (float& value : input) {
    value = static_cast<float>(c10::BFloat16(value * 0.08838834764831845f));
  }
  return input;
}

std::array<float, 128> quantize(std::array<float, 128> input) {
  float maximum = 1e-4;
  for (float value : input) {
    maximum = std::max(maximum, std::abs(value));
  }
  const float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
  for (float& value : input) {
    value = static_cast<float>(c10::Float8_e4m3fn(value / scale)) * scale;
  }
  return input;
}

torch::Tensor decode_index_cache(const torch::Tensor& cache) {
  auto bytes = cache.cpu().view({-1, 64 * 132});
  auto values = bytes.narrow(1, 0, 64 * 128)
                    .contiguous()
                    .view(torch::kFloat8_e4m3fn)
                    .to(torch::kFloat32)
                    .view({-1, 64, 128});
  auto scales = bytes.narrow(1, 64 * 128, 64 * 4)
                    .contiguous()
                    .view(torch::kFloat32)
                    .view({-1, 64, 1});
  return (values * scales).view({-1, 128});
}

torch::Tensor pool_reference(const torch::Tensor& keys,
                             const torch::Tensor& gates,
                             const torch::Tensor& ape) {
  auto k = keys.cpu().to(torch::kFloat32).contiguous();
  auto g = gates.cpu().to(torch::kFloat32).contiguous();
  auto a = ape.cpu().to(torch::kFloat32).contiguous();
  auto out = torch::empty({k.size(0) / 4, 128}, torch::kFloat32);
  for (int64_t pool = 0; pool < out.size(0); ++pool) {
    std::array<float, 128> value{};
    for (int64_t d = 0; d < 128; ++d) {
      float maximum = -INFINITY;
      for (int64_t j = 0; j < 4; ++j) {
        maximum = std::max(maximum,
                           g.data_ptr<float>()[(pool * 4 + j) * 128 + d] +
                               a.data_ptr<float>()[j * 128 + d]);
      }
      float numerator = 0, denominator = 0;
      for (int64_t j = 0; j < 4; ++j) {
        const float p = std::exp(g.data_ptr<float>()[(pool * 4 + j) * 128 + d] +
                                 a.data_ptr<float>()[j * 128 + d] - maximum);
        numerator += p * k.data_ptr<float>()[(pool * 4 + j) * 128 + d];
        denominator += p;
      }
      value[d] = static_cast<float>(c10::BFloat16(numerator / denominator));
    }
    value = quantize(rotate(value));
    std::copy(value.begin(), value.end(), out.data_ptr<float>() + pool * 128);
  }
  return out;
}

std::vector<int64_t> reference_tokens(const torch::Tensor& query,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& pools,
                                      int64_t position,
                                      int64_t topk) {
  std::vector<int64_t> tokens;
  if (position + 1 <= topk) {
    tokens.resize(position + 1);
    std::iota(tokens.begin(), tokens.end(), 0);
    return tokens;
  }
  auto q = query.cpu().to(torch::kFloat32).contiguous();
  auto w = weights.cpu().to(torch::kFloat32).contiguous();
  std::vector<double> scores((position + 1) / 4, 0);
  for (int64_t h = 0; h < 32; ++h) {
    std::array<float, 128> value{};
    std::copy_n(q.data_ptr<float>() + h * 128, 128, value.begin());
    value = quantize(rotate(value));
    for (int64_t p = 0; p < static_cast<int64_t>(scores.size()); ++p) {
      double dot = 0;
      for (int64_t d = 0; d < 128; ++d) {
        dot += value[d] * pools.data_ptr<float>()[p * 128 + d];
      }
      scores[p] += w.data_ptr<float>()[h] * std::max(dot, 0.0) / 64.0;
    }
  }
  std::vector<int64_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return scores[a] > scores[b];
  });
  for (int64_t i = 0; i < std::min<int64_t>(topk / 4, order.size()); ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      tokens.push_back(order[i] * 4 + j);
    }
  }
  for (int64_t j = ((position + 1) / 4) * 4; j <= position; ++j) {
    tokens.push_back(j);
  }
  std::sort(tokens.begin(), tokens.end());
  return tokens;
}

std::vector<int64_t> selected_tokens(const Selection& selection, int64_t row) {
  auto values = selection.tokens[row].cpu().contiguous();
  std::vector<int64_t> result;
  for (int64_t i = 0; i < values.numel(); ++i) {
    const int32_t token = values.data_ptr<int32_t>()[i];
    if (token >= 0) {
      result.push_back(token);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

TEST_F(GlmSparseTest, IndexPoolMatchesFormulaAcrossPageAndRequestBoundaries) {
  auto opts = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto q = torch::randn({526, 32, 128}, opts) * 0.1;
  auto k = torch::randn({526, 128}, opts) * 0.1;
  auto gate = torch::randn({526, 128}, opts) * 0.2;
  auto weights = torch::randn({526, 32}, opts.dtype(torch::kFloat32)) * 0.1;
  auto ape = torch::randn({4, 128}, opts.dtype(torch::kFloat32)) * 0.1;
  auto positions = torch::cat({torch::arange(257, torch::kInt64),
                               torch::arange(269, torch::kInt64)})
                       .to(device_);
  auto table = torch::tensor({{1, 3}, {0, 2}}, torch::kInt32).to(device_);
  auto cu = torch::tensor({0, 257, 526}, torch::kInt32).to(device_);
  auto ids = torch::tensor({1, 2}, torch::kInt32).to(device_);
  auto cache = torch::zeros({4, 64, 1, 132}, opts.dtype(torch::kUInt8));
  auto tail = torch::zeros({3, 2, 4, 128}, opts);
  auto selected = kernel::musa::glm5_next_kpool(q,
                                                k,
                                                gate,
                                                weights,
                                                ape,
                                                positions,
                                                cu,
                                                ids,
                                                ids,
                                                table,
                                                cache,
                                                tail,
                                                269,
                                                8);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto decoded = decode_index_cache(cache);
  for (int64_t request = 0; request < 2; ++request) {
    const int64_t offset = request == 0 ? 0 : 257;
    const int64_t size = request == 0 ? 257 : 269;
    auto expected = pool_reference(
        k.narrow(0, offset, size), gate.narrow(0, offset, size), ape);
    auto actual = torch::empty_like(expected);
    for (int64_t p = 0; p < expected.size(0); ++p) {
      const int64_t page = request == 0 ? (p < 64 ? 1 : 3) : (p < 64 ? 0 : 2);
      actual[p].copy_(decoded[page * 64 + p % 64]);
    }
    EXPECT_TRUE(torch::allclose(actual, expected, 0.08, 0.004));
    for (int64_t position : {0, 3, 7, 8, 10, 11, 12, 254, 255, 256}) {
      SCOPED_TRACE(::testing::Message()
                   << "req=" << request << " pos=" << position);
      EXPECT_EQ(selected_tokens(selected, offset + position),
                reference_tokens(q[offset + position],
                                 weights[offset + position],
                                 actual,
                                 position,
                                 8));
    }
  }
  EXPECT_EQ(tail[0].cpu().abs().sum().item<float>(), 0);
}

TEST_F(GlmSparseTest, ProductionTop512PoolsExpands2048TokensAndThreeTokenTail) {
  auto opts = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto q = torch::randn({2053, 32, 128}, opts) * 0.1;
  auto k = torch::randn({2053, 128}, opts) * 0.1;
  auto gates = torch::randn({2053, 128}, opts) * 0.2;
  auto weights = torch::randn({2053, 32}, opts.dtype(torch::kFloat32)) * 0.1;
  auto ape = torch::randn({4, 128}, opts.dtype(torch::kFloat32)) * 0.1;
  auto positions = torch::arange(2053, opts.dtype(torch::kInt64));
  auto cu = torch::tensor({0, 2053}, torch::kInt32).to(device_);
  auto states = torch::tensor({1}, torch::kInt32).to(device_);
  auto table = torch::arange(9, opts.dtype(torch::kInt32)).unsqueeze(0);
  auto cache = torch::zeros({9, 64, 1, 132}, opts.dtype(torch::kUInt8));
  auto tail = torch::zeros({2, 2, 4, 128}, opts);
  auto selected = kernel::musa::glm5_next_kpool(q,
                                                k,
                                                gates,
                                                weights,
                                                ape,
                                                positions,
                                                cu,
                                                states,
                                                states,
                                                table,
                                                cache,
                                                tail,
                                                2053,
                                                2048);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto pools = decode_index_cache(cache);
  EXPECT_EQ(selected.tokens.size(1), 2176);
  EXPECT_EQ(selected.lengths[2050].cpu().item<int32_t>(), 2051);
  for (int64_t position : {2047, 2048, 2049, 2050, 2051, 2052}) {
    SCOPED_TRACE(position);
    EXPECT_EQ(selected_tokens(selected, position),
              reference_tokens(
                  q[position], weights[position], pools, position, 2048));
  }
}

ModelArgs model_args(int64_t heads = 8,
                     int64_t hidden = 64,
                     int64_t rank = 32) {
  ModelArgs args;
  args.hidden_size(hidden);
  args.q_lora_rank(rank);
  args.n_heads(heads);
  args.qk_nope_head_dim(256);
  args.qk_rope_head_dim(0);
  args.v_head_dim(256);
  args.kv_lora_rank(512);
  args.index_n_heads(32);
  args.index_head_dim(128);
  args.index_kpool(4);
  args.index_topk(8);
  args.index_kpool_compress(true);
  args.index_kpool_always_select_tail(true);
  args.rms_norm_eps(1e-5);
  return args;
}

Weights make_weights(const ModelArgs& args) {
  Weights w;
  auto add = [&](const std::string& name, std::vector<int64_t> shape) {
    w[name] = torch::randn(shape, torch::kBFloat16) * 0.02;
  };
  add("q_a_proj.weight", {args.q_lora_rank(), args.hidden_size()});
  add("kv_a_proj_with_mqa.weight", {512, args.hidden_size()});
  add("q_b_proj.weight", {args.n_heads() * 256, args.q_lora_rank()});
  add("kv_b_proj.weight", {args.n_heads() * 512, 512});
  add("o_proj.weight", {args.hidden_size(), args.n_heads() * 256});
  w["q_a_layernorm.weight"] =
      torch::ones({args.q_lora_rank()}, torch::kBFloat16);
  w["kv_a_layernorm.weight"] = torch::ones({512}, torch::kBFloat16);
  add("indexer.wq_b.weight", {32 * 128, args.q_lora_rank()});
  add("indexer.wk.weight", {128, args.hidden_size()});
  add("indexer.weights_proj.weight", {32, args.hidden_size()});
  add("indexer.index_kpool_compress_gate", {128, args.hidden_size()});
  w["indexer.k_norm.weight"] = torch::ones({128}, torch::kFloat32);
  w["indexer.k_norm.bias"] = torch::randn({128}, torch::kFloat32) * 0.01;
  w["indexer.index_kpool_compress_ape"] =
      torch::randn({4, 128}, torch::kFloat32) * 0.1;
  return w;
}

AttentionMetadata metadata(int64_t start,
                           int64_t count,
                           const torch::Device& device) {
  AttentionMetadata meta{};
  meta.q_cu_seq_lens =
      torch::tensor({int64_t{0}, count}, torch::kInt32).to(device);
  meta.block_table = torch::tensor({{0, 1}}, torch::kInt32).to(device);
  meta.slot_mapping =
      torch::arange(start,
                    start + count,
                    torch::TensorOptions().dtype(torch::kInt64).device(device));
  meta.linear_state_indices = torch::tensor({1}, torch::kInt32).to(device);
  meta.max_seq_len = start + count;
  return meta;
}

// Independent dense K/V form. Projections are separate GEMMs, norms use their
// defining formulas, and the indexer uses scalar CPU reference math.
std::pair<torch::Tensor, std::vector<std::vector<int64_t>>> full_reference(
    const torch::Tensor& x,
    const Weights& weights,
    const ModelArgs& args) {
  auto linear = [&](const torch::Tensor& value, const std::string& name) {
    return torch::matmul(
        value, weights.at(name).to(value.device(), value.scalar_type()).t());
  };
  auto norm = [&](const torch::Tensor& value, const std::string& name) {
    auto f = value.to(torch::kFloat32);
    return (f * (f.square().mean(-1, true) + args.rms_norm_eps()).rsqrt() *
            weights.at(name).to(x.device(), torch::kFloat32))
        .to(torch::kBFloat16);
  };
  auto qc = norm(linear(x, "q_a_proj.weight"), "q_a_layernorm.weight");
  auto c =
      norm(linear(x, "kv_a_proj_with_mqa.weight"), "kv_a_layernorm.weight");
  auto query =
      linear(qc, "q_b_proj.weight").view({x.size(0), args.n_heads(), 256});
  auto expanded = linear(c, "kv_b_proj.weight")
                      .view({x.size(0), args.n_heads(), 512})
                      .to(torch::kFloat32);
  auto iq = linear(qc, "indexer.wq_b.weight").view({x.size(0), 32, 128});
  auto ik = linear(x, "indexer.wk.weight").to(torch::kFloat32);
  auto centered = ik - ik.mean(-1, true);
  ik = (centered * (centered.square().mean(-1, true) + 1e-6).rsqrt() *
            weights.at("indexer.k_norm.weight").to(x.device()) +
        weights.at("indexer.k_norm.bias").to(x.device()))
           .to(torch::kBFloat16);
  auto gate = linear(x, "indexer.index_kpool_compress_gate");
  auto hw = linear(x.to(torch::kFloat32), "indexer.weights_proj.weight");
  auto pools =
      pool_reference(ik, gate, weights.at("indexer.index_kpool_compress_ape"));
  auto out = torch::zeros({x.size(0), args.n_heads(), 256},
                          x.options().dtype(torch::kFloat32));
  std::vector<std::vector<int64_t>> selections;
  for (int64_t row = 0; row < x.size(0); ++row) {
    auto ids =
        reference_tokens(iq[row], hw[row], pools, row, args.index_topk());
    selections.push_back(ids);
    auto selected = torch::tensor(ids, torch::kInt64).to(x.device());
    auto kv = expanded.index_select(0, selected).transpose(0, 1);
    auto logits = torch::bmm(query[row].to(torch::kFloat32).unsqueeze(1),
                             kv.narrow(2, 0, 256).transpose(1, 2)) *
                  0.0625;
    out[row].copy_(
        torch::bmm(logits.softmax(-1), kv.narrow(2, 256, 256)).squeeze(1));
  }
  return {linear(out.to(torch::kBFloat16).flatten(1, 2), "o_proj.weight"),
          selections};
}

TEST_F(GlmSparseTest,
       CompleteForwardMatchesIndependentReferenceAndChunkedDecode) {
  auto args = model_args();
  auto weights = make_weights(args);
  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(weights));
  layer->verify_loaded_weights();
  auto x = torch::randn({17, 64}, opts) * 0.3;
  auto positions = torch::arange(17, opts.dtype(torch::kInt64));
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto full = layer->forward(positions, x, metadata(0, 17, device_), cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto expected = full_reference(x, weights, args);
  EXPECT_TRUE(torch::allclose(full.first.cpu().to(torch::kFloat32),
                              expected.first.cpu().to(torch::kFloat32),
                              0.04,
                              0.004));
  for (int64_t row = 0; row < 17; ++row) {
    EXPECT_EQ(selected_tokens(full.second, row), expected.second[row]);
  }
  auto chunk_cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  std::vector<torch::Tensor> outputs;
  int64_t start = 0;
  for (int64_t end : {1, 3, 4, 7, 12, 13, 14, 17}) {
    auto result = layer->forward(positions.narrow(0, start, end - start),
                                 x.narrow(0, start, end - start),
                                 metadata(start, end - start, device_),
                                 chunk_cache);
    outputs.push_back(result.first);
    for (int64_t row = 0; row < end - start; ++row) {
      EXPECT_EQ(selected_tokens(result.second, row),
                selected_tokens(full.second, start + row));
    }
    start = end;
  }
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(torch::cat(outputs, 0).cpu().to(torch::kFloat32),
                              full.first.cpu().to(torch::kFloat32),
                              0.01,
                              0.001));
  EXPECT_TRUE(torch::equal(cache.get_kpool_tail().cpu(),
                           chunk_cache.get_kpool_tail().cpu()));
  EXPECT_TRUE(torch::equal(cache.get_index_cache().cpu(),
                           chunk_cache.get_index_cache().cpu()));
}

TEST_F(GlmSparseTest,
       LoadsShardedCheckpointWithFp8BlockScalesAndReusesSelection) {
  auto args = model_args();
  auto source = make_weights(args);
  auto dense = source;
  Weights scales;
  for (const std::string& name : {"q_a_proj.weight",
                                  "kv_a_proj_with_mqa.weight",
                                  "q_b_proj.weight",
                                  "o_proj.weight"}) {
    const auto shape = source.at(name).sizes();
    auto scale = torch::rand({(shape[0] + 127) / 128, (shape[1] + 127) / 128},
                             torch::kFloat32) *
                     0.01 +
                 0.01;
    auto expanded = scale.repeat_interleave(128, 0)
                        .repeat_interleave(128, 1)
                        .narrow(0, 0, shape[0])
                        .narrow(1, 0, shape[1]);
    auto packed = (source.at(name).to(torch::kFloat32) / expanded)
                      .to(torch::kFloat8_e4m3fn);
    source[name] = packed;
    dense[name] = (packed.to(torch::kFloat32) * expanded).to(torch::kBFloat16);
    scales[name.substr(0, name.size() - 6) + "weight_scale_inv"] = scale;
  }
  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(source));
  for (const auto& [name, tensor] : scales) {
    layer->load_state_dict(StateDict({{name, tensor}}));
  }
  layer->verify_loaded_weights();
  Glm5NextSparseAttention baseline(args, parallel, opts);
  baseline->load_state_dict(StateDict(dense));
  baseline->verify_loaded_weights();
  auto x = torch::randn({17, 64}, opts) * 0.3;
  auto positions = torch::arange(17, opts.dtype(torch::kInt64));
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto baseline_cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto actual = layer->forward(positions, x, metadata(0, 17, device_), cache);
  auto expected =
      baseline->forward(positions, x, metadata(0, 17, device_), baseline_cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(actual.first.cpu().to(torch::kFloat32),
                              expected.first.cpu().to(torch::kFloat32),
                              0.005,
                              0.0005));
  Glm5NextSparseAttention reuse(args, parallel, opts, false);
  reuse->load_state_dict(StateDict(dense));
  reuse->verify_loaded_weights();
  auto reuse_cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto reused = reuse->forward(
      positions, x, metadata(0, 17, device_), reuse_cache, &expected.second);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(reused.first.cpu().to(torch::kFloat32),
                              expected.first.cpu().to(torch::kFloat32),
                              0.005,
                              0.0005));
  EXPECT_EQ(reuse_cache.get_kpool_tail().cpu().abs().sum().item<float>(), 0);
}

class CompletedReduction final : public c10d::Work {
 public:
  bool wait(std::chrono::milliseconds = std::chrono::milliseconds{0}) override {
    return true;
  }
};

class ReductionRecorder final : public ProcessGroup {
 public:
  explicit ReductionRecorder(int32_t rank, const torch::Device& device)
      : ProcessGroup(rank, 2, device) {}
  void allreduce(torch::Tensor& tensor) override {
    ++calls;
    partial = tensor.clone();
  }
  c10::intrusive_ptr<c10d::Work> allreduce_async(
      torch::Tensor& tensor) override {
    allreduce(tensor);
    return c10::make_intrusive<CompletedReduction>();
  }
  int32_t calls = 0;
  torch::Tensor partial;
};

TEST_F(GlmSparseTest,
       TensorParallelShardsAndOutputReductionMatchUnshardedLayer) {
  auto args = model_args(16);
  auto weights = make_weights(args);
  ProcessGroup one(0, 1, device_);
  ParallelArgs serial(0, 1, &one);
  serial.tp_group_ = &one;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention full(args, serial, opts);
  full->load_state_dict(StateDict(weights));
  full->verify_loaded_weights();
  auto x = torch::randn({13, 64}, opts) * 0.3;
  auto positions = torch::arange(13, opts.dtype(torch::kInt64));
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto expected = full->forward(positions, x, metadata(0, 13, device_), cache);
  std::vector<torch::Tensor> partials;
  for (int32_t rank = 0; rank < 2; ++rank) {
    ReductionRecorder group(rank, device_);
    ParallelArgs parallel(rank, 2, &group);
    parallel.tp_group_ = &group;
    Glm5NextSparseAttention layer(args, parallel, opts);
    layer->load_state_dict(StateDict(weights));
    layer->verify_loaded_weights();
    auto state = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
    auto output = layer->forward(positions, x, metadata(0, 13, device_), state);
    EXPECT_EQ(group.calls, 1);
    EXPECT_EQ(group.partial.device(), device_);
    partials.push_back(output.first.to(torch::kFloat32));
    for (int64_t row = 0; row < 13; ++row) {
      EXPECT_EQ(selected_tokens(output.second, row),
                selected_tokens(expected.second, row));
    }
  }
  auto reduced = (partials[0] + partials[1]).to(torch::kBFloat16);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(reduced.cpu().to(torch::kFloat32),
                              expected.first.cpu().to(torch::kFloat32),
                              0.04,
                              0.004));
}

TEST_F(GlmSparseTest, FullFp8LayerMatchesChunkedDecode) {
  auto args = model_args();
  auto weights = make_weights(args);
  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(weights));
  layer->verify_loaded_weights();
  auto x = torch::randn({17, 64}, opts) * 0.3;
  auto positions = torch::arange(17, opts.dtype(torch::kInt64));
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts, true);
  auto full = layer->forward(positions, x, metadata(0, 17, device_), cache);
  auto chunk_cache =
      Glm5NextSparseAttentionImpl::create_cache(2, 3, opts, true);
  std::vector<torch::Tensor> outputs;
  int64_t start = 0;
  for (int64_t end : {3, 4, 11, 12, 13, 14, 17}) {
    auto result = layer->forward(positions.narrow(0, start, end - start),
                                 x.narrow(0, start, end - start),
                                 metadata(start, end - start, device_),
                                 chunk_cache);
    outputs.push_back(result.first);
    for (int64_t row = 0; row < end - start; ++row) {
      EXPECT_EQ(selected_tokens(result.second, row),
                selected_tokens(full.second, start + row));
    }
    start = end;
  }
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(torch::cat(outputs, 0).cpu().to(torch::kFloat32),
                              full.first.cpu().to(torch::kFloat32),
                              0.02,
                              0.002));
  EXPECT_TRUE(
      torch::equal(cache.get_k_cache().cpu(), chunk_cache.get_k_cache().cpu()));
}

TEST_F(GlmSparseTest, BatchedRequestsPaddingTailRestoreAndSlotReuse) {
  auto args = model_args();
  auto weights = make_weights(args);
  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(weights));
  layer->verify_loaded_weights();
  auto x = torch::randn({11, 64}, opts) * 0.3;
  auto y = torch::randn({13, 64}, opts) * 0.3;
  auto positions = torch::cat({torch::arange(10, torch::kInt64),
                               torch::arange(13, torch::kInt64),
                               torch::tensor({-1}, torch::kInt64)})
                       .to(device_);
  AttentionMetadata batch{};
  batch.q_cu_seq_lens =
      torch::tensor({0, 10, 23, 24}, torch::kInt32).to(device_);
  batch.block_table =
      torch::tensor({{1, 3}, {0, 2}, {-1, -1}}, torch::kInt32).to(device_);
  batch.linear_state_indices =
      torch::tensor({1, 2, 0}, torch::kInt32).to(device_);
  batch.slot_mapping = torch::cat({torch::arange(256, 266, torch::kInt64),
                                   torch::arange(13, torch::kInt64),
                                   torch::tensor({-1}, torch::kInt64)})
                           .to(device_);
  batch.max_seq_len = 13;
  auto cache = Glm5NextSparseAttentionImpl::create_cache(4, 4, opts);
  auto actual = layer->forward(
      positions,
      torch::cat({x.narrow(0, 0, 10), y, torch::zeros({1, 64}, opts)}, 0),
      batch,
      cache);
  auto xcache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto ycache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto expected_x = layer->forward(torch::arange(10, opts.dtype(torch::kInt64)),
                                   x.narrow(0, 0, 10),
                                   metadata(0, 10, device_),
                                   xcache);
  auto expected_y = layer->forward(torch::arange(13, opts.dtype(torch::kInt64)),
                                   y,
                                   metadata(0, 13, device_),
                                   ycache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(
      torch::allclose(actual.first.narrow(0, 0, 10).cpu().to(torch::kFloat32),
                      expected_x.first.cpu().to(torch::kFloat32),
                      0.01,
                      0.001));
  EXPECT_TRUE(
      torch::allclose(actual.first.narrow(0, 10, 13).cpu().to(torch::kFloat32),
                      expected_y.first.cpu().to(torch::kFloat32),
                      0.01,
                      0.001));
  EXPECT_EQ(actual.first[23].cpu().abs().sum().item<float>(), 0);
  EXPECT_TRUE(selected_tokens(actual.second, 23).empty());
  EXPECT_EQ(cache.get_kpool_tail()[0].cpu().abs().sum().item<float>(), 0);
  auto saved_tail = cache.get_kpool_tail()[1].clone();
  auto restored_meta = metadata(10, 1, device_);
  restored_meta.block_table =
      torch::tensor({{1, 3}}, torch::kInt32).to(device_);
  restored_meta.slot_mapping += 256;
  restored_meta.linear_state_indices =
      torch::tensor({3}, torch::kInt32).to(device_);
  restored_meta.linear_state_read_indices =
      torch::tensor({1}, torch::kInt32).to(device_);
  auto restored = layer->forward(torch::tensor({10}, torch::kInt64).to(device_),
                                 x.narrow(0, 10, 1),
                                 restored_meta,
                                 cache);
  auto next = layer->forward(torch::tensor({10}, torch::kInt64).to(device_),
                             x.narrow(0, 10, 1),
                             metadata(10, 1, device_),
                             xcache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(restored.first.cpu().to(torch::kFloat32),
                              next.first.cpu().to(torch::kFloat32),
                              0.01,
                              0.001));
  EXPECT_TRUE(torch::equal(cache.get_kpool_tail()[1].cpu(), saved_tail.cpu()));
  EXPECT_TRUE(torch::equal(cache.get_kpool_tail()[3].cpu(),
                           xcache.get_kpool_tail()[1].cpu()));
  auto reset_meta = metadata(0, 1, device_);
  reset_meta.block_table = torch::tensor({{3, 1}}, torch::kInt32).to(device_);
  reset_meta.slot_mapping += 768;
  reset_meta.linear_state_indices =
      torch::tensor({3}, torch::kInt32).to(device_);
  auto reset = layer->forward(torch::tensor({0}, torch::kInt64).to(device_),
                              y.narrow(0, 0, 1),
                              reset_meta,
                              cache);
  auto fresh = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto fresh_result =
      layer->forward(torch::tensor({0}, torch::kInt64).to(device_),
                     y.narrow(0, 0, 1),
                     metadata(0, 1, device_),
                     fresh);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::equal(cache.get_kpool_tail()[3].cpu(),
                           fresh.get_kpool_tail()[1].cpu()));
  EXPECT_TRUE(torch::equal(reset.first.cpu(), fresh_result.first.cpu()));
}

// Launched in two fresh processes by the validation command. Each process owns
// a real MUSA device and MCCL ProcessGroup, rather than a test collective.
TEST_F(GlmSparseTest, McclTensorParallelWorker) {
  const char* rank_text = std::getenv("XLLM_GLM_TP_RANK");
  const char* port_text = std::getenv("XLLM_GLM_TP_PORT");
  if (rank_text == nullptr || port_text == nullptr) {
    GTEST_SKIP() << "Run with the two-process MCCL validation launcher";
  }
  const int32_t rank = std::stoi(rank_text);
  const torch::Device device(torch::kPrivateUse1, rank);
  c10::musa::MUSAGuard device_guard(device);
  const auto stream = c10::musa::getStreamFromPool(false, rank);
  c10::StreamGuard stream_guard(stream.unwrap());
  auto group = create_process_group(rank,
                                    rank,
                                    std::vector<int32_t>{0, 1},
                                    2,
                                    2,
                                    std::stoi(port_text),
                                    "127.0.0.1",
                                    "glm_sparse_tp",
                                    device);
  ParallelArgs parallel(rank, 2, group.get());
  parallel.tp_group_ = group.get();
  auto args = model_args(16);
  auto weights = make_weights(args);
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(weights));
  layer->verify_loaded_weights();
  auto x = (torch::randn({13, 64}, torch::kBFloat16) * 0.3).to(device);
  auto positions = torch::arange(13, opts.dtype(torch::kInt64));
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto actual = layer->forward(positions, x, metadata(0, 13, device), cache);
  ProcessGroup one(0, 1, device);
  ParallelArgs serial(0, 1, &one);
  serial.tp_group_ = &one;
  Glm5NextSparseAttention full(args, serial, opts);
  full->load_state_dict(StateDict(weights));
  full->verify_loaded_weights();
  auto full_cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto expected =
      full->forward(positions, x, metadata(0, 13, device), full_cache);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(actual.first.cpu().to(torch::kFloat32),
                              expected.first.cpu().to(torch::kFloat32),
                              0.04,
                              0.004));
}

TEST_F(GlmSparseTest, ProductionDimensionsWithSyntheticWeights) {
  auto args = model_args(64, 4096, 1536);
  args.index_topk(2048);
  auto weights = make_weights(args);
  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  Glm5NextSparseAttention layer(args, parallel, opts);
  layer->load_state_dict(StateDict(weights));
  layer->verify_loaded_weights();
  auto x = torch::randn({1, 4096}, opts) * 0.1;
  auto cache = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto actual = layer->forward(torch::tensor({0}, torch::kInt64).to(device_),
                               x,
                               metadata(0, 1, device_),
                               cache);
  auto raw =
      torch::matmul(x, weights.at("kv_a_proj_with_mqa.weight").to(device_).t())
          .to(torch::kFloat32);
  auto c =
      (raw * (raw.square().mean(-1, true) + 1e-5).rsqrt()).to(torch::kBFloat16);
  auto kv = torch::matmul(c, weights.at("kv_b_proj.weight").to(device_).t())
                .view({1, 64, 512});
  auto expected =
      torch::matmul(kv.narrow(2, 256, 256).contiguous().flatten(1, 2),
                    weights.at("o_proj.weight").to(device_).t());
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_EQ(actual.first.sizes(), torch::IntArrayRef({1, 4096}));
  EXPECT_EQ(actual.second.tokens.size(1), 2176);
  EXPECT_TRUE(torch::allclose(actual.first.cpu().to(torch::kFloat32),
                              expected.cpu().to(torch::kFloat32),
                              0.04,
                              0.01));
}
TEST_F(GlmSparseTest,
       DecoderLayerSparseMoeFp8ScaleInvMatchesReferenceInPrefillAndDecode) {
  auto args = model_args(8, 128, 128);
  args.n_layers(4)
      .layer_types({"linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "deepseek_sparse_attention"})
      .mlp_layer_types({"dense", "dense", "dense", "sparse"})
      .n_routed_experts(288)
      .num_experts_per_tok(8)
      .n_shared_experts(1)
      .moe_intermediate_size(128)
      .routed_scaling_factor(2.5f)
      .norm_topk_prob(true)
      .scoring_func("sigmoid")
      .topk_method("noaux_tc")
      .swiglu_limit(10.0f)
      .hidden_act("silu")
      .hc_mult(4)
      .hc_sinkhorn_iters(4)
      .hc_eps(1e-6f);

  const Glm5NextLayerRole dense_role = resolve_glm5_next_layer_role(args, 0);
  EXPECT_EQ(dense_role.attention, Glm5NextAttentionRole::KDA);
  EXPECT_EQ(dense_role.mlp, Glm5NextMlpRole::DENSE);
  const Glm5NextLayerRole sparse_role = resolve_glm5_next_layer_role(args, 3);
  EXPECT_EQ(sparse_role.attention, Glm5NextAttentionRole::DSA);
  EXPECT_EQ(sparse_role.mlp, Glm5NextMlpRole::SPARSE);

  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  QuantArgs quant_args;
  quant_args.quant_method("fp8");
  quant_args.weight_block_size({128, 128});
  const auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  ModelContext context(parallel, args, quant_args, opts);

  Weights attn_weights = make_weights(args);
  Weights full_fp8;
  Weights full_dense;
  for (const auto& [name, tensor] : attn_weights) {
    full_fp8["self_attn." + name] = tensor;
    full_dense["self_attn." + name] = tensor;
  }

  const int64_t hidden = args.hidden_size();
  const int64_t moe_inter = args.moe_intermediate_size();
  const int64_t mix_size = args.hc_mult() * (args.hc_mult() + 2);
  for (const std::string& prefix : {"attn_hc_pre.", "ffn_hc_pre."}) {
    auto fn =
        torch::randn({mix_size, args.hc_mult() * hidden}, torch::kBFloat16) *
        0.01f;
    auto base = torch::randn({mix_size}, torch::kFloat32) * 0.05f;
    auto scale = torch::randn({3}, torch::kFloat32) * 0.05f;
    full_fp8[prefix + "hc_fn"] = fn;
    full_fp8[prefix + "hc_base"] = base;
    full_fp8[prefix + "hc_scale"] = scale;
    full_dense[prefix + "hc_fn"] = fn;
    full_dense[prefix + "hc_base"] = base;
    full_dense[prefix + "hc_scale"] = scale;
  }
  auto in_norm = torch::ones({hidden}, torch::kBFloat16);
  auto post_norm = torch::ones({hidden}, torch::kBFloat16);
  full_fp8["input_layernorm.weight"] = in_norm;
  full_fp8["post_attention_layernorm.weight"] = post_norm;
  full_dense["input_layernorm.weight"] = in_norm;
  full_dense["post_attention_layernorm.weight"] = post_norm;

  auto gate_w = torch::randn({288, hidden}, torch::kBFloat16) * 0.02f;
  auto gate_b = torch::randn({288}, torch::kFloat32) * 0.01f;
  full_fp8["mlp.gate.weight"] = gate_w;
  full_fp8["mlp.gate.e_score_correction_bias"] = gate_b;
  full_dense["mlp.gate.weight"] = gate_w;
  full_dense["mlp.gate.e_score_correction_bias"] = gate_b;

  auto quantize_128 = [&](const std::string& key, int64_t rows, int64_t cols) {
    auto fp32 = torch::randn({rows, cols}, torch::kFloat32) * 0.02f;
    auto scale =
        torch::rand({(rows + 127) / 128, (cols + 127) / 128}, torch::kFloat32) *
            0.01f +
        0.01f;
    auto expanded = scale.repeat_interleave(128, 0)
                        .repeat_interleave(128, 1)
                        .narrow(0, 0, rows)
                        .narrow(1, 0, cols);
    auto fp8 = (fp32 / expanded).to(torch::kFloat8_e4m3fn);
    auto dequant = (fp8.to(torch::kFloat32) * expanded).to(torch::kBFloat16);
    full_fp8[key + ".weight"] = fp8;
    full_fp8[key + ".weight_scale_inv"] = scale;
    full_dense[key + ".weight"] = dequant;
  };

  quantize_128("mlp.shared_experts.gate_proj", moe_inter, hidden);
  quantize_128("mlp.shared_experts.up_proj", moe_inter, hidden);
  quantize_128("mlp.shared_experts.down_proj", hidden, moe_inter);
  for (int32_t e = 0; e < 288; ++e) {
    const std::string ep = "mlp.experts." + std::to_string(e);
    quantize_128(ep + ".gate_proj", moe_inter, hidden);
    quantize_128(ep + ".up_proj", moe_inter, hidden);
    quantize_128(ep + ".down_proj", hidden, moe_inter);
  }

  Glm5NextDecoderLayer fp8_layer(context, /*layer_id=*/3);
  fp8_layer->load_state_dict(StateDict(full_fp8));
  fp8_layer->verify_loaded_weights();

  ModelContext dense_context(parallel, args, QuantArgs(), opts);
  Glm5NextDecoderLayer dense_layer(dense_context, /*layer_id=*/3);
  dense_layer->load_state_dict(StateDict(full_dense));
  dense_layer->verify_loaded_weights();

  auto x_prefill = torch::randn({5, 4, hidden}, opts) * 0.1f;
  auto x_prefill_ref = x_prefill.clone();
  auto pos_prefill = torch::arange(5, opts.dtype(torch::kInt64));
  auto cache_fp8 = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  auto cache_ref = Glm5NextSparseAttentionImpl::create_cache(2, 3, opts);
  std::optional<torch::Tensor> residual;
  ModelInputParams prefill_params;
  prefill_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  auto out_prefill = fp8_layer->forward(x_prefill,
                                        residual,
                                        pos_prefill,
                                        metadata(0, 5, device_),
                                        cache_fp8,
                                        prefill_params);
  auto ref_prefill = dense_layer->forward(x_prefill_ref,
                                          residual,
                                          pos_prefill,
                                          metadata(0, 5, device_),
                                          cache_ref,
                                          prefill_params);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(out_prefill.cpu().to(torch::kFloat32),
                              ref_prefill.cpu().to(torch::kFloat32),
                              0.04,
                              0.02));

  auto x_decode = torch::randn({1, 4, hidden}, opts) * 0.1f;
  auto x_decode_ref = x_decode.clone();
  auto pos_decode = torch::tensor({5}, opts.dtype(torch::kInt64));
  ModelInputParams decode_params;
  decode_params.meta.batch_forward_type = BatchForwardType::DECODE;
  auto out_decode = fp8_layer->forward(x_decode,
                                       residual,
                                       pos_decode,
                                       metadata(5, 1, device_),
                                       cache_fp8,
                                       decode_params);
  auto ref_decode = dense_layer->forward(x_decode_ref,
                                         residual,
                                         pos_decode,
                                         metadata(5, 1, device_),
                                         cache_ref,
                                         decode_params);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::allclose(out_decode.cpu().to(torch::kFloat32),
                              ref_decode.cpu().to(torch::kFloat32),
                              0.04,
                              0.02));
}

TEST_F(GlmSparseTest, EngineKVCacheAndSharedSelectionCrossLayerForward) {
  constexpr int64_t hidden = 256;
  constexpr int64_t moe_inter = 128;
  auto args = model_args(8, hidden, 512);
  args.model_type("glm5_next");
  args.n_layers(2);
  args.enable_mla(true);
  args.hc_mult(4);
  args.hc_sinkhorn_iters(20);
  args.hc_eps(1e-6f);
  args.rms_norm_eps(1e-5f);
  args.hidden_act("silu");
  args.intermediate_size(512);
  args.moe_intermediate_size(moe_inter);
  args.n_routed_experts(288);
  args.num_experts_per_tok(8);
  args.n_shared_experts(1);
  args.routed_scaling_factor(2.5f);
  args.norm_topk_prob(true);
  args.n_group(1);
  args.topk_group(1);
  args.scoring_func("sigmoid");
  args.topk_method("noaux_tc");
  args.swiglu_limit(10.0f);
  args.layer_types({"deepseek_sparse_attention", "deepseek_sparse_attention"});
  args.mlp_layer_types({"dense", "dense"});
  args.indexer_types({"full", "shared"});

  KVCacheEstimateOptions est_options;
  est_options.dtype = torch::kBFloat16;
  est_options.kv_cache_dtype = "auto";
  est_options.cache_size_in_bytes = 64LL * 1024 * 1024;
  est_options.block_size = 256;
  est_options.world_size = 1;
  est_options.n_local_kv_heads = 1;
  est_options.max_seqs_per_batch = 2;
  est_options.kpool_layout = KPoolCacheLayout::COMPRESSED_WITH_TAIL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(args, est_options);
  EXPECT_EQ(capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(capacity.num_indexer_layers(), 1);
  EXPECT_EQ(capacity.kpool_tail_len(), 4);
  EXPECT_EQ(capacity.index_block_size(), 64 * 132);

  const KVCacheShape shape(capacity, args, /*world_size=*/1);
  ASSERT_TRUE(shape.has_index_cache_shape());
  ASSERT_TRUE(shape.has_kpool_tail_shape());
  EXPECT_EQ(shape.key_cache_shape()[1], 256);
  EXPECT_EQ(shape.key_cache_shape()[2], 1);
  EXPECT_EQ(shape.key_cache_shape()[3], 512);
  EXPECT_TRUE(shape.value_cache_shape().empty());
  EXPECT_EQ(shape.index_cache_shape()[1], 64);
  EXPECT_EQ(shape.index_cache_shape()[2], 1);
  EXPECT_EQ(shape.index_cache_shape()[3], 132);
  EXPECT_EQ(shape.kpool_tail_shape()[1], 2);
  EXPECT_EQ(shape.kpool_tail_shape()[2], 4);
  EXPECT_EQ(shape.kpool_tail_shape()[3], 128);

  KVCacheCreateOptions create_options;
  create_options.device(device_)
      .dtype(torch::kBFloat16)
      .num_layers(2)
      .layer_types(args.layer_types())
      .model_type(args.model_type())
      .enable_lighting_indexer(true)
      .indexer_cache_enabled_layers(
          resolve_indexer_cache_enabled_layers(args, args.n_layers()));

  KVCache cache_layer0(shape, create_options, /*layer_id=*/0);
  KVCache cache_layer1(shape, create_options, /*layer_id=*/1);
  EXPECT_TRUE(cache_layer0.get_k_cache().defined());
  EXPECT_TRUE(cache_layer0.get_index_cache().defined());
  EXPECT_EQ(cache_layer0.get_index_cache().scalar_type(), torch::kUInt8);
  EXPECT_TRUE(cache_layer0.get_kpool_tail().defined());
  EXPECT_EQ(cache_layer0.get_kpool_tail().scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(cache_layer1.get_k_cache().defined());
  EXPECT_FALSE(cache_layer1.get_index_cache().defined());
  EXPECT_FALSE(cache_layer1.get_kpool_tail().defined());

  ProcessGroup group(0, 1, device_);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  const auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  ModelContext context(parallel, args, QuantArgs(), opts);
  Glm5NextDecoderLayer layer0(context, /*layer_id=*/0);
  Glm5NextDecoderLayer layer1(context, /*layer_id=*/1);

  const int64_t mix_size = args.hc_mult() * (args.hc_mult() + 2);
  auto make_layer_weights = [&](bool include_indexer) {
    auto base = make_weights(args);
    Weights weights;
    for (const auto& [k, v] : base) {
      if (!include_indexer && k.rfind("indexer.", 0) == 0) {
        continue;
      }
      weights["self_attn." + k] = v;
    }
    weights["input_layernorm.weight"] = torch::ones({hidden}, torch::kBFloat16);
    weights["post_attention_layernorm.weight"] =
        torch::ones({hidden}, torch::kBFloat16);
    for (const std::string& prefix : {"attn_hc_pre.", "ffn_hc_pre."}) {
      weights[prefix + "hc_fn"] =
          torch::randn({mix_size, args.hc_mult() * hidden}, torch::kBFloat16) *
          0.01f;
      weights[prefix + "hc_base"] =
          torch::randn({mix_size}, torch::kFloat32) * 0.05f;
      weights[prefix + "hc_scale"] = torch::randn({3}, torch::kFloat32) * 0.05f;
    }
    weights["mlp.gate_proj.weight"] =
        torch::randn({512, hidden}, torch::kBFloat16) * 0.02f;
    weights["mlp.up_proj.weight"] =
        torch::randn({512, hidden}, torch::kBFloat16) * 0.02f;
    weights["mlp.down_proj.weight"] =
        torch::randn({hidden, 512}, torch::kBFloat16) * 0.02f;
    return weights;
  };

  layer0->load_state_dict(
      StateDict(make_layer_weights(/*include_indexer=*/true)));
  layer0->verify_loaded_weights();
  layer1->load_state_dict(
      StateDict(make_layer_weights(/*include_indexer=*/false)));
  layer1->verify_loaded_weights();

  auto x_prefill = torch::randn({5, 4, hidden}, opts) * 0.1f;
  auto pos_prefill = torch::arange(5, opts.dtype(torch::kInt64));
  std::optional<torch::Tensor> residual;
  std::optional<PendingMHC> pending_mhc;
  Selection shared_selection;
  ModelInputParams prefill_params;
  prefill_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  auto meta_prefill = metadata(0, 5, device_);

  x_prefill = layer0->forward(x_prefill,
                              residual,
                              pos_prefill,
                              meta_prefill,
                              cache_layer0,
                              prefill_params,
                              &pending_mhc,
                              /*is_last_layer=*/false,
                              &shared_selection);
  EXPECT_TRUE(shared_selection.lengths.defined());
  auto out_prefill = layer1->forward(x_prefill,
                                     residual,
                                     pos_prefill,
                                     meta_prefill,
                                     cache_layer1,
                                     prefill_params,
                                     &pending_mhc,
                                     /*is_last_layer=*/true,
                                     &shared_selection);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::isfinite(out_prefill.cpu()).all().item<bool>());

  auto x_decode = torch::randn({1, 4, hidden}, opts) * 0.1f;
  auto pos_decode = torch::tensor({5}, opts.dtype(torch::kInt64));
  ModelInputParams decode_params;
  decode_params.meta.batch_forward_type = BatchForwardType::DECODE;
  auto meta_decode = metadata(5, 1, device_);
  Selection decode_shared_selection;
  x_decode = layer0->forward(x_decode,
                             residual,
                             pos_decode,
                             meta_decode,
                             cache_layer0,
                             decode_params,
                             &pending_mhc,
                             /*is_last_layer=*/false,
                             &decode_shared_selection);
  auto out_decode = layer1->forward(x_decode,
                                    residual,
                                    pos_decode,
                                    meta_decode,
                                    cache_layer1,
                                    decode_params,
                                    &pending_mhc,
                                    /*is_last_layer=*/true,
                                    &decode_shared_selection);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_TRUE(torch::isfinite(out_decode.cpu()).all().item<bool>());
}
}  // namespace
}  // namespace xllm::layer
