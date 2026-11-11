/* Copyright 2026 The xLLM Authors. Licensed under the Apache License,
 * Version 2.0. */
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
struct Inputs {
  torch::Tensor cache_storage;
  int32_t ratio;
  bool full_tail;
  std::vector<int32_t> lengths, begins;
  torch::Tensor raw, scores, gamma, rope, query, begin, prefix, table,
      state_table;
  torch::Tensor state_raw, state_score, cache, latent, positions, slots,
      num_valid_compressed_rows, map;
};

Inputs make_inputs(std::vector<int32_t> lengths,
                   std::vector<int32_t> begins,
                   int32_t ratio,
                   int64_t padding = 3,
                   bool gathered = false,
                   bool packed = true,
                   bool bf16_gamma = false,
                   int32_t distribution = 0) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto ints = fp32.dtype(torch::kInt32);
  Inputs d;
  d.lengths = std::move(lengths);
  d.begins = std::move(begins);
  d.ratio = ratio;
  d.full_tail = true;
  const int64_t batch = d.lengths.size();
  std::vector<int32_t> q{0}, k{0};
  int32_t end = 1;
  for (int64_t seq = 0; seq < batch; ++seq) {
    q.emplace_back(q.back() + d.lengths[seq]);
    k.emplace_back(k.back() + (d.begins[seq] + d.lengths[seq]) / ratio -
                   d.begins[seq] / ratio);
    end = std::max(end, d.begins[seq] + d.lengths[seq]);
  }
  const int64_t total = q.back();
  std::vector<int32_t> mapping(total);
  std::iota(mapping.begin(), mapping.end(), 0);
  if (gathered) {
    int32_t physical = 0;
    for (int32_t rank = 0; rank < 8; ++rank) {
      for (int32_t part : {0, 1}) {
        for (int64_t seq = 0; seq < batch; ++seq) {
          const int32_t bucket = (d.lengths[seq] + 15) / 16;
          const int32_t half = part == 0 ? rank : 15 - rank;
          const int32_t first = std::min(half * bucket, d.lengths[seq]);
          const int32_t stop = std::min((half + 1) * bucket, d.lengths[seq]);
          for (int32_t row = first; row < stop; ++row) {
            mapping[q[seq] + row] = physical++;
          }
        }
      }
    }
  }
  const torch::ScalarType dtype =
      ratio == 2 ? torch::kFloat32 : torch::kBFloat16;
  const int64_t stride = packed ? 1024 : 512;
  auto owner = torch::full(
      {16 + (total + 3) * stride}, -19, fp32.dtype(dtype).device(device));
  d.raw = owner.as_strided({total + 3, 512}, {stride, 1}, 16);
  auto raw = torch::randn({total, 512}, fp32);
  if (distribution == 1) {
    raw *= 1e-12;
  } else if (distribution == 2) {
    raw = 1 + torch::arange(total * 512, ints)
                      .remainder(17)
                      .to(torch::kFloat32)
                      .view({total, 512}) /
                  512;
  }
  auto ids = torch::tensor(mapping, ints).to(device).to(torch::kInt64);
  d.raw.index_copy_(0, ids, raw.to(device, dtype));
  if (ratio == 2) {
    d.scores = packed ? owner.as_strided({total + 3, 512}, {stride, 1}, 528)
                      : torch::full_like(d.raw, -23);
    d.scores.index_copy_(0, ids, torch::randn({total, 512}, fp32).to(device));
  }
  if (gathered) {
    d.map = torch::tensor(mapping, ints).to(device);
  }
  d.gamma = (0.5 + torch::rand({512}, fp32))
                .to(device, bf16_gamma ? torch::kBFloat16 : torch::kFloat32);
  auto angles = torch::arange(end + 8, fp32).unsqueeze(1) *
                ((torch::arange(32, fp32) + 1) * 0.0001)
                    .repeat_interleave(2)
                    .unsqueeze(0);
  d.rope = torch::cat({angles.cos(), angles.sin()}, 1).to(device);
  const int64_t columns = (end + ratio * 128 - 1) / (ratio * 128);
  const int64_t max_rows =
      batch == 0 ? 0 : *std::max_element(d.lengths.begin(), d.lengths.end());
  const int64_t pages_per_seq =
      (max_rows + ratio * 128 - 1) / (ratio * 128) + 2;
  auto table = torch::empty({batch, columns}, ints);
  for (int64_t seq = 0; seq < batch; ++seq) {
    table[seq].copy_(torch::arange(columns, ints).remainder(pages_per_seq) +
                     seq * pages_per_seq);
  }
  d.table = table.to(device);
  d.state_table =
      torch::arange(batch * 2, ints).view({batch, 2}).flip({1}).to(device);
  auto state = torch::randn({batch * 2 + 1, 32, 1024}, fp32).to(device);
  d.state_raw =
      packed ? state.slice(2, 0, 512) : state.slice(2, 0, 512).contiguous();
  d.state_score =
      packed ? state.slice(2, 512) : state.slice(2, 512).contiguous();
  auto cache_owner = torch::full({batch * pages_per_seq + 3, 128, 512},
                                 -7,
                                 fp32.dtype(torch::kBFloat16).device(device));
  d.cache_storage = cache_owner;
  d.cache = cache_owner.slice(0, 1, cache_owner.size(0) - 1);
  d.query = torch::tensor(q, ints).to(device);
  d.begin = torch::tensor(d.begins, ints).to(device);
  d.prefix = torch::tensor(k, ints).to(device);
  d.latent = torch::full({k.back() + padding, 512}, -9, d.cache.options());
  d.positions = torch::full({k.back() + padding}, -9, ints.device(device));
  d.slots = torch::full_like(d.positions, -9);
  d.num_valid_compressed_rows = torch::full({1}, -9, ints.device(device));
  return d;
}

CompressKvMetadata metadata_for(const Inputs& d) {
  CompressKvMetadata metadata;
  metadata.query_start_loc = d.query;
  metadata.chunk_begins = d.begin;
  metadata.compressed_start_loc = d.prefix;
  metadata.compressed_block_table = d.table;
  if (d.map.defined()) {
    metadata.input_row_map = d.map;
  }
  return metadata;
}

CompressKvOutputs outputs_for(const Inputs& d) {
  CompressKvOutputs outputs;
  outputs.latent = d.latent;
  outputs.positions = d.positions;
  outputs.slots = d.slots;
  outputs.num_valid_rows = d.num_valid_compressed_rows;
  return outputs;
}

CompressKvPoolingInputs pooling_for(const Inputs& d) {
  CompressKvPoolingInputs pooling;
  pooling.scores = d.scores;
  pooling.state_block_table = d.state_table;
  pooling.state_raw = d.state_raw;
  pooling.state_score = d.state_score;
  pooling.save_full_tail = d.full_tail;
  return pooling;
}

void run(const Inputs& d) {
  std::optional<CompressKvPoolingInputs> pooling;
  if (d.ratio == 2) {
    pooling = pooling_for(d);
  }
  CompressKvOptions options;
  options.ratio = d.ratio;
  options.rms_norm_eps = 1e-20;
  fused_compress_kv_cache(d.raw,
                          d.gamma,
                          d.rope,
                          metadata_for(d),
                          pooling,
                          d.cache,
                          outputs_for(d),
                          options);
}

struct Expected {
  torch::Tensor latent, main, positions, slots, state_raw, state_score;
};

Expected reference(const Inputs& d, bool staged) {
  const torch::ScalarType dtype = staged ? torch::kFloat32 : torch::kFloat64;
  const auto raw = d.raw.cpu();
  const auto scores = d.scores.defined() ? d.scores.cpu() : torch::Tensor();
  const auto sr = d.state_raw.cpu(), ss = d.state_score.cpu();
  const auto st = d.state_table.cpu(), table = d.table.cpu();
  const auto gamma = d.gamma.cpu().to(dtype), rope = d.rope.cpu().to(dtype);
  const auto map = d.map.defined()
                       ? d.map.cpu().to(torch::kInt64)
                       : torch::arange(d.raw.size(0), torch::kInt64);
  auto state_raw = sr.clone(), state_score = ss.clone();
  std::vector<torch::Tensor> latent_list, position_list, slot_list;
  latent_list.reserve(d.lengths.size());
  position_list.reserve(d.lengths.size());
  slot_list.reserve(d.lengths.size());
  int64_t offset = 0;
  for (size_t seq = 0; seq < d.lengths.size(); ++seq) {
    const int32_t begin = d.begins[seq], length = d.lengths[seq],
                  end = begin + length;
    const int32_t first = begin / d.ratio, stop = end / d.ratio,
                  count = stop - first;
    if (count > 0) {
      auto values = torch::empty({count, d.ratio, 512},
                                 torch::TensorOptions().dtype(dtype));
      auto gates = torch::empty_like(values);
      for (int32_t member = 0; member < d.ratio; ++member) {
        const int32_t token = first * d.ratio + member;
        const int32_t history = token < begin ? 1 : 0;
        auto source = map.slice(0,
                                offset + token + history * d.ratio - begin,
                                offset + token + count * d.ratio - begin,
                                d.ratio);
        values.select(1, member)
            .slice(0, history)
            .copy_(raw.index_select(0, source));
        if (d.ratio == 2) {
          gates.select(1, member)
              .slice(0, history)
              .copy_(scores.index_select(0, source));
        }
        if (history) {
          const int32_t page =
              st[seq][(token / 128) % st.size(1)].item<int32_t>();
          values[0][member].copy_(sr[page][token % 32]);
          gates[0][member].copy_(ss[page][token % 32]);
        }
      }
      auto pooled = values.select(1, 0);
      if (d.ratio == 2) {
        auto exps = (gates - std::get<0>(gates.max(1, true))).exp();
        pooled = (values * (exps / exps.sum(1, true))).sum(1);
      }
      auto rounded = pooled.to(torch::kBFloat16).to(dtype);
      auto latent =
          (rounded / (rounded.square().sum(-1, true) / 512 + 1e-20).sqrt() *
           gamma)
              .to(torch::kBFloat16);
      auto pos = torch::arange(first, stop, torch::kInt32) * d.ratio;
      auto logical = torch::arange(first, stop, torch::kInt64);
      auto slots = table[seq]
                           .index_select(0, torch::floor_divide(logical, 128))
                           .to(torch::kInt64) *
                       128 +
                   logical.remainder(128);
      latent_list.emplace_back(latent);
      position_list.emplace_back(pos);
      slot_list.emplace_back(slots.to(torch::kInt32));
    }
    if (d.ratio == 2 && length > 0) {
      const int32_t tail = d.full_tail ? std::min(length, 32) : end % 2;
      for (int32_t token = end - tail; token < end; ++token) {
        const int32_t page =
            st[seq][(token / 128) % st.size(1)].item<int32_t>();
        const int64_t source = map[offset + token - begin].item<int64_t>();
        state_raw[page][token % 32].copy_(raw[source]);
        state_score[page][token % 32].copy_(scores[source]);
      }
    }
    offset += length;
  }
  Expected result;
  result.latent = latent_list.empty() ? torch::empty({0, 512}, torch::kBFloat16)
                                      : torch::cat(latent_list);
  result.positions = position_list.empty() ? torch::empty({0}, torch::kInt32)
                                           : torch::cat(position_list);
  result.slots = slot_list.empty() ? torch::empty({0}, torch::kInt32)
                                   : torch::cat(slot_list);
  result.main = result.latent.clone();
  if (result.latent.size(0) > 0) {
    auto pairs = result.latent.slice(1, 448).to(dtype).view({-1, 32, 2});
    auto coefficients =
        rope.index_select(0, result.positions.to(torch::kInt64));
    auto cos = coefficients.slice(1, 0, 64, 2),
         sin = coefficients.slice(1, 64, 128, 2);
    result.main.slice(1, 448, 512, 2)
        .copy_(pairs.select(2, 0) * cos - pairs.select(2, 1) * sin);
    result.main.slice(1, 449, 512, 2)
        .copy_(pairs.select(2, 1) * cos + pairs.select(2, 0) * sin);
  }
  result.state_raw = state_raw;
  result.state_score = state_score;
  return result;
}

std::pair<double, double> errors(const torch::Tensor& value,
                                 const torch::Tensor& expected) {
  auto ref = expected.to(torch::kFloat64),
       delta = value.to(torch::kFloat64) - ref;
  return {(delta.abs().sum() / ref.abs().sum().clamp_min(1e-9)).item<double>(),
          (delta.square().sum() / ref.square().sum().clamp_min(1e-18))
              .sqrt()
              .item<double>()};
}

void check_numeric(const torch::Tensor& value,
                   const torch::Tensor& baseline,
                   const torch::Tensor& expected) {
  EXPECT_TRUE(torch::isfinite(value).all().item<bool>());
  const auto measured = errors(value, expected),
             control = errors(baseline, expected);
  EXPECT_LE(measured.first, 10 * std::max(control.first, 1e-4));
  EXPECT_LE(measured.second, 10 * std::max(control.second, 1e-4));
}

void verify(Inputs& d, const std::function<void()>& execute) {
  const auto golden = reference(d, false), staged = reference(d, true);
  const auto before = d.cache.cpu();
  const auto guards = d.cache_storage.cpu();
  const auto raw = d.raw.cpu(), gamma = d.gamma.cpu(), rope = d.rope.cpu();
  const auto scores = d.scores.defined() ? d.scores.cpu() : torch::Tensor();
  const auto query = d.query.cpu(), begin = d.begin.cpu(),
             prefix = d.prefix.cpu();
  const auto table = d.table.cpu(), state_table = d.state_table.cpu();
  const auto map = d.map.defined() ? d.map.cpu() : torch::Tensor();
  execute();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(d.cache_storage[0].cpu(), guards[0]));
  EXPECT_TRUE(torch::equal(d.cache_storage[-1].cpu(), guards[-1]));
  const int64_t num_valid_compressed_rows = golden.latent.size(0),
                capacity = d.latent.size(0);
  EXPECT_EQ(d.num_valid_compressed_rows.item<int32_t>(),
            num_valid_compressed_rows);
  EXPECT_TRUE(
      torch::equal(d.positions.cpu().slice(0, 0, num_valid_compressed_rows),
                   golden.positions));
  EXPECT_TRUE(torch::equal(d.slots.cpu().slice(0, 0, num_valid_compressed_rows),
                           golden.slots));
  EXPECT_TRUE(
      d.positions.slice(0, num_valid_compressed_rows).eq(0).all().item<bool>());
  EXPECT_TRUE(
      d.slots.slice(0, num_valid_compressed_rows).eq(-1).all().item<bool>());
  EXPECT_TRUE(
      d.latent.slice(0, num_valid_compressed_rows).eq(0).all().item<bool>());
  EXPECT_EQ(d.latent.size(0), capacity);
  auto mask = torch::ones({before.numel() / 512}, torch::kBool);
  mask.index_fill_(0, golden.slots.to(torch::kInt64), false);
  auto actual = d.cache.cpu().view({-1, 512});
  EXPECT_TRUE(
      torch::equal(actual.index({mask}), before.view({-1, 512}).index({mask})));
  const auto written = actual.index_select(0, golden.slots.to(torch::kInt64));
  check_numeric(d.latent.cpu().slice(0, 0, num_valid_compressed_rows),
                staged.latent,
                golden.latent);
  check_numeric(written, staged.main, golden.main);
  EXPECT_TRUE(torch::equal(
      written.slice(1, 0, 448),
      d.latent.cpu().slice(0, 0, num_valid_compressed_rows).slice(1, 0, 448)));
  if (d.ratio == 2) {
    EXPECT_TRUE(torch::equal(d.state_raw.cpu(), golden.state_raw));
    EXPECT_TRUE(torch::equal(d.state_score.cpu(), golden.state_score));
    EXPECT_TRUE(torch::equal(d.scores.cpu(), scores));
  }
  EXPECT_TRUE(torch::equal(d.raw.cpu(), raw));
  EXPECT_TRUE(torch::equal(d.gamma.cpu(), gamma));
  EXPECT_TRUE(torch::equal(d.rope.cpu(), rope));
  EXPECT_TRUE(torch::equal(d.query.cpu(), query));
  EXPECT_TRUE(torch::equal(d.begin.cpu(), begin));
  EXPECT_TRUE(torch::equal(d.prefix.cpu(), prefix));
  EXPECT_TRUE(torch::equal(d.table.cpu(), table));
  EXPECT_TRUE(torch::equal(d.state_table.cpu(), state_table));
  if (map.defined()) {
    EXPECT_TRUE(torch::equal(d.map.cpu(), map));
  }
}

void check(Inputs& d) {
  verify(d, [&]() {
    const void* latent = d.latent.data_ptr();
    const void* positions = d.positions.data_ptr();
    const void* slots = d.slots.data_ptr();
    const void* num_valid_rows = d.num_valid_compressed_rows.data_ptr();
    run(d);
    EXPECT_EQ(d.latent.data_ptr(), latent);
    EXPECT_EQ(d.positions.data_ptr(), positions);
    EXPECT_EQ(d.slots.data_ptr(), slots);
    EXPECT_EQ(d.num_valid_compressed_rows.data_ptr(), num_valid_rows);
  });
}

TEST(FusedCompressKvCacheDeathTest, RejectsMismatchedAndIncompletePooling) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  auto d = make_inputs({6}, {1}, /*ratio=*/2);
  const auto invoke = [&](const std::optional<CompressKvPoolingInputs>& pooling,
                          int64_t ratio) {
    CompressKvOptions options;
    options.ratio = ratio;
    fused_compress_kv_cache(d.raw,
                            d.gamma,
                            d.rope,
                            metadata_for(d),
                            pooling,
                            d.cache,
                            outputs_for(d),
                            options);
  };
  EXPECT_DEATH(invoke(std::nullopt, /*ratio=*/2), "pooling inputs");
  EXPECT_DEATH(invoke(pooling_for(d), /*ratio=*/1), "pooling inputs");
  for (auto field : {&CompressKvPoolingInputs::scores,
                     &CompressKvPoolingInputs::state_block_table,
                     &CompressKvPoolingInputs::state_raw,
                     &CompressKvPoolingInputs::state_score}) {
    auto incomplete = pooling_for(d);
    incomplete.*field = torch::Tensor();
    EXPECT_DEATH(invoke(incomplete, /*ratio=*/2), "requires defined");
  }
}

TEST(FusedCompressKvCacheTest, UsesCurrentDeviceAcrossRepeatedSwitches) {
  uint32_t device_count = 0;
  ASSERT_EQ(cnrtGetDeviceCount(&device_count), cnrtSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "Requires two visible MLU devices";
  }
  for (int32_t device : {0, 1, 0}) {
    SCOPED_TRACE(device);
    const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, device));
    // Two requests force the Union launch and its device topology query.
    auto d = make_inputs({33, 1}, {127, 1}, /*ratio=*/2);
    check(d);
  }
}

TEST(FusedCompressKvCacheTest, SelectedBusinessShapes) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  torch::manual_seed(2026);
  for (int32_t ratio : {1, 2}) {
    for (int32_t batch : {1, 16, 32}) {
      for (int32_t history : {8192, 131072}) {
        SCOPED_TRACE(::testing::Message()
                     << ratio << "/" << batch << "/" << history);
        auto d = make_inputs(std::vector<int32_t>(batch, 6),
                             std::vector<int32_t>(batch, history + 1),
                             ratio);
        check(d);
      }
    }
    for (int32_t length : {8192, 32768}) {
      for (bool gathered : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << ratio << "/" << length << "/" << gathered);
        auto d = make_inputs({length}, {0}, ratio, 3, gathered);
        check(d);
      }
    }
  }
}

TEST(FusedCompressKvCacheTest, TileBoundariesAndFixedLiveCapacity) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  for (int32_t ratio : {1, 2}) {
    for (int32_t length : {15,
                           16,
                           17,
                           31,
                           32,
                           33,
                           127,
                           128,
                           129,
                           255,
                           256,
                           257,
                           4095,
                           4096,
                           4097}) {
      SCOPED_TRACE(::testing::Message() << ratio << "/" << length);
      auto d = make_inputs({length}, {1}, ratio);
      check(d);
    }
    for (int32_t capacity : (ratio == 1 ? std::vector<int32_t>{31, 32, 33}
                                        : std::vector<int32_t>{15, 16, 17})) {
      const int32_t length = ratio == 1 ? 30 : 28;
      auto d = make_inputs({length}, {0}, ratio, capacity - length / ratio);
      check(d);
    }
  }
}

TEST(FusedCompressKvCacheTest, EmptyUnbalancedAndRoundingLayouts) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  for (int32_t ratio : {1, 2}) {
    auto empty = make_inputs({}, {}, ratio, 0);
    empty.raw = empty.raw.narrow(0, 0, 0);
    if (ratio == 2) {
      empty.scores = empty.scores.narrow(0, 0, 0);
    }
    check(empty);
    auto zero = make_inputs({0, 0}, {1, 127}, ratio, 0);
    zero.raw = zero.raw.narrow(0, 0, 0);
    if (ratio == 2) {
      zero.scores = zero.scores.narrow(0, 0, 0);
    }
    check(zero);
    auto state_only = make_inputs({1}, {0}, ratio, 0);
    check(state_only);
    for (bool packed : {false, true}) {
      for (bool gamma : {false, true}) {
        for (int32_t distribution : {0, 1, 2}) {
          auto d = make_inputs({3, 0, 7, 1},
                               {127, 1, 31, 0},
                               ratio,
                               9,
                               true,
                               packed,
                               gamma,
                               distribution);
          d.cache = d.cache.view({d.cache.size(0), 128, 1, 512});
          check(d);
        }
      }
    }
  }
}

TEST(FusedCompressKvCacheTest, PersistentTailRejectResumeAndPageWrap) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  for (bool full : {false, true}) {
    auto d = make_inputs({6}, {269}, 2);
    d.full_tail = full;
    std::vector<int32_t> begins{123};
    begins.reserve(26);
    for (int32_t begin = full ? 125 : 129; begin <= 269; begin += 6) {
      begins.emplace_back(begin);
    }
    for (int32_t begin : begins) {
      SCOPED_TRACE(::testing::Message() << full << "/" << begin);
      d.begins = {begin};
      d.begin.fill_(begin);
      check(d);
    }
  }
}

TEST(FusedCompressKvCacheTest,
     GraphUpdatesMetadataMapAndZeroLiveOnCurrentStream) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  for (int32_t ratio : {1, 2}) {
    auto d = make_inputs({6, 6}, {123, 127}, ratio, 13, true);
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      run(d);
      graph.capture_end();
    }
    auto other = make_inputs({6}, {127}, ratio, 20);
    torch_mlu::synchronize();
    torch_mlu::MLUGraph other_graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      other_graph.capture_begin();
      run(other);
      other_graph.capture_end();
    }
    const void* address = d.latent.data_ptr();
    for (int32_t step = 0; step < 6; ++step) {
      d.lengths = step == 5 ? std::vector<int32_t>{0, 0}
                            : std::vector<int32_t>{6 - step, step};
      d.begins = {125 + step, 129 + step};
      std::vector<int32_t> q{0}, k{0};
      for (int32_t seq = 0; seq < 2; ++seq) {
        q.emplace_back(q.back() + d.lengths[seq]);
        k.emplace_back(k.back() + (d.begins[seq] + d.lengths[seq]) / ratio -
                       d.begins[seq] / ratio);
      }
      d.query.copy_(torch::tensor(q, torch::kInt32));
      d.prefix.copy_(torch::tensor(k, torch::kInt32));
      d.begin.copy_(torch::tensor(d.begins, torch::kInt32));
      d.raw.mul_(0.75);
      if (ratio == 2) {
        d.scores.add_(0.125);
      }
      d.table.copy_(d.table.roll(1, 1));
      d.state_table.copy_(d.state_table.roll(1, 1));
      d.map.copy_(d.map.roll(1));
      verify(d, [&]() { graph.replay(); });
      EXPECT_EQ(d.latent.data_ptr(), address);
      auto saved = d.latent.clone();
      verify(other, [&]() { other_graph.replay(); });
      EXPECT_TRUE(torch::equal(saved, d.latent));
    }
  }
}

TEST(FusedCompressKvCacheTest, DISABLED_RealAddressesBeyondTwoGiB) {
  const torch::DeviceGuard guard(torch::Device(torch::kPrivateUse1, 0));
  auto d = make_inputs({6}, {131071}, 2);
  constexpr int64_t kPage = (int64_t{1} << 31) / (128 * 512 * 2) + 1;
  d.cache = torch::full({kPage + 2, 128, 512}, -7, d.cache.options());
  d.table.fill_(kPage);
  d.state_raw = torch::randn({2 * kPage + 2, 32, 512}, d.state_raw.options());
  d.state_score = torch::randn_like(d.state_raw);
  d.state_table.fill_(2 * kPage);
  check(d);
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
