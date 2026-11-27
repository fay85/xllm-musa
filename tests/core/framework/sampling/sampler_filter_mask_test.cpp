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

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/constraint/xgrammar_machine.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/xgrammar_constraint.h"
#include "platform/platform.h"

namespace xllm {
namespace {
SamplingParameters make_greedy_params(int64_t batch_size) {
  SamplingParameters params;
  params.selected_token_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.sample_idxes =
      torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt));
  params.do_sample = torch::zeros({batch_size}, torch::kBool);
  params.sampling_mode = SamplingMode::GREEDY;

  return params;
}

bool uses_device_random_sample() {
  return Platform::is_mlu() || Platform::is_cuda() || Platform::is_dcu();
}

torch::Device get_random_sample_device() {
  if (uses_device_random_sample()) {
    return torch::Device(Platform::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

TEST(SamplerFilterMaskTest, TokenLookupUsesOriginalLogitsInMixedBatch) {
  if (uses_device_random_sample() && Platform::device_count() == 0) {
    GTEST_SKIP() << "Sampling backend device is unavailable";
  }

  SamplingParameters params = make_greedy_params(/*batch_size=*/3);
  params.logprobs = true;
  params.max_top_logprobs = 3;
  params.top_k = torch::ones({3}, torch::kLong);
  params.logprob_token_ids =
      torch::tensor({{2, 0}, {-1, -1}, {1, -1}}, torch::kInt);
  torch::Tensor logits =
      torch::tensor({{5.F, 4.F, 1.F}, {1.F, 5.F, 2.F}, {4.F, 2.F, 1.F}});
  const torch::Tensor original = logits.clone();
  const torch::Device device = get_random_sample_device();
  params = params.to(device, torch::kFloat32);
  logits = logits.to(device);

  const SampleOutput output = sample(logits, params);
  const torch::Tensor expected = original.log_softmax(/*dim=*/-1);
  const torch::Tensor next_tokens = output.next_tokens.to(torch::kCPU);
  const torch::Tensor top_tokens = output.top_tokens.to(torch::kCPU);
  const torch::Tensor top_logprobs = output.top_logprobs.to(torch::kCPU);
  EXPECT_TRUE(
      torch::equal(next_tokens, torch::tensor({0, 1, 0}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(top_tokens[0], torch::tensor({2, 0, -1}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(top_tokens[2], torch::tensor({1, -1, -1}, torch::kLong)));
  EXPECT_NEAR(top_logprobs.index({0, 0}).item<float>(),
              expected.index({0, 2}).item<float>(),
              1e-6F);
  EXPECT_NEAR(top_logprobs.index({0, 1}).item<float>(),
              expected.index({0, 0}).item<float>(),
              1e-6F);
  EXPECT_NEAR(top_logprobs.index({2, 0}).item<float>(),
              expected.index({2, 1}).item<float>(),
              1e-6F);
  EXPECT_TRUE(torch::isneginf(top_logprobs.index({2, 1})).item<bool>());
  EXPECT_EQ(top_tokens.index({1, 0}).item<int64_t>(), 1);
  EXPECT_TRUE(torch::isneginf(top_logprobs.index({1, 1})).item<bool>());
}

TEST(SamplerFilterMaskTest, TokenLookupUsesSampleIndicesBeforeMaskMutation) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/3);
  params.sample_idxes = torch::tensor({2, 0}, torch::kInt);
  params.do_sample = torch::zeros({2}, torch::kBool);
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logprob_token_ids = torch::tensor({{0, -1}, {1, 2}}, torch::kInt);
  params.filter_mask = torch::tensor({{-10.F, 0.F, 0.F}, {0.F, -10.F, 0.F}});
  torch::Tensor logits =
      torch::tensor({{4.F, 3.F, 2.F}, {0.F, 0.F, 0.F}, {1.F, 4.F, 2.F}});
  const torch::Tensor expected =
      logits.index_select(/*dim=*/0, params.sample_idxes).log_softmax(-1);

  const SampleOutput output = sample(logits, params);
  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({1, 0}, torch::kLong)));
  EXPECT_TRUE(torch::equal(output.top_tokens,
                           torch::tensor({{0, -1}, {1, 2}}, torch::kLong)));
  EXPECT_NEAR(output.top_logprobs.index({0, 0}).item<float>(),
              expected.index({0, 0}).item<float>(),
              1e-6F);
  EXPECT_NEAR(output.top_logprobs.index({1, 1}).item<float>(),
              expected.index({1, 2}).item<float>(),
              1e-6F);
}

TEST(SamplerFilterMaskTest,
     TokenLookupAllRowsPreservesOrderAndFloat32Precision) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    SCOPED_TRACE(dtype);
    SamplingParameters params = make_greedy_params(/*batch_size=*/2);
    params.logprobs = true;
    params.max_top_logprobs = 4;
    params.logprob_token_ids =
        torch::tensor({{2, 0, 2}, {-1, 3, 1}}, torch::kInt);
    torch::Tensor logits =
        torch::tensor({{5.F, 4.F, 1.F, 2.F}, {1.F, 5.F, 2.F, 3.F}}).to(dtype);
    const torch::Tensor original = logits.clone();
    const torch::Tensor original_logprobs =
        torch::log_softmax(original, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    torch::Tensor expected = torch::full({2, 4}, neg_inf);
    expected.index_put_({0, 0}, original_logprobs[0][2]);
    expected.index_put_({0, 1}, original_logprobs[0][0]);
    expected.index_put_({0, 2}, original_logprobs[0][2]);
    expected.index_put_({1, 1}, original_logprobs[1][3]);
    expected.index_put_({1, 2}, original_logprobs[1][1]);

    const SampleOutput output = sample(logits, params);

    EXPECT_EQ(output.top_logprobs.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(output.top_logprobs, expected));
    EXPECT_TRUE(torch::equal(
        output.top_tokens,
        torch::tensor({{2, 0, 2, -1}, {-1, 3, 1, -1}}, torch::kLong)));
    EXPECT_TRUE(
        torch::equal(output.next_tokens, torch::tensor({0, 1}, torch::kLong)));
    EXPECT_TRUE(torch::equal(logits, original));
  }
}

TEST(SamplerFilterMaskTest,
     TokenLookupWithoutValidRowsPreservesNormalSampling) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.logprobs = true;
  params.max_top_logprobs = 3;
  params.temperatures = torch::tensor({2.F, 0.5F});
  params.filter_mask = torch::tensor({{-10.F, 0.F, 0.F}, {0.F, -10.F, 0.F}});
  torch::Tensor logits = torch::tensor({{5.F, 4.F, 1.F}, {1.F, 5.F, 2.F}});
  torch::Tensor normal_logits = logits.clone();
  const SampleOutput normal_output = sample(normal_logits, params);
  params.logprob_token_ids = torch::full({2, 2}, -1, torch::kInt);

  const SampleOutput output = sample(logits, params);

  EXPECT_TRUE(torch::equal(output.next_tokens, normal_output.next_tokens));
  EXPECT_TRUE(torch::equal(output.logprobs, normal_output.logprobs));
  EXPECT_TRUE(torch::equal(output.top_tokens, normal_output.top_tokens));
  EXPECT_TRUE(torch::equal(output.top_logprobs, normal_output.top_logprobs));
  EXPECT_TRUE(torch::equal(logits, normal_logits));
}

TEST(SamplerFilterMaskTest,
     TokenLookupCompactsReorderedRowsBeforeTemperatureAndPenalties) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  SamplingParameters params = make_greedy_params(/*batch_size=*/5);
  params.sample_idxes = torch::tensor({4, 1, 3}, torch::kInt);
  params.do_sample = torch::zeros({3}, torch::kBool);
  params.logprobs = true;
  params.max_top_logprobs = 3;
  params.temperatures = torch::tensor({1.F, 2.F, 1.F, 0.5F, 4.F});
  params.unique_token_ids = torch::zeros({5, 1}, torch::kLong);
  params.unique_token_counts = torch::ones({5, 1}, torch::kInt);
  params.frequency_penalties = torch::ones({5});
  params.presence_penalties = torch::ones({5});
  params.repetition_penalties = torch::full({5}, 2.F);
  torch::Tensor logits = torch::tensor({{10.F, 3.F, 1.F},
                                        {6.F, 5.F, 1.F},
                                        {8.F, 2.F, 1.F},
                                        {4.F, 3.F, 2.F},
                                        {2.F, 4.F, 3.F}});
  const torch::Tensor original_logprobs = logits.log_softmax(/*dim=*/-1);
  torch::Tensor normal_logits = logits.clone();
  const SampleOutput normal_output = sample(normal_logits, params);
  params.logprob_token_ids =
      torch::tensor({{-1, -1}, {2, 0}, {1, -1}}, torch::kInt);
  const torch::Tensor logits_alias = logits;

  const SampleOutput output = sample(logits, params);

  torch::Tensor expected_values = torch::full({3, 3}, neg_inf);
  expected_values[0].copy_(normal_output.top_logprobs[0]);
  expected_values.index_put_({1, 0}, original_logprobs.index({1, 2}));
  expected_values.index_put_({1, 1}, original_logprobs.index({1, 0}));
  expected_values.index_put_({2, 0}, original_logprobs.index({3, 1}));
  torch::Tensor expected_tokens =
      torch::tensor({{-1, -1, -1}, {2, 0, -1}, {1, -1, -1}}, torch::kLong);
  expected_tokens[0].copy_(normal_output.top_tokens[0]);
  EXPECT_TRUE(torch::allclose(output.top_logprobs, expected_values));
  EXPECT_TRUE(torch::equal(output.top_tokens, expected_tokens));
  EXPECT_TRUE(torch::equal(output.next_tokens, normal_output.next_tokens));
  EXPECT_TRUE(torch::equal(output.logprobs, normal_output.logprobs));
  const torch::Tensor expected_logits = torch::tensor({{4.F, 3.F, 1.F},
                                                       {1.F, 2.5F, 0.5F},
                                                       {3.F, 2.F, 1.F},
                                                       {2.F, 6.F, 4.F},
                                                       {0.F, 1.F, 0.75F}});
  EXPECT_TRUE(torch::equal(logits, expected_logits));
  EXPECT_TRUE(torch::equal(logits_alias, expected_logits));
  EXPECT_TRUE(torch::equal(logits, normal_logits));
}

TEST(SamplerFilterMaskTest, GreedySamplingHonorsMixedRows) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.filter_mask =
      torch::tensor({{0.0F, -1.0e9F, 0.0F}, {0.0F, 0.0F, -1.0e9F}});
  torch::Tensor logits =
      torch::tensor({{1.0F, 100.0F, 2.0F}, {1.0F, 2.0F, 100.0F}});

  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(output.next_tokens.index({1}).item<int64_t>(), 1);
}

#if defined(USE_NPU) && defined(USE_XGRAMMAR)
TEST(SamplerFilterMaskTest, XGrammarMasksConstrainNpuSampling) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "XGrammar NPU sampling requires an available device";
  }
  std::vector<std::string> pieces;
  pieces.reserve(257);
  for (int32_t id = 0; id < 256; ++id) {
    pieces.emplace_back(1, static_cast<char>(id));
  }
  pieces.emplace_back();
  XGrammarCompiler compiler(pieces,
                            /*model_vocab_size=*/288,
                            std::vector<int32_t>{256},
                            /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler.compile_regex("a", error);
  ASSERT_NE(grammar, nullptr) << error;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  for (int64_t batch_size : {1, 2}) {
    for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      SCOPED_TRACE(batch_size);
      SCOPED_TRACE(dtype);
      std::shared_ptr<XGrammarConstraintMachine> machine =
          wrap_xgrammar_machine(grammar);
      ASSERT_NE(machine, nullptr);
      ConstraintState state = machine->initial_state(false);
      ASSERT_TRUE(state.initialized());
      for (const int32_t expected : {int32_t{'a'}, 256}) {
        SamplingParameters params = make_greedy_params(batch_size);
        std::vector<ConstraintState> states{state};
        if (batch_size == 2) {
          states.emplace_back();
        }
        params.filter_bitmask = build_filter_bitmask(states);
        ASSERT_EQ(params.filter_bitmask.size(1), 9);
        params = params.to(device, dtype);
        torch::Tensor logits = torch::zeros({batch_size, 288}, torch::kFloat32);
        logits.index_put_({0, 'x'}, 100.0F);
        logits.index_put_({0, expected}, 10.0F);
        if (batch_size == 2) {
          logits.index_put_({1, 'x'}, 100.0F);
        }
        logits = logits.to(device, dtype);
        const SampleOutput output = sample(logits, params);
        const torch::Tensor tokens = output.next_tokens.to(torch::kCPU);
        EXPECT_EQ(tokens[0].item<int32_t>(), expected);
        if (batch_size == 2) {
          EXPECT_EQ(tokens[1].item<int32_t>(), 'x');
        }
        ASSERT_TRUE(machine->advance(
            state, tokens[0].item<int32_t>(), HaltPolicy::RESPECT));
      }
    }
  }
}
#endif

TEST(SamplerFilterMaskTest, RandomSamplingCannotSelectDisallowedToken) {
  if (uses_device_random_sample() && Platform::device_count() == 0) {
    GTEST_SKIP() << "Random sampling backend device is unavailable";
  }

  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.do_sample = torch::ones({1}, torch::kBool);

  params.sampling_mode = SamplingMode::RANDOM;
  params.filter_mask = torch::tensor({{-1.0e9F, 0.0F, -1.0e9F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 100.0F}});

  const torch::Device device = get_random_sample_device();
  params = params.to(device, torch::kFloat32);
  logits = logits.to(device);

  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 1);
}

TEST(SamplerFilterMaskTest, PackedMaskFiltersCallerLogitsInPlace) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0b0101)}},
                    torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor logits = torch::tensor({{1.0F, 100.0F, 2.0F, 100.0F}});

  SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.index({0}).item<int64_t>(), 2);
  EXPECT_EQ(logits.index({0, 0}).item<float>(), 1.0F);
  EXPECT_LT(logits.index({0, 1}).item<float>(), -1.0F);
  EXPECT_EQ(logits.index({0, 2}).item<float>(), 2.0F);
  EXPECT_LT(logits.index({0, 3}).item<float>(), -1.0F);
}

TEST(SamplerFilterMaskTest, TopKSamplingPublishesDistribution) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU top-k sampling requires a device";
  }
  const torch::Device device = get_random_sample_device();
  SamplingParameters params = make_greedy_params(/*batch_size=*/2);
  params.do_sample = torch::ones({2}, torch::kBool);

  params.sampling_mode = SamplingMode::RANDOM;
  params.top_k = torch::ones({2}, torch::kLong);
  params.top_p = torch::ones({2});
  params = params.to(device, torch::kFloat32);
  torch::Tensor logits =
      torch::tensor({{3.0F, 2.0F, 1.0F}, {1.0F, 3.0F, 2.0F}}).to(device);

  const SampleOutput output = sample(logits, params);
  const torch::Tensor expected =
      torch::tensor({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}}).to(device);
  EXPECT_TRUE(torch::equal(logits.softmax(/*dim=*/-1), expected));
  EXPECT_TRUE(torch::equal(output.next_tokens, expected.argmax(/*dim=*/-1)));
}

TEST(SamplerFilterMaskTest, GreedySamplingPublishesDenseMask) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.filter_mask = torch::tensor({{-20.0F, 0.0F, 0.0F}});
  torch::Tensor logits = torch::tensor({{10.0F, 2.0F, 1.0F}});

  const SampleOutput output = sample(logits, params);

  EXPECT_EQ(output.next_tokens.item<int64_t>(), 1);
  EXPECT_TRUE(torch::equal(logits, torch::tensor({{-10.0F, 2.0F, 1.0F}})));
}

TEST(SamplerFilterMaskTest, SubsetSamplingPreservesOtherRows) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU top-k sampling requires a device";
  }
  const torch::Device device = get_random_sample_device();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    for (bool packed_mask : {false, true}) {
      SCOPED_TRACE(dtype);
      SCOPED_TRACE(packed_mask);
      SamplingParameters params = make_greedy_params(/*batch_size=*/3);
      params.sample_idxes = torch::tensor({2, 0}, torch::kLong);
      params.do_sample = torch::ones({2}, torch::kBool);

      params.sampling_mode = SamplingMode::RANDOM;
      params.top_k = torch::tensor({1, 1, 2}, torch::kLong);
      params.top_p = torch::ones({3});
      if (packed_mask) {
        params.filter_bitmask = torch::tensor({{15}, {15}, {13}}, torch::kInt);
      } else {
        params.filter_mask = torch::tensor(
            {{0.0F, neg_inf, 0.0F, 0.0F}, {0.0F, neg_inf, 0.0F, 0.0F}});
      }
      params = params.to(device, torch::kFloat32);
      torch::Tensor logits = torch::tensor({{4.0F, 3.0F, 2.0F, 1.0F},
                                            {3.0F, 2.0F, 1.0F, 0.0F},
                                            {1.0F, 2.0F, 4.0F, 3.0F}})
                                 .to(device, dtype);

      const SampleOutput output = sample(logits, params);

      const torch::Tensor expected =
          torch::tensor({{4.0F, neg_inf, neg_inf, neg_inf},
                         {3.0F, 2.0F, 1.0F, 0.0F},
                         {neg_inf, neg_inf, 4.0F, 3.0F}})
              .to(device);
      EXPECT_EQ(logits.scalar_type(), torch::kFloat32);
      EXPECT_TRUE(torch::equal(logits, expected));
      EXPECT_EQ(output.next_tokens[1].item<int64_t>(), 0);
    }
  }
}

TEST(SamplerFilterMaskTest, FullRowsKeepOrderWithSparseSelection) {
  for (bool logprobs : {false, true}) {
    SamplingParameters params = make_greedy_params(/*batch_size=*/3);
    params.selected_token_idxes =
        torch::tensor(/*values=*/{2, 5, 9}, /*options=*/torch::kLong);
    params.logprobs = logprobs;
    params.filter_mask = torch::tensor(
        {{0.F, -10.F, 0.F}, {0.F, 0.F, -10.F}, {-10.F, 0.F, 0.F}});
    torch::Tensor logits =
        torch::tensor({{1.F, 3.F, 2.F}, {3.F, 2.F, 1.F}, {2.F, 3.F, 1.F}});
    const torch::Tensor expected = logits + params.filter_mask;
    const SampleOutput output = sample(logits, params);
    EXPECT_TRUE(torch::equal(logits, expected));
    EXPECT_TRUE(torch::equal(output.next_tokens, expected.argmax(/*dim=*/-1)));
  }
}

TEST(SamplerFilterMaskTest, DenseProcessorReturnsSelectedRows) {
  SamplingParameters params = make_greedy_params(/*batch_size=*/3);
  params.sample_idxes = torch::tensor({2, 0}, torch::kLong);
  params.filter_mask = torch::tensor({{-10.F, 0.F}, {0.F, -10.F}});
  torch::Tensor logits = torch::tensor({{1.F, 2.F}, {3.F, 4.F}, {5.F, 6.F}});
  const torch::Tensor scores = process_logits(logits, params);
  EXPECT_TRUE(torch::equal(scores, torch::tensor({{-5.F, 6.F}, {1.F, -8.F}})));
}

void check_request_filters(int64_t top_k,
                           float top_p,
                           bool mixed,
                           const std::array<int64_t, 2>& kept_counts) {
  if (!Platform::is_mlu() || Platform::device_count() == 0) {
    GTEST_SKIP() << "MLU integration test requires a device";
  }
  const torch::Device device(Platform::type_torch(), 0);
  // Qwen3.5 and GLM-5.3-Flash vocabulary sizes.
  for (int64_t vocab_size : {248320, 154880}) {
    for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      SCOPED_TRACE(vocab_size);
      SCOPED_TRACE(dtype);
      RequestSamplingParam first;
      first.temperature = 1.F;
      first.top_k = top_k;
      first.top_p = top_p;
      RequestSamplingParam second = first;
      if (mixed) {
        second.top_k = 8;
        second.top_p = 1.F;
      }
      SamplingParameters params;
      params.init({&first, &second}, {0, 1}, {0, 1}, {}, {}, {});
      EXPECT_EQ(params.top_k.defined(), top_k > 0 || mixed);
      EXPECT_EQ(params.top_p.defined(), top_p != 1.F);
      ASSERT_TRUE(params.sampling_mode == SamplingMode::RANDOM);
      torch::Tensor original = torch::full({2, vocab_size}, -100.F);
      const torch::Tensor scores = -torch::arange(32, torch::kFloat32) * 0.2F;
      const torch::Tensor indices =
          torch::arange(32, torch::kLong) * 13 + vocab_size / 2;
      original[0].index_copy_(/*dim=*/0, indices, scores);
      original[1].index_copy_(/*dim=*/0, indices + 7, scores.flip({0}));
      original = original.to(dtype).to(torch::kFloat32);
      // Fixed support for the 32 sparse scores: top-p=.9 keeps 12 tokens,
      // top-k=8 keeps 8, and their intersection keeps 7. No sorting or
      // cumulative-probability implementation is duplicated in this oracle.
      torch::Tensor expected =
          torch::full_like(original, -std::numeric_limits<float>::infinity());
      const torch::Tensor first_indices =
          indices.slice(/*dim=*/0, /*start=*/0, /*end=*/kept_counts[0]);
      const torch::Tensor second_indices =
          (indices + 7)
              .slice(/*dim=*/0,
                     /*start=*/32 - kept_counts[1],
                     /*end=*/32);
      expected[0].index_copy_(
          0, first_indices, original[0].index_select(0, first_indices));
      expected[1].index_copy_(
          0, second_indices, original[1].index_select(0, second_indices));
      if (kept_counts[0] == 32) {
        expected[0].copy_(original[0]);
      }
      torch::Tensor logits = original.to(device, dtype);
      params = params.to(device, torch::kFloat32);
      const SampleOutput output = sample(logits, params);
      const torch::Tensor actual = logits.to(torch::kCPU);
      const torch::Tensor expected_probs = expected.softmax(/*dim=*/-1);
      EXPECT_EQ(logits.scalar_type(), torch::kFloat32);
      EXPECT_TRUE(torch::equal(actual.isneginf(), expected.isneginf()));
      EXPECT_TRUE(torch::allclose(actual.softmax(/*dim=*/-1),
                                  expected_probs,
                                  /*rtol=*/1e-5,
                                  /*atol=*/1e-7));
      const torch::Tensor tokens = output.next_tokens.to(torch::kCPU);
      EXPECT_TRUE(expected_probs.gather(/*dim=*/1, tokens.view({2, 1}))
                      .gt(0)
                      .all()
                      .item<bool>());
    }
  }
}

TEST(SamplerFilterMaskTest, RequestTopPOnlyUsesModelVocabulary) {
  check_request_filters(
      /*top_k=*/-1, /*top_p=*/0.9F, /*mixed=*/false, /*kept_counts=*/{12, 12});
}

TEST(SamplerFilterMaskTest, RequestTopKOnlyUsesModelVocabulary) {
  check_request_filters(
      /*top_k=*/8, /*top_p=*/1.F, /*mixed=*/false, /*kept_counts=*/{8, 8});
}

TEST(SamplerFilterMaskTest, RequestTopKTopPUsesModelVocabulary) {
  check_request_filters(
      /*top_k=*/8, /*top_p=*/0.9F, /*mixed=*/false, /*kept_counts=*/{7, 7});
}

TEST(SamplerFilterMaskTest, MixedRequestsUseModelVocabulary) {
  check_request_filters(
      /*top_k=*/-1, /*top_p=*/0.9F, /*mixed=*/true, /*kept_counts=*/{12, 8});
}

TEST(SamplerFilterMaskTest, MixedRequestsTreatZeroTopKAsDisabled) {
  check_request_filters(
      /*top_k=*/0, /*top_p=*/0.9F, /*mixed=*/true, /*kept_counts=*/{12, 8});
}

TEST(SamplerFilterMaskTest, RequestLargeTopKUsesModelVocabulary) {
  check_request_filters(/*top_k=*/10000,
                        /*top_p=*/0.9F,
                        /*mixed=*/false,
                        /*kept_counts=*/{12, 12});
}

TEST(SamplerFilterMaskTest, MixedTopKOnlyUsesModelVocabulary) {
  check_request_filters(
      /*top_k=*/-1, /*top_p=*/1.F, /*mixed=*/true, /*kept_counts=*/{32, 8});
}

TEST(SamplerFilterMaskTest, JointFiltersPreserveCrossingTokenAndDisabledTopK) {
  if (Platform::is_npu() || Platform::is_mlu()) {
    GTEST_SKIP() << "Dedicated sampling kernels require device tensors";
  }
  torch::Tensor logits = torch::tensor({{0.05F, 0.2F, 0.6F, 0.15F},
                                        {0.05F, 0.2F, 0.6F, 0.15F},
                                        {0.05F, 0.2F, 0.6F, 0.15F}})
                             .log();
  const torch::Tensor original = logits.clone();
  const torch::Tensor alias = logits;
  apply_top_k_top_p(logits,
                    torch::tensor({2, -1, 0}, torch::kLong),
                    torch::tensor({0.8F, 0.75F, 1.F}));
  const torch::Tensor kept =
      torch::tensor({{0, 1, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 1}}, torch::kBool);
  const torch::Tensor expected = original.masked_fill(
      kept.logical_not(), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(torch::equal(logits, expected));
  EXPECT_TRUE(torch::equal(alias, expected));
}

TEST(SamplerFilterMaskTest, BFloat16TopPUsesFloat32Probabilities) {
  if (Platform::is_npu() || Platform::is_mlu()) {
    GTEST_SKIP() << "Dedicated sampling kernels require device tensors";
  }
  torch::Tensor logits =
      torch::tensor({{0.F, -2.953125F, -10.F}}).to(torch::kBFloat16);
  const torch::Tensor alias = logits;
  apply_top_k_top_p(logits,
                    torch::tensor(/*values=*/{2}, /*options=*/torch::kLong),
                    torch::tensor(/*values=*/{0.95F}));
  const torch::Tensor expected =
      torch::tensor({{0.F,
                      -std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}})
          .to(torch::kBFloat16);
  EXPECT_TRUE(torch::equal(logits, expected));
  EXPECT_TRUE(torch::equal(alias, expected));
}

TEST(SamplerFilterMaskTest, TopPFilteringPublishesTargetDistribution) {
  if (uses_device_random_sample() && Platform::device_count() == 0) {
    GTEST_SKIP() << "Random sampling backend device is unavailable";
  }

  SamplingParameters params = make_greedy_params(/*batch_size=*/1);
  params.do_sample = torch::ones({1}, torch::kBool);

  params.sampling_mode = SamplingMode::RANDOM;
  params.top_p = torch::tensor({0.95F});
  const torch::Device device = get_random_sample_device();
  params = params.to(device, torch::kFloat32);
  torch::Tensor logits = torch::tensor({{3.0F, 2.0F, 1.0F, 0.0F}}).to(device);
  SampleOutput output = sample(logits, params);

  const torch::Tensor target_logits = logits;
  const torch::Tensor target_probs = target_logits.softmax(/*dim=*/-1);
  const torch::Tensor expected_probs =
      torch::tensor(
          {{3.0F, 2.0F, 1.0F, -std::numeric_limits<float>::infinity()}})
          .to(device)
          .softmax(/*dim=*/-1);
  EXPECT_TRUE(torch::allclose(target_probs, expected_probs));
  EXPECT_LT(output.next_tokens.item<int64_t>(), 3);
}

}  // namespace
}  // namespace xllm
