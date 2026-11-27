/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/kernels/mlu/dflash2_sampling.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm {

namespace {

std::pair<torch::Tensor, torch::Tensor> reference_sparse_paths(
    const torch::Tensor& ids,
    const torch::Tensor& log_probs,
    const torch::Tensor& noise,
    const torch::Tensor& do_sample,
    bool use_random_sampling_for_batch) {
  const int64_t batch = ids.size(0);
  const int64_t steps = ids.size(1);
  const int64_t top_k = ids.size(2);
  torch::Tensor tokens = torch::empty({batch, steps}, ids.options());
  torch::Tensor probs =
      torch::empty({batch, steps, top_k}, log_probs.options());
  torch::Tensor previous = torch::zeros({batch}, ids.options());
  torch::Tensor offsets = torch::arange(batch, ids.options()) * (steps * top_k);
  torch::Tensor rows = log_probs.contiguous().view({-1, top_k});
  for (int64_t step = 0; step < steps; ++step) {
    torch::Tensor row = rows.index_select(
        /*dim=*/0, previous + (offsets + step * top_k));
    torch::Tensor selected =
        (row + noise.select(/*dim=*/1, /*index=*/step)).argmax(/*dim=*/-1);
    tokens.select(/*dim=*/1, /*index=*/step)
        .copy_(ids.select(/*dim=*/1, /*index=*/step)
                   .gather(/*dim=*/1, selected.view({-1, 1}))
                   .view({-1}));
    torch::Tensor step_logits = row;
    if (!use_random_sampling_for_batch) {
      torch::Tensor onehot = torch::full_like(
          step_logits, /*fill_value=*/-std::numeric_limits<float>::infinity());
      onehot.scatter_(/*dim=*/1, selected.unsqueeze(/*dim=*/1), /*value=*/0.0);
      step_logits =
          torch::where(do_sample.view({batch, 1}), step_logits, onehot);
    }
    probs.select(/*dim=*/1, /*index=*/step).copy_(step_logits);
    previous = selected;
  }
  return {std::move(tokens), std::move(probs)};
}

class MluDFlash2SamplingGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.set_device();
    device_.init_device_context();
    inputs_ = {
        torch::arange(12, torch::kLong).view({1, 3, 4}).to(device_.unwrap()),
        torch::full(/*size=*/{1, 3, 4, 4}, /*fill_value=*/-std::log(/*x=*/4.0F))
            .to(device_.unwrap()),
        torch::zeros({1, 3, 4}).to(device_.unwrap()),
        torch::ones({1}, torch::kBool).to(device_.unwrap())};
  }

  void expect_rejected(const std::array<torch::Tensor, 4>& inputs) {
    std::array<torch::Tensor, 4> original;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].defined()) {
        continue;
      }
      original[i] = inputs[i].cpu().clone();
    }
    EXPECT_FALSE(kernel::mlu::supports_dflash2_sampling(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        /*use_random_sampling_for_batch=*/false,
        /*greedy_only=*/false));
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].defined()) {
        continue;
      }
      EXPECT_TRUE(torch::equal(inputs[i].cpu(), original[i]));
    }
  }

  Device device_{/*device_index=*/0};
  std::array<torch::Tensor, 4> inputs_;
};

TEST_F(MluDFlash2SamplingGuardTest, MissingOrHostInputsAreRejected) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    SCOPED_TRACE(i);
    auto missing = inputs_;
    missing[i] = torch::Tensor();
    expect_rejected(missing);
    auto host = inputs_;
    host[i] = host[i].cpu();
    expect_rejected(host);
  }
  auto invalid_rank = inputs_;
  invalid_rank[0] = invalid_rank[0].flatten();
  expect_rejected(invalid_rank);
}

TEST_F(MluDFlash2SamplingGuardTest, RandomSamplingDoesNotRequireDeviceMask) {
  for (const torch::Tensor& mask : {torch::Tensor(), inputs_[3].cpu()}) {
    ASSERT_TRUE(kernel::mlu::supports_dflash2_sampling(
        inputs_[0],
        inputs_[1],
        inputs_[2],
        mask,
        /*use_random_sampling_for_batch=*/true,
        /*greedy_only=*/false));
    auto result = kernel::mlu::sample_dflash2_paths(
        inputs_[0],
        inputs_[1],
        inputs_[2],
        mask,
        /*use_random_sampling_for_batch=*/true);
    EXPECT_TRUE(
        torch::equal(result.first.cpu(),
                     torch::tensor({0, 4, 8}, torch::kLong).view({1, 3})));
    EXPECT_TRUE(torch::allclose(
        result.second.cpu(),
        torch::full(/*size=*/{1, 3, 4}, /*fill_value=*/-std::log(/*x=*/4.0F))));
  }
}

TEST_F(MluDFlash2SamplingGuardTest, OtherDeviceInputsAreRejected) {
  if (Platform::device_count() < 2) {
    GTEST_SKIP() << "Requires two MLU devices";
  }
  Device other_device(/*device_index=*/1);
  other_device.set_device();
  other_device.init_device_context();
  for (size_t i = 0; i < inputs_.size(); ++i) {
    SCOPED_TRACE(i);
    auto inputs = inputs_;
    inputs[i] = inputs[i].cpu().to(other_device.unwrap());
    expect_rejected(inputs);
  }
  device_.set_device();
}

TEST(MluDFlash2RandomPathKernelTest,
     ExactSparseProbabilitiesWithStridedInputs) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(42);
  struct Shape {
    int64_t batch;
    int64_t steps;
    int64_t candidates;
  };
  const std::vector<Shape> shapes = {{15, 3, 5}, {33, 7, 33}};
  const std::vector<torch::ScalarType> id_types = {torch::kInt32, torch::kLong};
  const std::vector<bool> modes = {true, false};
  using ISlice = torch::indexing::Slice;
  for (const Shape& shape : shapes) {
    for (torch::ScalarType id_type : id_types) {
      for (bool use_random_sampling_for_batch : modes) {
        SCOPED_TRACE(::testing::Message()
                     << "B=" << shape.batch << " S=" << shape.steps
                     << " K=" << shape.candidates << " dtype=" << id_type
                     << " random=" << use_random_sampling_for_batch);
        torch::Tensor ids =
            torch::arange(shape.batch * shape.steps * shape.candidates * 2,
                          torch::TensorOptions().dtype(id_type))
                .view({shape.batch, shape.steps, shape.candidates * 2})
                .to(device.unwrap())
                .index(
                    {ISlice(), ISlice(), ISlice(0, torch::indexing::None, 2)});
        torch::Tensor log_probs =
            torch::log_softmax(torch::randn({shape.batch,
                                             shape.steps,
                                             shape.candidates,
                                             shape.candidates}),
                               /*dim=*/-1)
                .to(device.unwrap());
        // Repeated/interleaved backing storage preserves normalized values
        // while exposing a non-contiguous last dimension to the kernel.
        log_probs = log_probs.repeat_interleave(/*repeats=*/2, /*dim=*/3)
                        .index({ISlice(),
                                ISlice(),
                                ISlice(),
                                ISlice(0, torch::indexing::None, 2)});
        torch::Tensor noise =
            torch::randn({shape.batch, shape.steps, shape.candidates * 2})
                .to(device.unwrap())
                .index(
                    {ISlice(), ISlice(), ISlice(0, torch::indexing::None, 2)});
        torch::Tensor mask =
            (torch::arange(shape.batch * 2, torch::kLong) % 4 == 0)
                .to(device.unwrap())
                .index({ISlice(0, torch::indexing::None, 2)});
        torch::Tensor input_mask =
            use_random_sampling_for_batch ? torch::Tensor() : mask;
        auto expected = reference_sparse_paths(
            ids, log_probs, noise, input_mask, use_random_sampling_for_batch);
        auto result = kernel::mlu::sample_dflash2_paths(
            ids, log_probs, noise, input_mask, use_random_sampling_for_batch);
        EXPECT_EQ(result.first.scalar_type(), id_type);
        EXPECT_TRUE(torch::equal(result.first.cpu(), expected.first.cpu()));
        EXPECT_TRUE(torch::equal(result.second.cpu(), expected.second.cpu()));
        EXPECT_TRUE(torch::allclose(result.second.exp().sum(/*dim=*/-1).cpu(),
                                    torch::ones({shape.batch, shape.steps}),
                                    /*rtol=*/1e-5,
                                    /*atol=*/1e-6));
      }
    }
  }
}

TEST(MluDFlash2RandomPathKernelTest, CompactPathsMatchReferenceAcrossTiles) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(2026);
  constexpr int64_t kSteps = 3;
  const auto options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kFloat32);
  for (int64_t batch : {65, 129}) {
    torch::Tensor ids =
        torch::arange(batch * kSteps * 16, options.dtype(torch::kLong))
            .view({batch, kSteps, 16});
    torch::Tensor logs = torch::log_softmax(
        torch::randn({batch, kSteps, 16, 16}, options), /*dim=*/-1);
    torch::Tensor noise = torch::randn({batch, kSteps, 16}, options);
    torch::Tensor mask =
        torch::arange(batch, options.dtype(torch::kLong)).remainder(2).eq(0);
    for (bool all_random : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "batch=" << batch << " random=" << all_random);
      torch::Tensor active_mask = all_random ? torch::Tensor() : mask;
      auto expected =
          reference_sparse_paths(ids, logs, noise, active_mask, all_random);
      auto actual = kernel::mlu::sample_dflash2_paths(
          ids, logs, noise, active_mask, all_random);
      EXPECT_TRUE(torch::equal(actual.first.cpu(), expected.first.cpu()));
      EXPECT_TRUE(torch::equal(actual.second.cpu(), expected.second.cpu()));
    }
  }
}

TEST(MluDFlash2RandomPathKernelTest, FirstTieNanAndInfinityMatchTorch) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::Tensor ids =
      torch::arange(60, torch::kLong).view({4, 3, 5}).to(device.unwrap());
  torch::Tensor scores = torch::zeros({4, 3, 5, 5});
  scores.select(/*dim=*/0, /*index=*/1)
      .fill_(-std::numeric_limits<float>::infinity());
  scores.select(/*dim=*/0, /*index=*/2)
      .select(/*dim=*/2, /*index=*/2)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  scores.select(/*dim=*/0, /*index=*/2)
      .select(/*dim=*/2, /*index=*/3)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  scores.select(/*dim=*/0, /*index=*/3)
      .select(/*dim=*/2, /*index=*/1)
      .fill_(std::numeric_limits<float>::infinity());
  scores = scores.to(device.unwrap());
  torch::Tensor noise = torch::zeros({4, 3, 5}, scores.options());
  torch::Tensor mask =
      torch::tensor({true, false, true, false}).to(device.unwrap());
  const std::vector<bool> modes = {true, false};
  for (bool use_random_sampling_for_batch : modes) {
    auto expected = reference_sparse_paths(
        ids, scores, noise, mask, use_random_sampling_for_batch);
    auto result = kernel::mlu::sample_dflash2_paths(
        ids, scores, noise, mask, use_random_sampling_for_batch);
    EXPECT_TRUE(torch::equal(result.first.cpu(), expected.first.cpu()));
    EXPECT_TRUE(torch::allclose(result.second.cpu(),
                                expected.second.cpu(),
                                /*rtol=*/0,
                                /*atol=*/0,
                                /*equal_nan=*/true));
  }
}

}  // namespace
}  // namespace xllm
