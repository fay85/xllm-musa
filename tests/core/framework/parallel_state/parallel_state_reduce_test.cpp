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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {
namespace {

struct ReduceWorkState {
  int32_t launches = 0;
  int32_t waits = 0;
  bool fail_on_wait = false;
};

class DeferredAddWork final : public c10d::Work {
 public:
  DeferredAddWork(torch::Tensor tensor, std::shared_ptr<ReduceWorkState> state)
      : Work(/*rank=*/0, c10d::OpType::ALLREDUCE),
        tensor_(std::move(tensor)),
        state_(std::move(state)) {}

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    ++state_->waits;
    if (state_->fail_on_wait) {
      throw std::runtime_error("injected collective failure");
    }
    tensor_.add_(10);
    return true;
  }

 private:
  torch::Tensor tensor_;
  std::shared_ptr<ReduceWorkState> state_;
};

class DeferredAddProcessGroup final : public ProcessGroup {
 public:
  DeferredAddProcessGroup(int32_t world_size,
                          std::shared_ptr<ReduceWorkState> state)
      : ProcessGroup(/*rank=*/0, world_size, torch::Device(torch::kCPU)),
        state_(std::move(state)) {}

  c10::intrusive_ptr<c10d::Work> allreduce_async(
      torch::Tensor& input) override {
    ++state_->launches;
    return c10::make_intrusive<DeferredAddWork>(input, state_);
  }

 private:
  std::shared_ptr<ReduceWorkState> state_;
};

TEST(ParallelStateReduceTest, NullGroupPreservesStorageWithoutWork) {
  torch::Tensor input = torch::arange(6, torch::kFloat32);
  auto ctx = parallel_state::launch_reduce(input, /*process_group=*/nullptr);
  EXPECT_FALSE(ctx.work.defined());
  torch::Tensor output = parallel_state::finish_reduce(std::move(ctx));
  EXPECT_EQ(output.data_ptr(), input.data_ptr());
  EXPECT_TRUE(torch::equal(output, input));
}

TEST(ParallelStateReduceTest, SingleRankDoesNotLaunchCollective) {
  auto state = std::make_shared<ReduceWorkState>();
  DeferredAddProcessGroup group(/*world_size=*/1, state);
  torch::Tensor input = torch::ones({2, 3});
  auto ctx = parallel_state::launch_reduce(input, &group);
  EXPECT_FALSE(ctx.work.defined());
  EXPECT_EQ(state->launches, 0);
  EXPECT_EQ(state->waits, 0);
  EXPECT_EQ(parallel_state::finish_reduce(std::move(ctx)).data_ptr(),
            input.data_ptr());
}

TEST(ParallelStateReduceTest, FallbackIsDeferredAndWaitedOnce) {
  auto state = std::make_shared<ReduceWorkState>();
  DeferredAddProcessGroup group(/*world_size=*/2, state);
  torch::Tensor input = torch::arange(6, torch::kFloat32);
  torch::Tensor expected = input + 10;
  auto ctx = parallel_state::launch_reduce(input, &group);
  ASSERT_TRUE(ctx.work.defined());
  EXPECT_EQ(state->launches, 1);
  EXPECT_EQ(state->waits, 0);
  EXPECT_FALSE(torch::equal(ctx.tensor, expected));
  torch::Tensor output = parallel_state::finish_reduce(std::move(ctx));
  EXPECT_EQ(state->waits, 1);
  EXPECT_EQ(output.data_ptr(), input.data_ptr());
  EXPECT_TRUE(torch::equal(output, expected));
}

TEST(ParallelStateReduceTest, NoncontiguousFallbackDoesNotOverwriteSource) {
  auto state = std::make_shared<ReduceWorkState>();
  DeferredAddProcessGroup group(/*world_size=*/2, state);
  torch::Tensor input =
      torch::arange(12, torch::kFloat32).reshape({3, 4}).transpose(0, 1);
  ASSERT_FALSE(input.is_contiguous());
  torch::Tensor original = input.clone();
  auto ctx = parallel_state::launch_reduce(input, &group);
  EXPECT_TRUE(ctx.tensor.is_contiguous());
  EXPECT_NE(ctx.tensor.data_ptr(), input.data_ptr());
  torch::Tensor output = parallel_state::finish_reduce(std::move(ctx));
  EXPECT_TRUE(torch::equal(output, original + 10));
  EXPECT_TRUE(torch::equal(input, original));
}

TEST(ParallelStateReduceTest, ContextRetainsInputUntilDeferredConsumption) {
  auto state = std::make_shared<ReduceWorkState>();
  DeferredAddProcessGroup group(/*world_size=*/2, state);
  torch::Tensor input = torch::arange(8, torch::kFloat32);
  torch::Tensor expected = input + 10;
  auto ctx = parallel_state::launch_reduce(input, &group);
  input = torch::Tensor();
  torch::Tensor output = parallel_state::finish_reduce(std::move(ctx));
  EXPECT_TRUE(torch::equal(output, expected));
  EXPECT_EQ(state->waits, 1);
}

TEST(ParallelStateReduceTest, CollectiveFailurePropagates) {
  auto state = std::make_shared<ReduceWorkState>();
  state->fail_on_wait = true;
  DeferredAddProcessGroup group(/*world_size=*/2, state);
  auto ctx = parallel_state::launch_reduce(torch::ones({2, 4}), &group);
  EXPECT_THROW(parallel_state::finish_reduce(std::move(ctx)),
               std::runtime_error);
  EXPECT_EQ(state->launches, 1);
  EXPECT_EQ(state->waits, 1);
}

}  // namespace
}  // namespace xllm
