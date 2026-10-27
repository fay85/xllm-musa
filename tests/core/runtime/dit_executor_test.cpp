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

#include "core/runtime/dit_executor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "core/common/metrics.h"

namespace xllm {
namespace {

class RecordingDiTModel final : public DiTModel {
 public:
  explicit RecordingDiTModel(bool fail = false) : fail_(fail) {}

  DiTForwardOutput forward(const DiTForwardInput& input) override {
    ++calls_;
    eager_at_forward_ = COUNTER_VALUE(num_model_execution_total_eager);
    last_batch_size_ = input.batch_size;
    if (fail_) {
      throw std::runtime_error("injected forward failure");
    }
    DiTForwardOutput output;
    output.text_output = {"generated text"};
    return output;
  }

  torch::Device device() const override { return torch::Device(torch::kCPU); }
  const torch::TensorOptions& options() const override { return options_; }
  void load_model(std::unique_ptr<DiTModelLoader> /*loader*/) override {}

  int32_t calls_ = 0;
  int32_t last_batch_size_ = 0;
  double eager_at_forward_ = 0;

 private:
  bool fail_;
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
};

TEST(DiTExecutorMetricsTest, CountsTopLevelForwardNotRequestsOrDenoiseSteps) {
  RecordingDiTModel model;
  DiTExecutor executor(&model, runtime::Options());
  DiTForwardInput input;
  input.batch_size = 3;
  input.generation_params.num_inference_steps = 20;
  const double before = COUNTER_VALUE(num_model_execution_total_eager);

  const DiTForwardOutput output = executor.forward(input);

  EXPECT_EQ(COUNTER_VALUE(num_model_execution_total_eager), before + 1);
  EXPECT_EQ(model.eager_at_forward_, before + 1);
  EXPECT_EQ(model.calls_, 1);
  EXPECT_EQ(model.last_batch_size_, 3);
  ASSERT_EQ(output.text_output.size(), 1u);
  EXPECT_EQ(output.text_output[0], "generated text");
}

TEST(DiTExecutorMetricsTest, FailedForwardStillCountsOneAttempt) {
  RecordingDiTModel model(/*fail=*/true);
  DiTExecutor executor(&model, runtime::Options());
  const double before = COUNTER_VALUE(num_model_execution_total_eager);

  EXPECT_THROW(executor.forward(DiTForwardInput()), std::runtime_error);

  EXPECT_EQ(COUNTER_VALUE(num_model_execution_total_eager), before + 1);
  EXPECT_EQ(model.eager_at_forward_, before + 1);
  EXPECT_EQ(model.calls_, 1);
}

}  // namespace
}  // namespace xllm
