/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include "core/kernels/musa/attention_runner.h"
#include "core/kernels/musa/piecewise_graphs.h"

namespace xllm::runtime::cuda {
namespace test {

TEST(PiecewiseGraphsTest, EmptyReplayIsNoOp) {
  PiecewiseGraphs graphs;
  EXPECT_TRUE(graphs.empty());
  EXPECT_EQ(graphs.size(), 0u);
  EXPECT_EQ(graphs.num_graphs(), 0u);
  EXPECT_EQ(graphs.num_runners(), 0u);

  ::xllm::kernel::cuda::AttentionReplayParams params;
  params.actual_num_tokens = 0;
  graphs.replay(params);
}

TEST(PiecewiseGraphsTest, RunnerOnlyPiecewiseIsNonEmpty) {
  PiecewiseGraphs graphs;
  ::xllm::kernel::cuda::AttentionRunner runner;
  graphs.add_attention_runner(std::move(runner));

  EXPECT_FALSE(graphs.empty());
  EXPECT_EQ(graphs.size(), 1u);
  EXPECT_EQ(graphs.num_graphs(), 0u);
  EXPECT_EQ(graphs.num_runners(), 1u);
}

}  // namespace test
}  // namespace xllm::runtime::cuda
