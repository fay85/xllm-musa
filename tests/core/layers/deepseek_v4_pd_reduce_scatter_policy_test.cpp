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

#include "core/layers/musa/deepseek_v4/deepseek_v4_pd_reduce_scatter_policy.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV4PdReduceScatterPolicyTest, AcceptsValidatedBf16PrefillShape) {
  EXPECT_TRUE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/8192, /*local_rows=*/1024, /*world=*/8,
      /*prefill=*/true, /*bf16=*/true, /*skip_gather=*/false));
}

TEST(DeepseekV4PdReduceScatterPolicyTest, RejectsUnsafeShapesAndModes) {
  EXPECT_FALSE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/4096, /*local_rows=*/512, /*world=*/8,
      /*prefill=*/true, /*bf16=*/true, /*skip_gather=*/false));
  EXPECT_FALSE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/8192, /*local_rows=*/1024, /*world=*/4,
      /*prefill=*/true, /*bf16=*/true, /*skip_gather=*/false));
  EXPECT_FALSE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/8192, /*local_rows=*/1024, /*world=*/8,
      /*prefill=*/false, /*bf16=*/true, /*skip_gather=*/false));
  EXPECT_FALSE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/8192, /*local_rows=*/1024, /*world=*/8,
      /*prefill=*/true, /*bf16=*/false, /*skip_gather=*/false));
  EXPECT_FALSE(dsv4_pd_reduce_scatter_eligible(
      /*rows=*/8192, /*local_rows=*/1024, /*world=*/8,
      /*prefill=*/true, /*bf16=*/true, /*skip_gather=*/true));
}

}  // namespace
}  // namespace xllm::layer
