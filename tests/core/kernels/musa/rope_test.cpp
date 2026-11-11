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
#include <torch/torch.h>

#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::kernel {
namespace {

TEST(MusaRopeTest, MatchesReferenceWithAndWithoutKey) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::Device target = device.unwrap();
  const torch::Tensor positions = torch::tensor({1, 0}, torch::kInt64);
  const torch::Tensor cache =
      torch::tensor({{1.0f, 0.5f, 0.25f, -0.5f}, {0.75f, 0.25f, -0.25f, 1.0f}});
  const torch::Tensor selected = cache.index_select(0, positions);
  const torch::Tensor cos = selected.slice(-1, 0, 2).unsqueeze(1);
  const torch::Tensor sin = selected.slice(-1, 2, 4).unsqueeze(1);
  const auto reference = [&](const torch::Tensor& input) {
    const torch::Tensor x = input.slice(-1, 0, 2);
    const torch::Tensor y = input.slice(-1, 2, 4);
    return torch::cat({x * cos - y * sin, y * cos + x * sin}, -1);
  };
  const torch::Tensor query =
      torch::arange(16, torch::kFloat32).view({2, 2, 4});
  const torch::Tensor key =
      torch::arange(8, torch::kFloat32).view({2, 1, 4}) + 0.5;
  for (const bool with_key : {false, true}) {
    SCOPED_TRACE(with_key);
    RotaryParams params;
    params.q = query.to(target);
    if (with_key) {
      params.k = key.to(target);
    }
    params.position_ids = positions.to(target);
    params.precomputed_cos_sin = cache.to(target);
    params.interleaved = false;
    apply_rotary(params);
    EXPECT_TRUE(torch::equal(params.q.cpu(), reference(query)));
    if (with_key) {
      EXPECT_TRUE(torch::equal(params.k.cpu(), reference(key)));
    } else {
      EXPECT_FALSE(params.k.defined());
    }
  }
}

}  // namespace
}  // namespace xllm::kernel
