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

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <array>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

TEST(DeepseekV4StateGatherTest, ReadsPreviousPageAndWritesCurrentPage) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const c10::musa::MUSAStream stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor cache_cpu =
      torch::arange(4 * 128 * 16, cpu_options).reshape({4, 128, 16});
  torch::Tensor cache = cache_cpu.to(device);
  const std::array<torch::ScalarType, 2> integer_types{torch::kInt32,
                                                       torch::kInt64};
  for (const torch::ScalarType table_type : integer_types) {
    for (const torch::ScalarType position_type : integer_types) {
      for (const int64_t state_rows : {int64_t{8}, int64_t{128}}) {
        torch::Tensor slots =
            torch::empty({state_rows}, cache.options().dtype(torch::kInt64));
        torch::Tensor output = torch::empty({state_rows, 16}, cache.options());
        torch::Tensor table = torch::tensor({1, 2}, torch::kInt64)
                                  .view({1, 2})
                                  .to(device, table_type);
        torch::Tensor position =
            torch::tensor({128}, torch::kInt64).to(device, position_type);
        deepseek_v4_state_gather(cache,
                                 table,
                                 /*sequence_index=*/0,
                                 state_rows,
                                 /*block_size=*/128,
                                 slots,
                                 output,
                                 position);
        stream.synchronize();
        EXPECT_TRUE(torch::equal(output.to(torch::kCPU),
                                 cache_cpu[1].narrow(0, 0, state_rows)));
        EXPECT_TRUE(
            torch::equal(slots.to(torch::kCPU),
                         torch::arange(256, 256 + state_rows, torch::kInt64)));

        // After one step on the new page, both source and destination are
        // that page. Reuse the same device metadata storage as graph replay.
        position.fill_(129);
        deepseek_v4_state_gather(cache,
                                 table,
                                 /*sequence_index=*/0,
                                 state_rows,
                                 /*block_size=*/128,
                                 slots,
                                 output,
                                 position);
        stream.synchronize();
        EXPECT_TRUE(torch::equal(output.to(torch::kCPU),
                                 cache_cpu[2].narrow(0, 0, state_rows)));
      }
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::musa
