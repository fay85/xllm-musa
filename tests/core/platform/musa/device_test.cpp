/* Copyright 2026 The xLLM Authors.

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

#include "core/platform/device.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/platform/platform.h"

namespace xllm {
namespace {

TEST(MusaDeviceTest, InitializesNativeRuntimeBeforeTensorAllocation) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  // Registration and allocator initialization must also be safe to repeat.
  device.init_device_context();

  torch::Tensor tensor = torch::ones(
      {4},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  EXPECT_TRUE(torch::equal(tensor.cpu(), torch::ones({4})));

  // FlashinferWorkspace also allocates pinned host memory immediately after
  // initializing its device buffers.
  torch::Tensor pinned = torch::empty(
      {16}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
  EXPECT_TRUE(pinned.is_pinned());
}

}  // namespace
}  // namespace xllm
