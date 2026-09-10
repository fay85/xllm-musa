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

#include "core/kernels/musa/musa_tvmffi_stream.h"

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <array>
#include <cstdint>

namespace xllm::kernel::musa {
namespace {

constexpr std::array<int32_t, 2> kDeviceTypes{2, 12};

class TvmffiStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device unavailable";
    }
    for (size_t index = 0; index < kDeviceTypes.size(); ++index) {
      original_streams_[index] =
          TVMFFIEnvGetStream(kDeviceTypes[index], /*device_id=*/0);
    }
    initialized_ = true;
  }

  void TearDown() override {
    if (!initialized_) {
      return;
    }
    for (size_t index = 0; index < kDeviceTypes.size(); ++index) {
      void* previous = nullptr;
      EXPECT_EQ(TVMFFIEnvSetStream(kDeviceTypes[index],
                                   /*device_id=*/0,
                                   original_streams_[index],
                                   &previous),
                0);
    }
  }

  void expect_bound_stream(const c10::musa::MUSAStream& stream) {
    const torch::Device device(torch::kPrivateUse1, 0);
    bind_tvmffi_stream_to_current_torch_stream(device);
    for (const int32_t device_type : kDeviceTypes) {
      EXPECT_EQ(TVMFFIEnvGetStream(device_type, /*device_id=*/0),
                reinterpret_cast<void*>(stream.stream()));
    }
    EXPECT_EQ(c10::musa::getCurrentMUSAStream(/*device_index=*/0), stream);
  }

 private:
  bool initialized_ = false;
  std::array<void*, 2> original_streams_{};
};

TEST_F(TvmffiStreamTest, PreservesDefaultStreamAfterPooledBinding) {
  const c10::musa::MUSAStream pooled =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  {
    c10::StreamGuard stream_guard(pooled.unwrap());
    expect_bound_stream(pooled);
  }
  const c10::musa::MUSAStream default_stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard stream_guard(default_stream.unwrap());
  expect_bound_stream(default_stream);
}

TEST_F(TvmffiStreamTest, FollowsNonDefaultStreamChanges) {
  const c10::musa::MUSAStream first =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  const c10::musa::MUSAStream second =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  {
    c10::StreamGuard stream_guard(first.unwrap());
    expect_bound_stream(first);
  }
  {
    c10::StreamGuard stream_guard(second.unwrap());
    expect_bound_stream(second);
  }
}

TEST_F(TvmffiStreamTest, PreservesCaptureStreamBinding) {
  const c10::musa::MUSAStream stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  c10::StreamGuard stream_guard(stream.unwrap());
  ASSERT_EQ(
      musaStreamBeginCapture(stream.stream(), musaStreamCaptureModeThreadLocal),
      musaSuccess);
  expect_bound_stream(stream);
  musaGraph_t graph = nullptr;
  ASSERT_EQ(musaStreamEndCapture(stream.stream(), &graph), musaSuccess);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(musaGraphDestroy(graph), musaSuccess);
}

}  // namespace
}  // namespace xllm::kernel::musa
