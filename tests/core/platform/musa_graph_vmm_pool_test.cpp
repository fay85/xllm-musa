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

#include <Python.h>
#include <c10/core/StreamGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/kernels/musa/global_capture_instance.h"
#include "core/platform/musa_graph_handle.h"
#include "core/platform/musa_vmm_capture_pool.h"

namespace xllm {
namespace {

namespace allocator = c10::musa::MUSACachingAllocator;

class MusaGraphVmmPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::musa::device_count() == 0) {
      GTEST_SKIP() << "No MUSA device available";
    }
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    CHECK_EQ(
        PyRun_SimpleString("import torch\nimport torch_musa\ntorch.empty(1, "
                           "device='musa:0')\ntorch.musa.synchronize()\n"),
        0);

    c10::musa::set_device(0);
    allocator::init(c10::musa::device_count());
  }
};

TEST_F(MusaGraphVmmPoolTest, FullGraphResetReleasesExactlyOneReference) {
  MusaVmmCapturePool physical(/*device_index=*/0,
                              /*reserve_size=*/64 * 1024 * 1024);
  auto pool = physical.create_pool();
  const auto stream = c10::musa::getStreamFromPool(/*isHighPriority=*/false, 0);
  c10::StreamGuard guard(stream.unwrap());
  MusaGraphHandle graph;
  EXPECT_EQ(pool->use_count(), 1);
  graph.capture_begin(pool->id(), musaStreamCaptureModeThreadLocal, true);
  void* buffer = allocator::raw_alloc(1024);
  ASSERT_EQ(musaMemsetAsync(buffer, 37, 1024, stream.stream()), musaSuccess);
  graph.capture_end();
  allocator::raw_delete(buffer);
  EXPECT_EQ(pool->use_count(), 2);
  graph.replay();
  stream.synchronize();
  graph.reset();
  EXPECT_EQ(pool->use_count(), 1);
  graph.reset();
  EXPECT_EQ(pool->use_count(), 1);
}

TEST_F(MusaGraphVmmPoolTest, PiecewiseSegmentsReturnToThePoolOwner) {
  MusaVmmCapturePool physical(/*device_index=*/0,
                              /*reserve_size=*/64 * 1024 * 1024);
  auto pool = physical.create_pool();
  const auto stream = c10::musa::getStreamFromPool(/*isHighPriority=*/false, 0);
  c10::StreamGuard guard(stream.unwrap());
  auto& capture = runtime::musa::GlobalCaptureInstance::get_instance();
  capture.begin_capture(pool->id(), /*owns_user_pool_ref=*/true);
  for (int32_t i = 0; i < 3; ++i) {
    void* buffer = allocator::raw_alloc(1024);
    CHECK_EQ(musaMemsetAsync(buffer, i, 1024, stream.stream()), musaSuccess);
    allocator::raw_delete(buffer);
    if (i < 2) {
      capture.temporarily_end_graph();
      capture.temporarily_begin_graph();
    }
  }
  auto segments = capture.end_capture();
  EXPECT_EQ(segments->size(), 3);
  EXPECT_EQ(pool->use_count(), 4);
  kernel::musa::AttentionReplayParams params;
  segments->replay(params);
  stream.synchronize();
  segments.reset();
  EXPECT_EQ(pool->use_count(), 1);
}

class CapturedBuffers final {
 public:
  CapturedBuffers(MusaVmmCapturePool& physical,
                  size_t large_bytes,
                  uint8_t canary)
      : pool_(physical.create_pool()), canary_(canary) {
    const auto stream = c10::musa::getCurrentMUSAStream();
    graph_.capture_begin(pool_->id(), musaStreamCaptureModeThreadLocal, true);
    small_ = allocator::raw_alloc(1024);
    large_ = allocator::raw_alloc(large_bytes);
    CHECK_EQ(musaMemsetAsync(small_, canary_, 1024, stream.stream()),
             musaSuccess);
    CHECK_EQ(musaMemsetAsync(large_, canary_ + 1, large_bytes, stream.stream()),
             musaSuccess);
    graph_.capture_end();
    allocator::raw_delete(small_);
    allocator::raw_delete(large_);
  }

  c10::musa::MempoolId_t pool_id() { return pool_->id(); }

  void replay_and_check() {
    graph_.replay();
    c10::musa::getCurrentMUSAStream().synchronize();
    std::array<uint8_t, 128> small_host{};
    std::array<uint8_t, 128> large_host{};
    ASSERT_EQ(musaMemcpy(small_host.data(),
                         small_,
                         small_host.size(),
                         musaMemcpyDeviceToHost),
              musaSuccess);
    ASSERT_EQ(musaMemcpy(large_host.data(),
                         large_,
                         large_host.size(),
                         musaMemcpyDeviceToHost),
              musaSuccess);
    for (const uint8_t value : small_host) {
      EXPECT_EQ(value, canary_);
    }
    for (const uint8_t value : large_host) {
      EXPECT_EQ(value, canary_ + 1);
    }
  }

 private:
  // Native handles and all captured uses are destroyed before this owner.
  std::unique_ptr<c10::musa::MemPool> pool_;
  const uint8_t canary_;
  void* small_ = nullptr;  // Non-owning addresses kept alive by the graph pool.
  void* large_ = nullptr;
  MusaGraphHandle graph_;
};

TEST_F(MusaGraphVmmPoolTest, DifferentLayoutsAndRecapturesGetFreshPools) {
  MusaVmmCapturePool physical(/*device_index=*/0,
                              /*reserve_size=*/64 * 1024 * 1024);
  const auto stream = c10::musa::getStreamFromPool(/*isHighPriority=*/false, 0);
  c10::StreamGuard guard(stream.unwrap());
  // Same token bucket, different allocation layouts; the second capture grows
  // its native pool after obtaining a small block. Fresh pool IDs prevent old
  // cached segments from being mixed with allocations from the new VA.
  CapturedBuffers balanced(physical,
                           /*large_bytes=*/8 * 1024 * 1024,
                           /*canary=*/23);
  CapturedBuffers skewed(physical,
                         /*large_bytes=*/24 * 1024 * 1024,
                         /*canary=*/81);
  EXPECT_NE(balanced.pool_id(), skewed.pool_id());
  for (int32_t i = 0; i < 3; ++i) {
    balanced.replay_and_check();
    skewed.replay_and_check();
  }
  auto recaptured = std::make_unique<CapturedBuffers>(
      physical, /*large_bytes=*/8 * 1024 * 1024, /*canary=*/42);
  const auto old_id = recaptured->pool_id();
  recaptured->replay_and_check();
  recaptured.reset();
  recaptured = std::make_unique<CapturedBuffers>(
      physical, /*large_bytes=*/8 * 1024 * 1024, /*canary=*/43);
  EXPECT_NE(recaptured->pool_id(), old_id);
  recaptured->replay_and_check();
  balanced.replay_and_check();
}

}  // namespace
}  // namespace xllm
