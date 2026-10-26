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

#include "core/platform/graph_pool_ref.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {
namespace {

TEST(GraphPoolRefTest, UnmanagedResetDoesNotQueryPool) {
  GraphPoolRef ref;
  int32_t resets = 0;
  ref.reset([&]() { ++resets; },
            []() -> int64_t {
              ADD_FAILURE();
              return 0;
            },
            []() { ADD_FAILURE(); });
  EXPECT_EQ(resets, 1);
}

TEST(GraphPoolRefTest, MissingNativeReleaseIsCompensatedOnce) {
  GraphPoolRef ref;
  int64_t count = 2;
  int32_t releases = 0;
  ref.mark_captured();
  const auto reset = [&]() {
    ref.reset([]() {},
              [&]() { return count; },
              [&]() {
                --count;
                ++releases;
              });
  };
  reset();
  reset();
  EXPECT_EQ(count, 1);
  EXPECT_EQ(releases, 1);
}

TEST(GraphPoolRefTest, FixedSdkIsNotReleasedTwice) {
  GraphPoolRef ref;
  int64_t count = 2;
  int32_t releases = 0;
  ref.mark_captured();
  ref.reset([&]() { --count; }, [&]() { return count; }, [&]() { ++releases; });
  EXPECT_EQ(count, 1);
  EXPECT_EQ(releases, 0);
}

TEST(GraphPoolRefTest, PiecewiseSegmentsPreserveThePoolOwner) {
  for (const bool native_releases : {false, true}) {
    int64_t count = 1;
    std::vector<std::unique_ptr<GraphPoolRef>> segments;
    segments.reserve(4);
    for (int32_t i = 0; i < 4; ++i) {
      auto ref = std::make_unique<GraphPoolRef>();
      ++count;
      ref->mark_captured();
      segments.emplace_back(std::move(ref));
    }
    for (auto& ref : segments) {
      ref->reset([&]() { count -= native_releases ? 1 : 0; },
                 [&]() { return count; },
                 [&]() { --count; });
    }
    EXPECT_EQ(count, 1);
  }
}

TEST(GraphPoolRefTest, ResetAllowsAnotherCapture) {
  GraphPoolRef ref;
  int64_t count = 1;
  for (int32_t i = 0; i < 3; ++i) {
    ++count;
    ref.mark_captured();
    ref.reset([]() {}, [&]() { return count; }, [&]() { --count; });
    EXPECT_EQ(count, 1);
  }
}

TEST(GraphPoolRefTest, RejectsMissingPoolOwner) {
  GraphPoolRef ref;
  ref.mark_captured();
  EXPECT_DEATH(ref.reset([]() {}, []() { return 1; }, []() {}),
               "lost its user MemPool owner");
}

TEST(GraphPoolRefTest, RejectsUnrelatedReferenceChanges) {
  GraphPoolRef ref;
  ref.mark_captured();
  int64_t count = 3;
  EXPECT_DEATH(
      ref.reset([&]() { count -= 2; }, [&]() { return count; }, []() {}),
      "unrelated MemPool references");
}

TEST(GraphPoolRefTest, RejectsIneffectiveCompensation) {
  GraphPoolRef ref;
  ref.mark_captured();
  EXPECT_DEATH(ref.reset([]() {}, []() { return 2; }, []() {}),
               "exactly one MemPool reference");
}

TEST(GraphPoolRefTest, RejectsRecaptureWithoutReset) {
  GraphPoolRef ref;
  ref.mark_captured();
  EXPECT_DEATH(ref.mark_captured(), "reset before recapture");
}

}  // namespace
}  // namespace xllm
