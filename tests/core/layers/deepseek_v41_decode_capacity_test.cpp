// Copyright 2026 The xLLM Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include "core/layers/musa/deepseek_v41/decode_capacity.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV41DecodeCapacity, PreservesLegacyAndFallbackBuckets) {
  for (int64_t end = 1; end <= 4096; ++end) {
    EXPECT_EQ(deepseek_v41_decode_capacity(end, false),
              end <= 512 ? 512 : 4096);
    const int64_t capacity = deepseek_v41_decode_capacity(end, true);
    EXPECT_GE(capacity, end);
    if (end <= 512 || end > 3072) {
      EXPECT_EQ(capacity, deepseek_v41_decode_capacity(end, false));
    } else {
      EXPECT_EQ(capacity, 3072);
    }
  }
}

TEST(DeepseekV41DecodeCapacity, SizesUsingLastVerifyRowNotAcceptedPrefix) {
  for (int64_t rows = 1; rows <= 6; ++rows) {
    for (int64_t begin = 3065; begin <= 3073; ++begin) {
      const int64_t end = begin + rows;
      const int64_t capacity = deepseek_v41_decode_capacity(end, true);
      EXPECT_EQ(capacity, end <= 3072 ? 3072 : 4096);
      for (int64_t ratio : {1, 2}) {
        for (int64_t block_size : {64, 128}) {
          const int64_t pages =
              (capacity / ratio + block_size - 1) / block_size;
          EXPECT_GE(pages * block_size, end / ratio);
        }
      }
    }
  }
}
}  // namespace
}  // namespace xllm::layer
