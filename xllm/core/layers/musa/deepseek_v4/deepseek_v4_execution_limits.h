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
#pragma once
#include <cstdint>

namespace xllm::layer {
// Qualified kernel/communication shapes, not arbitrary tuning defaults.
inline constexpr int32_t kDsv4PrefillReduceWorldSize = 8;
inline constexpr int64_t kDsv4PrefillReduceMinRows = 4096;
// The compact MoE path has a lower qualified minimum than ReduceScatter.
inline constexpr int64_t kDsv4CompactPrefillMinRows = 2048;
inline constexpr int64_t kDsv4PrefillReduceMaxRows = 16384;
inline constexpr int64_t kDsv4FlashHiddenSize = 4096;
inline constexpr int64_t kDsv4PrefillReduceWorkspaceRows = 2048;
inline constexpr int64_t kDsv4PrefillAllreduceWorkspaceRows = 1024;
inline constexpr int64_t kDsv4GroupedVerifySmallRows = 77;
inline constexpr int64_t kDsv4GroupedVerifyOptimizedMinRows = 63;
inline constexpr int64_t kDsv4GroupedVerifyLargeRows = 112;
inline constexpr int64_t kDsv4VerifyWorkspaceRows = 128;
inline constexpr int64_t kDsv4VerifyQueryWidth = 7;

inline bool dsv4_grouped_verify_minimum_supported(int64_t rows) {
  return rows == kDsv4GroupedVerifyOptimizedMinRows ||
         rows == kDsv4GroupedVerifySmallRows ||
         rows == kDsv4GroupedVerifyLargeRows;
}
}  // namespace xllm::layer
