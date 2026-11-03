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
#include <charconv>
#include <cstddef>
#include <optional>
#include <string_view>

namespace xllm::musa {
inline constexpr size_t kDsv4DefaultPrefixSnapshotPercent = 80;
inline constexpr size_t kDsv4PercentDenominator = 100;

inline std::optional<size_t> parse_deepseek_v4_prefix_snapshot_percent(
    const char* value) {
  if (value == nullptr) {
    return kDsv4DefaultPrefixSnapshotPercent;
  }
  const std::string_view text(value);
  size_t percent = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), percent);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      percent == 0 || percent >= kDsv4PercentDenominator) {
    return std::nullopt;
  }
  return percent;
}

inline size_t deepseek_v4_prefix_snapshot_tokens(size_t prompt_tokens,
                                                 size_t block_size,
                                                 size_t percent) {
  if (block_size == 0 || percent == 0 || percent >= kDsv4PercentDenominator) {
    return 0;
  }
  // Divide before multiplying so even the largest size_t prompt cannot
  // overflow.
  const size_t unaligned =
      (prompt_tokens / kDsv4PercentDenominator) * percent +
      ((prompt_tokens % kDsv4PercentDenominator) * percent) /
          kDsv4PercentDenominator;
  return (unaligned / block_size) * block_size;
}
}  // namespace xllm::musa
