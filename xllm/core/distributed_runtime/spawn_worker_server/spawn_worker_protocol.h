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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "core/framework/config/model_config.h"

namespace xllm::spawn_worker_protocol {

inline constexpr int32_t kArgumentCount = 42;
inline constexpr int32_t kMinimumArgumentCount = 33;
inline constexpr int32_t kIndexerCacheDtypeArgumentIndex = 33;
inline constexpr int32_t kEnableMtpDraftBodyTp1ArgumentIndex = 34;
inline constexpr int32_t kTextEncoderTpSizeArgumentIndex = 35;
inline constexpr int32_t kEnableProbabilisticDraftArgumentIndex = 36;
inline constexpr int32_t kKvSplitSizeArgumentIndex = 37;
inline constexpr int32_t kCommBufferOffsetArgumentIndex = 38;
inline constexpr int32_t kModelImplArgumentIndex = 39;
inline constexpr int32_t kVaeSizeArgumentIndex = 40;
inline constexpr int32_t kWorkerConfigArgumentIndex = 41;
inline constexpr char kDefaultIndexerCacheDtype[] = "auto";

inline std::optional<std::string> parse_indexer_cache_dtype(
    int32_t argc,
    char* const argv[]) {
  if (argc < kMinimumArgumentCount || argv == nullptr) {
    return std::nullopt;
  }

  if (argc == kMinimumArgumentCount) {
    return std::string(kDefaultIndexerCacheDtype);
  }

  if (argv[kIndexerCacheDtypeArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  return std::string(argv[kIndexerCacheDtypeArgumentIndex]);
}

inline bool parse_enable_probabilistic_draft(int32_t argc, char* const argv[]) {
  return argv != nullptr && argc > kEnableProbabilisticDraftArgumentIndex &&
         argv[kEnableProbabilisticDraftArgumentIndex] != nullptr &&
         std::string_view(argv[kEnableProbabilisticDraftArgumentIndex]) == "1";
}

inline std::optional<std::string> parse_model_impl(int32_t argc,
                                                   char* const argv[]) {
  if (argc < kMinimumArgumentCount || argv == nullptr) {
    return std::nullopt;
  }
  if (argc <= kModelImplArgumentIndex) {
    // Pre-protocol-worker images never carry this argument; they predate the
    // python model implementation entirely, so native is the only answer.
    return std::string("native");
  }
  if (argv[kModelImplArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  // Alias handling and the canonical values live in ModelConfig so the
  // worker accepts exactly what the Master constructor accepts.
  return ModelConfig::normalize_model_impl(argv[kModelImplArgumentIndex]);
}

}  // namespace xllm::spawn_worker_protocol
