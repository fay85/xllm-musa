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

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>

#include "framework/model/model_args.h"

namespace xllm {

enum class LinearHeadSharding : int8_t {
  ATTENTION_TP,
  CP_TP,
};

struct LinearAttentionPolicy {
  LinearHeadSharding head_sharding = LinearHeadSharding::ATTENTION_TP;
  bool requires_unsplit_kv = false;
};

// The split width must already resolve kv_split_size=0 through CP.
inline std::optional<std::string> validate_linear_attention_kv_split(
    const ModelArgs& args,
    const LinearAttentionPolicy& policy,
    int32_t effective_kv_split_size) {
  if (policy.requires_unsplit_kv && has_linear_attention_layers(args) &&
      effective_kv_split_size > 1) {
    return "Linear attention requires effective kv_split_size == 1; "
           "set --kv_split_size=1 (kv_split_size=0 follows cp_size)";
  }
  return std::nullopt;
}

inline int64_t linear_head_tp_size(const LinearAttentionPolicy& policy,
                                   int64_t attention_tp_size,
                                   int64_t cp_size) {
  switch (policy.head_sharding) {
    case LinearHeadSharding::ATTENTION_TP:
      return attention_tp_size;
    case LinearHeadSharding::CP_TP:
      return attention_tp_size * cp_size;
  }
  LOG(FATAL) << "Invalid linear head sharding mode";
  return 0;
}

}  // namespace xllm
