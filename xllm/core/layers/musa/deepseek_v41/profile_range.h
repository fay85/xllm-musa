/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <ATen/record_function.h>
#include <torch/types.h>

#include <cstdint>
#include <string>
#include <string_view>

#include "core/layers/musa/deepseek_v41/light_timing.h"

namespace xllm::layer {

// A real native RecordFunction scope, captured by Kineto and the optional
// stack callback. Names/metadata are built only while a callback is active.
class DeepseekV41ProfileRange final {
 public:
  explicit DeepseekV41ProfileRange(std::string_view name,
                                   int64_t layer = -1,
                                   int64_t tokens = -1,
                                   int64_t context = -1,
                                   int64_t payload_bytes = -1,
                                   int64_t group_size = -1)
      : light_timing_(name, layer, tokens, context, payload_bytes, group_size) {
    if (!record_.isActive()) {
      return;
    }
    name_ = name;
    if (layer >= 0) {
      name_ += "/" + std::to_string(layer);
    }
    if (tokens >= 0) {
      name_ += "/q=" + std::to_string(tokens);
    }
    if (context >= 0) {
      name_ += "/kv=" + std::to_string(context);
    }
    record_.before(std::string_view(name_));
  }

 private:
  DeepseekV41LightTiming light_timing_;

  // The name must outlive RecordFunction's end callback.
  std::string name_;
  torch::RecordFunction record_{torch::RecordScope::USER_SCOPE};
};

}  // namespace xllm::layer
