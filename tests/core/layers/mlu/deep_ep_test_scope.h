/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <framework/core/device.h>

#include "layers/common/deep_ep.h"

namespace xllm::test {

// Declare after ParallelArgs/options and before models. DeepEP retains those
// references, and its process singleton must not survive their scenario scope.
class DeepEpTestScope final {
 public:
  DeepEpTestScope() = default;
  DeepEpTestScope(const DeepEpTestScope&) = delete;
  DeepEpTestScope& operator=(const DeepEpTestScope&) = delete;

  ~DeepEpTestScope() {
    torch_mlu::synchronize();
    layer::DeepEPManager::destroy();
  }
};

}  // namespace xllm::test
