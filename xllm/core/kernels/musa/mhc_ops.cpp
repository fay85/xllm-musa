/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kernels/musa/mhc_ops.h"

#include <glog/logging.h>

#include <vector>

namespace xllm::kernel::musa {

torch::Tensor mhc_expand(const torch::Tensor& hidden_states, int64_t hc_mult) {
  CHECK(hidden_states.defined() && hidden_states.dim() >= 2);
  CHECK_EQ(hidden_states.device().type(), torch::kPrivateUse1);
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK_GT(hc_mult, 0);
  CHECK_LE(hc_mult, 12);
  std::vector<int64_t> output_shape = hidden_states.sizes().vec();
  output_shape.insert(output_shape.end() - 1, hc_mult);
  return hidden_states.unsqueeze(-2).expand(output_shape).contiguous();
}

}  // namespace xllm::kernel::musa
