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

#include <cmath>
#include <limits>

#include "core/kernels/musa/kda_ops.h"
#include "core/kernels/musa/musa_tensor_validation.h"

namespace xllm::kernel::musa::detail {

inline void validate_kda(const KdaInputs& inputs,
                         const torch::Tensor& output,
                         const torch::Tensor& final_state) {
  CHECK(std::isfinite(inputs.options.gate_lower_bound));
  CHECK_LT(inputs.options.gate_lower_bound, 0.0f);
  CHECK(std::isfinite(inputs.options.scale));
  CHECK_GT(inputs.options.scale, 0.0f);
  const torch::Tensor& q = inputs.query;
  CHECK(q.defined());
  CHECK(q.device().is_privateuseone());
  CHECK_EQ(q.dim(), 4);
  CHECK_EQ(q.size(3), 128);
  CHECK_GT(q.size(0), 0);
  CHECK_GT(q.size(1), 0);
  CHECK_GT(q.size(2), 0);
  CHECK_LE(q.size(0), std::numeric_limits<int32_t>::max());
  CHECK_LE(q.size(1), std::numeric_limits<int32_t>::max());
  CHECK_LE(q.size(2), std::numeric_limits<int32_t>::max());
  for (const torch::Tensor* tensor :
       {&q, &inputs.key, &inputs.value, &inputs.raw_forget, &output}) {
    validate_musa_tensor(*tensor, q.device(), torch::kBFloat16);
    CHECK(tensor->sizes() == q.sizes());
  }
  validate_musa_tensor(inputs.raw_beta, q.device(), torch::kBFloat16);
  CHECK(inputs.raw_beta.sizes() ==
        torch::IntArrayRef({q.size(0), q.size(1), q.size(2)}));
  validate_musa_tensor(inputs.a_log, q.device(), torch::kFloat32);
  CHECK(inputs.a_log.sizes() == torch::IntArrayRef({q.size(2)}));
  validate_musa_tensor(inputs.dt_bias, q.device(), torch::kFloat32);
  CHECK(inputs.dt_bias.sizes() == torch::IntArrayRef({q.size(2), 128}));
  for (const torch::Tensor* tensor : {&inputs.initial_state, &final_state}) {
    validate_musa_tensor(*tensor, q.device(), torch::kFloat32);
    CHECK(tensor->sizes() ==
          torch::IntArrayRef({q.size(0), q.size(2), 128, 128}));
  }
  for (const torch::Tensor* tensor : {&q,
                                      &inputs.key,
                                      &inputs.value,
                                      &inputs.raw_forget,
                                      &inputs.raw_beta,
                                      &inputs.a_log,
                                      &inputs.dt_bias,
                                      &inputs.initial_state}) {
    CHECK(!output.is_alias_of(*tensor));
    CHECK(!final_state.is_alias_of(*tensor));
  }
  CHECK(!output.is_alias_of(final_state));
}

}  // namespace xllm::kernel::musa::detail
