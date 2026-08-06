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

#include "layers/musa/activation.h"

#include <ATen/ops/swish_glu.h>
#include <glog/logging.h>

#include <cmath>
#include <vector>

#include "kernels/ops_api.h"

namespace xllm::layer::musa {

namespace {

bool has_effective_swiglu_limit(double swiglu_limit) {
  return std::isfinite(swiglu_limit) && swiglu_limit > 0.0 &&
         swiglu_limit < 1000000.0;
}

torch::Tensor swiglu_with_clamp(const torch::Tensor& input,
                                double swiglu_limit) {
  CHECK(input.defined()) << "SwiGLU input is undefined.";
  CHECK_GT(input.dim(), 0) << "SwiGLU input must have at least one dimension.";
  const int64_t last_dim = input.size(-1);
  CHECK_EQ(last_dim % 2, 0)
      << "SwiGLU input last dimension must be even, got " << last_dim;
  const int64_t half_dim = last_dim / 2;
  const torch::ScalarType input_dtype = input.scalar_type();
  torch::Tensor gate = input.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)
                           .to(torch::kFloat32);
  torch::Tensor up = input
                         .slice(/*dim=*/-1,
                                /*start=*/half_dim,
                                /*end=*/last_dim)
                         .to(torch::kFloat32);
  gate = torch::clamp_max(gate, swiglu_limit);
  up = torch::clamp(up, -swiglu_limit, swiglu_limit);
  torch::Tensor output = torch::silu(gate) * up;
  return output.to(input_dtype);
}

}  // namespace

ActivationImpl::ActivationImpl(const std::string& act_mode,
                               bool is_gated,
                               double swiglu_limit)
    : act_mode_(act_mode), is_gated_(is_gated), swiglu_limit_(swiglu_limit) {
  CHECK(is_gated_) << "MUSA activation supports only gated activations.";
}

void ActivationImpl::forward(torch::Tensor& input, torch::Tensor& output) {
  if (is_gated_ && (act_mode_ == "silu" || act_mode_ == "swiglu")) {
    if (has_effective_swiglu_limit(swiglu_limit_)) {
      output = swiglu_with_clamp(input, swiglu_limit_);
    } else {
      output = at::swish_glu(input);
    }
    return;
  }

  CHECK(input.defined()) << "MUSA activation input is undefined.";
  CHECK_GT(input.dim(), 0) << "MUSA activation input must have a dimension.";
  CHECK_EQ(input.size(-1) % 2, 0)
      << "MUSA gated activation input dimension must be even.";
  CHECK(input.is_contiguous()) << "MUSA activation input must be contiguous.";
  std::vector<int64_t> expected_output_shape = input.sizes().vec();
  expected_output_shape.back() /= 2;
  if (!output.defined()) {
    output = torch::empty(expected_output_shape, input.options());
  }
  CHECK(output.sizes().vec() == expected_output_shape)
      << "MUSA activation output shape mismatch.";
  CHECK(output.scalar_type() == input.scalar_type())
      << "MUSA activation output dtype mismatch.";
  CHECK(output.device() == input.device())
      << "MUSA activation output device mismatch.";
  CHECK(output.is_contiguous()) << "MUSA activation output must be contiguous.";

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = input;
  activation_params.output = output;
  activation_params.act_mode = act_mode_;
  activation_params.is_gated = is_gated_;
  xllm::kernel::active(activation_params);
  output = activation_params.output;
}

}  // namespace xllm::layer::musa
