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

#include "core/layers/musa/hyper_connection.h"

#include <glog/logging.h>

#include <tuple>
#include <unordered_map>

#include "core/kernels/musa/mhc_ops.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::layer {
StateDict get_mhc_state(const StateDict& state_dict,
                        const std::string& module_prefix,
                        const std::string& checkpoint_prefix) {
  StateDict module_state = state_dict.get_dict_with_prefix(module_prefix);
  if (module_state.size() > 0) {
    return module_state;
  }
  std::unordered_map<std::string, torch::Tensor> tensors;
  const torch::Tensor fn = state_dict.get_tensor(checkpoint_prefix + "fn");
  const torch::Tensor base = state_dict.get_tensor(checkpoint_prefix + "base");
  const torch::Tensor scale =
      state_dict.get_tensor(checkpoint_prefix + "scale");
  if (fn.defined()) {
    tensors.emplace("hc_fn", fn);
  }
  if (base.defined()) {
    tensors.emplace("hc_base", base);
  }
  if (scale.defined()) {
    tensors.emplace("hc_scale", scale);
  }
  return StateDict(std::move(tensors));
}

MHCPreImpl::MHCPreImpl(int64_t hc_mult,
                       int64_t dim,
                       int64_t sinkhorn_iters,
                       double hc_eps,
                       double norm_eps,
                       const torch::TensorOptions& options)
    : hc_mult_(hc_mult),
      dim_(dim),
      sinkhorn_iters_(sinkhorn_iters),
      hc_eps_(hc_eps),
      norm_eps_(norm_eps) {
  CHECK_GT(hc_mult_, 0);
  CHECK_GT(dim_, 0);
  CHECK_GE(sinkhorn_iters_, 1);
  const int64_t mix_size = hc_mult_ * (hc_mult_ + 2);
  const int64_t hc_dim = hc_mult_ * dim_;
  torch::TensorOptions parameter_options =
      options.dtype(torch::kFloat32).requires_grad(false);
  hc_fn_ =
      register_parameter("hc_fn",
                         torch::empty({mix_size, hc_dim}, parameter_options),
                         /*requires_grad=*/false);
  parameter_options = parameter_options.dtype(torch::kFloat32);
  hc_base_ = register_parameter("hc_base",
                                torch::empty({mix_size}, parameter_options),
                                /*requires_grad=*/false);
  hc_scale_ = register_parameter("hc_scale",
                                 torch::empty({3}, parameter_options),
                                 /*requires_grad=*/false);
}

MHCPreOutput MHCPreImpl::forward(const torch::Tensor& residual) {
  CHECK_EQ(residual.size(-2), hc_mult_);
  CHECK_EQ(residual.size(-1), dim_);
  kernel::musa::MHCPreOutput output = kernel::musa::mhc_pre(residual,
                                                            hc_fn_,
                                                            hc_scale_,
                                                            hc_base_,
                                                            norm_eps_,
                                                            hc_eps_,
                                                            hc_eps_,
                                                            2.0,
                                                            sinkhorn_iters_);
  return {output.layer_input, output.post_mix, output.comb_mix};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MHCPreImpl::post_pre_norm(const torch::Tensor& x,
                          const torch::Tensor& residual,
                          const torch::Tensor& post,
                          const torch::Tensor& comb,
                          const torch::Tensor& norm_weight) {
  CHECK(norm_weight.defined() && norm_weight.device() == residual.device());
  CHECK(norm_weight.is_contiguous());
  CHECK_EQ(norm_weight.sizes(), torch::IntArrayRef({dim_}));
  torch::Tensor residual_current =
      kernel::musa::mhc_post(x, residual, post, comb);
  MHCPreOutput pre = forward(residual_current);
  torch::Tensor normalized = torch::empty_like(pre.output);
  if (pre.output.numel() > 0) {
    kernel::cuda::rms_norm(normalized, pre.output, norm_weight, norm_eps_);
  }
  return {normalized, residual_current, pre.post, pre.comb};
}

void MHCPreImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  torch::NoGradGuard no_grad;
  const torch::Tensor fn = state_dict.get_tensor("hc_fn");
  if (fn.defined()) {
    CHECK(!hc_fn_is_loaded_) << "weight already loaded for hc_fn";
    CHECK_EQ(hc_fn_.sizes(), fn.sizes()) << "weight size mismatch for hc_fn";
    // GLM stores hc_fn as BF16 numeric values. Convert before the FP32 GEMM;
    // a raw copy would reinterpret the encoding.
    hc_fn_.copy_(fn.scalar_type() == torch::kFloat32 ? fn
                                                     : fn.to(torch::kFloat32));
    hc_fn_is_loaded_ = true;
  }
  LOAD_WEIGHT(hc_base);
  LOAD_WEIGHT(hc_scale);
}

void MHCPreImpl::verify_loaded_weights(const std::string& prefix) const {
  CHECK(hc_fn_is_loaded_) << "weight is not loaded for " << prefix + "hc_fn";
  CHECK(hc_base_is_loaded_)
      << "weight is not loaded for " << prefix + "hc_base";
  CHECK(hc_scale_is_loaded_)
      << "weight is not loaded for " << prefix + "hc_scale";
}

std::tuple<torch::Tensor, torch::Tensor> MHCPostImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    bool compute_rms) {
  torch::Tensor output = kernel::musa::mhc_post(x, residual, post, comb);
  if (!compute_rms) {
    return {output, torch::Tensor()};
  }
  return {output, kernel::musa::mhc_inverse_rms(output, norm_eps_)};
}

MHCHeadImpl::MHCHeadImpl(int64_t hc_mult,
                         int64_t dim,
                         double hc_eps,
                         double norm_eps,
                         const torch::TensorOptions& options)
    : hc_mult_(hc_mult), dim_(dim), hc_eps_(hc_eps), norm_eps_(norm_eps) {
  torch::TensorOptions parameter_options =
      options.dtype(torch::kFloat32).requires_grad(false);
  hc_head_fn_ = register_parameter(
      "hc_head_fn",
      torch::empty({hc_mult_, hc_mult_ * dim_}, parameter_options),
      /*requires_grad=*/false);
  hc_head_base_ =
      register_parameter("hc_head_base",
                         torch::empty({hc_mult_}, parameter_options),
                         /*requires_grad=*/false);
  hc_head_scale_ = register_parameter("hc_head_scale",
                                      torch::empty({1}, parameter_options),
                                      /*requires_grad=*/false);
}

torch::Tensor MHCHeadImpl::forward(const torch::Tensor& residual) {
  CHECK_EQ(residual.size(-2), hc_mult_);
  CHECK_EQ(residual.size(-1), dim_);
  return kernel::musa::mhc_head(
      residual, hc_head_fn_, hc_head_scale_, hc_head_base_, norm_eps_, hc_eps_);
}

void MHCHeadImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  torch::NoGradGuard no_grad;
  const torch::Tensor fn = state_dict.get_tensor("hc_head_fn");
  if (fn.defined()) {
    CHECK(!hc_head_fn_is_loaded_) << "weight already loaded for hc_head_fn";
    CHECK_EQ(hc_head_fn_.sizes(), fn.sizes())
        << "weight size mismatch for hc_head_fn";
    hc_head_fn_.copy_(
        fn.scalar_type() == torch::kFloat32 ? fn : fn.to(torch::kFloat32));
    hc_head_fn_is_loaded_ = true;
  }
  LOAD_WEIGHT(hc_head_base);
  LOAD_WEIGHT(hc_head_scale);
}

}  // namespace xllm::layer
