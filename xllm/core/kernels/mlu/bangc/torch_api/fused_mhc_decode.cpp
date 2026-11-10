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

#include "kernels/mlu/bangc/include/fused_mhc_decode.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/fused_mhc_decode/launcher.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {

void fused_mhc_decode(const torch::Tensor& x,
                      const torch::Tensor& residual_in,
                      const torch::Tensor& post_in,
                      const torch::Tensor& comb_in,
                      const torch::Tensor& pre_in,
                      const torch::Tensor& hc_fn,
                      const torch::Tensor& hc_scale,
                      const torch::Tensor& hc_base,
                      const torch::Tensor& gamma,
                      int64_t sinkhorn_iter,
                      double hc_eps,
                      double norm_eps,
                      torch::Tensor& norm_out,
                      torch::Tensor& residual_out,
                      torch::Tensor& next_pre,
                      torch::Tensor& post_out,
                      torch::Tensor& comb_out) {
  CHECK(detail::check_tensor(x,
                             std::nullopt,
                             std::nullopt,
                             /*require_contiguous=*/false))
      << "x must be on MLU";
  CHECK_EQ(x.dim(), 2);
  CHECK_GT(sinkhorn_iter, 0);
  CHECK_LE(sinkhorn_iter, std::numeric_limits<int32_t>::max());
  CHECK(std::isfinite(hc_eps) && hc_eps >= 0) << "hc_eps must be finite, >= 0";
  CHECK(std::isfinite(norm_eps) && norm_eps > 0)
      << "norm_eps must be finite and positive";
  const torch::Device device = x.device();
  const int64_t tokens = x.size(0);
  CHECK_LE(tokens, int64_t{1} << 30) << "fused_mhc_decode: too many tokens";
  constexpr int64_t kHc = 4;
  constexpr int64_t kDim = 5120;
  constexpr int64_t kMix = 24;
  if (tokens == 0) {
    return;
  }
  CHECK(detail::check_tensor(x, torch::kBFloat16, device))
      << "x must have the expected dtype/device and be contiguous";
  CHECK(x.sizes() == torch::IntArrayRef({tokens, kDim}))
      << "x has an unexpected shape";
  CHECK(detail::check_tensor(residual_in, torch::kBFloat16, device))
      << "residual_in must have the expected dtype/device and be contiguous";
  CHECK(residual_in.sizes() == torch::IntArrayRef({tokens, kHc, kDim}))
      << "residual_in has an unexpected shape";
  CHECK(detail::check_tensor(post_in, torch::kFloat32, device))
      << "post_in must have the expected dtype/device and be contiguous";
  CHECK(post_in.sizes() == torch::IntArrayRef({tokens, kHc}))
      << "post_in has an unexpected shape";
  CHECK(detail::check_tensor(comb_in, torch::kFloat32, device))
      << "comb_in must have the expected dtype/device and be contiguous";
  CHECK(comb_in.sizes() == torch::IntArrayRef({tokens, kHc, kHc}))
      << "comb_in has an unexpected shape";
  CHECK(detail::check_tensor(pre_in, torch::kFloat32, device))
      << "pre_in must have the expected dtype/device and be contiguous";
  CHECK(pre_in.sizes() == torch::IntArrayRef({tokens, kHc}))
      << "pre_in has an unexpected shape";
  CHECK(detail::check_tensor(hc_fn, torch::kBFloat16, device))
      << "hc_fn must have the expected dtype/device and be contiguous";
  CHECK(hc_fn.sizes() == torch::IntArrayRef({kMix, kHc * kDim}))
      << "hc_fn has an unexpected shape";
  CHECK(detail::check_tensor(hc_scale, torch::kFloat32, device))
      << "hc_scale must have the expected dtype/device and be contiguous";
  CHECK(hc_scale.sizes() == torch::IntArrayRef({3}))
      << "hc_scale has an unexpected shape";
  CHECK(detail::check_tensor(hc_base, torch::kFloat32, device))
      << "hc_base must have the expected dtype/device and be contiguous";
  CHECK(hc_base.sizes() == torch::IntArrayRef({kMix}))
      << "hc_base has an unexpected shape";
  CHECK(detail::check_tensor(gamma, torch::kBFloat16, device))
      << "gamma must have the expected dtype/device and be contiguous";
  CHECK(gamma.sizes() == torch::IntArrayRef({kDim}))
      << "gamma has an unexpected shape";
  CHECK(detail::check_tensor(norm_out, torch::kBFloat16, device))
      << "norm_out must have the expected dtype/device and be contiguous";
  CHECK(norm_out.sizes() == torch::IntArrayRef({tokens, kDim}))
      << "norm_out has an unexpected shape";
  CHECK(detail::check_tensor(residual_out, torch::kBFloat16, device))
      << "residual_out must have the expected dtype/device and be contiguous";
  CHECK(residual_out.sizes() == torch::IntArrayRef({tokens, kHc, kDim}))
      << "residual_out has an unexpected shape";
  CHECK(detail::check_tensor(next_pre, torch::kFloat32, device))
      << "next_pre must have the expected dtype/device and be contiguous";
  CHECK(next_pre.sizes() == torch::IntArrayRef({tokens, kHc}))
      << "next_pre has an unexpected shape";
  CHECK(detail::check_tensor(post_out, torch::kFloat32, device))
      << "post_out must have the expected dtype/device and be contiguous";
  CHECK(post_out.sizes() == torch::IntArrayRef({tokens, kHc}))
      << "post_out has an unexpected shape";
  CHECK(detail::check_tensor(comb_out, torch::kFloat32, device))
      << "comb_out must have the expected dtype/device and be contiguous";
  CHECK(comb_out.sizes() == torch::IntArrayRef({tokens, kHc, kHc}))
      << "comb_out has an unexpected shape";

  const void* inputs[9] = {x.data_ptr(),
                           residual_in.data_ptr(),
                           post_in.data_ptr(),
                           comb_in.data_ptr(),
                           pre_in.data_ptr(),
                           hc_fn.data_ptr(),
                           hc_scale.data_ptr(),
                           hc_base.data_ptr(),
                           gamma.data_ptr()};
  void* const outputs[5] = {norm_out.data_ptr(),
                            residual_out.data_ptr(),
                            next_pre.data_ptr(),
                            post_out.data_ptr(),
                            comb_out.data_ptr()};
  const torch::DeviceGuard guard(device);
  const int64_t workspace_bytes = fused_mhc_decode_workspace_bytes(tokens);
  CHECK_GE(workspace_bytes, 0)
      << "fused_mhc_decode: device query failed or unsupported topology";
  const torch::Tensor workspace = torch::empty(
      {workspace_bytes}, torch::dtype(torch::kUInt8).device(device));
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const int32_t status =
      fused_mhc_decode_launch(static_cast<void*>(queue),
                              inputs,
                              outputs,
                              workspace.data_ptr(),
                              tokens,
                              static_cast<int32_t>(kDim),
                              static_cast<int32_t>(sinkhorn_iter),
                              static_cast<float>(hc_eps),
                              static_cast<float>(norm_eps));
  CHECK_EQ(status, 0) << "fused_mhc_decode_launch rejected the launch";
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
}

}  // namespace xllm::kernel::mlu::bangc
