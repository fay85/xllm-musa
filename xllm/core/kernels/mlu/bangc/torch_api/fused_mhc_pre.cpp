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

#include "kernels/mlu/bangc/include/fused_mhc_pre.h"

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/fused_mhc_pre/launcher.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_mhc_pre(const torch::Tensor& residual,
              const torch::Tensor& pre_in,
              const torch::Tensor& hc_fn,
              const torch::Tensor& hc_scale,
              const torch::Tensor& hc_base,
              const torch::Tensor& gamma,
              int64_t sinkhorn_iter,
              double hc_eps,
              double norm_eps) {
  CHECK(detail::check_tensor(residual, torch::kBFloat16))
      << "residual must have the expected dtype and be contiguous on MLU";
  CHECK_EQ(residual.dim(), 3);
  CHECK_GT(sinkhorn_iter, 0);
  CHECK_LE(sinkhorn_iter, std::numeric_limits<int32_t>::max());
  CHECK(std::isfinite(hc_eps) && hc_eps >= 0);
  CHECK(std::isfinite(norm_eps) && norm_eps > 0);
  const torch::Device device = residual.device();
  const torch::DeviceGuard guard(device);
  const int64_t tokens = residual.size(0);
  CHECK_LE(tokens, int64_t{1} << 30);
  constexpr int64_t kHc = 4;
  constexpr int64_t kDim = 5120;
  constexpr int64_t kMix = 24;
  CHECK(residual.sizes() == torch::IntArrayRef({tokens, kHc, kDim}))
      << "residual has an unexpected shape";
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
  torch::Tensor norm_out = torch::empty({tokens, kDim}, residual.options());
  torch::Tensor new_pre = torch::empty_like(pre_in);
  torch::Tensor post = torch::empty_like(pre_in);
  torch::Tensor comb = torch::empty({tokens, kHc, kHc}, pre_in.options());
  if (tokens == 0) {
    return {norm_out, new_pre, post, comb};
  }
  const void* inputs[6] = {residual.data_ptr(),
                           pre_in.data_ptr(),
                           hc_fn.data_ptr(),
                           hc_scale.data_ptr(),
                           hc_base.data_ptr(),
                           gamma.data_ptr()};
  void* const outputs[4] = {norm_out.data_ptr(),
                            new_pre.data_ptr(),
                            post.data_ptr(),
                            comb.data_ptr()};
  const int64_t workspace_bytes = fused_mhc_pre_workspace_bytes(tokens);
  CHECK_GE(workspace_bytes, 0) << "fused_mhc_pre: device query failed";
  const torch::Tensor workspace = torch::empty(
      {workspace_bytes}, torch::dtype(torch::kUInt8).device(device));
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const int32_t status =
      fused_mhc_pre_launch(static_cast<void*>(queue),
                           inputs,
                           outputs,
                           workspace.data_ptr(),
                           tokens,
                           static_cast<int32_t>(kDim),
                           static_cast<int32_t>(sinkhorn_iter),
                           static_cast<float>(hc_eps),
                           static_cast<float>(norm_eps));
  CHECK_EQ(status, 0) << "fused_mhc_pre_launch rejected the launch";
  CHECK_EQ(cnrtSuccess, cnrtGetLastError());
  return {norm_out, new_pre, post, comb};
}
}  // namespace xllm::kernel::mlu::bangc
