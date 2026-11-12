/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "core/kernels/musa/deepseek_v41_mhc_prenorm.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/layers/musa/deepseek_v41/profile_range.h"

namespace xllm::layer {

// Native V4.1 sublayer preparation. The decoder owns checkpoint parameters;
// this object retains their Tensor handles. A current sublayer's pre is
// returned for the NEXT sublayer; collapse always consumes carried_pre.
class DeepseekV41HyperConnection final {
 public:
  DeepseekV41HyperConnection(torch::Tensor fn,
                             torch::Tensor scale,
                             torch::Tensor base,
                             int64_t iterations,
                             double norm_eps,
                             double eps,
                             std::string mate_prenorm_directory = {})
      : fn_(std::move(fn)),
        scale_(std::move(scale)),
        base_(std::move(base)),
        iterations_(iterations),
        norm_eps_(norm_eps),
        eps_(eps),
        mate_prenorm_directory_(std::move(mate_prenorm_directory)) {
    CHECK(fn_.defined() && scale_.defined() && base_.defined());
    CHECK(fn_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(fn_.scalar_type(), torch::kFloat32);
    CHECK_EQ(scale_.scalar_type(), torch::kFloat32);
    CHECK_EQ(base_.scalar_type(), torch::kFloat32);
    CHECK_EQ(fn_.device(), scale_.device());
    CHECK_EQ(fn_.device(), base_.device());
    CHECK_EQ(fn_.dim(), 2);
    CHECK_EQ(fn_.size(0), 24);
    CHECK_GT(fn_.size(1), 0);
    CHECK_EQ(fn_.size(1) % 4, 0);
    CHECK_EQ(scale_.dim(), 1);
    CHECK_EQ(scale_.numel(), 3);
    CHECK_EQ(base_.dim(), 1);
    CHECK_EQ(base_.numel(), 24);
    CHECK(fn_.is_contiguous() && scale_.is_contiguous() &&
          base_.is_contiguous());
    CHECK_GE(iterations_, 1);
    CHECK_LE(iterations_, 1000);
    CHECK(std::isfinite(norm_eps_) && norm_eps_ > 0.0);
    CHECK(std::isfinite(eps_) && eps_ > 0.0 && eps_ < 1.0);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  prepare(const torch::Tensor& residual,
          const torch::Tensor& carried_pre) const {
    const DeepseekV41ProfileRange range("V41/mHC/prepare");
    check_residual(residual);
    CHECK(carried_pre.defined());
    CHECK_EQ(carried_pre.device(), residual.device());
    CHECK_EQ(carried_pre.scalar_type(), torch::kFloat32);
    CHECK_EQ(carried_pre.dim(), 2);
    CHECK_EQ(carried_pre.size(0), residual.size(0));
    CHECK_EQ(carried_pre.size(1), 4);
    torch::DeviceGuard guard(residual.device());
    const torch::Tensor packed = residual.contiguous();
    torch::Tensor mixes;
    if (!mate_prenorm_directory_.empty()) {
      mixes = kernel::musa::deepseek_v41_mhc_prenorm(
          packed.flatten(1), fn_, mate_prenorm_directory_, norm_eps_);
    } else {
      // Explicit reference mode, selected at construction; never an exception
      // fallback from a failed MATE invocation.
      const torch::Tensor flat = packed.flatten(1).to(torch::kFloat32);
      mixes = torch::empty({flat.size(0), 24}, flat.options());
      torch::Tensor tile = torch::empty({128, flat.size(1)}, flat.options());
      for (int64_t start = 0; start < flat.size(0); start += 128) {
        const int64_t count = std::min<int64_t>(128, flat.size(0) - start);
        tile.narrow(0, 0, count).copy_(flat.narrow(0, start, count));
        mixes.narrow(0, start, count)
            .copy_(torch::linear(tile, fn_).narrow(0, 0, count));
      }
      mixes = mixes * torch::rsqrt(flat.square().mean(-1, true) + norm_eps_);
    }
    auto [next_pre, post, comb] = kernel::musa::deepseek_v4_hc_pre_metadata(
        mixes.contiguous(), scale_, base_, 4, iterations_, eps_);
    torch::Tensor input =
        packed.size(0) == 0
            ? torch::empty({0, packed.size(2)}, packed.options())
            : kernel::musa::deepseek_v4_hc_pre_combine(
                  packed, carried_pre.contiguous());
    return {input, next_pre, post, comb};
  }

  torch::Tensor combine(const torch::Tensor& value,
                        const torch::Tensor& residual,
                        const torch::Tensor& post,
                        const torch::Tensor& comb) const {
    const DeepseekV41ProfileRange range("V41/mHC/post");
    check_residual(residual);
    CHECK_EQ(value.device(), residual.device());
    CHECK_EQ(value.scalar_type(), torch::kBFloat16);
    CHECK_EQ(value.dim(), 2);
    CHECK_EQ(value.size(0), residual.size(0));
    CHECK_EQ(value.size(1), residual.size(2));
    CHECK_EQ(post.device(), residual.device());
    CHECK_EQ(comb.device(), residual.device());
    CHECK_EQ(post.scalar_type(), torch::kFloat32);
    CHECK_EQ(comb.scalar_type(), torch::kFloat32);
    CHECK_EQ(post.dim(), 2);
    CHECK_EQ(post.size(0), residual.size(0));
    CHECK_EQ(post.size(1), 4);
    CHECK_EQ(comb.dim(), 3);
    CHECK_EQ(comb.size(0), residual.size(0));
    CHECK_EQ(comb.size(1), 4);
    CHECK_EQ(comb.size(2), 4);
    torch::DeviceGuard guard(residual.device());
    return kernel::musa::deepseek_v4_hc_post(value.contiguous(),
                                             residual.contiguous(),
                                             post.contiguous(),
                                             comb.contiguous());
  }

 private:
  void check_residual(const torch::Tensor& residual) const {
    CHECK(residual.defined());
    CHECK_EQ(residual.device(), fn_.device());
    CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
    CHECK_EQ(residual.dim(), 3);
    CHECK_EQ(residual.size(1), 4);
    CHECK_EQ(residual.size(2) * 4, fn_.size(1));
  }

  torch::Tensor fn_;
  torch::Tensor scale_;
  torch::Tensor base_;
  int64_t iterations_;
  double norm_eps_;
  double eps_;
  std::string mate_prenorm_directory_;
};

}  // namespace xllm::layer
