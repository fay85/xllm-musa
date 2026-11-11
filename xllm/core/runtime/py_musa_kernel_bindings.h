/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/utils.h>

#include <cmath>

#include "core/kernels/musa/deepseek_v41_attention_quant.h"
#include "core/kernels/musa/deepseek_v41_engram_gate.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm {

// Shared by the embedded runtime and the standalone numerical-test module.
// This binds the existing V4 kernel; it does not implement a second kernel.
inline void register_musa_kernel_bindings(pybind11::module_& module) {
  module.def("musa_v41_engram_gate",
             &kernel::musa::deepseek_v41_engram_gate,
             pybind11::arg("input"),
             pybind11::arg("kv"),
             pybind11::arg("q_weight"),
             pybind11::arg("k_weight"),
             pybind11::arg("mask") = std::nullopt,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "musa_shared_swiglu",
      [](const torch::Tensor& input, double limit) {
        CHECK(input.defined());
        CHECK(input.device().type() == torch::kPrivateUse1);
        CHECK(input.scalar_type() == torch::kBFloat16);
        CHECK_EQ(input.dim(), 2);
        CHECK_GT(input.size(1), 0);
        CHECK_EQ(input.size(1) % 2, 0);
        CHECK(input.is_contiguous());
        CHECK(std::isfinite(limit) && limit > 0.0 && limit < 1000000.0);
        torch::Tensor output =
            torch::empty({input.size(0), input.size(1) / 2}, input.options());
        if (input.size(0) == 0) {
          return output;
        }
        torch::Tensor result =
            kernel::musa::deepseek_v4_shared_swiglu(input, limit, output);
        CHECK(result.defined()) << "V4 shared SwiGLU rejected validated input";
        return result;
      },
      pybind11::arg("input"),
      pybind11::arg("limit"),
      pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def("musa_v41_attention_fp4_quant",
             &kernel::musa::deepseek_v41_attention_fp4_quant,
             pybind11::arg("input"),
             pybind11::arg("group_size"),
             pybind11::arg("e4m3_scale"),
             pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def("musa_v41_decode_moe",
             &kernel::musa::deepseek_v41_decode_moe,
             pybind11::arg("input"),
             pybind11::arg("expert_ids"),
             pybind11::arg("route_weights"),
             pybind11::arg("gate"),
             pybind11::arg("gate_scale"),
             pybind11::arg("up"),
             pybind11::arg("up_scale"),
             pybind11::arg("down"),
             pybind11::arg("down_scale"),
             pybind11::arg("first_expert"),
             pybind11::arg("limit"),
             pybind11::arg("block32_quant"),
             pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def("musa_v41_block32_batch_quant",
             &kernel::musa::deepseek_v41_block32_batch_quant,
             pybind11::arg("input"),
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("musa_v41_block32_batch_unpack",
             &kernel::musa::deepseek_v41_block32_batch_unpack,
             pybind11::arg("packed"),
             pybind11::arg("scales"),
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("musa_v41_block32_batch_reduce",
             &kernel::musa::deepseek_v41_block32_batch_reduce,
             pybind11::arg("partial"),
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("musa_v41_block32_batch_reduce_scaled",
             &kernel::musa::deepseek_v41_block32_batch_reduce_scaled,
             pybind11::arg("partial"),
             pybind11::arg("input_scales"),
             pybind11::arg("weight_scales"));
  module.def("musa_v41_partition_linear",
             &kernel::musa::deepseek_v41_partition_linear,
             pybind11::arg("input"),
             pybind11::arg("packed"),
             pybind11::arg("scales"),
             pybind11::arg("bmm_so_path"),
             pybind11::arg("max_partial_bytes"),
             pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def("musa_v41_block32_qdq",
             &kernel::musa::deepseek_v41_block32_qdq,
             pybind11::arg("input"),
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("musa_v41_packed_fp4_gemv",
             &kernel::musa::deepseek_v41_packed_fp4_gemv,
             pybind11::arg("input"),
             pybind11::arg("packed"),
             pybind11::arg("scales"),
             pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def("musa_v41_gate_topk",
             &kernel::musa::deepseek_v41_gate_topk,
             pybind11::arg("logits"),
             pybind11::arg("bias"),
             pybind11::arg("topk"),
             pybind11::arg("normalize"),
             pybind11::arg("scale"),
             pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def(
      "musa_partial_rope_fp32",
      [](const torch::Tensor& input,
         const torch::Tensor& cos,
         const torch::Tensor& sin,
         bool inverse) {
        CHECK(input.defined() && cos.defined() && sin.defined());
        CHECK(input.device().type() == torch::kPrivateUse1);
        CHECK_EQ(input.device(), cos.device());
        CHECK_EQ(input.device(), sin.device());
        CHECK_EQ(input.scalar_type(), torch::kFloat32);
        CHECK_EQ(cos.scalar_type(), torch::kFloat32);
        CHECK_EQ(sin.scalar_type(), torch::kFloat32);
        CHECK(input.dim() == 2 || input.dim() == 3);
        CHECK_EQ(cos.dim(), 2);
        CHECK_EQ(sin.sizes(), cos.sizes());
        CHECK_EQ(cos.size(0), input.size(0));
        CHECK_GT(cos.size(1), 0);
        CHECK_EQ(cos.size(1) % 2, 0);
        CHECK_LE(cos.size(1), input.size(-1));
        CHECK(input.is_contiguous() && cos.is_contiguous() &&
              sin.is_contiguous());
        torch::DeviceGuard device_guard(input.device());
        if (input.numel() == 0) {
          return torch::empty_like(input);
        }
        torch::Tensor output;
        CHECK(
            kernel::musa::deepseek_v4_partial_rope(input,
                                                   cos,
                                                   sin,
                                                   input.size(-1) - cos.size(1),
                                                   cos.size(1),
                                                   inverse,
                                                   output));
        return output;
      },
      pybind11::arg("input"),
      pybind11::arg("cos"),
      pybind11::arg("sin"),
      pybind11::arg("inverse"),
      pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def(
      "musa_hc_pre_combine",
      [](const torch::Tensor& residual, const torch::Tensor& pre) {
        CHECK(residual.defined() && pre.defined());
        CHECK(residual.device().type() == torch::kPrivateUse1);
        CHECK_EQ(residual.device(), pre.device());
        CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
        CHECK_EQ(pre.scalar_type(), torch::kFloat32);
        CHECK_EQ(residual.dim(), 3);
        CHECK_EQ(residual.size(1), 4);
        CHECK_GT(residual.size(2), 0);
        CHECK_EQ(pre.dim(), 2);
        CHECK_EQ(pre.size(0), residual.size(0));
        CHECK_EQ(pre.size(1), residual.size(1));
        CHECK(residual.is_contiguous() && pre.is_contiguous());
        torch::DeviceGuard device_guard(residual.device());
        if (residual.size(0) == 0) {
          return torch::empty({0, residual.size(2)}, residual.options());
        }
        return kernel::musa::deepseek_v4_hc_pre_combine(residual, pre);
      },
      pybind11::arg("residual"),
      pybind11::arg("pre"),
      pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def(
      "musa_hc_post",
      [](const torch::Tensor& input,
         const torch::Tensor& residual,
         const torch::Tensor& post,
         const torch::Tensor& comb) {
        CHECK(input.defined() && residual.defined() && post.defined() &&
              comb.defined());
        CHECK(residual.device().type() == torch::kPrivateUse1);
        CHECK_EQ(input.device(), residual.device());
        CHECK_EQ(post.device(), residual.device());
        CHECK_EQ(comb.device(), residual.device());
        CHECK_EQ(input.scalar_type(), torch::kBFloat16);
        CHECK_EQ(residual.scalar_type(), torch::kBFloat16);
        CHECK_EQ(post.scalar_type(), torch::kFloat32);
        CHECK_EQ(comb.scalar_type(), torch::kFloat32);
        CHECK_EQ(residual.dim(), 3);
        CHECK_EQ(residual.size(1), 4);
        CHECK_GT(residual.size(2), 0);
        CHECK_EQ(input.dim(), 2);
        CHECK_EQ(input.size(0), residual.size(0));
        CHECK_EQ(input.size(1), residual.size(2));
        CHECK_EQ(post.dim(), 2);
        CHECK_EQ(post.size(0), residual.size(0));
        CHECK_EQ(post.size(1), 4);
        CHECK_EQ(comb.dim(), 3);
        CHECK_EQ(comb.size(0), residual.size(0));
        CHECK_EQ(comb.size(1), 4);
        CHECK_EQ(comb.size(2), 4);
        CHECK(input.is_contiguous() && residual.is_contiguous() &&
              post.is_contiguous() && comb.is_contiguous());
        torch::DeviceGuard device_guard(residual.device());
        return kernel::musa::deepseek_v4_hc_post(input, residual, post, comb);
      },
      pybind11::arg("input"),
      pybind11::arg("residual"),
      pybind11::arg("post"),
      pybind11::arg("comb"),
      pybind11::call_guard<pybind11::gil_scoped_release>());

  module.def(
      "musa_hc_metadata",
      [](const torch::Tensor& mixes,
         const torch::Tensor& scale,
         const torch::Tensor& base,
         int64_t iterations,
         double eps) {
        CHECK(mixes.defined() && scale.defined() && base.defined());
        CHECK(mixes.device().type() == torch::kPrivateUse1);
        CHECK_EQ(mixes.device(), scale.device());
        CHECK_EQ(mixes.device(), base.device());
        CHECK_EQ(mixes.scalar_type(), torch::kFloat32);
        CHECK_EQ(scale.scalar_type(), torch::kFloat32);
        CHECK_EQ(base.scalar_type(), torch::kFloat32);
        CHECK_EQ(mixes.dim(), 2);
        CHECK_EQ(mixes.size(1), 24);
        CHECK_EQ(scale.dim(), 1);
        CHECK_EQ(scale.numel(), 3);
        CHECK_EQ(base.dim(), 1);
        CHECK_EQ(base.numel(), 24);
        CHECK(mixes.is_contiguous() && scale.is_contiguous() &&
              base.is_contiguous());
        CHECK_GE(iterations, 1);
        CHECK_LE(iterations, 1000);
        CHECK(std::isfinite(eps) && eps > 0.0 && eps < 1.0);
        torch::DeviceGuard device_guard(mixes.device());
        return kernel::musa::deepseek_v4_hc_pre_metadata(
            mixes, scale, base, /*hc_mult=*/4, iterations, eps);
      },
      pybind11::arg("mixes"),
      pybind11::arg("scale"),
      pybind11::arg("base"),
      pybind11::arg("iterations"),
      pybind11::arg("eps"),
      pybind11::call_guard<pybind11::gil_scoped_release>());
}

}  // namespace xllm
