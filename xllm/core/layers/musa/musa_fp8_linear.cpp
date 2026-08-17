/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/layers/musa/musa_fp8_linear.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/ops_api.h"

namespace xllm::layer::musa {

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  (void)input_scale;
  CHECK(weight.defined() && weight.dim() == 2)
      << "MUSA FP8 linear requires a 2D weight.";
  CHECK(weight_scale.defined())
      << "MUSA FP8 linear requires a loaded weight scale.";

  const bool native_blockwise =
      input.scalar_type() == torch::kBFloat16 &&
      (weight.scalar_type() == torch::kFloat8_e4m3fn ||
       weight.scalar_type() == torch::kFloat8_e5m2) &&
      weight_scale.scalar_type() == torch::kFloat32 &&
      weight_scale.dim() == 2 && weight.size(1) % 128 == 0 &&
      weight_scale.size(0) == (weight.size(0) + 127) / 128 &&
      weight_scale.size(1) == weight.size(1) / 128 &&
      (output_dtype == torch::kBFloat16 || output_dtype == torch::kFloat16) &&
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0;
  if (native_blockwise) {
    auto input_2d = input.reshape({-1, input.size(-1)}).contiguous();
    auto [quantized_input, input_group_scale] =
        kernel::musa::per_token_group_quant_fp8(input_2d, 128);
    if (quantized_input.defined() && input_group_scale.defined()) {
      auto output =
          kernel::musa::gemm_fp8_nt_groupwise(quantized_input,
                                              weight.contiguous(),
                                              input_group_scale,
                                              weight_scale.contiguous(),
                                              output_dtype);
      if (output.defined()) {
        if (bias.has_value() && bias.value().defined()) {
          output.add_(bias.value());
        }
        auto output_shape = input.sizes().vec();
        output_shape.back() = weight.size(0);
        return output.view(output_shape);
      }
    }
  }

  auto scale_fp32 = weight_scale.to(torch::kFloat32);
  torch::Tensor expanded_scale;
  if (scale_fp32.numel() == 1) {
    expanded_scale = scale_fp32.reshape({1, 1});
  } else if (scale_fp32.dim() == 2) {
    const int64_t block_rows =
        (weight.size(0) + scale_fp32.size(0) - 1) / scale_fp32.size(0);
    const int64_t block_cols =
        (weight.size(1) + scale_fp32.size(1) - 1) / scale_fp32.size(1);
    expanded_scale = scale_fp32.unsqueeze(1)
                         .unsqueeze(3)
                         .repeat({1, block_rows, 1, block_cols})
                         .reshape({scale_fp32.size(0) * block_rows,
                                   scale_fp32.size(1) * block_cols})
                         .slice(/*dim=*/0, /*start=*/0, /*end=*/weight.size(0))
                         .slice(/*dim=*/1, /*start=*/0, /*end=*/weight.size(1));
  } else if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(0)) {
    expanded_scale = scale_fp32.reshape({-1, 1});
  } else if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(1)) {
    expanded_scale = scale_fp32.reshape({1, -1});
  } else {
    LOG(FATAL) << "Unsupported MUSA FP8 weight scale shape "
               << scale_fp32.sizes() << " for weight " << weight.sizes();
  }

  auto dequantized_weight =
      (weight.to(torch::kFloat32) * expanded_scale).to(output_dtype);
  auto input_2d = input.view({-1, input.size(-1)});
  kernel::MatmulParams matmul_params;
  matmul_params.a = input_2d;
  matmul_params.b = dequantized_weight.contiguous();
  matmul_params.bias = bias;
  auto output = kernel::matmul(matmul_params);
  auto output_shape = input.sizes().vec();
  output_shape.back() = weight.size(0);
  return output.view(output_shape);
}

bool load_fp8_weight_scale(const StateDict& state_dict,
                           int64_t dim,
                           int64_t rank,
                           int64_t world_size,
                           torch::Tensor& weight_scale,
                           bool& weight_scale_is_loaded,
                           const torch::Tensor& weight) {
  if (weight_scale_is_loaded) {
    return true;
  }
  const auto load = [&](const char* name) {
    auto scale =
        dim >= 0 ? state_dict.get_sharded_tensor(name, dim, rank, world_size)
                 : state_dict.get_tensor(name);
    if (!scale.defined()) {
      return false;
    }
    torch::NoGradGuard no_grad;
    auto loaded = scale.to(torch::kFloat32).to(weight.device()).contiguous();
    if (weight_scale.defined()) {
      weight_scale.set_data(loaded);
    } else {
      weight_scale = loaded;
    }
    weight_scale_is_loaded = true;
    return true;
  };
  return load("weight_scale") || load("scale");
}

bool load_fused_fp8_weight_scale(const StateDict& state_dict,
                                 const std::vector<std::string>& prefixes,
                                 int64_t dim,
                                 int64_t rank,
                                 int64_t world_size,
                                 std::vector<torch::Tensor>& scale_list,
                                 torch::Tensor& weight_scale,
                                 bool& weight_scale_is_loaded,
                                 const torch::Tensor& weight) {
  if (weight_scale_is_loaded) {
    return true;
  }
  if (scale_list.size() < prefixes.size()) {
    scale_list.resize(prefixes.size());
  }
  for (size_t i = 0; i < prefixes.size(); ++i) {
    if (scale_list[i].defined()) {
      continue;
    }
    auto scale = state_dict.get_sharded_tensor(
        prefixes[i] + "scale", dim, rank, world_size);
    if (scale.defined()) {
      scale_list[i] = scale.to(torch::kFloat32);
    }
  }
  if (!std::all_of(
          scale_list.begin(), scale_list.end(), [](const torch::Tensor& scale) {
            return scale.defined();
          })) {
    return false;
  }
  torch::NoGradGuard no_grad;
  auto merged = torch::cat(scale_list, dim).to(weight.device()).contiguous();
  if (weight_scale.defined()) {
    weight_scale.set_data(merged);
  } else {
    weight_scale = merged;
  }
  weight_scale_is_loaded = true;
  scale_list.clear();
  return true;
}

}  // namespace xllm::layer::musa
