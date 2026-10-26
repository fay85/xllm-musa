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
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/ops_api.h"

namespace xllm::layer::musa {
namespace {

constexpr int64_t kDecodeWorkspaceRows = 64;
constexpr int64_t kHcLaneExpandedWorkspaceRows = kDecodeWorkspaceRows * 4;

struct Fp8LinearWorkspace {
  int64_t capacity_rows = 0;
  torch::Tensor output;
  torch::Tensor dequantized_weight;
};

Fp8LinearWorkspace& get_fp8_linear_workspace(const torch::Tensor& weight) {
  static thread_local std::unordered_map<std::uintptr_t, Fp8LinearWorkspace>
      workspaces;
  const std::uintptr_t key =
      reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  return workspaces[key];
}

torch::Tensor get_fp8_linear_output_workspace(const torch::Tensor& input,
                                              const torch::Tensor& weight,
                                              at::ScalarType output_dtype) {
  auto& workspace = get_fp8_linear_workspace(weight);
  const int64_t rows = input.numel() / input.size(-1);
  const int64_t cols = weight.size(0);
  if (workspace.capacity_rows < rows || !workspace.output.defined() ||
      workspace.output.size(1) != cols ||
      workspace.output.scalar_type() != output_dtype) {
    const int64_t new_capacity = std::max<int64_t>(
        rows, workspace.capacity_rows > 0 ? workspace.capacity_rows * 2 : 1);
    workspace.capacity_rows = new_capacity;
    workspace.output =
        torch::empty({new_capacity, cols}, input.options().dtype(output_dtype));
  }
  return workspace.output.narrow(/*dim=*/0, /*start=*/0, /*length=*/rows);
}

torch::Tensor get_requested_output_workspace(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    at::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output) {
  if (!output.has_value()) {
    return get_fp8_linear_output_workspace(input, weight, output_dtype);
  }
  const torch::Tensor& requested = output.value();
  const int64_t rows = input.numel() / input.size(-1);
  CHECK(requested.defined() && requested.device() == input.device())
      << "MUSA FP8 linear output must be defined on the input device";
  CHECK(requested.scalar_type() == output_dtype && requested.is_contiguous())
      << "MUSA FP8 linear output must be contiguous with the requested dtype";
  CHECK_EQ(requested.numel(), rows * weight.size(0))
      << "MUSA FP8 linear output has an incompatible shape";
  return requested.view({rows, weight.size(0)});
}

torch::Tensor expand_fp8_weight_scale(const torch::Tensor& weight_scale,
                                      const torch::Tensor& weight) {
  auto scale_fp32 = weight_scale.to(torch::kFloat32);
  if (scale_fp32.numel() == 1) {
    return scale_fp32.reshape({1, 1});
  }
  if (scale_fp32.dim() == 2) {
    const int64_t block_rows =
        (weight.size(0) + scale_fp32.size(0) - 1) / scale_fp32.size(0);
    const int64_t block_cols =
        (weight.size(1) + scale_fp32.size(1) - 1) / scale_fp32.size(1);
    return scale_fp32.unsqueeze(1)
        .unsqueeze(3)
        .repeat({1, block_rows, 1, block_cols})
        .reshape(
            {scale_fp32.size(0) * block_rows, scale_fp32.size(1) * block_cols})
        .slice(/*dim=*/0, /*start=*/0, /*end=*/weight.size(0))
        .slice(/*dim=*/1, /*start=*/0, /*end=*/weight.size(1));
  }
  if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(0)) {
    return scale_fp32.reshape({-1, 1});
  }
  if (scale_fp32.dim() == 1 && scale_fp32.size(0) == weight.size(1)) {
    return scale_fp32.reshape({1, -1});
  }
  LOG(FATAL) << "Unsupported MUSA FP8 weight scale shape " << scale_fp32.sizes()
             << " for weight " << weight.sizes();
  return torch::Tensor();
}

}  // namespace

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output) {
  (void)input_scale;
  CHECK(weight.defined() && weight.dim() == 2)
      << "MUSA FP8 linear requires a 2D weight.";
  CHECK(weight_scale.defined())
      << "MUSA FP8 linear requires a loaded weight scale.";

  auto input_2d = input.reshape({-1, input.size(-1)}).contiguous();
  const bool fp8_weight = weight.scalar_type() == torch::kFloat8_e4m3fn ||
                          weight.scalar_type() == torch::kFloat8_e5m2;
  const int64_t workspace_rows =
      output.has_value() ? kHcLaneExpandedWorkspaceRows : kDecodeWorkspaceRows;
  const char* disable_bf16_cache_env =
      std::getenv("XLLM_DSV4_DISABLE_FP8_BF16_CACHE");
  const bool disable_bf16_cache =
      disable_bf16_cache_env != nullptr &&
      std::string(disable_bf16_cache_env) == "1";
  if (fp8_weight && input_2d.size(0) > 0 &&
      input_2d.size(0) <= workspace_rows &&
      (output_dtype == torch::kBFloat16 || output_dtype == torch::kFloat16) &&
      !disable_bf16_cache) {
    auto& workspace = get_fp8_linear_workspace(weight);
    if (!workspace.dequantized_weight.defined() ||
        workspace.dequantized_weight.sizes() != weight.sizes() ||
        workspace.dequantized_weight.scalar_type() != output_dtype) {
      workspace.dequantized_weight =
          (weight.to(torch::kFloat32) *
           expand_fp8_weight_scale(weight_scale, weight))
              .to(output_dtype)
              .contiguous();
    }
    torch::Tensor output_workspace =
        get_requested_output_workspace(input_2d, weight, output_dtype, output);
    kernel::MatmulParams matmul_params;
    matmul_params.a = input_2d;
    matmul_params.b = workspace.dequantized_weight;
    matmul_params.bias = bias;
    matmul_params.c = output_workspace;
    auto output = kernel::matmul(matmul_params);
    auto output_shape = input.sizes().vec();
    output_shape.back() = weight.size(0);
    LOG_FIRST_N(INFO, 1)
        << "DeepSeek-V4 MUSA low-token BF16 linear cache enabled.";
    return output.view(output_shape);
  }

  const bool native_blockwise =
      input.scalar_type() == torch::kBFloat16 && fp8_weight &&
      weight_scale.scalar_type() == torch::kFloat32 &&
      weight_scale.dim() == 2 && weight.size(1) % 128 == 0 &&
      weight_scale.size(0) == (weight.size(0) + 127) / 128 &&
      weight_scale.size(1) == weight.size(1) / 128 &&
      (output_dtype == torch::kBFloat16 || output_dtype == torch::kFloat16) &&
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0;
  if (native_blockwise) {
    auto [quantized_input, input_group_scale] =
        kernel::musa::per_token_group_quant_fp8(input_2d, 128);
    if (quantized_input.defined() && input_group_scale.defined()) {
      torch::Tensor result =
          kernel::musa::gemm_fp8_nt_groupwise(quantized_input,
                                              weight.contiguous(),
                                              input_group_scale,
                                              weight_scale.contiguous(),
                                              output_dtype,
                                              output);
      if (result.defined()) {
        if (bias.has_value() && bias.value().defined()) {
          result.add_(bias.value());
        }
        auto output_shape = input.sizes().vec();
        output_shape.back() = weight.size(0);
        return result.view(output_shape);
      }
    }
  }

  auto dequantized_weight = (weight.to(torch::kFloat32) *
                             expand_fp8_weight_scale(weight_scale, weight))
                                .to(output_dtype);
  kernel::MatmulParams matmul_params;
  matmul_params.a = input_2d;
  matmul_params.b = dequantized_weight.contiguous();
  matmul_params.bias = bias;
  matmul_params.c = output;
  torch::Tensor result = kernel::matmul(matmul_params);
  auto output_shape = input.sizes().vec();
  output_shape.back() = weight.size(0);
  return result.view(output_shape);
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
