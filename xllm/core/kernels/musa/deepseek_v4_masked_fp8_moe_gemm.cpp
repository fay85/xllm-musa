/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <tvm/ffi/extra/stl.h>

#include <exception>
#include <fstream>
#include <string>
#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

bool is_fp8(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

bool has_supported_contract(const torch::Tensor& input,
                            const torch::Tensor& input_scale,
                            const torch::Tensor& weight,
                            const torch::Tensor& weight_scale,
                            const torch::Tensor& masked_tokens) {
  if (!input.defined() || !input_scale.defined() || !weight.defined() ||
      !weight_scale.defined() || !masked_tokens.defined() || input.dim() != 3 ||
      input_scale.dim() != 3 || weight.dim() != 3 || weight_scale.dim() != 3 ||
      masked_tokens.dim() != 1 || !is_fp8(input.scalar_type()) ||
      !is_fp8(weight.scalar_type()) ||
      input_scale.scalar_type() != torch::kFloat32 ||
      weight_scale.scalar_type() != torch::kFloat32 ||
      masked_tokens.scalar_type() != torch::kInt32 || input.stride(-1) != 1 ||
      weight.stride(-1) != 1 ||
      !(input_scale.stride(-1) == 1 || input_scale.stride(-2) == 1) ||
      !weight_scale.is_contiguous() || !masked_tokens.is_contiguous()) {
    return false;
  }

  const int64_t num_experts = input.size(0);
  const int64_t max_tokens = input.size(1);
  const int64_t k = input.size(2);
  const int64_t n = weight.size(1);
  if (num_experts <= 0 || max_tokens <= 0 || k <= 0 || n <= 0 || k % 128 != 0 ||
      n % 128 != 0 || weight.size(0) != num_experts || weight.size(2) != k ||
      input_scale.size(0) != num_experts || input_scale.size(1) != max_tokens ||
      input_scale.size(2) != k / 128 || weight_scale.size(0) != num_experts ||
      weight_scale.size(1) != n / 128 || weight_scale.size(2) != k / 128 ||
      masked_tokens.size(0) != num_experts) {
    return false;
  }

  return input.device() == input_scale.device() &&
         input.device() == weight.device() &&
         input.device() == weight_scale.device() &&
         input.device() == masked_tokens.device();
}

}  // namespace

bool deepseek_v4_masked_fp8_moe_available(const std::string& uri) {
  const std::string so_path = path_to_uri_so_lib(uri);
  if (!std::ifstream(so_path).good()) {
    return false;
  }
  try {
    const tvm::ffi::Function run = get_function(uri, "masked_moe_gemm_8bit");
    (void)run;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

torch::Tensor deepseek_v4_masked_fp8_moe_gemm(
    const torch::Tensor& input,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& masked_tokens,
    int64_t expected_tokens,
    const std::optional<torch::Tensor>& output,
    const std::string& uri) {
  if (!has_supported_contract(
          input, input_scale, weight, weight_scale, masked_tokens) ||
      expected_tokens <= 0) {
    return torch::Tensor();
  }

  const int64_t num_experts = input.size(0);
  const int64_t max_tokens = input.size(1);
  const int64_t n = weight.size(1);
  torch::Tensor out;
  if (output.has_value() && output->defined()) {
    out = *output;
    if (out.dim() != 3 || out.size(0) != num_experts ||
        out.size(1) != max_tokens || out.size(2) != n ||
        (out.scalar_type() != torch::kBFloat16 &&
         out.scalar_type() != torch::kFloat16) ||
        out.device() != input.device() || !out.is_contiguous()) {
      return torch::Tensor();
    }
  } else {
    out = torch::empty({num_experts, max_tokens, n},
                       input.options().dtype(torch::kBFloat16));
  }

  const std::string so_path = path_to_uri_so_lib(uri);
  if (!std::ifstream(so_path).good()) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_DEEPEP] Mate masked FP8 GEMM unavailable: " << so_path;
    return torch::Tensor();
  }

  try {
    bind_tvmffi_stream_to_current_torch_stream(input.device());
    tvm::ffi::Function run = get_function(uri, "masked_moe_gemm_8bit");
    run(std::make_tuple(to_ffi_tensor(input), to_ffi_tensor(input_scale)),
        std::make_tuple(to_ffi_tensor(weight), to_ffi_tensor(weight_scale)),
        to_ffi_tensor(masked_tokens),
        std::make_tuple(int64_t{1}, int64_t{128}, int64_t{128}),
        to_ffi_tensor(out),
        expected_tokens,
        to_ffi_optional_tensor(std::nullopt));
  } catch (const c10::Error& error) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_DEEPEP] Mate masked FP8 GEMM fallback: "
        << error.what_without_backtrace();
    return torch::Tensor();
  } catch (const std::exception& error) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_DEEPEP] Mate masked FP8 GEMM fallback: " << error.what();
    return torch::Tensor();
  }

  LOG_FIRST_N(INFO, 1) << "[DSV4_MUSA_DEEPEP] Mate masked FP8 GEMM active: E="
                       << num_experts << " M=" << max_tokens << " N=" << n
                       << " K=" << input.size(2);
  return out;
}

}  // namespace xllm::kernel::musa
