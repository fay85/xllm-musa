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

#include "layers/musa/linear.h"

#include <glog/logging.h>

#include <algorithm>

#include "util/env_var.h"

namespace xllm::layer::musa {
namespace {
constexpr int64_t kMatmulOutputBufMaxRows = 128;
}  // namespace

std::optional<torch::Tensor> MatmulOutputBuffers::get(
    const torch::Tensor& input,
    const torch::Tensor& weight) {
  if (input.dim() != 2 || weight.dim() != 2 || input.size(0) <= 0) {
    return std::nullopt;
  }
  const int64_t rows = input.size(0);
  if (rows > kMatmulOutputBufMaxRows) {
    return std::nullopt;
  }
  const int64_t columns = weight.size(0);
  const bool needs_realloc =
      !decode_output_buf_.defined() || decode_output_buf_.size(0) < rows ||
      decode_output_buf_.size(1) != columns ||
      decode_output_buf_.scalar_type() != input.scalar_type() ||
      decode_output_buf_.device() != input.device();
  if (needs_realloc) {
    const int64_t target_rows = decode_output_buf_.defined()
                                    ? std::max(rows, decode_output_buf_.size(0))
                                    : rows;
    decode_output_buf_ = torch::empty({target_rows, columns}, input.options());
  }
  return decode_output_buf_.narrow(0, 0, rows);
}

void set_persistent_output_buf(xllm::kernel::musa::MatmulParams& params,
                               MatmulOutputBuffers& output_buffers,
                               const torch::Tensor& input,
                               const torch::Tensor& weight) {
  params.output = output_buffers.get(input, weight);
}

namespace {

torch::Tensor dequantize_fp8_block_weight(const torch::Tensor& fp8_weight,
                                          const torch::Tensor& weight_scale_inv,
                                          int64_t block_n,
                                          int64_t block_k) {
  CHECK_EQ(fp8_weight.dim(), 2)
      << "block-fp8 weight must be 2D, got " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.dim(), 2)
      << "block-fp8 weight_scale_inv must be 2D, got "
      << weight_scale_inv.sizes();
  const int64_t n = fp8_weight.size(0);
  const int64_t k = fp8_weight.size(1);
  const int64_t n_tiles = (n + block_n - 1) / block_n;
  const int64_t k_tiles = (k + block_k - 1) / block_k;
  CHECK_EQ(weight_scale_inv.size(0), n_tiles)
      << "block-fp8 scale rows " << weight_scale_inv.sizes()
      << " mismatch weight " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.size(1), k_tiles)
      << "block-fp8 scale cols " << weight_scale_inv.sizes()
      << " mismatch weight " << fp8_weight.sizes();
  if (n % block_n == 0 && k % block_k == 0) {
    auto w = fp8_weight.to(torch::kBFloat16)
                 .reshape({n_tiles, block_n, k_tiles, block_k});
    auto s =
        weight_scale_inv.to(torch::kBFloat16).reshape({n_tiles, 1, k_tiles, 1});
    return (w * s).reshape({n, k});
  }
  auto expanded = weight_scale_inv.repeat_interleave(block_n, /*dim=*/0)
                      .repeat_interleave(block_k, /*dim=*/1)
                      .slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                      .slice(/*dim=*/1, /*start=*/0, /*end=*/k)
                      .to(torch::kBFloat16);
  return fp8_weight.to(torch::kBFloat16) * expanded;
}

torch::Tensor block_fp8_dequant_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& weight_scale_inv,
    const std::vector<int64_t>& weight_block_size,
    const std::optional<torch::Tensor>& bias,
    MatmulOutputBuffers& output_buffers) {
  const int64_t block_n = weight_block_size[0];
  const int64_t block_k = weight_block_size[1];
  auto weight_bf16 = dequantize_fp8_block_weight(
      weight_fp8, weight_scale_inv, block_n, block_k);
  xllm::kernel::musa::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_bf16;
  matmul_params.bias = bias;
  set_persistent_output_buf(matmul_params, output_buffers, input, weight_bf16);
  return xllm::kernel::musa::matmul(matmul_params);
}

torch::Tensor block_fp8_native_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& weight_scale_inv,
    const std::vector<int64_t>& weight_block_size,
    const std::optional<torch::Tensor>& bias,
    MatmulOutputBuffers& output_buffers) {
  const int64_t block_k = weight_block_size[1];
  auto in_shape = input.sizes().vec();
  const int64_t k = input.size(-1);
  CHECK_EQ(k % block_k, 0) << "native block-fp8 GEMM requires K % " << block_k
                           << " == 0, got K=" << k;
  torch::Tensor input_2d = input.reshape({-1, k}).contiguous();
  auto [a_fp8, a_scale] =
      xllm::kernel::musa::per_token_group_quant_fp8(input_2d, block_k);
  xllm::kernel::musa::Fp8BlockMatmulParams params;
  params.a = a_fp8;
  params.b = weight_fp8;
  params.a_scale = a_scale;
  CHECK_EQ(weight_scale_inv.scalar_type(), torch::kFloat32);
  CHECK(weight_scale_inv.is_contiguous());
  params.b_scale = weight_scale_inv;
  params.output_dtype = (input.scalar_type() == torch::kFloat16)
                            ? torch::kFloat16
                            : torch::kBFloat16;
  params.output = output_buffers.get(input_2d, weight_fp8);
  auto out = xllm::kernel::musa::fp8_block_matmul(params);
  if (bias.has_value() && bias.value().defined()) {
    out = out + bias.value().to(out.scalar_type());
  }
  in_shape.back() = weight_fp8.size(0);
  return out.reshape(in_shape);
}

}  // namespace

torch::Tensor block_fp8_forward(const torch::Tensor& input,
                                const torch::Tensor& weight_fp8,
                                const torch::Tensor& weight_scale_inv,
                                const std::vector<int64_t>& weight_block_size,
                                const std::optional<torch::Tensor>& bias,
                                MatmulOutputBuffers& output_buffers) {
  static const bool use_dequant = util::get_bool_env("XLLM_FP8_DEQUANT", false);
  if (use_dequant) {
    return block_fp8_dequant_forward(input,
                                     weight_fp8,
                                     weight_scale_inv,
                                     weight_block_size,
                                     bias,
                                     output_buffers);
  }
  return block_fp8_native_forward(input,
                                  weight_fp8,
                                  weight_scale_inv,
                                  weight_block_size,
                                  bias,
                                  output_buffers);
}

}  // namespace xllm::layer::musa
