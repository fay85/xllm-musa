/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "core/kernels/musa/musa_aot_config.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

void launch_moe_gemm(
    std::string_view mode,
    int64_t alignment,
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& input_scale = std::nullopt,
    const std::optional<torch::Tensor>& weight_scale = std::nullopt) {
  const bool fp8 = input.scalar_type() == torch::kFloat8_e4m3fn;
  const bool fp16 = output.scalar_type() == torch::kFloat16;
  CHECK(output.scalar_type() == torch::kBFloat16 || fp16);
  if (output.numel() == 0) {
    return;
  }
  if (!fp8) {
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_EQ(weights.scalar_type(), output.scalar_type());
  }
  const auto iterator = std::find_if(aot::kMoeGemmOperators.begin(),
                                     aot::kMoeGemmOperators.end(),
                                     [&](const aot::MoeGemmConfig& config) {
                                       return mode == config.mode &&
                                              alignment == config.alignment &&
                                              fp8 == config.fp8 &&
                                              fp16 == config.fp16;
                                     });
  CHECK(iterator != aot::kMoeGemmOperators.end())
      << "MUSA MoE GEMM configuration is not part of this build";
  const std::string uri = iterator->uri;
  const std::string object_path = get_musa_ops_root() + "/" + uri + "/kernel.o";
  TvmffiStreamGuard stream_guard(input.device());
  get_function(uri, uri)(object_path,
                         to_ffi_tensor_view(input),
                         to_ffi_tensor_view(weights),
                         to_ffi_tensor_view(output),
                         to_ffi_optional_tensor(input_scale),
                         to_ffi_optional_tensor(weight_scale),
                         /*scale_b1=*/to_ffi_optional_tensor(std::nullopt),
                         to_ffi_optional_tensor(tokens),
                         /*signal=*/to_ffi_optional_tensor(std::nullopt),
                         /*output_scale=*/to_ffi_optional_tensor(std::nullopt),
                         /*num_target_mp=*/std::optional<int64_t>());
}

void check_masked_moe_inputs(const torch::Tensor& input,
                             const torch::Tensor& weights,
                             const torch::Tensor& token_counts,
                             int64_t expected_tokens) {
  CHECK(input.defined() && weights.defined() && token_counts.defined());
  CHECK_EQ(input.dim(), 3) << "Masked MoE input must be [experts, tokens, K]";
  CHECK_EQ(weights.dim(), 3) << "Masked MoE weights must be [experts, N, K]";
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), input.size(0));
  CHECK_EQ(input.size(0), weights.size(0));
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(1) % (expected_tokens <= 128 ? 128 : 256), 0)
      << "Masked MoE storage must contain complete GEMM tiles";
  CHECK_EQ(input.size(2), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        token_counts.is_contiguous());
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);
  CHECK_GE(expected_tokens, 0);
}

}  // namespace

torch::Tensor masked_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& token_counts,
                                   torch::ScalarType output_dtype,
                                   int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);

  auto output = torch::empty({input.size(0), input.size(1), weights.size(1)},
                             input.options().dtype(output_dtype));
  launch_moe_gemm("masked",
                  expected_tokens <= 128 ? 128 : 256,
                  input,
                  weights,
                  token_counts,
                  output);
  return output;
}

torch::Tensor masked_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& token_counts,
                                  torch::ScalarType output_dtype,
                                  int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(input_scale.dim(), 3);
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1), input.size(1));
  CHECK_EQ(input_scale.size(2) * 128, input.size(2));
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));
  CHECK(input_scale.is_contiguous() && weight_scale.is_contiguous());

  auto output = torch::empty({input.size(0), input.size(1), weights.size(1)},
                             input.options().dtype(output_dtype));
  launch_moe_gemm("masked",
                  expected_tokens <= 128 ? 128 : 256,
                  input,
                  weights,
                  token_counts,
                  output,
                  input_scale,
                  weight_scale);
  return output;
}

torch::Tensor contiguous_moe_gemm_bf16(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& token_counts,
                                       torch::ScalarType output_dtype) {
  CHECK(input.defined() && weights.defined() && token_counts.defined())
      << "Mate contiguous BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous BF16 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous BF16 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous BF16 MoE tensors must be contiguous.";
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);

  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  launch_moe_gemm("contiguous", 128, input, weights, token_counts, output);
  return output;
}

torch::Tensor ragged_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& row_expert_ids,
                                   torch::ScalarType output_dtype,
                                   int64_t alignment) {
  CHECK(input.defined() && weights.defined() && row_expert_ids.defined())
      << "Mate Ragged BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged BF16 MoE tensors must be contiguous.";
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged BF16 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  launch_moe_gemm("ragged", alignment, input, weights, row_expert_ids, output);
  return output;
}

torch::Tensor contiguous_moe_gemm_fp8(const torch::Tensor& input,
                                      const torch::Tensor& input_scale,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& token_counts,
                                      torch::ScalarType output_dtype) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && token_counts.defined())
      << "Mate contiguous FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous FP8 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous FP8 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));

  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  launch_moe_gemm("contiguous",
                  128,
                  input,
                  weights,
                  token_counts,
                  output,
                  input_scale,
                  weight_scale);
  return output;
}

torch::Tensor ragged_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& row_expert_ids,
                                  torch::ScalarType output_dtype,
                                  int64_t alignment) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && row_expert_ids.defined())
      << "Mate Ragged FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged FP8 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  launch_moe_gemm("ragged",
                  alignment,
                  input,
                  weights,
                  row_expert_ids,
                  output,
                  input_scale,
                  weight_scale);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> moe_topk_softmax(
    const torch::Tensor& router_logits,
    int64_t topk) {
  CHECK(router_logits.defined());
  CHECK_EQ(router_logits.dim(), 2);
  CHECK(router_logits.is_contiguous());
  CHECK_EQ(router_logits.scalar_type(), torch::kBFloat16);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, router_logits.size(1));

  torch::Tensor logits = router_logits;
  return moe_fused_topk(logits,
                        topk,
                        /*renormalize=*/true,
                        /*correction_bias=*/std::nullopt,
                        /*scoring_func=*/"softmax");
}

}  // namespace xllm::kernel::musa
