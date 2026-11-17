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
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/musa/deepseek_v41_dense_block32.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::layer {

// Native dense projection: the default uses an exact BF16 weight carrier and
// activation QDQ. Explicit FP8 weights/scales select offline-compiled kernels.
class DeepseekV41Dense final {
 public:
  DeepseekV41Dense(torch::Tensor weight,
                   bool original_fp8_activation,
                   bool verify_skinny_projection = false,
                   torch::Tensor fp8_scale = {},
                   std::string prequant_directory = {})
      : weight_(std::move(weight)),
        original_fp8_activation_(original_fp8_activation),
        verify_skinny_projection_(verify_skinny_projection),
        fp8_scale_(std::move(fp8_scale)),
        prequant_directory_(std::move(prequant_directory)) {
    CHECK(weight_.defined());
    CHECK(weight_.device().type() == torch::kPrivateUse1);
    CHECK_EQ(weight_.scalar_type(),
             fp8_scale_.defined() ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    CHECK_EQ(weight_.dim(), 2);
    CHECK_GT(weight_.size(0), 0);
    CHECK_GT(weight_.size(1), 0);
    CHECK(!original_fp8_activation_ || weight_.size(1) % 32 == 0);
    if (fp8_scale_.defined()) {
      CHECK(original_fp8_activation_);
      CHECK(!prequant_directory_.empty());
      CHECK(weight_.is_contiguous() && fp8_scale_.is_contiguous());
      CHECK_EQ(fp8_scale_.device(), weight_.device());
      CHECK_EQ(fp8_scale_.scalar_type(), torch::kFloat32);
      CHECK_EQ(
          fp8_scale_.sizes(),
          torch::IntArrayRef({weight_.size(0) / 32, weight_.size(1) / 32}));
      CHECK_EQ(weight_.size(0) % 32, 0);
      CHECK_EQ(weight_.size(1) % 32, 0);
    } else {
      CHECK(prequant_directory_.empty());
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    CHECK(input.defined());
    CHECK_EQ(input.device(), weight_.device());
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_GE(input.dim(), 2);
    CHECK_EQ(input.size(-1), weight_.size(1));
    torch::DeviceGuard guard(input.device());
    if (fp8_scale_.defined()) {
      return fp8_forward(input);
    }
    torch::Tensor value = input;
    if (original_fp8_activation_) {
      value =
          std::get<0>(kernel::musa::deepseek_v41_block32_qdq(
                          input.reshape({-1, weight_.size(1)}).contiguous()))
              .reshape(input.sizes());
    }
    return project(value, weight_, verify_skinny_projection_);
  }

  static torch::Tensor project(const torch::Tensor& input,
                               const torch::Tensor& weight,
                               bool verify_skinny_projection) {
    if (verify_skinny_projection &&
        kernel::musa::deepseek_v41_q4_projection_supported(input, weight)) {
      return kernel::musa::deepseek_v41_q4_projection(input, weight);
    }
    return torch::linear(input, weight);
  }

 private:
  torch::Tensor fp8_forward(const torch::Tensor& input) const {
    const int64_t columns = weight_.size(0);
    const int64_t hidden = weight_.size(1);
    std::vector<int64_t> shape = input.sizes().vec();
    shape.back() = columns;
    if (input.numel() == 0) {
      return torch::empty(shape, input.options());
    }
    int64_t splits = columns == 1280 || columns == 512 ? 16 : 8;
    if (hidden == 1280 || hidden == 8192 || hidden == 1024) {
      splits = 4;
    }
    while (hidden % (32 * splits) != 0) {
      splits /= 2;
    }
    const torch::Tensor flat = input.reshape({-1, hidden}).contiguous();
    const int64_t rows = flat.size(0);
    // Small decode/verify runs one graph-compatible projection. Larger D-side
    // CED replay is explicitly tiled; it never retains a BF16 weight replica.
    if (rows <= 32) {
      return kernel::musa::deepseek_v41_dense_prequant_block32(
                 flat, weight_, fp8_scale_, splits, prequant_directory_)
          .reshape(shape);
    }
    std::vector<torch::Tensor> output;
    output.reserve((rows + 31) / 32);
    for (int64_t start = 0; start < rows; start += 32) {
      output.emplace_back(kernel::musa::deepseek_v41_dense_prequant_block32(
          flat.narrow(0, start, std::min<int64_t>(32, rows - start)),
          weight_,
          fp8_scale_,
          splits,
          prequant_directory_));
    }
    return torch::cat(output, 0).reshape(shape);
  }

  torch::Tensor weight_;
  bool original_fp8_activation_;
  bool verify_skinny_projection_;
  torch::Tensor fp8_scale_;
  std::string prequant_directory_;
};

// Shared/dense expert with BF16 or explicit FP8 projections. Both paths use
// the same SwiGLU semantics; decoder loading retains tensor ownership.
class DeepseekV41SharedMlp final {
 public:
  DeepseekV41SharedMlp(torch::Tensor gate,
                       torch::Tensor up,
                       torch::Tensor down,
                       double limit,
                       bool original_fp8_activation = false,
                       bool decode_packed_gate_up = false,
                       bool verify_skinny_projection = false,
                       const std::vector<torch::Tensor>& fp8_scales = {},
                       const std::string& prequant_directory = {})
      : gate_(std::move(gate)),
        up_(std::move(up)),
        down_(std::move(down)),
        limit_(limit),
        original_fp8_activation_(original_fp8_activation),
        verify_skinny_projection_(verify_skinny_projection) {
    CHECK(gate_.defined() && up_.defined() && down_.defined());
    CHECK(gate_.device().type() == torch::kPrivateUse1);
    const torch::ScalarType dtype =
        fp8_scales.empty() ? torch::kBFloat16 : torch::kFloat8_e4m3fn;
    CHECK_EQ(gate_.scalar_type(), dtype);
    CHECK_EQ(up_.scalar_type(), dtype);
    CHECK_EQ(down_.scalar_type(), dtype);
    CHECK_EQ(gate_.device(), up_.device());
    CHECK_EQ(gate_.device(), down_.device());
    CHECK_EQ(gate_.dim(), 2);
    CHECK_EQ(up_.sizes(), gate_.sizes());
    CHECK_EQ(down_.dim(), 2);
    CHECK_EQ(down_.size(0), gate_.size(1));
    CHECK_EQ(down_.size(1), gate_.size(0));
    CHECK_GT(gate_.size(0), 0);
    CHECK_GT(gate_.size(1), 0);
    CHECK(!original_fp8_activation_ ||
          (gate_.size(0) % 32 == 0 && gate_.size(1) % 32 == 0));
    CHECK(std::isfinite(limit_) && limit_ > 0.0 && limit_ < 1000000.0);
    if (!fp8_scales.empty()) {
      CHECK_EQ(fp8_scales.size(), 3);
      CHECK(!decode_packed_gate_up)
          << "FP8 shared candidate keeps separate gate/up projections";
      fp8_gate_ = std::make_unique<DeepseekV41Dense>(gate_,
                                                     original_fp8_activation_,
                                                     false,
                                                     fp8_scales[0],
                                                     prequant_directory);
      fp8_up_ = std::make_unique<DeepseekV41Dense>(up_,
                                                   original_fp8_activation_,
                                                   false,
                                                   fp8_scales[1],
                                                   prequant_directory);
      fp8_down_ = std::make_unique<DeepseekV41Dense>(down_,
                                                     original_fp8_activation_,
                                                     false,
                                                     fp8_scales[2],
                                                     prequant_directory);
    } else {
      CHECK(prequant_directory.empty());
    }
    if (decode_packed_gate_up) {
      packed_gate_up_ = torch::cat({gate_, up_}, 0).contiguous();
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    CHECK(input.defined());
    CHECK_EQ(input.device(), gate_.device());
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.size(1), gate_.size(1));
    torch::DeviceGuard guard(input.device());
    if (input.size(0) == 0) {
      return torch::empty_like(input);
    }
    if (fp8_gate_ != nullptr) {
      const torch::Tensor packed =
          torch::cat({fp8_gate_->forward(input), fp8_up_->forward(input)}, -1);
      const torch::Tensor activation =
          torch::empty({input.size(0), gate_.size(0)}, input.options());
      kernel::musa::deepseek_v4_shared_swiglu(packed, limit_, activation);
      return fp8_down_->forward(activation);
    }
    const torch::Tensor value = quantize(input);
    torch::Tensor packed;
    if (packed_gate_up_.defined() && input.size(0) == 1) {
      packed = torch::linear(value, packed_gate_up_);
    } else {
      packed = torch::cat(
          {DeepseekV41Dense::project(value, gate_, verify_skinny_projection_),
           DeepseekV41Dense::project(value, up_, verify_skinny_projection_)},
          -1);
    }
    const torch::Tensor activation =
        torch::empty({input.size(0), gate_.size(0)}, input.options());
    kernel::musa::deepseek_v4_shared_swiglu(packed, limit_, activation);
    return torch::linear(quantize(activation), down_);
  }

 private:
  torch::Tensor quantize(const torch::Tensor& input) const {
    if (original_fp8_activation_) {
      return std::get<0>(
          kernel::musa::deepseek_v41_block32_qdq(input.contiguous()));
    }
    return input;
  }

  torch::Tensor gate_;
  torch::Tensor up_;
  torch::Tensor packed_gate_up_;
  torch::Tensor down_;
  std::unique_ptr<DeepseekV41Dense> fp8_gate_;
  std::unique_ptr<DeepseekV41Dense> fp8_up_;
  std::unique_ptr<DeepseekV41Dense> fp8_down_;
  double limit_;
  bool original_fp8_activation_;
  bool verify_skinny_projection_;
};

// Small-M original-FP4 routed expert. No Python forward is called here.
// Quantization is selected at construction and applied to ALL three linears.
class DeepseekV41PackedRoutedMlp final {
 public:
  DeepseekV41PackedRoutedMlp(torch::Tensor gate,
                             torch::Tensor gate_scale,
                             torch::Tensor up,
                             torch::Tensor up_scale,
                             torch::Tensor down,
                             torch::Tensor down_scale,
                             double limit,
                             bool block32_quant)
      : weights_{std::move(gate), std::move(up), std::move(down)},
        scales_{std::move(gate_scale),
                std::move(up_scale),
                std::move(down_scale)},
        limit_(limit),
        block32_quant_(block32_quant) {
    CHECK(std::isfinite(limit_) && limit_ > 0.0 && limit_ < 1000000.0);
    for (int32_t i = 0; i < 3; ++i) {
      CHECK(weights_[i].defined() && scales_[i].defined());
      CHECK(weights_[i].device().type() == torch::kPrivateUse1);
      CHECK_EQ(weights_[i].device(), weights_[0].device());
      CHECK_EQ(scales_[i].device(), weights_[0].device());
      CHECK_EQ(weights_[i].scalar_type(), torch::kUInt8);
      CHECK_EQ(scales_[i].scalar_type(), torch::kUInt8);
      CHECK_EQ(weights_[i].dim(), 2);
      CHECK_EQ(scales_[i].dim(), 2);
      CHECK(weights_[i].is_contiguous() && scales_[i].is_contiguous());
      CHECK_GT(weights_[i].size(0), 0);
      CHECK_GT(weights_[i].size(1), 0);
      CHECK_EQ(weights_[i].size(1) % 16, 0);
      CHECK_EQ(scales_[i].size(0), weights_[i].size(0));
      CHECK_EQ(scales_[i].size(1) * 16, weights_[i].size(1));
    }
    CHECK_EQ(weights_[0].sizes(), weights_[1].sizes());
    CHECK_EQ(weights_[2].size(0), weights_[0].size(1) * 2);
    CHECK_EQ(weights_[2].size(1) * 2, weights_[0].size(0));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& route_weight) const {
    CHECK(input.defined() && route_weight.defined());
    CHECK_EQ(input.device(), weights_[0].device());
    CHECK_EQ(input.scalar_type(), torch::kBFloat16);
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.size(1), weights_[0].size(1) * 2);
    CHECK_LE(input.size(0), 8);
    CHECK_EQ(route_weight.device(), input.device());
    CHECK_EQ(route_weight.scalar_type(), torch::kFloat32);
    CHECK_EQ(route_weight.dim(), 1);
    CHECK_EQ(route_weight.size(0), input.size(0));
    torch::DeviceGuard guard(input.device());
    if (input.size(0) == 0) {
      return torch::empty_like(input);
    }
    const torch::Tensor quantized = quantize(input);
    const torch::Tensor gate = kernel::musa::deepseek_v41_packed_fp4_gemv(
                                   quantized, weights_[0], scales_[0])
                                   .to(torch::kFloat32);
    const torch::Tensor up = kernel::musa::deepseek_v41_packed_fp4_gemv(
                                 quantized, weights_[1], scales_[1])
                                 .to(torch::kFloat32);
    // Weighting after the down projection changes rounding/quantization.
    const torch::Tensor hidden =
        (torch::silu(torch::clamp_max(gate, limit_)) *
         torch::clamp(up, -limit_, limit_) * route_weight.unsqueeze(-1))
            .to(torch::kBFloat16);
    return kernel::musa::deepseek_v41_packed_fp4_gemv(
        quantize(hidden), weights_[2], scales_[2]);
  }

 private:
  torch::Tensor quantize(const torch::Tensor& input) const {
    const torch::Tensor contiguous = input.contiguous();
    if (block32_quant_) {
      return std::get<0>(kernel::musa::deepseek_v41_block32_qdq(contiguous));
    }
    return contiguous;
  }

  std::array<torch::Tensor, 3> weights_;
  std::array<torch::Tensor, 3> scales_;
  double limit_;
  bool block32_quant_;
};

// Owns a complete local EP slice; executor performs the collective once.
class DeepseekV41PackedRoutedExperts final {
 public:
  DeepseekV41PackedRoutedExperts(torch::Tensor gate,
                                 torch::Tensor gate_scale,
                                 torch::Tensor up,
                                 torch::Tensor up_scale,
                                 torch::Tensor down,
                                 torch::Tensor down_scale,
                                 int64_t first_expert,
                                 double limit,
                                 bool block32_quant)
      : weights_{std::move(gate), std::move(up), std::move(down)},
        scales_{std::move(gate_scale),
                std::move(up_scale),
                std::move(down_scale)},
        first_expert_(first_expert),
        limit_(limit),
        block32_quant_(block32_quant) {}

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& expert_ids,
                        const torch::Tensor& route_weights) const {
    return kernel::musa::deepseek_v41_decode_moe(input.contiguous(),
                                                 expert_ids.contiguous(),
                                                 route_weights.contiguous(),
                                                 weights_[0],
                                                 scales_[0],
                                                 weights_[1],
                                                 scales_[1],
                                                 weights_[2],
                                                 scales_[2],
                                                 first_expert_,
                                                 limit_,
                                                 block32_quant_);
  }

 private:
  std::array<torch::Tensor, 3> weights_;
  std::array<torch::Tensor, 3> scales_;
  int64_t first_expert_;
  double limit_;
  bool block32_quant_;
};

// Large-M expert candidate using the SAME native MATE adapter as Python.
class DeepseekV41PartitionRoutedMlp final {
 public:
  DeepseekV41PartitionRoutedMlp(torch::Tensor gate,
                                torch::Tensor gate_scale,
                                torch::Tensor up,
                                torch::Tensor up_scale,
                                torch::Tensor down,
                                torch::Tensor down_scale,
                                double limit,
                                std::string bmm_so_path,
                                int64_t max_partial_bytes)
      : weights_{std::move(gate), std::move(up), std::move(down)},
        scales_{std::move(gate_scale),
                std::move(up_scale),
                std::move(down_scale)},
        limit_(limit),
        bmm_so_path_(std::move(bmm_so_path)),
        max_partial_bytes_(max_partial_bytes) {
    CHECK(std::isfinite(limit_) && limit_ > 0.0 && limit_ < 1000000.0);
    CHECK(!bmm_so_path_.empty());
    CHECK_GT(max_partial_bytes_, 0);
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& route_weight) const {
    CHECK(input.defined() && route_weight.defined());
    CHECK_EQ(route_weight.device(), input.device());
    CHECK_EQ(route_weight.scalar_type(), torch::kFloat32);
    CHECK_EQ(route_weight.dim(), 1);
    CHECK_EQ(route_weight.size(0), input.size(0));
    const torch::Tensor gate = linear(input, 0).to(torch::kFloat32);
    const torch::Tensor up = linear(input, 1).to(torch::kFloat32);
    const torch::Tensor hidden =
        (torch::silu(torch::clamp_max(gate, limit_)) *
         torch::clamp(up, -limit_, limit_) * route_weight.unsqueeze(-1))
            .to(torch::kBFloat16);
    return linear(hidden, 2);
  }

 private:
  torch::Tensor linear(const torch::Tensor& input, int32_t index) const {
    return kernel::musa::deepseek_v41_partition_linear(
        input.contiguous(),
        weights_[index].contiguous(),
        scales_[index].contiguous(),
        bmm_so_path_,
        max_partial_bytes_);
  }

  std::array<torch::Tensor, 3> weights_;
  std::array<torch::Tensor, 3> scales_;
  double limit_;
  std::string bmm_so_path_;
  int64_t max_partial_bytes_;
};

class DeepseekV41Gate final {
 public:
  DeepseekV41Gate(torch::Tensor weight,
                  torch::Tensor bias,
                  std::optional<torch::Tensor> vision_bias,
                  int64_t image_token_id,
                  int64_t topk,
                  bool normalize,
                  double scale,
                  bool cache_fp32_weight = false,
                  bool use_bf16_fp32_projection = false)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        vision_bias_(vision_bias.value_or(torch::Tensor{})),
        image_token_id_(image_token_id),
        topk_(topk),
        normalize_(normalize),
        scale_(scale) {
    CHECK(weight_.defined() && bias_.defined());
    CHECK(weight_.device().type() == torch::kPrivateUse1);
    CHECK(weight_.scalar_type() == torch::kBFloat16 ||
          weight_.scalar_type() == torch::kFloat32);
    CHECK_EQ(bias_.device(), weight_.device());
    CHECK_EQ(bias_.scalar_type(), torch::kFloat32);
    CHECK_EQ(weight_.dim(), 2);
    CHECK_GT(weight_.size(1), 0);
    CHECK_EQ(bias_.dim(), 1);
    CHECK_EQ(bias_.size(0), weight_.size(0));
    CHECK_GT(topk_, 0);
    CHECK_LE(topk_, weight_.size(0));
    CHECK_LE(weight_.size(0), 1024);
    CHECK(std::isfinite(scale_));
    if (use_bf16_fp32_projection && weight_.scalar_type() == torch::kBFloat16) {
      // Retain the original BF16 operand before optional FP32 caching. Never
      // round a genuinely FP32 checkpoint merely to select this backend.
      weight_bf16_ = weight_;
    }
    if (cache_fp32_weight) {
      torch::DeviceGuard guard(weight_.device());
      // The weight is immutable after model construction. Materialize exactly
      // the same BF16-to-FP32 conversion once, without changing routing math.
      weight_ = weight_.to(torch::kFloat32);
    }
    if (vision_bias_.defined()) {
      CHECK_EQ(vision_bias_.device(), bias_.device());
      CHECK_EQ(vision_bias_.scalar_type(), torch::kFloat32);
      CHECK_EQ(vision_bias_.sizes(), bias_.sizes());
    }
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& input_ids) const {
    CHECK(input.defined());
    CHECK_EQ(input.device(), weight_.device());
    CHECK(input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat32);
    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(input.size(1), weight_.size(1));
    torch::DeviceGuard guard(input.device());
    torch::Tensor bias = bias_;
    if (vision_bias_.defined() && input_ids.has_value() &&
        input_ids->defined()) {
      CHECK_EQ(input_ids->device(), input.device());
      CHECK_EQ(input_ids->numel(), input.size(0));
      CHECK(input_ids->scalar_type() == torch::kInt32 ||
            input_ids->scalar_type() == torch::kInt64);
      bias = torch::where(
          input_ids->reshape({-1, 1}).eq(image_token_id_), vision_bias_, bias_);
    }
    torch::Tensor logits;
    if (weight_bf16_.defined() && input.scalar_type() == torch::kBFloat16 &&
        input.size(0) <= 6) {
      // Reuse the V4 MATE BF16-input / FP32-output projection. The guard
      // orders eager pool-stream producers/consumers and preserves capture.
      {
        kernel::musa::TvmffiStreamGuard stream_guard(input.device());
        logits = kernel::musa::deepseek_v4_gate_bf16_fp32(
            input, weight_bf16_, /*output=*/std::nullopt);
      }
      CHECK(logits.defined())
          << "Required V4.1 BF16->FP32 MATE Gate projection failed";
      CHECK_EQ(logits.scalar_type(), torch::kFloat32);
    } else {
      logits =
          torch::linear(input.to(torch::kFloat32), weight_.to(torch::kFloat32));
    }
    return kernel::musa::deepseek_v41_gate_topk(
        logits.contiguous(), bias.contiguous(), topk_, normalize_, scale_);
  }

 private:
  torch::Tensor weight_;
  torch::Tensor weight_bf16_;
  torch::Tensor bias_;
  torch::Tensor vision_bias_;
  int64_t image_token_id_;
  int64_t topk_;
  bool normalize_;
  double scale_;
};
}  // namespace xllm::layer
