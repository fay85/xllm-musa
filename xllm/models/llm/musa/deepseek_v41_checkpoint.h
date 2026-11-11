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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/state_dict/state_dict.h"

namespace xllm {

// Reads original mapped Safetensors, not a converted checkpoint. The caller
// retains the StateDict/mmap lifetime through model construction. CPU math
// here is offline weight loading; no model forward uses this decoder.
class DeepseekV41Checkpoint final {
 public:
  DeepseekV41Checkpoint(const std::vector<std::unique_ptr<StateDict>>& shards,
                        torch::TensorOptions options,
                        int64_t tp_rank,
                        int64_t tp_size)
      : shards_(shards),
        options_(std::move(options)),
        tp_rank_(tp_rank),
        tp_size_(tp_size) {
    CHECK(!shards_.empty());
    CHECK_GT(tp_size_, 0);
    CHECK_GE(tp_rank_, 0);
    CHECK_LT(tp_rank_, tp_size_);
    CHECK(options_.device().type() == torch::kPrivateUse1);
    CHECK(options_.dtype().toScalarType() == torch::kBFloat16);
  }

  torch::Tensor tensor(const std::string& name) const {
    for (const auto& shard : shards_) {
      torch::Tensor value = shard->get_tensor(name);
      if (value.defined()) {
        return value;
      }
    }
    LOG(FATAL) << "Required original V4.1 checkpoint tensor missing: " << name;
    return {};
  }

  bool has(const std::string& name) const {
    for (const auto& shard : shards_) {
      if (shard->has(name)) {
        return true;
      }
    }
    return false;
  }

  torch::Tensor copy(const std::string& name,
                     torch::ScalarType dtype,
                     std::optional<int64_t> shard_dim = std::nullopt) const {
    return shard(tensor(name), shard_dim)
        .to(options_.dtype(dtype))
        .contiguous();
  }

  torch::Tensor dense(const std::string& prefix,
                      std::optional<int64_t> shard_dim = std::nullopt) const {
    // Match the Python loader: original whole-matrix block scales first,
    // then TP slicing. Do not rebase scale blocks after a partial TP slice.
    const torch::Tensor decoded =
        dequant_fp8(tensor(prefix + ".weight"), tensor(prefix + ".scale"));
    return shard(decoded, shard_dim).to(options_).contiguous();
  }

  std::pair<torch::Tensor, torch::Tensor> experts(const std::string& prefix,
                                                  const std::string& projection,
                                                  int64_t first,
                                                  int64_t count) const {
    CHECK_GE(first, 0);
    CHECK_GT(count, 0);
    const std::string initial =
        prefix + "." + std::to_string(first) + "." + projection;
    const torch::Tensor first_weight = tensor(initial + ".weight");
    const torch::Tensor first_scale = tensor(initial + ".scale");
    CHECK_EQ(first_weight.dim(), 2);
    CHECK_EQ(first_scale.dim(), 2);
    CHECK_EQ(first_weight.element_size(), 1);
    CHECK_EQ(first_scale.element_size(), 1);
    CHECK_EQ(first_scale.size(0), first_weight.size(0));
    CHECK_EQ(first_scale.size(1) * 16, first_weight.size(1));
    const torch::Tensor weights =
        torch::empty({count, first_weight.size(0), first_weight.size(1)},
                     options_.dtype(torch::kUInt8));
    const torch::Tensor scales =
        torch::empty({count, first_scale.size(0), first_scale.size(1)},
                     options_.dtype(torch::kUInt8));
    for (int64_t index = 0; index < count; ++index) {
      const std::string key =
          prefix + "." + std::to_string(first + index) + "." + projection;
      const torch::Tensor weight = tensor(key + ".weight");
      const torch::Tensor scale = tensor(key + ".scale");
      CHECK_EQ(weight.sizes(), first_weight.sizes());
      CHECK_EQ(scale.sizes(), first_scale.sizes());
      CHECK_EQ(weight.element_size(), 1);
      CHECK_EQ(scale.element_size(), 1);
      // One direct H2D per slice. Never expand the full EP expert set to BF16
      // or materialize a second GPU stack with cat/stack during loading.
      weights.select(0, index).copy_(weight.view(torch::kUInt8));
      scales.select(0, index).copy_(scale.view(torch::kUInt8));
    }
    return {weights, scales};
  }

  static torch::Tensor dequant_fp8(const torch::Tensor& weight,
                                   const torch::Tensor& scale) {
    CHECK(weight.defined() && scale.defined());
    CHECK(weight.device().is_cpu() && scale.device().is_cpu());
    CHECK_EQ(weight.dim(), 2);
    CHECK_EQ(scale.dim(), 2);
    CHECK_EQ(weight.element_size(), 1);
    CHECK_EQ(scale.element_size(), 1);
    CHECK(weight.scalar_type() == torch::kUInt8 ||
          weight.scalar_type() == torch::kFloat8_e4m3fn);
    CHECK_EQ(scale.size(0), (weight.size(0) + 31) / 32);
    CHECK_EQ(scale.size(1), (weight.size(1) + 31) / 32);
    const torch::Tensor bytes = scale.view(torch::kUInt8).to(torch::kInt32);
    CHECK(!bytes.eq(255).any().item<bool>()) << "UE8M0 scale 255 is NaN";
    const torch::Tensor bits = torch::where(
        bytes.eq(0), 0x00400000, torch::bitwise_left_shift(bytes, 23));
    const torch::Tensor values =
        weight.contiguous().view(torch::kFloat8_e4m3fn).to(torch::kFloat32);
    const torch::Tensor factors = bits.contiguous()
                                      .view(torch::kFloat32)
                                      .repeat_interleave(32, 0)
                                      .repeat_interleave(32, 1)
                                      .narrow(0, 0, weight.size(0))
                                      .narrow(1, 0, weight.size(1));
    return (values * factors).to(torch::kBFloat16);
  }

 private:
  torch::Tensor shard(const torch::Tensor& value,
                      const std::optional<int64_t>& dim) const {
    if (!dim.has_value() || tp_size_ == 1) {
      return value;
    }
    CHECK_GE(dim.value(), 0);
    CHECK_LT(dim.value(), value.dim());
    CHECK_EQ(value.size(dim.value()) % tp_size_, 0);
    const int64_t width = value.size(dim.value()) / tp_size_;
    return value.narrow(dim.value(), tp_rank_ * width, width);
  }

  const std::vector<std::unique_ptr<StateDict>>& shards_;
  torch::TensorOptions options_;
  int64_t tp_rank_;
  int64_t tp_size_;
};

}  // namespace xllm
