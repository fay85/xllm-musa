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

#pragma once

#include <torch/torch.h>

#include <string>
#include <utility>

#include "layers/musa/qwen3_gated_delta_net_base.h"

namespace xllm {
namespace layer {

class Qwen3NextGatedDeltaNetImpl : public Qwen3GatedDeltaNetBaseImpl {
 public:
  Qwen3NextGatedDeltaNetImpl() = default;
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

 protected:
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options,
                             bool init_projections);

  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) override;
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) override;

  virtual void load_projection_state_dict(const StateDict& state_dict);
  virtual void verify_projection_weights(const std::string& prefix) const;

  void init_next_projections(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  ColumnParallelLinear qkvz_proj_{nullptr};
  ColumnParallelLinear ba_proj_{nullptr};
};
TORCH_MODULE(Qwen3NextGatedDeltaNet);

class Qwen3_5GatedDeltaNetImpl final : public Qwen3NextGatedDeltaNetImpl {
 public:
  Qwen3_5GatedDeltaNetImpl() = default;
  Qwen3_5GatedDeltaNetImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

 protected:
  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) override;
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) override;
  bool use_fla_ssm_state_layout() const override { return true; }
  bool uses_contiguous_qkvzba_layout() const override { return true; }

  void load_projection_state_dict(const StateDict& state_dict) override;
  void verify_projection_weights(const std::string& prefix) const override;

 private:
  bool use_merged_projections() const {
    return static_cast<bool>(qkvz_proj_);
  }

  torch::Tensor merge_qkvz_from_split_activations(const torch::Tensor& qkv,
                                                  const torch::Tensor& z) const;
  torch::Tensor merge_ba_from_split_activations(const torch::Tensor& b,
                                                const torch::Tensor& a) const;

  ColumnParallelLinear in_proj_qkv_{nullptr};
  ColumnParallelLinear in_proj_z_{nullptr};
  ColumnParallelLinear in_proj_b_{nullptr};
  ColumnParallelLinear in_proj_a_{nullptr};

  // Persistent buffers that replace the two `torch::cat` calls in
  // merge_qkvz_from_split_activations / merge_ba_from_split_activations.
  // Only used on the TP>1 fallback path (4 separate projections); the TP=1
  // merged path (qkvz_proj_ / ba_proj_) emits the concatenated output
  // directly from a single matmul, so no merge is needed.
  mutable torch::Tensor qkvz_merge_buf_;
  mutable torch::Tensor ba_merge_buf_;
};
TORCH_MODULE(Qwen3_5GatedDeltaNet);

}  // namespace layer
}  // namespace xllm
