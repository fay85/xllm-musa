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

#include <optional>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/dense_mlp.h"

namespace xllm {
namespace layer {

class DeepseekV4MoEImpl final : public torch::nn::Module {
 public:
  DeepseekV4MoEImpl() = default;
  DeepseekV4MoEImpl(const ModelArgs& model_args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);

  torch::Tensor forward_with_selected_experts(
      const torch::Tensor& hidden_states,
      const torch::Tensor& topk_weights,
      const torch::Tensor& topk_ids,
      const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

 private:
  void load_experts(const StateDict& state_dict);
  torch::Tensor apply_expert(const torch::Tensor& input,
                             int64_t local_expert_id) const;
  void ensure_dense_expert_storage();
  void ensure_fp8_expert_storage();
  void ensure_packed_fp4_expert_storage();
  bool try_load_packed_fp4_experts(const StateDict& state_dict,
                                   const std::vector<std::string>& w13_prefixes,
                                   const std::string& w2_prefix,
                                   int64_t rank,
                                   int64_t world_size);
  bool try_load_fp8_experts(const StateDict& state_dict,
                            const std::vector<std::string>& w13_prefixes,
                            const std::string& w2_prefix,
                            int64_t rank,
                            int64_t world_size);
  bool fp4_experts_loaded() const {
    return w13_packed_is_loaded_ && w2_packed_is_loaded_;
  }
  bool fp8_experts_loaded() const {
    return w13_fp8_is_loaded_ && w2_fp8_is_loaded_;
  }

  int64_t num_total_experts_ = 0;
  int64_t topk_ = 1;
  int64_t hidden_size_ = 0;
  int64_t intermediate_size_ = 0;
  int64_t local_intermediate_size_ = 0;
  int64_t num_experts_per_rank_ = 0;
  int64_t start_expert_id_ = 0;
  int64_t n_shared_experts_ = 0;
  double swiglu_limit_ = 0.0;
  std::string hidden_act_ = "silu";
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_ = nullptr;
  ProcessGroup* ep_pg_ = nullptr;

  DenseMLP shared_experts_{nullptr};
  torch::nn::Linear shared_expert_gate_{nullptr};
  bool shared_expert_gate_is_loaded_ = false;

  bool packed_fp4_experts_detected_ = false;
  bool fp8_experts_detected_ = false;
  torch::Tensor w13_packed_;
  torch::Tensor w13_scale_;
  torch::Tensor w2_packed_;
  torch::Tensor w2_scale_;
  bool w13_packed_is_loaded_ = false;
  bool w2_packed_is_loaded_ = false;
  std::vector<torch::Tensor> w1_packed_list_;
  std::vector<torch::Tensor> w3_packed_list_;
  std::vector<torch::Tensor> w1_scale_list_;
  std::vector<torch::Tensor> w3_scale_list_;
  std::vector<torch::Tensor> w2_packed_list_;
  std::vector<torch::Tensor> w2_scale_list_;

  torch::Tensor w13_fp8_;
  torch::Tensor w13_fp8_scale_;
  torch::Tensor w2_fp8_;
  torch::Tensor w2_fp8_scale_;
  bool w13_fp8_is_loaded_ = false;
  bool w2_fp8_is_loaded_ = false;
  std::vector<torch::Tensor> w1_fp8_list_;
  std::vector<torch::Tensor> w3_fp8_list_;
  std::vector<torch::Tensor> w1_fp8_scale_list_;
  std::vector<torch::Tensor> w3_fp8_scale_list_;
  std::vector<torch::Tensor> w2_fp8_list_;
  std::vector<torch::Tensor> w2_fp8_scale_list_;

  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w2);
};

TORCH_MODULE(DeepseekV4MoE);

}  // namespace layer
}  // namespace xllm
