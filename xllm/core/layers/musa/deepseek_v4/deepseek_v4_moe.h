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
#include "core/layers/common/activation.h"
#include "core/layers/common/linear.h"

namespace xllm {
namespace layer {
namespace dsv4_musa_sp {
struct DeepseekV4SPContext;
}

// DeepSeek-V4 shared experts use a MUSA-only low-token FP8 path.  The
// fallback remains the same parallel-linear implementation used by the
// original shared expert, so non-MUSA and prefill behavior are unchanged.
class DeepseekV4SharedExpertImpl final : public torch::nn::Module {
 public:
  DeepseekV4SharedExpertImpl() = default;
  DeepseekV4SharedExpertImpl(int64_t hidden_size,
                             int64_t intermediate_size,
                             const std::string& hidden_act,
                             const QuantArgs& quant_args,
                             ProcessGroup* process_group,
                             const torch::TensorOptions& options,
                             double swiglu_limit);

  torch::Tensor forward(const torch::Tensor& hidden_states);

  void load_state_dict(const StateDict& state_dict);
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& gate_up_names,
                       const std::string& down_name);

 private:
  bool fp8_decode_ready() const;
  void load_down_scale(const StateDict& state_dict,
                       const std::string& down_name);

  int64_t hidden_size_ = 0;
  int64_t intermediate_size_ = 0;
  int64_t local_intermediate_size_ = 0;
  double swiglu_limit_ = 0.0;
  ProcessGroup* process_group_ = nullptr;

  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
  Activation act_{nullptr};

  torch::Tensor down_weight_scale_;
  torch::Tensor decode_gate_up_workspace_;
  torch::Tensor decode_activation_workspace_;
  torch::Tensor decode_output_workspace_;
};

TORCH_MODULE(DeepseekV4SharedExpert);

class DeepseekV4MoEImpl final : public torch::nn::Module {
 public:
  DeepseekV4MoEImpl() = default;
  DeepseekV4MoEImpl(const ModelArgs& model_args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    int32_t layer_id);

  torch::Tensor forward_with_selected_experts(
      const torch::Tensor& hidden_states,
      const torch::Tensor& topk_weights,
      const torch::Tensor& topk_ids,
      const ModelInputParams& input_params,
      bool skip_tp_allreduce = false,
      const torch::Tensor& output_buffer = {});

  // Preserve the existing six-argument ABI for all non-SP callers.
  torch::Tensor forward_with_selected_experts(
      const torch::Tensor& hidden_states,
      const torch::Tensor& topk_weights,
      const torch::Tensor& topk_ids,
      const ModelInputParams& input_params,
      bool skip_tp_allreduce,
      const torch::Tensor& output_buffer,
      const dsv4_musa_sp::DeepseekV4SPContext* reduce_scatter_context);

  void load_state_dict(const StateDict& state_dict);

 private:
  void load_experts(const StateDict& state_dict);
  torch::Tensor apply_expert(const torch::Tensor& input,
                             int64_t local_expert_id) const;
  torch::Tensor forward_grouped_fp8(const torch::Tensor& input,
                                    const torch::Tensor& topk_ids,
                                    const torch::Tensor& topk_weights,
                                    bool compact_rows = false) const;
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
  bool try_load_quantized_experts(const StateDict& state_dict,
                                  const std::vector<std::string>& w13_prefixes,
                                  const std::string& w2_prefix,
                                  int64_t rank,
                                  int64_t world_size,
                                  bool packed_fp4);
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
  int32_t layer_id_ = 0;
  int32_t num_layers_ = 0;
  int64_t grouped_fp8_decode_min_tokens_ = 0;
  double swiglu_limit_ = 0.0;
  std::string hidden_act_ = "silu";
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_ = nullptr;
  ProcessGroup* ep_pg_ = nullptr;

  DeepseekV4SharedExpert shared_experts_{nullptr};
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
  // Decode workspaces are allocated while loading weights so MUSA graph
  // capture never observes an allocation or a changing storage address.
  torch::Tensor fp8_gemv_activation_workspace_;
  torch::Tensor fp8_gemv_routed_output_workspace_;
  torch::Tensor fp8_gemv_output_workspace_;
  torch::Tensor tp_custom_allreduce_output_workspace_;
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
