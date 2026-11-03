/* Copyright 2026 The xLLM Authors.

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

// Composition-based LoRA wrapper around ColumnParallelLinear.
//
// Used for gate_up_proj (fused gate+up in Qwen family). LoRA A/B come from
// PEFT under two distinct target modules: "gate_proj" and "up_proj". This
// wrapper reads both per-proj deltas from LoRARuntime and concatenates them
// along the output dim to match the base's fused output.
//
// LoRA math (fused gate_up):
//   gate_delta = B_gate @ A_gate @ x     A: [r, hidden]  B: [inter, r]
//   up_delta   = B_up   @ A_up   @ x
//   y = base(x)  # shape [T, 2*inter_local]
//   y += cat([gate_delta, up_delta], -1) * scaling

#pragma once

#include <torch/torch.h>

#include <string>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRAColumnParallelLinearImpl : public torch::nn::Module {
 public:
  LoRAColumnParallelLinearImpl() = default;

  // Signature mirrors ColumnParallelLinearImpl (with LinearExtraArgs).
  // Extra parameter:
  //   fused: "gate_up_proj" -> lookup A/B for "gate_proj" + "up_proj",
  //          concat delta along dim=-1.
  //   single proj name (e.g. "o_proj" not applicable here since that's Row)
  //     - reserved for future use.
  LoRAColumnParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool gather_output,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const std::string& proj_name,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  void load_state_dict(const StateDict& state_dict);
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);
  void load_state_dict(const StateDict& state_dict,
                       int32_t shard_tensor_count,
                       const std::vector<int64_t>& shard_sizes);

  torch::Tensor weight() const { return base_->weight(); }
  std::optional<torch::Tensor> get_input_scale() const {
    return base_->get_input_scale();
  }

  // Passthrough accessors so the wrapper stays drop-in compatible with
  // ColumnParallelLinearImpl for callers added after this wrapper was
  // written (see upstream linear.h). uses_w8a8_dynamic_quant()
  // deliberately reports false while a LoRA delta may apply to the
  // current batch: DenseMLPImpl's quantized-allgather fast path bypasses
  // forward(), which would silently drop the LoRA delta for adapter
  // batches.
  bool is_weight_loaded() const { return base_->is_weight_loaded(); }
  bool uses_w8a8_dynamic_quant() const;
  torch::Tensor forward_quantized(const W8A8DynamicInput& input) {
    return base_->forward_quantized(input);
  }

  void pretty_print(std::ostream& stream) const {
    stream << name() << " (LoRA-wrapped/" << proj_name_
           << ")  base_weight=" << base_->weight().sizes();
  }

 private:
  ColumnParallelLinear base_{nullptr};
  std::string proj_name_;  // "gate_up_proj" for fused, or single
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;      // per-rank local, total (fused = 2*inter)
  int64_t out_size_local_ = 0;    // per-rank, matches base
  int64_t inter_size_local_ = 0;  // per-rank, half of out_size_local_ if fused
  bool is_fused_gate_up_ = false;
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;
};
TORCH_MODULE(LoRAColumnParallelLinear);

}  // namespace layer
}  // namespace xllm
