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

#include "layers/musa/dense_mlp.h"

#include <glog/logging.h>

#include "common/flash_comm1_context.h"

namespace xllm::layer::musa {

DenseMLPImpl::DenseMLPImpl(int64_t hidden_size,
                           int64_t intermediate_size,
                           bool is_gated,
                           bool has_bias,
                           const std::string& hidden_act,
                           bool enable_result_reduction,
                           const QuantArgs& quant_args,
                           ProcessGroup* process_group,
                           const torch::TensorOptions& options,
                           const std::string& module_prefix,
                           double swiglu_limit,
                           bool apply_fc1_sequence_parallel)
    : is_gated_(is_gated),
      intermediate_size_(intermediate_size),
      process_group_(process_group),
      apply_fc1_sequence_parallel_(apply_fc1_sequence_parallel) {
  CHECK_NE(quant_args.quant_method(), kQuantMethodSmoothquant)
      << "MUSA dense MLP does not support SmoothQuant.";

  int64_t out_feature = is_gated_ ? intermediate_size_ * 2 : intermediate_size_;
  gate_up_proj_ = register_module("gate_up_proj",
                                  ColumnParallelLinear(hidden_size,
                                                       out_feature,
                                                       /*bias=*/has_bias,
                                                       /*gather_output=*/false,
                                                       quant_args,
                                                       process_group_,
                                                       options));

  act_ =
      register_module("act", Activation(hidden_act, is_gated_, swiglu_limit));

  const QuantArgs down_proj_quant_args =
      module_prefix.empty()
          ? quant_args
          : quant_args.for_module(module_prefix + ".down_proj");
  down_proj_ = register_module("down_proj",
                               RowParallelLinear(intermediate_size_,
                                                 hidden_size,
                                                 /*bias=*/has_bias,
                                                 /*input_is_parallelized=*/true,
                                                 enable_result_reduction,
                                                 down_proj_quant_args,
                                                 process_group_,
                                                 options));
}

torch::Tensor DenseMLPImpl::forward(const torch::Tensor& hidden_states) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  const bool use_fc1_sequence_parallel =
      apply_fc1_sequence_parallel_ && fc1_ctx && is_sequence_sharded(*fc1_ctx);
  torch::Tensor h = hidden_states;

  if (use_fc1_sequence_parallel) {
    h = gather_sequence(hidden_states, *fc1_ctx);
  }

  torch::Tensor gate_up = gate_up_proj_->forward(h);
  torch::Tensor output;
  act_->forward(gate_up, output);

  if (use_fc1_sequence_parallel) {
    return down_proj_->forward(output,
                               row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  }
  return down_proj_->forward(output);
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict) {
  gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict,
                                   const std::vector<std::string>& gate_up_name,
                                   const std::string& down_name) {
  if (is_gated_) {
    CHECK_EQ(gate_up_name.size(), 2);
    gate_up_proj_->load_state_dict(state_dict, gate_up_name);
  } else {
    CHECK_EQ(gate_up_name.size(), 1);
    gate_up_proj_->load_state_dict(
        state_dict.get_dict_with_prefix(gate_up_name[0]));
  }
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix(down_name));
}

}  // namespace xllm::layer::musa
