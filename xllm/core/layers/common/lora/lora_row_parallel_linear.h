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

// Composition-based LoRA wrapper around RowParallelLinear.
//
// Used for o_proj (attention output projection) and down_proj (MLP output).
// Both are Row-parallel (input already TP-sharded, output reduced).
//
// LoRA math (single proj):
//   y = base(x)
//   delta = B @ (A @ x) * scaling      A: [r, in_full]   B: [out, r]
//   return y + delta
//
// TP handling (mirrors SGLang RowParallelLinearWithLoRA):
//   * A is slice-sharded on in-dim at forward time: A_local = A[:, rank_slice]
//     Input arrives already sharded on in-dim, so this alignment is natural.
//   * B is kept full-width (replicated) — cheap because r is tiny.
//   * forward: tmp_local = x_local @ A_local^T -> [T, r]  (partial per rank)
//              tmp_full  = all_reduce(tmp_local)         (rank-dim, cheap)
//              delta     = tmp_full @ B^T * scaling      (replicated on ranks)
//              y         = base_output (already reduced by base) + delta
//
// Cost win over naive "all_reduce(delta)": we reduce a rank-dim tensor
// [T, r=16] instead of hidden-dim [T, out]. For Qwen3-30B r=16 vs out=2048
// that is 128x smaller.

#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRARowParallelLinearImpl : public torch::nn::Module {
 public:
  LoRARowParallelLinearImpl() = default;

  // Signature mirrors RowParallelLinearImpl exactly (drop-in replace).
  // Extra parameters:
  //   proj_name: which PEFT target module this wrapper serves. Passed to
  //     LoRARuntime::get_per_proj_delta at forward time. Must be one of
  //     "o_proj" or "down_proj" for Qwen family.
  LoRARowParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool input_is_parallelized,
      bool enable_result_reduction,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const std::string& proj_name,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  // Mirror of RowParallelLinearImpl::forward(input, reduce_mode) added upstream
  // for flash_comm1 sequence-sharded fc1 (see linear.h). Batches that carry no
  // adapter id route through to the base's 2-arg overload so the fc1
  // reduce-scatter / MMRS semantics apply unchanged; batches that do carry an
  // adapter cannot scatter (the delta folds into full-row output), so the
  // sharded modes CHECK-fail loudly instead of silently mis-shaping.
  torch::Tensor forward(torch::Tensor input, RowParallelReduceMode reduce_mode);

  // Passthrough so verify_loaded_weights() can poll the base's load state
  // through the wrapper (mirrors RowParallelLinearImpl::is_weight_loaded).
  bool is_weight_loaded() const { return base_->is_weight_loaded(); }

  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " (LoRA-wrapped/" << proj_name_
           << ")  base_weight=" << base_->weight().sizes();
  }

 private:
  RowParallelLinear base_{nullptr};
  std::string proj_name_;
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;

  // Cached TP topology so forward's hot path avoids a virtual dispatch
  // into ProcessGroup for the tp_rank / world_size on every call.
  int32_t tp_rank_ = 0;
  int32_t tp_world_size_ = 1;
  int64_t in_features_local_ = 0;

  // Fused-AR mode: base is constructed with enable_result_reduction=false
  // and the wrapper owns the collective. Set once in the ctor from the
  // enable_lora_row_parallel_fused_ar flag.
  bool fused_ar_ = false;
  // True iff we actually issue an AR in forward (fused_ar_ && tp>1).
  bool wrapper_owns_reduction_ = false;
};
TORCH_MODULE(LoRARowParallelLinear);

}  // namespace layer
}  // namespace xllm
