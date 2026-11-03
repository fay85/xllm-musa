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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

#include "core/framework/lora/lora_runtime.h"
#include "core/framework/model_context.h"

namespace xllm {

// Start the LoRA hot-swap executor for this engine. The runtime keeps the
// model's device / dtype so /v1/load_lora_adapter knows where to place
// weights, and set_model_device_dtype also spawns the pinned executor
// thread that owns the NPU context -- without it every dynamic load fails
// with "executor not started". The hot-swap install
// config passes the TP / MoE shape the executor needs to pick between the
// per-proj and MoE-experts installers.
//
// Called from the CausalLM base ctors so every model family gets it without
// per-model edits. Idempotent and last-writer-wins (single model per engine
// today); the executor itself only spawns once. No-op when LoRA is
// disabled, so stock deployments pay one enabled() check per model build.
inline void bootstrap_lora_runtime(const ModelContext& context) {
  auto& runtime = LoRARuntime::instance();
  if (!runtime.enabled()) {
    return;
  }

  const auto& options = context.get_tensor_options();
  const torch::Device device = options.device();
  const torch::ScalarType dtype = options.dtype().toScalarType();
  runtime.set_model_device_dtype(device, dtype);

  const ModelArgs& model_args = context.get_model_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();

  LoRARuntime::HotswapConfig cfg;
  const int32_t tp_size = std::max<int32_t>(1, parallel_args.tp_size());
  const int32_t tp_rank = parallel_args.tp_group_ != nullptr
                              ? parallel_args.tp_group_->rank()
                              : parallel_args.rank() % tp_size;
  if (parallel_args.tp_group_ == nullptr && tp_size > 1) {
    LOG(WARNING) << "[LoRA bootstrap] tp_group is unavailable; assuming TP "
                    "rank "
                 << tp_rank << " from global rank " << parallel_args.rank();
  }
  cfg.tp = TPInfo{tp_size, tp_rank};
  cfg.install_per_proj = true;
  // MoE discriminator: only MoE checkpoints carry activated experts, so a
  // positive num_experts_per_tok is the reliable "is MoE" probe. Dense
  // models default to 0 and skip the experts installer entirely.
  cfg.install_moe_experts = model_args.num_experts_per_tok() > 0;
  cfg.num_experts_total = model_args.num_experts();
  // Same shard math as FusedMoE (num_experts / ep_size, ep_rank * slice).
  const int32_t ep_size = std::max<int32_t>(1, parallel_args.ep_size());
  cfg.num_experts_per_rank = cfg.num_experts_total / ep_size;
  int32_t ep_rank = 0;
  if (ep_size > 1) {
    if (parallel_args.moe_ep_group_ != nullptr) {
      ep_rank = parallel_args.moe_ep_group_->rank();
    } else {
      // No EP group during early bootstrap: fall back to the common
      // innermost-EP rank layout and let the loud warning point at any
      // deployment where that assumption breaks.
      ep_rank = parallel_args.rank() % ep_size;
      LOG(WARNING) << "[LoRA bootstrap] ep_size=" << ep_size
                   << " but no moe_ep_group yet; assuming ep_rank=" << ep_rank
                   << " from rank " << parallel_args.rank();
    }
  }
  cfg.start_expert_id = ep_rank * cfg.num_experts_per_rank;
  cfg.moe_intermediate_size = model_args.moe_intermediate_size();
  runtime.set_hotswap_config(cfg);
  LOG(INFO) << "[LoRA bootstrap] hot-swap executor armed on device=" << device
            << " dtype=" << dtype << " tp_size=" << parallel_args.tp_size()
            << " moe=" << cfg.install_moe_experts;
}

}  // namespace xllm
