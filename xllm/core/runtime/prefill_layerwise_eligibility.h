/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace xllm {
namespace runtime {
struct Options;
}

// The prefill offload owner is only valid on an instance that never executes
// decode: its Device slots are overwritten layer by layer, while the scheduler
// still addresses the same logical blocks. Supported speculative targets keep
// their draft context in a separate full cache outside the offload owner.
//
// This object owns the complete target admission: worker lifecycle (backend and
// task), topology, graph mode, and the cache/budget predicates. Engine and
// worker entry points must consume it instead of re-checking individual flags.
// The string fields are copies, not views, so a stored eligibility stays valid
// independently of the Options/ModelArgs it was built from.
class PrefillLayerwiseEligibility final {
 public:
  bool npu = false;
  bool torch_kernel_backend = false;
  bool llm_backend = false;
  bool generate_task = false;
  bool pd_prefill_only = false;
  bool push_transfer = false;
  bool enable_prefill_piecewise_graph = false;
  bool is_draft_engine = false;
  bool enable_sleep_mode = false;
  bool enable_xtensor = false;
  bool enable_kvcache_store = false;
  double host_blocks_factor = 1.0;
  int32_t cp_size = 1;
  int32_t layerwise_split_size = 1;
  int32_t kv_split_size_effective = 1;
  bool python_model = false;
  bool replicated_index = false;
  // True when the deployment declares any speculative intent. The target
  // engine's own Options never carry enable_speculative_decode (it is set on
  // the worker options), so num_speculative_tokens also signals intent.
  bool has_speculative_decode = false;
  bool supported_speculative_target = false;
  std::string kv_cache_dtype;
  std::string model_type;

  bool supported() const { return !error().has_value(); }

  std::optional<std::string> error() const {
    if (model_type != "glm_moe_dsa") {
      return "Prefill layerwise offload currently supports glm_moe_dsa only.";
    }
    if (!npu) {
      return "Prefill layerwise offload requires NPU.";
    }
    if (!torch_kernel_backend) {
      return "Prefill layerwise offload currently supports only the TORCH "
             "kernel backend; ATB has no model TP group to prepare KV over.";
    }
    if (!llm_backend) {
      return "Prefill layerwise offload requires the LLM worker lifecycle.";
    }
    if (!generate_task) {
      return "Prefill layerwise offload requires task=generate; embedding "
             "workers do not drive the layerwise KV lifecycle.";
    }
    if (!pd_prefill_only) {
      return "Prefill layerwise offload requires a disaggregated PREFILL "
             "instance without PD online-offline co-location.";
    }
    if (!push_transfer) {
      return "Prefill layerwise offload currently requires PD PUSH.";
    }
    if (is_draft_engine) {
      return "Prefill layerwise offload owns the target cache only; the draft "
             "engine must keep its ordinary cache.";
    }
    if (cp_size < 1) {
      return "Prefill layerwise offload requires CP >= 1.";
    }
    if (layerwise_split_size != 1) {
      return "Prefill layerwise offload requires layerwise_split_size=1.";
    }
    if (kv_split_size_effective < 1 || cp_size % kv_split_size_effective != 0) {
      return "Prefill layerwise offload requires a positive effective "
             "kv_split_size that divides cp_size.";
    }
    if (kv_split_size_effective > 1 && (!python_model || !replicated_index)) {
      return "Prefill layerwise offload with KV split requires Python CP, "
             "replicated Index writes; otherwise set kv_split_size=1.";
    }
    if (has_speculative_decode && !supported_speculative_target) {
      return "Prefill layerwise offload only supports speculative decoding for "
             "Python glm_moe_dsa targets with DFlash, DFlash2, or MTP.";
    }
    if (enable_prefill_piecewise_graph) {
      return "Prefill layerwise offload does not support piecewise prefill "
             "graphs.";
    }
    if (enable_sleep_mode) {
      return "Prefill layerwise offload does not support sleep mode.";
    }
    if (enable_xtensor) {
      return "Prefill layerwise offload does not support XTensor.";
    }
    if (enable_kvcache_store || host_blocks_factor > 1.0) {
      return "Prefill layerwise offload owns its Host data; Host/Store "
             "hierarchy cannot be combined.";
    }
    if (kv_cache_dtype != "auto") {
      return "Prefill layerwise offload requires kv_cache_dtype=auto.";
    }
    return std::nullopt;
  }
};

PrefillLayerwiseEligibility build_prefill_layerwise_eligibility(
    const runtime::Options& options,
    std::string_view model_type,
    std::string_view model_impl);

}  // namespace xllm
