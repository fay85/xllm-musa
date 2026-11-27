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

#include "runtime/prefill_layerwise_eligibility.h"

#include "core/common/instance_role_util.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "platform/platform.h"
#include "runtime/options.h"

namespace xllm {

PrefillLayerwiseEligibility build_prefill_layerwise_eligibility(
    const runtime::Options& options,
    std::string_view model_type,
    std::string_view model_impl) {
  PrefillLayerwiseEligibility eligibility;
  eligibility.npu = Platform::is_npu();
  eligibility.torch_kernel_backend = options.npu_kernel_backend() == "TORCH";
  eligibility.llm_backend = options.backend() == "llm";
  eligibility.generate_task = options.task_type() == "generate";
  eligibility.pd_prefill_only = is_pd_prefill_role(options);
  eligibility.push_transfer = options.kv_cache_transfer_mode() == "PUSH";
  eligibility.enable_prefill_piecewise_graph =
      options.enable_prefill_piecewise_graph();
  eligibility.is_draft_engine = options.is_draft_engine();
  eligibility.enable_sleep_mode = options.enable_sleep_mode();
  eligibility.enable_kvcache_store = options.enable_kvcache_store();
  eligibility.host_blocks_factor = options.host_blocks_factor();
  eligibility.cp_size = options.cp_size();
  eligibility.layerwise_split_size =
      ParallelConfig::get_instance().layerwise_split_size();
  eligibility.kv_split_size_effective =
      ParallelConfig::get_instance().kv_split_size_effective();
  eligibility.python_model = ModelConfig::is_python_model_impl(model_impl);
  eligibility.replicated_index =
      eligibility.kv_split_size_effective > 1 &&
      declared_cp_index_write_mode() == kCpIndexWriteModeReplicated;
  eligibility.has_speculative_decode = options.enable_speculative_decode() ||
                                       options.num_speculative_tokens() > 0;
  const std::string_view algorithm = options.speculative_algorithm();
  eligibility.supported_speculative_target =
      model_type == "glm_moe_dsa" && eligibility.python_model &&
      (SpeculativeConfig::is_dflash_algorithm(algorithm) ||
       SpeculativeConfig::is_dflash2_algorithm(algorithm) ||
       SpeculativeConfig::is_mtp_algorithm(algorithm));
  eligibility.kv_cache_dtype = std::string(options.kv_cache_dtype());
  eligibility.model_type = std::string(model_type);
  return eligibility;
}

}  // namespace xllm
