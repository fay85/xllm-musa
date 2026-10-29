/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/execution_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

#if defined(USE_MUSA)
constexpr bool kMusaDefault_enable_graph = true;
#else
constexpr bool kMusaDefault_enable_graph = false;
#endif

DEFINE_bool(
    enable_graph,
    kMusaDefault_enable_graph,
    "Whether to enable graph execution for decode phase. When enabled, "
    "the engine uses graph mode (CUDA Graph for GPU, ACL Graph for NPU, "
    "MLU Graph, or DCU Graph) to optimize decode performance by reducing "
    "kernel launch overhead and device idle time.");

DEFINE_bool(disable_graph_warmup,
            false,
            "Whether to skip synthetic graph warmup during engine startup. "
            "When enabled, graphs are captured lazily by real requests.");

DEFINE_bool(enable_graph_double_buffer,
            true,
            "Whether to enable double-buffered ACL graph persistent params "
            "and graph instances for NPU schedule-overlap decode.");

#if defined(USE_MUSA)
constexpr bool kMusaDefault_enable_graph_mode_decode_no_padding = true;
#else
constexpr bool kMusaDefault_enable_graph_mode_decode_no_padding = false;
#endif

DEFINE_bool(enable_graph_mode_decode_no_padding,
            kMusaDefault_enable_graph_mode_decode_no_padding,
            "Whether to enable graph execution for decode phase without "
            "padding. If true, graph will be captured with every actual num "
            "tokens, as stride is 1.");

DEFINE_bool(enable_prefill_piecewise_graph,
            false,
            "Whether to enable piecewise graph execution for prefill phase "
            "when graph mode is enabled. When enabled, attention operations "
            "use eager mode while other operations are captured in device "
            "graphs.");

constexpr bool kEnableGraphVmmPoolDefault = true;

DEFINE_bool(enable_graph_vmm_pool,
            kEnableGraphVmmPoolDefault,
            "Whether to enable VMM-backed graph memory pool for multi-shape "
            "graph memory reuse.");

DEFINE_int32(max_tokens_for_graph_mode,
             2048,
             "Maximum number of tokens for graph execution. "
             "If 0, no limit is applied.");

DEFINE_int32(acl_graph_decode_batch_size_limit,
             16,
             "Decode batch size threshold for ACL graph on NPU. "
             "When actual decode batch_size > this value, ACL graph decode "
             "falls back to eager mode to avoid OOM.");

DEFINE_bool(enable_shm,
            false,
            "Whether to enable shared memory for executing model.");

DEFINE_bool(use_contiguous_input_buffer,
            true,
            "Whether to use contiguous device input buffer for executing "
            "model.");

DEFINE_uint64(input_shm_size,
              1024,
              "Input shared memory size, default is 1GB.");

DEFINE_uint64(output_shm_size,
              128,
              "Output shared memory size, default is 128MB.");

DEFINE_int32(random_seed, -1, "Random seed for random number generator.");

DEFINE_string(
    python_graph_backend,
    "off",
    "Graph backend for the Python model executor. "
    "Defaults to aclgraph on NPU when --enable_graph is enabled. "
    "Values: off (eager), cudagraphs (CUDA decode graph with eager prefill), "
    "aclgraph (NPU decode graph with eager prefill), "
    "or any torch.compile backend name.");

DEFINE_bool(
    enable_fia_decode,
    false,
    "Enable FIA for Qwen3.5 decode attention. Applies to both target and MTP "
    "draft models. Prefill attention is unaffected.");

DEFINE_int32(
    musa_prefill_graph_token_stride,
    64,
    "Token bucket stride for single-sequence MUSA Prefill graphs. "
    "Decode buckets and exact multi-sequence GDN layouts are unchanged.");

DEFINE_int32(musa_prefill_graph_max_entries,
             32,
             "Maximum retained Prefill graph keys per MUSA executor. Existing "
             "keys replay at the limit; unseen keys execute eager before any "
             "capture or VMM address-space allocation.");

DEFINE_int32(musa_prefill_warmup_reserve_mb,
             2048,
             "Additional MiB excluded from the KV budget for MUSA piecewise "
             "Prefill preparation peaks and graph overhead. Applies once to "
             "the shared target/draft memory estimate, in addition to the "
             "max_memory_utilization headroom.");

namespace xllm {

void ExecutionConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(disable_graph_warmup);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph_double_buffer);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph_mode_decode_no_padding);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_prefill_piecewise_graph);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph_vmm_pool);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_tokens_for_graph_mode);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(acl_graph_decode_batch_size_limit);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_shm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_contiguous_input_buffer);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(input_shm_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(output_shm_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(random_seed);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(python_graph_backend);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_fia_decode);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(musa_prefill_graph_token_stride);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(musa_prefill_graph_max_entries);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(musa_prefill_warmup_reserve_mb);
}

void ExecutionConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph);
  XLLM_CONFIG_ASSIGN_FROM_JSON(disable_graph_warmup);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph_double_buffer);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph_mode_decode_no_padding);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_prefill_piecewise_graph);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph_vmm_pool);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_tokens_for_graph_mode);
  XLLM_CONFIG_ASSIGN_FROM_JSON(acl_graph_decode_batch_size_limit);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_shm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_contiguous_input_buffer);
  XLLM_CONFIG_ASSIGN_FROM_JSON(input_shm_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(output_shm_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(random_seed);
  XLLM_CONFIG_ASSIGN_FROM_JSON(python_graph_backend);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_fia_decode);
  XLLM_CONFIG_ASSIGN_FROM_JSON(musa_prefill_graph_token_stride);
  XLLM_CONFIG_ASSIGN_FROM_JSON(musa_prefill_graph_max_entries);
  XLLM_CONFIG_ASSIGN_FROM_JSON(musa_prefill_warmup_reserve_mb);
}

void ExecutionConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ExecutionConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, disable_graph_warmup);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph_double_buffer);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph_mode_decode_no_padding);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_prefill_piecewise_graph);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph_vmm_pool);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_tokens_for_graph_mode);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, acl_graph_decode_batch_size_limit);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_shm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, use_contiguous_input_buffer);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, input_shm_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, output_shm_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, random_seed);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, python_graph_backend);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_fia_decode);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, musa_prefill_graph_token_stride);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, musa_prefill_graph_max_entries);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, musa_prefill_warmup_reserve_mb);
}

ExecutionConfig& ExecutionConfig::get_instance() {
  static ExecutionConfig config;
  return config;
}

void ExecutionConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
  CHECK_GT(musa_prefill_graph_token_stride(), 0);
  CHECK_EQ(musa_prefill_graph_token_stride() % 16, 0)
      << "MUSA Prefill stride must be a multiple of 16";
  CHECK_GT(musa_prefill_graph_max_entries(), 0);
  CHECK_GE(musa_prefill_warmup_reserve_mb(), 0);
}

}  // namespace xllm
