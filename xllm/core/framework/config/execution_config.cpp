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

#include <algorithm>
#include <limits>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/platform/platform.h"

DEFINE_bool(
    enable_graph,
    false,
    "Whether to enable graph execution for decode phase. When enabled, "
    "the engine uses graph mode (CUDA Graph for GPU, ACL Graph for NPU, "
    "MLU Graph, or DCU Graph) to optimize decode performance by reducing "
    "kernel launch overhead and device idle time.");

DEFINE_bool(disable_graph_warmup,
            false,
            "Whether to skip synthetic graph warmup during engine startup. "
            "When enabled, graphs are captured lazily by real requests.");

DEFINE_string(graph_capture_sizes,
              "",
              "Comma-separated positive decode graph capture sizes in "
              "DP-local token rows, e.g. 1,2,4,8,16. JSON uses an integer "
              "array. Supported only by native and Python ACL graph "
              "paths on NPU for LLM/VLM decode, subject to existing model "
              "graph support. Nonempty values are rejected on other devices; "
              "prefill graph policy is unchanged. Sizes are rounded up to each "
              "runner's query width within "
              "the largest configured size and existing capacity limits. "
              "Unspecified keeps the default policy; no-padding and adaptive "
              "speculative decode or native packed ATB speculative kernels "
              "are not supported with this option.");

DEFINE_bool(enable_graph_double_buffer,
            true,
            "Whether to enable double-buffered ACL graph persistent params "
            "and graph instances for NPU schedule-overlap decode.");

DEFINE_bool(enable_graph_mode_decode_no_padding,
            false,
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

DEFINE_int32(forward_execution_timeout_ms,
             300000,
             "Deadline in milliseconds for a dispatched worker execution. "
             "Must be positive; idle workers are not timed out.");

DEFINE_int32(random_seed, -1, "Random seed for random number generator.");

DEFINE_bool(
    enable_fia_decode,
    false,
    "Enable FIA for Qwen3.5 decode attention. Applies to both target and MTP "
    "draft models. Prefill attention is unaffected.");

DEFINE_bool(
    enable_graph_aux_hidden_states,
    false,
    "Let the deepseek_v41 target publish its transiently-incompatible aux "
    "capture through a persistent graph buffer instead of staying eager. "
    "The transient aux capture returns a (hidden, aux) tuple; the "
    "deepseek_v41 graph path (CSA2 backend + fused ops) cannot replay that "
    "tuple, so with this flag off that one target is pinned to eager. "
    "Every other aux-capture target replays the transient tuple through "
    "the graph fine (fixed-address pool allocation since #240) and is "
    "never affected by this flag. When enabled the v41 capture buffer "
    "becomes persistent (fixed address, graph-capturable) and the "
    "executor re-attaches the aux slice to the graph output (gate chain: "
    "py_causal_lm -> deepseek_v41.py AuxHiddenCapture -> "
    "decode_acl_graph -> executor aux publish).");

namespace xllm {
namespace {

std::vector<int32_t> parse_graph_capture_sizes(const nlohmann::json& value) {
  CHECK(value.is_array() && !value.empty())
      << "graph_capture_sizes must be a nonempty array of positive int32 "
         "values";
  std::vector<int32_t> sizes;
  sizes.reserve(value.size());
  for (const auto& size : value) {
    CHECK(size.is_number_integer() && size > 0 &&
          size <= std::numeric_limits<int32_t>::max())
        << "graph_capture_sizes must contain only positive int32 values";
    sizes.emplace_back(size.get<int32_t>());
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

}  // namespace

void ExecutionConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(disable_graph_warmup);
  if (FLAGS_graph_capture_sizes.empty()) {
    graph_capture_sizes({});
  } else {
    graph_capture_sizes(parse_graph_capture_sizes(
        nlohmann::json::parse("[" + FLAGS_graph_capture_sizes + "]",
                              nullptr,
                              /*allow_exceptions=*/false)));
  }
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
  XLLM_CONFIG_ASSIGN_FROM_FLAG(forward_execution_timeout_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(random_seed);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_fia_decode);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_graph_aux_hidden_states);
}

void ExecutionConfig::from_json(const JsonReader& json) {
  if (json.contains("python_graph_backend")) {
    LOG(WARNING)
        << "python_graph_backend has been removed and is ignored. Set "
        << "enable_graph=true to request graph execution; torch.compile "
        << "backends are no longer supported.";
  }
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph);
  XLLM_CONFIG_ASSIGN_FROM_JSON(disable_graph_warmup);
  if (!config::is_flag_specified("graph_capture_sizes") &&
      json.contains("graph_capture_sizes")) {
    graph_capture_sizes(
        parse_graph_capture_sizes(*json.resolve("graph_capture_sizes")));
    const std::string serialized = nlohmann::json(graph_capture_sizes()).dump();
    FLAGS_graph_capture_sizes = serialized.substr(1, serialized.size() - 2);
  }
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
  XLLM_CONFIG_ASSIGN_FROM_JSON(forward_execution_timeout_ms);
  XLLM_CONFIG_ASSIGN_FROM_JSON(random_seed);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_fia_decode);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_graph_aux_hidden_states);
}

void ExecutionConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ExecutionConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, disable_graph_warmup);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, graph_capture_sizes);
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
      config_json, default_config, forward_execution_timeout_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, random_seed);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_fia_decode);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_graph_aux_hidden_states);
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
  validate();
}

void ExecutionConfig::validate() const {
  CHECK_GT(forward_execution_timeout_ms(), 0);
  if (graph_capture_sizes().empty()) {
    return;
  }
  // TODO: Enable MLU after its owners implement and validate
  // configured capture buckets, graph warmup, and overflow eager routing.
  CHECK(Platform::is_npu()) << "graph_capture_sizes is only supported on NPU";
  CHECK(enable_graph()) << "graph_capture_sizes requires enable_graph=true";
  CHECK(!enable_graph_mode_decode_no_padding())
      << "graph_capture_sizes cannot be combined with "
         "enable_graph_mode_decode_no_padding";
}

}  // namespace xllm
