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

#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/xllm_build_info.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "dit_master.h"
#if defined(USE_NPU)
#include "framework/parallel_state/npu_rank_table_env.h"
#endif
#include "core/platform/device_name_utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "llm_engine.h"
#include "llm_master.h"
#include "models/cp_capability.h"
#include "models/model_registry.h"
#include "platform/platform.h"
#include "runtime/options.h"
#include "speculative_engine.h"
#include "util/model_config_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"
#include "vlm_engine.h"
#include "vlm_master.h"

namespace brpc {
DECLARE_bool(graceful_quit_on_sigterm);
DECLARE_bool(graceful_quit_on_sighup);
}  // namespace brpc

namespace xllm {
namespace {

enum class VlmPdTransferSupport { UNSUPPORTED, PUSH_ONLY, PUSH_OR_PULL };

VlmPdTransferSupport vlm_speculative_pd_transfer_support(
    const std::string& model_type,
    std::string_view algorithm) {
  if (SpeculativeConfig::is_qwen3_5_mtp_or_dflash2(model_type, algorithm)) {
    return Platform::is_mlu() ? VlmPdTransferSupport::PUSH_ONLY
                              : VlmPdTransferSupport::PUSH_OR_PULL;
  }
  if (SpeculativeConfig::is_oxygenvlm_mtp_or_dflash2(model_type, algorithm)) {
    return Platform::is_mlu() ? VlmPdTransferSupport::PUSH_ONLY
                              : VlmPdTransferSupport::PUSH_OR_PULL;
  }
  if (Platform::is_mlu() && model_type == "glm5_next" &&
      SpeculativeConfig::is_mtp_algorithm(algorithm)) {
    return VlmPdTransferSupport::PUSH_ONLY;
  }
  if (SpeculativeConfig::is_glm5_next_dflash2(model_type, algorithm) ||
      SpeculativeConfig::is_deepseek_v41_dspark(model_type, algorithm) ||
      SpeculativeConfig::is_qwen3_5_dspark(model_type, algorithm) ||
      SpeculativeConfig::is_qwen3_5_python_dflash(
          model_type, algorithm, ModelConfig::get_instance().model_impl()) ||
      SpeculativeConfig::is_qwen3_vl_eagle3(model_type, algorithm) ||
      SpeculativeConfig::is_oxygenvlm_dspark(model_type, algorithm)) {
    return VlmPdTransferSupport::PUSH_OR_PULL;
  }
  return VlmPdTransferSupport::UNSUPPORTED;
}

void apply_runtime_kv_cache_options(const Options& source,
                                    runtime::Options& destination) {
  CHECK(!source.enable_kvcache_store() || source.host_blocks_factor() > 1.0)
      << "KV cache Store requires Host cache blocks (host_blocks_factor > 1).";
  destination.host_blocks_factor(source.host_blocks_factor())
      .enable_kvcache_store(source.enable_kvcache_store())
      .store_protocol(source.store_protocol())
      .store_rdma_devices(source.store_rdma_devices())
      .store_master_server_address(source.store_master_server_address())
      .store_metadata_server(source.store_metadata_server())
      .store_local_hostname(source.store_local_hostname())
      .prefetch_batch_size(source.prefetch_batch_size())
      .prefetch_valid_blocks_per_batch(source.prefetch_valid_blocks_per_batch())
      .prefetch_timeout(source.prefetch_timeout())
      .layers_wise_copy_batchs(source.layers_wise_copy_batchs())
      .kv_cache_dtype(source.kv_cache_dtype());
}

void validate_layerwise_split_size_startup_config(const Options& options,
                                                  const std::string& model_type,
                                                  int32_t global_world_size) {
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const int32_t layerwise_split_size = parallel_config.layerwise_split_size();
  validate_layerwise_split_size_config(layerwise_split_size);
  if (layerwise_split_size <= 1) {
    return;
  }

  CHECK_GE(options.dp_size(), 1) << "dp_size must be >= 1.";
  CHECK_GT(global_world_size, 0) << "world_size must be > 0.";
  const int32_t dp_cp_size = options.dp_size() * options.cp_size();
  CHECK_EQ(global_world_size % dp_cp_size, 0)
      << "world_size (" << global_world_size
      << ") must be divisible by dp_size * cp_size (" << dp_cp_size << ").";
  const int32_t attn_tp_size = global_world_size / dp_cp_size;
  std::string resolved_model_type = model_type;
  if (resolved_model_type.empty()) {
    resolved_model_type =
        util::get_model_type(options.model_path(), options.backend());
  }
  validate_layerwise_split_enablement(
      layerwise_split_size, attn_tp_size, resolved_model_type);

  CHECK_LE(options.host_blocks_factor(), 1.0)
      << "layerwise_split_size > 1 does not support hierarchy host cache.";
  CHECK_EQ(parallel_config.cp_size(), 1)
      << "layerwise_split_size > 1 does not support context parallelism.";
  CHECK_EQ(parallel_config.kv_split_size_effective(), 1)
      << "layerwise_split_size > 1 does not support KV split.";
}

}  // namespace

std::optional<std::string> validate_vlm_speculative_pd_transfer_mode(
    const Options& options,
    const std::string& model_type) {
  if (!options.enable_disagg_pd() ||
      options.kv_cache_transfer_mode() == "PUSH") {
    return std::nullopt;
  }
  // Native PULL assumes matching SSM checkpoint strides on prefill and decode.
  const std::string& algorithm = options.speculative_algorithm();
  if (vlm_speculative_pd_transfer_support(model_type, algorithm) ==
      VlmPdTransferSupport::PUSH_ONLY) {
    return "MLU " + model_type + " VLM " + algorithm +
           " with disaggregated PD requires "
           "kv_cache_transfer_mode=PUSH";
  }
  return std::nullopt;
}

std::optional<std::string> validate_qwen_dcp_topology(int32_t global_world_size,
                                                      int32_t dp_size,
                                                      int32_t kv_split_size) {
  if (kv_split_size < 1) {
    return "kv_split_size must resolve to a value greater than or equal to 1";
  }
  if (kv_split_size == 1) {
    return std::nullopt;
  }
  if (global_world_size < 1 || dp_size < 1) {
    return "Qwen DCP requires positive world_size and dp_size";
  }
  if (global_world_size % dp_size != 0) {
    return "Qwen DCP requires world_size divisible by dp_size";
  }
  const int32_t tp_size = global_world_size / dp_size;
  if (kv_split_size > tp_size || tp_size % kv_split_size != 0) {
    return "Qwen DCP kv_split_size must divide the TP size within each DP "
           "replica";
  }
  return std::nullopt;
}

namespace {

void print_startup_banner(const std::filesystem::path& model_path,
                          const std::string& backend,
                          int32_t node_rank) {
  if (node_rank != 0) {
    return;
  }

  constexpr std::string_view kAnsiRed = "\033[31m";
  constexpr std::string_view kAnsiReset = "\033[0m";
  const bool use_color = ::isatty(::fileno(stderr));

  std::array<std::string_view, 4> x_logo = {
      "      ", "▀█▄ ▀ ", "  █▶  ", "▄█▀ ▄ "};
  std::array<std::string_view, 4> llm_logo = {"█     █     █▄   ▄█",
                                              "█     █     █ ▀▄▀ █",
                                              "█     █     █     █",
                                              "█▄▄▄▄ █▄▄▄▄ █     █"};

  LOG(INFO) << "";
  LOG(INFO) << x_logo[0] << llm_logo[0];
  if (use_color) {
    LOG(INFO) << kAnsiRed << x_logo[1] << kAnsiReset << llm_logo[1]
              << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << kAnsiRed << x_logo[2] << kAnsiReset << llm_logo[2]
              << "  model   " << model_path.string();
    LOG(INFO) << kAnsiRed << x_logo[3] << kAnsiReset << llm_logo[3]
              << "  backend " << backend;
  } else {
    LOG(INFO) << x_logo[1] << llm_logo[1] << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << x_logo[2] << llm_logo[2] << "  model   "
              << model_path.string();
    LOG(INFO) << x_logo[3] << llm_logo[3] << "  backend " << backend;
  }
  LOG(INFO) << "";
}

}  // namespace

namespace {

#if defined(USE_NPU)
void validate_rank_tablefile_backend() {
  const EPLBConfig& eplb_config = EPLBConfig::get_instance();
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (!eplb_config.rank_tablefile().empty() &&
      parallel_config.communication_backend() != "hccl") {
    LOG(FATAL) << "--rank_tablefile requires --communication_backend=hccl, "
               << "but got --communication_backend="
               << parallel_config.communication_backend();
  }
}

void resolve_npu_kernel_backend_for_options(Options* options) {
  CHECK(options != nullptr) << "options must not be null";
  if (options->backend() == "dit") {
    return;
  }

  // Python model executor builds the compute graph in Python (torch/torch_npu),
  // bypassing ATB C++ kernels entirely — force TORCH backend so that kernel
  // dispatch picks pure-torch implementations for reshape_and_cache etc.
  if (ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl())) {
    options->npu_kernel_backend("TORCH");
    KernelConfig::get_instance().npu_kernel_backend("TORCH");
    LOG(INFO) << "Forced npu_kernel_backend=TORCH for python model_impl";
    return;
  }

  const std::string model_type =
      util::get_model_type(options->model_path(), options->backend());
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration(model_type,
                                  options->npu_kernel_backend(),
                                  &effective_backend,
                                  &resolved_name,
                                  &error_message)) {
    LOG(FATAL) << error_message;
  }

  options->npu_kernel_backend(effective_backend);
  KernelConfig::get_instance().npu_kernel_backend(effective_backend);
  LOG(INFO) << "Resolved npu_kernel_backend=" << effective_backend
            << " for model_type=" << model_type;
}
#endif

}  // namespace

Master::Master(const Options& options, EngineType type)
    : options_(options),
      engine_type_(type),
      master_status_(options.master_status()) {
  // Offline inference enters through pybind and never parses flags, so the
  // option is the authoritative source here; the serving binary already filled
  // it from --model_impl.
  const std::optional<std::string> model_impl =
      ModelConfig::normalize_model_impl(options_.model_impl());
  CHECK(model_impl.has_value()) << "model_impl must be native, python, or py";
  ModelConfig::get_instance().model_impl(model_impl.value());
  const auto model_path =
      std::filesystem::path(options_.model_path()).lexically_normal();
  if (options_.host_blocks_factor() > 1.0) {
    // VLM and VLMSSM share the LLM host-cache path. DeepSeek-V4.1 serving
    // uses --backend=vlm with DSpark, and Store requires host_blocks_factor
    // above 1. Gate only the speculative algorithms that do not share the
    // target cache.
    const bool supports_host_offload =
        type == EngineType::LLM || type == EngineType::VLM ||
        ((type == EngineType::SSM || type == EngineType::VLMSSM) &&
         SpeculativeConfig::supports_host_kv_cache(
             options_.speculative_algorithm()));
    CHECK(supports_host_offload)
        << "Basic host KV cache offload supports LLM/VLM engines and "
           "model-based speculative engines only.";
  }
  const auto visible_devices = DeviceNameUtils::parse_devices("auto");
  // parse_devices("auto") answers {kCPU} on a host with no visible
  // accelerator; serving still catches that inside get_device_idx, but the
  // offline branch takes the set as-is, so require a real device for both
  // paths here.
  CHECK(!visible_devices.empty() && !visible_devices.front().is_cpu())
      << "At least one accelerator device must be visible.";
  CHECK(!(options_.enable_offline_inference() && options_.nnodes() > 1))
      << "Offline inference drives every device of a single node through "
         "spawned workers; multi-node offline inference is not supported. "
         "Use online serving for nnodes > 1.";
  std::vector<torch::Device> devices;
  if (options_.enable_offline_inference()) {
    // A single offline process drives every visible device: rank 0 runs here
    // and the remaining ranks are spawned workers, so the device set is the
    // whole visible set.
    devices = visible_devices;
  } else {
    // Serving runs one worker per process: take this process's own device and
    // keep node_rank as the distributed identity.
    const int32_t visible_device_count =
        static_cast<int32_t>(visible_devices.size());
    const int32_t device_idx = DeviceNameUtils::get_device_idx(
        options_.node_rank(), options_.nnodes(), visible_device_count);
    devices = {visible_devices[device_idx]};
  }
  // Every device hosts one worker and every node contributes its own devices.
  // Offline is pinned to nnodes == 1 by the CHECK above (spawned workers do
  // not propagate the topology), so devices here is always the single node's
  // visible set; revisit this product if multi-node offline ever lands.
  const int32_t global_world_size =
      static_cast<int32_t>(devices.size()) * options_.nnodes();
  const int32_t kv_split_size =
      ParallelConfig::get_instance().kv_split_size_effective();
  const bool npu_dcp_requested =
      Platform::is_npu() && options_.cp_size() == 1 && kv_split_size > 1;
  const bool native_qwen_dcp_requested =
      npu_dcp_requested && !ModelConfig::is_python_model_impl(
                               ModelConfig::get_instance().model_impl());
  if (npu_dcp_requested) {
    const std::optional<std::string> dcp_topology_error =
        validate_qwen_dcp_topology(
            global_world_size, options_.dp_size(), kv_split_size);
    CHECK(!dcp_topology_error.has_value()) << dcp_topology_error.value();
  }
  std::string model_type;
  if (options_.backend() == "llm" || native_qwen_dcp_requested ||
      (options_.cp_size() > 1 && Platform::uses_model_cp_sharding()) ||
      (ModelConfig::is_python_model_impl(
           ModelConfig::get_instance().model_impl()) &&
       options_.num_speculative_tokens() > 0)) {
    model_type = util::get_model_type(model_path, options_.backend());
  }
  // Reject unsupported speculative combinations before creating workers.
  if (util::is_deepseek_v41_model_type(model_type)) {
    const bool v41_speculative_requested =
        options_.num_speculative_tokens() > 0 ||
        !options_.draft_model_path().value_or("").empty();
    if (v41_speculative_requested) {
      CHECK(SpeculativeConfig::is_dspark_algorithm(
          options_.speculative_algorithm()))
          << "DeepSeek-V4.1 speculative decoding requires DSpark.";
    }
  }
  const std::optional<std::string> speculative_error =
      ModelConfig::validate_python_speculative_decode(
          ModelConfig::get_instance().model_impl(),
          model_type,
          options_.speculative_algorithm(),
          options_.num_speculative_tokens());
  CHECK(!speculative_error.has_value()) << speculative_error.value();
  const std::optional<std::string> cp_error =
      validate_model_cp(options_, type, model_type, global_world_size);
  CHECK(!cp_error.has_value()) << cp_error.value();
  options_.enable_mla(util::should_enable_mla(model_path, options_.backend()));
  print_startup_banner(model_path, options_.backend(), options_.node_rank());
  LOG(INFO) << "Master init options: " << options_.to_string();
  ParallelConfig::get_instance().cp_size(options_.cp_size());
  // cp_size <= 1 -> "disabled", otherwise "model" (model-side CP).
  const char* cp_sharding_stage =
      options_.cp_size() <= 1 ? "disabled" : "model";
  LOG(INFO) << "Resolved CP config: cp_size=" << options_.cp_size()
            << ", world_size=" << global_world_size
            << ", dp_size=" << options_.dp_size()
            << ", ep_size=" << options_.ep_size()
            << ", cp_sharding_stage=" << cp_sharding_stage
            << ", instance_role=" << options_.instance_role().to_string();

  // Allow brpc receive SIGTREM and SIGINT signal.
  brpc::FLAGS_graceful_quit_on_sigterm = true;
  brpc::FLAGS_graceful_quit_on_sighup = true;

#if defined(USE_NPU)
  EPLBConfig& eplb_config = EPLBConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (options.rank_tablefile().has_value()) {
    eplb_config.rank_tablefile(options.rank_tablefile().value());
  }
  if (options.communication_backend().has_value()) {
    parallel_config.communication_backend(
        options.communication_backend().value());
  }
  validate_rank_tablefile_backend();
  parallel_state::sync_torch_npu_rank_table_file_env(
      eplb_config.rank_tablefile());
  if (options.expert_parallel_degree().has_value()) {
    eplb_config.expert_parallel_degree(
        options.expert_parallel_degree().value());
  }
  if (options.enable_eplb().has_value()) {
    eplb_config.enable_eplb(options.enable_eplb().value());
  }
  if (options.redundant_experts_num().has_value()) {
    eplb_config.redundant_experts_num(options.redundant_experts_num().value());
  }
  if (options.eplb_update_interval().has_value()) {
    eplb_config.eplb_update_interval(options.eplb_update_interval().value());
  }
  if (options.eplb_min_peak_load_improvement().has_value()) {
    eplb_config.eplb_min_peak_load_improvement(
        options.eplb_min_peak_load_improvement().value());
  }
  CHECK(!eplb_config.enable_eplb() ||
        (type != EngineType::VLM && type != EngineType::VLMSSM))
      << "EPLB does not support VLM engines because VLMEngine has no EPLB "
         "control plane.";
  resolve_npu_kernel_backend_for_options(&options_);
#endif
  if (native_qwen_dcp_requested) {
    const bool supported_runtime =
        options_.npu_kernel_backend() == "TORCH" && type == EngineType::LLM &&
        options_.task_type() == "generate" && options_.ep_size() == 1 &&
        !options_.enable_eplb().value_or(false) &&
        options_.expert_parallel_degree().value_or(0) <= 1 &&
        options_.is_local() && options_.rank_tablefile().value_or("").empty() &&
        options_.host_blocks_factor() <= 1.0 &&
        !options_.enable_kvcache_store() && !options_.enable_disagg_pd() &&
        options_.instance_role() == InstanceRole::DEFAULT &&
        options_.draft_model_path().value_or("").empty() &&
        options_.num_speculative_tokens() == 0 &&
        ExecutionConfig::get_instance().enable_fia_decode() &&
        (model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text");
    CHECK(supported_runtime)
        << "Qwen3.5 DCP supports only replicated-GQA local NPU TORCH LLM "
           "generation without CP, EP/EPLB, host/KVStore offload, "
           "P/D, or "
           "speculative decoding, and requires enable_fia_decode=true";
  }
  validate_layerwise_split_size_startup_config(
      options_, model_type, global_world_size);
  ParallelConfig::get_instance().enable_multi_stream_parallel(
      options.enable_multi_stream_parallel() && (options.nnodes() > 1));
  if (ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    LOG(FATAL)
        << "Multi-stream parallel is refactoring now, will be supported later.";
  }
  // construct engine
  LOG(INFO) << "Creating engine with devices: "
            << DeviceNameUtils::to_string(devices);

  if (options_.enable_disagg_pd()) {
    // Enable service routing in disagg pd mode
    options_.enable_service_routing(true);
    if (options_.instance_role() == InstanceRole::PREFILL) {
      // Disable schedule overlap for prefill instance in disagg pd mode
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for prefill instance "
                      "in disagg pd mode.";
    }
  }

  if (options_.task_type() == "mm_embed") {
    options_.enable_chunked_prefill(false);
    options_.enable_prefix_cache(false);
    SchedulerConfig::get_instance().enable_chunked_prefill(false);
    KVCacheConfig::get_instance().enable_prefix_cache(false);
    LOG(WARNING) << "Disabling chunked prefill and prefix cache for "
                    "task=mm_embed to process all multimodal inputs.";
  }

  if (type == EngineType::VLM) {
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .model_id(options_.model_id())
        .devices(devices)
        .backend(options.backend())
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_encoder_cache_size(options.max_encoder_cache_size())
        .max_processor_cache_items(options.max_processor_cache_items())
        .max_linear_state_cache_slots(options.max_linear_state_cache_slots())
        .task_type(options.task_type())
        .enable_mla(options_.enable_mla())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .instance_name(options_.instance_name())
        .instance_role(options_.instance_role())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill());

    apply_runtime_kv_cache_options(options_, eng_options);

    auto engine = std::make_unique<VLMEngine>(eng_options);
    engine_ = std::move(engine);
  } else if (type == EngineType::SSM || type == EngineType::VLMSSM) {
    if (type == EngineType::VLMSSM) {
      const std::string model_type =
          util::get_model_type(options_.model_path(), options_.backend());
      const std::string& algorithm = options_.speculative_algorithm();
      const bool supports_pd =
          vlm_speculative_pd_transfer_support(model_type, algorithm) !=
          VlmPdTransferSupport::UNSUPPORTED;
      // JoyaiASR + Suffix serves aggregated requests through the service
      // router, but its draft has no P/D bootstrap embedding, so disaggregated
      // PD stays rejected by the supports_pd gate below.
      const bool supports_routing =
          supports_pd ||
          SpeculativeConfig::is_joyai_asr_suffix(model_type, algorithm);
      CHECK(!options_.enable_disagg_pd() || supports_pd)
          << "VLM speculative PD does not support model_type=" << model_type
          << ", speculative_algorithm=" << algorithm
          << ", platform=" << Platform::type_str();
      const auto transfer_mode_error =
          validate_vlm_speculative_pd_transfer_mode(options_, model_type);
      CHECK(!transfer_mode_error.has_value()) << transfer_mode_error.value();
      CHECK(!options_.enable_service_routing() || supports_routing)
          << "VLM speculative service routing does not support model_type="
          << model_type << ", speculative_algorithm=" << algorithm
          << ", platform=" << Platform::type_str();
    }
    // create a speculative engine if draft model path is provided
    const std::string draft_model_path =
        options_.draft_model_path().value_or("");
    const bool use_suffix_spec = SpeculativeConfig::is_suffix_algorithm(
        options_.speculative_algorithm());
    CHECK(use_suffix_spec || !draft_model_path.empty())
        << "draft model path is required unless --speculative_algorithm=Suffix";
    // Draft model shares the same devices as the target model.
    const auto& draft_devices = devices;
    LOG(INFO) << "Using draft devices: "
              << DeviceNameUtils::to_string(draft_devices);
    runtime::Options spec_options;
    spec_options.model_path(options_.model_path())
        .model_id(options_.model_id())
        .draft_model_path(draft_model_path)
        .devices(devices)
        .draft_devices(draft_devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_encoder_cache_size(options_.max_encoder_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .num_speculative_tokens(options_.num_speculative_tokens())
        .speculative_algorithm(options_.speculative_algorithm())
        .enable_probabilistic_draft(options_.enable_probabilistic_draft())
        .enable_mtp_draft_body_tp1(options_.enable_mtp_draft_body_tp1())
        .speculative_suffix_cache_max_depth(
            options_.speculative_suffix_cache_max_depth())
        .speculative_suffix_max_cached_requests(
            options_.speculative_suffix_max_cached_requests())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode());
    apply_runtime_kv_cache_options(options_, spec_options);

    if (type == EngineType::VLMSSM) {
      engine_ =
          std::make_unique<SpeculativeEngineBase<VLMEngine>>(spec_options);
    } else {
      engine_ =
          std::make_unique<SpeculativeEngineBase<LLMEngine>>(spec_options);
    }
  } else if (type == EngineType::LLM) {
    if (options_.task_type() == "embed" || options.task_type() == "mm_embed") {
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for embedding model, "
                      "avoiding performance degradation.";
    }
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .server_idx(options_.server_idx())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .enable_sleep_mode(options_.enable_sleep_mode())
        .model_id(options_.model_id());
    apply_runtime_kv_cache_options(options_, eng_options);

    engine_ = std::make_unique<LLMEngine>(eng_options);
  } else if (type == EngineType::DIT) {
    // construct dit engine
    runtime::Options eng_options;
    eng_options.model_path(options.model_path())
        .model_id(options.model_id())
        .devices(devices)
        .backend(options.backend())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .spawn_worker_path(options_.spawn_worker_path())
        .disable_log_stats(options_.disable_log_stats())
        .max_memory_utilization(options_.max_memory_utilization())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .task_type(options_.task_type())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .node_rank(options_.node_rank())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .tp_size(options_.tp_size())
        .sp_size(options_.sp_size())
        .cfg_size(options_.cfg_size())
        .vae_size(options_.vae_size())
        .text_encoder_tp_size(options_.text_encoder_tp_size());

    auto dit_engine = std::make_unique<DiTEngine>(eng_options);
    engine_ = std::move(dit_engine);
  } else {
    LOG(WARNING) << "Not supported llm engine type: "
                 << static_cast<size_t>(type);
  }

  if (!is_leader()) {
    const std::string master_node_addr =
        options_.master_node_addr().value_or("");
    if (master_node_addr.empty()) {
      LOG(FATAL) << "Multi-node serving requires --master_node_addr, "
                    "current value is empty.";
    }
  }
}

std::atomic<bool> Master::idle_running_{false};

void Master::handle_shutdown_signal(int /*signum*/) {
  idle_running_.store(false, std::memory_order_relaxed);
}

Master::~Master() {
  idle_running_.store(false, std::memory_order_relaxed);
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::wait() {
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::run() {
  idle_running_.store(true, std::memory_order_relaxed);
  signal(SIGINT, Master::handle_shutdown_signal);
  signal(SIGTERM, Master::handle_shutdown_signal);

  idle_thread_ = std::thread([]() {
    while (idle_running_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  } else if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  } else if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  } else {
    LOG(FATAL) << "Failed to create master, backend is" << backend;
    return nullptr;
  }
}

}  // namespace xllm
