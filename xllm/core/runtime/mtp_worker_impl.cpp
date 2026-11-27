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

#include "mtp_worker_impl.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <unordered_set>

#include "common/metrics.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "core/framework/block/block_utils.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/constraint/draft_validation.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/runtime/draft_model_config.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/speculative/mtp_async_input_builder.h"
#include "core/framework/speculative/mtp_async_replay_builder.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "core/runtime/json_object_decode_adapter.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/speculative_worker_impl.h"
#include "runtime/speculative_worker_utils.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {

using speculative_worker::broadcast_speculative_token_ids;
using speculative_worker::clear_all_output_embeddings;
using speculative_worker::clear_selected_embeddings;
using speculative_worker::record_metadata_ready_event;
using speculative_worker::wait_metadata_ready_event;

void clear_draft_state(ModelInputParams& params) {
  params.clear_linear_attention_state();
  params.num_accepted_tokens = torch::Tensor();
  params.num_accepted_tokens_host.clear();
}

bool has_active_dp_tokens(const ForwardInput& input) {
  const ParallelInput& parallel = input.input_params.parallel;
  const std::vector<int32_t>& token_nums =
      parallel.raw_dp_global_token_nums.empty()
          ? parallel.dp_global_token_nums
          : parallel.raw_dp_global_token_nums;
  return std::any_of(token_nums.begin(), token_nums.end(), [](int32_t count) {
    return count > 0;
  });
}

void append_request_failures(ForwardOutput& destination,
                             const ForwardOutput& source) {
  destination.failed_request_ids.insert(destination.failed_request_ids.end(),
                                        source.failed_request_ids.begin(),
                                        source.failed_request_ids.end());
  destination.unsafe_request_ids.insert(destination.unsafe_request_ids.end(),
                                        source.unsafe_request_ids.begin(),
                                        source.unsafe_request_ids.end());
}

bool should_broadcast_speculative_token_ids(const ParallelArgs& parallel_args,
                                            bool enable_spec_token_broadcast,
                                            SamplingMode sampling_mode) {
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  return use_orthogonal_cp_consensus ||
         (enable_spec_token_broadcast && sampling_mode != SamplingMode::GREEDY);
}

constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;

void record_current_metadata_ready_event(ForwardInput& input, Stream& stream) {
  wait_metadata_ready_event(input, stream);
  record_metadata_ready_event(stream, input);
}

void record_output_ready_event(ForwardOutput& output, Stream& stream) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    const int32_t ret = stream.synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  }
  output.ready_event = event;
}

void finalize_output_on_stream(ForwardOutput& output,
                               Stream& stream,
                               bool allow_async) {
  if (allow_async) {
    record_output_ready_event(output, stream);
    return;
  }
  const int32_t ret = stream.synchronize();
  CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  output.retained_inputs.clear();
}

#if defined(USE_NPU)
void clear_expanded_spec_verify_graph_input(ModelInputParams& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
}

bool build_expanded_spec_verify_graph_host_input(
    ModelInputParams& input_params) {
  clear_expanded_spec_verify_graph_input(input_params);
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return false;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return false;
  }
  std::vector<int32_t> expanded_kv_seq_lens =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  if (expanded_kv_seq_lens.empty()) {
    return false;
  }

  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  input_params.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
  return true;
}

void bind_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                           const torch::Device& device,
                                           bool kv_lens_already_bound,
                                           int32_t block_size,
                                           bool defer_paged_layout = false) {
  if (!input_params.graph.use_expanded_decode_for_spec_verify_attention) {
    return;
  }
  const auto& q_seq_lens = input_params.attention.host.q_seq_lens;
  if (!kv_lens_already_bound) {
    torch::Tensor expanded_kv_seq_lens_host =
        torch::tensor(input_params.graph.expanded_kv_seq_lens_vec,
                      torch::TensorOptions()
                          .dtype(torch::kInt)
                          .device(torch::kCPU)
                          .pinned_memory(true));
    input_params.graph.expanded_kv_seq_lens =
        expanded_kv_seq_lens_host.to(device, /*non_blocking=*/true);
  }

  torch::Tensor expanded_block_tables =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_block_tables(
          input_params.attention.device.block_tables, q_seq_lens);
  if (defer_paged_layout) {
    // The accepted host lengths are not yet available. Keep the fixed device
    // rows for async correction, and build paging once after the host flush.
    input_params.graph.expanded_block_tables = std::move(expanded_block_tables);
    return;
  }
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      input_params,
      input_params.graph.expanded_kv_seq_lens,
      expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      block_size);
}

void build_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                            const torch::Device& device,
                                            int32_t block_size,
                                            bool defer_paged_layout = false) {
  build_expanded_spec_verify_graph_host_input(input_params);
  bind_expanded_spec_verify_graph_input(
      input_params, device, false, block_size, defer_paged_layout);
}

#endif

void clear_ready_events(ForwardInput& input) {
  input.metadata_ready_event.reset();
}

std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput& processed_input) {
  speculative_worker::WorkerStreamOptions stream_options;
  stream_options.record_input_ready_event = &prepare_stream != &compute_stream;
  stream_options.record_output_ready_event = false;
  return speculative_worker::run_worker_no_sync(worker,
                                                input,
                                                prepare_stream,
                                                compute_stream,
                                                &processed_input,
                                                stream_options);
}

torch::Tensor clone_host_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU host tensor";
  return tensor.contiguous().clone();
}

torch::Tensor clone_host_block_table(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU host tensor";
  if (tensor.layout() == torch::kStrided && tensor.dim() == 2 &&
      tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous() &&
      !tensor.is_neg() && !tensor.is_conj()) {
    // Keep an independent snapshot without launching an ATen parallel copy
    // team on every TP rank. Match clone's non-pinned CPU allocation.
    torch::Tensor snapshot =
        torch::empty(tensor.sizes(), tensor.options().pinned_memory(false));
    if (tensor.numel() != 0) {
      std::memcpy(snapshot.data_ptr(), tensor.data_ptr(), tensor.nbytes());
    }
    return snapshot;
  }
  return clone_host_tensor(tensor);
}

void stabilize_decode_host_tensors(ForwardInput& input) {
  input.token_ids_host = clone_host_tensor(input.token_ids_host);
  input.positions_host = clone_host_tensor(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_host_block_table(input.input_params.attention.host.block_tables);
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    block_table = clone_host_block_table(block_table);
  }
}

void set_token_ids_device_tensor(ForwardInput& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options,
                                 Stream& compute_stream) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  input.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids =
      safe_to(flat_token_ids, token_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

torch::Tensor to_cpu_int_tensor_for_read(const torch::Tensor& values) {
  return safe_to(values.flatten(),
                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU),
                 false)
      .contiguous();
}

void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = {token_ids_host.data_ptr<int32_t>(),
                              static_cast<size_t>(token_ids_host.numel())};
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const EmbeddingCache::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
    CHECK(state.embedding.defined())
        << "MTP decode target state embedding is undefined, request_id="
        << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}

void replace_host_token_placeholders(ForwardInput& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

  input.device_tensors_ready = false;
  torch::Tensor replacement_cpu = to_cpu_int_tensor_for_read(replacements);
  int32_t* token_ids = input.token_ids_host.data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = {
      replacement_cpu.data_ptr<int32_t>(),
      static_cast<size_t>(replacement_cpu.numel())};

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids =
        safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
    input.device_tensors_ready = true;
  }
}

void set_positions_tensor(ForwardInput& input,
                          const torch::Tensor& positions_host,
                          const torch::TensorOptions& device_options) {
  input.device_tensors_ready = false;
  input.positions_host = positions_host;
  input.positions =
      safe_to(input.positions_host, device_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

runtime::Options mtp_target_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options mtp_draft_options(const runtime::Options& options) {
  runtime::Options draft_options = options;
  draft_options.enable_schedule_overlap(false)
      .backend("llm")
      .is_draft_engine(true)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(true);
  return draft_options;
}

ParallelArgs mtp_draft_parallel_args(const ParallelArgs& parallel_args,
                                     const runtime::Options& options) {
  if (!options.enable_mtp_draft_body_tp1()) {
    return parallel_args;
  }
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "MTP draft body TP1 requires a single-rank process group";
  ParallelArgs draft_args = parallel_args;
  draft_args.rank(0)
      .world_size(1)
      .dp_size(1)
      .ep_size(1)
      .cp_size(1)
      .tp_size(1)
      .sp_size(1);
  draft_args.mapping_data(nlohmann::json{});
  draft_args.process_group_ = parallel_args.single_rank_group_;
  draft_args.dp_local_process_group_ = parallel_args.single_rank_group_;
  draft_args.lm_head_group_ = parallel_args.tp_group_;
  draft_args.tp_group_ = parallel_args.single_rank_group_;
  draft_args.cp_group_ = parallel_args.single_rank_group_;
  draft_args.moe_ep_group_ = parallel_args.single_rank_group_;
  draft_args.moe_tp_group_ = parallel_args.single_rank_group_;
  return draft_args;
}

}  // namespace

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             WorkerType worker_type)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    mtp_target_options(options),
                    mtp_draft_options(options),
                    worker_type) {}

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             const runtime::Options& target_options,
                             const runtime::Options& draft_options,
                             WorkerType worker_type)
    : DraftModelSpecWorkerImpl(
          parallel_args,
          device,
          options,
          target_options,
          worker_type,
          [&parallel_args, &device, &options, &draft_options] {
            return std::make_unique<LLMWorkerImpl>(
                mtp_draft_parallel_args(parallel_args, options),
                device,
                mtp_draft_options(draft_options));
          }) {}

MTPWorkerImpl::~MTPWorkerImpl() = default;

void MTPWorkerImpl::apply_target_model_overrides(ModelArgs& args) const {
  SpeculativeWorkerImpl::apply_target_model_overrides(args);
#if defined(USE_NPU)
  if (SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm()) &&
      (util::is_deepseek_v4_model_type(args.model_type()) ||
       is_glm5_next_target_model(args))) {
    args.num_speculative_tokens(options_.num_speculative_tokens());
  }
#endif
}

void MTPWorkerImpl::apply_draft_model_overrides(ModelArgs& args) const {
  configure_mtp_family_draft(args, draft_impl_->get_options());
}

void MTPWorkerImpl::on_target_model_loaded() {
  embedding_size_ = impl_->hidden_size();
  target_spec_verify_mode_ = mtp_async::classify_target_spec_verify_mode(
      context_.get_model_args().model_type(),
      ModelConfig::is_python_model_impl(context_.get_model_impl()));
#if defined(USE_NPU)
  if (supports_expanded_spec_verify()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Expanded MTP verify requires the NPU Torch backend";
  }
#endif
}

void MTPWorkerImpl::on_draft_model_loaded(
    const std::string& /*model_weights_path*/) {
  const std::string& draft_model_type =
      draft_impl_->context_.get_model_args().model_type();
  combined_draft_execution_path_ =
      mtp_async::classify_combined_draft_execution_path(draft_model_type);
  init_draft_context_policy();
}

int64_t MTPWorkerImpl::get_embedding_placeholder_size() const {
  // DeepSeek-V4 MTP stashes the pre-hc_head 3D hidden flattened to
  // [num_tokens, hc_mult*hidden], so the cache placeholder must cover
  // hc_mult*hidden per row.
  if (impl_ != nullptr) {
    const ModelArgs& args = impl_->context_.get_model_args();
    return mtp_hidden_state_width(args);
  }
  return static_cast<int64_t>(embedding_size_);
}

KVCacheShape MTPWorkerImpl::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  if (options_.enable_mtp_draft_body_tp1()) {
    return build_draft_kv_cache_shape(target_kv_cache_shape,
                                      /*draft_world_size=*/1);
  }
  return target_kv_cache_shape;
}

bool MTPWorkerImpl::positions_are_decoupled_from_kv_length() const {
  if (impl_ == nullptr) {
    return false;
  }
  const ModelArgs& target_args = impl_->context_.get_model_args();
  return !target_args.rope_scaling_mrope_section().empty();
}

bool MTPWorkerImpl::supports_expanded_spec_verify() const {
  return target_spec_verify_mode_ ==
         mtp_async::TargetSpecVerifyMode::EXPANDED_VERIFY;
}

bool MTPWorkerImpl::supports_explicit_spec_verify_replay_update() const {
  return mtp_async::supports_native_spec_verify_replay_update(
      target_spec_verify_mode_,
      ModelConfig::is_python_model_impl(context_.get_model_impl()));
}

bool MTPWorkerImpl::should_use_explicit_spec_verify_replay_update(
    const ForwardInput& input) const {
#if defined(USE_NPU)
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  const int64_t num_sequences = input.input_params.meta.num_sequences;
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !::xllm::ExecutionConfig::get_instance()
           .enable_graph_mode_decode_no_padding() ||
      !supports_explicit_spec_verify_replay_update() || impl_ == nullptr ||
      options_.num_speculative_tokens() == 0 || options_.block_size() <= 0 ||
      !mtp_async::has_speculative_verify_block_table_layout(
          block_tables, input.input_params.multi_block_tables, num_sequences)) {
    return false;
  }
  const int64_t block_table_width =
      block_tables.defined()
          ? spec_verify_block_table_width(block_tables)
          : mtp_async::speculative_verify_block_table_capacity(
                impl_->context_.get_model_args().max_position_embeddings(),
                options_.block_size());
  if (block_table_width <= 0 ||
      block_table_width > kMaxSpecVerifyGraphUpdateBlockTableWidth) {
    return false;
  }
  const int64_t spec_width =
      static_cast<int64_t>(options_.num_speculative_tokens()) + 1;
  return kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
      spec_width, options_.block_size());
#else
  (void)input;
  return false;
#endif
}

int64_t MTPWorkerImpl::spec_verify_block_table_width(
    const torch::Tensor& block_tables) const {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  CHECK_GT(options_.block_size(), 0);
  int64_t required_width = block_tables.size(1);
  if (impl_ != nullptr) {
    const int64_t declared_capacity =
        mtp_async::speculative_verify_block_table_capacity(
            impl_->context_.get_model_args().max_position_embeddings(),
            options_.block_size());
    CHECK_LE(required_width, declared_capacity)
        << "block table width exceeds the model position capacity";
    required_width = declared_capacity;
  }
  return required_width;
}

int64_t MTPWorkerImpl::active_spec_verify_block_table_width(
    const ModelInputParams& input_params) const {
  CHECK_GT(options_.block_size(), 0);
  const int64_t effective_kv_seq_len =
      std::max<int64_t>(input_params.meta.kv_max_seq_len, 1);
  return (effective_kv_seq_len + options_.block_size() - 1) /
         options_.block_size();
}

torch::Tensor MTPWorkerImpl::acquire_spec_verify_control_block_table(
    int64_t num_sequences,
    int64_t block_table_capacity) {
  CHECK_GT(num_sequences, 0);
  CHECK_GT(block_table_capacity, 0);

  const int64_t row_capacity = std::max(
      num_sequences, static_cast<int64_t>(options_.max_seqs_per_batch()));
  const bool needs_allocation =
      !spec_verify_control_block_table_buffer_.defined() ||
      spec_verify_control_block_table_buffer_.scalar_type() != torch::kInt32 ||
      spec_verify_control_block_table_buffer_.dim() != 2 ||
      spec_verify_control_block_table_buffer_.size(0) < row_capacity ||
      spec_verify_control_block_table_buffer_.size(1) < block_table_capacity;
  if (needs_allocation) {
    spec_verify_control_block_table_buffer_ =
        mtp_async::make_speculative_verify_control_block_table(
            row_capacity, block_table_capacity);
  }

  return spec_verify_control_block_table_buffer_
      .narrow(/*dim=*/0, /*start=*/0, num_sequences)
      .narrow(/*dim=*/1, /*start=*/0, block_table_capacity);
}

void MTPWorkerImpl::ensure_spec_verify_control_block_table(
    ModelInputParams& input_params,
    int64_t num_sequences,
    int64_t active_width) {
  auto& block_tables = input_params.attention.host.block_tables;
  if (block_tables.defined()) {
    return;
  }
  CHECK(!input_params.multi_block_tables.empty())
      << "missing model-managed block tables for spec verify";
  CHECK(impl_ != nullptr) << "target model must be initialized";
  const int64_t block_table_capacity =
      mtp_async::speculative_verify_block_table_capacity(
          impl_->context_.get_model_args().max_position_embeddings(),
          options_.block_size());
  CHECK_GT(active_width, 0);
  CHECK_LE(active_width, block_table_capacity)
      << "active spec verify block table exceeds declared capacity";
  // Keep one reusable maximum-capacity backing buffer, but expose only the
  // columns consumed by this generic expanded-verify step to bound metadata
  // packing costs.
  block_tables = acquire_spec_verify_control_block_table(num_sequences,
                                                         block_table_capacity)
                     .narrow(/*dim=*/1, /*start=*/0, active_width);
}

bool MTPWorkerImpl::use_chunked_prefill_spec_verify_path() const {
  return target_spec_verify_mode_ ==
             mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
         supports_expanded_spec_verify();
}

bool MTPWorkerImpl::uses_speculative_linear_state_checkpoints() const {
#if defined(USE_NPU)
  if (impl_ == nullptr ||
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }
  const std::string& model_type = impl_->context_.get_model_args().model_type();
  return is_qwen3_5_target_model_type(model_type) ||
         is_glm5_next_target_model_type(model_type);
#else
  return false;
#endif
}

void MTPWorkerImpl::init_draft_context_policy() {
  std::string resolved_draft_model_name;
  std::string error_message;
  CHECK(resolve_model_registration_name(
      draft_impl_->context_.get_model_args().model_type(),
      &resolved_draft_model_name,
      &error_message))
      << error_message;
  const MtpModelCapabilities capabilities =
      ModelRegistry::get_mtp_capabilities(resolved_draft_model_name);
  const bool is_python_target =
      ModelConfig::is_python_model_impl(context_.get_model_impl());
  const bool is_python_draft =
      ModelConfig::is_python_model_impl(draft_impl_->context_.get_model_impl());
  const bool legacy_tail =
      capabilities.allow_tail_fallback && (is_python_target || is_python_draft);
  const DraftContextUpdate update =
      capabilities.supports_accepted_span_replay && !legacy_tail
          ? DraftContextUpdate::ACCEPTED_SPAN_REPLAY
          : DraftContextUpdate::TAIL_EXTEND;
  std::string resolved_target_model_name;
  CHECK(resolve_model_registration_name(context_.get_model_args().model_type(),
                                        &resolved_target_model_name,
                                        &error_message))
      << error_message;
  const MtpModelCapabilities target_capabilities =
      ModelRegistry::get_mtp_capabilities(resolved_target_model_name);
  const bool target_capable =
      target_spec_verify_mode_ ==
              mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL
          ? target_capabilities.supports_causal_replay_target
          : target_capabilities.supports_expanded_replay_target;
  const bool replay_pair =
      target_capable &&
      target_capabilities.replay_family != MtpReplayFamily::NONE &&
      target_capabilities.replay_family == capabilities.replay_family;
  CHECK(mtp_async::replay_compatible(target_spec_verify_mode_,
                                     update,
                                     replay_pair,
                                     capabilities.supports_accepted_span_replay,
                                     is_python_target,
                                     is_python_draft))
      << "Incompatible draft context update: target="
      << context_.get_model_args().model_type()
      << ", verify_mode=" << static_cast<int32_t>(target_spec_verify_mode_)
      << ", draft=" << draft_impl_->context_.get_model_args().model_type()
      << ", update=" << static_cast<int32_t>(update);
  const auto replay_policy = mtp_async::draft_context_replay_semantics(
      update, options_.num_speculative_tokens());
  if (draft_context_replay_policy_.has_value()) {
    CHECK_EQ(draft_context_replay_policy_->full_target_replay,
             replay_policy.full_target_replay)
        << "Draft context update policy changed after initialization.";
    CHECK_EQ(draft_context_replay_policy_->target_expansion_width,
             replay_policy.target_expansion_width)
        << "Draft context update policy changed after initialization.";
    CHECK_EQ(draft_context_replay_policy_->draft_position_offset,
             replay_policy.draft_position_offset)
        << "Draft context update policy changed after initialization.";
    return;
  }
  draft_context_replay_policy_ = replay_policy;
}

bool MTPWorkerImpl::requires_full_target_replay() const {
  CHECK(draft_context_replay_policy_.has_value())
      << "Draft context replay policy is not initialized.";
  return draft_context_replay_policy_->full_target_replay;
}

void MTPWorkerImpl::prepare_work_before_execute(const ForwardInput& input,
                                                ForwardInput& processed_input) {
  // Composite skips CP prepare; leaves run it on their execution streams.
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
}

bool MTPWorkerImpl::owns_npu_parallel_input_prepare() const { return false; }

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
  const bool use_prelaunched_first_draft =
      input.input_params.meta.batch_forward_type.is_decode() &&
      can_use_combined_first_draft(input) &&
      pending_draft_context_matches(input);
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // The preceding validation may have speculatively submitted draft-0 before
    // the scheduler learned that the batch had finished.  Keep its graph/input
    // buffers alive until the queued work completes, then discard the result.
    // This is a batch-exit slow path and is never taken in steady decode.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain final MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  flush_pending_target_context();

  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    ForwardInput target_prepared;
    ForwardInput draft_prepared;
    auto output = run_worker_no_sync_impl(
        *impl_, input, *prepare_stream_, *compute_stream_, target_prepared);
    auto draft_output = run_worker_no_sync_impl(*draft_impl_,
                                                input,
                                                *prepare_stream_,
                                                *compute_stream_,
                                                draft_prepared);
    if (draft_output.has_value()) {
      transfer_retained_inputs(*output, draft_output.value());
      append_request_failures(*output, draft_output.value());
    }
    clear_all_output_embeddings(*output);
    finalize_output_on_stream(
        *output, *compute_stream_, enable_schedule_overlap());
    return output;
  } else {
    ForwardInput draft_extend_prepared;
    std::vector<ForwardInput> draft_step_prepared(
        options_.num_speculative_tokens());
    ForwardInput target_prepared;
    std::vector<ForwardOutput> draft_outputs;
    draft_outputs.reserve(options_.num_speculative_tokens());

    // Empty ranks still replay the same draft graph as ranks with target
    // context.
    ForwardInput draft_input = input;
    draft_input.input_params.embedding.input_embedding =
        embedding_cache_->embedding_placeholder().unsqueeze(0);
    ForwardInput new_input = draft_input;
    new_input.input_params.meta.mtp_draft_step = 0;
    CHECK(draft_context_replay_policy_.has_value())
        << "Draft context replay policy is not initialized.";
    const int32_t extend_width =
        draft_context_replay_policy_->target_expansion_width;
    for (int32_t& token_num :
         new_input.input_params.parallel.dp_global_token_nums) {
      token_num *= extend_width;
    }
    for (int32_t& token_num :
         new_input.input_params.parallel.raw_dp_global_token_nums) {
      token_num *= extend_width;
    }
    new_input.input_params.expert.eplb_decode_token_mask =
        eplb::expand_decode_token_mask(
            new_input.input_params.expert.eplb_decode_token_mask, extend_width);
    if (use_prelaunched_first_draft) {
      draft_outputs.emplace_back(
          std::move(pending_draft_context_.output.value()));
      draft_extend_prepared = std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_outputs.emplace_back(run_worker_no_sync_impl(*draft_impl_,
                                                         new_input,
                                                         *prepare_stream_,
                                                         *compute_stream_,
                                                         draft_extend_prepared)
                                     .value());
    }

    for (int32_t i = 1; i < options_.num_speculative_tokens(); ++i) {
      // Idle DP shards must replay the same compute/reuse graph as busy peers.
      draft_input.input_params.meta.mtp_draft_step = i;
      draft_outputs.emplace_back(run_worker_no_sync_impl(*draft_impl_,
                                                         draft_input,
                                                         *prepare_stream_,
                                                         *compute_stream_,
                                                         draft_step_prepared[i])
                                     .value());
    }

    prepare_empty_validate_inputs(input, new_input);
    ForwardOutput output = run_worker_no_sync_impl(*impl_,
                                                   new_input,
                                                   *prepare_stream_,
                                                   *compute_stream_,
                                                   target_prepared)
                               .value();
    for (ForwardOutput& draft_output : draft_outputs) {
      transfer_retained_inputs(output, draft_output);
      append_request_failures(output, draft_output);
    }
    clear_all_output_embeddings(output);
    finalize_output_on_stream(
        output, *compute_stream_, enable_schedule_overlap());
    if (can_prelaunch_next_first_draft(input)) {
      ForwardInput next_first_draft_input = draft_input;
      for (int32_t& token_num :
           next_first_draft_input.input_params.parallel.dp_global_token_nums) {
        token_num *= 2;
      }
      for (int32_t& token_num : next_first_draft_input.input_params.parallel
                                    .raw_dp_global_token_nums) {
        token_num *= 2;
      }
      submit_pending_first_draft(input, std::move(next_first_draft_input));
    }
    return output;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_prefill(
    const ForwardInput& input) {
  flush_pending_target_context();

  Timer timer;
  ForwardInput target_prepared;
  ForwardInput draft_prepared;

  // run the target model to get first token and hidden states
  ForwardOutput output =
      run_worker_no_sync_impl(
          *impl_, input, *prepare_stream_, *compute_stream_, target_prepared)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());
  if (!output.sample_output.embeddings.defined() &&
      !output.failed_request_ids.empty()) {
    return output;
  }

  // MTP path that depends on hidden states.
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  check_draft_input_embedding(embeddings, "prefill");

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill seeds the MTP decode cache. Under orthogonal CP x TP each
    // CP shard samples independently unless this token is synchronized across
    // both axes; caching divergent tokens makes the first decode input disagree
    // with the non-driver CP shard's DecodeState.
    if (should_broadcast_speculative_token_ids(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.sampling_mode)) {
      broadcast_speculative_token_ids(output.sample_output.next_tokens,
                                      parallel_args_);
    }
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      replace_host_token_placeholders(prefill_input,
                                      -1,
                                      output.sample_output.next_tokens,
                                      prefill_input.token_ids.options());
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      record_current_metadata_ready_event(prefill_input, *compute_stream_);
    }
  }
  // generate kv cache for draft model
  timer.reset();
  ForwardOutput draft_output = run_worker_no_sync_impl(*draft_impl_,
                                                       prefill_input,
                                                       *prepare_stream_,
                                                       *compute_stream_,
                                                       draft_prepared)
                                   .value();
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    draft_vocab_.remap(draft_output.sample_output);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Prefer embeddings (global-real after CP merge); selected_embeddings is
    // a fallback.
    const torch::Tensor& target_hidden =
        output.sample_output.selected_embeddings.defined()
            ? output.sample_output.selected_embeddings
            : embeddings;
    torch::Tensor bootstrap_embeddings = target_hidden;
    if (bootstrap_embeddings.size(0) !=
        static_cast<int64_t>(
            input.input_params.embedding.embedding_ids.size())) {
      torch::Tensor bootstrap_idxes =
          input.sampling_params.selected_token_idxes.to(
              torch::dtype(torch::kLong).device(bootstrap_embeddings.device()));
      bootstrap_embeddings =
          bootstrap_embeddings.index_select(/*dim=*/0, bootstrap_idxes);
    }
    output.sample_output.embeddings = bootstrap_embeddings.detach();
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        target_hidden,
        input.sampling_params.selected_token_idxes,
        input.positions);
    clear_selected_embeddings(output);
  } else {
    clear_all_output_embeddings(output);
  }

  transfer_retained_inputs(output, draft_output);
  append_request_failures(output, draft_output);
  finalize_output_on_stream(
      output, *compute_stream_, enable_schedule_overlap());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !::xllm::EPLBConfig::get_instance().enable_eplb()) {
    if (!output.failed_request_ids.empty()) {
      return make_failed_output(std::move(output.failed_request_ids),
                                std::move(output.unsafe_request_ids));
    }
    return std::nullopt;
  }
  return output;
}

void MTPWorkerImpl::prepare_prefill_inputs(const ForwardInput& input,
                                           ForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  prefill_input = input.to(device_, dtype_);
  prepare_draft_sampling(prefill_input.sampling_params);
  clear_ready_events(prefill_input);
  // The draft model feeds on the target's hidden states (input_embedding set
  // below), never on raw multimodal input: strip the copied mm_data so the
  // draft executor does not drive vision encode on the text-only draft model
  // (whose python object has no ``encode``) and skips the device copy of
  // pixel_values it would never consume.
  prefill_input.input_params.multimodal.mm_data = MMBatchData();
  auto& input_params = prefill_input.input_params;
  // The Qwen draft is a pure full-attention model; without this cleanup the
  // target's recurrent slot metadata makes MTP prefill enter a stateful path
  // it has neither a validity mask nor a recurrent cache for.
  // Full-attention KPool drafts still own per-request compression tails.
  if (!draft_impl_->has_request_state_cache()) {
    input_params.clear_linear_attention_state();
  }
  auto& extra_token_ids = input_params.embedding.extra_token_ids;

  const torch::Tensor& token_ids = input.token_ids_host;
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(token_ids.numel());
  for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
    int32_t q_len = input_params.get_q_seq_len(i);
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.device_tensors_ready = false;
  prefill_input.token_ids_host = make_cpu_int_tensor(new_token_ids);
  prefill_input.token_ids = safe_to(prefill_input.token_ids_host,
                                    prefill_input.positions.options(),
                                    /*non_blocking=*/true);
  prefill_input.device_tensors_ready = true;
  record_metadata_ready_event(*prepare_stream_, prefill_input);
}

void MTPWorkerImpl::prepare_draft_sampling(
    SamplingParameters& sampling_params) const {
  sampling_params.filter_mask = torch::Tensor();
  sampling_params.filter_bitmask = torch::Tensor();
  if (!options_.enable_probabilistic_draft()) {
    force_greedy_draft_sampling(sampling_params);
  } else {
    sampling_params.logprobs = false;
    sampling_params.max_top_logprobs = 0;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  if (use_chunked_prefill_spec_verify_path()) {
    stabilize_decode_host_tensors(input);
  }
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const bool combined_first_draft_available =
      can_use_combined_first_draft(input);
  // Reuse draft-0 prelaunched for this same batch.
  const bool use_prelaunched_first_draft =
      combined_first_draft_available && pending_draft_context_matches(input);
  const bool matching_device_target_context =
      pending_target_context_matches(input);
  // Consume this batch's pending target context directly on device.
  const bool use_device_target_context =
      combined_first_draft_available && matching_device_target_context &&
      device_target_context_ready_for_batch(input);
  // Keep the device-side accepted state alive across a first-transition Host
  // cache flush. The prelaunched draft can be valid before the batch is marked
  // device-context ready, while flush_pending_target_context() clears the
  // owning context below.
  const torch::Tensor accepted_tokens = pending_target_context_.accepted_tokens;
  const torch::Tensor accepted_embeddings =
      pending_target_context_.accepted_embeddings;
  const torch::Tensor target_base_positions =
      pending_target_context_.base_positions;
  const torch::Tensor target_base_kv_seq_lens =
      pending_target_context_.base_kv_seq_lens;
  const StreamEventPtr target_context_ready_event =
      pending_target_context_.ready_event;
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // A batch transition invalidates the speculative prelaunch.  Drain it
    // before releasing its graph/input buffers; this slow path is outside
    // steady decode and preserves cache/buffer lifetime correctness.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain stale MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  if (!use_device_target_context) {
    // Batch transitions are uncommon in steady decode.  Materialize the most
    // recent target state only for that fallback; the normal path below never
    // synchronizes the worker thread with the NPU.
    flush_pending_target_context();
    if (matching_device_target_context) {
      // The first target-context publication for a new batch establishes the
      // scheduler's corrected position/KV base. Subsequent publications can
      // derive that base fully on device without waiting for the scheduler.
      device_context_ready_embedding_ids_ =
          input.input_params.embedding.embedding_ids;
      device_context_ready_request_ids_ =
          input.input_params.embedding.request_ids;
    } else if (!device_target_context_ready_for_batch(input)) {
      device_context_ready_embedding_ids_.clear();
      device_context_ready_request_ids_.clear();
    }
  }
  std::vector<ForwardOutput> draft_outputs;
  ForwardInput current_draft_input, validate_input, next_step_input;
  std::vector<ForwardInput> draft_prepared(num_speculative_tokens);
  Timer timer;
  CHECK(embedding_cache_ != nullptr) << "MTP embedding cache is not allocated";

  const auto& embedding = input.input_params.embedding;
  if (embedding.mtp_bootstrap_embeddings.defined()) {
    CHECK(input.token_ids_host.defined())
        << "MTP bootstrap requires host token ids";
    CHECK(input.token_ids_host.device().is_cpu())
        << "MTP bootstrap host token ids must be on CPU";
    CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
        << "MTP bootstrap host token ids must be int32";

    torch::Tensor bootstrap_embeddings =
        safe_to(embedding.mtp_bootstrap_embeddings,
                torch::dtype(dtype_).device(device_));
    CHECK_EQ(bootstrap_embeddings.size(0),
             static_cast<int64_t>(embedding.mtp_bootstrap_row_idxes.size()))
        << "MTP bootstrap row count mismatch";

    Slice<int32_t> token_ids = {
        input.token_ids_host.data_ptr<int32_t>(),
        static_cast<size_t>(input.token_ids_host.numel())};
    for (int32_t i = 0;
         i < static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
         ++i) {
      const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
      CHECK_GE(row_idx, 0) << "MTP bootstrap row index should be valid";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.embedding_ids.size()))
          << "MTP bootstrap row index exceeds embedding ids";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.request_ids.size()))
          << "MTP bootstrap row index exceeds request ids";
      CHECK_LT(static_cast<int64_t>(row_idx), input.token_ids_host.numel())
          << "MTP bootstrap row index exceeds token ids";
      embedding_cache_->write_mtp_bootstrap_context(
          embedding.embedding_ids[row_idx],
          embedding.request_ids[row_idx],
          token_ids[row_idx],
          bootstrap_embeddings[i],
          input.positions.dim() == 2 ? input.positions.select(1, row_idx)
                                     : torch::Tensor());
    }
  }

  ForwardInput metadata_template = input;
  // A prelaunched draft already owns its first-step inputs.
  if (!use_prelaunched_first_draft && use_device_target_context) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Build fixed-shape host metadata immediately while target verification is
    // still running. Use the maximum accepted draft offset for conservative
    // graph planning; actual values replace every device tensor below.
    std::vector<EmbeddingCache::DecodeState> template_states =
        mtp_async::advance_first_draft_template(
            metadata_template,
            embedding_cache_->embedding_placeholder(),
            num_speculative_tokens);
    prepare_draft_extend_inputs(metadata_template,
                                template_states,
                                current_draft_input,
                                /*force_two_rows=*/true);
    wait_metadata_ready_event(current_draft_input, *compute_stream_);
    clear_ready_events(current_draft_input);

    mtp_async::prepare_next_draft_from_accepted_state(
        current_draft_input,
        input,
        accepted_tokens,
        accepted_embeddings,
        embedding_cache_->embedding_placeholder(),
        target_base_positions,
        target_base_kv_seq_lens,
        /*use_chunked_prefill=*/false,
        /*rebuild_expanded_decode_metadata=*/true,
        block_size());
  } else if (!use_prelaunched_first_draft) {
    // First decode after prefill and batch transitions use the host cache.
    std::vector<EmbeddingCache::DecodeState> last_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    CHECK_EQ(last_states.size(),
             input.input_params.embedding.embedding_ids.size())
        << "decode target state count mismatch";
    // Synthetic graph warmup requests do not have a target decode history.
    // Draft preparation already falls back to the initialized placeholder for
    // invalid states, so reserve strict cache validation for real requests.
    if (!input.input_params.meta.is_graph_warmup) {
      check_mtp_decode_states(last_states,
                              input.input_params.embedding.request_ids,
                              input.token_ids_host,
                              enable_schedule_overlap());
    }
    update_decode_step_input(input, last_states);
    metadata_template = input;
    prepare_draft_extend_inputs(input, last_states, current_draft_input);
  }
  const bool use_continuous_dsa_drafts =
      (use_device_target_context || use_prelaunched_first_draft) &&
      combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  const bool use_continuous_python_drafts =
      (use_device_target_context || use_prelaunched_first_draft) &&
      supports_torch_combined_first_draft(input);
  const bool use_continuous_drafts =
      use_continuous_dsa_drafts || use_continuous_python_drafts;
  // Continuous drafting does not imply an expanded target layout: native
  // GLM/ATB retains generic verify correction.
  const bool use_expanded_verify_metadata =
      use_continuous_drafts && supports_expanded_spec_verify() &&
      input.input_params.multi_block_tables.empty();
  std::vector<ForwardInput> later_draft_inputs;
  torch::Tensor accepted_base_positions;
  torch::Tensor accepted_base_kv_seq_lens;
  if (use_continuous_drafts) {
    later_draft_inputs.resize(num_speculative_tokens);
    const ForwardInput& combined_draft_input =
        use_prelaunched_first_draft ? pending_draft_context_.prepared_input
                                    : current_draft_input;
    const int64_t batch_size = input.input_params.meta.num_sequences;
    CHECK_EQ(combined_draft_input.positions.numel(), batch_size * 2)
        << "combined draft positions must contain [repair,current] rows";
    CHECK_EQ(
        combined_draft_input.input_params.attention.device.kv_seq_lens.numel(),
        batch_size * 2)
        << "combined draft KV lengths must contain [repair,current] rows";
    accepted_base_positions =
        combined_draft_input.positions.view({batch_size, 2}).select(1, 1);
    accepted_base_kv_seq_lens =
        combined_draft_input.input_params.attention.device.kv_seq_lens
            .view({batch_size, 2})
            .select(1, 1);

    for (int32_t draft_idx = 1; draft_idx < num_speculative_tokens;
         ++draft_idx) {
      // Only the fixed B layout is needed on prepare_stream. Reusing offset 0
      // avoids extending an already conservative Host template past the
      // scheduler-allocated block range; device metadata is replaced below.
      prepare_draft_inputs(metadata_template,
                           later_draft_inputs[draft_idx],
                           /*position_offset=*/0);
    }
  }

  const auto materialize_pending_target_host_state =
      [&]() -> const std::vector<int32_t>& {
    flush_pending_target_context();
    std::vector<EmbeddingCache::DecodeState> resolved_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    // The scheduler input contains the conservative overlap placeholder.
    // Force cache correction before comparing it with the accepted target.
    input.token_ids_host = torch::full_like(input.token_ids_host, -1);
    update_decode_step_input(input, resolved_states);
    check_mtp_decode_states(resolved_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            /*allow_overlap_fake_token=*/false);
    metadata_template = input;
    return input.input_params.attention.host.kv_seq_lens;
  };

  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state;
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft && !use_continuous_drafts &&
        prepare_static_mtp_graph_tasks_before_final_draft(input);
    if (reuse_mtp_topk_state) {
      current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
    }
    current_draft_input.input_params.meta.mtp_draft_step = draft_idx;
    std::optional<ForwardOutput> draft_output_opt;
    if (use_prelaunched_first_draft && draft_idx == 0) {
      draft_output_opt = std::move(pending_draft_context_.output);
      draft_prepared[draft_idx] =
          std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_output_opt = run_worker_no_sync_impl(*draft_impl_,
                                                 current_draft_input,
                                                 *compute_stream_,
                                                 *compute_stream_,
                                                 draft_prepared[draft_idx]);
    }

    if ((use_device_target_context || use_prelaunched_first_draft) &&
        !use_continuous_drafts && draft_idx == 0) {
      // The next draft forward is already queued behind target validation.
      // It can start immediately when rejection sampling finishes while the
      // worker materializes the accepted state for later draft/target metadata
      // on CPU.  This synchronization is therefore outside the NPU critical
      // path rather than sitting between target and draft launches.
      materialize_pending_target_host_state();
    }

    if (use_continuous_drafts && is_final_draft) {
      // Queue target metadata before the Host target-context wait. The fixed
      // template can be copied immediately; its real device values are
      // corrected after a prepare-stream wait on the previous target event.
      prepare_validate_inputs(metadata_template,
                              validate_input,
                              /*static_graph_tasks_prepared=*/false,
                              /*record_ready_event=*/false,
                              /*defer_expanded_paging=*/
                              use_expanded_verify_metadata);
      {
        c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
        CHECK(prepare_stream_->wait_event(target_context_ready_event))
            << "failed to wait pending target state on prepare stream";
        if (use_expanded_verify_metadata) {
          mtp_async::prepare_expanded_target_verify_from_accepted_state(
              validate_input,
              accepted_tokens,
              target_base_positions,
              target_base_kv_seq_lens,
              block_size());
        } else {
          mtp_async::prepare_target_verify_from_accepted_state(
              validate_input,
              accepted_tokens,
              target_base_positions,
              target_base_kv_seq_lens,
              block_size());
        }
        validate_input.retained_device_tensors = {
            accepted_tokens, target_base_positions, target_base_kv_seq_lens};
        if (!use_expanded_verify_metadata) {
          record_metadata_ready_event(*prepare_stream_, validate_input);
        }
      }

      // Host cache materialization is still required before staging the next
      // target context, but it no longer blocks target metadata submission.
      const std::vector<int32_t>& accepted_base_kv_seq_lens =
          materialize_pending_target_host_state();
      if (use_expanded_verify_metadata) {
        // Build exact host and paged metadata on the prepare stream, then
        // publish one event covering both device and paging correction.
        c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
        mtp_async::refresh_expanded_target_verify_metadata(
            validate_input, accepted_base_kv_seq_lens, block_size());
        record_metadata_ready_event(*prepare_stream_, validate_input);
      }
    }

    // Overlap next-step input preparation with async draft forward.
    if (is_final_draft) {
      if (!use_continuous_drafts) {
        prepare_validate_inputs(
            metadata_template, validate_input, static_graph_tasks_prepared);
      }
    } else if (use_continuous_drafts) {
      next_step_input = std::move(later_draft_inputs[draft_idx + 1]);
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      wait_metadata_ready_event(next_step_input, *compute_stream_);
      clear_ready_events(next_step_input);
      mtp_async::prepare_later_draft_from_device_base(next_step_input,
                                                      input,
                                                      accepted_base_positions,
                                                      accepted_base_kv_seq_lens,
                                                      draft_idx + 1,
                                                      block_size());
    } else {
      prepare_draft_inputs(metadata_template, next_step_input, draft_idx + 1);
    }

    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.emplace_back(std::move(draft_output_opt.value()));
    const SamplingParameters& draft_sampling_params =
        draft_prepared[draft_idx].sampling_params;
    {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      if (reuse_mtp_topk_state) {
        mtp_topk_state = mtp_async::select_mtp_topk_state_for_next_step(
            draft_outputs.back().mtp_topk_state, draft_sampling_params);
      }
      // Keep draft tokens consistent across the consensus group.
      if (should_broadcast_speculative_token_ids(
              parallel_args_,
              get_optimization_config().enable_spec_token_broadcast,
              draft_sampling_params.sampling_mode)) {
        SampleOutput& draft_sample = draft_outputs.back().sample_output;
        broadcast_speculative_token_ids(draft_sample.next_tokens,
                                        parallel_args_);
      }
      auto& draft_output = draft_outputs.back();
      if (options_.enable_probabilistic_draft() &&
          draft_sampling_params.sampling_mode != SamplingMode::GREEDY) {
        draft_output.sample_output.draft_logits =
            std::move(draft_output.logits);
      }
      draft_vocab_.remap(draft_output.sample_output);
    }
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    check_draft_input_embedding(last_output.embeddings, "decode");
    current_draft_input = std::move(next_step_input);
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options(),
                                *compute_stream_);
    if (last_output.embeddings.defined()) {
      current_draft_input.input_params.embedding.input_embedding =
          last_output.embeddings;
      // input_embedding is produced and consumed on compute_stream_; FIFO
      // ordering replaces the same-stream EventRecord/EventWait pair.
    }
    record_current_metadata_ready_event(current_draft_input, *compute_stream_);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return run_validate(input, draft_outputs, validate_input);
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    Stream& compute_stream) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_draft_tokens = static_cast<int32_t>(draft_outputs.size());
  const int32_t num_val_tokens = num_draft_tokens + 1;
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";

  CHECK_EQ(validate_input.token_ids.numel(),
           static_cast<int64_t>(num_sequences) * num_val_tokens)
      << "MTP verify token count must match the fixed-width batch";

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  validate_input.device_tensors_ready = false;
  auto& fused_draft_tokens =
      validate_input.input_params.graph.spec_verify_draft_token_sources;
  fused_draft_tokens.clear();
#if defined(USE_NPU)
  const bool use_fused_verify_token_update =
      validate_input.input_params.graph.spec_verify_source_addresses_stable &&
      validate_input.input_params.graph.input_tokens_override.defined() &&
      supports_explicit_spec_verify_replay_update() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          num_val_tokens, options_.block_size());
#else
  constexpr bool use_fused_verify_token_update = false;
#endif
  std::vector<torch::Tensor> draft_token_columns;
  draft_token_columns.reserve(draft_outputs.size());
  for (const ForwardOutput& draft_output : draft_outputs) {
    const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
    CHECK(next_tokens.defined() && next_tokens.numel() == num_sequences)
        << "expected one draft token per sequence";
    if (use_fused_verify_token_update) {
      CHECK(next_tokens.scalar_type() == torch::kInt64 &&
            next_tokens.device() == validate_input.token_ids.device() &&
            next_tokens.is_contiguous())
          << "fused speculative verify requires contiguous int64 draft tokens";
      fused_draft_tokens.emplace_back(next_tokens.flatten());
    }
    if (!use_fused_verify_token_update || !input.json_object_states.empty()) {
      draft_token_columns.emplace_back(
          safe_to(next_tokens.flatten(), token_options, /*non_blocking=*/true));
    }
  }
  if (!draft_token_columns.empty()) {
    const torch::Tensor draft_token_ids =
        torch::stack(draft_token_columns, /*dim=*/1);
    if (!input.json_object_states.empty()) {
      JsonObjectDecodeAdapter::prepare_block_verify_mask(
          input,
          safe_to(draft_token_ids, /*options=*/torch::kCPU),
          validate_input,
          device_);
    }
    if (!use_fused_verify_token_update) {
      auto verify_tokens =
          validate_input.token_ids.view({num_sequences, num_val_tokens});
      verify_tokens.slice(/*dim=*/1, /*start=*/1, /*end=*/num_val_tokens)
          .copy_(draft_token_ids, /*non_blocking=*/true);
    }
  }
  validate_input.device_tensors_ready = true;
  if (!use_fused_verify_token_update || !input.json_object_states.empty()) {
    record_metadata_ready_event(compute_stream, validate_input);
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  Timer timer;
  ForwardInput target_prepared;
  fill_validate_input_from_draft_outputs(
      input, draft_outputs, validate_input, *compute_stream_);
  ForwardOutput target_output = run_worker_no_sync_impl(*impl_,
                                                        validate_input,
                                                        *compute_stream_,
                                                        *compute_stream_,
                                                        target_prepared)
                                    .value();
  for (const ForwardOutput& draft_output : draft_outputs) {
    append_request_failures(target_output, draft_output);
  }
  const double target_latency_ms = timer.elapsed_milliseconds();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_latency_ms / 1000.0);

  const bool prelaunch_next_first_draft = can_prelaunch_next_first_draft(input);
  ForwardInput next_first_draft_input;
  if (prelaunch_next_first_draft) {
    // This input is independent of the accepted token.  Prepare it on the
    // auxiliary stream while target verification is still executing; the
    // compute stream consumes it through a device-side event after rejection
    // sampling, with no host synchronization.
    prepare_next_first_draft_template(input, next_first_draft_input);
  }

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    val_output = validate(input.sampling_params,
                          validate_input.sampling_params,
                          draft_outputs,
                          target_output,
                          validate_input.json_object_invalid_draft);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  const int64_t batch_size = input.input_params.meta.num_sequences;
  const int64_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const torch::Tensor& validate_positions = validate_input.positions;
  CHECK_EQ(validate_positions.numel(), batch_size * num_val_tokens)
      << "validate positions must contain one row per speculative token";
  const torch::Tensor& validate_kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens;
  CHECK_GE(validate_kv_seq_lens.numel(), batch_size)
      << "validate KV lengths must be sequence-scoped";
  torch::Tensor accepted_tokens_host =
      acquire_accepted_tokens_host_buffer(val_output.next_tokens);
  torch::Tensor accepted_tokens_cpu_result = accepted_tokens_host;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    if (should_broadcast_speculative_token_ids(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.sampling_mode)) {
      broadcast_speculative_token_ids(val_output.next_tokens, parallel_args_);
    }

    base_positions = validate_positions.view({batch_size, num_val_tokens})
                         .select(/*dim=*/1, /*index=*/0)
                         .contiguous();
    base_kv_seq_lens = mtp_async::extract_target_base_kv_seq_lens(
        validate_kv_seq_lens,
        batch_size,
        num_val_tokens,
        use_chunked_prefill_spec_verify_path());

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers consensus, base-state derivation, and the D2H copy.
    target_context_ready_event = compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  std::vector<size_t> failed_sequence_rows;
  if (!input.json_object_states.empty()) {
    CHECK_EQ(accepted_tokens_host.dim(), 2)
        << "MTP JSON accepted output must be [sequence, token]";
    CHECK_EQ(input.sample_sequence_ids.size(), input.json_object_states.size())
        << "MTP JSON sequence ids must match grammar state rows";
    CHECK_EQ(input.input_params.embedding.embedding_ids.size(),
             input.json_object_states.size())
        << "MTP JSON embedding ids must match grammar state rows";
    CHECK(target_context_ready_event == nullptr ||
          target_context_ready_event->synchronize())
        << "failed to wait for accepted MTP tokens before JSON replay";

    const torch::Tensor accepted_tokens =
        accepted_tokens_host.to(torch::kInt64).contiguous();
    const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
        detail::verify_json_accepted_rows(
            input.json_object_states,
            accepted_tokens.const_data_ptr<int64_t>(),
            static_cast<size_t>(accepted_tokens.size(0)),
            static_cast<size_t>(accepted_tokens.size(1)));
    std::unordered_set<std::string> failed_sequence_ids;
    failed_sequence_ids.reserve(input.json_object_errors.size() +
                                target_output.json_object_errors.size() +
                                mismatches.size());
    for (const JsonObjectOutputError& error : input.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON input error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const JsonObjectOutputError& error :
         target_output.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON output error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const detail::JsonAcceptedTokenMismatch& mismatch : mismatches) {
      const std::string& sample_sequence_id =
          input.sample_sequence_ids[mismatch.sequence_index];
      CHECK(!sample_sequence_id.empty())
          << "MTP JSON mismatch requires a sampled sequence id";
      LOG(ERROR) << "MTP JSON accepted output replay mismatch: sequence_id="
                 << sample_sequence_id
                 << ", token_offset=" << mismatch.token_offset
                 << ", token_id=" << mismatch.token_id
                 << ", committed_tokens=" << mismatch.committed_tokens
                 << ", state_fingerprint=" << mismatch.state_fingerprint;
      if (failed_sequence_ids.emplace(sample_sequence_id).second) {
        target_output.json_object_errors.push_back(
            {sample_sequence_id,
             "accepted MTP output violates json_object grammar, token_id=" +
                 std::to_string(mismatch.token_id)});
      }
    }
    failed_sequence_rows.reserve(failed_sequence_ids.size());
    for (size_t sequence_index = 0;
         sequence_index < input.sample_sequence_ids.size();
         ++sequence_index) {
      if (failed_sequence_ids.contains(
              input.sample_sequence_ids[sequence_index])) {
        failed_sequence_rows.emplace_back(sequence_index);
      }
    }
    CHECK_EQ(failed_sequence_rows.size(), failed_sequence_ids.size())
        << "MTP JSON errors must reference sampled rows in the current batch";
  }
  const bool has_failed_sequence_rows = !failed_sequence_rows.empty();
  stage_target_context_write(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             target_context_ready_event,
                             std::move(accepted_tokens_host),
                             std::move(failed_sequence_rows));
  if (prelaunch_next_first_draft && !has_failed_sequence_rows) {
    // Submit the next iteration's first draft before returning to the
    // scheduler.  This is the actual asynchronous boundary: scheduler/host
    // accepted-state work can no longer sit between target validation and the
    // next draft launch.
    enqueue_next_first_draft(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             std::move(next_first_draft_input));
  }
  target_output.ready_event = target_context_ready_event;

  // Target validation consumes all draft outputs on the same compute stream;
  // keep their prepared inputs alive on target_output until the target-context
  // event completes. Copy (not move): draft_outputs live beyond this loop.
  for (const ForwardOutput& draft_output : draft_outputs) {
    copy_retained_inputs(target_output, draft_output);
  }

  if (!enable_schedule_overlap()) {
    flush_pending_target_context();
    target_output.retained_inputs.clear();
    target_output.ready_event.reset();
    val_output.next_tokens = std::move(accepted_tokens_cpu_result);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !::xllm::EPLBConfig::get_instance().enable_eplb()) {
    if (!target_output.failed_request_ids.empty()) {
      return make_failed_output(std::move(target_output.failed_request_ids),
                                std::move(target_output.unsafe_request_ids));
    }
    return std::nullopt;
  }
  clear_all_output_embeddings(target_output);
  finalize_verify_output(target_output, std::move(val_output), input);
  return target_output;
}

void MTPWorkerImpl::stage_target_context_write(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    torch::Tensor base_positions,
    torch::Tensor base_kv_seq_lens,
    StreamEventPtr ready_event,
    torch::Tensor accepted_tokens_host,
    std::vector<size_t> failed_rows) {
  CHECK(!pending_target_context_.accepted_tokens.defined())
      << "previous MTP target context must be flushed before staging another";
  pending_target_context_.embedding_ids =
      input.input_params.embedding.embedding_ids;
  pending_target_context_.request_ids =
      input.input_params.embedding.request_ids;
  pending_target_context_.accepted_tokens = validate_output.next_tokens;
  pending_target_context_.accepted_tokens_host =
      std::move(accepted_tokens_host);
  pending_target_context_.accepted_embeddings = validate_output.embeddings;
  pending_target_context_.base_positions = std::move(base_positions);
  pending_target_context_.base_kv_seq_lens = std::move(base_kv_seq_lens);
  pending_target_context_.json_constrained_rows.clear();
  pending_target_context_.json_constrained_rows.reserve(
      input.json_object_states.size());
  for (const ConstraintState& state : input.json_object_states) {
    pending_target_context_.json_constrained_rows.emplace_back(
        state.initialized() ? 1U : 0U);
  }
  pending_target_context_.failed_rows = std::move(failed_rows);
  pending_target_context_.ready_event = std::move(ready_event);
}

torch::Tensor MTPWorkerImpl::acquire_accepted_tokens_host_buffer(
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";
  CHECK(!pending_target_context_.accepted_tokens.defined())
      << "accepted-token host buffer is still in use";

  const int64_t required_capacity = accepted_tokens.numel();
  const int64_t configured_capacity =
      static_cast<int64_t>(options_.max_seqs_per_batch()) *
      (static_cast<int64_t>(options_.num_speculative_tokens()) + 1);
  const bool needs_allocation =
      !accepted_tokens_host_buffer_.defined() ||
      accepted_tokens_host_buffer_.scalar_type() !=
          accepted_tokens.scalar_type() ||
      accepted_tokens_host_buffer_.numel() < required_capacity;
  if (needs_allocation) {
    const int64_t capacity = std::max(required_capacity, configured_capacity);
    accepted_tokens_host_buffer_ = torch::empty(
        {capacity},
        accepted_tokens.options().device(torch::kCPU).pinned_memory(true));
  }

  return accepted_tokens_host_buffer_
      .narrow(/*dim=*/0, /*start=*/0, required_capacity)
      .view(accepted_tokens.sizes());
}

bool MTPWorkerImpl::pending_target_context_matches(
    const ForwardInput& input) const {
  return pending_target_context_.failed_rows.empty() &&
         pending_target_context_.accepted_tokens.defined() &&
         pending_target_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_target_context_.request_ids ==
             input.input_params.embedding.request_ids;
}

bool MTPWorkerImpl::device_target_context_ready_for_batch(
    const ForwardInput& input) const {
  return device_context_ready_embedding_ids_ ==
             input.input_params.embedding.embedding_ids &&
         device_context_ready_request_ids_ ==
             input.input_params.embedding.request_ids;
}

void MTPWorkerImpl::flush_pending_target_context() {
  if (!pending_target_context_.accepted_tokens.defined()) {
    return;
  }
  CHECK(pending_target_context_.ready_event == nullptr ||
        pending_target_context_.ready_event->synchronize())
      << "failed to wait for pending MTP target context";
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_validation_tokens = num_speculative_tokens + 1;
  const torch::Tensor accepted_tokens =
      pending_target_context_.accepted_tokens_host.contiguous();
  CHECK_EQ(accepted_tokens.numel() % num_validation_tokens, 0)
      << "MTP validation output width mismatch";
  const torch::Tensor output_tokens =
      accepted_tokens.view({-1, num_validation_tokens});
  if (!pending_target_context_.json_constrained_rows.empty()) {
    CHECK_EQ(pending_target_context_.json_constrained_rows.size(),
             static_cast<size_t>(output_tokens.size(0)))
        << "MTP JSON row metadata mismatch";
  }
  int64_t constrained_accepted = 0;
  int64_t plain_accepted = 0;
  int64_t constrained_draft = 0;
  int64_t plain_draft = 0;
  for (int64_t sequence_idx = 0; sequence_idx < output_tokens.size(0);
       ++sequence_idx) {
    const bool constrained =
        !pending_target_context_.json_constrained_rows.empty() &&
        pending_target_context_
                .json_constrained_rows[static_cast<size_t>(sequence_idx)] != 0U;
    int64_t rejected = 0;
    for (int32_t token_idx = 0; token_idx < num_validation_tokens;
         ++token_idx) {
      if (output_tokens.index({sequence_idx, token_idx}).item<int64_t>() < 0) {
        ++rejected;
      }
    }
    const int64_t accepted =
        num_speculative_tokens -
        std::min<int64_t>(rejected, num_speculative_tokens);
    if (constrained) {
      constrained_accepted += accepted;
      constrained_draft += num_speculative_tokens;
    } else {
      plain_accepted += accepted;
      plain_draft += num_speculative_tokens;
    }
  }
  COUNTER_ADD(speculative_num_accepted_tokens_constrained_total,
              constrained_accepted);
  COUNTER_ADD(speculative_num_accepted_tokens_plain_total, plain_accepted);
  COUNTER_ADD(speculative_num_draft_tokens_constrained_total,
              constrained_draft);
  COUNTER_ADD(speculative_num_draft_tokens_plain_total, plain_draft);
  if (pending_target_context_.failed_rows.empty()) {
    embedding_cache_->write_target_context(
        pending_target_context_.embedding_ids,
        pending_target_context_.request_ids,
        pending_target_context_.accepted_tokens_host,
        pending_target_context_.accepted_embeddings,
        options_.num_speculative_tokens());
  } else {
    CHECK(pending_target_context_.request_ids.empty() ||
          pending_target_context_.request_ids.size() ==
              pending_target_context_.embedding_ids.size())
        << "target context request ids must match embedding ids";
    std::unordered_set<size_t> failed_row_set(
        pending_target_context_.failed_rows.begin(),
        pending_target_context_.failed_rows.end());
    CHECK_EQ(failed_row_set.size(), pending_target_context_.failed_rows.size())
        << "target context failed rows must be unique";

    std::vector<int32_t> failed_embedding_ids;
    failed_embedding_ids.reserve(pending_target_context_.failed_rows.size());
    for (const size_t failed_row : pending_target_context_.failed_rows) {
      CHECK_LT(failed_row, pending_target_context_.embedding_ids.size())
          << "target context failed row exceeds embedding ids";
      failed_embedding_ids.emplace_back(
          pending_target_context_.embedding_ids[failed_row]);
    }
    embedding_cache_->clear(failed_embedding_ids);

    for (size_t sequence_index = 0;
         sequence_index < pending_target_context_.embedding_ids.size();
         ++sequence_index) {
      if (failed_row_set.contains(sequence_index)) {
        continue;
      }
      const std::vector<int32_t> row_embedding_ids = {
          pending_target_context_.embedding_ids[sequence_index]};
      const std::vector<std::string> row_request_ids =
          pending_target_context_.request_ids.empty()
              ? std::vector<std::string>()
              : std::vector<std::string>{
                    pending_target_context_.request_ids[sequence_index]};
      embedding_cache_->write_target_context(
          row_embedding_ids,
          row_request_ids,
          pending_target_context_.accepted_tokens_host.narrow(
              /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
          pending_target_context_.accepted_embeddings.narrow(
              /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
          options_.num_speculative_tokens());
    }
  }
  pending_target_context_ = PendingTargetContext();
}

bool MTPWorkerImpl::npu_combined_draft_ready() const {
#if defined(USE_NPU)
  if (draft_impl_ == nullptr ||
      draft_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }

  return device_.unwrap().is_privateuseone();
#else
  return false;
#endif
}

bool MTPWorkerImpl::supports_combined_first_draft_execution() const {
#if defined(USE_NPU)
  if (!npu_combined_draft_ready()) {
    return false;
  }
  const std::string& npu_backend =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend();
  return mtp_async::supports_combined_draft_configuration(
      combined_draft_execution_path_, npu_backend, parallel_args_.dp_size());
#else
  return false;
#endif
}

bool MTPWorkerImpl::supports_torch_combined_first_draft(
    const ForwardInput& input) const {
#if defined(USE_NPU)
  if (!npu_combined_draft_ready()) {
    return false;
  }

  const std::string& target_model_type = context_.get_model_args().model_type();
  const std::string& draft_model_type =
      draft_impl_->context_.get_model_args().model_type();
  const std::string& npu_backend =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend();
  const bool supported =
      mtp_async::supports_torch_combined_first_draft_configuration(
          target_model_type,
          draft_model_type,
          npu_backend,
          parallel_args_.dp_size(),
          ModelConfig::is_python_model_impl(context_.get_model_impl()),
          ModelConfig::is_python_model_impl(
              draft_impl_->context_.get_model_impl()),
          !input.input_params.multi_block_tables.empty(),
          context_.get_model_args().index_topk(),
          draft_impl_->context_.get_model_args().index_topk());
  LOG_FIRST_N(INFO, 1)
      << "MTP TORCH combined first draft configuration (admission is checked "
         "per input)"
      << ", target=" << target_model_type << ", draft=" << draft_model_type
      << ", backend=" << npu_backend << ", dp_size=" << parallel_args_.dp_size()
      << ", python_target="
      << ModelConfig::is_python_model_impl(context_.get_model_impl())
      << ", python_draft="
      << ModelConfig::is_python_model_impl(
             draft_impl_->context_.get_model_impl())
      << ", index_topk=" << context_.get_model_args().index_topk() << "/"
      << draft_impl_->context_.get_model_args().index_topk();
  return supported;
#else
  static_cast<void>(input);
  return false;
#endif
}

bool MTPWorkerImpl::can_use_combined_first_draft(
    const ForwardInput& input) const {
  if (!enable_schedule_overlap()) {
    return false;
  }
  // DSV32 needs positive topk for expanded verify. Zero topk rejects both
  // prelaunch and continuous drafting; its native combined path is unsupported,
  // so it stays on regular in-loop drafting, not generic verify correction.
  return supports_torch_combined_first_draft(input) ||
         supports_combined_first_draft_execution();
}

bool MTPWorkerImpl::can_prelaunch_next_first_draft(
    const ForwardInput& input) const {
  if (!can_use_combined_first_draft(input)) {
    return false;
  }
  const bool requires_dp_symmetric_prelaunch =
      parallel_args_.dp_size() > 1 &&
      combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  if (requires_dp_symmetric_prelaunch) {
    return has_active_dp_tokens(input);
  }
  return device_target_context_ready_for_batch(input);
}

void MTPWorkerImpl::prepare_next_first_draft_template(
    const ForwardInput& input,
    ForwardInput& combined_input) {
  CHECK(embedding_cache_ != nullptr);

  ForwardInput metadata_template = input;
  std::vector<EmbeddingCache::DecodeState> template_states =
      mtp_async::advance_first_draft_template(
          metadata_template,
          embedding_cache_->embedding_placeholder(),
          options_.num_speculative_tokens());

  prepare_draft_extend_inputs(metadata_template,
                              template_states,
                              combined_input,
                              /*force_two_rows=*/true,
                              /*wait_for_compute_stream=*/false);
  combined_input.skip_sampling_for_logits_only = false;
}

void MTPWorkerImpl::enqueue_next_first_draft(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    ForwardInput combined_input) {
  CHECK(validate_output.next_tokens.defined());
  CHECK(validate_output.embeddings.defined());
  CHECK(embedding_cache_ != nullptr);

  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  wait_metadata_ready_event(combined_input, *compute_stream_);
  clear_ready_events(combined_input);

  // Interleave [repair, current] rows in one decode batch. Every transformer
  // layer projects both rows, writes both KV rows, and only then launches
  // PagedAttention. Same-stream ordering therefore makes repair KV visible to
  // the current row without a host wait or a separate repair forward.
  mtp_async::prepare_next_draft_from_accepted_state(
      combined_input,
      input,
      validate_output.next_tokens,
      validate_output.embeddings,
      embedding_cache_->embedding_placeholder(),
      base_positions,
      base_kv_seq_lens,
      /*use_chunked_prefill=*/false,
      /*rebuild_expanded_decode_metadata=*/false,
      block_size());

  submit_pending_first_draft(input, std::move(combined_input));
}

void MTPWorkerImpl::submit_pending_first_draft(
    const ForwardInput& batch_identity_input,
    ForwardInput draft_input) {
  CHECK(!pending_draft_context_.output.has_value())
      << "MTP first-draft prelaunch was not consumed";
  pending_draft_context_.embedding_ids =
      batch_identity_input.input_params.embedding.embedding_ids;
  pending_draft_context_.request_ids =
      batch_identity_input.input_params.embedding.request_ids;
  pending_draft_context_.dp_global_token_nums =
      batch_identity_input.input_params.parallel.dp_global_token_nums;
  pending_draft_context_.raw_dp_global_token_nums =
      batch_identity_input.input_params.parallel.raw_dp_global_token_nums;
  pending_draft_context_.dp_global_batch_generations =
      batch_identity_input.input_params.parallel.dp_global_batch_generations;
  draft_input.input_params.meta.mtp_draft_step = 0;
  pending_draft_context_.output =
      run_worker_no_sync_impl(*draft_impl_,
                              draft_input,
                              *compute_stream_,
                              *compute_stream_,
                              pending_draft_context_.prepared_input);
  CHECK(pending_draft_context_.output.has_value())
      << "failed to prelaunch next MTP first draft";
  LOG_FIRST_N(INFO, 1) << "MTP first-draft prelaunch submitted, target="
                       << context_.get_model_args().model_type() << ", draft="
                       << draft_impl_->context_.get_model_args().model_type();
}

bool MTPWorkerImpl::pending_draft_context_matches(
    const ForwardInput& input) const {
  return pending_draft_context_.output.has_value() &&
         pending_draft_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_draft_context_.request_ids ==
             input.input_params.embedding.request_ids &&
         pending_draft_context_.dp_global_token_nums ==
             input.input_params.parallel.dp_global_token_nums &&
         pending_draft_context_.raw_dp_global_token_nums ==
             input.input_params.parallel.raw_dp_global_token_nums &&
         pending_draft_context_.dp_global_batch_generations ==
             input.input_params.parallel.dp_global_batch_generations;
}

void MTPWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();
  const bool positions_decoupled = positions_are_decoupled_from_kv_length();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};
  std::vector<int32_t> positions_vec;
  positions_vec.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len = calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    int32_t expected_kv_len =
        positions_decoupled ? current_kv_len : current_position + 1;
    if (use_chunked_prefill_spec_verify_path()) {
      const torch::Tensor& block_tables =
          input.input_params.attention.host.block_tables;
      if (block_tables.defined() && block_tables.dim() == 2 &&
          seq_id < block_tables.size(0)) {
        const int32_t allocated_kv_len =
            static_cast<int32_t>(block_tables.size(1) * block_size());
        const int32_t validate_width = options_.num_speculative_tokens() + 1;
        const int32_t max_valid_position = allocated_kv_len - validate_width;
        if (current_position > max_valid_position) {
          CHECK_GT(allocated_kv_len, 0)
              << "decode context has empty block table, seq_id=" << seq_id;
          CHECK_GE(max_valid_position, 0)
              << "decode context block table is too small for validation, "
              << "seq_id=" << seq_id
              << ", allocated_kv_len=" << allocated_kv_len
              << ", validate_width=" << validate_width;
          CHECK_LE(current_position - max_valid_position,
                   options_.num_speculative_tokens() + 1)
              << "decode context position exceeds allocated blocks, seq_id="
              << seq_id << ", current_position=" << current_position
              << ", current_kv_len=" << current_kv_len
              << ", allocated_kv_len=" << allocated_kv_len;
          current_position = max_valid_position;
          expected_kv_len = current_position + 1;
          current_kv_len = std::min(current_kv_len, expected_kv_len);
        }
      }
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len < expected_kv_len) {
      // Qwen3.5/MiMo MTP can receive a scheduler KV length that has not yet
      // caught up with the speculative placeholder resolved into
      // current_position. Normalize only the lag explainable by the current
      // speculative step.
      CHECK_LE(expected_kv_len - current_kv_len,
               options_.num_speculative_tokens() + 1)
          << "decode context kv_len lag is too large, seq_id=" << seq_id
          << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len;
      current_kv_len = expected_kv_len;
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len > expected_kv_len) {
      // The first decode step can carry the prompt KV length while the decode
      // position is still initialized to zero. Align the position to the KV
      // context before building the MTP draft input.
      current_position = current_kv_len - 1;
      expected_kv_len = current_kv_len;
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back((use_cache_correction || use_fake_context)
                                   ? state.token_id
                                   : input_token_id);
    positions_vec.emplace_back(current_position);
    append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = make_cpu_int_tensor(token_ids_vec);
  input.positions_host = make_cpu_int_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  // The corrected host fields supersede the serialized scheduler snapshot.
  input.input_host_buffer_has_layout = false;
  input.device_tensors_ready = false;
}

void MTPWorkerImpl::prepare_empty_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  CHECK_EQ(input.input_params.meta.num_sequences, 0)
      << "empty target validation requires an empty rank";
  validate_input = input;

  ModelInputParams& input_params = validate_input.input_params;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  scale_speculative_parallel_token_counts(input_params, num_val_tokens);

  if (!use_chunked_prefill_spec_verify_path()) {
    return;
  }

  input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  input_params.is_spec_verify = true;

#if defined(USE_NPU)
  if (parallel_args_.dp_size() <= 1 ||
      !is_qwen3_5_target_model_type(context_.get_model_args().model_type()) ||
      !supports_explicit_spec_verify_replay_update()) {
    return;
  }

  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  clear_ready_events(validate_input);
  validate_input.input_host_buffer_has_layout = false;
  validate_input.device_tensors_ready = false;

  CHECK(input_params.attention.host.q_seq_lens.empty())
      << "empty Qwen3.5 target validation must not contain real query lengths";
  CHECK(input_params.attention.host.kv_seq_lens.empty())
      << "empty Qwen3.5 target validation must not contain real KV lengths";
  input_params.attention.host.q_cu_seq_lens.clear();
  input_params.attention.host.new_cache_slots.clear();
  input_params.attention.host.kv_cache_tokens_nums.clear();
  input_params.attention.host.graph_q_seq_lens_data = nullptr;
  input_params.attention.host.graph_kv_seq_lens_data = nullptr;
  input_params.attention.device = AttentionDeviceInput{};
  input_params.embedding.input_embedding = torch::Tensor();
  input_params.clear_linear_attention_state();
  input_params.meta.actual_num_sequences = 0;
  input_params.meta.q_max_seq_len = num_val_tokens;
  input_params.meta.kv_max_seq_len = 1;
  input_params.parallel.query_start_loc.clear();
  input_params.num_accepted_tokens_host.assign(1, 0);
  input_params.num_accepted_tokens = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kInt).device(device_));
  input_params.graph.attn_mask = torch::Tensor();
  input_params.graph.tiling_data = torch::Tensor();
  input_params.graph.input_tokens_override = torch::Tensor();
  input_params.graph.spec_verify_draft_token_sources.clear();
  input_params.graph.spec_verify_source_addresses_stable = false;
  input_params.graph.spec_verify_static_graph_tasks_prepared = false;

  CHECK_GT(options_.block_size(), 0)
      << "empty expanded metadata requires a positive cache block size";
  int64_t block_table_width = 0;
  if (impl_ != nullptr) {
    const int64_t max_position_embeddings =
        impl_->context_.get_model_args().max_position_embeddings();
    if (max_position_embeddings > 0) {
      block_table_width = mtp_async::speculative_verify_block_table_capacity(
          max_position_embeddings, options_.block_size());
    }
  }
  const torch::Tensor& host_block_tables =
      input.input_params.attention.host.block_tables;
  if (block_table_width <= 0 && host_block_tables.defined() &&
      host_block_tables.dim() == 2 && host_block_tables.size(1) > 0) {
    block_table_width = host_block_tables.size(1);
  }
  const torch::Tensor& device_block_tables =
      input.input_params.attention.device.block_tables;
  if (block_table_width <= 0 && device_block_tables.defined() &&
      device_block_tables.dim() == 2 && device_block_tables.size(1) > 0) {
    block_table_width = device_block_tables.size(1);
  }
  CHECK_GT(block_table_width, 0)
      << "empty expanded metadata requires block-table width from the model "
         "position capacity or input block tables";
  clear_expanded_spec_verify_graph_input(input_params);
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt).device(device_);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      input_params,
      torch::ones({1}, int_options),
      torch::zeros({1, block_table_width}, int_options),
      std::vector<int32_t>{1},
      options_.block_size());
#endif
}

void MTPWorkerImpl::prepare_validate_inputs(const ForwardInput& input,
                                            ForwardInput& validate_input,
                                            bool static_graph_tasks_prepared,
                                            bool record_ready_event,
                                            bool defer_expanded_paging) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  validate_input.skip_sampling_for_logits_only = true;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  const auto token_options = validate_input.token_ids.options();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      should_use_explicit_spec_verify_replay_update(input);
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  const auto decode_context = make_decode_batch_context(input);
  if (!positions_are_decoupled_from_kv_length()) {
    for (int32_t req_id = 0; req_id < num_sequences; ++req_id) {
      const int32_t kv_len =
          calc_kv_len(decode_context.kv_seq_lens, req_id, /*offset=*/0);
      CHECK_EQ(decode_context.positions[req_id] + 1, kv_len)
          << "validate position/kv_len mismatch, req_id=" << req_id;
    }
  }
  DecodeExpansionOptions decode_options;
  decode_options.num_speculative_tokens = options_.num_speculative_tokens();
  decode_options.block_size = block_size();
  decode_options.per_request = use_chunked_prefill_spec_verify_path();
  auto decode_buffers = expand_decode_inputs(decode_context, decode_options);
  if (impl_->has_request_state_cache()) {
    set_kpool_verify_metadata(
        input_params, std::vector<int32_t>(num_sequences, num_val_tokens));
  }
  build_decode_input(
      validate_input,
      decode_buffers,
      decode_options,
      /*materialize_tokens=*/!use_explicit_spec_verify_replay_update);

  if (!input.json_object_states.empty()) {
    // Validation installs exact per-position JSON masks after the draft
    // tokens are known. Do not repeat the inherited one-row mask here.
    validate_input.sampling_params.filter_mask = torch::Tensor();
    validate_input.sampling_params.filter_bitmask = torch::Tensor();
  }
  auto& validate_sampling_params = validate_input.sampling_params;
#if defined(USE_NPU)
  // The generic builder uses repeat_interleave on device tensors.  For a
  // single greedy sequence that work only expands [0] and [false] into fixed
  // validation controls, yet it lands on the final-draft -> target dependency
  // chain.  Reuse stable controls after warmup and retain the generic builder
  // for sampling, penalties, multi-sequence batches and other backends.
  const bool use_stable_greedy_validate_sampling =
      use_explicit_spec_verify_replay_update &&
      validate_sampling_params.sampling_mode == SamplingMode::GREEDY &&
      validate_sampling_params.selected_token_idxes.defined() &&
      validate_sampling_params.selected_token_idxes.numel() == 1 &&
      validate_sampling_params.sample_idxes.defined() &&
      validate_sampling_params.sample_idxes.numel() == 1 &&
      validate_sampling_params.do_sample.defined() &&
      validate_sampling_params.do_sample.numel() == 1 &&
      !validate_sampling_params.frequency_penalties.defined() &&
      !validate_sampling_params.presence_penalties.defined() &&
      !validate_sampling_params.repetition_penalties.defined() &&
      !validate_sampling_params.temperatures.defined() &&
      !validate_sampling_params.top_p.defined() &&
      !validate_sampling_params.top_k.defined() &&
      !validate_sampling_params.unique_token_ids.defined() &&
      !validate_sampling_params.unique_token_counts.defined() &&
      !validate_sampling_params.unique_token_ids_lens.defined();
  if (use_stable_greedy_validate_sampling) {
    if (!mtp_validate_greedy_indices_.defined() ||
        mtp_validate_greedy_indices_.numel() != total_num_val_tokens) {
      mtp_validate_greedy_indices_ = torch::arange(
          total_num_val_tokens,
          torch::TensorOptions().dtype(torch::kInt).device(device_));
      mtp_validate_greedy_do_sample_ = torch::zeros(
          {total_num_val_tokens},
          torch::TensorOptions().dtype(torch::kBool).device(device_));
    }
    validate_sampling_params.selected_token_idxes =
        mtp_validate_greedy_indices_;
    validate_sampling_params.sample_idxes = mtp_validate_greedy_indices_;
    validate_sampling_params.do_sample = mtp_validate_greedy_do_sample_;

    validate_sampling_params.sampling_mode = SamplingMode::GREEDY;
  } else {
    spec_verify::expand_verify_sampling_params(validate_sampling_params,
                                               num_val_tokens,
                                               total_num_val_tokens,
                                               device_);
  }
#else
  spec_verify::expand_verify_sampling_params(
      validate_sampling_params, num_val_tokens, total_num_val_tokens, device_);
#endif

  scale_speculative_parallel_token_counts(input_params, num_val_tokens);

  std::vector<int32_t> accepted_prefix_lengths;
  if (use_chunked_prefill_spec_verify_path()) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens_vec;
      q_cu_seq_lens_vec.reserve(input_params.meta.num_sequences + 1);
      q_cu_seq_lens_vec.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
    }
  }
  if (use_chunked_prefill_spec_verify_path() ||
      uses_speculative_linear_state_checkpoints()) {
    accepted_prefix_lengths.assign(num_sequences, 1);
    if (embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths = embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          options_.num_speculative_tokens() + 1);
    }
    // num_accepted_tokens selects the previous step's conv/SSM checkpoint
    // (checkpoint index = nat - 1) and stays equal to the accepted count.
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    if (!use_explicit_spec_verify_replay_update) {
      input_params.num_accepted_tokens =
          torch::tensor(accepted_prefix_lengths, token_options);
    }
  }

#if defined(USE_NPU)
  if (use_explicit_spec_verify_replay_update) {
    build_expanded_spec_verify_graph_host_input(input_params);

    auto& attention = input_params.attention;
    ensure_spec_verify_control_block_table(
        input_params,
        num_sequences,
        mtp_async::speculative_verify_block_table_capacity(
            impl_->context_.get_model_args().max_position_embeddings(),
            options_.block_size()));
    CHECK(attention.host.block_tables.defined());
    CHECK_EQ(attention.host.block_tables.dim(), 2);
    CHECK_EQ(attention.host.block_tables.size(0), num_sequences);
    CHECK_EQ(attention.host.block_tables.scalar_type(), torch::kInt32);
    const int64_t active_block_table_width =
        attention.host.block_tables.size(1);
    const int64_t verify_block_table_width =
        spec_verify_block_table_width(attention.host.block_tables);
    if (active_block_table_width != verify_block_table_width) {
      torch::Tensor padded_block_tables = torch::zeros(
          {num_sequences, verify_block_table_width},
          attention.host.block_tables.options().device(torch::kCPU));
      padded_block_tables.narrow(1, 0, active_block_table_width)
          .copy_(attention.host.block_tables);
      attention.host.block_tables = std::move(padded_block_tables);
    }

    const bool initialize_stable_buffer =
        !spec_verify_attention_host_buffer_.defined();
    attention.attention_host_buffer = spec_verify_attention_host_buffer_;
    attention.attention_device_buffer = spec_verify_attention_device_buffer_;
    attention.attention_buffer_bytes = 0;
    attention.attention_buffer_capacity =
        spec_verify_attention_buffer_capacity_;
    attention.attention_buffer_owner = spec_verify_attention_buffer_owner_;

    const int64_t expanded_block_rows = static_cast<int64_t>(
        input_params.graph.expanded_kv_seq_lens_vec.size());
    CHECK_EQ(expanded_block_rows, total_num_val_tokens);
    const torch::Tensor dense_block_source =
        attention.host.block_tables.contiguous();
    std::vector<int32_t> expanded_block_tables_dense(
        static_cast<size_t>(expanded_block_rows * verify_block_table_width));
    const int32_t* block_row = dense_block_source.data_ptr<int32_t>();
    for (int64_t row = 0; row < expanded_block_rows; ++row) {
      const int64_t sequence_index = row / num_val_tokens;
      std::memcpy(
          expanded_block_tables_dense.data() + row * verify_block_table_width,
          block_row + sequence_index * verify_block_table_width,
          static_cast<size_t>(verify_block_table_width) * sizeof(int32_t));
    }
    torch::Tensor expanded_block_tables_flat;
    std::vector<AttentionInput::PackedIntInput> extra_int_inputs;
    extra_int_inputs.reserve(/*new_cap=*/7);
    extra_int_inputs.push_back({&decode_buffers.out_token_ids,
                                &validate_input.token_ids_host,
                                &validate_input.token_ids});
    extra_int_inputs.push_back({&decode_buffers.out_positions,
                                &validate_input.positions_host,
                                &validate_input.positions});
    if (!input_params.embedding.linear_state_ids.empty()) {
      extra_int_inputs.push_back(
          {&input_params.embedding.linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
      if (input_params.embedding.linear_state_read_ids.empty()) {
        input_params.embedding.linear_state_read_ids =
            input_params.embedding.linear_state_ids;
      }
      extra_int_inputs.push_back(
          {&input_params.embedding.linear_state_read_ids,
           nullptr,
           &input_params.embedding.linear_state_read_indices});
    }
    if (!input_params.num_accepted_tokens_host.empty()) {
      extra_int_inputs.push_back({&accepted_prefix_lengths,
                                  nullptr,
                                  &input_params.num_accepted_tokens});
    }
    if (!input_params.graph.expanded_kv_seq_lens_vec.empty()) {
      extra_int_inputs.push_back({&input_params.graph.expanded_kv_seq_lens_vec,
                                  nullptr,
                                  &input_params.graph.expanded_kv_seq_lens});
    }
    extra_int_inputs.push_back(
        {&expanded_block_tables_dense, nullptr, &expanded_block_tables_flat});
    attention.rebuild_device_buffer(
        device_,
        extra_int_inputs,
        initialize_stable_buffer
            ? AttentionInput::BufferReusePolicy::GROWABLE
            : AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
    if (initialize_stable_buffer && num_sequences > 0) {
      // Source views are part of the explicit replay contract. Reserve enough
      // packed storage for any configured decode batch before the first graph
      // captures their addresses, so a later larger batch cannot relocate the
      // buffer and invalidate an already captured variant.
      const uint64_t max_sequences =
          static_cast<uint64_t>(options_.max_seqs_per_batch());
      const uint64_t batch_scale =
          (max_sequences + static_cast<uint64_t>(num_sequences) - 1) /
          static_cast<uint64_t>(num_sequences);
      const uint64_t reserve_capacity =
          attention.attention_buffer_bytes * batch_scale;
      if (reserve_capacity > attention.attention_buffer_capacity) {
        attention.reserve_device_buffer_capacity(reserve_capacity, device_);
        attention.rebuild_device_buffer(
            device_,
            extra_int_inputs,
            AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
      }
    }
    spec_verify_attention_host_buffer_ = attention.attention_host_buffer;
    spec_verify_attention_device_buffer_ = attention.attention_device_buffer;
    spec_verify_attention_buffer_capacity_ =
        attention.attention_buffer_capacity;
    input_params.graph.expanded_block_tables = expanded_block_tables_flat.view(
        {expanded_block_rows, verify_block_table_width});
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        input_params,
        input_params.graph.expanded_kv_seq_lens,
        input_params.graph.expanded_block_tables,
        input_params.graph.expanded_kv_seq_lens_vec,
        block_size());
    input_params.graph.input_tokens_override = validate_input.token_ids;
    input_params.graph.spec_verify_source_addresses_stable = true;
  } else {
    if (supports_expanded_spec_verify()) {
      ensure_spec_verify_control_block_table(
          input_params,
          num_sequences,
          active_spec_verify_block_table_width(input_params));
    }
    input_params.attention.rebuild_device_buffer(device_);
    if (supports_expanded_spec_verify()) {
      build_expanded_spec_verify_graph_input(
          input_params, device_, block_size(), defer_expanded_paging);
    }
  }
#else
  input_params.attention.rebuild_device_buffer(device_);
#endif
  validate_input.device_tensors_ready = true;
  // This metadata is independent of the in-flight final draft. Keep it on the
  // auxiliary stream and hand it to the compute stream with a device event.
#if defined(USE_NPU)
  input_params.graph.spec_verify_static_graph_tasks_prepared =
      use_explicit_spec_verify_replay_update && static_graph_tasks_prepared;
#endif
  if (record_ready_event) {
    record_metadata_ready_event(*prepare_stream_, validate_input);
  }
}

bool MTPWorkerImpl::prepare_static_mtp_graph_tasks_before_final_draft(
    const ForwardInput& input) {
#if defined(USE_NPU)
  if (!should_use_explicit_spec_verify_replay_update(input) ||
      input.input_params.embedding.linear_state_ids.size() != 1 ||
      embedding_cache_ == nullptr ||
      input.input_params.embedding.embedding_ids.empty()) {
    return false;
  }
  const auto& block_tables = input.input_params.attention.host.block_tables;
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      block_tables.size(0) != 1) {
    return false;
  }
  const std::vector<int32_t> accepted_prefix_lengths =
      embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          options_.num_speculative_tokens() + 1);
  if (accepted_prefix_lengths.size() != 1) {
    return false;
  }
  const int64_t verify_block_table_width =
      spec_verify_block_table_width(block_tables);
  const auto& kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  if (kv_seq_lens.size() != 1) {
    return false;
  }
  const int64_t spec_width = options_.num_speculative_tokens() + 1;
  const int64_t base_kv_seq_len = kv_seq_lens.front();
  const int64_t spec_verify_max_kv_seq_len = base_kv_seq_len + spec_width - 1;
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = input.input_params.embedding.linear_state_ids.front(),
      .num_accepted_tokens = accepted_prefix_lengths.front(),
      .spec_width = spec_width,
      .block_table_width = verify_block_table_width,
      .base_kv_seq_len = base_kv_seq_len,
      .max_kv_seq_len = spec_verify_max_kv_seq_len,
  };
  auto* llm_target = dynamic_cast<LLMWorkerImpl*>(impl_.get());
  if (llm_target == nullptr) {
    return false;
  }
  return llm_target->prepare_static_mtp_graph_tasks(signal, *compute_stream_);
#else
  (void)input;
  return false;
#endif
}

void MTPWorkerImpl::prepare_draft_replay_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input) {
  // Called on the prepare stream after waiting for target context writes.
  const bool dp_enabled = parallel_args_.dp_size() > 1;
  CHECK(draft_context_replay_policy_.has_value())
      << "Draft context replay policy is not initialized.";
  const int32_t uniform_width =
      draft_context_replay_policy_->target_expansion_width;
  mtp_async::DraftReplayInputPlan plan =
      mtp_async::build_draft_replay_input_plan(
          base_input,
          last_states,
          embedding_cache_->embedding_placeholder(),
          options_.block_size() * parallel_args_.kv_split_size_effective(),
          uniform_width,
          extend_input.input_params.meta.is_graph_warmup,
          Platform::is_mlu() ? -1 : 0);
  auto& input_params = extend_input.input_params;
  if (!draft_impl_->has_request_state_cache()) {
    clear_draft_state(input_params);
  }
  const int32_t num_sequences = input_params.meta.num_sequences;
  auto& rows = plan.rows;
  const int32_t num_rows = static_cast<int32_t>(rows.out_positions.size());
  set_token_position_tensors(extend_input,
                             rows.out_token_ids,
                             rows.out_positions,
                             base_input.token_ids.options(),
                             base_input.positions.options());
  if (!plan.mrope_positions.empty()) {
    CHECK_EQ(plan.mrope_positions.size(), static_cast<size_t>(num_rows));
    std::vector<int32_t> rotary(3 * num_rows);
    for (int32_t axis = 0; axis < 3; ++axis) {
      for (int32_t row = 0; row < num_rows; ++row) {
        rotary[axis * num_rows + row] = plan.mrope_positions[row][axis];
      }
    }
    torch::Tensor rotary_host =
        make_cpu_int_tensor(rotary).reshape({3, num_rows});
    extend_input.positions_host = rotary_host;
    extend_input.positions =
        safe_to(rotary_host, base_input.positions.options(), true);
  }
  input_params.meta.num_sequences = num_rows;
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input_params.is_spec_verify = false;
  input_params.attention.host.kpool_query_lens = plan.kpool_query_lens;
  input_params.attn_metadata.reset();
  update_decode_attention_metadata(input_params,
                                   rows,
                                   /*q_max_seq_len=*/1,
                                   std::move(rows.out_q_seq_lens),
                                   std::move(rows.out_q_cu_seq_lens),
                                   rows.meta.kv_max_seq_len,
                                   std::move(rows.out_kv_seq_lens),
                                   /*update_block_tables=*/true);
  // A draft step executes its own row geometry (one decode row per replay row),
  // while base_input's execution batch still describes the target step. Publish
  // the draft's layout so the Python executor's InputBatch metadata matches the
  // rows it is handed.
  update_draft_execution_batch_metadata(input_params, num_rows);
  input_params.attention.rebuild_device_buffer(device_);
  for (auto& embedding : plan.embeddings) {
    embedding = embedding.to(device_);
  }
  input_params.embedding.input_embedding = torch::stack(plan.embeddings);

  for (auto* counts : {&input_params.parallel.dp_global_token_nums,
                       &input_params.parallel.raw_dp_global_token_nums}) {
    if (dp_enabled) {
      for (int32_t& count : *counts) {
        count *= uniform_width;
      }
    } else if (counts->size() == 1) {
      counts->front() = num_rows;
    }
  }
  auto& mask = input_params.expert.eplb_decode_token_mask;
  if (dp_enabled) {
    mask = eplb::expand_decode_token_mask(mask, uniform_width);
  } else if (mask.defined()) {
    auto source_rows =
        torch::tensor(plan.source_sequences, torch::dtype(torch::kLong))
            .to(mask.device());
    auto valid_rows = torch::tensor(plan.valid_rows, torch::dtype(torch::kInt))
                          .to(mask.options());
    mask = mask.reshape({-1}).index_select(0, source_rows) * valid_rows;
  }
  auto& sampling = extend_input.sampling_params;
  const auto idx_options = sampling.selected_token_idxes.defined()
                               ? sampling.selected_token_idxes.options()
                               : torch::dtype(torch::kInt).device(device_);
  sampling.selected_token_idxes =
      safe_to(make_cpu_int_tensor(plan.selected_rows),
              idx_options,
              /*non_blocking=*/true);
  if (!sampling.sample_idxes.defined()) {
    sampling.sample_idxes = torch::arange(num_sequences, idx_options);
  }
  extend_input.device_tensors_ready = true;
  check_draft_input_embedding(
      extend_input.input_params.embedding.input_embedding, "target replay");
  record_metadata_ready_event(*prepare_stream_, extend_input);
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input,
    bool force_two_rows,
    bool wait_for_compute_stream) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  // Regular draft preparation may consume tensors produced by the previous
  // compute. The placeholder-only first-draft prelaunch has no such dependency.
  if (wait_for_compute_stream) {
    prepare_stream_->wait_stream(*compute_stream_);
  }
  extend_input = base_input;
  prepare_draft_sampling(extend_input.sampling_params);
  clear_ready_events(extend_input);
  extend_input.device_tensors_ready = false;
  if (requires_full_target_replay()) {
    prepare_draft_replay_inputs(base_input, last_states, extend_input);
    return;
  }
  auto& input_params = extend_input.input_params;
  if (!draft_impl_->has_request_state_cache()) {
    clear_draft_state(input_params);
  }
  const int32_t num_sequences = input_params.meta.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";
  const bool requires_uniform_rows =
      Platform::is_npu() && is_glm5_next_target_model_type(
                                impl_->context_.get_model_args().model_type());
  const bool use_uniform_two_rows = should_use_uniform_two_draft_rows(
      last_states, force_two_rows, dp_enabled, requires_uniform_rows);
  DecodeBatchContext row_ctx = make_decode_batch_context(base_input);
  torch::TensorOptions token_options = extend_input.token_ids.options();
  torch::TensorOptions position_options = extend_input.positions.options();
  Slice<int32_t> token_ids = {
      base_input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(base_input.token_ids_host.numel())};

  DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences * 2);
  buf.out_positions.reserve(num_sequences * 2);
  buf.out_new_cache_slots.reserve(num_sequences * 2);
  buf.out_kv_seq_lens.reserve(num_sequences * 2);
  buf.out_q_seq_lens.reserve(num_sequences * 2);
  buf.out_q_cu_seq_lens.reserve(num_sequences * 2);
  buf.out_block_tables.reserve(static_cast<size_t>(num_sequences) * 2 *
                               row_ctx.block_table_stride);
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);

  auto to_worker_device = [this](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device() == device_) {
      return tensor;
    }
    return tensor.to(device_);
  };

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = to_worker_device(placeholder);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      DecodeTokenInput row;
      row.req_id = seq_id;
      row.token_id = token_id >= 0 ? token_id : 0;
      row.position_offset = position_offset;
      row.append_kv_len = true;
      row.append_q_len_one = true;
      row.append_block_table = true;
      append_decode_token(row_ctx, row, block_size(), buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(to_worker_device(embedding));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    const bool sequence_uses_two_rows =
        use_uniform_two_rows || state.all_draft_accepted;
    if (sequence_uses_two_rows) {
      int32_t prev_token_id = state.prev_token_id;
      int32_t prev_position_offset = -1;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        // The first DP draft extension has no preceding target token.
        // Reuse the current verifier state for the padding row.
        prev_token_id = state.token_id;
        prev_embedding = state.embedding;
      }
      CHECK_GE(prev_token_id, 0)
          << "Eagle/MTP draft extend previous row requires a real token";
      add_row(prev_token_id, prev_position_offset, prev_embedding);
      if (prev_is_placeholder) {
        // Redirect to padding block 0 to avoid overwriting correct KV cache.
        buf.out_new_cache_slots.back() = 0;
      }
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);
  }

  if (draft_impl_->has_request_state_cache()) {
    auto& spans = input_params.attention.host.kpool_query_lens;
    spans.clear();
    spans.reserve(num_sequences);
    int32_t begin = 0;
    for (const int32_t selected : selected_row_idx) {
      spans.emplace_back(selected + 1 - begin);
      begin = selected + 1;
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  set_token_position_tensors(extend_input,
                             buf.out_token_ids,
                             buf.out_positions,
                             token_options,
                             position_options);
  // The draft executes one DECODE row per token, with metadata sized to its
  // actual input instead of the target prefill or verification batch.
  update_draft_execution_batch_metadata(input_params,
                                        extend_input.positions.numel());
  input_params.meta.num_sequences =
      static_cast<int32_t>(buf.out_positions.size());
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  update_decode_attention_metadata(input_params,
                                   buf,
                                   /*q_max_seq_len=*/1,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(buf.out_q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/true);
  if (supports_expanded_spec_verify()) {
    input_params.attention.host.q_cu_seq_lens.clear();
    input_params.attention.host.q_cu_seq_lens.reserve(
        input_params.meta.num_sequences + 1);
    input_params.attention.host.q_cu_seq_lens.emplace_back(0);
    for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
      input_params.attention.host.q_cu_seq_lens.emplace_back(
          input_params.attention.host.q_cu_seq_lens.back() +
          input_params.get_q_seq_len(i));
    }
  }
  input_params.attention.rebuild_device_buffer(device_);

  input_params.embedding.input_embedding = torch::stack(expanded_embeddings);
  check_draft_input_embedding(input_params.embedding.input_embedding,
                              "decode extend");

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
      for (int32_t& token_num :
           input_params.parallel.raw_dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] =
          static_cast<int32_t>(buf.out_positions.size());
    }
  }
  input_params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      input_params.expert.eplb_decode_token_mask,
      static_cast<int32_t>(buf.out_positions.size()) / num_sequences);

#if defined(USE_NPU)
  // The extend layout is the 2B cache variant during steady overlap decode.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  if (use_uniform_two_rows) {
    // Uniform extension appends two rows per sequence and select the second
    // row.  Build the tiny control tensor directly on device; copying a
    // temporary pinned CPU tensor forces its allocator to synchronize before
    // the asynchronous H2D has completed.
    params.selected_token_idxes = torch::arange(
        /*start=*/1,
        /*end=*/2 * num_sequences,
        /*step=*/2,
        idx_options);
  } else {
    params.selected_token_idxes = safe_to(make_cpu_int_tensor(selected_row_idx),
                                          idx_options,
                                          /*non_blocking=*/true);
  }
  if (!params.sample_idxes.defined()) {
    // This control tensor is always the identity mapping. Generate it directly
    // on device instead of allocating a short-lived pinned H2D source.
    params.sample_idxes = torch::arange(
        /*start=*/0, /*end=*/num_sequences, idx_options);
  }
  extend_input.device_tensors_ready = true;
  record_metadata_ready_event(*prepare_stream_, extend_input);
}

bool MTPWorkerImpl::should_use_uniform_two_draft_rows(
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    bool force_two_rows,
    bool dp_enabled,
    bool requires_uniform_rows) {
  if (force_two_rows || dp_enabled) {
    return true;
  }
  return requires_uniform_rows &&
         std::any_of(last_states.begin(),
                     last_states.end(),
                     [](const EmbeddingCache::DecodeState& state) {
                       return state.all_draft_accepted;
                     });
}

void MTPWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
                                         ForwardInput& draft_input,
                                         int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input;
  if (requires_full_target_replay()) {
    // The first proposal used the target hidden at current_position - 1.
    // Recursive proposals advance from that draft position; target validation
    // continues to use the unshifted input template.
    CHECK(draft_context_replay_policy_.has_value())
        << "Draft context replay policy is not initialized.";
    position_offset += draft_context_replay_policy_->draft_position_offset;
    draft_input.input_params.attn_metadata.reset();
  }
  prepare_draft_sampling(draft_input.sampling_params);
  clear_ready_events(draft_input);
  draft_input.device_tensors_ready = false;

  auto& input_params = draft_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  if (!draft_impl_->has_request_state_cache()) {
    clear_draft_state(input_params);
  }
  const int32_t num_sequences = input_params.meta.num_sequences;
  if (draft_impl_->has_request_state_cache()) {
    input_params.attention.host.kpool_query_lens.assign(num_sequences, 1);
    input_params.is_spec_verify = false;
    // The first draft already restored its request slot. Subsequent drafts
    // append to that live state instead of replaying a prefix restore.
    input_params.embedding.linear_state_read_ids =
        input_params.embedding.linear_state_ids;
    input_params.embedding.linear_state_read_indices =
        input_params.embedding.linear_state_indices;
  }
  DecodeBatchContext row_ctx = make_decode_batch_context(input);
  DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    DecodeTokenInput row;
    row.req_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    append_decode_token(row_ctx, row, block_size(), buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(
      draft_input, make_cpu_int_tensor(buf.out_positions), position_options);
  update_decode_attention_metadata(
      input_params,
      buf,
      input_params.meta.q_max_seq_len,
      std::move(input_params.attention.host.q_seq_lens),
      std::move(input_params.attention.host.q_cu_seq_lens),
      buf.meta.kv_max_seq_len,
      std::move(buf.out_kv_seq_lens));
  if (supports_expanded_spec_verify()) {
    input_params.attention.host.q_cu_seq_lens.clear();
    input_params.attention.host.q_cu_seq_lens.reserve(
        input_params.meta.num_sequences + 1);
    input_params.attention.host.q_cu_seq_lens.emplace_back(0);
    for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
      input_params.attention.host.q_cu_seq_lens.emplace_back(
          input_params.attention.host.q_cu_seq_lens.back() +
          input_params.get_q_seq_len(i));
    }
  }
  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result. Prepare
  // them concurrently; token ids and embeddings are filled on compute_stream.
  record_metadata_ready_event(*prepare_stream_, draft_input);
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const SamplingParameters& verify_sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardOutput& target_output,
    const std::vector<uint8_t>& invalid_draft) {
  const bool probabilistic =
      options_.enable_probabilistic_draft() &&
      sampling_params.sampling_mode != SamplingMode::GREEDY;
  std::vector<torch::Tensor> draft_tokens;
  std::vector<torch::Tensor> draft_logits_steps;
  draft_tokens.reserve(draft_outputs.size());
  if (probabilistic) {
    draft_logits_steps.reserve(draft_outputs.size());
  }
  for (const auto& draft_output : draft_outputs) {
    draft_tokens.emplace_back(draft_output.sample_output.next_tokens);
    if (probabilistic) {
      draft_logits_steps.emplace_back(draft_output.sample_output.draft_logits);
    }
  }
  const auto draft_token_ids = torch::stack(draft_tokens, /*dim=*/1);
  const auto draft_logits = probabilistic
                                ? torch::stack(draft_logits_steps, /*dim=*/1)
                                : torch::Tensor();
  const auto verify_draft_token_ids =
      JsonObjectDecodeAdapter::mask_invalid_draft_tokens(draft_token_ids,
                                                         invalid_draft);
  return run_rejection_sampling(verify_sampling_params,
                                verify_draft_token_ids,
                                draft_logits,
                                target_output.logits,
                                target_output.sample_output.embeddings);
}

}  // namespace xllm
