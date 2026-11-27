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

#include "core/runtime/dflash_worker_impl.h"

#include <c10/util/SmallVector.h>
#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/instance_role_util.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_sampling_input.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/kv_shard_batch_metadata.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#if defined(USE_NPU)
#include "core/kernels/npu/utils.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/layers/npu_torch/deepseek_sparse_attention.h"
#endif
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/platform/platform.h"
#include "core/runtime/dflash/dflash_context_positions.h"
#include "core/runtime/draft_model_config.h"
#include "core/runtime/json_object_decode_adapter.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/speculative_worker_utils.h"
#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"
#include "core/util/timer.h"
#include "core/util/utils.h"

namespace xllm {
namespace {

using speculative_worker::clear_all_output_embeddings;
using speculative_worker::clear_selected_embeddings;
using speculative_worker::record_metadata_ready_event;
using speculative_worker::wait_metadata_ready_event;

runtime::Options target_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options draft_options(const runtime::Options& options,
                               bool enable_dflash_xfia) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .backend("llm")
      .is_draft_engine(true)
      .enable_dflash_xfia(enable_dflash_xfia)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

// Resolve the draft topology once, before constructing its worker. Shared
// output weights retain the target communication layout.
struct DraftWorkerConfig {
  ParallelArgs parallel_args;
  runtime::Options options;
};

DraftWorkerConfig resolve_draft_worker_config(
    const ParallelArgs& target_parallel_args,
    const runtime::Options& options,
    bool enable_dflash_xfia) {
  DraftWorkerConfig config{target_parallel_args,
                           draft_options(options, enable_dflash_xfia)};
  if (Platform::uses_cp_independent_tp_for_block_diffusion_draft()) {
    if (target_parallel_args.cp_size() > 1) {
      config.parallel_args =
          parallel_state::get_cp_independent_tp_parallel_args(
              target_parallel_args);
      if (!SpeculativeConfig::is_dspark_algorithm(
              options.speculative_algorithm()) &&
          config.parallel_args.lm_head_group_ == nullptr) {
        config.parallel_args.lm_head_group_ = target_parallel_args.tp_group_;
      }
    }
    config.options.cp_size(config.parallel_args.cp_size())
        .tp_size(config.parallel_args.tp_size());
  }
  LOG(INFO) << "Draft parallel layout: algorithm="
            << options.speculative_algorithm()
            << ", rank=" << config.parallel_args.rank()
            << ", target_cp=" << target_parallel_args.cp_size()
            << ", target_tp=" << target_parallel_args.tp_size()
            << ", draft_cp=" << config.parallel_args.cp_size()
            << ", draft_tp=" << config.parallel_args.tp_size()
            << ", draft_tp_group_size="
            << (config.parallel_args.tp_group_ != nullptr
                    ? config.parallel_args.tp_group_->world_size()
                    : 1);
  return config;
}

void expand_block_parallel_sequence_rows(ModelInputParams& input_params,
                                         int32_t query_width) {
  input_params.meta.num_sequences *= query_width;
  if (input_params.meta.actual_num_sequences > 0) {
    input_params.meta.actual_num_sequences *= query_width;
  }
}

// Stage a host int32 vector to `device` on the caller's active stream.
torch::Tensor cpu_int_vec_to_device(const std::vector<int32_t>& values,
                                    const Device& device) {
  return safe_to(
      make_cpu_int_tensor(values),
      torch::TensorOptions().dtype(torch::kInt).device(device.unwrap()),
      /*non_blocking=*/true);
}

void repeat_sampling_tensor(torch::Tensor& tensor, int32_t repeats) {
  if (tensor.defined()) {
    tensor = tensor.repeat_interleave(/*repeats=*/repeats, /*dim=*/0);
  }
}

void repeat_sampling_params(SamplingParameters& sampling_params,
                            int32_t repeats) {
  repeat_sampling_tensor(sampling_params.frequency_penalties, repeats);
  repeat_sampling_tensor(sampling_params.presence_penalties, repeats);
  repeat_sampling_tensor(sampling_params.repetition_penalties, repeats);
  repeat_sampling_tensor(sampling_params.temperatures, repeats);
  repeat_sampling_tensor(sampling_params.top_p, repeats);
  repeat_sampling_tensor(sampling_params.top_k, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_counts, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids_lens, repeats);
  repeat_sampling_tensor(sampling_params.do_sample, repeats);
}

// Stages the drafted block to pinned host memory with an async copy on the
// current compute stream. The JSON grammar replay needs the draft tokens on
// the host to advance per-sequence grammar states; issuing the copy right
// after the draft forward lets the validate-input host prep overlap with it.
// The caller must record a copy-completion event and wait before reading.
torch::Tensor stage_draft_tokens_to_pinned_host(
    const torch::Tensor& draft_token_ids) {
  CHECK_EQ(draft_token_ids.dim(), 2)
      << "DFlash JSON grammar replay requires [batch, block] draft ids";
  torch::Tensor pinned = torch::empty(draft_token_ids.sizes(),
                                      torch::TensorOptions()
                                          .dtype(draft_token_ids.dtype())
                                          .device(torch::kCPU)
                                          .pinned_memory(true));
  pinned.copy_(draft_token_ids, /*non_blocking=*/true);
  return pinned;
}

#if defined(USE_NPU)
void build_dflash_expanded_spec_verify_graph_input(
    ModelInputParams& input_params,
    const torch::Device& device,
    int32_t block_size) {
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return;
  }
  const auto& q_seq_lens = input_params.attention.host.q_seq_lens;
  const auto& kv_seq_lens = input_params.attention.host.kv_seq_lens;
  CHECK(!q_seq_lens.empty());
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size());
  std::vector<int32_t> expanded_kv_seq_lens =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  torch::Tensor expanded_kv_seq_lens_device =
      torch::tensor(expanded_kv_seq_lens,
                    torch::TensorOptions()
                        .dtype(torch::kInt)
                        .device(torch::kCPU)
                        .pinned_memory(true))
          .to(device, /*non_blocking=*/true);

  torch::Tensor expanded_block_tables =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_block_tables(
          input_params.attention.device.block_tables, q_seq_lens);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      input_params,
      expanded_kv_seq_lens_device,
      expanded_block_tables,
      std::move(expanded_kv_seq_lens),
      block_size);
}
#endif

// Rebuilds the [B] plain-RoPE decode base positions (kv_len - 1) shared by the
// synchronous and asynchronous decode paths. MRoPE axis rows carry a request
// delta, so the KV lengths are the only authoritative source.
std::vector<int32_t> rebuild_decode_base_positions(
    const ModelInputParams& input_params) {
  const int32_t num_sequences = input_params.meta.num_sequences;
  std::vector<int32_t> positions;
  positions.reserve(static_cast<size_t>(num_sequences));
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t kv_len = calc_kv_len(
        input_params.attention.host.kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_GT(kv_len, 0) << "DFlash decode KV length must be positive, seq="
                        << seq_id;
    positions.emplace_back(kv_len - 1);
  }
  return positions;
}

void normalize_draft_decode_positions(ForwardInput& input) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  if (!dflash_detail::is_mrope_context_positions(input.positions_host,
                                                 num_sequences)) {
    return;
  }

  // Only this draft-local copy is rewritten. The shared target input retains
  // its original MRoPE coordinates for verification.
  input.positions_host =
      make_cpu_int_tensor(rebuild_decode_base_positions(input.input_params));
}

std::vector<int64_t> build_accepted_context_rows(
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens_cpu,
    int32_t block_size,
    DecodeBuildBuffers& buf) {
  const int32_t batch_size = static_cast<int32_t>(accepted_tokens_cpu.size(0));
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  CHECK_EQ(input.input_params.meta.num_sequences, batch_size)
      << "DFlash accepted token batch mismatch.";

  DecodeBatchContext row_ctx = make_decode_batch_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_positions.reserve(buf.out_positions.size() +
                            static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_new_cache_slots.reserve(
      buf.out_new_cache_slots.size() +
      static_cast<size_t>(accepted_tokens_cpu.numel()));

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      DecodeTokenInput row;
      row.req_id = seq_id;
      row.position_offset = token_idx;
      row.append_token = false;
      row.append_kv_len = false;
      append_decode_token(row_ctx, row, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "DFlash accepted context slots/positions mismatch.";
  return accepted_idxes;
}

}  // namespace

bool enable_dflash_proposal_xfia(const torch::Device& device,
                                 const runtime::Options& options) {
  if (!SpeculativeConfig::is_dflash_algorithm(
          options.speculative_algorithm()) &&
      !SpeculativeConfig::is_dflash2_algorithm(
          options.speculative_algorithm())) {
    return false;
  }
#if defined(USE_NPU)
  const bool is_ascend_a3 = kernel::npu::is_ascend_a3();
#else
  const bool is_ascend_a3 = false;
#endif
  return is_ascend_a3 &&
         ModelConfig::is_python_model_impl(
             ModelConfig::get_instance().model_impl()) &&
         device.is_privateuseone() && options.enable_schedule_overlap() &&
         (!options.enable_disagg_pd() ||
          options.instance_role() == InstanceRole::DECODE);
}

// The target worker type must flow through to the speculative composite: a
// VLM target built as the LLM worker keeps the text-only model bridge, so its
// executor never creates the multimodal pipeline and image requests silently
// lose their embeddings. Text-only deployments pass WorkerType::LLM explicitly.
DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   WorkerType worker_type)
    : DraftModelSpecWorkerImpl(
          parallel_args,
          device,
          options,
          target_options(options),
          worker_type,
          [&parallel_args, &device, &options] {
            // The draft consumes global token-aligned aux hidden states.
            // MLU restores these rows for GLM-5-Next with DFlash2. NPU admits
            // the proven aux-hidden CP restore pairings (design.md §3.1
            // Gate B4): DeepSeek-V4 on NPU, or GLM-5-Next / glm_moe_dsa on
            // NPU with the DFlash2 algorithm specifically. GLM-5-Next's
            // aux-hidden-buffer CP restore is proven correct
            // (test_glm5_next_cp.py's
            // test_layer_capture_restores_global_rows_with_context_parallelism)
            // but was only verified for DFlash2's dispatch path; other
            // block-diffusion algorithms (DFlash(1), DSpark) stay rejected.
            // glm_moe_dsa (GLM-5.2/5.3) joins the same exemption: glm5_2.py
            // merges the target output and the aux-hidden buffer with one
            // cp_merge_rows permutation (test_glm5_2_cp_ragged.py's
            // test_cp_aux_hidden_is_merged_with_target_in_capture_order),
            // so the DFlash2 draft consumes token-aligned global rows. Its
            // exemption is python-impl only -- the native/ATB glm_moe_dsa
            // has no aux capture and keeps the startup-time rejection.
            bool allow_cp = false;
            // Native MLU V4.1 restores captured target features to global rows
            // before handing them to DSpark; the existing draft path is reused.
            if (parallel_args.cp_size() > 1 && Platform::is_mlu() &&
                !ModelConfig::is_python_model_impl(
                    ModelConfig::get_instance().model_impl())) {
              allow_cp = util::is_deepseek_v41_model_type(util::get_model_type(
                             options.model_path(), options.backend())) &&
                         SpeculativeConfig::is_dspark_algorithm(
                             options.speculative_algorithm());
            }
            if (parallel_args.cp_size() > 1 &&
                (Platform::is_npu() || Platform::is_mlu())) {
              const std::string model_type =
                  util::get_model_type(options.model_path(), options.backend());
              allow_cp = allow_cp ||
                         SpeculativeConfig::is_glm5_next_dflash2(
                             model_type, options.speculative_algorithm()) ||
                         (Platform::is_npu() &&
                          (util::is_deepseek_v4_model_type(model_type) ||
                           SpeculativeConfig::is_glm_moe_dsa_dflash2(
                               model_type,
                               options.speculative_algorithm(),
                               ModelConfig::get_instance().model_impl())));
            }
            if (!allow_cp) {
              CHECK_LE(parallel_args.cp_size(), 1)
                  << "Block-diffusion speculative decoding does not support "
                     "context parallelism (cp_size > 1).";
            }
            const DraftWorkerConfig config = resolve_draft_worker_config(
                parallel_args,
                options,
                enable_dflash_proposal_xfia(device, options));
            return std::make_unique<LLMWorkerImpl>(
                config.parallel_args, device, config.options);
          }) {
  const bool python_dflash =
      SpeculativeConfig::is_dflash_algorithm(options.speculative_algorithm()) &&
      Platform::is_npu() &&
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl());
  async_output_enabled_ = python_dflash && enable_schedule_overlap();
  proposal_xfia_enabled_ = enable_dflash_proposal_xfia(device, options);
  // Python block-query graphs need proposal metadata on non-XFIA NPUs too.
  proposal_attention_enabled_ =
      proposal_xfia_enabled_ ||
      (Platform::is_npu() && ModelConfig::is_python_model_impl(
                                 ModelConfig::get_instance().model_impl()));
  draft_prelaunch_enabled_for_init_ =
      async_output_enabled_ && proposal_xfia_enabled_ &&
      parallel_args.kv_split_size_effective() == 1;
}

DFlashWorkerImpl::~DFlashWorkerImpl() { retire_async_output(); }

void DFlashWorkerImpl::apply_target_model_overrides(ModelArgs& args) const {
  // DFlash draft attends each block non-causally, which the shared QWen3 model
  // only wires up on the chunked-prefill mask path. Without it the draft falls
  // back to a causal mask and proposal quality silently degrades, so require
  // the flag rather than accept a misconfigured run.
  CHECK(::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
      << "Block-diffusion speculative decoding requires "
         "--enable_chunked_prefill=true.";
  // Python MoE capacity is sized from ModelArgs and must accommodate the
  // complete verification width even for block-diffusion drafts.
  args.num_speculative_tokens(options_.num_speculative_tokens());
  configure_block_diffusion_target(args, options_);
}

void DFlashWorkerImpl::apply_draft_model_overrides(ModelArgs& args) const {
  configure_block_diffusion_draft(args, draft_impl_->get_options());
}

void DFlashWorkerImpl::on_target_model_loaded() {
  target_is_hybrid_recurrent_ =
      has_linear_attention_layers(context_.get_model_args());
}

void DFlashWorkerImpl::on_draft_model_loaded(
    const std::string& model_weights_path) {
  if (proposal_attention_enabled_ && (SpeculativeConfig::is_dflash_algorithm(
                                          options_.speculative_algorithm()) ||
                                      SpeculativeConfig::is_dflash2_algorithm(
                                          options_.speculative_algorithm()))) {
    draft_prelaunch_enabled_for_init_ =
        draft_prelaunch_enabled_for_init_ &&
        draft_impl_->supports_device_only_decode_inputs();
    proposal_xfia_enabled_ = draft_prelaunch_enabled_for_init_;
    draft_impl_->set_device_only_decode_inputs(proposal_xfia_enabled_);
    LOG(INFO) << "DFlash device-only/prelaunch enabled="
              << draft_prelaunch_enabled_for_init_;
  }

  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse block-diffusion draft config: " << config_path;
  // The root-level aliases win over the ``text_config`` fallback and any
  // drift surfaces as a WARNING (see read_mask_token_id).
  mask_token_id_ = read_mask_token_id(reader);
  CHECK_GE(mask_token_id_, 0)
      << "Block-diffusion draft config requires mask_token_id, "
         "dflash_config.mask_token_id, dspark_noise_token_id, "
         "text_config.dspark_noise_token_id, or text_config.mask_token_id.";

  const int64_t draft_vocab_size =
      draft_impl_->context_.get_model_args().vocab_size();
  CHECK_GT(draft_vocab_size, 0)
      << "Block-diffusion draft vocab_size must be set.";
  CHECK_LT(mask_token_id_, draft_vocab_size)
      << "Block-diffusion mask_token_id (" << mask_token_id_
      << ") must be < draft vocab_size (" << draft_vocab_size << ").";
  if (SpeculativeConfig::is_dflash2_algorithm(
          options_.speculative_algorithm())) {
#if defined(USE_NPU)
    // The graph requirement guards the decode-side spec-verify replay, so it
    // is exempt only for an instance that never receives a decode batch:
    // is_pd_prefill_only() rather than a bare role comparison. The role flag
    // alone is not enough -- without enable_disagg_pd the continuous
    // scheduler still schedules decode on an instance whose role says
    // PREFILL, and under enable_pd_ooc the PD prefill instance decodes its
    // offline requests -- either case would silently run the eager
    // validation this CHECK rejects instead of aborting at startup. Note the
    // exemption cannot lean on graph warmup: no warmup runs at all when
    // enable_graph=false, which is the only configuration this CHECK fails.
    CHECK(!target_is_hybrid_recurrent_ || is_pd_prefill_only(options_) ||
          ::xllm::ExecutionConfig::get_instance().enable_graph())
        << "DFlash2 with a hybrid recurrent target requires ACL Graph: the "
           "expanded spec-verify replay path preserves the accepted GDN "
           "checkpoint, while eager validation is not lossless.";
#endif
  }
  // Context hidden comes from the target.
  const ModelArgs& target_args = impl_->context_.get_model_args();

  // Probabilistic rejection requires aligned vocabularies.
  if (options_.enable_probabilistic_draft()) {
    const int64_t target_vocab_size = target_args.vocab_size();
    CHECK_EQ(draft_vocab_size, target_vocab_size)
        << "enable_probabilistic_draft requires the draft vocab ("
        << draft_vocab_size << ") to equal the target vocab ("
        << target_vocab_size << ").";
  }
  const int64_t num_target_layers =
      static_cast<int64_t>(target_args.layers_to_capture().size());
  CHECK_GT(num_target_layers, 0)
      << "Block-diffusion draft config requires dspark_target_layer_ids, "
         "target_layer_ids, dflash_config.target_layer_ids, or "
         "aux_hidden_state_layer_ids.";
  expected_context_hidden_size_ =
      static_cast<int64_t>(target_args.hidden_size()) * num_target_layers;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const ForwardInput& input) {
  commit_pending_context();
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    std::optional<ForwardOutput> output =
        speculative_worker::run_worker_no_sync(
            *impl_, input, *prepare_stream_, *compute_stream_);
    // Active prefill ranks write the draft context KV without a draft forward.
    // Keep idle ranks symmetric: a draft MoE forward here would enter EP
    // collectives that active ranks never join and deadlock the whole group.
    // Sync the target forward before its staged input can be reused.
    compute_stream_->synchronize();
    if (output.has_value()) {
      clear_all_output_embeddings(output.value());
    }
    return output;
  }

  // Idle ranks still use the active ranks' draft width for DP token counts.
  // Their local query geometry follows the draft's dummy input contract.
  const int32_t draft_width = dflash_detail::decode_draft_width(
      options_.num_speculative_tokens(), sample_from_anchor());
  const bool use_block_parallel_rows = uses_block_parallel_draft_decode();
  ForwardInput query_input = input;
  dflash_detail::reset_draft_model_metadata(query_input.input_params);
  query_input.input_params.use_dflash_proposal_attention =
      input.input_params.use_dflash_proposal_attention ||
      (proposal_attention_enabled_ && !input.input_params.meta.is_graph_warmup);
  query_input.input_params.meta.batch_forward_type = draft_batch_forward_type();
  if (draft_impl_->context_.get_model_args().dummy_token_count() == 1) {
    // The metadata builder uses zero query length to materialize a dummy row.
    query_input.input_params.meta.q_max_seq_len = 0;
  } else {
    query_input.input_params.meta.q_max_seq_len =
        use_block_parallel_rows ? 1 : draft_width;
  }
  if (use_block_parallel_rows) {
    expand_block_parallel_sequence_rows(query_input.input_params, draft_width);
  }
  scale_speculative_parallel_token_counts(query_input.input_params,
                                          draft_width);
  expand_draft_block_tables(query_input.input_params);
  // Warmup only: prime the draft; its output is unused. Keep it alive until the
  // sync below so the no-sync draft input is not freed while the target forward
  // launched next can reuse the buffer.
  std::optional<ForwardOutput> draft_output =
      speculative_worker::run_worker_no_sync(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_);

  ForwardInput validate_input = input;
  validate_input.input_params.use_dflash_proposal_attention = false;
  validate_input.input_params.sampling.reset();
  // Idle decode joins the busy ranks' target validate. Mark the same spec
  // verify flag so every rank captures one expanded graph instead of a
  // separate width-1 graph on the dummy row.
  validate_input.input_params.is_spec_verify = target_is_hybrid_recurrent_;
  // DSpark's N-wide draft geometry must be rescaled to (N+1) for the target's
  // anchor + drafts forward.
  scale_speculative_parallel_token_counts(
      validate_input.input_params, options_.num_speculative_tokens() + 1);
  // Mirror prepare_validate_inputs on the idle shard: a NON-expanded verify
  // pack must export its rows-per-sequence so this fake-single-row batch folds
  // to the same sequence bucket as the busy ranks. Both runners derive the
  // graph key from the SAME group-wide token counts, so a missing width would
  // size this rank locally (width 1 -> 1 sequence-in-1-row vs the busy rank's
  // one-sequence-in-(N+1)-rows) and desynchronize the runner choice: busy
  // ranks replay the target verify graph while this rank falls to eager, and
  // the HCCL collectives in the target forward deadlock. The chunked/expanded
  // layout keeps width 1 (its width is carried by the query-chunk metadata).
  if (validate_input.input_params.is_spec_verify) {
    // Match the active shard's validate mode without turning the idle shard
    // into a real query. Keep q_max_seq_len == 0 so metadata preparation uses
    // its one-row dummy layout and does not update live recurrent state.
    validate_input.input_params.meta.batch_forward_type =
        BatchForwardType::CHUNKED_PREFILL;
  } else {
    validate_input.input_params.meta.num_speculative_tokens =
        options_.num_speculative_tokens();
  }

  ForwardOutput output =
      speculative_worker::run_worker_no_sync(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  finish_empty_decode(input, output, draft_output);
  clear_all_output_embeddings(output);
  return output;
}

void DFlashWorkerImpl::finish_empty_decode(
    const ForwardInput& input,
    ForwardOutput& output,
    std::optional<ForwardOutput>& draft_output) {
  if (!draft_prelaunch_enabled_for_init_ ||
      input.input_params.meta.is_graph_warmup) {
    compute_stream_->synchronize();
    return;
  }
  StreamEventPtr ready = compute_stream_->record_event();
  CHECK(ready != nullptr);
  // The next draft must reach DP graph admission while the current target is
  // running, including on idle ranks. Retire the previous idle inputs only
  // after this step's draft and target have both been submitted.
  if (idle_decode_output_.ready_event != nullptr) {
    CHECK(idle_decode_output_.ready_event->synchronize());
  }
  idle_decode_output_.ready_event = std::move(ready);
  idle_decode_output_.retained_inputs = take_retained_inputs(output);
  output.ready_event.reset();
  if (draft_output.has_value()) {
    transfer_retained_inputs(idle_decode_output_, draft_output.value());
  }
  // This worker owns retirement. The empty output carries no token that the
  // generic scheduler must wait to read, and no input it must keep alive.
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const ForwardInput& input) {
  commit_pending_context();
  Timer timer;
  ForwardInput processed_target_input;
  ForwardOutput output =
      speculative_worker::run_worker_no_sync(
          *impl_,
          input,
          *prepare_stream_,
          *compute_stream_,
          &processed_target_input,
          /*options=*/{},
          [this](ModelInputParams& params) {
            params.keep_cp_aux_in_gather_order =
                options_.cp_size() > 1 &&
                params.meta.batch_forward_type.no_decode() &&
                SpeculativeConfig::is_glm_moe_dsa_dflash2(
                    impl_->context_.get_model_args().model_type(),
                    options_.speculative_algorithm(),
                    ModelConfig::get_instance().model_impl());
          })
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  torch::Tensor restore_index = std::move(output.aux_hidden_restore_index);
  if (embeddings.defined()) {
    const int64_t num_context_tokens =
        restore_index.defined() ? restore_index.numel() : embeddings.size(0);
    const bool is_mrope_positions = dflash_detail::is_mrope_context_positions(
        processed_target_input.positions_host, num_context_tokens);
    if (is_mrope_positions) {
      ModelInputParams& input_params = processed_target_input.input_params;
      const int32_t num_sequences = input_params.meta.num_sequences;
      CHECK_GE(num_sequences, 0);
      std::vector<int32_t> kv_seq_lens;
      std::vector<int32_t> q_seq_lens;
      kv_seq_lens.reserve(static_cast<size_t>(num_sequences));
      q_seq_lens.reserve(static_cast<size_t>(num_sequences));
      for (int32_t seq = 0; seq < num_sequences; ++seq) {
        kv_seq_lens.emplace_back(calc_kv_len(
            input_params.attention.host.kv_seq_lens, seq, /*offset=*/0));
        q_seq_lens.emplace_back(input_params.get_q_seq_len(seq));
      }
      const std::vector<int32_t> token_aligned_positions =
          dflash_detail::build_token_aligned_context_positions(
              kv_seq_lens, q_seq_lens, num_context_tokens);
      torch::Tensor normalized_positions_host =
          make_cpu_int_tensor(token_aligned_positions);
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      torch::Tensor normalized_positions_device =
          safe_to(normalized_positions_host,
                  torch::TensorOptions().dtype(torch::kInt).device(device_),
                  /*non_blocking=*/true);
      // Normalize the processed input itself so every downstream consumer,
      // including grouped SWA slot building, observes the same 1-D contract.
      processed_target_input.positions_host =
          std::move(normalized_positions_host);
      processed_target_input.positions = std::move(normalized_positions_device);
    }

    Slice<int32_t> positions = {
        processed_target_input.positions_host.data_ptr<int32_t>(),
        static_cast<size_t>(processed_target_input.positions_host.numel())};
    CHECK_EQ(positions.size(), static_cast<size_t>(num_context_tokens))
        << "DFlash prefill hidden/position count mismatch.";
    torch::Tensor context_cache_slots =
        processed_target_input.input_params.attention.device.new_cache_slots;
    if (!processed_target_input.input_params.multi_block_tables.empty()) {
      const std::vector<int32_t> grouped_swa_slots =
          build_grouped_prefill_swa_slots(processed_target_input,
                                          options_.block_size());
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      context_cache_slots = cpu_int_vec_to_device(grouped_swa_slots, device_);
    }
    CHECK(context_cache_slots.defined())
        << "DFlash prefill requires context cache slots.";
    CHECK_EQ(static_cast<size_t>(context_cache_slots.numel()), positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    torch::Tensor context_positions = processed_target_input.positions;
    if (restore_index.defined()) {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      std::tie(context_positions, context_cache_slots) =
          dflash_detail::scatter_cp_context_metadata(context_positions,
                                                     context_cache_slots,
                                                     restore_index,
                                                     embeddings.size(0));
    }
    timer.reset();
    KvPushFailureIds failures = write_context_kv(processed_target_input,
                                                 embeddings,
                                                 context_positions,
                                                 context_cache_slots);
    output.failed_request_ids.insert(
        output.failed_request_ids.end(),
        std::make_move_iterator(failures.failed_request_ids.begin()),
        std::make_move_iterator(failures.failed_request_ids.end()));
    output.unsafe_request_ids.insert(
        output.unsafe_request_ids.end(),
        std::make_move_iterator(failures.unsafe_request_ids.begin()),
        std::make_move_iterator(failures.unsafe_request_ids.end()));
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
  }

  const bool target_prefill_failed =
      !embeddings.defined() && !output.failed_request_ids.empty();
  if (!target_prefill_failed && !is_pd_prefill_only(options_) &&
      input.sampling_params.selected_token_idxes.defined()) {
    const auto& embedding_ids = input.input_params.embedding.embedding_ids;
    torch::Tensor bootstrap_embeddings = embeddings;
    if (restore_index.defined() ||
        embeddings.size(0) != static_cast<int64_t>(embedding_ids.size())) {
      bootstrap_embeddings = dflash_detail::select_prefill_context_rows(
          embeddings,
          input.sampling_params.selected_token_idxes,
          restore_index);
    }
    // Local decode seeds its continuation cache from the prefill target.
    embedding_cache_->write_prefill_target_context(
        embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        bootstrap_embeddings);
  }
  clear_all_output_embeddings(output);

  consumed_context_tensors_.clear();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    if (output.failed_request_ids.empty()) {
      return std::nullopt;
    }
    return make_failed_output(std::move(output.failed_request_ids),
                              std::move(output.unsafe_request_ids));
  }
  return output;
}

void DFlashWorkerImpl::retire_async_output() {
  if (idle_decode_output_.ready_event != nullptr) {
    CHECK(idle_decode_output_.ready_event->synchronize());
  }
  if (pending_context_.has_value()) {
    CHECK(pending_context_->context_ready->synchronize());
  }
  if (!consumed_context_tensors_.empty()) {
    compute_stream_->synchronize();
  }
}

std::vector<int64_t> DFlashWorkerImpl::build_batch_to_pending_row_mapping(
    const ForwardInput& input) const {
  if (!pending_context_.has_value()) {
    return {};
  }
  const ForwardInput& previous = pending_context_->identity;
  const auto& previous_requests = previous.input_params.embedding.request_ids;
  const auto& current_requests = input.input_params.embedding.request_ids;
  const auto& previous_embeddings =
      previous.input_params.embedding.embedding_ids;
  const auto& current_embeddings = input.input_params.embedding.embedding_ids;
  const bool use_sequence_ids = !previous.sample_sequence_ids.empty() ||
                                !input.sample_sequence_ids.empty();
  if (use_sequence_ids &&
      (previous.sample_sequence_ids.size() != previous_requests.size() ||
       input.sample_sequence_ids.size() != current_requests.size())) {
    return {};
  }
  std::unordered_map<std::string, int64_t> previous_rows;
  previous_rows.reserve(previous_requests.size());
  for (size_t row = 0; row < previous_requests.size(); ++row) {
    const std::string& identity = use_sequence_ids
                                      ? previous.sample_sequence_ids[row]
                                      : previous_requests[row];
    if (!previous_rows.emplace(identity, static_cast<int64_t>(row)).second) {
      return {};
    }
  }
  std::vector<int64_t> rows;
  rows.reserve(current_requests.size());
  for (size_t row = 0; row < current_requests.size(); ++row) {
    const std::string& identity = use_sequence_ids
                                      ? input.sample_sequence_ids[row]
                                      : current_requests[row];
    const auto found = previous_rows.find(identity);
    if (found == previous_rows.end()) {
      return {};
    }
    const size_t previous_row = static_cast<size_t>(found->second);
    if (previous_requests[previous_row] != current_requests[row] ||
        previous_embeddings[previous_row] != current_embeddings[row]) {
      return {};
    }
    rows.emplace_back(found->second);
  }
  return rows;
}

void DFlashWorkerImpl::commit_pending_context() {
  if (!pending_context_.has_value()) {
    return;
  }
  CHECK(pending_context_->d2h_ready->synchronize())
      << "Failed to synchronize DFlash accepted tokens";
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  embedding_cache_->write_target_context(
      pending_context_->identity.input_params.embedding.embedding_ids,
      pending_context_->identity.input_params.embedding.request_ids,
      pending_context_->accepted_host,
      pending_context_->embeddings,
      options_.num_speculative_tokens());
  consumed_context_tensors_.reserve(consumed_context_tensors_.size() + 3);
  consumed_context_tensors_.emplace_back(pending_context_->accepted_device);
  consumed_context_tensors_.emplace_back(pending_context_->embeddings);
  consumed_context_tensors_.emplace_back(pending_context_->base_positions);
  pending_context_.reset();
}

void DFlashWorkerImpl::prepare_validate_output_async(
    const ForwardInput& input) {
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  // The async accepted-context path consumes base_positions as one
  // plain-RoPE position per sequence (kv_len - 1) to locate draft cache
  // slots. MRoPE targets carry [3, B] axis coordinates instead, whose
  // axis-0 row includes a request-specific delta and must not be used as
  // the logical position. Rebuild from kv_seq_lens, mirroring the
  // synchronous path's normalize_draft_decode_positions; the shared target
  // input keeps its MRoPE coordinates for verification. The rebuilt values
  // are written element-wise straight into the final kLong pinned buffer
  // (data_ptr<int64_t>), keeping this step at one pinned allocation + one
  // H2D copy as before the fix.
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const bool is_mrope_positions = dflash_detail::is_mrope_context_positions(
      input.positions_host, num_sequences);
  torch::Tensor host_base_positions =
      torch::empty({static_cast<int64_t>(num_sequences)},
                   torch::TensorOptions()
                       .dtype(torch::kLong)
                       .device(torch::kCPU)
                       .pinned_memory(true));
  if (is_mrope_positions) {
    const std::vector<int32_t> base_positions =
        rebuild_decode_base_positions(input.input_params);
    CHECK_EQ(static_cast<int64_t>(base_positions.size()),
             host_base_positions.numel());
    int64_t* base_positions_data = host_base_positions.data_ptr<int64_t>();
    for (size_t seq = 0; seq < base_positions.size(); ++seq) {
      base_positions_data[seq] = static_cast<int64_t>(base_positions[seq]);
    }
  } else {
    host_base_positions.copy_(input.positions_host);
  }
  prepared_base_positions_ = host_base_positions.to(device_, true);
  std::optional<ModelInputParams> expanded_draft_params;
  if (draft_replicated_block_pages() > 1) {
    expanded_draft_params.emplace(input.input_params);
    expand_draft_block_tables(*expanded_draft_params);
  }
  const ModelInputParams& params = expanded_draft_params.has_value()
                                       ? *expanded_draft_params
                                       : input.input_params;
  prepared_context_block_table_ = dflash_async::prepare_draft_cache_block_table(
      params.attention.device.block_tables,
      params.multi_block_tables,
      params.meta.num_sequences,
      device_);
  prepared_context_ready_ = prepare_stream_->record_event();
  CHECK(prepared_context_ready_ != nullptr);
}

void DFlashWorkerImpl::finalize_validate_output_async(
    const ForwardInput& input,
    SampleOutput output,
    ForwardOutput& target_output) {
  CHECK(!pending_context_.has_value());
  CHECK(prepared_context_ready_ != nullptr);
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(prepared_context_ready_));
  PendingContext pending;
  pending.identity = input;
  pending.accepted_device = output.next_tokens;
  pending.embeddings = output.embeddings;
  pending.base_positions = std::move(prepared_base_positions_);
  pending.accepted_host = torch::empty(
      output.next_tokens.sizes(),
      output.next_tokens.options().device(torch::kCPU).pinned_memory(true));
  pending.accepted_host.copy_(pending.accepted_device, true);
  pending.d2h_ready = compute_stream_->record_event();
  CHECK(pending.d2h_ready != nullptr);
  dflash_async::ContextKvInputs context =
      dflash_async::prepare_context_kv(pending.accepted_device,
                                       pending.base_positions,
                                       prepared_context_block_table_,
                                       options_.block_size());
#if defined(USE_MLU)
  ModelInputParams draft_input_params = input.input_params;
  draft_input_params.parallel.layer_synchronizer.reset();
  const ModelInputParams& context_input_params = draft_input_params;
#else
  const ModelInputParams& context_input_params = input.input_params;
#endif
  ModelOutput written = draft_impl_->write_context_kv(
      output.embeddings.reshape({output.next_tokens.numel(), -1}),
      context.positions,
      context.slots,
      context_input_params);
  CHECK(written.hidden_states.defined());
  pending.context_ready = compute_stream_->record_event();
  CHECK(pending.context_ready != nullptr);

  if (!input.json_object_states.empty()) {
    // Replay only after the asynchronous accepted-token copy is visible on
    // host. The context KV write can finish independently through
    // context_ready.
    CHECK(pending.d2h_ready->synchronize());
    JsonObjectDecodeAdapter::check_verified_tokens(
        input, pending.accepted_host, target_output);
  }

  auto retained = std::make_shared<ForwardInput>(input);
  retained->retained_device_tensors.reserve(
      retained->retained_device_tensors.size() + 6 +
      consumed_context_tensors_.size());
  retained->retained_device_tensors.emplace_back(
      std::move(prepared_context_block_table_.blocks));
  retained->retained_device_tensors.emplace_back(pending.accepted_device);
  retained->retained_device_tensors.emplace_back(pending.embeddings);
  retained->retained_device_tensors.emplace_back(pending.base_positions);
  retained->retained_device_tensors.emplace_back(std::move(context.positions));
  retained->retained_device_tensors.emplace_back(std::move(context.slots));
  retained->retained_device_tensors.insert(
      retained->retained_device_tensors.end(),
      std::make_move_iterator(consumed_context_tensors_.begin()),
      std::make_move_iterator(consumed_context_tensors_.end()));
  consumed_context_tensors_.clear();
  target_output.retained_inputs.emplace_back(std::move(retained));
  output.next_tokens = pending.accepted_host;
  finalize_verify_output(target_output, std::move(output), input);
  target_output.ready_event = pending.context_ready;
  target_output.requires_host_retirement = true;
  pending_context_ = std::move(pending);
  prepared_context_ready_.reset();
  VLOG(10) << "DFlash step-local output submitted: batch_id="
           << input.input_params.meta.batch_id
           << ", sequences=" << input.input_params.meta.num_sequences;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;
  const bool can_prelaunch = can_prepare_without_compute_stream_wait(input);
  std::vector<int64_t> pending_rows;
  if (can_prelaunch) {
    pending_rows = build_batch_to_pending_row_mapping(input);
  }
  auto draft_output = [&] {
    if (!pending_rows.empty()) {
      auto draft = run_decode_draft_from_pending(input, pending_rows);
      VLOG(1) << "DFlash draft prelaunch submitted: batch_id="
              << input.input_params.meta.batch_id
              << ", sequences=" << pending_rows.size();
      commit_pending_context();
      prepare_decode_input(input);
      prepare_validate_inputs(input, validate_input);
      return draft;
    }
    commit_pending_context();
    prepare_decode_input(input);
    return run_decode_draft(input, validate_input);
  }();
  if (can_prelaunch) {
    prepare_target_metadata(validate_input);
  }
  const bool async_output = async_output_enabled_ &&
                            !input.input_params.meta.is_graph_warmup &&
                            input.transfer_kv_infos.empty();
  auto output = run_validate(input, validate_input, draft_output, async_output);
  if (!async_output) {
    consumed_context_tensors_.clear();
  }
  return output;
}

void DFlashWorkerImpl::prepare_target_metadata(ForwardInput& validate_input) {
  ForwardInput prepared;
  impl_->prepare_work_before_execute_on_stream(
      validate_input, prepared, *prepare_stream_, false, false, false);
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  prepared.input_params.attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(
              prepared.input_params,
              impl_->context_.get_model_args().enable_mla(),
              std::nullopt,
              device_.unwrap()));
  speculative_worker::record_metadata_ready_event(*prepare_stream_, prepared);
  validate_input = std::move(prepared);
}

bool DFlashWorkerImpl::can_prepare_without_compute_stream_wait(
    const ForwardInput& input) const {
  if (!draft_prelaunch_enabled_for_init_ ||
      !can_prepare_device_draft_inputs(input)) {
    return false;
  }
  const auto rows = make_decode_batch_context(input);
  const int32_t max_query_tokens_with_pending_acceptance =
      2 * (options_.num_speculative_tokens() + 1);
  return has_decode_lookahead_capacity(
      rows,
      max_query_tokens_with_pending_acceptance,
      block_size(),
      draft_impl_->context_.get_model_args().max_position_embeddings());
}

bool DFlashWorkerImpl::can_prepare_device_draft_inputs(
    const ForwardInput& input) const {
  const auto& sampling = input.sampling_params;
  if (SpeculativeConfig::is_dflash_algorithm(
          options_.speculative_algorithm()) &&
      (sampling.unique_token_ids.defined() ||
       sampling.frequency_penalties.defined() ||
       sampling.presence_penalties.defined() ||
       sampling.repetition_penalties.defined())) {
    return false;
  }
  if (!proposal_xfia_enabled_ || input.input_params.meta.is_graph_warmup ||
      !input.input_params.meta.batch_forward_type.is_decode() ||
      input.input_params.meta.num_sequences <= 0 ||
      !input.input_params.multi_block_tables.empty() ||
      input.input_params.meta.requires_host_restore ||
      !input.input_params.block_copy.swap_blocks.empty() ||
      (input.input_params.block_copy.src_block_indices.defined() &&
       input.input_params.block_copy.src_block_indices.numel() > 0) ||
      !input.transfer_kv_infos.empty() ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size()) {
    return false;
  }
  const auto rows = make_decode_batch_context(input);
  return !rows.model_managed_multiblock &&
         has_decode_lookahead_capacity(
             rows,
             options_.num_speculative_tokens() + 1,
             block_size(),
             draft_impl_->context_.get_model_args().max_position_embeddings());
}

std::pair<ForwardInput, torch::Tensor>
DFlashWorkerImpl::prepare_pending_draft_query(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.use_dflash_proposal_attention = true;
  ForwardInput prepared;
  draft_impl_->prepare_work_before_execute_on_stream(
      query, prepared, *prepare_stream_, true, false);
  torch::Tensor row_indices;
  torch::Tensor host_row_indices;
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    host_row_indices = torch::empty(
        {static_cast<int64_t>(rows.size())},
        torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    std::copy(rows.begin(), rows.end(), host_row_indices.data_ptr<int64_t>());
    row_indices = host_row_indices.to(device_, torch::kLong, true);
    prepared.metadata_ready_event = prepare_stream_->record_event();
  }
  CHECK(prepared.metadata_ready_event != nullptr);
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(prepared.metadata_ready_event));
  torch::Tensor accepted =
      pending_context_->accepted_device.index_select(0, row_indices);
  torch::Tensor base_positions =
      pending_context_->base_positions.index_select(0, row_indices);
  dflash_async::NextDraftInputs next = dflash_async::prepare_next_draft(
      accepted,
      base_positions,
      prepared.input_params.attention.device.block_tables,
      mask_token_id_,
      options_.block_size());
  prepared.token_ids.copy_(next.query_tokens);
  prepared.positions.copy_(next.query_positions);
  prepared.input_params.attention.device.new_cache_slots.copy_(
      next.query_slots);
  prepared.input_params.attention.device.kv_seq_lens.copy_(next.kv_lengths);
  prepared.input_params.attn_metadata.reset();
  prepared.metadata_ready_event.reset();
  prepared.device_tensors_ready = true;
  prepared.retained_device_tensors.reserve(
      prepared.retained_device_tensors.size() + 6);
  prepared.retained_device_tensors.insert(
      prepared.retained_device_tensors.end(),
      {row_indices,
       host_row_indices,
       accepted,
       base_positions,
       pending_context_->accepted_device,
       pending_context_->base_positions});
  return {std::move(prepared), std::move(next.anchor_tokens)};
}

void DFlashWorkerImpl::prepare_decode_input(ForwardInput& input) {
  CHECK(embedding_cache_ != nullptr)
      << "DFlash embedding cache is not allocated";

  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  CHECK_EQ(last_states.size(),
           input.input_params.embedding.embedding_ids.size())
      << "DFlash decode target state count mismatch";

  update_decode_step_input(input, last_states);
}

ForwardOutput DFlashWorkerImpl::run_decode_draft(const ForwardInput& input,
                                                 ForwardInput& validate_input) {
  Timer timer;
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  query_input.input_params.use_dflash_proposal_attention =
      proposal_attention_enabled_ && !input.input_params.meta.is_graph_warmup;
  auto draft =
      execute_greedy_draft_query(std::move(query_input), /*prepared=*/false);
  prepare_validate_inputs(input, validate_input);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft;
}

ForwardOutput DFlashWorkerImpl::run_decode_draft_from_pending(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  Timer timer;
  auto prepared = prepare_pending_draft_query(input, rows).first;
  auto draft =
      execute_greedy_draft_query(std::move(prepared), /*prepared=*/true);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft;
}

ForwardOutput DFlashWorkerImpl::execute_greedy_draft_query(ForwardInput query,
                                                           bool prepared) {
  ForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query, processed_input, *prepare_stream_);
  }
  ModelSamplingInput sampling;
  sampling.parameters = processed_input.sampling_params;
  processed_input.input_params.sampling = std::move(sampling);
  ForwardOutput draft_output =
      draft_impl_
          ->execute_no_sync_on_stream(processed_input, *compute_stream_, false)
          .value();
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  torch::Tensor tokens;
  if (draft_output.draft_token_ids.defined()) {
    tokens = draft_output.draft_token_ids;
    CHECK_EQ(
        tokens.sizes(),
        torch::IntArrayRef({processed_input.input_params.meta.num_sequences,
                            num_speculative_tokens}));
  } else {
    tokens = draft_output.sample_output.next_tokens;
  }
  // Keep draft tokens consistent across tensor-parallel ranks.
  maybe_broadcast_spec_tokens(tokens);
  // Draft emits the whole block in one forward; reshape the flat outputs into
  // [batch, num_speculative_tokens] instead of splitting into per-step outputs.
  const int32_t num_draft_tokens = static_cast<int32_t>(tokens.numel());
  CHECK_EQ(num_draft_tokens % num_speculative_tokens, 0)
      << "DFlash draft token count mismatch.";
  const int32_t batch_size = num_draft_tokens / num_speculative_tokens;
  draft_output.draft_token_ids =
      tokens.view({batch_size, num_speculative_tokens});
  draft_output.logits.reset();
  return draft_output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::run_validate(
    const ForwardInput& input,
    ForwardInput& validate_input,
    ForwardOutput& draft_output,
    bool async_output) {
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  Timer timer;
  const auto& draft_token_ids = draft_output.draft_token_ids;
  const bool has_json_states = !input.json_object_states.empty();
  torch::Tensor json_draft_tokens_host;
  StreamEventPtr json_draft_tokens_ready;
  if (has_json_states) {
    json_draft_tokens_host = stage_draft_tokens_to_pinned_host(draft_token_ids);
    json_draft_tokens_ready = compute_stream_->record_event();
    CHECK(json_draft_tokens_ready != nullptr);
  }
  wait_metadata_ready_event(validate_input, *compute_stream_);
  const int64_t num_sequences = draft_token_ids.size(/*dim=*/0);
  const int64_t num_verify_tokens = options_.num_speculative_tokens() + 1;
  torch::Tensor verify_token_ids =
      validate_input.token_ids.view({num_sequences, num_verify_tokens});
  validate_input.device_tensors_ready = false;
  verify_token_ids.slice(/*dim=*/1, /*start=*/1)
      .copy_(
          safe_to(draft_token_ids.view({num_sequences, num_verify_tokens - 1}),
                  validate_input.token_ids.options(),
                  /*non_blocking=*/true),
          /*non_blocking=*/true);
  validate_input.device_tensors_ready = true;
  // The target prepares ACL graph buffers on a separate stream.
  record_metadata_ready_event(*compute_stream_, validate_input);
  if (has_json_states) {
    prepare_json_validation_states(
        input, json_draft_tokens_host, json_draft_tokens_ready, validate_input);
  }
  // Stage host-originating template data before target launch: a nominally
  // asynchronous pageable H2D can otherwise drain target's execution stream.
  if (async_output) {
    prepare_validate_output_async(input);
  }
  ForwardOutput target_output =
      speculative_worker::run_worker_no_sync(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  target_output.failed_request_ids.insert(
      target_output.failed_request_ids.end(),
      std::make_move_iterator(draft_output.failed_request_ids.begin()),
      std::make_move_iterator(draft_output.failed_request_ids.end()));
  target_output.unsafe_request_ids.insert(
      target_output.unsafe_request_ids.end(),
      std::make_move_iterator(draft_output.unsafe_request_ids.begin()),
      std::make_move_iterator(draft_output.unsafe_request_ids.end()));
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());
  timer.reset();
  const auto verify_draft_token_ids =
      JsonObjectDecodeAdapter::mask_invalid_draft_tokens(
          draft_token_ids, validate_input.json_object_invalid_draft);
  SampleOutput val_output =
      run_rejection_sampling(validate_input.sampling_params,
                             verify_draft_token_ids,
                             draft_output.draft_logits,
                             target_output.logits,
                             target_output.sample_output.embeddings);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // target forward and validate()'s reads share compute_stream_, so they
  // serialize without a cross-stream wait; the sync below only makes the
  // accepted tokens host-visible for the D2H copy and context-cache write.
  maybe_broadcast_spec_tokens(val_output.next_tokens);
  if (async_output) {
    finalize_validate_output_async(input, std::move(val_output), target_output);
    transfer_retained_inputs(target_output, draft_output);
    return target_output;
  }
  compute_stream_->synchronize();
  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  KvPushFailureIds failures = write_target_context_to_cache(input, val_output);
  target_output.failed_request_ids.insert(
      target_output.failed_request_ids.end(),
      std::make_move_iterator(failures.failed_request_ids.begin()),
      std::make_move_iterator(failures.failed_request_ids.end()));
  target_output.unsafe_request_ids.insert(
      target_output.unsafe_request_ids.end(),
      std::make_move_iterator(failures.unsafe_request_ids.begin()),
      std::make_move_iterator(failures.unsafe_request_ids.end()));
  if (has_json_states) {
    JsonObjectDecodeAdapter::check_verified_tokens(
        input, val_output.next_tokens, target_output);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    if (target_output.failed_request_ids.empty()) {
      return std::nullopt;
    }
    return make_failed_output(std::move(target_output.failed_request_ids),
                              std::move(target_output.unsafe_request_ids));
  }
  finalize_verify_output(target_output, std::move(val_output), input);
  return target_output;
}

void DFlashWorkerImpl::prepare_json_validation_states(
    const ForwardInput& input,
    const torch::Tensor& draft_tokens_host,
    const StreamEventPtr& draft_tokens_ready,
    ForwardInput& validate_input) {
  CHECK_EQ(draft_tokens_host.dim(), 2);
  CHECK_EQ(draft_tokens_host.size(/*dim=*/1),
           options_.num_speculative_tokens());
  CHECK(draft_tokens_ready != nullptr && draft_tokens_ready->synchronize())
      << "failed to wait for DFlash draft tokens for grammar replay";
  JsonObjectDecodeAdapter::prepare_block_verify_mask(
      input, draft_tokens_host, validate_input, device_);
}

void DFlashWorkerImpl::maybe_broadcast_spec_tokens(torch::Tensor& tokens) {
  if (get_optimization_config().enable_spec_token_broadcast) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    speculative_worker::broadcast_tokens_in_group(
        tokens,
        speculative_worker::sampling_process_group(
            draft_impl_->get_parallel_args()));
  }
}

void DFlashWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "DFlash decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  const bool is_mrope_positions =
      dflash_detail::is_mrope_context_positions(positions_cpu, num_sequences);
  torch::Tensor corrected_mrope_positions;
  if (is_mrope_positions) {
    // Keep the host positions pinned so the subsequent non-blocking H2D copy
    // remains asynchronous on the target validate path.
    corrected_mrope_positions = torch::empty(positions_cpu.sizes(),
                                             torch::TensorOptions()
                                                 .dtype(torch::kInt)
                                                 .device(torch::kCPU)
                                                 .pinned_memory(true));
    corrected_mrope_positions.copy_(positions_cpu);
  }
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "DFlash decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "DFlash decode context position seq_id out of range, seq_id="
        << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    // Rewrite fake input tokens to the last committed real token so KV cache
    // scatter has a valid id; only apply the recorded position offset when the
    // cached state is still valid.
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    const int32_t current_kv_len = calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    // Only text positions can be checked independently against the KV length.
    // MRoPE's raw axis-0 value includes a request-specific delta, so leave the
    // optional empty instead of asserting a tautological equality.
    std::optional<int32_t> current_position;
    if (!is_mrope_positions) {
      current_position = input_positions[seq_id] + position_offset;
    }

    token_ids_vec.emplace_back(rewrite_fake_token ? state.token_id
                                                  : input_token_id);
    if (current_position.has_value()) {
      CHECK_EQ(*current_position + 1, current_kv_len)
          << "DFlash decode context position/kv_len mismatch, seq_id=" << seq_id
          << ", current_position=" << *current_position
          << ", current_kv_len=" << current_kv_len;
      positions_vec.emplace_back(*current_position);
    } else {
      // MRoPE's raw axis-0 value includes a request-specific delta and cannot
      // be compared directly with kv_len. Check the derived draft position
      // instead of asserting a tautological position + 1 == kv_len.
      CHECK_GT(current_kv_len, 0)
          << "DFlash decode requires a positive kv_len for MRoPE, seq_id="
          << seq_id << ", current_kv_len=" << current_kv_len;
      // Preserve all three target axes, including the request's MRoPE delta.
      // Draft-only consumers normalize their local copies from current_kv_len.
      // Decode emits identical positions on all three MRoPE axes. Apply the
      // cache correction to every axis to preserve that equality.
      corrected_mrope_positions.select(/*dim=*/1, seq_id).add_(position_offset);
    }
    append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = make_cpu_int_tensor(token_ids_vec);
  if (is_mrope_positions) {
    input.positions_host = std::move(corrected_mrope_positions);
  } else {
    input.positions_host = make_cpu_int_tensor(positions_vec);
  }
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  // The host fields above supersede the serialized scheduler snapshot.
  // Re-unpacking it would restore the stale tokens, positions and KV lengths
  // the cache correction just replaced.
  input.input_host_buffer_has_layout = false;
  input.device_tensors_ready = false;
}

void DFlashWorkerImpl::prepare_validate_inputs(const ForwardInput& input,
                                               ForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  ForwardInput prepared_input = input;
  prepared_input.metadata_ready_event.reset();
  const bool use_linear_spec_verify = target_is_hybrid_recurrent_;
  prepared_input.input_params.is_spec_verify = use_linear_spec_verify;
  SpeculativeWorkerImpl::prepare_validate_inputs(prepared_input,
                                                 validate_input);
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  if (use_linear_spec_verify) {
    ModelInputParams& input_params = validate_input.input_params;
    if (Platform::uses_compressed_kpool_cache() &&
        impl_->has_request_state_cache()) {
      set_kpool_verify_metadata(
          input_params,
          std::vector<int32_t>(input.input_params.meta.num_sequences,
                               options_.num_speculative_tokens() + 1));
    }
    std::vector<int32_t> accepted_prefix_lengths(
        input.input_params.meta.num_sequences, 1);
    if (embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths = embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          options_.num_speculative_tokens() + 1);
    }
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths,
                      validate_input.token_ids.options().dtype(torch::kInt32));
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens;
      q_cu_seq_lens.reserve(input_params.attention.host.q_seq_lens.size() + 1);
      q_cu_seq_lens.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens.emplace_back(q_cu_seq_lens.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens);
    }
    // The generic builder materializes the device buffer before the hybrid
    // cumulative lengths above are canonicalized.  Rebind it so GDN sees
    // [0, q_len_0, ...] rather than the stale pre-verify host layout.
    input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
    if (options_.enable_graph()) {
      build_dflash_expanded_spec_verify_graph_input(
          input_params, device_.unwrap(), block_size());
    }
#endif
  }
  record_metadata_ready_event(*prepare_stream_, validate_input);
}

void DFlashWorkerImpl::expand_draft_block_tables(
    ModelInputParams& input_params) const {
  const int64_t replicated_block_pages = draft_replicated_block_pages();
  if (replicated_block_pages <= 1) {
    return;
  }
  input_params.attention.host.block_tables =
      layer::expand_replicated_block_table(
          input_params.attention.host.block_tables, replicated_block_pages);
  input_params.attention.device.block_tables =
      layer::expand_replicated_block_table(
          input_params.attention.device.block_tables, replicated_block_pages);
}

bool DFlashWorkerImpl::uses_block_parallel_draft_decode() const {
  return sample_from_anchor() && !Platform::is_mlu() &&
         util::is_deepseek_v4_dspark_model_type(
             draft_impl_->context_.get_model_args().model_type());
}

void DFlashWorkerImpl::prepare_query_inputs(const ForwardInput& input,
                                            ForwardInput& query_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input;
  query_input.device_tensors_ready = false;
  normalize_draft_decode_positions(query_input);
  ModelInputParams& input_params = query_input.input_params;
  // DFlash-family drafts are pure full-attention models. The copied target
  // input may belong to a hybrid model, so discard its recurrent-cache state
  // before rebuilding the draft query geometry.
  dflash_detail::reset_draft_model_metadata(input_params);
  input_params.embedding.input_embedding = torch::Tensor();
  if (!input.json_object_states.empty() ||
      input.sampling_params.filter_bitmask.defined() ||
      input.sampling_params.filter_mask.defined()) {
    // The sequence-level constraint masks (builtin JSON states or the
    // installed XGrammar bitmask) do not model expanded draft positions.
    // Enforce grammar on target verify rows instead, while preserving other
    // sampling masks for unconstrained inputs. Include filter_mask-only rows.
    query_input.sampling_params.filter_bitmask = torch::Tensor();
    query_input.sampling_params.filter_mask = torch::Tensor();
  }
  // The draft addresses its own pool, so both the row builder below and the
  // draft's attention must see this rank's expanded block table. Expand before
  // building the rows: calc_slot_id indexes it with the physical dim.
  expand_draft_block_tables(input_params);

  const bool sample_anchor = sample_from_anchor();
  const int32_t query_width = dflash_detail::decode_draft_width(
      options_.num_speculative_tokens(), sample_anchor);
  const bool use_block_parallel_rows = uses_block_parallel_draft_decode();
  const auto decode_context = make_decode_batch_context(query_input);
  if (use_block_parallel_rows) {
    CHECK(sample_anchor);
    CHECK(decode_context.model_managed_multiblock);
  }
  DecodeExpansionOptions decode_options;
  decode_options.num_speculative_tokens = query_width - 1;
  decode_options.block_size = options_.block_size();
  decode_options.per_request = !use_block_parallel_rows;
  decode_options.causal = false;
  decode_options.mask_token_id = mask_token_id_;
  auto buf = expand_decode_inputs(decode_context, decode_options);

  std::vector<int32_t> selected_idxes;
  selected_idxes.reserve(buf.out_token_ids.size());
  const int32_t first_sample = sample_anchor ? 0 : 1;
  for (int32_t begin = 0;
       begin < static_cast<int32_t>(buf.out_token_ids.size());
       begin += query_width) {
    for (int32_t pos = first_sample; pos < query_width; ++pos) {
      selected_idxes.emplace_back(begin + pos);
    }
  }
  // DFlash emits query_width rows per seq unconditionally, so DP shape
  // symmetry holds by construction. Catch scheduler regressions that break
  // the invariant (see MTP dp_enabled idle-rank branch).
  const int32_t num_sequences_query = input.input_params.meta.num_sequences;
  CHECK_EQ(static_cast<int32_t>(buf.out_positions.size()),
           num_sequences_query * query_width)
      << "DFlash per-seq row count must be uniform (query_width=" << query_width
      << ", num_sequences=" << num_sequences_query << ")";

  set_token_position_tensors(query_input,
                             buf.out_token_ids,
                             buf.out_positions,
                             input.token_ids.options(),
                             input.positions.options());
  input_params.meta.batch_forward_type = draft_batch_forward_type();
  if (use_block_parallel_rows) {
    expand_block_parallel_sequence_rows(input_params, query_width);
  }
  update_decode_attention_metadata(input_params,
                                   buf,
                                   use_block_parallel_rows ? 1 : query_width,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(buf.out_q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/
                                   use_block_parallel_rows);
  update_execution_batch_metadata(
      input_params,
      std::vector<int32_t>(static_cast<size_t>(num_sequences_query),
                           query_width));
  scale_speculative_parallel_token_counts(input_params, query_width);
  input_params.attention.rebuild_device_buffer(device_);

  torch::TensorOptions idx_options =
      torch::TensorOptions().dtype(torch::kInt).device(device_);
  // Pinned-host + async H2D on prepare_stream_ (the file's idiom), so the copy
  // overlaps instead of a blocking non-pinned transfer every decode step.
  query_input.sampling_params.selected_token_idxes =
      safe_to(make_cpu_int_tensor(selected_idxes),
              idx_options,
              /*non_blocking=*/true);
  query_input.sampling_params.sample_idxes =
      torch::arange(static_cast<int64_t>(selected_idxes.size()), idx_options);
  force_greedy_draft_sampling(query_input.sampling_params);
  repeat_sampling_params(query_input.sampling_params,
                         options_.num_speculative_tokens());
  query_input.device_tensors_ready = true;
}

KvPushFailureIds DFlashWorkerImpl::write_context_kv(
    const ForwardInput& input,
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions_device,
    const torch::Tensor& new_cache_slots_device) {
  if constexpr (Platform::requires_auxiliary_kv_push_rejection()) {
    CHECK(options_.kv_cache_transfer_mode() != "PUSH" ||
          input.transfer_kv_infos.empty())
        << "DFlash draft KV PUSH is not implemented for "
        << Platform::type_str();
  }
  CHECK(context_hidden.defined()) << "DFlash context hidden is undefined.";
  CHECK_EQ(context_hidden.dim(), 2) << "DFlash context hidden must be 2D.";
  CHECK_EQ(context_hidden.size(1), expected_context_hidden_size_)
      << "DFlash context hidden size must be hidden_size * "
      << "target_layer_ids.size().";

  CHECK(context_hidden.device() == device_.unwrap())
      << "DFlash context hidden must already be on the compute device.";

  // Both the target forward that produced context_hidden and this pass run
  // on compute_stream_, so no explicit event dance is needed — the stream
  // orders them. Model methods below use torch ops on the same stream.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(input.input_params.synchronize_draft_layer())
      << "Failed to wait for Draft Host KV load completion.";

#if defined(USE_MLU)
  // The target's completed layer events must not be recorded by the draft.
  ModelInputParams draft_input_params = input.input_params;
  draft_input_params.parallel.layer_synchronizer.reset();
  const ModelInputParams& context_input_params = draft_input_params;
#else
  const ModelInputParams& context_input_params = input.input_params;
#endif

#if defined(USE_NPU) || defined(USE_MLU)
  // Draft context KV uses one completion event per draft layer, independently
  // of the target's KV push.
  KVTransferCompletion kv_transfers;
  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            draft_impl_->context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;
#elif defined(USE_MLU)
    std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<MLULayerSynchronizerImpl>(
            draft_impl_->context_.get_model_args().n_layers());
    draft_input_params.parallel.layer_synchronizer = layer_synchronizer;
#endif
    // Cancel hook: aborting the layer synchronizer unblocks a push thread
    // spinning in synchronize_layer, so it stops reading source KV blocks,
    // reports its keys as failed, and completes.
    kv_transfers.add(kv_cache_transfer_->push_kv_blocks_async(
                         input.transfer_kv_infos,
                         draft_impl_->context_.get_parallel_args(),
                         layer_synchronizer,
                         /*is_spec_draft=*/true),
                     unique_transfer_request_ids(input.transfer_kv_infos),
                     [layer_synchronizer] { layer_synchronizer->abort(); });
  }
#endif

  ModelOutput scatter_output =
      draft_impl_->write_context_kv(context_hidden,
                                    positions_device,
                                    new_cache_slots_device,
                                    context_input_params);
  // Model returns an empty ModelOutput (hidden_states undefined) when the
  // per-layer completion event fails to record mid-scatter under
  // PD-PUSH transfer. Fail fast instead of silently continuing: subsequent
  // layers won't have been scattered and the PD transfer side would block
  // forever waiting on the missing per-layer event.
  CHECK(scatter_output.hidden_states.defined())
      << "DFlash context-KV scatter failed (layer_synchronizer record_event "
         "returned false); PD-PUSH transfer would deadlock.";

  // The context-KV scatter is the last device op of the step and is launched
  // no-sync. Under schedule-overlap the scheduler dispatches the next step's
  // host prep as soon as this step's host call returns, while this scatter may
  // still be in flight. Sync so the next step's draft query reads a fully
  // written context-KV cache instead of a partially scattered one.
  compute_stream_->synchronize();

#if defined(USE_NPU) || defined(USE_MLU)
  // Wait for the draft KV push (if any) so the source draft cache is not
  // overwritten by the next step while the transfer is still reading it
  // (mirrors step_internal's wait_kv_push()). No-op when no push was issued.
  return finalize_kv_push_failures(kv_transfers,
                                   input.transfer_kv_infos,
                                   options_.kv_cache_transfer_mode(),
                                   options_.instance_role(),
                                   draft_impl_->context_.get_parallel_args(),
                                   device_);
#else
  return {};
#endif
}

KvPushFailureIds DFlashWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  torch::Tensor accepted_tokens = validate_output.next_tokens;
  CHECK(accepted_tokens.defined()) << "DFlash accepted tokens are undefined.";
  if (accepted_tokens.scalar_type() != torch::kInt64) {
    accepted_tokens = accepted_tokens.to(torch::kInt64);
  }
  DCHECK(accepted_tokens.is_contiguous())
      << "DFlash accepted tokens must be contiguous (guaranteed by upstream "
         ".to(kCPU) and .to(kInt64) branches); check for stride changes if "
         "this fires.";

  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  DecodeBuildBuffers buf;
  // Accepted-context rows use draft-local token positions and the draft's own
  // pool. Keep the shared input unchanged for target verification, and copy it
  // only when MRoPE normalization or replicated block-table expansion is
  // needed.
  const bool needs_position_normalization =
      input.positions_host.defined() &&
      dflash_detail::is_mrope_context_positions(
          input.positions_host, input.input_params.meta.num_sequences);
  const int64_t replicated_block_pages = draft_replicated_block_pages();
  std::optional<ForwardInput> processed_draft_input;
  if (needs_position_normalization || replicated_block_pages > 1) {
    processed_draft_input.emplace(input);
    normalize_draft_decode_positions(*processed_draft_input);
    if (replicated_block_pages > 1) {
      expand_draft_block_tables(processed_draft_input->input_params);
    }
  }
  const ForwardInput& draft_input =
      processed_draft_input.has_value() ? *processed_draft_input : input;
  std::vector<int64_t> accepted_idxes = build_accepted_context_rows(
      draft_input, accepted_tokens, options_.block_size(), buf);
  torch::TensorOptions host_index_options = torch::TensorOptions()
                                                .dtype(torch::kLong)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
  torch::TensorOptions device_index_options =
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_embeddings.device());
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  torch::Tensor accepted_index =
      safe_to(torch::tensor(accepted_idxes, host_index_options),
              device_index_options,
              /*non_blocking=*/true);
  torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, accepted_index);
  torch::Tensor positions_device =
      cpu_int_vec_to_device(buf.out_positions, device_);
  torch::Tensor new_cache_slots_device =
      cpu_int_vec_to_device(buf.out_new_cache_slots, device_);
  // Publish the prepare_stream_ work (index_select producing context_hidden +
  // pinned H2D copies for positions/slots) so compute_stream_ waits for it
  // before the model reads these tensors. Without this, torch does not
  // enforce cross-stream ordering: the write_context_kv scatter could launch
  // before index_select finishes, producing corrupt KV cache. The prefill
  // caller runs its producer on compute_stream_, so no event is needed there.
  StreamEventPtr context_hidden_ready_event =
      prepare_stream_->record_event_or_sync();
  if (context_hidden_ready_event != nullptr) {
    CHECK(compute_stream_->wait_event(context_hidden_ready_event))
        << "failed to wait DFlash context hidden ready event";
  }
  KvPushFailureIds failures = write_context_kv(
      draft_input, context_hidden, positions_device, new_cache_slots_device);
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "DFlash target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens());
  return failures;
}

}  // namespace xllm
