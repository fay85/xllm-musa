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

#include "runtime/dflash/dflash2_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>

#include "common/metrics.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "framework/sampling/gumbel_sampling.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"
#include "runtime/speculative_worker_utils.h"
#include "util/timer.h"

namespace xllm {

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options,
                                     WorkerType worker_type)
    : DFlashWorkerImpl(parallel_args, device, options, worker_type),
      sampling_process_group_(
          speculative_worker::sampling_process_group(parallel_args)) {
  async_output_enabled_ = enable_schedule_overlap() &&
                          !options_.enable_adaptive_speculative_decode();
  proposal_xfia_enabled_ =
      enable_dflash_proposal_xfia(parallel_args, device, options);
  draft_prelaunch_enabled_for_init_ =
      async_output_enabled_ && proposal_xfia_enabled_ &&
      parallel_args_.kv_split_size_effective() == 1;
}

DFlash2WorkerImpl::~DFlash2WorkerImpl() { retire_async_output(); }

std::optional<ForwardOutput> DFlash2WorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;
  const bool can_prelaunch = can_prelaunch_draft_for_this_step(input);
  std::vector<int64_t> pending_rows;
  if (can_prelaunch) {
    pending_rows = build_batch_to_pending_row_mapping(input);
  }
  DraftBlock draft_block;
  if (!pending_rows.empty()) {
    draft_block = run_decode_draft_from_pending(input, pending_rows);
    commit_pending_context();
    prepare_decode_input(input);
    prepare_validate_inputs(input, validate_input);
  } else {
    commit_pending_context();
    prepare_decode_input(input);
    draft_block = run_decode_draft(input, validate_input);
  }
  if (can_prelaunch) {
    prepare_target_metadata(validate_input);
  }
  const bool async_output = async_output_enabled_ &&
                            !input.input_params.meta.is_graph_warmup &&
                            input.transfer_kv_infos.empty();
  auto output = run_validate(input, draft_block, validate_input, async_output);
  if (!async_output) {
    consumed_context_tensors_.clear();
  }
  return output;
}

void DFlash2WorkerImpl::prepare_target_metadata(ForwardInput& validate_input) {
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

std::optional<ForwardOutput> DFlash2WorkerImpl::step_prefill(
    const ForwardInput& input) {
  commit_pending_context();
  std::optional<ForwardOutput> output = DFlashWorkerImpl::step_prefill(input);
  consumed_context_tensors_.clear();
  return output;
}

std::optional<ForwardOutput> DFlash2WorkerImpl::step_empty(
    const ForwardInput& input) {
  commit_pending_context();
  ForwardInput empty_input = input;
  empty_input.input_params.use_dflash_proposal_attention =
      proposal_xfia_enabled_ && !input.input_params.meta.is_graph_warmup &&
      input.input_params.meta.batch_forward_type.is_decode();
  return DFlashWorkerImpl::step_empty(empty_input);
}

bool DFlash2WorkerImpl::can_prepare_without_compute_stream_wait(
    const ForwardInput& input) const {
  return can_prelaunch_draft_for_this_step(input);
}

bool DFlash2WorkerImpl::can_prelaunch_draft_for_this_step(
    const ForwardInput& input) const {
  if (!draft_prelaunch_enabled_for_init_ ||
      !can_prepare_device_draft_inputs(input)) {
    return false;
  }
  const auto rows = specBuilder::make_decode_row_context(input);
  const int32_t max_query_tokens_with_pending_acceptance =
      2 * (options_.num_speculative_tokens() + 1);
  return specBuilder::has_decode_lookahead_capacity(
      rows,
      max_query_tokens_with_pending_acceptance,
      options_.block_size(),
      draft_impl_->context_.get_model_args().max_position_embeddings());
}

bool DFlash2WorkerImpl::can_prepare_device_draft_inputs(
    const ForwardInput& input) const {
  if (!proposal_xfia_enabled_ || input.input_params.meta.is_graph_warmup ||
      !input.input_params.meta.batch_forward_type.is_decode() ||
      input.input_params.meta.num_sequences <= 0 ||
      !input.input_params.multi_block_tables.empty() ||
      input.input_params.meta.requires_host_restore ||
      !input.input_params.block_copy.swap_blocks.empty() ||
      (input.input_params.block_copy.src_block_indices.defined() &&
       input.input_params.block_copy.src_block_indices.numel() > 0) ||
      !input.transfer_kv_infos.empty() ||
      !constraint_states_allow_prelaunch(input) ||
      input.input_params.embedding.mtp_bootstrap_embeddings.defined() ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size()) {
    return false;
  }
  const auto rows = specBuilder::make_decode_row_context(input);
  return !rows.model_managed_multiblock &&
         specBuilder::has_decode_lookahead_capacity(
             rows,
             options_.num_speculative_tokens() + 1,
             options_.block_size(),
             draft_impl_->context_.get_model_args().max_position_embeddings());
}

torch::Tensor DFlash2WorkerImpl::prepare_draft_noise(
    const ForwardInput& input) const {
  torch::Tensor noise = sample_gumbel_noise(
      input.input_params.meta.num_sequences,
      options_.num_speculative_tokens(),
      draft_impl_->context_.get_model_args().dflash2_selector_top_k(),
      input.sampling_params,
      draft_impl_->device());
  return noise;
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.use_dflash_proposal_attention =
      proposal_xfia_enabled_ && !input.input_params.meta.is_graph_warmup;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_GT(batch_size, 0);
  CHECK(input.token_ids_host.defined());
  CHECK_GE(input.token_ids_host.numel(), batch_size);
  torch::Tensor anchor_tokens;
  torch::Tensor noise;
  ForwardInput device_input = input;
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    anchor_tokens = input.token_ids_host.slice(0, 0, batch_size)
                        .to(draft_impl_->device(), torch::kLong);
    noise = prepare_draft_noise(input);
    device_input.sampling_params =
        input.sampling_params.to(draft_impl_->device(), torch::kFloat32);
    query.metadata_ready_event = prepare_stream_->record_event_or_sync();
  }
  DraftBlock block = execute_draft_query(
      device_input, std::move(query), anchor_tokens, noise, false);
  block.retained_tensors.reserve(block.retained_tensors.size() + 2);
  block.retained_inputs.reserve(block.retained_inputs.size() + 1);
  block.retained_tensors.emplace_back(std::move(anchor_tokens));
  block.retained_tensors.emplace_back(std::move(noise));
  block.retained_inputs.emplace_back(
      std::make_shared<ForwardInput>(std::move(device_input)));
  prepare_validate_inputs(input, validate_input);
  return block;
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::run_decode_draft_from_pending(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.use_dflash_proposal_attention = true;
  query.skip_sampling_for_logits_only = true;
  query.return_selected_hidden = true;
  ForwardInput prepared;
  draft_impl_->prepare_work_before_execute_on_stream(
      query, prepared, *prepare_stream_, true, false);
  ForwardInput device_input = input;
  torch::Tensor row_indices;
  torch::Tensor host_row_indices;
  torch::Tensor noise;
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    device_input.sampling_params =
        input.sampling_params.to(draft_impl_->device(), torch::kFloat32);
    noise = prepare_draft_noise(input);
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
  DraftBlock block = execute_draft_query(
      device_input, prepared, next.anchor_tokens, noise, true);
  block.retained_inputs.reserve(block.retained_inputs.size() + 2);
  block.retained_inputs.emplace_back(
      std::make_shared<ForwardInput>(std::move(prepared)));
  block.retained_inputs.emplace_back(
      std::make_shared<ForwardInput>(std::move(device_input)));
  block.retained_tensors.reserve(block.retained_tensors.size() + 8);
  block.retained_tensors.insert(block.retained_tensors.end(),
                                {row_indices,
                                 host_row_indices,
                                 noise,
                                 accepted,
                                 base_positions,
                                 next.anchor_tokens,
                                 pending_context_->accepted_device,
                                 pending_context_->base_positions});
  return block;
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::execute_draft_query(
    const ForwardInput& input,
    ForwardInput query_input,
    const torch::Tensor& anchor_token_ids,
    const torch::Tensor& gumbel_noise,
    bool prepared) {
  Stream& stream = *compute_stream_;
  Timer timer;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  query_input.skip_sampling_for_logits_only = true;
  query_input.return_selected_hidden = true;
  ForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query_input);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query_input,
        processed_input,
        *prepare_stream_,
        /*record_ready_event=*/prepare_stream_.get() != compute_stream_.get());
  }
  ModelSamplingInput sampling;
  sampling.parameters = input.sampling_params;
  sampling.parameters.return_probs = true;
  sampling.gumbel_noise = gumbel_noise;
  processed_input.input_params.sampling = std::move(sampling);
  draft_impl_->set_hierarchy_layer_synchronizer(processed_input.input_params);
  if (!input.sampling_params.all_greedy_sample &&
      sampling_process_group_ != nullptr &&
      sampling_process_group_->world_size() > 1) {
    c10::StreamGuard stream_guard = stream.set_stream_guard();
    speculative_worker::wait_metadata_ready_event(processed_input, stream);
    torch::Tensor noise = gumbel_noise;
    sampling_process_group_->broadcast(noise, 0);
  }
  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             stream,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value());
  if (draft_output->draft_proposal.has_value()) {
    DraftBlock block;
    block.proposal = std::move(*draft_output->draft_proposal);
    CHECK_EQ(block.proposal.token_ids().sizes(),
             torch::IntArrayRef({batch_size, num_speculative_tokens}));
    CHECK(block.proposal.draft_probs().has_value());
    block.retained_inputs = take_retained_inputs(*draft_output);
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
    return block;
  }
  CHECK(draft_output->logits.defined());
  CHECK(draft_output->selected_hidden.defined());
  const int64_t num_rows = draft_output->logits.size(0);
  CHECK_EQ(num_rows, static_cast<int64_t>(batch_size) * num_speculative_tokens);
  torch::Tensor unary_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  torch::Tensor hidden_states = draft_output->selected_hidden.view(
      {batch_size,
       num_speculative_tokens,
       draft_output->selected_hidden.size(-1)});

  BlockSampleOutput sampled;
  {
    c10::StreamGuard stream_guard = stream.set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    SamplingParameters sampling_params = input.sampling_params.to(
        unary_logits.device(), unary_logits.scalar_type());
    sampled = sample_path(candidates,
                          sampling_params,
                          gumbel_noise,
                          unary_logits.size(/*dim=*/-1));
  }

  DraftBlock draft_block;
  // DFlash2 samples selector paths from a sparse top-k distribution; the
  // dense per-token proposal must be retained so rejection recovery stays
  // exact. The selected-only probs carry no extra information for the
  // verifier, so only token_ids and the dense proposal feed the DraftProposal.
  draft_block.proposal = DraftProposal(std::move(sampled.token_ids),
                                       std::move(sampled.dense_probs));
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

DFlash2WorkerImpl::BlockSampleOutput DFlash2WorkerImpl::sample_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size) const {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k}));
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // Pre-compute log_softmax(edge_logits / temperature) for every step at
  // once.  The previous implementation re-scaled and re-normalized the
  // gathered row inside each step through the generic sampler, which lowered
  // to an AICPU multinomial plus a per-step consensus broadcast and left the
  // compute stream idle between steps.  Gumbel-max over the same
  // log-probabilities draws from exactly the same distribution:
  //   argmax(log_softmax(logits / T) + g) ~ categorical(softmax(logits / T))
  // and greedy rows (zeroed noise) collapse to plain argmax.  Target-side
  // truncation, penalties, and grammar constraints are applied by
  // verification and must not be applied a second time to the selector's
  // top-k candidate distribution.
  torch::Tensor edge_log_probs = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_log_probs, sampling_params.temperatures, batch_size);
  }
  edge_log_probs = torch::log_softmax(edge_log_probs, /*dim=*/-1);

  if (auto sampled = fused_sample(candidates.candidate_ids,
                                  edge_log_probs,
                                  gumbel_noise,
                                  sampling_params,
                                  vocab_size)) {
    return std::move(*sampled);
  }

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor candidate_probs =
      torch::empty({batch_size, num_steps, top_k}, float_options);
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());
  torch::Tensor batch_offsets =
      torch::arange(batch_size, candidates.candidate_ids.options()) *
      (num_steps * top_k);
  // Flatten all batch/step/predecessor rows once. Selecting a step before
  // reshaping would copy a noncontiguous [B, K, K] slice when B > 1.
  torch::Tensor edge_rows = edge_log_probs.view({-1, top_k});

  using ISlice = torch::indexing::Slice;
  for (int64_t step = 0; step < num_steps; ++step) {
    // NPU gather on the middle dimension of a 3-D tensor transposes both
    // inputs and its output. A complete-row lookup preserves the same values
    // and ordering without those per-step layout conversions.
    torch::Tensor row_indices =
        previous_indices + (batch_offsets + step * top_k);
    torch::Tensor row_log_probs =
        edge_rows.index_select(/*dim=*/0, row_indices);
    torch::Tensor sampled_indices = gumbel_argmax(
        row_log_probs, gumbel_noise.select(/*dim=*/1, /*index=*/step));

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    // Keep the per-step proposal distribution semantics identical to the
    // previous sampler path: random rows expose the full softmax over the
    // top-k candidates, deterministic rows expose a one-hot at the argmax,
    // and mixed batches select per row via do_sample. The exp() lives inside
    // the consuming branches so greedy-only batches skip it.
    torch::Tensor step_probs;
    if (sampling_params.all_random_sample) {
      step_probs = row_log_probs.exp();
    } else {
      torch::Tensor greedy_probs =
          torch::zeros({batch_size, top_k}, float_options);
      greedy_probs.scatter_(/*dim=*/1,
                            sampled_indices.view({-1, 1}),
                            /*value=*/1.0);
      if (sampling_params.all_greedy_sample) {
        step_probs = greedy_probs;
      } else {
        step_probs =
            torch::where(sampling_params.do_sample.view({batch_size, 1}),
                         row_log_probs.exp(),
                         greedy_probs);
      }
    }
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    candidate_probs.index_put_({ISlice(), step, ISlice()}, step_probs);
    previous_indices = sampled_indices;
  }

  torch::Tensor dense_probs =
      torch::zeros({batch_size, num_steps, vocab_size}, float_options);
  dense_probs.scatter_(
      /*dim=*/-1, candidates.candidate_ids, candidate_probs);
  return {.token_ids = std::move(token_ids),
          .dense_probs = std::move(dense_probs)};
}

}  // namespace xllm
