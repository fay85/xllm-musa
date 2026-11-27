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

#include "runtime/dspark_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/model/model_sampling_input.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/speculative_worker_utils.h"
#include "util/timer.h"

namespace xllm {

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   WorkerType worker_type)
    : DFlashWorkerImpl(parallel_args, device, options, worker_type),
      sampling_process_group_(speculative_worker::sampling_process_group(
          draft_impl_->get_parallel_args())) {}

DSparkWorkerImpl::~DSparkWorkerImpl() { retire_async_output(); }

void DSparkWorkerImpl::on_draft_model_loaded(
    const std::string& model_weights_path) {
  DFlashWorkerImpl::on_draft_model_loaded(model_weights_path);
  async_output_enabled_ = enable_schedule_overlap();
  const bool supports_device_inputs =
      draft_impl_->supports_device_only_decode_inputs();
  draft_prelaunch_enabled_for_init_ =
      can_prelaunch_draft_for_init(supports_device_inputs);
  draft_impl_->set_device_only_decode_inputs(draft_prelaunch_enabled_for_init_);
  LOG(INFO) << "DSpark async output enabled=" << async_output_enabled_;
  LOG(INFO) << "DSpark device-only decode inputs supported="
            << supports_device_inputs << ", draft prelaunch enabled="
            << draft_prelaunch_enabled_for_init_;
}

bool DSparkWorkerImpl::can_prelaunch_draft_for_init(
    bool supports_device_inputs) const {
  const bool draft_prelaunch_pending_support_gate =
      parallel_args_.cp_size() == 1 &&
      parallel_args_.kv_split_size_effective() == 1;
  return async_output_enabled_ && draft_prelaunch_pending_support_gate &&
         !options_.enable_probabilistic_draft() && supports_device_inputs;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;
  std::vector<int64_t> pending_rows;
  if (can_prepare_without_compute_stream_wait(input)) {
    pending_rows = build_batch_to_pending_row_mapping(input);
  }

  auto draft_output = [&] {
    if (!pending_rows.empty()) {
      auto draft = run_decode_draft_from_pending(input, pending_rows);
      commit_pending_context();
      prepare_decode_input(input);
      prepare_validate_inputs(input, validate_input);
      return draft;
    }
    commit_pending_context();
    prepare_decode_input(input);
    return run_decode_draft(input, validate_input);
  }();

  const bool async_output =
      async_output_enabled_ && !input.input_params.meta.is_graph_warmup;
  return run_validate(input, validate_input, draft_output, async_output);
}

ForwardOutput DSparkWorkerImpl::run_decode_draft(const ForwardInput& input,
                                                 ForwardInput& validate_input) {
  // Same input build as DFlash, but sample_from_anchor()==true makes the query
  // block N-wide and every position predicts a draft token.
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  query_input.input_params.use_dflash_proposal_attention = true;

  return execute_draft_query(input,
                             std::move(query_input),
                             /*prepared=*/false,
                             &validate_input);
}

ForwardOutput DSparkWorkerImpl::execute_draft_query(
    const ForwardInput& input,
    ForwardInput query_input,
    bool prepared,
    ForwardInput* validate_input) {
  Timer timer;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  query_input.skip_sampling_for_logits_only = true;

  ForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query_input);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query_input, processed_input, *prepare_stream_);
  }
  // Take the draft sampling params before any move: clear the sequence-level
  // constraint mask from input.sampling_params (it does not model the expanded
  // draft positions) while keeping every other setting.
  const SamplingParameters query_sampling_params =
      dflash_detail::make_draft_sampling_params(input.sampling_params);
  ModelSamplingInput sampling;
  sampling.parameters = query_sampling_params;
  if (!options_.enable_probabilistic_draft()) {
    force_greedy_draft_sampling(sampling.parameters);
  }
  sampling.return_draft_logits =
      options_.enable_probabilistic_draft() &&
      sampling.parameters.sampling_mode != SamplingMode::GREEDY;
  processed_input.input_params.sampling = std::move(sampling);

  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             *compute_stream_,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value())
      << "DSpark draft forward must return an output.";
  if (validate_input != nullptr) {
    prepare_validate_inputs(input, *validate_input);
  }

  if (!draft_output->draft_token_ids.defined()) {
    torch::Tensor base_logits = draft_output->logits.view(
        {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
    // H2D the sampling params on prepare_stream_ (overlapping the draft
    // forward) instead of on compute_stream_ after the forward is submitted.
    // Prepared path: skip the prepare_stream_ H2D here (its prepare_stream
    // section has already been recorded and waited on by the caller); the
    // params stay host-side, but after the index-tensor clearing below the
    // .to() payload is scalars and empty tensors only, so the one
    // compute-stream conversion is negligible.
    StreamEventPtr sampling_params_ready;
    if (!prepared) {
      c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
      // Store the converted params on processed_input so they outlive
      // execute_draft_query: a local would be destroyed on return while
      // compute_stream_ is still consuming them (non_blocking H2D).
      processed_input.input_params.sampling->parameters =
          query_sampling_params.to(draft_impl_->device(), torch::kFloat32);
      // Record the H2D so compute_stream_ only consumes the params after the
      // copy has completed (ordering across the two streams).
      sampling_params_ready = prepare_stream_->record_event();
      CHECK(sampling_params_ready != nullptr);
    }

    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Wait for the prepare_stream_ H2D before consuming the params.
    if (sampling_params_ready != nullptr) {
      CHECK(compute_stream_->wait_event(sampling_params_ready));
    }
    torch::Tensor anchor_token_ids =
        processed_input.token_ids.view({batch_size, num_speculative_tokens})
            .select(/*dim=*/1, /*index=*/0)
            .to(device_, /*dtype=*/torch::kLong);
    // Drop the index tensors first: sample_block clears them in its own
    // local copy anyway, so converting them here would just waste a
    // blocking H2D on data that gets discarded.
    processed_input.input_params.sampling->parameters.selected_token_idxes =
        torch::Tensor();
    processed_input.input_params.sampling->parameters.sample_idxes =
        torch::Tensor();
    SamplingParameters typed_params =
        processed_input.input_params.sampling->parameters.to(
            base_logits.device(), base_logits.scalar_type());
    if (!options_.enable_probabilistic_draft()) {
      force_greedy_draft_sampling(typed_params);
    }
    auto [draft_token_ids, draft_logits] =
        sample_block(base_logits, anchor_token_ids, typed_params);
    draft_output->draft_token_ids = std::move(draft_token_ids);
    draft_output->draft_logits = std::move(draft_logits);
    draft_output->logits.reset();
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return std::move(*draft_output);
}

bool DSparkWorkerImpl::can_prepare_without_compute_stream_wait(
    const ForwardInput& input) const {
  // Constraint masks never block prelaunch: the draft query clears the
  // constraint bitmask (drafts sample unconstrained) and the verify rows are
  // rebuilt from this step's fresh states, both inside prepare_query_inputs
  // and the validate path.
  if (!draft_prelaunch_enabled_for_init_ ||
      input.input_params.meta.is_graph_warmup ||
      !input.input_params.meta.batch_forward_type.is_decode() ||
      input.input_params.meta.num_sequences <= 0 ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size()) {
    return false;
  }
  const auto& block_copy = input.input_params.block_copy;
  if (!block_copy.swap_blocks.empty() ||
      (block_copy.src_block_indices.defined() &&
       block_copy.src_block_indices.numel() > 0)) {
    return false;
  }
  const auto& sampling = input.sampling_params;
  return sampling.sampling_mode == SamplingMode::GREEDY &&
         !sampling.unique_token_ids.defined() &&
         !sampling.frequency_penalties.defined() &&
         !sampling.presence_penalties.defined() &&
         !sampling.repetition_penalties.defined();
}

ForwardOutput DSparkWorkerImpl::run_decode_draft_from_pending(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.use_dflash_proposal_attention = true;
  query.skip_sampling_for_logits_only = true;
  query.return_selected_hidden = false;
  ForwardInput prepared;
  draft_impl_->prepare_work_before_execute_on_stream(
      query, prepared, *prepare_stream_, true, false);
  dflash_async::DraftCacheBlockTable block_table;
  torch::Tensor row_indices;
  torch::Tensor host_row_indices;
  const bool block_parallel_rows = uses_block_parallel_draft_decode();
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    // A block-parallel proposal exports one metadata row per query token.
    // Device preparation consumes one ring table per logical sequence.
    std::vector<torch::Tensor> tables =
        prepared.input_params.multi_block_tables;
    if (block_parallel_rows) {
      const int32_t width = options_.num_speculative_tokens();
      for (torch::Tensor& table : tables) {
        table = table.slice(/*dim=*/0, /*start=*/0, table.size(0), width);
      }
    }
    block_table = dflash_async::prepare_draft_cache_block_table(
        prepared.input_params.attention.device.block_tables,
        tables,
        input.input_params.meta.num_sequences,
        device_);
    if (block_parallel_rows) {
      // Reuse the staged SWA table for Python graph refresh. A CPU table
      // would introduce a blocking H2D copy on the compute stream while
      // submitting the proposal, serializing it behind the target verify.
      CHECK(!prepared.input_params.multi_block_tables.empty());
      prepared.input_params.multi_block_tables.front() =
          block_table.blocks.repeat_interleave(
              options_.num_speculative_tokens(),
              /*dim=*/0);
    }
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
  dflash_async::NextDraftInputs next =
      dflash_async::prepare_next_draft(accepted,
                                       base_positions,
                                       block_table,
                                       mask_token_id_,
                                       options_.block_size(),
                                       options_.num_speculative_tokens());
  prepared.token_ids.copy_(next.query_tokens);
  prepared.positions.copy_(next.query_positions);
  prepared.input_params.attention.device.new_cache_slots.copy_(
      next.query_slots);
  if (block_parallel_rows) {
    // Match build_query_rows: every query row sees the whole block.
    prepared.input_params.attention.device.kv_seq_lens.copy_(
        next.kv_lengths.repeat_interleave(options_.num_speculative_tokens()));
  } else {
    prepared.input_params.attention.device.kv_seq_lens.copy_(next.kv_lengths);
  }
  prepared.input_params.attn_metadata.reset();
  prepared.metadata_ready_event.reset();
  prepared.device_tensors_ready = true;
  prepared.retained_device_tensors = {std::move(block_table.blocks),
                                      std::move(row_indices),
                                      std::move(host_row_indices),
                                      std::move(accepted),
                                      std::move(base_positions),
                                      std::move(next.anchor_tokens),
                                      pending_context_->accepted_device,
                                      pending_context_->base_positions};
  auto draft =
      execute_draft_query(input, std::move(prepared), /*prepared=*/true);
  VLOG(1) << "DSpark step-local draft submitted: batch_id="
          << input.input_params.meta.batch_id << ", sequences=" << rows.size();
  return draft;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_prefill(
    const ForwardInput& input) {
  commit_pending_context();
  std::optional<ForwardOutput> output = DFlashWorkerImpl::step_prefill(input);
  consumed_context_tensors_.clear();
  return output;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_empty(
    const ForwardInput& input) {
  commit_pending_context();
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    return DFlashWorkerImpl::step_empty(input);
  }
  ForwardInput empty_input = input;
  empty_input.input_params.use_dflash_proposal_attention = true;
  ModelSamplingInput sampling;
  force_greedy_draft_sampling(sampling.parameters);
  empty_input.input_params.sampling = std::move(sampling);
  return DFlashWorkerImpl::step_empty(empty_input);
}

std::pair<torch::Tensor, torch::Tensor> DSparkWorkerImpl::sample_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling_params) const {
  CHECK_EQ(base_logits.dim(), 3)
      << "DSpark base_logits must be [num_reqs, n_spec, draft_vocab].";
  CHECK_EQ(anchor_token_ids.dim(), 1)
      << "DSpark anchor_token_ids must be [num_reqs].";
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0))
      << "DSpark anchor token batch must match base_logits.";
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kLong)
      << "DSpark anchor_token_ids must use int64.";
  CHECK_EQ(anchor_token_ids.device(), base_logits.device())
      << "DSpark anchor_token_ids and base_logits must share a device.";

  const int64_t num_reqs = base_logits.size(/*dim=*/0);
  const int64_t num_speculative_tokens = base_logits.size(/*dim=*/1);
  const int64_t draft_vocab_size = base_logits.size(/*dim=*/2);
  const bool return_draft_logits =
      options_.enable_probabilistic_draft() &&
      sampling_params.sampling_mode != SamplingMode::GREEDY;
  SamplingParameters step_sampling_params = sampling_params;
  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt).device(base_logits.device());
  step_sampling_params.selected_token_idxes = torch::empty({0}, index_options);
  step_sampling_params.sample_idxes = torch::empty({0}, index_options);
  step_sampling_params.logprobs = false;
  step_sampling_params.max_top_logprobs = 0;
  step_sampling_params.use_beam_search = false;

  torch::Tensor token_ids = torch::empty(
      {num_reqs, num_speculative_tokens},
      torch::TensorOptions().dtype(torch::kLong).device(base_logits.device()));
  std::vector<torch::Tensor> draft_logits_steps;
  if (return_draft_logits) {
    draft_logits_steps.reserve(num_speculative_tokens);
  }
  using ISlice = torch::indexing::Slice;
  torch::Tensor previous_token_ids = anchor_token_ids;
  for (int64_t token_idx = 0; token_idx < num_speculative_tokens; ++token_idx) {
    torch::Tensor markov_bias =
        draft_impl_->dspark_markov_bias(previous_token_ids);
    CHECK_EQ(markov_bias.dim(), 2)
        << "DSpark Markov bias must be [num_reqs, draft_vocab].";
    CHECK_EQ(markov_bias.size(0), num_reqs)
        << "DSpark Markov bias batch must match base_logits.";
    CHECK_EQ(markov_bias.size(1), draft_vocab_size)
        << "DSpark Markov bias width must equal the draft vocabulary.";

    torch::Tensor step_logits =
        base_logits.select(/*dim=*/1, /*index=*/token_idx) + markov_bias;
    SampleOutput sample_output = sample(step_logits, step_sampling_params);
    synchronize_sampled_token_ids(sample_output.next_tokens,
                                  step_sampling_params);
    // Proposals and Markov feedback both use target token ids.
    if (return_draft_logits) {
      sample_output.draft_logits = std::move(step_logits);
    }
    draft_vocab_.remap(sample_output);
    token_ids.index_put_({ISlice(), token_idx}, sample_output.next_tokens);
    if (return_draft_logits) {
      draft_logits_steps.emplace_back(std::move(sample_output.draft_logits));
    }
    previous_token_ids = std::move(sample_output.next_tokens);
  }

  return {std::move(token_ids),
          return_draft_logits ? torch::stack(draft_logits_steps, /*dim=*/1)
                              : torch::Tensor()};
}

void DSparkWorkerImpl::synchronize_sampled_token_ids(
    torch::Tensor& sampled_token_ids,
    const SamplingParameters& sampling_params) const {
  if (sampling_params.sampling_mode == SamplingMode::GREEDY ||
      sampling_process_group_ == nullptr ||
      sampling_process_group_->world_size() <= 1) {
    return;
  }
  sampled_token_ids = sampled_token_ids.contiguous();
  sampling_process_group_->broadcast(sampled_token_ids, /*root_rank=*/0);
}

}  // namespace xllm
