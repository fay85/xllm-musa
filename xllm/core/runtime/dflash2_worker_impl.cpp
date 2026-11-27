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

#include "core/runtime/dflash2_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "core/common/metrics.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/runtime/speculative_worker_utils.h"
#include "core/util/timer.h"

namespace xllm {

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options,
                                     WorkerType worker_type)
    : DFlashWorkerImpl(parallel_args, device, options, worker_type),
      sampling_process_group_(speculative_worker::sampling_process_group(
          draft_impl_->get_parallel_args())) {
  async_output_enabled_ = enable_schedule_overlap();
  draft_prelaunch_enabled_for_init_ =
      async_output_enabled_ && proposal_xfia_enabled_;
}

torch::Tensor DFlash2WorkerImpl::prepare_draft_noise(
    const ForwardInput& input) const {
  auto sampling_params =
      dflash_detail::make_draft_sampling_params(input.sampling_params);
  if (!options_.enable_probabilistic_draft()) {
    force_greedy_draft_sampling(sampling_params);
  }
  auto noise = gumbel_noise(
      sampling_params,
      {input.input_params.meta.num_sequences,
       options_.num_speculative_tokens(),
       draft_impl_->context_.get_model_args().dflash2_selector_top_k()},
      draft_impl_->device());
  if (sampling_params.sampling_mode != SamplingMode::GREEDY) {
    noise.neg_();
  }
  return noise;
}

ForwardOutput DFlash2WorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.use_dflash_proposal_attention =
      proposal_attention_enabled_ && !input.input_params.meta.is_graph_warmup;
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
  auto draft = execute_draft_query(device_input,
                                   std::move(query),
                                   anchor_tokens,
                                   noise,
                                   /*prepared=*/false);
  device_input.retained_device_tensors.emplace_back(std::move(anchor_tokens));
  device_input.retained_device_tensors.emplace_back(std::move(noise));
  draft.retained_inputs.emplace_back(
      std::make_shared<ForwardInput>(std::move(device_input)));
  prepare_validate_inputs(input, validate_input);
  return draft;
}

ForwardOutput DFlash2WorkerImpl::run_decode_draft_from_pending(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  ForwardInput device_input = input;
  torch::Tensor noise;
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    speculative_worker::wait_metadata_ready_event(input, *prepare_stream_);
    device_input.sampling_params =
        input.sampling_params.to(draft_impl_->device(), torch::kFloat32);
    noise = prepare_draft_noise(input);
  }
  auto [prepared, anchor_tokens] = prepare_pending_draft_query(input, rows);
  auto draft = execute_draft_query(device_input,
                                   std::move(prepared),
                                   anchor_tokens,
                                   noise,
                                   /*prepared=*/true);
  device_input.retained_device_tensors.emplace_back(std::move(anchor_tokens));
  device_input.retained_device_tensors.emplace_back(std::move(noise));
  draft.retained_inputs.emplace_back(
      std::make_shared<ForwardInput>(std::move(device_input)));
  return draft;
}

ForwardOutput DFlash2WorkerImpl::execute_draft_query(
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
  sampling.parameters =
      dflash_detail::make_draft_sampling_params(input.sampling_params);
  if (!options_.enable_probabilistic_draft()) {
    force_greedy_draft_sampling(sampling.parameters);
  }
  sampling.gumbel_noise = gumbel_noise;
  processed_input.input_params.sampling = std::move(sampling);
  // The draft input derives from the target input that step_async's guarded
  // set_hierarchy_layer_synchronizer already processed, so
  // requires_host_restore is cleared (or the promise failed before this
  // query was built): the missing-handle throw cannot fire here. The call
  // still installs the draft's own host-restore synchronizer when one was
  // registered for the draft transfer.
  draft_impl_->set_hierarchy_layer_synchronizer(processed_input.input_params);
  if (options_.enable_probabilistic_draft() &&
      input.sampling_params.sampling_mode != SamplingMode::GREEDY &&
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
  if (draft_output->draft_token_ids.defined()) {
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
    return std::move(*draft_output);
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

  auto [draft_token_ids, draft_logits] = [&] {
    c10::StreamGuard stream_guard = stream.set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    // Drafts sample unconstrained: the sequence-level constraint mask does
    // not model the expanded draft positions, and the verify rows enforce the
    // grammar per position. Clear it (keeping every other sampling setting)
    // instead of sampling under a wrongly-shaped mask.
    SamplingParameters sampling_params =
        dflash_detail::make_draft_sampling_params(input.sampling_params)
            .to(unary_logits.device(), unary_logits.scalar_type());
    if (!options_.enable_probabilistic_draft()) {
      force_greedy_draft_sampling(sampling_params);
    }
    return sample_path(candidates,
                       sampling_params,
                       gumbel_noise,
                       unary_logits.size(/*dim=*/-1));
  }();

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  draft_output->draft_token_ids = std::move(draft_token_ids);
  draft_output->draft_logits = std::move(draft_logits);
  draft_output->logits.reset();
  draft_output->selected_hidden.reset();
  return std::move(*draft_output);
}

std::pair<torch::Tensor, torch::Tensor> DFlash2WorkerImpl::sample_path(
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

  torch::Tensor edge_logits =
      candidates.edge_logits.to(/*dtype=*/torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_temperatures(edge_logits, sampling_params.temperatures);
  }
  if (auto sampled = fused_sample(candidates.candidate_ids,
                                  edge_logits,
                                  gumbel_noise,
                                  sampling_params,
                                  vocab_size)) {
    return std::move(*sampled);
  }

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  const bool return_draft_logits =
      sampling_params.sampling_mode != SamplingMode::GREEDY;
  torch::Tensor candidate_logits;
  if (return_draft_logits) {
    candidate_logits =
        torch::empty({batch_size, num_steps, top_k}, float_options);
  }
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());
  torch::Tensor batch_offsets =
      torch::arange(batch_size, candidates.candidate_ids.options()) *
      (num_steps * top_k);
  // Flatten all batch/step/predecessor rows once. Selecting a step before
  // reshaping would copy a noncontiguous [B, K, K] slice when B > 1.
  torch::Tensor edge_rows = edge_logits.view({-1, top_k});

  using ISlice = torch::indexing::Slice;
  for (int64_t step = 0; step < num_steps; ++step) {
    // NPU gather on the middle dimension of a 3-D tensor transposes both
    // inputs and its output. A complete-row lookup preserves the same values
    // and ordering without those per-step layout conversions.
    torch::Tensor row_indices =
        previous_indices + (batch_offsets + step * top_k);
    torch::Tensor transition_logits =
        edge_rows.index_select(/*dim=*/0, row_indices);
    torch::Tensor sampled_indices =
        (transition_logits + gumbel_noise.select(/*dim=*/1, /*index=*/step))
            .argmax(/*dim=*/-1);

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    if (return_draft_logits) {
      if (sampling_params.sampling_mode == SamplingMode::MIXED) {
        torch::Tensor greedy_logits = torch::full_like(
            transition_logits,
            /*fill_value=*/-std::numeric_limits<float>::infinity());
        greedy_logits.scatter_(
            /*dim=*/1, sampled_indices.unsqueeze(/*dim=*/1), /*value=*/0.0);
        transition_logits =
            torch::where(sampling_params.do_sample.view({batch_size, 1}),
                         transition_logits,
                         greedy_logits);
      }
      candidate_logits.index_put_({ISlice(), step, ISlice()},
                                  transition_logits);
    }
    previous_indices = sampled_indices;
  }

  if (!return_draft_logits) {
    return {std::move(token_ids), torch::Tensor()};
  }
  torch::Tensor draft_logits =
      torch::full({batch_size, num_steps, vocab_size},
                  -std::numeric_limits<float>::infinity(),
                  float_options);
  draft_logits.scatter_(/*dim=*/-1, candidates.candidate_ids, candidate_logits);
  return {std::move(token_ids), std::move(draft_logits)};
}

}  // namespace xllm
