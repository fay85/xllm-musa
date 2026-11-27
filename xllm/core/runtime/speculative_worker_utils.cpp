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

#include "runtime/speculative_worker_utils.h"

#include <glog/logging.h>

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/llm_worker_impl.h"

namespace xllm::speculative_worker {

// Speculative state is replicated across every model rank within one DP
// replica. With orthogonal CP x TP, tp_group_ covers only one CP shard, while
// cp_group_ connects the same TP rank across CP shards. Broadcast along both
// axes so every replica caches and consumes the same sampled token, without
// crossing into another DP replica through the world group.
void broadcast_speculative_token_ids(torch::Tensor& tokens,
                                     const ParallelArgs& parallel_args) {
  // DeepSeek-V4 TORCH publishes orthogonal TP and CP groups. Other backends
  // retain their existing single-group speculative broadcast behavior.
  ProcessGroup* tp_group =
      speculative_worker::sampling_process_group(parallel_args);
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  if (!use_orthogonal_cp_consensus) {
    broadcast_tokens_in_group(tokens, tp_group);
    return;
  }

  broadcast_tokens_in_group(tokens, parallel_args.tp_group_);
  if (parallel_args.cp_group_ != parallel_args.tp_group_) {
    broadcast_tokens_in_group(tokens, parallel_args.cp_group_);
  }
}

ProcessGroup* sampling_process_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank) {
  if (process_group == nullptr || process_group->world_size() <= 1 ||
      !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  process_group->broadcast(tokens, root_rank);
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  input.metadata_ready_event = stream.record_event_or_sync();
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

std::optional<ForwardOutput> run_worker_no_sync(
    WorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput* processed_output,
    WorkerStreamOptions options,
    const std::function<void(ModelInputParams&)>& adjust_prepared_params) {
  if (processed_output == nullptr) {
    ForwardInput processed_input;
    return run_worker_no_sync(worker,
                              input,
                              prepare_stream,
                              compute_stream,
                              &processed_input,
                              options,
                              adjust_prepared_params);
  }
  ForwardInput& processed_input = *processed_output;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream, options.record_input_ready_event);
  if (adjust_prepared_params) {
    adjust_prepared_params(processed_input.input_params);
  }
  if (!options.record_output_ready_event) {
    if (auto* llm_worker = dynamic_cast<LLMWorkerImpl*>(&worker);
        llm_worker != nullptr) {
      return llm_worker->execute_no_sync_on_stream(
          processed_input, compute_stream, /*record_ready_event=*/false);
    }
  }
  return worker.execute_no_sync_on_stream(processed_input, compute_stream);
}

}  // namespace xllm::speculative_worker
