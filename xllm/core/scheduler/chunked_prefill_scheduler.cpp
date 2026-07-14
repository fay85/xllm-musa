/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/chunked_prefill_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

// LCM helper (std::lcm is C++17; the existing codebase already uses
// std::gcd elsewhere, so we depend on that and inline the lcm to avoid
// header churn). Returns 0 when either argument is 0 so callers can use
// the result as a multiplier safely.
inline size_t lcm_size_t(size_t a, size_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  return (a / std::gcd(a, b)) * b;
}

// Round a chunked-prefill chunk size DOWN to a multiple of
// `lcm(2 * cp_size, kv_split_size * block_size)`. Two constraints stack
// here after the KV-split / CP decoupling refactor:
//   (a) `num_tokens % (2 * cp_size) == 0` so that worker-side
//       `cp_partition_inplace` produces two equally-sized halves per CP rank
//       (this is the constraint BatchInputBuilder used to enforce via
//       q_seq_len right-padding to 2*cp_size; we now push it to chunk
//       granularity so it is satisfied without padding).
//   (b) `prefix_len % (kv_split_size * block_size) == 0` so the KV-shard
//       prefix geometry (compute_in_prefix_slots /
//       compute_prefix_rank_geometry) stays aligned at block boundaries across
//       chunks. When KV split is off (kv_split_size == 1) this term collapses
//       to block_size and only constraint (a) effectively governs.
//
// The function intentionally does NOTHING and returns `num_tokens` when:
//   - cp_size <= 1 AND kv_split_size <= 1 (no constraints at all)
//   - block_size <= 0 (degenerate config)
//   - num_tokens == 0
//   - The current chunk is the LAST one for this sequence (i.e.
//     num_tokens >= remaining_in_seq). The final chunk's residual is
//     handled by BatchInputBuilder's q_seq_len right-padding to 2*cp_size.
//   - num_tokens is already smaller than the alignment unit. Aligning down
//     to zero would starve the chunk; the caller's downstream budgets
//     stay coherent if we leave the chunk as-is.
inline size_t maybe_align_cp_chunk_tokens(size_t num_tokens,
                                          int32_t cp_size,
                                          int32_t kv_split_size,
                                          int32_t block_size,
                                          size_t remaining_in_seq) {
  if (cp_size <= 1 && kv_split_size <= 1) {
    return num_tokens;
  }
  if (block_size <= 0 || num_tokens == 0) {
    return num_tokens;
  }
  if (num_tokens >= remaining_in_seq) {
    return num_tokens;
  }
  const size_t cp_term =
      cp_size > 1 ? static_cast<size_t>(2) * static_cast<size_t>(cp_size) : 1;
  const size_t kv_term =
      kv_split_size > 1
          ? static_cast<size_t>(kv_split_size) * static_cast<size_t>(block_size)
          : 1;
  const size_t alignment = lcm_size_t(cp_term, kv_term);
  if (alignment == 0 || num_tokens < alignment) {
    return num_tokens;
  }
  return (num_tokens / alignment) * alignment;
}

bool require_homogeneous_graph_batch() {
  const ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  return execution_config.enable_graph() &&
         execution_config.enable_prefill_piecewise_graph();
}

bool enable_packed_prefill() {
  return ExecutionConfig::get_instance().enable_packed_prefill();
}

// Opt-in Phase D instrumentation: packed-group composition per prepare_batch.
bool sched_pack_log_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_SCHED_PACK_LOG");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

// Cap how many pure-prefill requests may share one packed step.
// 0 / unset = no cap (admit until token/seq budget). Useful to A/B 2+2+1
// vs greedy 3+2 packing under rate=5 C=5.
size_t max_packed_prefill_seqs() {
  static const size_t limit = [] {
    const char* env = std::getenv("XLLM_MAX_PACKED_PREFILL_SEQS");
    if (env == nullptr || env[0] == '\0') {
      return static_cast<size_t>(0);
    }
    const long parsed = std::strtol(env, nullptr, 10);
    return parsed > 0 ? static_cast<size_t>(parsed) : static_cast<size_t>(0);
  }();
  return limit;
}

size_t count_prefill_sequences(
    const std::vector<Sequence*>& running_sequences) {
  size_t n = 0;
  for (Sequence* seq : running_sequences) {
    if (seq != nullptr && seq->stage() != SequenceStage::DECODE) {
      ++n;
    }
  }
  return n;
}

const char* sequence_stage_name(SequenceStage stage) {
  switch (stage) {
    case SequenceStage::PREFILL:
      return "PREFILL";
    case SequenceStage::CHUNKED_PREFILL:
      return "CHUNKED";
    case SequenceStage::DECODE:
      return "DECODE";
    default:
      return "UNKNOWN";
  }
}

}  // namespace

ChunkedPrefillScheduler::ChunkedPrefillScheduler(Engine* engine,
                                                 const Options& options)
    : ContinuousScheduler(engine, options) {}

ChunkedPrefillScheduler::~ChunkedPrefillScheduler() {
  // release all requests in the priority queue
  while (!waiting_priority_queue_->empty()) {
    waiting_priority_queue_->pop_top();
  }

  // release all requests in the running priority queue
  while (!running_queue_->empty()) {
    running_queue_->pop_top();
  }
}

size_t ChunkedPrefillScheduler::effective_prefill_chunk_limit(
    size_t batch_token_budget) const {
  const size_t configured_chunk_limit =
      static_cast<size_t>(options_.max_tokens_per_chunk_for_prefill());
  if (!options_.enable_adaptive_prefill_oneshot()) {
    return configured_chunk_limit;
  }

  // One-shot is a latency optimization for an idle instance. Preserve normal
  // chunking whenever decode/chunked work is already active or another request
  // is waiting, so fairness and decode inter-token latency do not change.
  if (!running_queue_->empty() || !running_queue_offline_->empty()) {
    return configured_chunk_limit;
  }
  const size_t online_waiting = waiting_priority_queue_->size();
  const size_t offline_waiting = waiting_priority_queue_offline_->size();
  if (online_waiting + offline_waiting != 1) {
    return configured_chunk_limit;
  }

  const RequestPriorityQueue* waiting_queue =
      online_waiting == 1 ? waiting_priority_queue_.get()
                          : waiting_priority_queue_offline_.get();
  const std::shared_ptr<Request> request = waiting_queue->top();
  if (request->finished() || request->cancelled() || request->preempted() ||
      request->sequences().size() != 1) {
    return configured_chunk_limit;
  }

  Sequence* sequence = request->sequences().front().get();
  const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
  if (sequence->finished() || sequence->stage() != SequenceStage::PREFILL ||
      cached_tokens != 0 || sequence->num_tokens() <= cached_tokens) {
    return configured_chunk_limit;
  }

  const size_t remaining_tokens = sequence->num_tokens() - cached_tokens;
  if (remaining_tokens <= configured_chunk_limit ||
      remaining_tokens > batch_token_budget) {
    return configured_chunk_limit;
  }
  return remaining_tokens;
}

void ChunkedPrefillScheduler::handle_running_queue_requests(
    const size_t max_tokens_per_chunk_for_prefill,
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_preempted_requests,
    std::vector<Sequence*>& prefill_stage_sequences,
    std::unique_ptr<RequestPriorityQueue>& running_queue,
    bool& budget_exhausted,
    bool& blocks_exhausted,
    bool require_homogeneous_batch) {
  std::optional<bool> batch_is_prefill;
  if (require_homogeneous_batch && !running_sequences_.empty()) {
    batch_is_prefill =
        running_sequences_.front()->stage() != SequenceStage::DECODE;
  }

  while (!running_queue->empty() &&
         remaining_token_budget > min_speculative_tokens_required_ &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(running_queue->top());
    const bool request_is_prefill =
        request->sequences().front()->stage() != SequenceStage::DECODE;
    if (require_homogeneous_batch && batch_is_prefill.has_value() &&
        batch_is_prefill.value() != request_is_prefill) {
      break;
    }
    // One independent prefill request per piecewise-graph batch. Multiple
    // decode requests remain allowed. When packed prefill is enabled,
    // multiple prefill requests are admitted up to the token budget.
    if (require_homogeneous_batch && !enable_packed_prefill() &&
        request_is_prefill &&
        batch_is_prefill.has_value() && batch_is_prefill.value()) {
      break;
    }
    const size_t packed_prefill_cap = max_packed_prefill_seqs();
    if (enable_packed_prefill() && request_is_prefill &&
        packed_prefill_cap > 0 &&
        count_prefill_sequences(running_sequences_) >= packed_prefill_cap) {
      break;
    }

    const size_t num_sequences = request->sequences().size();
    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;

    for (auto& sequence : request->sequences()) {
      // skip finished sequence.
      if (sequence->finished()) {
        continue;
      }

      // no budget left

      // The max tokens current sequence can handle.
      const size_t assume_max_tokens =
          std::min(max_tokens_per_chunk_for_prefill,
                   remaining_token_budget - allocated_tokens);

      size_t num_tokens = sequence->num_tokens();
      size_t kv_cache_tokens_num = sequence->kv_state().kv_cache_tokens_num();

      // FIXME: It does not consider the
      // acutual handle token changes when enabling prefix cache. Need to
      // refactor to seperate counting actual handling token and allocating
      // blocks in `allocate_blocks_for`. Distinguish the number of tokens
      // actually handled by current sequence in the prefill stage and the
      // decode stage. Now Partially use `num_tokens_to_handle` to replace
      // `current_step_handle_tokens`.
      size_t num_tokens_to_handle =
          sequence->is_chunked_prefill_stage()
              ? std::min(assume_max_tokens, num_tokens - kv_cache_tokens_num)
              : 1 + min_speculative_tokens_required_;

      if (allocated_seqs + 1 > remaining_seq_budget ||
          allocated_tokens + num_tokens_to_handle > remaining_token_budget) {
        has_enough_budget = false;
        break;
      }
      // for chunked prefill, we need to estimate latency according to
      // `current_step_handle_tokens` which is totally precise.
      double seq_estimate_latency = 0.0;
      if (options_.enable_latency_aware_schedule()) {
        seq_estimate_latency = profile_manager_->predict_step_time(
            num_tokens_to_handle + kv_cache_tokens_num,
            kv_cache_tokens_num,
            false);
        if (estimate_latency + allocated_estimate_latency +
                seq_estimate_latency >
            latency_budget) {
          has_enough_budget = false;
          break;
        }
      }

      // actual tokens be handled,
      // decode: 1 + num_speculative_tokens
      // prefill: std::min(seq.num_tokens(), min_tokens_per_iter_for_prefill)
      size_t current_step_handle_tokens = 0;
      // no budget left
      if (!allocate_blocks_for(
              sequence.get(), assume_max_tokens, &current_step_handle_tokens)) {
        has_enough_blocks = false;
        break;
      }

      // for chunked prefill, we need to estimate latency according to
      // current_step_handle_tokens.
      // TO IMPROVE and REFACTOR: seperate allocate
      // prefix cache blocks and compute prefix cache match to estimate latency
      // before acctually allocating blocks

      // if (sequence->if_cache_block_for_prefill()) {
      //   kv_cache_manager_->cache(sequence.get());
      // }

      // the new request do chunked prefill
      if (sequence->kv_state().kv_cache_tokens_num() == 0 ||
          sequence->is_chunked_prefill_stage()) {
        prefill_stage_sequences.emplace_back(sequence.get());
      }

      // update the allocated tokens for the sequence
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue->pop_top();
      if (require_homogeneous_batch && !batch_is_prefill.has_value()) {
        batch_is_prefill = request_is_prefill;
      }
      // add the request to the batch
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      cache_in_batch_prefix(candidate_sequences, candidate_token_budgets);
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      estimate_latency += allocated_estimate_latency;
      continue;
    }

    // budget exhausted, do partially schedule the request
    if (!has_enough_budget) {
      handle_abnormal_request(running_queue.get(),
                              candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              allocated_estimate_latency,
                              remaining_token_budget,
                              remaining_seq_budget,
                              estimate_latency,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      budget_exhausted = true;
      break;
    }

    // memory exhausted, preempt lowest priority request and retry.
    // preemptable_requests_ only contain decoding requests.
    // Maybe improve: for prefill stage sequence, we know how many blocks it
    // needs
    if (options_.enable_online_preempt_offline() && !request->offline() &&
        !running_queue_offline_->empty()) {
      std::shared_ptr<Request> request_to_preempt =
          running_queue_offline_->back();
      ++num_preempted_requests;
      kv_cache_manager_->deallocate(request_to_preempt.get());
      running_queue_offline_->pop_back();
      // add preemptable request to waiting priority queue
      request_to_preempt->set_preempted();
      waiting_priority_queue_offline_->push(request_to_preempt);
      continue;
    } else if (running_queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue->back();
      if (request_to_preempt.get() != request.get()) {
        ++num_preempted_requests;
        // TO IMPROVE: kv cache offload to cpu
        kv_cache_manager_->deallocate(request_to_preempt.get());
        running_queue_->pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        if (request_to_preempt->offline()) {
          waiting_priority_queue_offline_->push(request_to_preempt);
        } else {
          waiting_priority_queue_->push(request_to_preempt);
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(running_queue.get(),
                            candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            allocated_estimate_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            estimate_latency,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    blocks_exhausted = true;
    break;
  }
}

void ChunkedPrefillScheduler::handle_prefill_requests(
    const size_t max_tokens_per_chunk_for_prefill,
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_preempted_requests,
    std::vector<Sequence*>& prefill_stage_sequences,
    RequestPriorityQueue* waiting_priority_queue,
    bool& budget_exhausted,
    bool& blocks_exhausted,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  const bool require_homogeneous_batch = require_homogeneous_graph_batch();
  // Refuse to mix waiting prefills into an already-scheduled decode (or
  // other) batch under homogeneous graph mode. When enable_packed_prefill
  // is on, multiple pure prefills are still packed in the loop below as
  // long as running_sequences_ starts empty for this call.
  if (require_homogeneous_batch && !running_sequences_.empty()) {
    return;
  }

  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contian many sequences, so we should check here.
  while (!waiting_priority_queue->empty() && remaining_token_budget > 0 &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    if (require_homogeneous_batch && !enable_packed_prefill() &&
        !running_sequences_.empty()) {
      break;
    }
    const size_t packed_prefill_cap = max_packed_prefill_seqs();
    if (enable_packed_prefill() && packed_prefill_cap > 0 &&
        count_prefill_sequences(running_sequences_) >= packed_prefill_cap) {
      break;
    }

    // Memory-utilization short-circuit (added by the CP PR): when not in
    // disagg-PD mode, refuse to admit additional waiting prefill requests
    // once KV-cache utilization crosses the threshold so that decode tail
    // latency stays bounded. In disagg-PD the prefill instance has its own
    // memory budgeting, so we skip this check.
    if (!options_.enable_disagg_pd() &&
        kv_cache_manager_->kv_cache_utilization() >=
            SchedulerConfig::get_instance()
                .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(waiting_priority_queue->top());
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      waiting_priority_queue->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (require_homogeneous_batch && !running_sequences_.empty()) {
      size_t request_chunk_tokens = 0;
      size_t request_active_sequences = 0;
      for (const auto& sequence : request->sequences()) {
        if (sequence->finished()) {
          continue;
        }
        const size_t cached_tokens =
            sequence->kv_state().kv_cache_tokens_num();
        const size_t remaining_tokens =
            sequence->num_tokens() > cached_tokens
                ? sequence->num_tokens() - cached_tokens
                : 0;
        request_chunk_tokens +=
            std::min(max_tokens_per_chunk_for_prefill, remaining_tokens);
        ++request_active_sequences;
      }
      if (request_chunk_tokens > remaining_token_budget ||
          request_active_sequences > remaining_seq_budget) {
        budget_exhausted = true;
        break;
      }
    }

    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    size_t allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      size_t num_tokens = prefill_sequence->num_tokens();
      // The max tokens current sequence can handle.
      const size_t assume_max_tokens =
          std::min(max_tokens_per_chunk_for_prefill,
                   remaining_token_budget - allocated_tokens);

      num_tokens = std::min(assume_max_tokens, num_tokens);

      // CP-aware chunk alignment: when CP is enabled and this is not the
      // last chunk for the sequence, snap the chunk size down to a multiple
      // of cp_size * block_size so worker-side cp_partition produces
      // balanced former/latter chunks and the prefix-cache invariant
      // `prefix_len % (cp_size * block_size) == 0` is maintained chunk to
      // chunk (see cp_kv_split.md §6.7). For non-CP, this is a no-op.
      const size_t kv_cache_tokens_num_for_align =
          prefill_sequence->kv_state().kv_cache_tokens_num();
      const size_t remaining_in_seq =
          prefill_sequence->num_tokens() > kv_cache_tokens_num_for_align
              ? prefill_sequence->num_tokens() - kv_cache_tokens_num_for_align
              : 0;
      // kv_split_size: FLAGS_kv_split_size > 0 takes priority (independent of
      // cp_size); otherwise fall back to cp_size for legacy parity. We read
      // FLAGS directly here because the scheduler holds no ParallelArgs.
      const int32_t kv_split_for_align =
          ::xllm::ParallelConfig::get_instance().kv_split_size_effective();
      num_tokens = maybe_align_cp_chunk_tokens(num_tokens,
                                               options_.cp_size(),
                                               kv_split_for_align,
                                               kv_cache_manager_->block_size(),
                                               remaining_in_seq);

      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // OPTIMIZE: for prefill requests, need to compute num_tokens_to_handle
      // after prefix cache match
      size_t seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        size_t kv_cache_tokens_num =
            prefill_sequence->kv_state().kv_cache_tokens_num();
        seq_estimate_latency = profile_manager_->predict_step_time(
            num_tokens + kv_cache_tokens_num, kv_cache_tokens_num, false);
        if (estimate_latency + allocated_estimate_latency +
                seq_estimate_latency >
            latency_budget) {
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      size_t current_step_handle_tokens = 0;
      if (!allocate_blocks_for(prefill_sequence.get(),
                               num_tokens,
                               &current_step_handle_tokens)) {
        can_schedule = false;
        if (options_.enable_online_preempt_offline() && !request->offline() &&
            !running_queue_offline_->empty()) {
          size_t num_request_to_evict = 0;
          // according to the prefill_sequence num tokens to check if can
          // allocate blocks for it through evict
          size_t max_handle_num_tokens =
              current_step_handle_tokens +
              prefill_sequence->kv_state().kv_cache_tokens_num();
          bool enough_to_evict =
              check_if_enough_to_evict(running_queue_offline_.get(),
                                       prefill_sequence.get(),
                                       max_handle_num_tokens,
                                       num_request_to_evict);
          if (enough_to_evict) {
            for (size_t i = 0; i < num_request_to_evict; ++i) {
              std::shared_ptr<Request> request_to_preempt =
                  running_queue_offline_->back();
              ++num_preempted_requests;
              kv_cache_manager_->deallocate(request_to_preempt.get());
              running_queue_offline_->pop_back();
              // add preemptable request to waiting priority queue
              // TO IMPROVE?: not process this offline request in current batch
              request_to_preempt->set_preempted();
              waiting_priority_queue_offline_->push(request_to_preempt);
            }
            if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                             max_handle_num_tokens)) {
              LOG(ERROR) << "Should be able to allocate after preempting "
                         << num_request_to_evict
                         << " offline requests, but failed.";
              can_schedule = false;
            } else {
              can_schedule = true;
            }
          }
        }
        if (!can_schedule) {
          kv_cache_manager_->deallocate(prefill_sequence.get());
          blocks_exhausted = true;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(current_step_handle_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_->deallocate(seq);
      }
      break;
    }

    prefill_stage_sequences.insert(prefill_stage_sequences.end(),
                                   prefill_sequences.begin(),
                                   prefill_sequences.end());
    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    estimate_latency += allocated_estimate_latency;
    waiting_priority_queue->pop_top();
    running_requests_.emplace_back(request);
    request->record_num_prefix_cache_tokens();
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget);
  }
  // maybe can pre-compute if prompt beyond length
  if (running_sequences_.empty() && !waiting_priority_queue->empty() &&
      running_queue_->empty()) {
    std::shared_ptr<Request> request(waiting_priority_queue->top());
    waiting_priority_queue->pop_top();
    kv_cache_manager_->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budegt.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }
}

void ChunkedPrefillScheduler::handle_remaining_budget(
    const size_t max_tokens_per_chunk_for_prefill,
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    std::vector<Sequence*>& prefill_stage_sequences,
    bool& blocks_exhausted) {
  size_t prefill_stage_seq_idx = 0;
  for (size_t i = 0; i < running_sequences_.size() &&
                     prefill_stage_seq_idx < prefill_stage_sequences.size();
       ++i) {
    if (prefill_stage_sequences[prefill_stage_seq_idx] !=
        running_sequences_[i]) {
      continue;
    }
    ++prefill_stage_seq_idx;
    Sequence* sequence = running_sequences_[i];
    size_t& token_budget = running_sequences_budgets_[i];

    // add previous allocated tokens back
    remaining_token_budget += token_budget;
    const size_t sequence_token_budget =
        std::min(remaining_token_budget, max_tokens_per_chunk_for_prefill);
    if (options_.enable_latency_aware_schedule()) {
      auto origin_estimate_seq_latency =
          profile_manager_->predict_step_time(sequence, false);
      // subtract the allocated latency back
      estimate_latency -= origin_estimate_seq_latency;

      // check latency budget
      auto cur_estimate_seq_latency = profile_manager_->predict_step_time(
          sequence_token_budget,
          sequence->kv_state().kv_cache_tokens_num(),
          false);
      if (estimate_latency + cur_estimate_seq_latency > latency_budget) {
        // beyond latency budget, break
        break;
      }
      estimate_latency += cur_estimate_seq_latency;
    }

    size_t current_step_handle_tokens = 0;
    // no memory left
    if (!allocate_blocks_for(
            sequence, sequence_token_budget, &current_step_handle_tokens)) {
      blocks_exhausted = true;
      break;
    }

    // update the allocated tokens for the sequence
    token_budget = current_step_handle_tokens;
    CHECK(remaining_token_budget >= current_step_handle_tokens);
    remaining_token_budget -= current_step_handle_tokens;

    // no budget left
    if (remaining_token_budget == 0) {
      break;
    }
  }
}

std::vector<Batch> ChunkedPrefillScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting_priority_queue_
  while (request_queue_.read(request)) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      if (request->offline()) {
        waiting_priority_queue_offline_->push(request);
      } else {
        waiting_priority_queue_->push(request);
      }
    } else {
      // request from prefill instance in disagge pd mode.
      // NOTE: running_requests_ keep a batch of requests in running state,
      //   sorted by priority from high to low.
      running_requests_.emplace_back(request);
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  // insert running requests back to the priority queue, iterating from the
  // lowest priority to the highest
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    // finished request is set to nullptr
    if (*it == nullptr) {
      continue;
    }
    handle_running_requests(*it);
    // unified multi priority strategy
    if ((*it)->offline()) {
      running_queue_offline_->push(*it);
    } else {
      running_queue_->push(*it);
    }

    // push the request front to the priority deque
    // running_queue_->push(request, false /*if_back*/);
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // for new request which in prefill stage, this means we handle
  // std::min(seq.num_tokens(), max_tokens_per_chunk_for_prefill)
  // num tokens per step.
  // seq.num_tokens(): `prompt tokens` or `(prompt tokens + partial
  // generated tokens when request be preempted and recomputed)`.
  //
  // If we don't set `max_tokens_per_chunk_for_prefill`, the default value is
  // int32_t::max(), which means we process high-priority prefill first.
  // If set `max_tokens_per_chunk_for_prefill`, for example 512,
  // If a high-priority prefill request is very long, each step only
  // handle 2028 promote tolkens, the scheduler will allocate
  // part of the resources to subsequent prefill requests
  // for execution, thereby preventing subsequent requests from being
  // blocked by the first extremely long request.
  //

  const size_t configured_prefill_chunk_limit =
      static_cast<size_t>(options_.max_tokens_per_chunk_for_prefill());

  // maintain estimate_latency for current batch for support requests with
  // different tpot. Currently only support one unified tpot for ChunkedPrefill.
  // TO IMPROVE: use min remaining time (i.e. tpot_slo - elapsed_time) of the
  // reuquest in current decode queue to replace latency_budget.
  double latency_budget = options_.max_global_tpot_ms();
  // size_t estimate_latency =
  // profile_manager_->get_constant_overhead()->get_constant_overhead(); // add
  // constant overhead ahead
  double estimate_latency = profile_manager_->get_constant_overhead();
  // Max tokens be handled in once chunked prefill schedule.
  size_t remaining_token_budget = options_.enable_profile_token_budget()
                                      ? profile_manager_->get_token_budget()
                                      : options_.max_tokens_per_batch();
  const size_t max_tokens_per_chunk_for_prefill =
      effective_prefill_chunk_limit(remaining_token_budget);
  const bool adaptive_oneshot =
      max_tokens_per_chunk_for_prefill > configured_prefill_chunk_limit;
  size_t remaining_seq_budget = options_.max_seqs_per_batch();
  size_t num_preempted_requests = 0;
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  // keep the requests in prefill stage
  std::vector<Sequence*> prefill_stage_sequences;
  const bool require_homogeneous_batch = require_homogeneous_graph_batch();
  const bool prioritize_waiting_prefill =
      require_homogeneous_batch &&
      (!waiting_priority_queue_->empty() ||
       !waiting_priority_queue_offline_->empty());

  // step-1. handle requests from running_queue_
  // "decode-maximal batching" principle:
  //  Prioritize adding Decode requests to the batch until
  //  the KV cache space they occupy reaches the upper limit.
  //  Subsequently, based on the remaining token quota,
  //  split sequences requiring Prefill into chunks and
  //  add them to the batch.
  //
  if (!prioritize_waiting_prefill) {
    handle_running_queue_requests(max_tokens_per_chunk_for_prefill,
                                  latency_budget,
                                  estimate_latency,
                                  remaining_token_budget,
                                  remaining_seq_budget,
                                  num_preempted_requests,
                                  prefill_stage_sequences,
                                  running_queue_,
                                  budget_exhausted,
                                  blocks_exhausted,
                                  require_homogeneous_batch);
  }

  // step-2. handle new prefill request
  // new prefill request can not preempt any running requests.
  // TODO: replace condition budget_exhausted and blocks_exhausted with actual
  // condition Otherwise, fragmentation may occur due to processing some long
  // requests first
  if (!budget_exhausted && !blocks_exhausted) {
    // NOTE: for latency schedule, currently only consider tpot for chunked
    // prefill. when no decode requests, not consider latency_budget.
    if (running_sequences_.empty()) {
      latency_budget = std::numeric_limits<int32_t>::max();
    }
    handle_prefill_requests(max_tokens_per_chunk_for_prefill,
                            latency_budget,
                            estimate_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            num_preempted_requests,
                            prefill_stage_sequences,
                            waiting_priority_queue_.get(),
                            budget_exhausted,
                            blocks_exhausted,
                            finished_requests);
  }
  // handle offline
  // online prefill may be too long to allocate, but decode just need
  // num_speculative_tokens+1 token per step
  if (remaining_token_budget > 0 && !prioritize_waiting_prefill) {
    handle_running_queue_requests(max_tokens_per_chunk_for_prefill,
                                  latency_budget,
                                  estimate_latency,
                                  remaining_token_budget,
                                  remaining_seq_budget,
                                  num_preempted_requests,
                                  prefill_stage_sequences,
                                  running_queue_offline_,
                                  budget_exhausted,
                                  blocks_exhausted,
                                  require_homogeneous_batch);
  }
  if (!budget_exhausted && !blocks_exhausted) {
    if (running_sequences_.empty()) {
      latency_budget = std::numeric_limits<int32_t>::max();
    }
    handle_prefill_requests(max_tokens_per_chunk_for_prefill,
                            latency_budget,
                            estimate_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            num_preempted_requests,
                            prefill_stage_sequences,
                            waiting_priority_queue_offline_.get(),
                            budget_exhausted,
                            blocks_exhausted,
                            finished_requests);
  }

  // step-3. remaining_token_budget > 0, try to allocate more tokens to
  // prefill stage reuquests.
  if (remaining_token_budget > 0 && latency_budget > estimate_latency) {
    handle_remaining_budget(max_tokens_per_chunk_for_prefill,
                            latency_budget,
                            estimate_latency,
                            remaining_token_budget,
                            prefill_stage_sequences,
                            blocks_exhausted);
  }

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(running_requests_,
                                      running_sequences_,
                                      running_sequences_budgets_);

  bool is_batches_empty =
      (std::all_of(batches.begin(), batches.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      }));
  if (!is_batches_empty) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  if (sched_pack_log_enabled() && !is_batches_empty) {
    size_t n_prefill = 0;
    size_t n_chunked = 0;
    size_t n_decode = 0;
    size_t token_budget_sum = 0;
    std::ostringstream seq_oss;
    for (size_t i = 0; i < running_sequences_.size(); ++i) {
      Sequence* seq = running_sequences_[i];
      const size_t budget = running_sequences_budgets_[i];
      token_budget_sum += budget;
      const SequenceStage stage = seq->stage();
      if (stage == SequenceStage::PREFILL) {
        ++n_prefill;
      } else if (stage == SequenceStage::CHUNKED_PREFILL) {
        ++n_chunked;
      } else {
        ++n_decode;
      }
      if (i > 0) {
        seq_oss << ',';
      }
      seq_oss << sequence_stage_name(stage) << ':' << budget << '/'
              << seq->num_tokens() << '@' << seq->kv_cache_tokens_num();
    }
    // Decode-only steps are extremely frequent; keep Phase D logs on
    // batches that actually contain prefill/chunked work.
    if (n_prefill > 0 || n_chunked > 0) {
      LOG(INFO) << "[SCHED_PACK] n_seq=" << running_sequences_.size()
                << " n_prefill=" << n_prefill << " n_chunked=" << n_chunked
                << " n_decode=" << n_decode
                << " token_budget=" << token_budget_sum
                << " adaptive_oneshot=" << adaptive_oneshot
                << " prioritize_waiting=" << prioritize_waiting_prefill
                << " packed_prefill=" << enable_packed_prefill()
                << " waiting=" << waiting_priority_queue_->size()
                << " running_q=" << running_queue_->size()
                << " sched_ms=" << timer.elapsed_milliseconds()
                << " seqs=[" << seq_oss.str() << "]";
    }
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            waiting_priority_queue_->size() + running_queue_->size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(kv_cache_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(kv_cache_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(kv_cache_manager_->num_used_blocks()));

  return batches;
}

bool ChunkedPrefillScheduler::allocate_blocks_for(
    Sequence* sequence,
    size_t token_budget,
    size_t* current_step_handle_tokens) {
  // token budget should be large enough for one speculative decoding step
  CHECK_GT(token_budget, min_speculative_tokens_required_);

  allocate_shared_blocks_for(sequence);

  // number of tokens in the kv cache, which are already processed
  const size_t kv_cache_tokens_num = sequence->kv_cache_tokens_num();
  // the total number tokens for the sequence can be handled till now.
  // there may some tokens can not be handled once when enable chunked prefill.
  size_t max_handle_num_tokens =
      std::min(kv_cache_tokens_num + token_budget, sequence->num_tokens());

  // speculative decoding specific logic,
  // prefill stage don't need speculative decoding.
  //
  // if in decoding stage
  if (options_.num_speculative_tokens() > 0 &&
      !sequence->is_chunked_prefill_stage() && kv_cache_tokens_num > 0) {
    max_handle_num_tokens += min_speculative_tokens_required_;
  }

  // make sure the sequence proceeds forward
  CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *current_step_handle_tokens = max_handle_num_tokens - kv_cache_tokens_num;
  // allocate blocks for the sequence
  return kv_cache_manager_->allocate(sequence, max_handle_num_tokens);
}

void ChunkedPrefillScheduler::allocate_shared_blocks_for(Sequence* sequence) {
  if (sequence->kv_state().num_blocks(BlockType::KV) == 0) {
    // allocate shared blocks
    kv_cache_manager_->allocate_shared(sequence);
    return;
  }
  if (sequence->is_chunked_prefill_stage()) {
    const size_t max_tokens_per_chunk_for_prefill =
        std::max(options_.max_tokens_per_chunk_for_prefill(), 64);
    size_t total_chunked_size =
        (sequence->num_tokens() + max_tokens_per_chunk_for_prefill - 1) /
        max_tokens_per_chunk_for_prefill;
    if (total_chunked_size <
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency()) {
      kv_cache_manager_->allocate_shared(sequence);
      return;
    }
    size_t prefix_cache_interval =
        (total_chunked_size +
         ::xllm::SchedulerConfig::get_instance().chunked_match_frequency() -
         1) /
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency();
    size_t cur_chunked_index = sequence->kv_state().kv_cache_tokens_num() /
                               max_tokens_per_chunk_for_prefill;
    if (cur_chunked_index % prefix_cache_interval == 0) {
      // allocate shared blocks
      kv_cache_manager_->allocate_shared(sequence);
    }
  }
}

}  // namespace xllm
