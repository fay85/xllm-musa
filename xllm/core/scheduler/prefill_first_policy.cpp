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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

#include "core/framework/config/scheduler_config.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {

namespace {

size_t estimate_remaining_prefill_tokens(const std::shared_ptr<Request>& req,
                                         const SchedulerState& state,
                                         const BlockManagerPool* pool) {
  if (req == nullptr || req->sequences().empty() ||
      req->sequences()[0] == nullptr) {
    return 0;
  }
  Sequence* seq = req->sequences()[0].get();
  const size_t total = seq->num_tokens();
  size_t cached = seq->kv_state().kv_cache_tokens_num();
  if (cached == 0 && pool != nullptr && state.enable_prefix_cache) {
    const int32_t dp_rank = std::max<int32_t>(0, seq->dp_rank());
    cached = pool->probe_prefix_hit_tokens(seq, dp_rank);
  }
  return total > cached ? total - cached : 0;
}

// Running decodes keep the device until a waiting prefill request has been
// queued for decode_priority_max_wait_ms.
bool should_defer_new_prefill(const SchedulerState& state) {
  const int32_t max_wait_ms =
      ::xllm::SchedulerConfig::get_instance().decode_priority_max_wait_ms();
  if (max_wait_ms <= 0 || state.decode_queue.empty() ||
      state.prefill_queue.empty()) {
    return false;
  }
  const absl::Time wait_deadline =
      absl::Now() - absl::Milliseconds(max_wait_ms);
  for (const std::shared_ptr<Request>& request : state.prefill_queue) {
    if (request->created_time() <= wait_deadline) {
      return false;
    }
  }
  return true;
}

}  // namespace

// =============================================================================
// PrefillFirstPolicy::schedule
// =============================================================================

void PrefillFirstPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  // === Requeue phase ===
  if (batch_mode_.priority_strategy == "fcfs") {
    // ContinuousScheduler / PrefillOnlyScheduler with FCFS:
    // Respect last_step_prefill for insertion order.
    if (state.last_step_prefill) {
      // Forward iterate: push_back for decode maintains FCFS order.
      // Continuations go to chunk_queue front (reverse order preserved by
      // collecting then pushing).
      for (auto it = state.running_requests.cbegin();
           it != state.running_requests.cend();
           ++it) {
        if (*it == nullptr) {
          continue;
        }
        auto req = *it;
        handle_running_requests(req, state);
        if (batch_mode_.enable_chunked_prefill &&
            req->sequences()[0]->is_chunked_prefill_stage()) {
          state.chunk_queue.push(req, /*if_back=*/true);
        } else {
          state.decode_queue.push(req, /*if_back=*/true);
        }
      }
    } else {
      // last step was decode -- all requests here are decode stage
      for (auto it = state.running_requests.crbegin();
           it != state.running_requests.crend();
           ++it) {
        if (*it == nullptr) {
          continue;
        }
        auto req = *it;
        handle_running_requests(req, state);
        state.decode_queue.push(req, /*if_back=*/false);
      }
    }
  } else {
    // Priority / deadline / multi_slo_and_prio in exclusive mode.
    for (auto it = state.running_requests.cbegin();
         it != state.running_requests.cend();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      auto req = *it;
      handle_running_requests(req, state);
      if (batch_mode_.enable_chunked_prefill &&
          req->sequences()[0]->is_chunked_prefill_stage()) {
        state.chunk_queue.push(req);
      } else {
        state.decode_queue.push(req);
      }
    }
  }
  reset_batch_state(state);

  // Requeue / MPMC drain can shuffle FCFS order. Sort waiting prefill
  // queues by created_time so earlier requests stay at the front.
  const auto* block_pool =
      state.enable_prefix_cache
          ? dynamic_cast<const BlockManagerPool*>(state.kv_cache_manager)
          : nullptr;
  const bool is_prefill_instance =
      state.options.instance_role().has_value() &&
      state.options.instance_role().value() == InstanceRole::PREFILL;
  if (batch_mode_.priority_strategy == "fcfs") {
    auto fcfs_cmp = create_comparator("fcfs", /*is_reversed=*/true);
    if (!state.prefill_queue.empty() && state.prefill_queue.supports_sort()) {
      state.prefill_queue.sort(fcfs_cmp);
      if (is_prefill_instance && state.prefill_queue.size() > 1) {
        const absl::Time now = absl::Now();
        const size_t max_chunk =
            std::max<size_t>(1, options_.max_tokens_per_chunk_for_prefill());
        constexpr double kPrefillScoreBucketWidth = 128.0;
        std::unordered_map<const Request*, double> srpt_scores;
        srpt_scores.reserve(state.prefill_queue.size());
        for (const std::shared_ptr<Request>& req : state.prefill_queue) {
          if (req == nullptr) {
            continue;
          }
          const size_t rem =
              estimate_remaining_prefill_tokens(req, state, block_pool);
          const double wait_ms = std::max(
              0.0, absl::ToDoubleMilliseconds(now - req->created_time()));
          const bool is_fast = rem <= max_chunk;
          const double score =
              is_fast ? (static_cast<double>(rem) - 2.0 * wait_ms)
                      : (static_cast<double>(max_chunk + rem) - 10.0 * wait_ms);
          srpt_scores.emplace(req.get(),
                              std::floor(score / kPrefillScoreBucketWidth));
        }
        state.prefill_queue.sort(
            [&srpt_scores, &fcfs_cmp](const std::shared_ptr<Request>& a,
                                      const std::shared_ptr<Request>& b) {
              if (a == nullptr || b == nullptr) {
                return a != nullptr;
              }
              const double score_a = srpt_scores.at(a.get());
              const double score_b = srpt_scores.at(b.get());
              if (score_a != score_b) {
                return score_a < score_b;
              }
              return fcfs_cmp(a, b);
            });
      }
    }
    if (!state.chunk_queue.empty() && state.chunk_queue.supports_sort()) {
      state.chunk_queue.sort(fcfs_cmp);
    }
  }

  static thread_local const Request* last_deferred_chunk_req = nullptr;
  static thread_local absl::Time deferred_chunk_since = absl::InfinitePast();
  static thread_local int32_t consecutive_fast_passes = 0;
  static thread_local int32_t total_fast_passes_on_req = 0;
  static thread_local double accumulated_deferred_ms_on_req = 0.0;
  const Request* oldest_chunk_req =
      !state.chunk_queue.empty() ? state.chunk_queue.top().get() : nullptr;
  if (oldest_chunk_req != last_deferred_chunk_req) {
    last_deferred_chunk_req = oldest_chunk_req;
    deferred_chunk_since = absl::InfinitePast();
    consecutive_fast_passes = 0;
    total_fast_passes_on_req = 0;
    accumulated_deferred_ms_on_req = 0.0;
  }
  const absl::Time sched_now = absl::Now();
  const double active_deferred_ms =
      (consecutive_fast_passes > 0 &&
       deferred_chunk_since != absl::InfinitePast())
          ? std::max(0.0,
                     absl::ToDoubleMilliseconds(sched_now - deferred_chunk_since))
          : 0.0;
  const double chunk_deferred_ms =
      accumulated_deferred_ms_on_req + active_deferred_ms;
  // Cap both per-chunk-boundary passes (<=1) and cumulative deferral across the
  // entire life of a multi-chunk cold request (<=2 passes, <700ms total) so
  // warm bursts never push a 1.8s 24K cold prefill out to 4-6s.
  const bool allow_chunk_deferral =
      consecutive_fast_passes < 1 && total_fast_passes_on_req < 2 &&
      chunk_deferred_ms < 700.0;

  // When a prefix-cached request is split at a fine linear-state checkpoint
  // boundary (e.g. 2048 + short final tail), let its single-step final tail
  // finish ahead of multi-chunk cold continuations in chunk_queue subject to
  // the bounded deferral cap on oldest_chunk_req.
  if (batch_mode_.priority_strategy == "fcfs" && is_prefill_instance &&
      state.chunk_queue.size() > 1 && state.chunk_queue.supports_sort() &&
      allow_chunk_deferral) {
    auto fcfs_cmp = create_comparator("fcfs", /*is_reversed=*/true);
    const size_t max_chunk =
        std::max<size_t>(1, options_.max_tokens_per_chunk_for_prefill());
    constexpr double kChunkScoreBucketWidth = 512.0;
    std::unordered_map<const Request*, double> chunk_scores;
    chunk_scores.reserve(state.chunk_queue.size());
    for (const std::shared_ptr<Request>& req : state.chunk_queue) {
      if (req == nullptr) {
        continue;
      }
      const size_t rem =
          estimate_remaining_prefill_tokens(req, state, block_pool);
      const double wait_ms = std::max(
          0.0, absl::ToDoubleMilliseconds(sched_now - req->created_time()));
      const bool is_one_step = rem <= max_chunk;
      const double multi_floor = static_cast<double>(max_chunk) + 1024.0;
      const double score =
          is_one_step
              ? (static_cast<double>(rem) - 2.0 * wait_ms)
              : std::max(multi_floor,
                         static_cast<double>(max_chunk + rem) - 10.0 * wait_ms);
      chunk_scores.emplace(req.get(),
                           std::floor(score / kChunkScoreBucketWidth));
    }
    state.chunk_queue.sort(
        [&chunk_scores, &fcfs_cmp](const std::shared_ptr<Request>& a,
                                   const std::shared_ptr<Request>& b) {
          if (a == nullptr || b == nullptr) {
            return a != nullptr;
          }
          const double score_a = chunk_scores.at(a.get());
          const double score_b = chunk_scores.at(b.get());
          if (score_a != score_b) {
            return score_a < score_b;
          }
          return fcfs_cmp(a, b);
        });
  }

  // A completed D2H release must first be available to the decode request
  // that triggered preemption. Do not let the restore waiter reclaim the
  // released blocks before that request can retry.
  const bool retry_decode_before_restore =
      !state.decode_restore_waiting.empty() &&
      !state.kv_cache_manager->has_pending_async_block_release();
  if (retry_decode_before_restore) {
    budget.latency_budget = options_.max_global_tpot_ms();
    adjust_latency_budget_and_reorder(&state.decode_queue,
                                      /*second_queue=*/nullptr,
                                      budget.latency_budget,
                                      /*for_prefill=*/false,
                                      state);
    schedule_decode_from_queue(&state.decode_queue, state, budget);
    if (!state.running_sequences.empty() ||
        state.kv_cache_manager->has_pending_async_block_release()) {
      return;
    }
  }
  schedule_decode_restore(state, budget);

  // === Schedule phase ===
  budget.latency_budget = options_.max_global_ttft_ms();

  // Compute urgency + reorder for both prefill queues together.
  adjust_latency_budget_and_reorder(&state.chunk_queue,
                                    &state.prefill_queue,
                                    budget.latency_budget,
                                    /*for_prefill=*/true,
                                    state);

  // Schedule chunked prefill continuations first (they already have partial
  // KV), unless a single-chunk request in prefill_queue can finish in one step
  // while chunk_queue still has multiple chunks remaining.
  std::vector<size_t> reserved_full_footprint(
      static_cast<size_t>(state.options.dp_size()), 0);
  bool fast_single_chunk_first = false;
  bool run_one_step_alone = false;
  size_t diag_rem_new = 0;
  size_t diag_rem_chunk = 0;
  if (is_prefill_instance && batch_mode_.enable_chunked_prefill &&
      !state.chunk_queue.empty() && !state.prefill_queue.empty() &&
      !should_defer_new_prefill(state) && allow_chunk_deferral) {
    const size_t max_chunk =
        std::max<size_t>(1, options_.max_tokens_per_chunk_for_prefill());
    const size_t rem_new = estimate_remaining_prefill_tokens(
        state.prefill_queue.top(), state, block_pool);
    const size_t rem_chunk = estimate_remaining_prefill_tokens(
        state.chunk_queue.top(), state, block_pool);
    diag_rem_new = rem_new;
    diag_rem_chunk = rem_chunk;

    const bool single_seq_prefill =
        should_limit_prefill_requests(state) && state.options.dp_size() <= 1;
    const size_t chunk_take = std::min(
        rem_chunk, std::min(max_chunk, budget.remaining_token_budget));
    const size_t leftover_after_chunk =
        single_seq_prefill
            ? 0
            : (budget.remaining_token_budget > chunk_take
                   ? budget.remaining_token_budget - chunk_take
                   : 0);
    // Only preempt an in-progress cold chunk for a genuinely short single-step
    // request (<= min(max_chunk/2, 4096) tokens and at most half of rem_chunk),
    // preventing ~7K warm tails from starving 8K cold chunks.
    const size_t max_fast_preempt_tokens =
        std::min<size_t>(max_chunk / 2, 4096);
    const bool srpt_advantage = (rem_new * 2 <= rem_chunk);
    if (rem_new > 0 && rem_new <= max_fast_preempt_tokens && srpt_advantage) {
      const size_t block_size = static_cast<size_t>(
          std::max<int32_t>(1, state.kv_cache_manager->block_size()));
      for (const std::shared_ptr<Request>& req : state.chunk_queue) {
        if (req == nullptr || req->sequences().empty() ||
            req->sequences()[0] == nullptr) {
          continue;
        }
        Sequence* s = req->sequences()[0].get();
        const int32_t r = std::max<int32_t>(0, s->dp_rank());
        if (static_cast<size_t>(r) < reserved_full_footprint.size()) {
          reserved_full_footprint[r] +=
              (s->num_tokens() + block_size - 1) / block_size;
        }
      }
      if (rem_new > leftover_after_chunk) {
        run_one_step_alone = true;
      } else {
        fast_single_chunk_first = true;
      }
    }
  }

  if (fast_single_chunk_first) {
    schedule_prefill_from_queue(
        &state.prefill_queue, state, budget, finished, reserved_full_footprint);
    schedule_prefill_from_queue(
        &state.chunk_queue, state, budget, finished, reserved_full_footprint);
  } else if (run_one_step_alone) {
    schedule_prefill_from_queue(
        &state.prefill_queue, state, budget, finished, reserved_full_footprint);
    if (state.running_sequences.empty()) {
      schedule_prefill_from_queue(
          &state.chunk_queue, state, budget, finished, reserved_full_footprint);
    }
  } else {
    schedule_prefill_from_queue(
        &state.chunk_queue, state, budget, finished, reserved_full_footprint);
    if (!should_defer_new_prefill(state)) {
      schedule_prefill_from_queue(
          &state.prefill_queue, state, budget, finished, reserved_full_footprint);
    }
  }

  if (is_prefill_instance && oldest_chunk_req != nullptr &&
      !state.running_requests.empty()) {
    bool oldest_scheduled = false;
    for (const auto& r : state.running_requests) {
      if (r.get() == oldest_chunk_req) {
        oldest_scheduled = true;
        break;
      }
    }
    if (oldest_scheduled) {
      if (consecutive_fast_passes > 0 &&
          deferred_chunk_since != absl::InfinitePast()) {
        accumulated_deferred_ms_on_req += std::max(
            0.0, absl::ToDoubleMilliseconds(absl::Now() - deferred_chunk_since));
      }
      consecutive_fast_passes = 0;
      deferred_chunk_since = absl::InfinitePast();
    } else {
      if (consecutive_fast_passes == 0) {
        deferred_chunk_since = absl::Now();
      }
      ++consecutive_fast_passes;
      ++total_fast_passes_on_req;
      VLOG(1) << "[PREFILL_SRPT] deferred oldest_chunk="
              << oldest_chunk_req->request_id()
              << " step_passes=" << consecutive_fast_passes
              << " total_passes=" << total_fast_passes_on_req
              << " deferred_ms=" << chunk_deferred_ms
              << " rem_new=" << diag_rem_new
              << " rem_chunk=" << diag_rem_chunk;
    }
  }

  if (!state.running_sequences.empty()) {
    state.last_step_prefill = true;
  }

  // If no prefill sequences were scheduled, try decode.
  if (!retry_decode_before_restore && state.running_sequences.empty()) {
    budget.latency_budget = options_.max_global_tpot_ms();
    adjust_latency_budget_and_reorder(&state.decode_queue,
                                      /*second_queue=*/nullptr,
                                      budget.latency_budget,
                                      /*for_prefill=*/false,
                                      state);
    schedule_decode_from_queue(&state.decode_queue, state, budget);
  }
}

// =============================================================================
// PrefillFirstPolicy::adjust_latency_budget_and_reorder
// =============================================================================

void PrefillFirstPolicy::adjust_latency_budget_and_reorder(
    RequestPriorityQueue* first_queue,
    RequestPriorityQueue* second_queue,
    double& latency_budget,
    bool for_prefill,
    const SchedulerState& state) {
  if (batch_mode_.priority_strategy != "multi_slo_and_prio") {
    return;
  }
  bool both_empty = (first_queue == nullptr || first_queue->empty()) &&
                    (second_queue == nullptr || second_queue->empty());
  if (both_empty) {
    return;
  }
  CHECK(state.profile_manager != nullptr);

  const double constant_overhead =
      state.profile_manager->get_constant_overhead();
  double total_exec_time = 0.0;
  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();

  // Lambda to process one queue's requests for metrics.
  auto process_queue = [&](RequestPriorityQueue* queue) {
    if (queue == nullptr || queue->empty()) {
      return;
    }
    for (auto it = queue->begin(); it != queue->end(); ++it) {
      auto request = *it;
      auto& sequence = request->sequences()[0];
      sequence->set_estimated_latency(
          state.profile_manager->predict_step_time(sequence.get(), false));
      request->set_elapsed_time_ms();
      request->set_deadline_ms();
      request->set_starved(false);

      const int32_t remaining_time = request->get_remaining_time();
      total_exec_time += sequence->estimated_latency();
      if (request->tpot_slo_ms() < min_tpot) {
        min_tpot = request->tpot_slo_ms();
      }
      if (remaining_time < sequence->estimated_latency() + constant_overhead) {
        continue;
      }
      if (remaining_time < min_remaining_time) {
        min_remaining_time = remaining_time;
      }
    }
  };

  process_queue(first_queue);
  process_queue(second_queue);

  if (!for_prefill) {
    int32_t latency_budget_threshold = static_cast<int32_t>(0.65 * min_tpot);
    latency_budget = std::max(min_remaining_time, latency_budget_threshold);
  }

  const double lambda = SchedulerConfig::get_instance().aggressive_coeff();
  double load_judge_func = 0.0;
  if (for_prefill) {
    load_judge_func = total_exec_time + constant_overhead;
  } else {
    const double denominator =
        std::max(latency_budget - constant_overhead, 1e-6);
    load_judge_func = total_exec_time * latency_budget / denominator;
  }

  // Lambda to set urgency on one queue's requests.
  auto set_urgency = [&](RequestPriorityQueue* queue) {
    if (queue == nullptr || queue->empty()) {
      return;
    }
    for (auto it = queue->begin(); it != queue->end(); ++it) {
      auto request = *it;
      auto& sequence = request->sequences()[0];

      if (SchedulerConfig::get_instance().enable_starve_prevent()) {
        const int32_t starve_unit_time = sequence->is_prefill_stage()
                                             ? -request->ttft_slo_ms()
                                             : -request->tpot_slo_ms();
        const int32_t starve_time_threshold = static_cast<int32_t>(
            SchedulerConfig::get_instance().starve_threshold() *
            starve_unit_time);
        if (request->get_remaining_time() < starve_time_threshold) {
          request->set_starved(true);
        }
      }

      if (request->get_remaining_time() < lambda * load_judge_func) {
        request->set_urgency(Urgency::URGENT);
      } else {
        request->set_urgency(Urgency::NORMAL);
      }
    }
  };

  set_urgency(first_queue);
  set_urgency(second_queue);

  // Sort each queue independently with the same comparator.
  auto comparator = create_comparator("multi_slo_and_prio", true);
  if (first_queue != nullptr && !first_queue->empty()) {
    CHECK(first_queue->supports_sort())
        << "multi_slo_and_prio requires sortable request queue.";
    first_queue->sort(comparator);
  }
  if (second_queue != nullptr && !second_queue->empty()) {
    CHECK(second_queue->supports_sort())
        << "multi_slo_and_prio requires sortable request queue.";
    second_queue->sort(comparator);
  }

  // Adjust latency budget for prefill based on top request across both queues.
  if (for_prefill) {
    constexpr int32_t kSmallPositiveTimeMs = 2;
    if (min_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
      latency_budget = min_remaining_time;
    } else {
      // Find the top request across both queues.
      std::shared_ptr<Request> top_request;
      if (first_queue != nullptr && !first_queue->empty()) {
        top_request = first_queue->top();
      }
      if (second_queue != nullptr && !second_queue->empty()) {
        if (top_request == nullptr ||
            comparator(second_queue->top(), top_request)) {
          top_request = second_queue->top();
        }
      }
      if (top_request != nullptr) {
        const int32_t top_remaining_time = top_request->get_remaining_time();
        if (top_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
          latency_budget = top_remaining_time;
        } else {
          const auto& top_sequence = top_request->sequences()[0];
          latency_budget = top_sequence->estimated_latency() +
                           constant_overhead + kSmallPositiveTimeMs;
        }
      }
    }
  }
}

}  // namespace xllm
