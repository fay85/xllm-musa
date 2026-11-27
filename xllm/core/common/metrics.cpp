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

#include "common/metrics.h"

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <string>

#include "core/common/types.h"

namespace xllm {

bvar::Adder<int64_t> COUNTER_engine_inflight_requests(
    "engine_inflight_requests");

void record_request_status(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      COUNTER_INC(request_status_total_ok);
      break;
    case StatusCode::CANCELLED:
      COUNTER_INC(request_status_total_cancelled);
      break;
    case StatusCode::UNKNOWN:
      COUNTER_INC(request_status_total_unknown);
      break;
    case StatusCode::INVALID_ARGUMENT:
      COUNTER_INC(request_status_total_invalid_argument);
      break;
    case StatusCode::DEADLINE_EXCEEDED:
      COUNTER_INC(request_status_total_deadline_exceeded);
      break;
    case StatusCode::RESOURCE_EXHAUSTED:
      COUNTER_INC(request_status_total_resource_exhausted);
      break;
    default:
      COUNTER_INC(request_status_total_unknown);
      break;
  }
}

void set_multi_gauge_value(
    bvar::MultiDimension<bvar::Status<double>>& metric,
    const bvar::MultiDimension<bvar::Status<double>>::key_type& labels,
    double value) {
  bvar::Status<double>* gauge = metric.get_stats(labels);
  if (gauge == nullptr) {
    LOG_EVERY_N(WARNING, 100) << "Failed to update multi gauge for labels "
                              << absl::StrJoin(labels, ", ");
    return;
  }
  gauge->set_value(value);
}

}  // namespace xllm

// llm server impl metrics
DEFINE_COUNTER(request_status_total_ok, "Total number of request status OK");
DEFINE_COUNTER(request_status_total_cancelled,
               "Total number of request status CANCELLED");
DEFINE_COUNTER(request_status_total_unknown,
               "Total number of request status UNKNOWN");
DEFINE_COUNTER(request_status_total_invalid_argument,
               "Total number of request status INVALID_ARGUMENT");
DEFINE_COUNTER(request_status_total_deadline_exceeded,
               "Total number of request status DEADLINE_EXCEEDED");
DEFINE_COUNTER(request_status_total_resource_exhausted,
               "Total number of request status RESOURCE_EXHAUSTED");
DEFINE_COUNTER(request_status_total_unauthenticated,
               "Total number of request status UNAUTHENTICATED");
DEFINE_COUNTER(request_status_total_unavailable,
               "Total number of request status UNAVAILABLE");
DEFINE_COUNTER(request_status_total_unimplemented,
               "Total number of request status UNIMPLEMENTED");

DEFINE_COUNTER(request_handling_latency_seconds_chat,
               "Latency of chat request handling in seconds");
DEFINE_COUNTER(request_handling_latency_seconds_completion,
               "Latency of completion request handling in seconds");

DEFINE_COUNTER(tokenization_latency_seconds,
               "Prompt tokenization latency in seconds");
DEFINE_COUNTER(chat_template_latency_seconds,
               "Chat template latency in seconds");

// multimodal metrics
DEFINE_HISTOGRAM(mm_transfer_latency_milliseconds,
                 "Latency of multimodal input transfer in milliseconds");
DEFINE_HISTOGRAM(mm_processor_latency_milliseconds,
                 "Latency of multimodal input processor in milliseconds");
DEFINE_MULTI_COUNTER(mm_input_stage_failures_total,
                     "stage",
                     "Failed multimodal request stages: transfer or processor");
DEFINE_MULTI_COUNTER_3_LABELS(mm_input_items_total,
                              "modality",
                              "representation",
                              "result",
                              "Completed multimodal input handler calls");
DEFINE_MULTI_COUNTER_2_LABELS(mm_cache_lookups_total,
                              "cache",
                              "outcome",
                              "Actual multimodal cache lookups");

// DiT request metrics
DEFINE_HISTOGRAM(dit_input_parse_latency_milliseconds,
                 "Host latency of actual DiT input parsing in milliseconds");
DEFINE_HISTOGRAM(dit_output_build_latency_milliseconds,
                 "Host latency of DiT output construction in milliseconds");
DEFINE_MULTI_HISTOGRAM(dit_input_stage_host_latency_microseconds,
                       "stage",
                       "Host latency of DiT media decode or tensor "
                       "deserialization in microseconds");
DEFINE_MULTI_COUNTER(
    dit_input_stage_failures_total,
    "stage",
    "Normal-return failures of DiT media decode or tensor deserialization");

DEFINE_HISTOGRAM(encoder_cache_hit_rate,
                 "Histogram of encoder cache hit rate in percent");

// Resident blocks/state slots, including evictable prefixes. Host copies use
// their own manager.
bvar::MultiDimension<bvar::Status<double>>
    MULTI_GAUGE_kv_cache_pool_occupied_resources(
        "kv_cache_pool_occupied_resources",
        {"model_id", "server_idx", "dp_rank", "block_type", "memory_type"});

// block manager metrics
DEFINE_COUNTER(prefix_cache_latency_seconds_insert,
               "Latency of prefix cache insert in seconds");
DEFINE_COUNTER(prefix_cache_latency_seconds_match,
               "Latency of prefix cache match in seconds");
DEFINE_COUNTER(prefix_cache_latency_seconds_evict,
               "Latency of prefix cache evict in seconds");
DEFINE_HISTOGRAM(prefix_cache_operation_latency_microseconds_insert,
                 "Individual prefix-cache insert operation duration");
DEFINE_HISTOGRAM(prefix_cache_operation_latency_microseconds_match,
                 "Individual prefix-cache match operation duration");
DEFINE_HISTOGRAM(prefix_cache_operation_latency_microseconds_evict,
                 "Individual prefix-cache evict operation duration");
DEFINE_COUNTER(prefix_cache_invalid_hit_samples_total,
               "Rejected paired prefix-hit samples whose hits exceed prompt");

DEFINE_COUNTER(prefix_cache_prompt_tokens_total,
               "Prompt tokens paired with first-allocation stage prefix hits");
DEFINE_COUNTER(prefix_cache_hit_tokens_total,
               "Process-cumulative prefix-cache hit tokens");
DEFINE_GAUGE(prefix_cache_token_hit_rate_perc,
             "Process-cumulative prefix-cache token hit rate in percent");

DEFINE_MULTI_COUNTER(store_prefetch_master_sessions_total,
                     "stop_reason",
                     "Measured Store prefetch sessions by bounded stop reason");
DEFINE_COUNTER(store_prefetch_master_prompt_tokens_total,
               "Prompt tokens in measured Master Store prefetch sessions");
DEFINE_COUNTER(store_prefetch_master_host_hit_tokens_total,
               "Host prefix tokens present before the measured Store GET");
DEFINE_COUNTER(store_prefetch_master_target_tokens_total,
               "Total cacheable target tokens, including the Host prefix");
DEFINE_COUNTER(store_prefetch_master_fetched_tokens_total,
               "Effective Store-fetched tokens retained for publication");
DEFINE_HISTOGRAM(store_prefetch_master_latency_microseconds,
                 "Master prefetch session duration including tier queries");
DEFINE_MULTI_COUNTER(store_prefetch_worker_sessions_total,
                     "status",
                     "Measured Worker sessions: completed or failed");
DEFINE_COUNTER(store_prefetch_worker_requested_units_total,
               "Units requested for measured Worker sessions");
DEFINE_COUNTER(store_prefetch_worker_executed_units_total,
               "Real units attempted, excluding batch wire padding");
DEFINE_COUNTER(store_prefetch_worker_hit_units_total,
               "Gated-hit units, including reads beyond a prefix miss");
DEFINE_COUNTER(store_prefetch_worker_requested_blocks_total,
               "Logical blocks requested in executed GET batches");
DEFINE_COUNTER(store_prefetch_worker_hit_blocks_total,
               "Logical block GET hits, not reusable prefix tokens");
DEFINE_HISTOGRAM(store_prefetch_worker_hit_blocks_per_session,
                 "Logical GET hits per closed Worker session, including zero");
DEFINE_HISTOGRAM(
    store_prefetch_worker_block_hit_rate_basis_points,
    "Per-session GET block hit rate in basis points; nonempty GETs");
// Successfully read physical bytes, including unusable tails. Integers retain
// small increments at TiB-scale totals through all bvar serialization paths.
bvar::Adder<uint64_t> COUNTER_store_prefetch_worker_read_bytes_total(
    "store_prefetch_worker_read_bytes_total");
bvar::MultiDimension<bvar::Adder<uint64_t>>
    MULTI_COUNTER_store_prefetch_worker_read_tier_objects_total(
        "store_prefetch_worker_read_tier_objects_total",
        {"tier"});
bvar::MultiDimension<bvar::Adder<uint64_t>>
    MULTI_COUNTER_store_prefetch_worker_read_tier_bytes_total(
        "store_prefetch_worker_read_tier_bytes_total",
        {"tier"});
DEFINE_COUNTER(store_prefetch_worker_batches_total,
               "GET batches executed in measured Worker sessions");
DEFINE_MULTI_HISTOGRAM(store_prefetch_worker_latency_microseconds,
                       "stage",
                       "Per-session queue_wait/get/tier_query/control_wait/"
                       "total duration; total excludes post-close probing");
DEFINE_MULTI_COUNTER(store_prefetch_worker_probe_sessions_total,
                     "status",
                     "Post-close probe results: complete, invalid or skipped");
DEFINE_COUNTER(store_prefetch_worker_probed_units_total,
               "Tail units probed, not additional GET or reusable prefix hits");
DEFINE_COUNTER(store_prefetch_worker_probed_present_units_total,
               "Present tail units in valid rank-local metadata snapshots");

DEFINE_COUNTER(allocate_blocks_latency_seconds,
               "Latency of blocks allocation in seconds");

namespace {

std::mutex prefix_cache_hit_metrics_mutex;
xllm::PrefixCacheHitWindow prefix_cache_hit_window;

}  // namespace

bvar::PassiveStatus<double> GAUGE_prefix_cache_token_hit_rate_window_perc(
    "prefix_cache_token_hit_rate_window_perc",
    [](void*) { return prefix_cache_hit_window.hit_rate(); },
    nullptr);

bvar::PassiveStatus<int64_t> GAUGE_prefix_cache_token_hit_rate_window_valid(
    "prefix_cache_token_hit_rate_window_valid",
    [](void*) -> int64_t {
      return prefix_cache_hit_window.has_samples() ? 1 : 0;
    },
    nullptr);

namespace xllm {

void record_prefix_cache_hit_metrics(size_t prompt_tokens, size_t hit_tokens) {
  if (hit_tokens > prompt_tokens) {
    COUNTER_INC(prefix_cache_invalid_hit_samples_total);
    LOG_EVERY_N(WARNING, 100)
        << "Rejected paired prefix-cache hit sample: hit=" << hit_tokens
        << ", prompt=" << prompt_tokens;
    return;
  }
  if (prompt_tokens == 0) {
    return;
  }

  const std::lock_guard<std::mutex> lock(prefix_cache_hit_metrics_mutex);
  prefix_cache_hit_window.record(prompt_tokens, hit_tokens);
  COUNTER_ADD(prefix_cache_prompt_tokens_total, prompt_tokens);
  COUNTER_ADD(prefix_cache_hit_tokens_total, hit_tokens);
  const double prompt_total = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_total = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  GAUGE_SET(prefix_cache_token_hit_rate_perc, hit_total * 100.0 / prompt_total);
}

}  // namespace xllm

// sequence metrics
DEFINE_COUNTER(detokenization_latency_seconds_stream,
               "Latency of stream detokenization in seconds");
DEFINE_COUNTER(detokenization_latency_seconds_non_stream,
               "Latency of non-stream detokenization in seconds");

// executor metrics
DEFINE_COUNTER(num_model_execution_total_eager,
               "Total number of model execution");

DEFINE_COUNTER(mooncake_transfer_completed_total_read,
               "Total number of completed MoonCake READ transfers");
DEFINE_COUNTER(mooncake_transfer_completed_total_write,
               "Total number of completed MoonCake WRITE transfers");
DEFINE_COUNTER(mooncake_transfer_bytes_total_read,
               "Total bytes completed by MoonCake READ transfers");
DEFINE_COUNTER(mooncake_transfer_bytes_total_write,
               "Total bytes completed by MoonCake WRITE transfers");
DEFINE_COUNTER(mooncake_transfer_failed_total,
               "Total number of failed MoonCake transfers");
DEFINE_COUNTER(
    kv_transfer_destruction_pending_total,
    "Total KV transfers a worker step left pending at KVTransferCompletion "
    "destruction without calling wait()");
DEFINE_COUNTER(
    kv_transfer_destruction_abandoned_total,
    "Total KV transfers abandoned at KVTransferCompletion destruction after "
    "cancel and drain; their push threads ignored abort and may still read "
    "source KV blocks");
DEFINE_HISTOGRAM(mooncake_transfer_latency_microseconds_read,
                 "MoonCake READ transfer latency in microseconds");
DEFINE_HISTOGRAM(mooncake_transfer_latency_microseconds_write,
                 "MoonCake WRITE transfer latency in microseconds");

// worker metrics
DEFINE_COUNTER(execution_latency_seconds_model,
               "Latency of model execution in seconds");
DEFINE_COUNTER(execution_latency_seconds_logits_processing,
               "Latency of logits processing in seconds");
DEFINE_COUNTER(execution_latency_seconds_sampling,
               "Latency of sampling in seconds");

DEFINE_COUNTER(json_object_mask_cache_hits_total,
               "JSON object mask cache hit count");
DEFINE_COUNTER(json_object_mask_cache_misses_total,
               "JSON object mask cache miss count");
DEFINE_HISTOGRAM(json_object_mask_vocab_scan_latency_microseconds,
                 "JSON object mask vocabulary scan latency in microseconds");
DEFINE_HISTOGRAM(json_object_mask_batch_build_latency_microseconds,
                 "JSON object CPU mask batch build latency in microseconds");
DEFINE_HISTOGRAM(json_object_mask_device_copy_latency_microseconds,
                 "JSON object blocking device copy latency in microseconds");
DEFINE_HISTOGRAM(
    json_object_mask_transfer_submission_latency_microseconds,
    "JSON object non-blocking device copy submission latency in microseconds");
DEFINE_COUNTER(json_object_mask_build_calls_normal_total,
               "JSON object normal mask build calls");
DEFINE_COUNTER(json_object_mask_build_calls_target_total,
               "JSON object target mask build calls");
DEFINE_COUNTER(json_object_mask_build_rows_normal_total,
               "JSON object normal mask build rows");
DEFINE_COUNTER(json_object_mask_build_rows_target_total,
               "JSON object target mask build rows");
DEFINE_COUNTER(json_object_mask_build_constrained_rows_normal_total,
               "JSON object constrained rows in normal mask builds");
DEFINE_COUNTER(json_object_mask_build_constrained_rows_target_total,
               "JSON object constrained rows in target mask builds");

// scheduler metrics
DEFINE_GAUGE(num_pending_requests,
             "Requests queued or executing in master preprocessing, sampled "
             "at scheduler reporting time");
DEFINE_MULTI_GAUGE_2_LABELS(scheduler_queue_requests,
                            "queue",
                            "server_idx",
                            "Scheduler-owned queue and preparation entries, "
                            "excluding selected batch requests; asynchronous "
                            "entries are sampled");
DEFINE_MULTI_GAUGE(scheduler_waiting_requests,
                   "server_idx",
                   "Sampled queued scheduler entries, excluding active "
                   "dispatch, resource preparation, master preprocessing and "
                   "selected batches; "
                   "asynchronous handoff samples are not transactional");
DEFINE_GAUGE(num_running_requests, "Number of running requests in scheduler");
DEFINE_GAUGE(num_waiting_requests, "Number of waiting requests in scheduler");
DEFINE_GAUGE(num_preempted_requests,
             "Number of preempted requests in scheduler");
DEFINE_COUNTER(scheduler_preempted_requests_total,
               "Scheduling preemption events, including repeated preemptions");
DEFINE_GAUGE(num_offline_decode_preempt_offline_requests,
             "Number of offline decode preempt offline requests in scheduler");
DEFINE_GAUGE(num_online_decode_preempt_online_requests,
             "Number of online decode preempt online requests in scheduler");
DEFINE_GAUGE(num_online_prefill_preempt_offline_requests,
             "Number of online prefill preempt offline requests in scheduler");
DEFINE_GAUGE(num_online_decode_preempt_offline_requests,
             "Number of online decode preempt offline requests in scheduler");

DEFINE_GAUGE(num_running_sequences, "Number of running sequences");

DEFINE_GAUGE(kv_cache_utilization_perc,
             "Minimum DP admission pressure as an unscaled ratio: maximum "
             "used / minimum capacity across admission leaves; may exceed 1, "
             "not physical device memory occupancy");
DEFINE_MULTI_GAUGE(kv_cache_dp_utilization_perc,
                   "dp_rank",
                   "DP admission pressure in percent: maximum used / minimum "
                   "capacity across admission leaves; may exceed 100, not "
                   "physical device memory occupancy");
DEFINE_MULTI_GAUGE_2_LABELS(block_type_dp_utilization_perc,
                            "block_type",
                            "dp_rank",
                            "Effective per-leaf block/slot utilization per DP "
                            "in percent: evictable prefix-cache residents "
                            "excluded, not physical slot or memory occupancy");
DEFINE_GAUGE(kv_cache_max_utilization_perc,
             "Maximum DP admission pressure in percent: maximum used / "
             "minimum capacity across admission leaves; may exceed 100, not "
             "physical device memory occupancy");
DEFINE_GAUGE(num_blocks_in_prefix_cache,
             "Number of blocks in the prefix cache");
DEFINE_GAUGE(num_free_blocks, "Maximum free admission-cache blocks across DP");
DEFINE_GAUGE(num_used_blocks,
             "Minimum effective used admission-cache blocks across DP");

DEFINE_COUNTER(scheduling_latency_seconds, "Latency of scheduling in seconds");

DEFINE_COUNTER(num_processing_tokens_total_prompt,
               "Total number of processing prompt tokens");
DEFINE_COUNTER(num_processing_tokens_total_generated,
               "Total number of processing generated tokens");
DEFINE_COUNTER(request_input_tokens_total,
               "Prompt tokens counted once at online request admission");
DEFINE_COUNTER(request_output_tokens_total,
               "Confirmed generated tokens in accepted online response deltas, "
               "including tokens hidden by decoding or stop rules");

DEFINE_HISTOGRAM(num_prompt_tokens_per_request,
                 "Histogram of the prompt token number per request");

DEFINE_HISTOGRAM(num_generated_tokens_per_request,
                 "Histogram of the generated token number per request");

// Preserve the legacy sequence-local TTFT alongside the request-level metric.
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
DEFINE_HISTOGRAM(server_time_to_first_token_latency_milliseconds,
                 "Time from preprocessing admission to scheduler first-token "
                 "result readiness, or accepted PD ingress handoff ACK, once "
                 "per request, in milliseconds; excludes detokenization and "
                 "HTTP delivery");
// Request-level phase latencies. Microseconds preserve sub-millisecond phases.
// Only complete local single-output samples enter these recorders.
DEFINE_HISTOGRAM(xllm_request_ttft_microseconds,
                 "Request server time-to-first-token latency in microseconds");
DEFINE_HISTOGRAM(xllm_request_queue_microseconds,
                 "Request queue-wait latency in microseconds");
DEFINE_HISTOGRAM(xllm_request_prefill_microseconds,
                 "Request prefill latency in microseconds");
DEFINE_HISTOGRAM(xllm_request_decode_microseconds,
                 "Request decode latency in microseconds");
DEFINE_HISTOGRAM(xllm_request_tpot_microseconds,
                 "Request mean time-per-output-token in microseconds");
// PD samples are separate from complete local requests. Imported upstream
// Queue/Prefill samples are never re-recorded on Decode.
DEFINE_MULTI_HISTOGRAM(xllm_pd_request_queue_microseconds,
                       "role",
                       "PD role-local admitted-to-first-batch wait in "
                       "microseconds, independently for prefill and decode");
DEFINE_MULTI_HISTOGRAM(xllm_pd_request_prefill_microseconds,
                       "role",
                       "PD Prefill first-batch-to-first-core-output latency "
                       "in microseconds, recorded at stage completion");
DEFINE_MULTI_HISTOGRAM(xllm_pd_request_decode_microseconds,
                       "role",
                       "Successful PD Decode local received-token-to-last-core-"
                       "output latency in microseconds, including local "
                       "preparation and scheduling waits");
DEFINE_MULTI_HISTOGRAM(xllm_pd_request_tpot_microseconds,
                       "role",
                       "PD local Decode duration per newly committed local "
                       "token in microseconds; received tokens excluded");
bvar::MultiDimension<bvar::Adder<int64_t>>
    MULTI_COUNTER_xllm_request_phase_skipped_total(
        "xllm_request_phase_skipped_total",
        {"phase", "reason"});
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");
DEFINE_HISTOGRAM(inter_token_latency_microseconds,
                 "Histogram of inter token latency in microseconds");

// response metrics
DEFINE_COUNTER(responsing_latency_seconds_stream,
               "Latency of stream responding in seconds");
DEFINE_COUNTER(responsing_latency_seconds_non_stream,
               "Latency of non-stream responding in seconds");

DEFINE_HISTOGRAM(end_2_end_latency_milliseconds,
                 "Histogram of end to end latency in milliseconds");

// xllm service metrics
DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");

DEFINE_COUNTER(server_request_total_ok,
               "Total number of ok request that server processed");
DEFINE_COUNTER(server_request_total_limit,
               "Total number of limit request that server processed");
DEFINE_COUNTER(server_request_total_fail,
               "Total number of fail request that server processed");
DEFINE_MULTI_COUNTER(api_request_terminal_total,
                     "outcome",
                     "Terminal outcome of inference API entries");

DEFINE_GAUGE(num_concurrent_requests,
             "Number of concurrent requests in server");

DEFINE_GAUGE(xllm_cpu_num, "The cpu number pre instance for xllm");
DEFINE_GAUGE(xllm_cpu_utilization, "The cpu utilization pre instance for xllm");
DEFINE_GAUGE(xllm_gpu_num, "The gpu number pre instance for xllm");
DEFINE_GAUGE(xllm_gpu_utilization, "The gpu utilization pre instance for xllm");

// speculative metrics
DEFINE_COUNTER(speculative_execution_latency_seconds_draft,
               "Latency of draft execution in seconds");
DEFINE_COUNTER(speculative_execution_latency_seconds_target,
               "Latency of target execution in seconds");
DEFINE_COUNTER(speculative_execution_latency_seconds_validation,
               "Latency of validation in seconds");

DEFINE_COUNTER(speculative_num_drafts_total,
               "Total number of speculative proposal sequences");
DEFINE_COUNTER(speculative_num_accepted_tokens_total,
               "Total number of accepted tokens in validation");
DEFINE_COUNTER(speculative_num_draft_tokens_total,
               "Total number of draft tokens");
DEFINE_COUNTER(speculative_num_committed_tokens_total,
               "Total number of tokens committed by speculative decode");
DEFINE_MULTI_COUNTER(speculative_num_accepted_tokens_per_pos,
                     "position",
                     "Accepted speculative tokens by zero-based draft "
                     "position");
DEFINE_GAUGE(speculative_mean_acceptance_length,
             "Cumulative acceptance length: mean tokens committed per "
             "speculative proposal sequence, including the target bonus "
             "token (1.0 without accepted draft tokens)");
DEFINE_COUNTER(speculative_num_accepted_tokens_constrained_total,
               "Accepted speculative tokens from constrained rows");
DEFINE_COUNTER(speculative_num_accepted_tokens_plain_total,
               "Accepted speculative tokens from plain rows");
DEFINE_COUNTER(speculative_num_draft_tokens_constrained_total,
               "Draft speculative tokens from constrained rows");
DEFINE_COUNTER(speculative_num_draft_tokens_plain_total,
               "Draft speculative tokens from plain rows");

// proto metrics
DEFINE_COUNTER(proto_latency_seconds_proto2i,
               "Latency of proto2i convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_i2proto,
               "Latency of i2proto convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_proto2o,
               "Latency of proto2o convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_o2proto,
               "Latency of o2proto convert in seconds");

// engine metrics
DEFINE_COUNTER(prepare_input_latency_seconds,
               "Latency of preparing input in seconds");

DEFINE_HISTOGRAM(prepare_input_latency_microseconds,
                 "Latency of preparing input in microseconds");
// multi node metrics
DEFINE_COUNTER(worker_service_latency_seconds,
               "Worker service execution latency in seconds");
DEFINE_COUNTER(engine_latency_seconds, "Engine execution latency in seconds");

// memory metrics
DEFINE_GAUGE(total_memory_size_in_kilobytes, "Total memory size in kilobytes");
DEFINE_GAUGE(weight_size_in_kilobytes, "Weight size in kilobytes");
DEFINE_GAUGE(total_kv_cache_size_in_kilobytes, "KV cache size in kilobytes");
DEFINE_GAUGE(total_activation_size_in_kilobytes,
             "Total activation size in kilobytes");

DEFINE_MULTI_HISTOGRAM(active_kv_cache_size_in_kilobytes,
                       "dp_rank",
                       "Active kv cache size in kilobytes per dp rank");
DEFINE_MULTI_HISTOGRAM(
    prefill_active_activation_size_in_kilobytes,
    "dp_rank",
    "Active activation size in kilobytes per dp rank during prefill phase");
DEFINE_MULTI_HISTOGRAM(
    decode_active_activation_size_in_kilobytes,
    "dp_rank",
    "Active activation size in kilobytes per dp rank during decode phase");

DEFINE_COUNTER(engine_heartbeat_total,
               "Total Native Engine heartbeat attempts");
DEFINE_COUNTER(engine_heartbeat_failures_total,
               "Total failed Native Engine heartbeat attempts");
DEFINE_COUNTER(kv_event_publish_failures_total,
               "Total failed Native KV event publish attempts");
DEFINE_HISTOGRAM(kv_event_publish_batch_size,
                 "Number of Native KV events in one publish attempt");
DEFINE_GAUGE(kv_event_queue_length,
             "Pending Native KV events waiting to be published");
