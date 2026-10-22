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

#include "core/layers/musa/pcp_context.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/common/types.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/kernels/ops_api.h"
#include "core/util/env_var.h"

namespace xllm::layer::musa_pcp {
namespace {

// Splits land on the 256-row tile of the FP8 prefill GEMMs, where a partial
// tile costs a full one. That also keeps the 64-row Mate GDN chunk partition
// of the unsplit prompt.
constexpr int32_t kRowAlignment = 256;
// Handoffs pair the first rank with the second as broadcasts.
constexpr int32_t kSupportedCpSize = 2;
// Prefill cost of a rank relative to its per-row projection and MLP cost:
// causal attention over (query, key) pairs, whose keys include the rows
// already in the cache, and writing the K/V rows of the other rank into its
// cache. Measured for Qwen3.8 FP8 on MTT S5000.
constexpr double kAttentionCostPerPair = 9.4e-6;
constexpr double kOtherRankKvCostPerRow = 0.04;

bool pcp_enabled() {
  static const bool enabled = util::get_bool_env("XLLM_MUSA_PCP", true);
  return enabled;
}

int32_t num_ranks(const Context& context) {
  return static_cast<int32_t>(context.row_offsets.size()) - 1;
}

int32_t last_rank(const Context& context) { return num_ranks(context) - 1; }

int32_t rows_of(const Context& context, int32_t rank) {
  return context.row_offsets[rank + 1] - context.row_offsets[rank];
}

double rank_cost(int32_t begin,
                 int32_t end,
                 int32_t history_rows,
                 int32_t other_rank_kv_rows) {
  const double rows = static_cast<double>(end - begin);
  const double first_keys = static_cast<double>(history_rows) + begin;
  const double last_keys = static_cast<double>(history_rows) + end;
  const double pairs = 0.5 * (last_keys * last_keys - first_keys * first_keys);
  return rows + kAttentionCostPerPair * pairs +
         kOtherRankKvCostPerRow * other_rank_kv_rows;
}

// Returns the aligned row count of the first rank that best balances both
// ranks, or 0 when the prompt cannot give each rank a chunk. The second rank
// writes the K/V rows of the first inside the layers; the first writes those
// of the second after the layers only when the final state is replicated.
int32_t balanced_first_rows(int32_t num_rows,
                            int32_t history_rows,
                            bool replicate_final_state) {
  int32_t best_rows = 0;
  double best_gap = std::numeric_limits<double>::max();
  for (int32_t rows = kRowAlignment; rows < num_rows; rows += kRowAlignment) {
    const int32_t second_rows = num_rows - rows;
    const double first_cost = rank_cost(
        /*begin=*/0,
        rows,
        history_rows,
        /*other_rank_kv_rows=*/replicate_final_state ? second_rows : 0);
    const double second_cost = rank_cost(
        rows, num_rows, history_rows, /*other_rank_kv_rows=*/rows);
    const double gap = std::abs(first_cost - second_cost);
    if (gap < best_gap) {
      best_gap = gap;
      best_rows = rows;
    }
  }
  return best_rows;
}

torch::Tensor slots_of(const Context& context, int32_t rank) {
  return context.global_slot_mapping.narrow(
      /*dim=*/0,
      /*start=*/context.row_offsets[rank],
      /*length=*/rows_of(context, rank));
}

// The root never consumes what it sends, so only receivers make their stream
// wait for the transfer.
void broadcast_rows(torch::Tensor& rows, int32_t root_rank, Context& context) {
  if (context.cp_rank == root_rank) {
    context.cp_group->broadcast_async(rows, root_rank);
    context.in_flight_sends.emplace_back(rows);
    return;
  }
  context.cp_group->broadcast(rows, root_rank);
}

void broadcast_cache_rows(torch::Tensor& cache,
                          const torch::Tensor& indices,
                          int32_t root_rank,
                          Context& context) {
  torch::Tensor rows = cache.index_select(/*dim=*/0, indices);
  broadcast_rows(rows, root_rank, context);
  if (context.cp_rank != root_rank) {
    cache.index_put_({indices}, rows);
  }
}

void write_kv_rows(const torch::Tensor& kv_rows,
                   const torch::Tensor& slot_mapping,
                   const torch::Tensor& k_cache,
                   const torch::Tensor& v_cache,
                   int64_t num_kv_heads,
                   int64_t head_dim) {
  const int64_t kv_size = num_kv_heads * head_dim;
  xllm::kernel::ReshapePagedCacheParams params;
  params.key = kv_rows.narrow(/*dim=*/-1, /*start=*/0, /*length=*/kv_size)
                   .view({-1, num_kv_heads, head_dim});
  params.value =
      kv_rows.narrow(/*dim=*/-1, /*start=*/kv_size, /*length=*/kv_size)
          .view({-1, num_kv_heads, head_dim});
  params.k_cache = k_cache;
  params.v_cache = v_cache;
  params.slot_mapping = slot_mapping;
  xllm::kernel::reshape_paged_cache(params);
}

}  // namespace

void build_context(const AttentionMetadata& global_metadata,
                   const ParallelArgs& parallel_args,
                   int32_t num_rows,
                   std::optional<Context>* context) {
  context->reset();
  ProcessGroup* cp_group = parallel_args.cp_group_;
  if (!pcp_enabled() || cp_group == nullptr ||
      cp_group->world_size() != kSupportedCpSize) {
    return;
  }
  if (global_metadata.is_spec_verify || global_metadata.prefill_without_cache ||
      (!global_metadata.is_prefill && !global_metadata.is_chunked_prefill) ||
      global_metadata.q_seq_lens_vec.size() != 1 ||
      global_metadata.kv_seq_lens_vec.size() != 1) {
    return;
  }
  // Rows already in the cache: a prefix-cache hit or earlier prompt chunks.
  const int32_t history_rows =
      global_metadata.kv_seq_lens_vec[0] - global_metadata.q_seq_lens_vec[0];
  if (history_rows < 0) {
    return;
  }
  // Later prompt chunks and prefix-cache hits read every rank's cache. Only a
  // prefill instance whose prompt fits one chunk, with prefix caching off,
  // hands its cache to decode from the last rank alone.
  const bool single_chunk_prompt =
      global_metadata.is_prefill &&
      num_rows <
          SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const bool replicate_final_state =
      InstanceRole(::xllm::DisaggPDConfig::get_instance().instance_role()) !=
          InstanceRole::PREFILL ||
      KVCacheConfig::get_instance().enable_prefix_cache() ||
      !single_chunk_prompt;
  const int32_t first_rows =
      balanced_first_rows(num_rows, history_rows, replicate_final_state);
  if (first_rows == 0) {
    return;
  }

  Context& pcp_context = context->emplace();
  pcp_context.cp_group = cp_group;
  pcp_context.cp_rank = cp_group->rank();
  pcp_context.row_offsets = {0, first_rows, num_rows};
  pcp_context.max_rows = std::max(first_rows, num_rows - first_rows);
  pcp_context.replicate_final_state = replicate_final_state;
  pcp_context.global_slot_mapping = global_metadata.slot_mapping;

  // `num_rows` may include graph padding past the prompt, and a captured graph
  // replays any history length. Device lengths are derived from the global
  // ones on the stream; host lengths stay at the bounds used for launches.
  const int32_t begin = pcp_context.row_offsets[pcp_context.cp_rank];
  const int32_t end = pcp_context.row_offsets[pcp_context.cp_rank + 1];
  const int32_t q_len = end - begin;
  const bool continues_rows = begin > 0 || global_metadata.is_chunked_prefill;
  const torch::Device device = global_metadata.q_cu_seq_lens.device();

  AttentionMetadata& local = pcp_context.local_metadata;
  local = global_metadata;
  // A rank's keys cover the history and every row up to its own end.
  local.q_cu_seq_lens =
      (global_metadata.q_cu_seq_lens - begin).clamp(/*min=*/0, /*max=*/q_len);
  local.kv_cu_seq_lens =
      global_metadata.kv_cu_seq_lens -
      (global_metadata.q_cu_seq_lens - end).clamp_min(/*min=*/0);
  const torch::Tensor rows_after_end =
      (global_metadata.q_cu_seq_lens.narrow(
           /*dim=*/0, /*start=*/1, /*length=*/1) -
       end)
          .clamp_min(/*min=*/0);
  if (global_metadata.q_seq_lens.defined()) {
    local.q_seq_lens =
        (global_metadata.q_seq_lens - begin).clamp(/*min=*/0, /*max=*/q_len);
  }
  if (global_metadata.kv_seq_lens.defined()) {
    local.kv_seq_lens = global_metadata.kv_seq_lens - rows_after_end;
  }
  local.q_seq_lens_vec = {q_len};
  local.kv_seq_lens_vec = {history_rows + end};
  local.slot_mapping = global_metadata.slot_mapping.narrow(
      /*dim=*/0, /*start=*/begin, /*length=*/q_len);
  if (global_metadata.mrope_cos.defined()) {
    local.mrope_cos = global_metadata.mrope_cos.narrow(
        /*dim=*/0, /*start=*/begin, /*length=*/q_len);
    local.mrope_sin = global_metadata.mrope_sin.narrow(
        /*dim=*/0, /*start=*/begin, /*length=*/q_len);
  }
  local.max_query_len = q_len;
  // A chunked graph replays any history, so its key length bound is the
  // block-table capacity.
  local.max_seq_len =
      global_metadata.is_chunked_prefill
          ? global_metadata.block_table.size(/*dim=*/1) *
                static_cast<int64_t>(KVCacheConfig::get_instance().block_size())
          : end;
  local.total_kv_len = history_rows + end;
  local.is_prefill = !continues_rows;
  local.is_chunked_prefill = continues_rows;
  local.has_initial_states =
      begin > 0 ? torch::full(
                      {1},
                      /*fill_value=*/true,
                      torch::TensorOptions().dtype(torch::kBool).device(device))
                : global_metadata.has_initial_states;
  local.fa3_metadata.fa3_scheduler_metadata = torch::Tensor();
  local.pcp_context = &pcp_context;
}

bool receives_prefix_state(const Context& context) {
  return context.cp_rank > 0;
}

torch::Tensor shard_rows(const torch::Tensor& global,
                         const Context& context,
                         int64_t dim) {
  return global
      .narrow(dim,
              context.row_offsets[context.cp_rank],
              rows_of(context, context.cp_rank))
      .contiguous();
}

torch::Tensor gather_rows(const torch::Tensor& local, const Context& context) {
  const int32_t local_rows = static_cast<int32_t>(local.size(0));
  torch::Tensor send = local.contiguous();
  if (local_rows < context.max_rows) {
    std::vector<int64_t> pad_shape = send.sizes().vec();
    pad_shape[0] = context.max_rows - local_rows;
    send = torch::cat({send, torch::empty(pad_shape, send.options())},
                      /*dim=*/0);
  }
  // [cp_size, max_rows, ...]
  torch::Tensor gathered = context.cp_group->allgather_base_sync(send);
  const int32_t cp_size = num_ranks(context);
  if (context.row_offsets.back() == cp_size * context.max_rows) {
    return gathered.flatten(/*start_dim=*/0, /*end_dim=*/1);
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(cp_size);
  for (int32_t rank = 0; rank < cp_size; ++rank) {
    parts.emplace_back(gathered[rank].narrow(
        /*dim=*/0, /*start=*/0, /*length=*/rows_of(context, rank)));
  }
  return torch::cat(parts, /*dim=*/0);
}

void expect_prefix_rows(torch::IntArrayRef shape,
                        const torch::TensorOptions& options,
                        Context& context) {
  if (context.cp_rank == 0) {
    return;
  }
  PendingReceive receive;
  receive.rows = torch::empty(shape, options);
  receive.work = context.cp_group->broadcast_async(
      receive.rows, /*root_rank=*/context.cp_rank - 1);
  context.pending_receives.emplace_back(std::move(receive));
}

void expect_prefix_cache_row(const torch::Tensor& cache, Context& context) {
  std::vector<int64_t> shape = cache.sizes().vec();
  shape[0] = 1;
  expect_prefix_rows(shape, cache.options(), context);
}

torch::Tensor take_prefix_rows(Context& context) {
  CHECK(!context.pending_receives.empty())
      << "context-parallel prefill took rows that were never posted";
  PendingReceive receive = std::move(context.pending_receives.front());
  context.pending_receives.pop_front();
  if (receive.work.defined()) {
    receive.work->wait();
  }
  return receive.rows;
}

void receive_prefix_cache_row(torch::Tensor& cache,
                              const torch::Tensor& indices,
                              Context& context) {
  if (context.cp_rank == 0) {
    return;
  }
  cache.index_put_({indices}, take_prefix_rows(context));
}

void send_prefix_rows(const torch::Tensor& rows, Context& context) {
  if (context.cp_rank + 1 >= num_ranks(context)) {
    return;
  }
  torch::Tensor send = rows.contiguous();
  broadcast_rows(send, /*root_rank=*/context.cp_rank, context);
}

void send_prefix_cache_row(const torch::Tensor& cache,
                           const torch::Tensor& indices,
                           Context& context) {
  if (context.cp_rank + 1 >= num_ranks(context)) {
    return;
  }
  send_prefix_rows(cache.index_select(/*dim=*/0, indices), context);
}

void expect_prefix_kv(int64_t num_kv_heads,
                      int64_t head_dim,
                      const torch::TensorOptions& options,
                      Context& context) {
  const int64_t first_rows = rows_of(context, /*rank=*/0);
  expect_prefix_rows(
      {first_rows, 2 * num_kv_heads * head_dim}, options, context);
}

void exchange_kv(const torch::Tensor& key,
                 const torch::Tensor& value,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 KVCache& kv_cache,
                 Context& context) {
  const int64_t kv_size = num_kv_heads * head_dim;
  torch::Tensor local_kv = torch::cat(
      {key.reshape({-1, kv_size}), value.reshape({-1, kv_size})}, /*dim=*/-1);
  const torch::Tensor k_cache = kv_cache.get_k_cache();
  const torch::Tensor v_cache = kv_cache.get_v_cache();
  write_kv_rows(local_kv,
                slots_of(context, context.cp_rank),
                k_cache,
                v_cache,
                num_kv_heads,
                head_dim);
  if (context.cp_rank == 0) {
    send_prefix_rows(local_kv, context);
  } else {
    write_kv_rows(take_prefix_rows(context),
                  slots_of(context, /*rank=*/0),
                  k_cache,
                  v_cache,
                  num_kv_heads,
                  head_dim);
  }
  if (!context.replicate_final_state) {
    return;
  }
  const int32_t root_rank = last_rank(context);
  DeferredKv deferred;
  deferred.rows = context.cp_rank == root_rank
                      ? local_kv
                      : torch::empty({rows_of(context, root_rank), 2 * kv_size},
                                     local_kv.options());
  deferred.k_cache = k_cache;
  deferred.v_cache = v_cache;
  deferred.num_kv_heads = num_kv_heads;
  deferred.head_dim = head_dim;
  context.deferred_kv.emplace_back(std::move(deferred));
}

void defer_final_state(const torch::Tensor& cache,
                       const torch::Tensor& indices,
                       Context& context) {
  if (!context.replicate_final_state) {
    return;
  }
  DeferredState deferred;
  deferred.cache = cache;
  deferred.indices = indices;
  context.deferred_states.emplace_back(std::move(deferred));
}

torch::Tensor finish(const torch::Tensor& local_hidden_states,
                     Context& context) {
  const int32_t root_rank = last_rank(context);
  for (DeferredKv& deferred : context.deferred_kv) {
    broadcast_rows(deferred.rows, root_rank, context);
    if (context.cp_rank != root_rank) {
      write_kv_rows(deferred.rows,
                    slots_of(context, root_rank),
                    deferred.k_cache,
                    deferred.v_cache,
                    deferred.num_kv_heads,
                    deferred.head_dim);
    }
  }
  for (DeferredState& deferred : context.deferred_states) {
    broadcast_cache_rows(deferred.cache, deferred.indices, root_rank, context);
  }
  context.deferred_kv.clear();
  context.deferred_states.clear();
  torch::Tensor hidden_states = gather_rows(local_hidden_states, context);
  context.in_flight_sends.clear();
  return hidden_states;
}

}  // namespace xllm::layer::musa_pcp
