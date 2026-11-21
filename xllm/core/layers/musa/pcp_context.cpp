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
#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/kernels/ops_api.h"
#include "core/util/env_var.h"

namespace xllm::layer::musa_pcp {
namespace {

// Splits land on the 256-row tile of the FP8 prefill GEMMs, where a partial
// tile costs a full one. That also keeps the 64-row Mate GDN chunk partition
// of the unsplit prompt.
constexpr int32_t kRowAlignment = 256;
// Multi-sequence PCP splits each sequence on a 64-token Mate GDN chunk boundary
// so the chunked recurrence across rank 0 and rank 1 matches an unsplit prompt.
constexpr int32_t kBatchRowAlignment = 64;
constexpr int32_t kMinBatchSeqRows = 128;
constexpr int32_t kMinBatchTailRows = 32;
constexpr int32_t kDefaultConvStateLen = 3;
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

bool pcp_multibatch_enabled() {
  return util::get_bool_env("XLLM_MUSA_PCP_MULTIBATCH", true);
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
                            bool replicate_final_state,
                            int32_t alignment = kRowAlignment,
                            int32_t min_second_rows = 1) {
  int32_t best_rows = 0;
  double best_gap = std::numeric_limits<double>::max();
  for (int32_t rows = alignment; rows + min_second_rows <= num_rows;
       rows += alignment) {
    const int32_t second_rows = num_rows - rows;
    const double first_cost = rank_cost(
        /*begin=*/0,
        rows,
        history_rows,
        /*other_rank_kv_rows=*/replicate_final_state ? second_rows : 0);
    const double second_cost =
        rank_cost(rows, num_rows, history_rows, /*other_rank_kv_rows=*/rows);
    const double gap = std::abs(first_cost - second_cost);
    if (gap < best_gap) {
      best_gap = gap;
      best_rows = rows;
    }
  }
  return best_rows;
}

torch::Tensor slots_of(const Context& context, int32_t rank) {
  if (context.is_batch) {
    return rank == 0 ? context.rank0_slot_mapping : context.rank1_slot_mapping;
  }
  return context.global_slot_mapping.narrow(
      /*dim=*/0,
      /*start=*/context.row_offsets[rank],
      /*length=*/rows_of(context, rank));
}

// The root never consumes what it sends, so only receivers make their stream
// wait for the transfer.

bool pcp_custom_dtype(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0 || !tensor.is_musa()) {
    return false;
  }
  const auto dtype = tensor.scalar_type();
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

bool pcp_custom_ar_requested() {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_PCP_CUSTOM_AR", false);
  return enabled;
}

// CP=2 broadcast and allgather, expressed as a sum all-reduce. Chunks stay
// inside the 8MiB custom-allreduce buffer. Both ranks must call together.
std::optional<torch::Tensor> custom_sum(torch::Tensor input,
                                        ProcessGroup* group) {
  if (group == nullptr || !pcp_custom_ar_requested() ||
      !musa_custom_allreduce_enabled(group) || !pcp_custom_dtype(input)) {
    return std::nullopt;
  }
  input = input.contiguous();
  const int64_t elem = static_cast<int64_t>(input.element_size());
  constexpr int64_t kMaxBytes = 8 * 1024 * 1024;
  int64_t align_elems = 1;
  while (align_elems < 16 && (align_elems * elem) % 16 != 0) {
    ++align_elems;
  }
  const int64_t max_elems = (kMaxBytes / elem) / align_elems * align_elems;
  if (max_elems <= 0) {
    return std::nullopt;
  }
  torch::Tensor flat = input.reshape({-1});
  const int64_t bytes = flat.numel() * elem;
  const int64_t padded_bytes = (bytes + 15) & ~static_cast<int64_t>(15);
  if (padded_bytes != bytes) {
    const int64_t padded_elems = padded_bytes / elem;
    torch::Tensor padded = torch::zeros({padded_elems}, flat.options());
    padded.narrow(/*dim=*/0, /*start=*/0, /*length=*/flat.numel()).copy_(flat);
    flat = padded;
  }
  torch::Tensor output = torch::empty_like(flat);
  for (int64_t offset = 0; offset < flat.numel(); offset += max_elems) {
    const int64_t count = std::min(max_elems, flat.numel() - offset);
    torch::Tensor in_slice = flat.narrow(/*dim=*/0, offset, count);
    torch::Tensor out_slice = output.narrow(/*dim=*/0, offset, count);
    const std::optional<torch::Tensor> reduced =
        musa_allreduce_custom(group, in_slice, out_slice);
    if (!reduced.has_value()) {
      CHECK_EQ(offset, 0) << "PCP custom all-reduce failed after a chunk";
      return std::nullopt;
    }
  }
  if (output.numel() != input.numel()) {
    output = output.narrow(/*dim=*/0, /*start=*/0, /*length=*/input.numel());
  }
  return output.view(input.sizes());
}

bool custom_broadcast(torch::Tensor& rows,
                      int32_t root_rank,
                      Context& context) {
  if (!pcp_custom_ar_requested() || !pcp_custom_dtype(rows) ||
      !musa_custom_allreduce_enabled(context.cp_group)) {
    return false;
  }
  const bool root = context.cp_rank == root_rank;
  torch::Tensor input =
      root ? rows.contiguous() : torch::zeros(rows.sizes(), rows.options());
  const std::optional<torch::Tensor> reduced =
      custom_sum(input, context.cp_group);
  CHECK(reduced.has_value())
      << "PCP custom all-reduce failed after the CP group enabled it";
  if (!root) {
    rows.copy_(*reduced);
  }
  context.in_flight_sends.push_back(input);
  context.in_flight_sends.push_back(*reduced);
  LOG_FIRST_N(INFO, 1) << "[PCP_CUSTOM_AR] broadcast"
                       << " root=" << root_rank << " bytes=" << rows.nbytes();
  return true;
}

void broadcast_rows(torch::Tensor& rows, int32_t root_rank, Context& context) {
  if (custom_broadcast(rows, root_rank, context)) {
    return;
  }
  if (context.cp_rank == root_rank) {
    c10::intrusive_ptr<c10d::Work> work =
        context.cp_group->broadcast_async(rows, root_rank);
    context.in_flight_sends.emplace_back(rows);
    if (work.defined()) {
      context.in_flight_send_works.emplace_back(std::move(work));
    }
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
      global_metadata.q_seq_lens_vec.empty() ||
      global_metadata.q_seq_lens_vec.size() !=
          global_metadata.kv_seq_lens_vec.size()) {
    return;
  }
  const size_t num_seqs = global_metadata.q_seq_lens_vec.size();
  if (num_seqs > 1) {
    if (!pcp_multibatch_enabled()) {
      return;
    }
    int64_t total_q_rows = 0;
    std::vector<int32_t> history_vec(num_seqs, 0);
    bool any_history = false;
    for (size_t b = 0; b < num_seqs; ++b) {
      const int32_t q_b = global_metadata.q_seq_lens_vec[b];
      const int32_t kv_b = global_metadata.kv_seq_lens_vec[b];
      const int32_t h_b = kv_b - q_b;
      if (q_b < kMinBatchSeqRows || h_b < 0) {
        return;
      }
      history_vec[b] = h_b;
      any_history = any_history || (h_b > 0);
      total_q_rows += q_b;
    }
    if (total_q_rows != static_cast<int64_t>(num_rows)) {
      return;
    }
    const int32_t max_chunk =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    bool all_single_chunk = global_metadata.is_prefill && !any_history;
    if (all_single_chunk && max_chunk > 0) {
      for (size_t b = 0; b < num_seqs; ++b) {
        if (global_metadata.q_seq_lens_vec[b] >= max_chunk) {
          all_single_chunk = false;
          break;
        }
      }
    }
    const bool replicate_final_state =
        InstanceRole(::xllm::DisaggPDConfig::get_instance().instance_role()) !=
            InstanceRole::PREFILL ||
        KVCacheConfig::get_instance().enable_prefix_cache() ||
        !all_single_chunk;

    std::vector<int32_t> front_lens(num_seqs, 0);
    std::vector<int32_t> back_lens(num_seqs, 0);
    int32_t total_front_rows = 0;
    int32_t total_back_rows = 0;
    for (size_t b = 0; b < num_seqs; ++b) {
      const int32_t q_b = global_metadata.q_seq_lens_vec[b];
      const int32_t f_b = balanced_first_rows(q_b,
                                              history_vec[b],
                                              replicate_final_state,
                                              kBatchRowAlignment,
                                              kMinBatchTailRows);
      if (f_b < kDefaultConvStateLen ||
          (q_b - f_b) < kDefaultConvStateLen) {
        return;
      }
      front_lens[b] = f_b;
      back_lens[b] = q_b - f_b;
      total_front_rows += f_b;
      total_back_rows += (q_b - f_b);
    }

    Context& pcp_context = context->emplace();
    pcp_context.cp_group = cp_group;
    pcp_context.cp_rank = cp_group->rank();
    pcp_context.batch_size = static_cast<int32_t>(num_seqs);
    pcp_context.is_batch = true;
    pcp_context.row_offsets = {
        0, total_front_rows, total_front_rows + total_back_rows};
    pcp_context.max_rows = std::max(total_front_rows, total_back_rows);
    pcp_context.replicate_final_state = replicate_final_state;
    pcp_context.global_slot_mapping = global_metadata.slot_mapping;
    pcp_context.rank0_seq_ends.reserve(num_seqs);

    // Pack all int64 permutation/index vectors into a single H2D copy.
    const int64_t total_rows_i64 =
        static_cast<int64_t>(total_front_rows + total_back_rows);
    const int64_t conv_idx_count =
        static_cast<int64_t>(num_seqs) * kDefaultConvStateLen;
    std::vector<int64_t> idx_bundle;
    idx_bundle.resize(static_cast<size_t>(2 * total_rows_i64 + conv_idx_count));
    int64_t* r0_ptr = idx_bundle.data();
    int64_t* r1_ptr = r0_ptr + total_front_rows;
    int64_t* unperm_ptr = r1_ptr + total_back_rows;
    int64_t* conv_ptr = unperm_ptr + total_rows_i64;

    int32_t global_off = 0;
    int32_t r0_off = 0;
    int32_t r1_off = 0;
    for (size_t b = 0; b < num_seqs; ++b) {
      const int32_t q_b = global_metadata.q_seq_lens_vec[b];
      const int32_t f_b = front_lens[b];
      const int32_t r_b = back_lens[b];
      for (int32_t t = 0; t < f_b; ++t) {
        r0_ptr[r0_off + t] = global_off + t;
        unperm_ptr[global_off + t] = r0_off + t;
      }
      r0_off += f_b;
      pcp_context.rank0_seq_ends.push_back(r0_off);
      for (int32_t c = 0; c < kDefaultConvStateLen; ++c) {
        conv_ptr[b * kDefaultConvStateLen + c] =
            r0_off - kDefaultConvStateLen + c;
      }
      for (int32_t t = 0; t < r_b; ++t) {
        r1_ptr[r1_off + t] = global_off + f_b + t;
        unperm_ptr[global_off + f_b + t] = total_front_rows + r1_off + t;
      }
      r1_off += r_b;
      global_off += q_b;
    }

    const torch::Device device = global_metadata.q_cu_seq_lens.device();
    torch::Tensor idx_device =
        torch::tensor(idx_bundle, torch::TensorOptions().dtype(torch::kLong))
            .to(device);
    pcp_context.rank0_row_indices =
        idx_device.narrow(/*dim=*/0, /*start=*/0, /*length=*/total_front_rows);
    pcp_context.rank1_row_indices = idx_device.narrow(
        /*dim=*/0, /*start=*/total_front_rows, /*length=*/total_back_rows);
    pcp_context.unpermute_indices = idx_device.narrow(
        /*dim=*/0, /*start=*/total_rows_i64, /*length=*/total_rows_i64);
    pcp_context.conv_prefix_row_indices =
        idx_device.narrow(/*dim=*/0,
                          /*start=*/2 * total_rows_i64,
                          /*length=*/conv_idx_count);

    if (global_metadata.slot_mapping.defined()) {
      const torch::Device slot_dev = global_metadata.slot_mapping.device();
      const torch::Tensor r0_slot_idx =
          slot_dev == device ? pcp_context.rank0_row_indices
                             : pcp_context.rank0_row_indices.to(slot_dev);
      const torch::Tensor r1_slot_idx =
          slot_dev == device ? pcp_context.rank1_row_indices
                             : pcp_context.rank1_row_indices.to(slot_dev);
      pcp_context.rank0_slot_mapping =
          global_metadata.slot_mapping.index_select(/*dim=*/0, r0_slot_idx);
      pcp_context.rank1_slot_mapping =
          global_metadata.slot_mapping.index_select(/*dim=*/0, r1_slot_idx);
    }

    const bool is_rank0 = (pcp_context.cp_rank == 0);
    const std::vector<int32_t>& local_q_vec = is_rank0 ? front_lens : back_lens;
    std::vector<int32_t> local_kv_vec(num_seqs, 0);
    int32_t max_q_len = 0;
    int32_t max_kv_len = 0;
    int64_t total_kv_len = 0;

    // Pack [q_cu (B+1) | kv_cu (B+1) | q_lens (B) | kv_lens (B)] into one H2D.
    const int64_t b_i64 = static_cast<int64_t>(num_seqs);
    std::vector<int32_t> meta_i32(static_cast<size_t>(4 * b_i64 + 2), 0);
    int32_t* q_cu_ptr = meta_i32.data();
    int32_t* kv_cu_ptr = q_cu_ptr + (b_i64 + 1);
    int32_t* q_len_ptr = kv_cu_ptr + (b_i64 + 1);
    int32_t* kv_len_ptr = q_len_ptr + b_i64;
    for (size_t b = 0; b < num_seqs; ++b) {
      const int32_t q_b = local_q_vec[b];
      const int32_t kv_b = is_rank0 ? (history_vec[b] + front_lens[b])
                                    : global_metadata.kv_seq_lens_vec[b];
      local_kv_vec[b] = kv_b;
      q_cu_ptr[b + 1] = q_cu_ptr[b] + q_b;
      kv_cu_ptr[b + 1] = kv_cu_ptr[b] + kv_b;
      q_len_ptr[b] = q_b;
      kv_len_ptr[b] = kv_b;
      max_q_len = std::max(max_q_len, q_b);
      max_kv_len = std::max(max_kv_len, kv_b);
      total_kv_len += kv_b;
    }
    torch::Tensor meta_device =
        torch::tensor(meta_i32, torch::TensorOptions().dtype(torch::kInt32))
            .to(device);

    AttentionMetadata& local = pcp_context.local_metadata;
    local = global_metadata;
    local.q_cu_seq_lens =
        meta_device.narrow(/*dim=*/0, /*start=*/0, /*length=*/b_i64 + 1);
    local.kv_cu_seq_lens = meta_device.narrow(
        /*dim=*/0, /*start=*/b_i64 + 1, /*length=*/b_i64 + 1);
    if (global_metadata.q_seq_lens.defined()) {
      local.q_seq_lens = meta_device.narrow(
          /*dim=*/0, /*start=*/2 * (b_i64 + 1), /*length=*/b_i64);
    }
    if (global_metadata.kv_seq_lens.defined()) {
      local.kv_seq_lens = meta_device.narrow(
          /*dim=*/0, /*start=*/2 * (b_i64 + 1) + b_i64, /*length=*/b_i64);
    }
    local.q_seq_lens_vec = local_q_vec;
    local.kv_seq_lens_vec = std::move(local_kv_vec);
    if (global_metadata.slot_mapping.defined()) {
      local.slot_mapping = slots_of(pcp_context, pcp_context.cp_rank);
    }
    if (global_metadata.mrope_cos.defined()) {
      local.mrope_cos =
          shard_rows(global_metadata.mrope_cos, pcp_context, /*dim=*/0);
      local.mrope_sin =
          shard_rows(global_metadata.mrope_sin, pcp_context, /*dim=*/0);
    }
    local.max_query_len = max_q_len;
    local.max_seq_len = max_kv_len;
    local.total_kv_len = total_kv_len;
    const bool continues_rows =
        !is_rank0 || any_history || global_metadata.is_chunked_prefill;
    local.is_prefill = !continues_rows;
    local.is_chunked_prefill = continues_rows;
    local.has_initial_states =
        !is_rank0
            ? torch::full(
                  {pcp_context.batch_size},
                  /*fill_value=*/true,
                  torch::TensorOptions().dtype(torch::kBool).device(device))
            : global_metadata.has_initial_states;
    local.fa3_metadata.fa3_scheduler_metadata = torch::Tensor();
    local.pcp_context = &pcp_context;
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

void join_pending_sends(Context& context) {
  for (auto& work : context.in_flight_send_works) {
    if (work.defined()) {
      work->wait();
    }
  }
  context.in_flight_send_works.clear();
}

bool receives_prefix_state(const Context& context) {
  return context.cp_rank > 0;
}

torch::Tensor shard_rows(const torch::Tensor& global,
                         const Context& context,
                         int64_t dim) {
  if (context.is_batch) {
    const int64_t norm_dim = dim < 0 ? global.dim() + dim : dim;
    const torch::Tensor& raw_idx =
        context.cp_rank == 0 ? context.rank0_row_indices
                             : context.rank1_row_indices;
    const torch::Tensor idx =
        raw_idx.device() == global.device() ? raw_idx
                                            : raw_idx.to(global.device());
    return global.index_select(norm_dim, idx).contiguous();
  }
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
  torch::Tensor gathered;
  if (pcp_custom_ar_requested() && pcp_custom_dtype(send) &&
      musa_custom_allreduce_enabled(context.cp_group)) {
    std::vector<int64_t> stacked_shape = send.sizes().vec();
    stacked_shape.insert(stacked_shape.begin(), num_ranks(context));
    torch::Tensor stacked = torch::zeros(stacked_shape, send.options());
    stacked[context.cp_rank].copy_(send);
    const std::optional<torch::Tensor> reduced =
        custom_sum(stacked, context.cp_group);
    CHECK(reduced.has_value()) << "PCP custom allgather failed";
    gathered = *reduced;
    LOG_FIRST_N(INFO, 1) << "[PCP_CUSTOM_AR] allgather bytes="
                         << stacked.nbytes();
  } else {
    gathered = context.cp_group->allgather_base_sync(send);
  }
  const int32_t cp_size = num_ranks(context);
  torch::Tensor packed;
  if (context.row_offsets.back() == cp_size * context.max_rows) {
    packed = gathered.flatten(/*start_dim=*/0, /*end_dim=*/1);
  } else {
    std::vector<torch::Tensor> parts;
    parts.reserve(cp_size);
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      parts.emplace_back(gathered[rank].narrow(
          /*dim=*/0, /*start=*/0, /*length=*/rows_of(context, rank)));
    }
    packed = torch::cat(parts, /*dim=*/0);
  }
  if (!context.is_batch) {
    return packed;
  }
  const torch::Tensor idx =
      context.unpermute_indices.device() == packed.device()
          ? context.unpermute_indices
          : context.unpermute_indices.to(packed.device());
  return packed.index_select(/*dim=*/0, idx);
}

void expect_prefix_rows(torch::IntArrayRef shape,
                        const torch::TensorOptions& options,
                        Context& context) {
  if (context.cp_rank == 0) {
    return;
  }
  PendingReceive receive;
  receive.rows = torch::empty(shape, options);
  // Custom all-reduce is a joint call with the sender, so the receiver only
  // reserves storage here and enters the collective in take_prefix_rows.
  if (!pcp_custom_ar_requested() || !pcp_custom_dtype(receive.rows) ||
      !musa_custom_allreduce_enabled(context.cp_group)) {
    receive.work = context.cp_group->broadcast_async(
        receive.rows, /*root_rank=*/context.cp_rank - 1);
  }
  context.pending_receives.emplace_back(std::move(receive));
}

void expect_prefix_cache_row(const torch::Tensor& cache, Context& context) {
  std::vector<int64_t> shape = cache.sizes().vec();
  shape[0] = context.batch_size;
  expect_prefix_rows(shape, cache.options(), context);
}

void expect_conv_prefix(int64_t conv_state_len,
                        int64_t qkv_dim,
                        const torch::TensorOptions& options,
                        Context& context) {
  if (context.is_batch) {
    expect_prefix_rows(
        {context.batch_size, conv_state_len, qkv_dim}, options, context);
  } else {
    expect_prefix_rows({conv_state_len, qkv_dim}, options, context);
  }
}

torch::Tensor extract_conv_prefix(const torch::Tensor& local_rows,
                                  int64_t conv_state_len,
                                  int64_t qkv_dim,
                                  const Context& context) {
  torch::Tensor cols =
      local_rows.size(-1) == qkv_dim
          ? local_rows
          : local_rows.narrow(/*dim=*/-1, /*start=*/0, /*length=*/qkv_dim);
  if (!context.is_batch) {
    CHECK_GE(cols.size(0), conv_state_len);
    return cols
        .narrow(/*dim=*/0,
                /*start=*/cols.size(0) - conv_state_len,
                /*length=*/conv_state_len)
        .contiguous();
  }
  torch::Tensor row_indices;
  if (conv_state_len == kDefaultConvStateLen &&
      context.conv_prefix_row_indices.defined()) {
    row_indices = context.conv_prefix_row_indices.device() == cols.device()
                      ? context.conv_prefix_row_indices
                      : context.conv_prefix_row_indices.to(cols.device());
  } else {
    CHECK_EQ(static_cast<int32_t>(context.rank0_seq_ends.size()),
             context.batch_size);
    std::vector<int64_t> cpu_idx;
    cpu_idx.reserve(static_cast<size_t>(context.batch_size * conv_state_len));
    for (int32_t seq_end : context.rank0_seq_ends) {
      CHECK_GE(seq_end, conv_state_len);
      for (int64_t c = 0; c < conv_state_len; ++c) {
        cpu_idx.push_back(seq_end - conv_state_len + c);
      }
    }
    row_indices =
        torch::tensor(cpu_idx, torch::TensorOptions().dtype(torch::kLong))
            .to(cols.device());
  }
  return cols.index_select(/*dim=*/0, row_indices)
      .view({context.batch_size, conv_state_len, qkv_dim})
      .contiguous();
}

torch::Tensor take_prefix_rows(Context& context) {
  CHECK(!context.pending_receives.empty())
      << "context-parallel prefill took rows that were never posted";
  PendingReceive receive = std::move(context.pending_receives.front());
  context.pending_receives.pop_front();
  if (receive.work.defined()) {
    receive.work->wait();
  } else if (!custom_broadcast(receive.rows, context.cp_rank - 1, context)) {
    LOG(FATAL) << "PCP custom all-reduce was selected but did not run";
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
                      const torch::TensorOptions& projected_options,
                      Context& context) {
  const torch::ScalarType wire_dtype = projected_options.dtype().toScalarType();
  CHECK(wire_dtype == torch::kBFloat16 || wire_dtype == torch::kFloat16 ||
        wire_dtype == torch::kFloat32)
      << "PCP KV receive requires the projection compute dtype, not the cache "
         "storage dtype";
  const int64_t first_rows = rows_of(context, /*rank=*/0);
  expect_prefix_rows(
      {first_rows, 2 * num_kv_heads * head_dim}, projected_options, context);
}

void exchange_kv(const torch::Tensor& key,
                 const torch::Tensor& value,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 KVCache& kv_cache,
                 Context& context) {
  CHECK_EQ(key.scalar_type(), value.scalar_type());
  CHECK_EQ(key.device(), value.device());
  CHECK(key.scalar_type() == torch::kBFloat16 ||
        key.scalar_type() == torch::kFloat16 ||
        key.scalar_type() == torch::kFloat32)
      << "PCP KV sends projected rows before cache-storage conversion";
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
    const torch::Tensor prefix_kv = take_prefix_rows(context);
    CHECK_EQ(prefix_kv.scalar_type(), local_kv.scalar_type());
    CHECK_EQ(prefix_kv.device(), local_kv.device());
    CHECK_EQ(prefix_kv.size(/*dim=*/-1), 2 * kv_size);
    write_kv_rows(prefix_kv,
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

void complete_deferred_transfers(Context& context) {
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
}

torch::Tensor finish_state_only(const torch::Tensor& local_hidden_states,
                                Context& context) {
  complete_deferred_transfers(context);
  // Join async MCCL sends onto the active compute/capture stream before
  // exiting the forward pass, since gather_rows is skipped here.
  join_pending_sends(context);
  context.in_flight_sends.clear();
  return local_hidden_states;
}

torch::Tensor finish(const torch::Tensor& local_hidden_states,
                     Context& context) {
  complete_deferred_transfers(context);
  join_pending_sends(context);
  torch::Tensor hidden_states = gather_rows(local_hidden_states, context);
  context.in_flight_sends.clear();
  return hidden_states;
}

}  // namespace xllm::layer::musa_pcp
