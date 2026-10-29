/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "core/layers/musa/deepseek_v4/deepseek_v4_pcp_context.h"

#include <glog/logging.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace xllm::layer::dsv4_musa_pcp {
namespace {

constexpr int32_t kSupportedCpSize = 2;
// Minimum rows per group; shorter prompts stay on the plain TP+SP path.
constexpr int64_t kMinRowsPerGroup = 1024;
// Row alignment for the split point (matches the SP row quantisation the
// attention kernels expect).
constexpr int64_t kRowAlignment = 128;

void synchronize_peer_groups(const Context& context) {
  CHECK(context.global_group != nullptr);
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  // A fast CP pair must not launch spinning MCCL work while another pair
  // still allocates buffers or finishes a TP collective on the same devices.
  musa_long_prefill_host_barrier(context.global_group);
}

bool pcp_env_enabled() {
  const char* setting = std::getenv("XLLM_DSV4_PCP");
  return setting != nullptr && std::string(setting) == "1";
}

// Causal prefill cost model, mirroring musa_pcp::rank_cost: rows plus
// attention pairs over the causal history.  The later group additionally
// writes the earlier group's K/V rows into its cache.
double group_cost(int64_t begin, int64_t end) {
  const double rows = static_cast<double>(end - begin);
  const double first_keys = static_cast<double>(begin);
  const double last_keys = static_cast<double>(end);
  const double pairs = 0.5 * (last_keys * last_keys - first_keys * first_keys);
  return rows + 9.4e-6 * pairs;
}

int64_t balanced_front_rows(int64_t total_rows) {
  int64_t best_rows = 0;
  double best_gap = std::numeric_limits<double>::max();
  for (int64_t rows = kMinRowsPerGroup; rows + kMinRowsPerGroup <= total_rows;
       rows += kRowAlignment) {
    const double gap =
        std::abs(group_cost(0, rows) - group_cost(rows, total_rows));
    if (gap < best_gap) {
      best_gap = gap;
      best_rows = rows;
    }
  }
  return best_rows;
}

}  // namespace

bool enabled() { return pcp_env_enabled(); }

void build_context(int64_t total_rows,
                   const ParallelArgs& parallel_args,
                   const torch::Tensor& global_tokens,
                   std::optional<Context>* context) {
  context->reset();
  // Graph capture replays a synthetic prefill with internal buffers; a p2p
  // collective on the capturing stream is rejected by MCCL outright.
  if (::xllm::kernel::musa::is_stream_capturing()) {
    return;
  }
  ProcessGroup* cross_group = parallel_args.cp_group_;
  const int32_t cp_size = parallel_args.cp_size();
  if (!pcp_env_enabled() || cross_group == nullptr ||
      cp_size != kSupportedCpSize) {
    return;
  }
  CHECK_EQ(cross_group->world_size(), cp_size)
      << "DeepSeek-V4 PCP requires a cross-group communicator sized cp_size.";
  // Empty batches, graph probes, and DSpark draft forwards (a handful of
  // tokens through the same model layer) never enter PCP: they publish no KV
  // to the decode side, and their shapes do not fit the two-group exchange.
  // Real prefill prompts below the split threshold also stay out; their KV
  // handoff is covered by the front group's full-prompt cache write below.
  if (total_rows < kMinRowsPerGroup) {
    return;
  }
  // Prompts too short to balance still split at the midpoint: every rank
  // keeps at least one valid row (empty groups break zero-work kernels down
  // the MoE/attention stack), and the bidirectional KV exchange below makes
  // both caches complete either way.
  ProcessGroup* moe_group = parallel_args.moe_tp_group_;
  const int64_t global_tp_size =
      parallel_args.tp_group_->world_size() * cp_size;
  const bool uniform_moe_rows = parallel_args.ep_size() == 1 &&
                                moe_group != nullptr &&
                                moe_group->world_size() == global_tp_size &&
                                total_rows % global_tp_size == 0;
  // Equal global rank ranges let MoE return only each rank's rows with the
  // same native reduce-scatter already used by TP8 prefill. The compressor
  // continuation contract carries partial C128 blocks across the midpoint.
  const int64_t front_rows =
      uniform_moe_rows || total_rows < 2 * kMinRowsPerGroup
          ? total_rows / 2
          : balanced_front_rows(total_rows);
  if (front_rows < 0 || front_rows >= total_rows) {
    if (total_rows < 2 * kMinRowsPerGroup) {
      // fall through with front_rows == 0
    } else {
      return;
    }
  }
  Context value;
  value.cross_group = cross_group;
  value.global_group = parallel_args.process_group_;
  value.moe_group = moe_group;
  value.global_input_ids = global_tokens;
  CHECK(value.global_group != nullptr);
  CHECK_EQ(value.global_group->world_size(), global_tp_size);
  value.cp_rank = cross_group->rank();
  value.cp_size = cp_size;
  value.front_rows = front_rows;
  value.total_rows = total_rows;

  // Group-local SP: every rank of the group runs the group's token range
  // through the existing TP sequence-parallel split inside the group-local
  // tp_group (width world/cp under CP).
  dsv4_musa_sp::DeepseekV4SPContext group_sp;
  group_sp.process_group = parallel_args.tp_group_;
  group_sp.rank =
      parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_->rank() : 0;
  group_sp.world_size = parallel_args.tp_group_ != nullptr
                            ? parallel_args.tp_group_->world_size()
                            : 1;
  const int64_t group_rows_count = group_rows(value);
  CHECK_GT(group_rows_count, 0) << "DeepSeek-V4 PCP group has negative rows.";
  group_sp.total_tokens = group_rows_count;
  group_sp.local_tokens = std::max<int64_t>(
      (group_rows_count + group_sp.world_size - 1) / group_sp.world_size, 1);
  group_sp.global_row_start = value.cp_rank == 0 ? 0 : front_rows;
  // local_offset stays GROUP-RELATIVE (local_shard/pad_local assume it lies
  // inside [0, total_tokens)); attention adds global_row_start wherever an
  // absolute prompt position is needed (RoPE slices, causal query starts).
  group_sp.local_offset = std::min<int64_t>(
      static_cast<int64_t>(group_sp.rank) * group_sp.local_tokens,
      group_rows_count);
  group_sp.local_valid_tokens = std::max<int64_t>(
      0,
      std::min<int64_t>(group_sp.local_tokens,
                        group_rows_count - group_sp.local_offset));
  if (global_tokens.defined() && global_tokens.numel() == total_rows) {
    group_sp.global_input_ids = shard_rows(global_tokens, value).contiguous();
  }
  if (uniform_moe_rows) {
    auto global_sp = group_sp;
    global_sp.process_group = value.global_group;
    global_sp.rank = value.global_group->rank();
    global_sp.world_size = value.global_group->world_size();
    global_sp.total_tokens = total_rows;
    global_sp.global_row_start = 0;
    global_sp.local_offset = global_sp.rank * global_sp.local_tokens;
    global_sp.global_input_ids = global_tokens;
    CHECK_EQ(global_sp.local_tokens * global_sp.world_size, total_rows);
    CHECK_EQ(global_sp.local_offset,
             group_sp.global_row_start + group_sp.local_offset);
    value.global_sp = std::move(global_sp);
  }
  value.group_sp = std::move(group_sp);
  *context = std::move(value);
  LOG_FIRST_N(INFO, 1) << "[DSV4_PCP] cp_rank=" << context->value().cp_rank
                       << " front_rows=" << front_rows
                       << " total_rows=" << total_rows
                       << " group_rows=" << group_rows_count << " sp_local="
                       << context->value().group_sp->local_valid_tokens << "/"
                       << context->value().group_sp->local_tokens;
}

torch::Tensor shard_rows(const torch::Tensor& global, const Context& context) {
  if (!global.defined()) {
    return global;
  }
  const int64_t start = context.cp_rank == 0 ? 0 : context.front_rows;
  return global.narrow(/*dim=*/0, start, group_rows(context)).contiguous();
}

torch::Tensor exchange_rows_global_slots(const torch::Tensor& local_rows,
                                         const torch::Tensor& global_slots,
                                         Context& context) {
  CHECK(local_rows.defined() && global_slots.defined());
  const int64_t total_rows = global_slots.numel();
  const int64_t local_row_count = local_rows.size(0);
  CHECK_GT(total_rows, local_row_count);
  // The causal split is deliberately unequal. Each group derives the same
  // front/back compressed row counts from its own output and the global map.
  Context compressed_context;
  compressed_context.cross_group = context.cross_group;
  compressed_context.global_group = context.global_group;
  compressed_context.cp_rank = context.cp_rank;
  compressed_context.cp_size = context.cp_size;
  compressed_context.total_rows = total_rows;
  compressed_context.front_rows =
      context.cp_rank == 0 ? local_row_count : total_rows - local_row_count;
  LOG_FIRST_N(INFO, 8) << "[DSV4_PCP_COMP_ROWS] cp_rank=" << context.cp_rank
                       << " front=" << compressed_context.front_rows
                       << " back=" << total_rows - compressed_context.front_rows
                       << " total=" << total_rows;
  // Both groups need complete history before the next chunk. Returning the
  // assembled rows also keeps every cache write aligned with its global slots.
  return relay_kv(local_rows, compressed_context);
}

torch::Tensor relay_kv(torch::Tensor rows, Context& context) {
  // Bidirectional half exchange via two broadcasts.  Raw p2p on the 2-rank
  // group aborts inside MCCL 2.28.9 (unhandled error at ProcessGroupMCCL
  // 3321) on the first real-data exchange inside the attention forward,
  // while broadcast on the same group runs reliably (the TP groups use it
  // constantly).  Each rank broadcasts its own half; both then concatenate
  // in global row order.  The root's own rows are its contribution.
  const bool is_front = context.cp_rank == 0;
  const int64_t front = context.front_rows;
  const int64_t back = context.total_rows - front;
  const int64_t own_rows = rows.size(0);
  CHECK_EQ(own_rows, is_front ? front : back)
      << "PCP relay received unexpected group row count";
  // The completion rendezvous below finishes all broadcasts before this
  // local buffer is released. Retaining one full FFN buffer per layer wastes
  // gigabytes and prevents the allocator from reusing stable storage.
  torch::Tensor send_buffer = rows.contiguous();

  auto front_shape = rows.sizes().vec();
  front_shape[0] = front;
  torch::Tensor front_half =
      is_front ? send_buffer : torch::empty(front_shape, rows.options());
  auto back_shape = rows.sizes().vec();
  back_shape[0] = back;
  torch::Tensor back_half =
      is_front ? torch::empty(back_shape, rows.options()) : send_buffer;

  synchronize_peer_groups(context);
  context.cross_group->broadcast(front_half, /*root_rank=*/0);
  context.cross_group->broadcast(back_half, /*root_rank=*/1);
  synchronize_peer_groups(context);
  return torch::cat({front_half, back_half}, 0);
}
torch::Tensor relay_state(torch::Tensor state, Context& context) {
  // Recurrent compressor/indexer state flows forward inside a chunk (front
  // group's end-of-half state seeds the later group) and back at the chunk
  // boundary (the later group's end-of-chunk state seeds the front group's
  // next chunk).  One call performs the forward leg on the front group and
  // both legs bookkeeping-free on the later group; the chunk-boundary
  // backward leg is relay_state_back.
  if (context.cp_rank == 0) {
    context.cross_group->broadcast(state, /*root_rank=*/0);
    return state;
  }
  if (!state.defined()) {
    return state;
  }
  torch::Tensor received = torch::empty_like(state);
  context.cross_group->broadcast(received, /*root_rank=*/0);
  return received;
}

void relay_state_back(torch::Tensor& state_slot, Context& context) {
  // Chunk boundary: the later group just finished the chunk; hand its final
  // recurrent state back so the front group continues the next chunk from
  // the true end-of-chunk state.
  if (context.cp_rank == 1) {
    context.cross_group->broadcast(state_slot, /*root_rank=*/1);
    return;
  }
  if (!state_slot.defined()) {
    return;
  }
  torch::Tensor received = torch::empty_like(state_slot);
  context.cross_group->broadcast(received, /*root_rank=*/1);
  state_slot = received;
}

void finish(torch::Tensor& local_hidden_states, Context& context) {
  if (context.global_sp.has_value()) {
    CHECK_EQ(local_hidden_states.size(0), context.global_sp->local_tokens);
    local_hidden_states = dsv4_musa_sp::gather_global(
        local_hidden_states, context.global_sp.value());
    LOG(INFO) << "[DSV4_PCP_FINISH] exit cp_rank=" << context.cp_rank
              << " local_rows=" << local_hidden_states.size(0)
              << " global_gather=1";
    return;
  }
  // The cross-group communicator pairs corresponding TP ranks in each CP group.
  // Both ranks must enter the same broadcast collective; returning early on
  // the back-group master leaves the front-group broadcast unmatched.
  // Broadcast exchange (p2p on this 2-rank group costs a fixed ~218s inside
  // MCCL 2.28.9 regardless of size; broadcast on the same group is fast).
  // The back rank broadcasts its rows to the front rank; the front rank
  // already has its own.
  LOG(INFO) << "[DSV4_PCP_FINISH] enter cp_rank=" << context.cp_rank
            << " local_rows=" << local_hidden_states.size(0)
            << " front_rows=" << context.front_rows
            << " total_rows=" << context.total_rows;
  synchronize_peer_groups(context);
  if (context.cp_rank == 1) {
    context.cross_group->broadcast(local_hidden_states, /*root_rank=*/1);
  } else {
    auto back_shape = local_hidden_states.sizes().vec();
    back_shape[0] = context.total_rows - context.front_rows;
    torch::Tensor back =
        torch::empty(back_shape, local_hidden_states.options());
    context.cross_group->broadcast(back, /*root_rank=*/1);
    local_hidden_states = torch::cat({local_hidden_states, back}, 0);
  }
  // Keep both CP groups on the same full hidden state before the next chunk.
  // Without this second broadcast, CP1 retains only its local rows while CP0
  // advances with the assembled prompt, which deadlocks the next chunk's
  // collective path for long prompts with uneven group sizes.
  if (context.cp_rank == 0) {
    context.cross_group->broadcast(local_hidden_states, /*root_rank=*/0);
  } else {
    auto full_shape = local_hidden_states.sizes().vec();
    full_shape[0] = context.total_rows;
    torch::Tensor full =
        torch::empty(full_shape, local_hidden_states.options());
    context.cross_group->broadcast(full, /*root_rank=*/0);
    local_hidden_states = std::move(full);
  }
  synchronize_peer_groups(context);
  LOG(INFO) << "[DSV4_PCP_FINISH] exit cp_rank=" << context.cp_rank
            << " local_rows=" << local_hidden_states.size(0);
}

}  // namespace xllm::layer::dsv4_musa_pcp
