/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_execution_limits.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::layer::dsv4_musa_sp {

struct DeepseekV4SPContext {
  ProcessGroup* process_group = nullptr;
  int32_t rank = 0;
  int32_t world_size = 1;
  int64_t total_tokens = 0;
  int64_t local_tokens = 0;
  int64_t local_valid_tokens = 0;
  int64_t local_offset = 0;
  // Absolute position of this SP shard inside the full global prompt. Plain
  // TP+SP leaves it zero; a PCP group's SP context carries the owning CP
  // group's row start so RoPE tables and causal offsets stay global.
  int64_t global_row_start = 0;
  // Hash routing consumes the same unsharded ids in every layer. Retaining
  // the model-owned view avoids another token collective around each MoE.
  torch::Tensor global_input_ids;
};

struct GatherGlobalAsync {
  torch::Tensor local_contiguous;
  torch::Tensor gathered;
  c10::intrusive_ptr<c10d::Work> work;
};

enum class GatherSlot : size_t {
  kKv = 0,
  kAuxOrMoe = 1,
};

inline constexpr int64_t kDsv4SpKvGatherCols = 576;

struct SpGatherWorkspaces {
  std::array<torch::Tensor, 2> slots;
};

inline std::mutex& sp_gather_workspace_mutex() {
  static std::mutex mu;
  return mu;
}

inline std::unordered_map<int32_t, SpGatherWorkspaces>&
sp_gather_workspaces_map() {
  static std::unordered_map<int32_t, SpGatherWorkspaces> by_device;
  return by_device;
}

inline torch::Tensor ensure_sp_gather_slot(GatherSlot slot,
                                           int64_t required_bytes,
                                           const torch::Device& device) {
  std::lock_guard<std::mutex> lock(sp_gather_workspace_mutex());
  auto& buf = sp_gather_workspaces_map()[device.index()]
                  .slots[static_cast<size_t>(slot)];
  if (!buf.defined() || buf.device() != device ||
      buf.scalar_type() != torch::kUInt8 || buf.numel() < required_bytes) {
    buf = torch::empty(
        {required_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }
  return buf;
}

inline torch::Tensor try_get_sp_gather_slot(GatherSlot slot,
                                            int64_t required_bytes,
                                            const torch::Device& device) {
  std::lock_guard<std::mutex> lock(sp_gather_workspace_mutex());
  const auto it = sp_gather_workspaces_map().find(device.index());
  if (it == sp_gather_workspaces_map().end()) {
    return {};
  }
  const auto& buf = it->second.slots[static_cast<size_t>(slot)];
  if (!buf.defined() || buf.device() != device ||
      buf.scalar_type() != torch::kUInt8 || buf.numel() < required_bytes) {
    return {};
  }
  return buf;
}

inline bool long_prefill_rendezvous_enabled(
    const DeepseekV4SPContext& context) {
  const char* setting = std::getenv("XLLM_DSV4_LONG_PREFILL_RENDEZVOUS");
  return context.total_tokens >= 4096 && setting != nullptr &&
         std::string(setting) == "1";
}

inline bool prefill_step_barrier_enabled(const DeepseekV4SPContext& context) {
  const char* setting = std::getenv("XLLM_DSV4_PREFILL_STEP_BARRIER");
  return context.total_tokens >= 4096 && setting != nullptr &&
         std::string(setting) == "1";
}

inline void ensure_prefill_workspaces(
    const DeepseekV4SPContext& context,
    const torch::Device& device,
    torch::ScalarType dtype = torch::kBFloat16) {
  const int64_t padded_tokens = std::max<int64_t>(
      1, static_cast<int64_t>(context.world_size) * context.local_tokens);
  const int64_t elem_bytes =
      std::max<int64_t>(1, static_cast<int64_t>(c10::elementSize(dtype)));
  ensure_sp_gather_slot(
      GatherSlot::kKv, padded_tokens * kDsv4SpKvGatherCols * elem_bytes, device);
  ensure_sp_gather_slot(
      GatherSlot::kAuxOrMoe,
      padded_tokens * kDsv4FlashHiddenSize * elem_bytes,
      device);
  if (prefill_step_barrier_enabled(context)) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_PREFILL_STEP_BARRIER] total_tokens="
                         << context.total_tokens;
    CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
    musa_long_prefill_host_barrier(context.process_group);
  }
}

inline void begin_long_prefill_collective(const DeepseekV4SPContext& context) {
  if (!long_prefill_rendezvous_enabled(context)) {
    return;
  }
  // Do not let a fast rank launch spinning MCCL work while another rank is
  // still producing local tensors or reclaiming allocator storage. This is
  // opt-in and long-prefill-only; short B1 and Graph decode never enter it.
  LOG_FIRST_N(INFO, 1) << "[DSV4_LONG_PREFILL_RENDEZVOUS] total_tokens="
                       << context.total_tokens;
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  musa_long_prefill_host_barrier(context.process_group);
}

inline void finish_long_prefill_collective(const DeepseekV4SPContext& context) {
  if (long_prefill_rendezvous_enabled(context)) {
    CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  }
}

inline std::optional<DeepseekV4SPContext> build_context(
    const torch::Tensor& tokens,
    bool is_prefill,
    ProcessGroup* process_group) {
  if (!is_prefill || process_group == nullptr ||
      process_group->world_size() <= 1 || !tokens.defined() ||
      tokens.dim() != 1) {
    return std::nullopt;
  }

  const int32_t world_size = process_group->world_size();
  const int64_t total_tokens = tokens.numel();
  if (total_tokens < world_size) {
    return std::nullopt;
  }

  const int64_t local_tokens = (total_tokens + world_size - 1) / world_size;
  const int32_t rank = process_group->rank();
  const int64_t local_offset = std::min<int64_t>(
      static_cast<int64_t>(rank) * local_tokens, total_tokens);
  const int64_t local_valid_tokens = std::max<int64_t>(
      0, std::min<int64_t>(local_tokens, total_tokens - local_offset));
  DeepseekV4SPContext context;
  context.process_group = process_group;
  context.rank = rank;
  context.world_size = world_size;
  context.total_tokens = total_tokens;
  context.local_tokens = local_tokens;
  context.local_valid_tokens = local_valid_tokens;
  context.local_offset = local_offset;
  context.global_input_ids = tokens.contiguous();
  return context;
}

inline torch::Tensor local_shard(const torch::Tensor& tensor,
                                 const DeepseekV4SPContext& context) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK_GT(tensor.dim(), 0)
      << "DeepSeek-V4 sequence parallel expects a token dimension";
  CHECK_EQ(tensor.size(0), context.total_tokens)
      << "DeepSeek-V4 sequence parallel input does not match global tokens";
  torch::Tensor shard =
      tensor.narrow(/*dim=*/0, context.local_offset, context.local_valid_tokens)
          .contiguous();
  if (context.local_valid_tokens == context.local_tokens) {
    return shard;
  }
  return kernel::musa::deepseek_v4_sp_pad_local(shard, context.local_tokens);
}

inline torch::Tensor pad_local(const torch::Tensor& tensor,
                               const DeepseekV4SPContext& context) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK_GT(tensor.dim(), 0)
      << "DeepSeek-V4 sequence parallel expects a token dimension";
  CHECK_EQ(tensor.size(0), context.local_valid_tokens)
      << "DeepSeek-V4 sequence parallel valid tensor has unexpected rows";
  if (context.local_valid_tokens == context.local_tokens) {
    return tensor.contiguous();
  }
  return kernel::musa::deepseek_v4_sp_pad_local(tensor.contiguous(),
                                                context.local_tokens);
}

inline torch::Tensor gather_global(
    const torch::Tensor& local_tensor,
    const DeepseekV4SPContext& context,
    std::optional<GatherSlot> slot = std::nullopt) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  CHECK_EQ(local_tensor.size(0), context.local_tokens)
      << "DeepSeek-V4 sequence parallel local tensor has unexpected rows";
  auto contiguous = local_tensor.contiguous();
  auto shape = contiguous.sizes().vec();
  shape.insert(shape.begin(), context.world_size);
  torch::Tensor gathered;
  bool using_workspace = false;
  if (slot.has_value() && contiguous.numel() > 0) {
    const int64_t total_bytes =
        static_cast<int64_t>(context.world_size) * contiguous.numel() *
        static_cast<int64_t>(contiguous.element_size());
    torch::Tensor workspace = try_get_sp_gather_slot(
        *slot, total_bytes, contiguous.device());
    if (workspace.defined() && workspace.data_ptr() != contiguous.data_ptr()) {
      gathered = workspace.narrow(/*dim=*/0, /*start=*/0, /*length=*/total_bytes)
                     .view(contiguous.scalar_type())
                     .view(shape);
      using_workspace = true;
    }
  }
  if (long_prefill_rendezvous_enabled(context)) {
    // Explicit opt-in rendezvous fallback: allocate before peer sync so a
    // lagging rank never reclaims allocator storage mid-collective.
    if (!using_workspace) {
      gathered = torch::empty(shape, contiguous.options());
    }
    begin_long_prefill_collective(context);
    auto work =
        context.process_group->allgather_base_async(contiguous, gathered);
    work->wait();
    finish_long_prefill_collective(context);
  } else if (using_workspace) {
    // Persistent pre-allocated workspace avoids per-layer torch::empty() and
    // host synchronization; Work::wait() records the stream dependency.
    auto work =
        context.process_group->allgather_base_async(contiguous, gathered);
    work->wait();
  } else {
    gathered = context.process_group->allgather_base_sync(contiguous);
  }
  CHECK_EQ(gathered.dim(), local_tensor.dim() + 1);
  gathered = gathered.flatten(/*start_dim=*/0, /*end_dim=*/1);
  CHECK_GE(gathered.size(0), context.total_tokens);
  return gathered.narrow(/*dim=*/0, /*start=*/0, context.total_tokens);
}

inline GatherGlobalAsync launch_gather_global(
    const torch::Tensor& local_tensor,
    const DeepseekV4SPContext& context,
    std::optional<GatherSlot> slot = std::nullopt) {
  CHECK(local_tensor.defined());
  CHECK_EQ(local_tensor.size(0), context.local_tokens)
      << "DeepSeek-V4 sequence parallel local tensor has unexpected rows";
  GatherGlobalAsync async;
  async.local_contiguous = local_tensor.contiguous();
  auto gathered_shape = async.local_contiguous.sizes().vec();
  gathered_shape.insert(gathered_shape.begin(), context.world_size);
  bool using_workspace = false;
  if (slot.has_value() && async.local_contiguous.numel() > 0) {
    const int64_t total_bytes =
        static_cast<int64_t>(context.world_size) *
        async.local_contiguous.numel() *
        static_cast<int64_t>(async.local_contiguous.element_size());
    torch::Tensor workspace = try_get_sp_gather_slot(
        *slot, total_bytes, async.local_contiguous.device());
    if (workspace.defined() &&
        workspace.data_ptr() != async.local_contiguous.data_ptr()) {
      async.gathered =
          workspace.narrow(/*dim=*/0, /*start=*/0, /*length=*/total_bytes)
              .view(async.local_contiguous.scalar_type())
              .view(gathered_shape);
      using_workspace = true;
    }
  }
  if (!using_workspace) {
    async.gathered =
        torch::empty(gathered_shape, async.local_contiguous.options());
  }
  begin_long_prefill_collective(context);
  async.work = context.process_group->allgather_base_async(
      async.local_contiguous, async.gathered);
  return async;
}

inline torch::Tensor finish_gather_global(GatherGlobalAsync async,
                                          const DeepseekV4SPContext& context) {
  CHECK(async.gathered.defined());
  if (async.work.defined()) {
    async.work->wait();
  }
  finish_long_prefill_collective(context);
  auto gathered = async.gathered.flatten(/*start_dim=*/0, /*end_dim=*/1);
  CHECK_GE(gathered.size(0), context.total_tokens);
  return gathered.narrow(/*dim=*/0, /*start=*/0, context.total_tokens);
}

}  // namespace xllm::layer::dsv4_musa_sp
