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

#include <torch/torch.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
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
  // Hash routing consumes the same unsharded ids in every layer. Retaining
  // the model-owned view avoids another token collective around each MoE.
  torch::Tensor global_input_ids;
};

struct GatherGlobalAsync {
  torch::Tensor local_contiguous;
  torch::Tensor gathered;
  c10::intrusive_ptr<c10d::Work> work;
};

inline bool long_prefill_rendezvous_enabled(const DeepseekV4SPContext& context) {
  const char* setting = std::getenv("XLLM_DSV4_LONG_PREFILL_RENDEZVOUS");
  return context.total_tokens >= 4096 && setting != nullptr &&
         std::string(setting) == "1";
}

inline void begin_long_prefill_collective(const DeepseekV4SPContext& context) {
  if (!long_prefill_rendezvous_enabled(context)) {
    return;
  }
  // Do not let a fast rank launch spinning MCCL work while another rank is
  // still producing local tensors or reclaiming allocator storage. This is
  // opt-in and long-prefill-only; short B1 and Graph decode never enter it.
  LOG_FIRST_N(INFO, 1)
      << "[DSV4_LONG_PREFILL_RENDEZVOUS] total_tokens=" << context.total_tokens;
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

inline torch::Tensor gather_global(const torch::Tensor& local_tensor,
                                   const DeepseekV4SPContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  CHECK_EQ(local_tensor.size(0), context.local_tokens)
      << "DeepSeek-V4 sequence parallel local tensor has unexpected rows";
  torch::Tensor gathered;
  if (long_prefill_rendezvous_enabled(context)) {
    // Allocate before the peer rendezvous: an allocation on a lagging rank
    // must not encounter another already-enqueued collective on that rank.
    auto contiguous = local_tensor.contiguous();
    auto shape = contiguous.sizes().vec();
    shape.insert(shape.begin(), context.world_size);
    gathered = torch::empty(shape, contiguous.options());
    begin_long_prefill_collective(context);
    auto work = context.process_group->allgather_base_async(contiguous, gathered);
    work->wait();
    finish_long_prefill_collective(context);
  } else {
    gathered =
        context.process_group->allgather_base_sync(local_tensor.contiguous());
  }
  CHECK_EQ(gathered.dim(), local_tensor.dim() + 1);
  gathered = gathered.flatten(/*start_dim=*/0, /*end_dim=*/1);
  CHECK_GE(gathered.size(0), context.total_tokens);
  return gathered.narrow(/*dim=*/0, /*start=*/0, context.total_tokens);
}

inline GatherGlobalAsync launch_gather_global(
    const torch::Tensor& local_tensor,
    const DeepseekV4SPContext& context) {
  CHECK(local_tensor.defined());
  CHECK_EQ(local_tensor.size(0), context.local_tokens)
      << "DeepSeek-V4 sequence parallel local tensor has unexpected rows";
  GatherGlobalAsync async;
  async.local_contiguous = local_tensor.contiguous();
  auto gathered_shape = async.local_contiguous.sizes().vec();
  gathered_shape.insert(gathered_shape.begin(), context.world_size);
  async.gathered =
      torch::empty(gathered_shape, async.local_contiguous.options());
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
