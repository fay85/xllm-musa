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

#include "core/framework/parallel_state/parallel_state.h"
#include "core/util/utils.h"

#if defined(USE_MUSA)
#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#endif

namespace xllm {
namespace parallel_state {

GatherAsyncCtx launch_gather(const torch::Tensor& input,
                             ProcessGroup* process_group,
                             const std::vector<int32_t>& token_num_list) {
  GatherAsyncCtx ctx;
  if (!process_group) {
    ctx.input = input.contiguous();
    ctx.stacked = ctx.input.unsqueeze(0);
    ctx.token_num_list = {static_cast<int32_t>(input.size(0))};
    return ctx;
  }

  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;
  if (world_size == 1) {
    ctx.input = input.contiguous();
    ctx.stacked = ctx.input.unsqueeze(0);
    ctx.token_num_list = token_num_list;
    return ctx;
  }

  torch::Tensor contiguous_input = input.contiguous();
  CHECK_EQ(contiguous_input.size(0), token_num_list[rank])
      << "sequence-parallel gather local length mismatch: rank=" << rank
      << ", world_size=" << world_size
      << ", local_tensor_tokens=" << contiguous_input.size(0)
      << ", expected_tokens=" << token_num_list[rank];

  const int32_t max_num_tokens = xllm::util::max(token_num_list);
  const int32_t num_padding = max_num_tokens - token_num_list[rank];
  torch::Tensor padded_input = contiguous_input;
  if (num_padding > 0) {
    auto pad_shape = contiguous_input.sizes().vec();
    pad_shape[0] = num_padding;
    torch::Tensor pad_tensor =
        torch::zeros(pad_shape, contiguous_input.options());
    padded_input = torch::cat({contiguous_input, pad_tensor}, /*dim=*/0);
  }

  ctx.input = padded_input;
  auto stacked_shape = ctx.input.sizes().vec();
  stacked_shape.insert(stacked_shape.begin(), world_size);
  ctx.stacked = torch::empty(stacked_shape, ctx.input.options());
  ctx.token_num_list = token_num_list;
  ctx.work = process_group->allgather_base_async(ctx.input, ctx.stacked);
  return ctx;
}

ReduceAsyncCtx launch_reduce(torch::Tensor input, ProcessGroup* process_group) {
  ReduceAsyncCtx ctx;
  if (!process_group || process_group->world_size() <= 1) {
    ctx.tensor = std::move(input);
    return ctx;
  }

  ctx.tensor = input.contiguous();
#if defined(USE_MUSA)
  std::optional<torch::Tensor> custom_output =
      musa_allreduce_tp_custom(process_group, ctx.tensor);
  if (custom_output.has_value()) {
    // Custom AR enqueues on the producer/consumer stream and returns separate
    // output storage. Preserve that tensor instead of changing the in-place
    // ProcessGroup API or waiting for an unrelated MCCL work handle.
    ctx.tensor = std::move(custom_output.value());
    return ctx;
  }
#endif
  ctx.work = process_group->allreduce_async(ctx.tensor);
  return ctx;
}

}  // namespace parallel_state
}  // namespace xllm
