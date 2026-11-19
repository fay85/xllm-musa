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

#pragma once

#include <torch_musa/csrc/distributed/ProcessGroupMCCL.h>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  ProcessGroupImpl(int32_t global_rank,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
    c10::intrusive_ptr<c10d::ProcessGroupMCCL::Options> pg_options =
        c10d::ProcessGroupMCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    pg_options->group_name = group_name;
#endif
    int32_t rank = global_rank;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
      rank = local_rank;
    }

    auto store = create_tcp_store(host, port, rank);
    pg_ = std::make_unique<c10d::ProcessGroupMCCL>(
        store, rank, rank_size, pg_options);
    custom_all_reduce_ = std::make_unique<MusaCustomAllReduce>(
        rank, rank_size, group_name, device, store);
  }

  // The caller already resolved this rank's position inside group_ranks.
  ProcessGroupImpl(int32_t global_rank,
                   int32_t local_rank,
                   const std::vector<int32_t>& /*group_ranks*/,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
    c10::intrusive_ptr<c10d::ProcessGroupMCCL::Options> pg_options =
        c10d::ProcessGroupMCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    pg_options->group_name = group_name;
#endif
    auto store = create_tcp_store(host, port, local_rank);
    pg_ = std::make_unique<c10d::ProcessGroupMCCL>(
        store, local_rank, rank_size, pg_options);
    // Explicit CP ranks take this constructor. Prefill PCP broadcasts and
    // allgathers on cp_group, and those collectives use this all-reduce.
    if (group_name == "cp_group") {
      custom_all_reduce_ = std::make_unique<MusaCustomAllReduce>(
          local_rank, rank_size, group_name, device, store);
    }
  }

  // Enqueue a blocking-call broadcast on the caller's compute stream.
  // This is deliberately separate from broadcast_async(): MCCL may return
  // no Work for asyncOp=false, while the public async API requires a handle.
  void broadcast_on_compute_stream(torch::Tensor& input, int32_t root_rank) {
    CHECK(pg_ != nullptr);
    CHECK(input.is_contiguous());
    CHECK_EQ(input.device(), device());
    CHECK_GE(root_rank, 0);
    CHECK_LT(root_rank, pg_->getSize());
    if (pg_->getSize() <= 1) {
      return;
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(1);
    tensors.emplace_back(input);
    c10d::BroadcastOptions options;
    options.rootRank = root_rank;
    options.asyncOp = false;
    auto work = pg_->broadcast(tensors, options);
    if (work.defined()) {
      work->wait();
    }
  }

  // Keep a dependent prefill gather on the caller's stream, without changing
  // the public asynchronous allgather contract or requiring a Work handle.
  void allgather_on_compute_stream(const torch::Tensor& input,
                                  torch::Tensor& output) {
    CHECK(pg_ != nullptr);
    CHECK(input.defined() && output.defined());
    CHECK(input.is_contiguous() && output.is_contiguous());
    CHECK_EQ(input.device(), device());
    CHECK_EQ(output.device(), device());
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_EQ(output.dim(), input.dim() + 1);
    CHECK_EQ(output.size(0), pg_->getSize());
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      CHECK_EQ(output.size(dim + 1), input.size(dim));
    }
    CHECK_EQ(output.numel(), input.numel() * pg_->getSize());
    CHECK_NE(input.data_ptr(), output.data_ptr());
    CHECK_GT(pg_->getSize(), 1);
    torch::Tensor input_buffer = input;
    c10d::AllgatherOptions options;
    options.asyncOp = false;
    auto work = pg_->_allgather_base(output, input_buffer, options);
    if (work.defined()) {
      work->wait();
    }
  }

  // Same tensor contract as reduce_scatter(), with deterministic ownership
  // on the caller's compute stream instead of a separate MCCL stream.
  void reduce_scatter_on_compute_stream(torch::Tensor& input,
                                        torch::Tensor& output) {
    CHECK(pg_ != nullptr);
    CHECK(input.defined() && output.defined());
    CHECK(input.is_contiguous() && output.is_contiguous());
    CHECK_EQ(input.device(), device());
    CHECK_EQ(output.device(), device());
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_GT(output.numel(), 0);
    CHECK_EQ(input.numel(), output.numel() * pg_->getSize());
    CHECK_NE(input.data_ptr(), output.data_ptr());
    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> outputs;
    inputs.reserve(1);
    outputs.reserve(1);
    inputs.emplace_back(input);
    outputs.emplace_back(output);
    c10d::ReduceScatterOptions options;
    options.reduceOp = c10d::ReduceOp::SUM;
    options.asyncOp = false;
    auto work = pg_->reduce_scatter_tensor_coalesced(outputs, inputs, options);
    if (work.defined()) {
      work->wait();
    }
  }

  std::optional<torch::Tensor> allreduce_custom(
      torch::Tensor& input,
      const std::optional<torch::Tensor>& output = std::nullopt) {
    if (custom_all_reduce_) {
      return custom_all_reduce_->allreduce(input, output);
    }
    return std::nullopt;
  }

  std::optional<torch::Tensor> allreduce_tp_custom(torch::Tensor& input) {
    if (custom_all_reduce_) {
      return custom_all_reduce_->allreduce_tp(input);
    }
    return std::nullopt;
  }

  bool custom_allreduce_enabled() const {
    return custom_all_reduce_ != nullptr && custom_all_reduce_->enabled();
  }

  void long_prefill_host_barrier() {
    CHECK(custom_all_reduce_ != nullptr);
    custom_all_reduce_->long_prefill_host_barrier();
  }

 private:
  std::unique_ptr<MusaCustomAllReduce> custom_all_reduce_;
};

}  // namespace xllm
