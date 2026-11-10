/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/npu/paged_attention_graph_operation.h"

#include <acl/acl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace xllm::kernel::npu {
namespace {

using GraphTestParameters =
    std::tuple<torch::ScalarType, int64_t, int64_t, int64_t>;

class PagedAttentionGraphOperationTest
    : public ::testing::TestWithParam<GraphTestParameters> {};

TEST_P(PagedAttentionGraphOperationTest, ReplayRetilesAcrossSequenceLengths) {
  if (c10::impl::getDeviceGuardImpl(torch::kPrivateUse1)->deviceCount() == 0) {
    GTEST_SKIP() << "NPU is unavailable.";
  }
  const auto [dtype, batch_size, num_heads, spec_width] = GetParam();
  torch::set_num_threads(2);
  constexpr int64_t kHeadSize = 256;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kBlocksPerSequence = 257;
  const int64_t query_rows = batch_size * spec_width;
  const torch::Device device(torch::kPrivateUse1, 0);
  const c10::DeviceGuard device_guard(device);
  const auto stream =
      c10_npu::getStreamFromPool(/*isHighPriority=*/false, device.index());
  const c10_npu::NPUStreamGuard stream_guard(stream);
  torch::manual_seed(20261103);
  const auto cpu_options = torch::TensorOptions().dtype(dtype);
  torch::Tensor query_cpu =
      torch::randn({query_rows, num_heads, kHeadSize}, cpu_options);
  const torch::Tensor key_cpu = torch::randn(
      {batch_size * kBlocksPerSequence, kBlockSize, 1, kHeadSize}, cpu_options);
  const torch::Tensor value_cpu = torch::randn_like(key_cpu);
  torch::Tensor query =
      spec_width == 1 ? query_cpu.to(device)
                      : torch::empty({query_rows, num_heads + 1, kHeadSize},
                                     cpu_options.device(device))
                            .narrow(1, 0, num_heads);
  query.copy_(query_cpu);
  const torch::Tensor key = key_cpu.to(device);
  const torch::Tensor value = value_cpu.to(device);
  const torch::Tensor canonical_block_table =
      torch::arange(batch_size * kBlocksPerSequence,
                    torch::TensorOptions().dtype(torch::kInt32))
          .view({batch_size, kBlocksPerSequence})
          .repeat_interleave(spec_width, /*dim=*/0);
  torch::Tensor block_table =
      spec_width == 1
          ? canonical_block_table.to(device)
          : torch::empty(
                {query_rows, kBlocksPerSequence + 3},
                torch::TensorOptions().dtype(torch::kInt32).device(device))
                .narrow(1, 0, kBlocksPerSequence);
  block_table.copy_(canonical_block_table);
  torch::Tensor output = torch::empty_like(query);
  auto resources = std::make_shared<PagedAttentionGraphResources>(1);
  PagedAttentionGraphOperation operation(
      resources, query, key, value, block_table, output, /*scale=*/1.0 / 16.0);

  // Initialize the operator before capture, then capture a short-context plan.
  operation.prepare(std::vector<int32_t>(query_rows, 16));
  operation.execute();
  ASSERT_EQ(aclrtSynchronizeStream(stream.stream()), ACL_SUCCESS);
  c10_npu::NPUGraph graph;
  c10_npu::NPUEvent ready(ACL_EVENT_EXTERNAL);
  graph.capture_begin();
  ready.block(stream);
  ready.reset(stream);
  c10_npu::graph_task_group_begin(stream);
  operation.execute();
  const auto handle = c10_npu::graph_task_group_end(stream);
  graph.capture_end();

  const auto update_stream = c10_npu::getStreamFromPool(
      /*isHighPriority=*/false, device.index());
  // Cross KV-block and split-plan boundaries in both directions. The same
  // graph and output addresses must remain valid throughout all replays.
  int32_t step = 0;
  for (const int32_t length :
       {16, 17, 128, 129, 2048, 2049, 32824, 32825, 32826, 2048, 16}) {
    query_cpu.neg_();
    query.copy_(query_cpu);
    torch::Tensor block_order =
        torch::arange(kBlocksPerSequence, torch::kInt64);
    if (step++ % 2 != 0) {
      block_order[0] = 1;
      block_order[1] = 0;
    }
    const torch::Tensor current_block_table =
        canonical_block_table.index_select(1, block_order);
    block_table.copy_(current_block_table);
    std::vector<int32_t> lengths(query_rows);
    std::vector<torch::Tensor> reference;
    reference.reserve(query_rows);
    for (int64_t query_row = 0; query_row < query_rows; ++query_row) {
      const int64_t row = query_row / spec_width;
      const int64_t token = query_row % spec_width;
      // Each draft token sees its own causal KV prefix, even though all the
      // tokens of one sequence share the same physical cache blocks.
      lengths[query_row] =
          std::max<int32_t>(1, length - static_cast<int32_t>(row)) +
          static_cast<int32_t>(token);
      const int64_t used_blocks =
          (lengths[query_row] + kBlockSize - 1) / kBlockSize;
      const torch::Tensor indices = current_block_table[query_row]
                                        .narrow(0, 0, used_blocks)
                                        .to(torch::kInt64);
      const torch::Tensor row_key = key_cpu.index_select(0, indices)
                                        .reshape({-1, kHeadSize})
                                        .narrow(0, 0, lengths[query_row])
                                        .to(torch::kFloat32);
      const torch::Tensor row_value = value_cpu.index_select(0, indices)
                                          .reshape({-1, kHeadSize})
                                          .narrow(0, 0, lengths[query_row])
                                          .to(torch::kFloat32);
      reference.emplace_back(
          torch::matmul(query_cpu[query_row].to(torch::kFloat32), row_key.t())
              .div(16.0)
              .softmax(-1)
              .matmul(row_value));
    }
    graph.replay();
    {
      const c10_npu::NPUStreamGuard update_guard(update_stream);
      operation.prepare(lengths);
      c10_npu::graph_task_update_begin(update_stream, handle);
      operation.execute();
      c10_npu::graph_task_update_end(update_stream);
      ready.record(update_stream);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream.stream()), ACL_SUCCESS);
    const torch::Tensor actual = output.to(torch::kCPU).to(torch::kFloat32);
    const torch::Tensor expected = torch::stack(reference);
    ASSERT_TRUE(
        torch::allclose(actual, expected, /*rtol=*/0.02, /*atol=*/0.002))
        << "length=" << length << ", batch=" << batch_size
        << ", heads=" << num_heads << ", dtype=" << dtype
        << ", spec_width=" << spec_width
        << ", max_abs_error=" << (actual - expected).abs().max().item<float>();
  }
  graph.reset();
  EXPECT_EQ(resources->stream_count(), 1);
  // A second node reuses the stream's context; another stream gets its own.
  PagedAttentionGraphOperation same_stream(
      resources, query, key, value, block_table, output, /*scale=*/1.0 / 16.0);
  EXPECT_EQ(resources->stream_count(), 1);
  const auto other_stream = c10_npu::getStreamFromPool(false, device.index());
  const c10_npu::NPUStreamGuard other_guard(other_stream);
  PagedAttentionGraphOperation other(
      resources, query, key, value, block_table, output, /*scale=*/1.0 / 16.0);
  EXPECT_EQ(resources->stream_count(), 2);
}

INSTANTIATE_TEST_SUITE_P(
    DecodeAndExpandedVerify,
    PagedAttentionGraphOperationTest,
    ::testing::Values(GraphTestParameters{torch::kHalf, 1, 4, 1},
                      GraphTestParameters{torch::kBFloat16, 4, 8, 3}));

}  // namespace
}  // namespace xllm::kernel::npu
