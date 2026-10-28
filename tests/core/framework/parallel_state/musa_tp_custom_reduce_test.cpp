/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Manual two-rank integration probe. It is deliberately not registered with
// CTest: the caller must first reserve two GPUs and a free rendezvous port.
// Run one process per rank with the same visible GPU list:
//   musa_tp_custom_reduce_test <rank:0|1> <port> <custom|mccl>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_musa/csrc/aten/musa/MUSAGraph.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"
#include "core/platform/stream.h"

namespace {

void check_reduction(xllm::ProcessGroup* group,
                     torch::Tensor input,
                     const torch::Tensor& expected_cpu,
                     bool expect_custom) {
  torch::Tensor original = input.clone();
  auto ctx = xllm::parallel_state::launch_reduce(input, group);
  CHECK_EQ(ctx.work.defined(), !expect_custom)
      << "Unexpected collective backend";
  const bool output_aliases_input = ctx.tensor.is_alias_of(input);
  torch::Tensor output = xllm::parallel_state::finish_reduce(std::move(ctx));
  CHECK(torch::equal(output.to(torch::kCPU), expected_cpu))
      << "TP2 reduction output differs from the exact sum";
  if (expect_custom) {
    CHECK(!output_aliases_input);
    CHECK(torch::equal(input.to(torch::kCPU), original.to(torch::kCPU)))
        << "Custom reduction overwrote its input";
  }
}

void run_probe(int32_t rank, int32_t port, bool use_custom) {
  CHECK_EQ(setenv("XLLM_MUSA_ENABLE_TP_CUSTOM_AR",
                  use_custom ? "1" : "0",
                  /*overwrite=*/1),
           0);
  const torch::Device device("musa:" + std::to_string(rank));
  xllm::Device runtime_device(device);
  runtime_device.set_device();
  auto group = xllm::create_process_group(rank,
                                          /*world_size=*/2,
                                          /*rank_size=*/2,
                                          port,
                                          /*trans=*/false,
                                          "127.0.0.1",
                                          "tp_group",
                                          device);
  xllm::Stream stream(device);
  auto stream_guard = stream.set_stream_guard();
  torch::NoGradGuard no_grad;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  for (int64_t rows : {int64_t{1}, int64_t{16}, int64_t{64}}) {
    torch::Tensor pattern = torch::arange(rows * 5120, torch::kInt64)
                                .remainder(17)
                                .reshape({rows, 5120})
                                .to(torch::kBFloat16);
    torch::Tensor input = pattern.to(device) + rank;
    check_reduction(group.get(), std::move(input), pattern * 2 + 1, use_custom);
  }

  // Model layers enqueue many producer kernels and reductions without a host
  // synchronization between them. Synchronous constant-input checks can miss
  // peer-cache visibility or stream-ordering failures in this workload.
  std::vector<torch::Tensor> queued_outputs;
  std::vector<torch::Tensor> queued_inputs;
  queued_outputs.reserve(96);
  queued_inputs.reserve(96);
  constexpr int64_t kQueuedRows[] = {
      1, 16, 64, 199, 208, 229, 241, 436, 521, 800};
  for (int32_t iteration = 0; iteration < 96; ++iteration) {
    const int64_t rows = kQueuedRows[iteration % 10];
    const int32_t value = rank + 1 + iteration % 19;
    torch::Tensor left = torch::ones({rows, 128}, options);
    torch::Tensor right = torch::full({128, 5120}, value, options);
    torch::Tensor produced = torch::matmul(left, right);
    queued_inputs.emplace_back(produced);
    auto ctx = xllm::parallel_state::launch_reduce(produced, group.get());
    CHECK_EQ(ctx.work.defined(), !use_custom);
    queued_outputs.emplace_back(
        xllm::parallel_state::finish_reduce(std::move(ctx)));
  }
  CHECK_EQ(stream.synchronize(), 0);
  for (int32_t iteration = 0; iteration < 96; ++iteration) {
    const int64_t rows = kQueuedRows[iteration % 10];
    const int32_t sum = 128 * (3 + 2 * (iteration % 19));
    CHECK(torch::equal(queued_outputs[iteration].to(torch::kCPU),
                       torch::full({rows, 5120}, sum, torch::kBFloat16)))
        << "Queued producer/reduction mismatch at iteration " << iteration;
    if (use_custom) {
      const int32_t value = 128 * (rank + 1 + iteration % 19);
      CHECK(torch::equal(queued_inputs[iteration].to(torch::kCPU),
                         torch::full({rows, 5120}, value, torch::kBFloat16)))
          << "Queued reduction overwrote its input at iteration " << iteration;
    }
  }
  LOG(INFO) << "MUSA_TP2_QUEUED_REDUCE_PROBE_PASS rank=" << rank;

  // Both ranks select MCCL before execution for unsupported dtype/size.
  torch::Tensor integers =
      torch::full({2, 17}, rank + 1, options.dtype(torch::kInt64));
  check_reduction(group.get(),
                  std::move(integers),
                  torch::full({2, 17}, 3, torch::kInt64),
                  /*expect_custom=*/false);
  torch::Tensor large = torch::full({820, 5120}, rank + 1, options);
  check_reduction(group.get(),
                  std::move(large),
                  torch::full({820, 5120}, 3, torch::kBFloat16),
                  /*expect_custom=*/false);

  torch::Tensor graph_input = torch::full({16, 5120}, rank + 1, options);
  check_reduction(group.get(),
                  graph_input,
                  torch::full({16, 5120}, 3, torch::kBFloat16),
                  use_custom);
  graph_input.fill_(rank + 1);
  CHECK_EQ(stream.synchronize(), 0);

  at::musa::MUSAGraph graph;
  graph.capture_begin(/*pool=*/{0, 0}, musaStreamCaptureModeThreadLocal);
  auto captured = xllm::parallel_state::launch_reduce(graph_input, group.get());
  CHECK_EQ(captured.work.defined(), !use_custom);
  torch::Tensor graph_output =
      xllm::parallel_state::finish_reduce(std::move(captured));
  graph.capture_end();
  CHECK_EQ(stream.synchronize(), 0);
  xllm::finalize_musa_custom_allreduce_graphs();

  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    graph_input.fill_(rank + 1 + iteration);
    graph.replay();
    CHECK_EQ(stream.synchronize(), 0);
    CHECK(torch::equal(
        graph_output.to(torch::kCPU),
        torch::full({16, 5120}, 3 + iteration * 2, torch::kBFloat16)))
        << "Graph replay used stale data";
    if (use_custom) {
      CHECK(torch::equal(
          graph_input.to(torch::kCPU),
          torch::full({16, 5120}, rank + 1 + iteration, torch::kBFloat16)));
    }
  }
  graph.reset();
  CHECK_EQ(stream.synchronize(), 0);
  // Finish a common MCCL operation before either rank releases its IPC memory.
  torch::Tensor done = torch::ones({1}, options.dtype(torch::kInt64));
  group->allreduce(done);
  CHECK_EQ(stream.synchronize(), 0);
  LOG(INFO) << "MUSA_TP2_REDUCE_PROBE_PASS rank=" << rank
            << ", backend=" << (use_custom ? "custom" : "mccl");
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 4)
      << "Usage: musa_tp_custom_reduce_test <rank:0|1> <port> <custom|mccl>";
  const int32_t rank = std::stoi(argv[1]);
  const int32_t port = std::stoi(argv[2]);
  const std::string backend = argv[3];
  CHECK_GE(rank, 0);
  CHECK_LT(rank, 2);
  CHECK(backend == "custom" || backend == "mccl");
  run_probe(rank, port, backend == "custom");
  return 0;
}
