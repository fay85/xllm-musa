/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/layers/musa/deepseek_v41/deep_ep_ll_runtime.h"
#include "core/platform/musa_graph_handle.h"

// Standalone native process, not a Python extension: this covers interpreter
// bootstrap plus forty consecutive LL pairs and empty-rank graph replay.
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 3) << "Usage: runtime_test tcp://host:port output_directory";
  CHECK(std::getenv("RANK") != nullptr);
  const int64_t rank = std::stoll(std::getenv("RANK"));
  xllm::layer::DeepseekV41DeepEpLlRuntime runtime(rank,
                                                  /*world_size=*/8,
                                                  rank,
                                                  /*capacity=*/16,
                                                  /*hidden=*/5120,
                                                  /*experts=*/384,
                                                  argv[1]);
  const torch::Device device(torch::kPrivateUse1, rank);
  torch::DeviceGuard guard(device);
  torch::NoGradGuard no_grad;
  const auto stream = c10::musa::getStreamFromPool(false, rank);
  c10::StreamGuard stream_guard(stream.unwrap());
  const torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  int64_t checks = 0;
  for (const int64_t rows : {1, 4, 6}) {
    const int64_t local_rows = rank == 0 ? rows : 0;
    torch::Tensor input = torch::ones({local_rows, 5120}, options);
    torch::Tensor ids =
        torch::zeros({local_rows, 2}, options.dtype(torch::kInt64));
    torch::Tensor weights =
        torch::full({local_rows, 2}, 0.5, options.dtype(torch::kFloat32));
    const auto forward = [&]() {
      torch::Tensor value = input;
      for (int64_t layer = 0; layer < 40; ++layer) {
        auto received = runtime.bridge().dispatch(value, ids);
        // Identity expert math makes any drop, duplicate weighting or stale
        // buffer visible without loading the full checkpoint.
        value = runtime.bridge().combine(received.first, ids, weights);
      }
      return value;
    };
    ids.select(1, 1).fill_(383);
    for (int64_t iteration = 0; iteration < 2; ++iteration) {
      torch::Tensor output = forward();
      runtime.barrier();
      CHECK(torch::equal(output.cpu(), input.cpu()));
      ++checks;
    }
    runtime.reset_for_capture();
    xllm::MusaGraphHandle graph;
    torch::Tensor output;
    {
      const auto capture_stream = c10::musa::getStreamFromPool(false, rank);
      c10::StreamGuard capture_guard(capture_stream.unwrap());
      graph.capture_begin({0, 0}, musaStreamCaptureModeThreadLocal);
      output = forward();
      graph.capture_end();
    }
    runtime.reset_for_capture();
    LOG(INFO) << "Starting cross-stream replay, rank=" << rank
              << ", rows=" << rows;
    for (int64_t variant = 0; variant < 6; ++variant) {
      input.fill_(variant + 2);
      ids.select(1, 0).fill_(variant * 48);
      ids.select(1, 1).fill_(383 - variant * 48);
      runtime.barrier();
      graph.replay();
      runtime.barrier();
      CHECK(torch::equal(output.cpu(), input.cpu()));
      ++checks;
    }
    LOG(INFO) << "Starting replay without per-iteration CPU barrier, rank="
              << rank << ", rows=" << rows;
    std::vector<torch::Tensor> outputs;
    outputs.reserve(6);
    for (int64_t variant = 0; variant < 6; ++variant) {
      input.fill_(variant + 9);
      ids.select(1, 0).fill_(variant * 48);
      ids.select(1, 1).fill_(383 - variant * 48);
      graph.replay();
      outputs.emplace_back(output.clone());
    }
    runtime.barrier();
    for (int64_t variant = 0; variant < 6; ++variant) {
      CHECK(torch::equal(
          outputs[variant].cpu(),
          torch::full(
              {local_rows, 5120}, variant + 9, options.device(torch::kCPU))));
      ++checks;
    }
    LOG(INFO) << "Native LL forty-layer replay passed, rank=" << rank
              << ", rows=" << rows << ", checks=" << checks;
  }
  runtime.barrier();
  std::ofstream result(std::string(argv[2]) + "/rank" + std::to_string(rank) +
                       ".json");
  CHECK(result.good());
  result << "{\"rank\":" << rank << ",\"checks\":" << checks
         << ",\"standalone_native_bootstrap\":true,\"layers_per_forward\":40,"
            "\"empty_non_owner_ranks\":true,\"cross_stream_replay\":true,"
            "\"barrierless_replay\":true,\"passed\":true}\n";
  return 0;
}
