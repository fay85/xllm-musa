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

#include "core/kernels/musa/piecewise_graphs.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>

namespace xllm::runtime::musa {

void PiecewiseGraphs::add_graph(std::unique_ptr<MusaGraphSegment>&& graph) {
  CHECK(graph != nullptr) << "Graph segment must not be null";
  graphs_.push_back(std::move(graph));
  instructions_.push_back(InstructionType::GRAPH);
}

void PiecewiseGraphs::add_attention_runner(
    ::xllm::kernel::musa::AttentionRunner&& runner) {
  attention_runners_.push_back(
      std::make_unique<::xllm::kernel::musa::AttentionRunner>(
          std::move(runner)));
  instructions_.push_back(InstructionType::RUNNER);
}

void PiecewiseGraphs::replay(
    const ::xllm::kernel::musa::AttentionReplayParams& runner_params) {
  CHECK_GT(graphs_.size(), 0) << "Piecewise graph must have at least one graph";

  std::size_t graph_idx = 0;
  std::size_t runner_idx = 0;

  for (const InstructionType instruction : instructions_) {
    if (instruction == InstructionType::GRAPH) {
      CHECK_LT(graph_idx, graphs_.size()) << "Graph index out of range";
      graphs_[graph_idx]->replay();
      ++graph_idx;
    } else {
      CHECK_LT(runner_idx, attention_runners_.size())
          << "Runner index out of range";
      attention_runners_[runner_idx]->run_replay(runner_params);
      ++runner_idx;
    }
  }

  CHECK_EQ(graph_idx, graphs_.size()) << "Not all graphs were replayed";
  CHECK_EQ(runner_idx, attention_runners_.size())
      << "Not all runners were replayed";
}

}  // namespace xllm::runtime::musa
