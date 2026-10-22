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

#pragma once

#include "torch_musa/csrc/core/MUSAStream.h"

// torch_musa's native MUSAGraph header expects this stream alias.
namespace at::musa {
using c10::musa::MUSAStream;
}  // namespace at::musa

#include "torch_musa/csrc/aten/musa/MUSAGraph.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "core/kernels/musa/attention_runner.h"

namespace xllm::runtime::musa {

using MusaGraphSegment = at::musa::MUSAGraph;

class PiecewiseGraphs final {
 public:
  enum class InstructionType {
    GRAPH,
    RUNNER,
  };

  void add_graph(std::unique_ptr<MusaGraphSegment>&& graph);

  void add_attention_runner(::xllm::kernel::musa::AttentionRunner&& runner);

  std::size_t size() const { return graphs_.size(); }

  std::size_t num_runners() const { return attention_runners_.size(); }

  void replay(const ::xllm::kernel::musa::AttentionReplayParams& runner_params);

 private:
  std::vector<std::unique_ptr<MusaGraphSegment>> graphs_;
  std::vector<std::unique_ptr<::xllm::kernel::musa::AttentionRunner>>
      attention_runners_;
  std::vector<InstructionType> instructions_;
};

}  // namespace xllm::runtime::musa
