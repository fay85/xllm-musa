/* Copyright 2025-2026 The xLLM Authors.

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
#include <glog/logging.h>

#include <memory>
#include <vector>

#include "core/kernels/musa/attention_runner.h"
#include "torch_musa/csrc/core/MUSAStream.h"

// torch_musa's native MUSAGraph header expects this stream alias.
namespace at::musa {
using c10::musa::MUSAStream;
}  // namespace at::musa

#include "torch_musa/csrc/aten/musa/MUSAGraph.h"

namespace xllm::runtime::musa {

class PiecewiseGraphs final {
 public:
  enum class InstructionType { kGraph, kRunner };

  PiecewiseGraphs() = default;
  ~PiecewiseGraphs() = default;
  PiecewiseGraphs(PiecewiseGraphs&&) noexcept = default;
  PiecewiseGraphs& operator=(PiecewiseGraphs&&) noexcept = default;

  void add_graph(std::unique_ptr<at::musa::MUSAGraph>&& graph);
  void add_attention_runner(::xllm::kernel::musa::AttentionRunner&& runner);
  void replay(const ::xllm::kernel::musa::AttentionReplayParams& runner_params);
  size_t size() const { return instructions_.size(); }
  bool empty() const { return instructions_.empty(); }
  size_t num_graphs() const { return graphs_.size(); }
  size_t num_runners() const;
  bool requires_plan_info() const { return requires_plan_info_; }
  bool has_gdn_prefill_runner() const { return has_gdn_prefill_runner_; }

 private:
  std::vector<std::unique_ptr<at::musa::MUSAGraph>> graphs_;
  std::vector<std::unique_ptr<::xllm::kernel::musa::AttentionRunner>>
      attention_runners_;
  std::vector<InstructionType> instructions_;
  bool requires_plan_info_ = false;
  bool has_gdn_prefill_runner_ = false;
};

}  // namespace xllm::runtime::musa
