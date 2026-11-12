/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "core/framework/parallel_state/process_group.h"

namespace xllm::layer {
// One IPC workspace shared by sequential GLM layers. Attention and the shared
// expert remain TP8; routed experts hold 36 complete experts per rank.
class MusaMegaMoe final {
 public:
  static std::shared_ptr<MusaMegaMoe> acquire(ProcessGroup* group);
  explicit MusaMegaMoe(ProcessGroup* group);
  ~MusaMegaMoe();
  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& ids,
                        const torch::Tensor& weights,
                        const torch::Tensor& w13,
                        const torch::Tensor& s13,
                        const torch::Tensor& w2,
                        const torch::Tensor& s2,
                        double clamp,
                        const torch::Tensor& output_buffer = {});
  static std::pair<torch::Tensor, torch::Tensor> transform_gate_up(
      const torch::Tensor& weight,
      const torch::Tensor& scale);

 private:
  ProcessGroup* group_;
  torch::Tensor buffer_;
  torch::Tensor pointers_;
  std::vector<torch::Tensor> views_;
};
}  // namespace xllm::layer
