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

#include <vector>

namespace xllm::layer::musa {

class PiecewiseGraphMatmulBufferPool final {
 public:
  torch::Tensor get(const torch::Tensor& input, const torch::Tensor& weight);
  torch::Tensor get_gated_rms_norm_output(const torch::Tensor& input);
  void freeze();

 private:
  torch::Tensor get_tensor(std::vector<torch::Tensor>& buffers,
                           c10::IntArrayRef sizes,
                           const torch::TensorOptions& options,
                           const char* name);

  std::vector<torch::Tensor> output_bufs_;
  std::vector<torch::Tensor> gated_rms_norm_output_bufs_;
  bool frozen_ = false;
};

class PiecewiseGraphMatmulBufferScope final {
 public:
  explicit PiecewiseGraphMatmulBufferScope(
      PiecewiseGraphMatmulBufferPool* buffer_pool);
  ~PiecewiseGraphMatmulBufferScope();

  PiecewiseGraphMatmulBufferScope(const PiecewiseGraphMatmulBufferScope&) =
      delete;
  PiecewiseGraphMatmulBufferScope& operator=(
      const PiecewiseGraphMatmulBufferScope&) = delete;

  static PiecewiseGraphMatmulBufferPool* current_buffer_pool();

 private:
  PiecewiseGraphMatmulBufferPool* previous_buffer_pool_;
};

}  // namespace xllm::layer::musa
