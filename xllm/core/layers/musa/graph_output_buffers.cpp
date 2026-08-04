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

#include "layers/musa/graph_output_buffers.h"

#include <glog/logging.h>

namespace xllm::layer::musa {
namespace {

thread_local PiecewiseGraphMatmulBufferPool* active_buffer_pool = nullptr;

}  // namespace

torch::Tensor PiecewiseGraphMatmulBufferPool::get_tensor(
    std::vector<torch::Tensor>& buffers,
    c10::IntArrayRef sizes,
    const torch::TensorOptions& options,
    const char* name) {
  for (const torch::Tensor& buffer : buffers) {
    if (buffer.sizes().equals(sizes) &&
        buffer.scalar_type() == options.dtype().toScalarType() &&
        buffer.device() == options.device()) {
      return buffer;
    }
  }
  CHECK(!frozen_) << "Piecewise graph " << name
                  << " shape was not prepared during eager warmup";
  torch::Tensor buffer = torch::empty(sizes, options);
  buffers.emplace_back(buffer);
  return buffer;
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get(const torch::Tensor& input,
                                                  const torch::Tensor& weight) {
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weight.dim(), 2);
  return get_tensor(output_bufs_,
                    {input.size(0), weight.size(0)},
                    input.options(),
                    "matmul buffer");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gated_rms_norm_output(
    const torch::Tensor& input) {
  CHECK_GE(input.dim(), 1);
  CHECK_GT(input.numel(), 0);
  const int64_t last_dim = input.size(-1);
  return get_tensor(gated_rms_norm_output_bufs_,
                    {input.numel() / last_dim, last_dim},
                    input.options(),
                    "gated RMSNorm buffer")
      .view(input.sizes());
}

void PiecewiseGraphMatmulBufferPool::freeze() { frozen_ = true; }

PiecewiseGraphMatmulBufferScope::PiecewiseGraphMatmulBufferScope(
    PiecewiseGraphMatmulBufferPool* buffer_pool)
    : previous_buffer_pool_(active_buffer_pool) {
  CHECK(buffer_pool != nullptr);
  active_buffer_pool = buffer_pool;
}

PiecewiseGraphMatmulBufferScope::~PiecewiseGraphMatmulBufferScope() {
  active_buffer_pool = previous_buffer_pool_;
}

PiecewiseGraphMatmulBufferPool*
PiecewiseGraphMatmulBufferScope::current_buffer_pool() {
  return active_buffer_pool;
}

}  // namespace xllm::layer::musa
