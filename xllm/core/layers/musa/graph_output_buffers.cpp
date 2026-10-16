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

void PiecewiseGraphMatmulBufferPool::reset_rings(
    std::vector<BufferRing>& rings) {
  for (auto& ring : rings) {
    ring.next = 0;
  }
}

void PiecewiseGraphMatmulBufferPool::reset_forward_slots() {
  reset_rings(output_bufs_);
  reset_rings(gated_rms_norm_output_bufs_);
  reset_rings(gdn_query_bufs_);
  reset_rings(gdn_key_bufs_);
  reset_rings(gdn_value_bufs_);
  reset_rings(gdn_output_bufs_);
  reset_rings(gdn_gate_bufs_);
  reset_rings(gdn_beta_bufs_);
  reset_rings(gdn_initial_state_bufs_);
  reset_rings(gdn_final_state_bufs_);
  reset_rings(gdn_kkt_bufs_);
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_tensor(
    std::vector<BufferRing>& rings,
    c10::IntArrayRef sizes,
    const torch::TensorOptions& options,
    const char* name) {
  for (BufferRing& ring : rings) {
    if (ring.bufs.empty()) {
      continue;
    }
    const torch::Tensor& probe = ring.bufs.front();
    if (probe.sizes().equals(sizes) &&
        probe.scalar_type() == options.dtype().toScalarType() &&
        probe.device() == options.device()) {
      if (ring.next < ring.bufs.size()) {
        return ring.bufs[ring.next++];
      }
      CHECK(!frozen_) << "Piecewise graph " << name
                      << " exhausted buffer ring during capture/replay; "
                         "shape was under-provisioned in eager warmup";
      torch::Tensor buffer = torch::empty(sizes, options);
      ring.bufs.emplace_back(buffer);
      ring.next = ring.bufs.size();
      return buffer;
    }
  }
  CHECK(!frozen_) << "Piecewise graph " << name
                  << " shape was not prepared during eager warmup";
  BufferRing ring;
  torch::Tensor buffer = torch::empty(sizes, options);
  ring.bufs.emplace_back(buffer);
  ring.next = 1;
  rings.emplace_back(std::move(ring));
  return buffer;
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get(
    const torch::Tensor& input,
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

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_query(
    const torch::Tensor& input) {
  return get_tensor(gdn_query_bufs_, input.sizes(), input.options(), "GDN Q");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_key(
    const torch::Tensor& input) {
  return get_tensor(gdn_key_bufs_, input.sizes(), input.options(), "GDN K");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_value(
    const torch::Tensor& input) {
  return get_tensor(gdn_value_bufs_, input.sizes(), input.options(), "GDN V");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_output(
    const torch::Tensor& input) {
  return get_tensor(
      gdn_output_bufs_, input.sizes(), input.options(), "GDN output");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_gate(
    const torch::Tensor& input) {
  return get_tensor(gdn_gate_bufs_,
                    input.sizes(),
                    input.options().dtype(torch::kFloat32),
                    "GDN gate");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_beta(
    const torch::Tensor& input) {
  return get_tensor(gdn_beta_bufs_,
                    input.sizes(),
                    input.options().dtype(torch::kFloat32),
                    "GDN beta");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_initial_state(
    const torch::Tensor& reference) {
  return get_tensor(gdn_initial_state_bufs_,
                    reference.sizes(),
                    reference.options().dtype(torch::kFloat32),
                    "GDN initial state");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_final_state(
    const torch::Tensor& reference) {
  return get_tensor(gdn_final_state_bufs_,
                    reference.sizes(),
                    reference.options().dtype(torch::kFloat32),
                    "GDN final state");
}

torch::Tensor PiecewiseGraphMatmulBufferPool::get_gdn_kkt(
    const torch::Tensor& key,
    int64_t num_v_heads) {
  return get_tensor(gdn_kkt_bufs_,
                    {1, key.size(1), num_v_heads, 64},
                    key.options(),
                    "GDN KKT");
}

void PiecewiseGraphMatmulBufferPool::freeze() { frozen_ = true; }

PiecewiseGraphMatmulBufferScope::PiecewiseGraphMatmulBufferScope(
    PiecewiseGraphMatmulBufferPool* buffer_pool)
    : previous_buffer_pool_(active_buffer_pool) {
  CHECK(buffer_pool != nullptr);
  active_buffer_pool = buffer_pool;
  active_buffer_pool->reset_forward_slots();
}

PiecewiseGraphMatmulBufferScope::~PiecewiseGraphMatmulBufferScope() {
  active_buffer_pool = previous_buffer_pool_;
}

PiecewiseGraphMatmulBufferPool*
PiecewiseGraphMatmulBufferScope::current_buffer_pool() {
  return active_buffer_pool;
}

}  // namespace xllm::layer::musa
