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

#include "core/kernels/musa/kda.h"

#include <c10/musa/MUSAGuard.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <cmath>
#include <limits>
#include <string>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

// Installed MATE chunk_kda_workspace.hpp / mate.kda._allocate_kda_workspace
// define this shared prepare/recurrence FFI ABI in activation elements.

// MATE jit/kda_ops.py fixes chunk=16 and head_dim=128; xLLM adds no limit.
constexpr int64_t kChunkSize = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kDecayedElements = 2 * kChunkSize * kHeadDim;
constexpr int64_t kRestoredElements = kChunkSize * kHeadDim;
constexpr int64_t kMatrixElements = kChunkSize * kChunkSize;
constexpr int64_t kRestoredOffset = kDecayedElements;
constexpr int64_t kPOffset = kRestoredOffset + kRestoredElements;
constexpr int64_t kInverseOffset = kPOffset + kMatrixElements;
constexpr int64_t kTotalOffset = kInverseOffset + kMatrixElements;
constexpr int64_t kRecordStride = kTotalOffset + kHeadDim;
// MATE's ChunkKdaMp31SchedulePolicy requires five prepare CTAs per MP.
constexpr int64_t kPrepareCtasPerMp = 5;

void check_musa(musaError_t status) {
  CHECK_EQ(status, musaSuccess) << musaGetErrorString(status);
}

torch::Tensor as_4d(const torch::Tensor& tensor, bool varlen) {
  return varlen && tensor.dim() == 3 ? tensor.unsqueeze(0) : tensor;
}

void check_tensor(const torch::Tensor& tensor,
                  const torch::Device& device,
                  torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK(tensor.device() == device);
  CHECK(tensor.scalar_type() == dtype);
}

std::string dtype_suffix(torch::ScalarType dtype) {
  return dtype == torch::kHalf ? "fp16" : "bf16";
}

void launch(const torch::Tensor& q,
            const torch::Tensor& k,
            const torch::Tensor& v,
            const torch::Tensor& g,
            const torch::Tensor& beta,
            torch::Tensor& out,
            const torch::Tensor& initial,
            const torch::Tensor& final,
            const torch::Tensor& cu,
            const std::optional<torch::Tensor>& a_log,
            const std::optional<torch::Tensor>& dt_bias,
            double scale,
            double lower_bound,
            int32_t mp_count) {
  // Match MATE's exported function names. Heads and token counts remain
  // runtime operands; no model-specific shape buckets are required.
  const std::string dtype = dtype_suffix(q.scalar_type());
  const std::string state_dtype =
      initial.scalar_type() == torch::kFloat32 ? "fp32" : dtype;
  const std::string prepare_uri = "prepare_kda_dtype_" + dtype + "_gate_" +
                                  (a_log.has_value() ? "1" : "0") +
                                  "_varlen_1_normqk_1_seqlens_i64_arch_mp31";
  const std::string recur_uri = "kda_recurrence_dtype_" + dtype +
                                "_varlen_1_seqlens_i64_state_" + state_dtype +
                                "_in_1_out_1_cs_0_arch_mp31";
  // Resolve both modules before mutating output/state.
  auto prepare = get_function(prepare_uri, prepare_uri);
  auto recur = get_function(recur_uri, recur_uri);

  int64_t records = (q.size(1) + kChunkSize - 1) / kChunkSize + cu.numel() - 1;
  int64_t heads = v.size(2);
  auto workspace = torch::empty({records, heads, kRecordStride}, q.options());
  auto field = [&](int64_t offset, int64_t rows, int64_t cols) {
    return workspace.as_strided({records, heads, rows, cols},
                                {heads * kRecordStride, kRecordStride, cols, 1},
                                offset);
  };
  auto decayed = field(/*offset=*/0, 2 * kChunkSize, kHeadDim);
  auto restored = field(kRestoredOffset, kChunkSize, kHeadDim);
  auto inverse = field(kInverseOffset, kChunkSize, kChunkSize);
  auto p = field(kPOffset, kChunkSize, kChunkSize);
  auto total = workspace.as_strided({records, heads, kHeadDim},
                                    {heads * kRecordStride, kRecordStride, 1},
                                    kTotalOffset);
  auto metadata = torch::empty({mp_count * kPrepareCtasPerMp, 4},
                               q.options().dtype(torch::kInt32));
  auto indices =
      torch::arange(cu.numel() - 1, q.options().dtype(torch::kInt32));
  // Destroy the guard before scratch tensors, and order outputs before the
  // caller performs further Torch operations.
  TvmffiStreamGuard stream_guard(q.device());
  prepare(to_ffi_tensor(q),
          to_ffi_tensor(k),
          to_ffi_tensor(g),
          to_ffi_tensor(beta),
          to_ffi_tensor(decayed),
          to_ffi_tensor(restored),
          to_ffi_tensor(inverse),
          to_ffi_tensor(p),
          to_ffi_tensor(total),
          to_ffi_tensor(metadata),
          to_ffi_tensor(cu),
          to_ffi_optional_tensor(a_log),
          to_ffi_optional_tensor(dt_bias),
          scale,
          lower_bound,
          /*use_qk_l2norm_in_kernel=*/true);
  recur(to_ffi_tensor(v),
        to_ffi_tensor(out),
        to_ffi_tensor(decayed),
        to_ffi_tensor(restored),
        to_ffi_tensor(inverse),
        to_ffi_tensor(p),
        to_ffi_tensor(total),
        to_ffi_tensor(indices),
        to_ffi_tensor(initial),
        to_ffi_tensor(final),
        to_ffi_tensor(cu),
        ffi::Optional<ffi::Tensor>());
}
}  // namespace

int64_t flash_kda_workspace_size(int64_t /*total_tokens*/,
                                 int64_t /*num_heads*/,
                                 int64_t /*num_sequences*/) {
  return 0;
}

void flash_kda(const torch::Tensor& q_input,
               const torch::Tensor& k_input,
               const torch::Tensor& v_input,
               const torch::Tensor& g_input,
               const torch::Tensor& beta_input,
               std::optional<double> scale,
               torch::Tensor& out_input,
               const std::optional<torch::Tensor>& a_log,
               const std::optional<torch::Tensor>& dt_bias,
               double lower_bound,
               const std::optional<torch::Tensor>& initial_state,
               const std::optional<torch::Tensor>& final_state,
               const std::optional<torch::Tensor>& cu_seqlens) {
  CHECK(q_input.device().is_privateuseone());
  c10::musa::MUSAGuard device_guard(q_input.device());
  bool varlen = cu_seqlens.has_value();
  auto q = as_4d(q_input, varlen);
  auto k = as_4d(k_input, varlen);
  auto v = as_4d(v_input, varlen);
  auto g = as_4d(g_input, varlen);
  auto out = as_4d(out_input, varlen);
  auto beta =
      varlen && beta_input.dim() == 2 ? beta_input.unsqueeze(0) : beta_input;
  CHECK(q.scalar_type() == torch::kHalf || q.scalar_type() == torch::kBFloat16);
  for (const auto& tensor : {q, k, v, g, out}) {
    check_tensor(tensor, q.device(), q.scalar_type());
    CHECK_EQ(tensor.dim(), 4);
    CHECK_EQ(tensor.size(3), 128) << "MATE chunk KDA requires D=128";
    CHECK_EQ(tensor.stride(3), 1);
  }
  CHECK(k.sizes() == q.sizes());
  CHECK(g.sizes() == v.sizes());
  CHECK(out.sizes() == v.sizes());
  CHECK_EQ(q.size(0), v.size(0));
  CHECK_EQ(q.size(1), v.size(1));
  CHECK_GT(q.size(2), 0);
  CHECK_EQ(v.size(2) % q.size(2), 0);
  check_tensor(beta, q.device(), q.scalar_type());
  CHECK(beta.sizes() == v.sizes().slice(0, 3));
  beta = beta.contiguous();
  CHECK_EQ(a_log.has_value(), dt_bias.has_value());
  if (a_log.has_value()) {
    check_tensor(*a_log, q.device(), torch::kFloat32);
    check_tensor(*dt_bias, q.device(), torch::kFloat32);
    CHECK(a_log->sizes() == torch::IntArrayRef({v.size(2)}));
    CHECK(dt_bias->sizes() == torch::IntArrayRef({v.size(2), 128}));
    CHECK(a_log->is_contiguous());
    CHECK(dt_bias->is_contiguous());
  }
  torch::Tensor cu;
  if (varlen) {
    CHECK_EQ(q.size(0), 1);
    CHECK(cu_seqlens->device() == q.device());
    CHECK_EQ(cu_seqlens->dim(), 1);
    CHECK(cu_seqlens->scalar_type() == torch::kInt32 ||
          cu_seqlens->scalar_type() == torch::kInt64);
    CHECK_GE(cu_seqlens->numel(), 1);
    cu = cu_seqlens->to(torch::kInt64).contiguous();
  } else {
    cu = torch::arange(q.size(0) + 1, q.options().dtype(torch::kInt64)) *
         q.size(1);
    q = q.reshape({1, q.size(0) * q.size(1), q.size(2), 128});
    k = k.reshape_as(q);
    g = g.reshape({1, v.size(0) * v.size(1), v.size(2), 128});
    beta = beta.reshape({1, v.size(0) * v.size(1), v.size(2)});
    v = v.reshape_as(g);
    // Preserve arbitrary caller output strides by copying back after
    // recurrence.
    out = torch::empty_like(v);
  }
  int64_t nseq = cu.numel() - 1;
  torch::ScalarType state_dtype =
      initial_state.has_value() ? initial_state->scalar_type()
      : final_state.has_value() ? final_state->scalar_type()
                                : q.scalar_type();
  CHECK(state_dtype == q.scalar_type() || state_dtype == torch::kFloat32);
  for (const auto& state : {initial_state, final_state}) {
    if (!state.has_value()) {
      continue;
    }
    check_tensor(*state, q.device(), state_dtype);
    CHECK(state->sizes() == torch::IntArrayRef({nseq, v.size(2), 128, 128}));
    CHECK(state->is_contiguous());
  }
  auto opts = q.options().dtype(state_dtype);
  auto initial = initial_state.has_value()
                     ? *initial_state
                     : torch::zeros({nseq, v.size(2), 128, 128}, opts);
  auto final =
      final_state.has_value() ? *final_state : torch::empty_like(initial);
  if (q.size(1) == 0 || nseq == 0) {
    final.copy_(initial);
    return;
  }
  for (int64_t size : {q.size(1), q.size(2), v.size(2), nseq}) {
    CHECK_LE(size, std::numeric_limits<int32_t>::max());
  }
  int device = q.get_device();
  musaDeviceProp properties;
  check_musa(musaGetDeviceProperties(&properties, device));
  CHECK_EQ(properties.major * 10 + properties.minor, 31)
      << "Installed MATE chunk KDA kernels target MP31";
  launch(q,
         k,
         v,
         g,
         beta,
         out,
         initial,
         final,
         cu,
         a_log,
         dt_bias,
         scale.value_or(1.0 / std::sqrt(128.0)),
         lower_bound,
         properties.multiProcessorCount);
  if (!varlen) {
    out_input.copy_(out.reshape_as(out_input));
  }
}
}  // namespace xllm::kernel::musa
