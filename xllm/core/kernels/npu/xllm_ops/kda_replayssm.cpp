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

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

// Geometry of the Ascend 910_93 ReplaySSM operators.
constexpr int64_t kVerifyWidth = 8;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kFeatures = kNumHeads * kHeadDim;
constexpr int64_t kCorrectionMaxElements = std::numeric_limits<int32_t>::max();

torch::Device correction_device(const torch::Tensor& tensor) {
  CHECK(tensor.defined());
  const torch::Device device = tensor.device();
  CHECK(device.is_privateuseone()) << "ReplaySSM requires an NPU device";
  return device;
}

void check_correction_tensor(const torch::Tensor& tensor,
                             torch::ScalarType dtype,
                             const torch::Device& device,
                             bool require_contiguous = true,
                             bool allow_empty = false) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  const int64_t elements = tensor.numel();
  CHECK((allow_empty || elements > 0) && elements <= kCorrectionMaxElements)
      << "Invalid ReplaySSM tensor element count: " << elements;
  CHECK(!require_contiguous || tensor.is_contiguous());
  const int64_t format = at_npu::native::custom_ops::get_npu_format(tensor);
  CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCL ||
        format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NCDHW)
      << "ReplaySSM requires base NPU storage format, got " << format;
}

int64_t check_correction_state_shape(const torch::Tensor& state) {
  constexpr int64_t trailing_dims[] = {kNumHeads, kHeadDim, kHeadDim};
  CHECK(state.dim() == 4 &&
        state.sizes().slice(1) == torch::IntArrayRef(trailing_dims))
      << "Invalid ReplaySSM state shape: " << state.sizes();
  return state.size(0);
}

struct CorrectionAddressRange {
  uintptr_t begin;
  uintptr_t end;
};

CorrectionAddressRange correction_address_range(const torch::Tensor& tensor,
                                                int64_t span) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const uint64_t bytes = span * tensor.element_size();
  CHECK_LE(bytes, std::numeric_limits<uintptr_t>::max() - begin);
  return {begin, begin + bytes};
}

CorrectionAddressRange correction_address_range(const torch::Tensor& tensor) {
  return correction_address_range(tensor, tensor.numel());
}

CorrectionAddressRange correction_record_range(const torch::Tensor& record,
                                               int64_t num_slots) {
  const int64_t shape[] = {num_slots, kVerifyWidth, kNumHeads, kHeadDim};
  CHECK_EQ(record.sizes(), torch::IntArrayRef(shape));
  return correction_address_range(record);
}

void check_correction_disjoint(CorrectionAddressRange first,
                               CorrectionAddressRange second) {
  CHECK(first.end <= second.begin || second.end <= first.begin)
      << "ReplaySSM writable record storage must be disjoint";
}

void check_fold_disjoint(std::vector<CorrectionAddressRange>& states,
                         const std::vector<CorrectionAddressRange>& inputs) {
  std::sort(states.begin(), states.end(), [](const auto& a, const auto& b) {
    return a.begin < b.begin;
  });
  for (size_t i = 1; i < states.size(); ++i) {
    CHECK_LE(states[i - 1].end, states[i].begin)
        << "KDA correction mutable states must not overlap";
  }
  // Disjoint state ranges have increasing end addresses as well as starts.
  // Each read-only input can overlap only if the last state starting before
  // its end extends into it. Read-only inputs may overlap each other.
  for (const CorrectionAddressRange& input : inputs) {
    auto next = std::lower_bound(
        states.begin(),
        states.end(),
        input.end,
        [](const auto& state, uintptr_t end) { return state.begin < end; });
    CHECK(next == states.begin() || std::prev(next)->end <= input.begin)
        << "KDA correction mutable state and read-only inputs must not overlap";
  }
}

void record_correction_stream(const torch::Tensor& tensor,
                              c10_npu::NPUStream stream) {
  c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(),
                                             stream);
}

}  // namespace

torch::Tensor kda_replayssm_verify_record(const torch::Tensor& query,
                                          const torch::Tensor& key,
                                          const torch::Tensor& value,
                                          const torch::Tensor& gate,
                                          const torch::Tensor& beta,
                                          const torch::Tensor& state,
                                          const torch::Tensor& a_log,
                                          const torch::Tensor& dt_bias,
                                          const torch::Tensor& state_indices,
                                          torch::Tensor& normalized_keys,
                                          torch::Tensor& corrections,
                                          torch::Tensor& decays,
                                          double scale) {
  const torch::Device device = correction_device(query);
  check_correction_tensor(
      query, torch::kBFloat16, device, /*require_contiguous=*/false);
  CHECK_EQ(query.dim(), 3);
  const int64_t num_tokens = query.size(0);
  const int64_t token_stride = query.stride(0);
  const int64_t qkv_shape[] = {num_tokens, kNumHeads, kHeadDim};
  const int64_t qkv_strides[] = {token_stride, kHeadDim, 1};
  CHECK_EQ(query.sizes(), torch::IntArrayRef(qkv_shape));
  CHECK_EQ(query.strides(), torch::IntArrayRef(qkv_strides));
  CHECK(token_stride == kFeatures || token_stride == 3 * kFeatures);
  CHECK_EQ(num_tokens % kVerifyWidth, 0);
  for (const torch::Tensor& tensor : {key, value}) {
    check_correction_tensor(
        tensor, torch::kBFloat16, device, /*require_contiguous=*/false);
    CHECK_EQ(tensor.sizes(), query.sizes());
    CHECK_EQ(tensor.strides(), query.strides());
  }
  const int64_t qkv_span = (num_tokens - 1) * token_stride + kFeatures;
  CHECK_LE(qkv_span, kCorrectionMaxElements);
  const int64_t num_seqs = num_tokens / kVerifyWidth;
  check_correction_tensor(state_indices, torch::kInt32, device);
  CHECK_EQ(state_indices.sizes(), torch::IntArrayRef(num_seqs));
  check_correction_tensor(gate, torch::kBFloat16, device);
  check_correction_tensor(beta, torch::kBFloat16, device);
  CHECK_EQ(gate.sizes(), query.sizes());
  const int64_t beta_shape[] = {num_tokens, kNumHeads};
  CHECK_EQ(beta.sizes(), torch::IntArrayRef(beta_shape));
  for (const torch::Tensor& tensor :
       {state, a_log, dt_bias, normalized_keys, corrections, decays}) {
    check_correction_tensor(tensor, torch::kFloat32, device);
  }
  const int64_t num_slots = check_correction_state_shape(state);
  CHECK_EQ(a_log.sizes(), torch::IntArrayRef(kNumHeads));
  CHECK_EQ(dt_bias.sizes(), torch::IntArrayRef(kFeatures));
  const float kernel_scale = static_cast<float>(scale);
  CHECK(std::isfinite(kernel_scale) && kernel_scale > 0.0f)
      << "Invalid ReplaySSM scale: " << scale;

  const CorrectionAddressRange inputs[] = {
      correction_address_range(query, qkv_span),
      correction_address_range(key, qkv_span),
      correction_address_range(value, qkv_span),
      correction_address_range(gate),
      correction_address_range(beta),
      correction_address_range(state),
      correction_address_range(a_log),
      correction_address_range(dt_bias),
      correction_address_range(state_indices)};
  const CorrectionAddressRange records[] = {
      correction_record_range(normalized_keys, num_slots),
      correction_record_range(corrections, num_slots),
      correction_record_range(decays, num_slots)};
  for (size_t i = 0; i < std::size(records); ++i) {
    for (const CorrectionAddressRange input : inputs) {
      check_correction_disjoint(records[i], input);
    }
    for (size_t j = 0; j < i; ++j) {
      check_correction_disjoint(records[i], records[j]);
    }
  }

  torch::Tensor output = torch::empty(query.sizes(), query.options());
  const c10::DeviceGuard device_guard(device);
  const c10_npu::NPUStream stream =
      c10_npu::getCurrentNPUStream(device.index());
  for (const torch::Tensor& tensor : {query,
                                      key,
                                      value,
                                      gate,
                                      beta,
                                      state,
                                      a_log,
                                      dt_bias,
                                      state_indices,
                                      normalized_keys,
                                      corrections,
                                      decays,
                                      output}) {
    record_correction_stream(tensor, stream);
  }
  EXEC_NPU_CMD(aclnnKdaReplayssmVerifyRecord,
               query,
               key,
               value,
               gate,
               beta,
               state,
               a_log,
               dt_bias,
               state_indices,
               scale,
               token_stride,
               normalized_keys,
               corrections,
               decays,
               output);
  return output;
}

void kda_replayssm_correction_fold(
    const std::vector<torch::Tensor>& states,
    const std::vector<torch::Tensor>& normalized_keys,
    const std::vector<torch::Tensor>& corrections,
    const std::vector<torch::Tensor>& decays,
    const torch::Tensor& state_indices,
    const torch::Tensor& accepted_lengths) {
  CHECK(!states.empty());
  CHECK_EQ(normalized_keys.size(), states.size());
  CHECK_EQ(corrections.size(), states.size());
  CHECK_EQ(decays.size(), states.size());
  const torch::Device device = correction_device(states.front());
  check_correction_tensor(state_indices,
                          torch::kInt32,
                          device,
                          /*require_contiguous=*/true,
                          /*allow_empty=*/true);
  check_correction_tensor(accepted_lengths,
                          torch::kInt32,
                          device,
                          /*require_contiguous=*/true,
                          /*allow_empty=*/true);
  CHECK_EQ(state_indices.dim(), 1);
  CHECK_EQ(accepted_lengths.sizes(), state_indices.sizes());
  const int64_t num_seqs = state_indices.numel();
  if (num_seqs == 0) {
    return;
  }
  CHECK_LE(states.size(), kCorrectionMaxElements / kNumHeads / num_seqs);
  const int64_t num_layers = static_cast<int64_t>(states.size());
  const c10::DeviceGuard device_guard(device);
  const c10_npu::NPUStream stream =
      c10_npu::getCurrentNPUStream(device.index());
  std::vector<CorrectionAddressRange> state_ranges;
  std::vector<CorrectionAddressRange> input_ranges;
  state_ranges.reserve(states.size());
  input_ranges.reserve(3 * states.size() + 2);
  input_ranges.emplace_back(correction_address_range(state_indices));
  input_ranges.emplace_back(correction_address_range(accepted_lengths));
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const torch::Tensor* tensors[] = {&states[layer],
                                      &normalized_keys[layer],
                                      &corrections[layer],
                                      &decays[layer]};
    for (const torch::Tensor* tensor : tensors) {
      check_correction_tensor(*tensor, torch::kFloat32, device);
      record_correction_stream(*tensor, stream);
    }
    const torch::Tensor& state = states[layer];
    const int64_t num_slots = check_correction_state_shape(state);
    state_ranges.emplace_back(correction_address_range(state));
    for (const torch::Tensor& record :
         {normalized_keys[layer], corrections[layer], decays[layer]}) {
      input_ranges.emplace_back(correction_record_range(record, num_slots));
    }
  }
  check_fold_disjoint(state_ranges, input_ranges);
  record_correction_stream(state_indices, stream);
  record_correction_stream(accepted_lengths, stream);
  const torch::TensorList state_list(states);
  const torch::TensorList key_list(normalized_keys);
  const torch::TensorList correction_list(corrections);
  const torch::TensorList decay_list(decays);
  EXEC_NPU_CMD(aclnnKdaReplayssmCorrectionFold,
               state_list,
               key_list,
               correction_list,
               decay_list,
               state_indices,
               accepted_lengths,
               state_list);
}

}  // namespace xllm::kernel::npu
