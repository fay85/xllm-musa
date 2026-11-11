/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <musa.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "core/kernels/musa/musa_aot_config.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

constexpr const char* kKernelName = "fused_moe_kernel";
constexpr int64_t kTopK = 8;
constexpr int64_t kGroupSize = 128;

enum class MoeStage : int32_t { GATE = 0, DOWN = 1 };

using KernelConfig = aot::FusedMoeKernel;

struct KernelArguments {
  uint64_t input;
  uint64_t weights;
  uint64_t output;
  uint64_t input_scale;
  uint64_t weight_scale;
  uint64_t topk_weights;
  uint64_t sorted_token_ids;
  uint64_t expert_ids;
  uint64_t num_tokens_post_padded;
  int32_t output_size;
  int32_t input_size;
  int32_t max_padded_tokens;
  int32_t valid_assignments;
  int32_t input_stride;
  int32_t weight_expert_stride;
  int32_t weight_output_stride;
  int32_t bias_expert_stride;
  int32_t bias_output_stride;
  int32_t output_assignment_stride;
  int32_t input_scale_row_stride;
  int32_t weight_scale_expert_stride;
  int32_t weight_scale_output_stride;
};

const KernelConfig* find_config(int64_t batch_size,
                                bool use_fp8,
                                MoeStage stage = MoeStage::GATE) {
  for (const KernelConfig& config : aot::kFusedMoeKernels) {
    if (config.batch_size == batch_size && config.use_fp8 == use_fp8 &&
        config.down == (stage == MoeStage::DOWN)) {
      return &config;
    }
  }
  return nullptr;
}

void check_driver_result(MUresult result, const char* operation) {
  if (result == MUresult::MUSA_SUCCESS) {
    return;
  }
  const char* error = nullptr;
  muGetErrorString(result, &error);
  CHECK(false) << operation << " failed with MUSA driver error "
               << static_cast<int32_t>(result) << ": "
               << (error == nullptr ? "unknown" : error);
}

class KernelHandle final {
 public:
  explicit KernelHandle(const std::string& path) {
    check_driver_result(muModuleLoad(&module_, path.c_str()), "muModuleLoad");
    check_driver_result(muModuleGetFunction(&function_, module_, kKernelName),
                        "muModuleGetFunction");
  }

  ~KernelHandle() {
    if (module_ != nullptr) {
      muModuleUnload(module_);
    }
  }

  KernelHandle(const KernelHandle&) = delete;
  KernelHandle& operator=(const KernelHandle&) = delete;
  MUfunction function() const { return function_; }

 private:
  MUmodule module_ = nullptr;
  MUfunction function_ = nullptr;
};

KernelHandle& get_kernel(int32_t device_index,
                         int64_t batch_size,
                         MoeStage stage,
                         bool use_fp8) {
  static std::mutex mutex;
  static std::unordered_map<int64_t, KernelHandle> handles;

  const int64_t key =
      ((static_cast<int64_t>(device_index) * 8 + batch_size - 1) * 2 +
       static_cast<int32_t>(stage)) *
          2 +
      static_cast<int32_t>(use_fp8);
  std::lock_guard<std::mutex> lock(mutex);
  const auto iterator = handles.find(key);
  if (iterator != handles.end()) {
    return iterator->second;
  }

  const KernelConfig* config = find_config(batch_size, use_fp8, stage);
  CHECK(config != nullptr) << "Unsupported MUSA fused MoE AOT batch";
  const std::string root = get_musa_ops_root();
  validate_musa_ops_bundle(root);
  const std::string path = root + "/" + config->path;
  return handles.try_emplace(key, path).first->second;
}

uint64_t tensor_pointer(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
}

int32_t checked_int32(int64_t value, const char* name) {
  CHECK_GE(value, 0) << name << " must be non-negative.";
  CHECK_LE(value, static_cast<int64_t>(INT32_MAX))
      << name << " exceeds int32 range: " << value;
  return static_cast<int32_t>(value);
}

torch::Tensor activate_gate_up(const torch::Tensor& gate_up) {
  torch::Tensor output;
  active(output, gate_up, "silu", /*is_gated=*/true);
  return output;
}

torch::Tensor launch_gemm(const torch::Tensor& input,
                          const torch::Tensor& input_scale,
                          const torch::Tensor& weights,
                          const torch::Tensor& weight_scale,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& sorted_token_ids,
                          const torch::Tensor& expert_ids,
                          const torch::Tensor& num_tokens_post_padded,
                          MoeStage stage,
                          bool use_fp8) {
  const int64_t batch_size = topk_weights.size(0);
  const int64_t assignment_count = batch_size * kTopK;
  const KernelConfig* config = find_config(batch_size, use_fp8, stage);
  CHECK(config != nullptr) << "Unsupported MUSA fused MoE AOT stage";
  const KernelConfig& launch_config = *config;
  torch::Tensor output;
  if (stage == MoeStage::GATE) {
    output = torch::empty({assignment_count, weights.size(1)},
                          input.options().dtype(torch::kBFloat16));
  } else {
    output = torch::empty({batch_size, kTopK, weights.size(1)},
                          input.options().dtype(torch::kBFloat16));
  }

  KernelArguments arguments = {
      .input = tensor_pointer(input),
      .weights = tensor_pointer(weights),
      .output = tensor_pointer(output),
      .input_scale = tensor_pointer(input_scale),
      .weight_scale = tensor_pointer(weight_scale),
      .topk_weights = tensor_pointer(topk_weights),
      .sorted_token_ids = tensor_pointer(sorted_token_ids),
      .expert_ids = tensor_pointer(expert_ids),
      .num_tokens_post_padded = tensor_pointer(num_tokens_post_padded),
      .output_size = checked_int32(weights.size(1), "output_size"),
      .input_size = checked_int32(weights.size(2), "input_size"),
      .max_padded_tokens =
          checked_int32(sorted_token_ids.size(0), "max_padded_tokens"),
      .valid_assignments = checked_int32(assignment_count, "valid_assignments"),
      .input_stride = checked_int32(input.stride(0), "input_stride"),
      .weight_expert_stride =
          checked_int32(weights.stride(0), "weight_expert_stride"),
      .weight_output_stride =
          checked_int32(weights.stride(1), "weight_output_stride"),
      .bias_expert_stride = 0,
      .bias_output_stride = 0,
      .output_assignment_stride =
          checked_int32(output.stride(-2), "output_assignment_stride"),
      .input_scale_row_stride =
          input_scale.defined()
              ? checked_int32(input_scale.stride(0), "input_scale_row_stride")
              : 0,
      .weight_scale_expert_stride =
          weight_scale.defined() ? checked_int32(weight_scale.stride(0),
                                                 "weight_scale_expert_stride")
                                 : 0,
      .weight_scale_output_stride =
          weight_scale.defined() ? checked_int32(weight_scale.stride(1),
                                                 "weight_scale_output_stride")
                                 : 0,
  };
  std::array<void*, 22> parameters = {
      &arguments.input,
      &arguments.weights,
      &arguments.output,
      &arguments.input_scale,
      &arguments.weight_scale,
      &arguments.topk_weights,
      &arguments.sorted_token_ids,
      &arguments.expert_ids,
      &arguments.num_tokens_post_padded,
      &arguments.output_size,
      &arguments.input_size,
      &arguments.max_padded_tokens,
      &arguments.valid_assignments,
      &arguments.input_stride,
      &arguments.weight_expert_stride,
      &arguments.weight_output_stride,
      &arguments.bias_expert_stride,
      &arguments.bias_output_stride,
      &arguments.output_assignment_stride,
      &arguments.input_scale_row_stride,
      &arguments.weight_scale_expert_stride,
      &arguments.weight_scale_output_stride,
  };

  c10::musa::MUSAGuard device_guard(input.device().index());
  KernelHandle& kernel =
      get_kernel(input.device().index(), batch_size, stage, use_fp8);
  const uint32_t grid_x = static_cast<uint32_t>(
      ((sorted_token_ids.size(0) + launch_config.block_m - 1) /
       launch_config.block_m) *
      ((weights.size(1) + launch_config.block_n - 1) / launch_config.block_n));
  MUstream stream = reinterpret_cast<MUstream>(
      c10::musa::getCurrentMUSAStream(input.device().index()).stream());
  check_driver_result(muLaunchKernel(kernel.function(),
                                     grid_x,
                                     1,
                                     1,
                                     launch_config.threads_per_block,
                                     1,
                                     1,
                                     launch_config.shared_memory,
                                     stream,
                                     parameters.data(),
                                     nullptr),
                      "muLaunchKernel");
  return output;
}

void check_common_inputs(const torch::Tensor& hidden_states,
                         const torch::Tensor& w13,
                         const torch::Tensor& w2,
                         const torch::Tensor& topk_weights,
                         const torch::Tensor& topk_ids) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK(hidden_states.device().is_privateuseone());
  CHECK(hidden_states.is_contiguous());
  CHECK_GT(hidden_states.size(1), 0);
  CHECK_EQ(hidden_states.size(1) % 128, 0);
  CHECK_EQ(w13.dim(), 3);
  CHECK_EQ(w2.dim(), 3);
  CHECK(w13.is_contiguous() && w2.is_contiguous());
  CHECK_EQ(topk_weights.dim(), 2);
  CHECK_EQ(topk_ids.dim(), 2);
  CHECK_EQ(topk_weights.size(0), hidden_states.size(0));
  CHECK_EQ(topk_ids.size(0), hidden_states.size(0));
  CHECK_EQ(topk_weights.size(1), kTopK);
  CHECK_EQ(topk_ids.size(1), kTopK);
  CHECK_EQ(topk_weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(topk_ids.scalar_type(), torch::kInt32);
  CHECK(topk_weights.is_contiguous() && topk_ids.is_contiguous());
  CHECK_EQ(w13.size(0), w2.size(0));
  CHECK_EQ(w13.size(2), hidden_states.size(1));
  CHECK_EQ(w13.size(1) % 2, 0);
  CHECK_EQ(w2.size(1), hidden_states.size(1));
  CHECK_EQ(w2.size(2), w13.size(1) / 2);
  CHECK_GT(w2.size(2), 0);
  CHECK_EQ(w2.size(2) % 128, 0);
  CHECK(hidden_states.device() == w13.device() &&
        hidden_states.device() == w2.device() &&
        hidden_states.device() == topk_weights.device() &&
        hidden_states.device() == topk_ids.device());
  for (const auto& tensor : {hidden_states, w13, w2, topk_weights, topk_ids}) {
    CHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16, 0)
        << "MUSA fused MoE AOT tensors must be 16-byte aligned";
  }
}

void check_fp8_inputs(const torch::Tensor& hidden_states,
                      const torch::Tensor& w13,
                      const torch::Tensor& w13_scale,
                      const torch::Tensor& w2,
                      const torch::Tensor& w2_scale,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& topk_ids) {
  check_common_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  CHECK_EQ(w13.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(w2.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(w13_scale.dim(), 3);
  CHECK_EQ(w2_scale.dim(), 3);
  CHECK_EQ(w13_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(w2_scale.scalar_type(), torch::kFloat32);
  CHECK(w13_scale.is_contiguous() && w2_scale.is_contiguous());
  CHECK_EQ(tensor_pointer(w13_scale) % 16, 0);
  CHECK_EQ(tensor_pointer(w2_scale) % 16, 0);
  CHECK_EQ(w13_scale.size(0), w13.size(0));
  CHECK_EQ(w13_scale.size(1) * kGroupSize, w13.size(1));
  CHECK_EQ(w13_scale.size(2) * kGroupSize, w13.size(2));
  CHECK_EQ(w2_scale.size(0), w2.size(0));
  CHECK_EQ(w2_scale.size(1) * kGroupSize, w2.size(1));
  CHECK_EQ(w2_scale.size(2) * kGroupSize, w2.size(2));
  CHECK(hidden_states.device() == w13_scale.device() &&
        hidden_states.device() == w2_scale.device() &&
        hidden_states.device() == w2.device());
}

void check_bf16_inputs(const torch::Tensor& hidden_states,
                       const torch::Tensor& w13,
                       const torch::Tensor& w2,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& topk_ids) {
  check_common_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  CHECK_EQ(w13.scalar_type(), torch::kBFloat16);
  CHECK_EQ(w2.scalar_type(), torch::kBFloat16);
}

void prepare_kernels(const torch::Device& device, bool use_fp8) {
  static std::mutex mutex;
  static std::unordered_set<int64_t> prepared;
  const int32_t device_index = device.index();
  const int64_t key =
      static_cast<int64_t>(device_index) * 2 + static_cast<int32_t>(use_fp8);
  std::lock_guard<std::mutex> lock(mutex);
  if (prepared.contains(key)) {
    return;
  }

  c10::musa::MUSAGuard device_guard(device_index);
  for (const KernelConfig& config : aot::kFusedMoeKernels) {
    if (config.use_fp8 != use_fp8) {
      continue;
    }
    get_kernel(device_index,
               config.batch_size,
               config.down ? MoeStage::DOWN : MoeStage::GATE,
               use_fp8);
  }
  prepared.insert(key);
}

}  // namespace

bool fused_moe_aot_available(int64_t num_tokens) {
  return find_config(num_tokens, /*use_fp8=*/true) != nullptr;
}

bool fused_moe_bf16_aot_available(int64_t num_tokens) {
  return find_config(num_tokens, /*use_fp8=*/false) != nullptr;
}

void prepare_fused_moe_aot(const torch::Device& device) {
  prepare_kernels(device, /*use_fp8=*/true);
}

void prepare_fused_moe_bf16_aot(const torch::Device& device) {
  prepare_kernels(device, /*use_fp8=*/false);
}

torch::Tensor fused_moe_aot_fp8(const torch::Tensor& hidden_states,
                                const torch::Tensor& w13,
                                const torch::Tensor& w13_scale,
                                const torch::Tensor& w2,
                                const torch::Tensor& w2_scale,
                                const torch::Tensor& topk_weights,
                                const torch::Tensor& topk_ids) {
  check_fp8_inputs(
      hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids);
  const KernelConfig* config =
      find_config(hidden_states.size(0), /*use_fp8=*/true);
  CHECK(config != nullptr)
      << "MUSA fused MoE AOT supports decode batches 1 through 8; got "
      << hidden_states.size(0);

  auto [sorted_token_ids, expert_ids, num_tokens_post_padded] =
      moe_align_block_size(topk_ids, w13.size(0), config->block_m);
  auto [quantized_hidden, hidden_scale] =
      per_token_group_quant_fp8(hidden_states, kGroupSize);
  auto gate_up = launch_gemm(quantized_hidden,
                             hidden_scale,
                             w13,
                             w13_scale,
                             topk_weights,
                             sorted_token_ids,
                             expert_ids,
                             num_tokens_post_padded,
                             MoeStage::GATE,
                             /*use_fp8=*/true);
  auto activated = activate_gate_up(gate_up);
  auto [quantized_activated, activated_scale] =
      per_token_group_quant_fp8(activated, kGroupSize);
  auto down = launch_gemm(quantized_activated,
                          activated_scale,
                          w2,
                          w2_scale,
                          topk_weights,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          MoeStage::DOWN,
                          /*use_fp8=*/true);
  return moe_sum_reduce(down);
}

torch::Tensor fused_moe_aot_bf16(const torch::Tensor& hidden_states,
                                 const torch::Tensor& w13,
                                 const torch::Tensor& w2,
                                 const torch::Tensor& topk_weights,
                                 const torch::Tensor& topk_ids) {
  check_bf16_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  const KernelConfig* config =
      find_config(hidden_states.size(0), /*use_fp8=*/false);
  CHECK(config != nullptr)
      << "MUSA BF16 fused MoE AOT supports decode batches 1 through 8; got "
      << hidden_states.size(0);

  auto [sorted_token_ids, expert_ids, num_tokens_post_padded] =
      moe_align_block_size(topk_ids, w13.size(0), config->block_m);
  const torch::Tensor no_scale;
  auto gate_up = launch_gemm(hidden_states,
                             no_scale,
                             w13,
                             no_scale,
                             topk_weights,
                             sorted_token_ids,
                             expert_ids,
                             num_tokens_post_padded,
                             MoeStage::GATE,
                             /*use_fp8=*/false);
  auto activated = activate_gate_up(gate_up);
  auto down = launch_gemm(activated,
                          no_scale,
                          w2,
                          no_scale,
                          topk_weights,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          MoeStage::DOWN,
                          /*use_fp8=*/false);
  return moe_sum_reduce(down);
}

}  // namespace xllm::kernel::musa
