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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "common/flash_comm1_context.h"
#include "core/framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {
namespace musa {
class MatmulOutputBuffers final {
 public:
  std::optional<torch::Tensor> get(const torch::Tensor& input,
                                   const torch::Tensor& weight);

 private:
  torch::Tensor decode_output_buf_;
};

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl final : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool bias,
                           bool gather_output,
                           const QuantArgs& quant_args,
                           ProcessGroup* process_group,
                           const torch::TensorOptions& options,
                           int32_t output_replicas = 1);

  explicit ColumnParallelLinearImpl(const ModelContext& context);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);

  // load_state_dict for merged weights with variable shard sizes
  void load_state_dict(const StateDict& state_dict,
                       int32_t shard_tensor_count,
                       const std::vector<int64_t>& shard_sizes);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  torch::Tensor weight() const { return weight_; }
  std::optional<torch::Tensor> bias() const;
  ProcessGroup* process_group() const { return process_group_; }
  bool is_weight_loaded() const { return weight_is_loaded_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(bias);

  DEFINE_FUSED_WEIGHT(
      weight_scale_inv);  // Block-wise FP8 inverse-scale grid
                          // [ceil(N/bn), ceil(K/bk)] (DeepSeek-style).

  int64_t rank_;
  int64_t world_size_;
  int64_t weight_rank_;
  int64_t weight_world_size_;
  // whether to gather the output
  bool gather_output_;
  torch::Device device_;
  // parallel process group
  ProcessGroup* process_group_;

  // quantization args
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  std::optional<std::string> resolved_weight_quant_method_;
  bool block_fp8_resolved_unquantized_ = false;
  mutable MatmulOutputBuffers output_buffers_;
};
TORCH_MODULE(ColumnParallelLinear);

class QKVParallelLinearImpl final : public torch::nn::Module {
 public:
  QKVParallelLinearImpl(int64_t hidden_size,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        int64_t head_size,
                        int64_t num_kv_head_replicas,
                        bool bias,
                        bool gather_output,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options,
                        const QuantArgs& quant_args = QuantArgs{});

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight().sizes() << " " << weight().device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }
  bool is_weight_loaded() const { return weight_is_loaded_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(bias);

  DEFINE_FUSED_WEIGHT(
      weight_scale_inv);  // Block-wise FP8 inverse-scale grid
                          // [ceil(N/bn), ceil(K/bk)] (DeepSeek-style).

  int64_t rank_;
  int64_t world_size_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_size_;
  int64_t num_kv_head_replicas_;
  // whether to gather the output
  bool gather_output_;
  torch::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  // quantization args
  QuantArgs quant_args_;
  std::optional<std::string> resolved_weight_quant_method_;
  bool block_fp8_resolved_unquantized_ = false;
  mutable MatmulOutputBuffers output_buffers_;
};
TORCH_MODULE(QKVParallelLinear);

// Linear layer with row parallelism.

//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelLinearImpl final : public torch::nn::Module {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        bool bias,
                        bool input_is_parallelized,
                        bool enable_result_reduction,
                        const QuantArgs& quant_args,
                        ProcessGroup* process_group,
                        const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input);

  torch::Tensor forward(torch::Tensor input, RowParallelReduceMode reduce_mode);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }
  ProcessGroup* process_group() const { return process_group_; }

 private:
  torch::Tensor forward_impl(torch::Tensor input,
                             RowParallelReduceMode reduce_mode);

  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);

  DEFINE_WEIGHT(
      weight_scale_inv);  // Block-wise FP8 inverse-scale grid
                          // [ceil(N/bn), ceil(K/bk)] (DeepSeek-style).

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // whether to reduce the results
  bool enable_result_reduction_;

  // parallel process group
  ProcessGroup* process_group_;

  int64_t rank_;
  int64_t world_size_;

  // quantization args
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  std::optional<std::string> resolved_weight_quant_method_;
  bool block_fp8_resolved_unquantized_ = false;
  mutable MatmulOutputBuffers output_buffers_;
};

TORCH_MODULE(RowParallelLinear);

class ReplicatedLinearImpl final : public torch::nn::Module {
 public:
  ReplicatedLinearImpl(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       const QuantArgs& quant_args,
                       const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }
  std::optional<torch::Tensor> bias() const;

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);

  QuantArgs quant_args_;
  torch::TensorOptions options_;
  std::optional<std::string> resolved_weight_quant_method_;

  mutable MatmulOutputBuffers output_buffers_;
};
TORCH_MODULE(ReplicatedLinear);

}  // namespace musa
}  // namespace layer
}  // namespace xllm
