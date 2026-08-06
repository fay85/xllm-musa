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

#include "layers/musa/linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/layers/common/quant_utils.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/musa/musa_ops_api.h"

namespace xllm {
namespace layer {
namespace musa {

std::optional<torch::Tensor> MatmulOutputBuffers::get(
    const torch::Tensor& input,
    const torch::Tensor& weight) {
  constexpr int64_t kMatmulOutputBufMaxRows = 256;
  if (input.dim() != 2 || weight.dim() != 2 || input.size(0) <= 0 ||
      input.size(0) > kMatmulOutputBufMaxRows) {
    return std::nullopt;
  }
  const int64_t rows = input.size(0);
  const int64_t columns = weight.size(0);
  const bool needs_realloc =
      !decode_output_buf_.defined() || decode_output_buf_.size(0) < rows ||
      decode_output_buf_.size(1) != columns ||
      decode_output_buf_.scalar_type() != input.scalar_type() ||
      decode_output_buf_.device() != input.device();
  if (needs_realloc) {
    decode_output_buf_ =
        torch::empty({kMatmulOutputBufMaxRows, columns}, input.options());
  }
  return decode_output_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/rows);
}

namespace {

torch::Tensor matmul_with_output_buffer(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    MatmulOutputBuffers& output_buffers) {
  return xllm::kernel::musa::matmul(
      input, weight, bias, output_buffers.get(input, weight));
}

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

bool is_block_fp8_quant(const QuantArgs& quant_args) {
  return quant_args.quant_method() == kQuantMethodFp8 &&
         quant_args.weight_block_size().size() == 2 &&
         quant_args.weight_block_size()[0] == 128 &&
         quant_args.weight_block_size()[1] == 128;
}

void check_quantization_supported(const QuantArgs& quant_args) {
  CHECK_NE(quant_args.quant_method(), kQuantMethodSmoothquant)
      << "MUSA linear does not support SmoothQuant.";
  if (quant_args.quant_method() == kQuantMethodFp8) {
    CHECK(is_block_fp8_quant(quant_args))
        << "MUSA linear supports only FP8 weight_block_size=[128, 128].";
  }
}

torch::Tensor block_fp8_matmul(const torch::Tensor& input,
                               const torch::Tensor& weight_fp8,
                               const torch::Tensor& weight_scale_inv,
                               const std::vector<int64_t>& weight_block_size,
                               const std::optional<torch::Tensor>& bias,
                               MatmulOutputBuffers& output_buffers) {
  CHECK_EQ(weight_block_size.size(), 2);
  CHECK_EQ(weight_block_size[0], 128);
  CHECK_EQ(weight_block_size[1], 128);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  const int64_t block_k = weight_block_size[1];
  std::vector<int64_t> input_shape = input.sizes().vec();
  const int64_t k = input.size(-1);
  CHECK_EQ(k % block_k, 0) << "native block-fp8 GEMM requires K % " << block_k
                           << " == 0, got K=" << k;
  torch::Tensor input_2d = input.reshape({-1, k}).contiguous();
  auto [a_fp8, a_scale] =
      xllm::kernel::musa::per_token_group_quant_fp8(input_2d, block_k);
  CHECK_EQ(weight_scale_inv.scalar_type(), torch::kFloat32);
  CHECK(weight_scale_inv.is_contiguous());
  const torch::ScalarType output_dtype = torch::kBFloat16;
  torch::Tensor output = xllm::kernel::musa::gemm_fp8_nt_groupwise(
      a_fp8,
      weight_fp8,
      a_scale,
      weight_scale_inv,
      output_dtype,
      output_buffers.get(input_2d, weight_fp8));
  if (bias.has_value() && bias.value().defined()) {
    output.add_(bias.value().to(output.scalar_type()));
  }
  input_shape.back() = weight_fp8.size(0);
  return output.reshape(input_shape);
}

bool is_w8a8_dynamic_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8_dynamic";
}

bool is_w8a8_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8";
}

void prepare_weight_for_linear_load(
    torch::nn::Module* module,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const std::optional<std::string>& resolved_weight_quant_method,
    torch::Tensor& weight_tensor,
    bool& weight_is_loaded) {
  CHECK(!is_w8a8_quant(resolved_weight_quant_method) &&
        !is_w8a8_dynamic_quant(resolved_weight_quant_method))
      << "MUSA linear does not support W8A8 checkpoints.";
  if (quant_args.quant_descs().empty() &&
      !quant_args.is_compressed_tensors_w8a8_dynamic()) {
    return;
  }

  CHECK(weight_tensor.defined())
      << "weight must be registered before lazy quant fallback";
  const int64_t out_features = weight_tensor.size(0);
  const int64_t in_features = weight_tensor.size(1);
  std::vector<weight::LazyParameterSpec> specs;
  specs.reserve(1);
  specs.emplace_back(weight::LazyParameterSpec{&weight_tensor,
                                               &weight_is_loaded,
                                               "weight",
                                               {out_features, in_features},
                                               options});
  weight::ensure_parameter_storage(module, specs);
}

}  // namespace

ColumnParallelLinearImpl::ColumnParallelLinearImpl(const ModelContext& context)
    : ColumnParallelLinearImpl(
          context.get_model_args().hidden_size(),
          context.get_model_args().vocab_size(),
          /*bias=*/false,
          /*gather_output=*/true,
          QuantArgs{},  // do not use quantization for lm_head
          context.get_parallel_args().lm_head_group_ != nullptr
              ? context.get_parallel_args().lm_head_group_
              : context.get_parallel_args().tp_group_,
          context.get_tensor_options()) {}

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    int32_t output_replicas)
    : gather_output_(gather_output),
      device_(options.device()),
      process_group_(process_group),
      quant_args_(quant_args),
      options_(options) {
  check_quantization_supported(quant_args_);
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  int32_t valid_output_replicas = output_replicas;
  if (valid_output_replicas <= 0 || world_size_ % valid_output_replicas != 0 ||
      (valid_output_replicas != 1 && gather_output)) {
    valid_output_replicas = 1;
  }
  weight_rank_ = rank_ / valid_output_replicas;
  weight_world_size_ = world_size_ / valid_output_replicas;
  CHECK(out_features % weight_world_size_ == 0)
      << "out_features " << out_features
      << " not divisible by weight_world_size " << weight_world_size_;
  const int64_t out_features_per_partition = out_features / weight_world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (is_block_fp8_quant(quant_args_)) {
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles =
        (out_features_per_partition + block_n - 1) / block_n;
    const int64_t k_tiles = (in_features + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  torch::Tensor output;

  if (is_block_fp8_quant(quant_args_)) {
    if (weight_.scalar_type() == torch::kFloat8_e4m3fn) {
      output = block_fp8_matmul(input,
                                weight_,
                                weight_scale_inv_,
                                quant_args_.weight_block_size(),
                                bias,
                                output_buffers_);
    } else {
      output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
    }
  } else {
    output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  if (is_block_fp8_quant(quant_args_)) {
    if (!block_fp8_resolved_unquantized_ && !weight_scale_inv_is_loaded_ &&
        state_dict.has("weight") && !state_dict.has("weight_scale_inv")) {
      block_fp8_resolved_unquantized_ = true;
      weight_.set_data(torch::empty(weight_.sizes(), options_));
      weight_is_loaded_ = false;
    }
    if (!block_fp8_resolved_unquantized_) {
      LOAD_SHARDED_WEIGHT(weight_scale_inv, 0);
    }
    LOAD_SHARDED_WEIGHT(weight, 0);
  } else {
    LOAD_SHARDED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  if (is_block_fp8_quant(quant_args_)) {
    if (!block_fp8_resolved_unquantized_ && !weight_scale_inv_is_loaded_) {
      for (const auto& prefix : prefixes) {
        if (state_dict.has(prefix + "weight") &&
            !state_dict.has(prefix + "weight_scale_inv")) {
          block_fp8_resolved_unquantized_ = true;
          break;
        }
      }
      if (block_fp8_resolved_unquantized_) {
        weight_.set_data(torch::empty(weight_.sizes(), options_));
        weight_is_loaded_ = false;
      }
    }
    if (!block_fp8_resolved_unquantized_) {
      LOAD_FUSED_WEIGHT(weight_scale_inv, 0);
    }
  }
  LOAD_FUSED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

// load_state_dict for merged weights with variable shard sizes
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  if (is_block_fp8_quant(quant_args_)) {
    if (!block_fp8_resolved_unquantized_ && !weight_scale_inv_is_loaded_ &&
        state_dict.has("weight") && !state_dict.has("weight_scale_inv")) {
      block_fp8_resolved_unquantized_ = true;
      weight_.set_data(torch::empty(weight_.sizes(), options_));
      weight_is_loaded_ = false;
    }
    if (!block_fp8_resolved_unquantized_) {
      LOAD_WEIGHT(weight_scale_inv);
      if (weight_scale_inv_is_loaded_) {
        CHECK_EQ(world_size, 1)
            << "block-fp8 merged-variable-shard scale loading requires TP=1";
      }
    }
    LOAD_MERGED_WEIGHT_V2(weight, 0);
  } else {
    LOAD_MERGED_WEIGHT_V2(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT_V2(bias, 0);
  }
}

QKVParallelLinearImpl::QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args) {
  check_quantization_supported(quant_args_);
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8: fused QKV FP8 weight [out,hidden] + BF16 inverse-scale
    // grid [ceil(out/bn), ceil(hidden/bk)].
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles =
        (out_features_per_partition + block_n - 1) / block_n;
    const int64_t k_tiles = (hidden_size + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (is_block_fp8_quant(quant_args_)) {
    if (weight_.scalar_type() == torch::kFloat8_e4m3fn) {
      output = block_fp8_matmul(input,
                                weight_,
                                weight_scale_inv_,
                                quant_args_.weight_block_size(),
                                bias,
                                output_buffers_);
    } else {
      output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
    }
  } else {
    output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

void QKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  if (is_block_fp8_quant(quant_args_) && !block_fp8_resolved_unquantized_ &&
      !weight_scale_inv_is_loaded_) {
    for (const auto& prefix : prefixes) {
      if (state_dict.has(prefix + "weight") &&
          !state_dict.has(prefix + "weight_scale_inv")) {
        block_fp8_resolved_unquantized_ = true;
        break;
      }
    }
    if (block_fp8_resolved_unquantized_) {
      weight_.set_data(torch::empty(weight_.sizes(), options_));
      weight_is_loaded_ = false;
    }
  }
  LOAD_QKV_WEIGHT(weight, 0, num_kv_head_replicas_);
  if (bias_.defined()) {
    LOAD_QKV_WEIGHT(bias, 0, num_kv_head_replicas_);
  }
  if (is_block_fp8_quant(quant_args_)) {
    if (!block_fp8_resolved_unquantized_) {
      LOAD_QKV_WEIGHT(weight_scale_inv, 0, num_kv_head_replicas_);
    }
  }
}

void QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int32_t shard_tensor_count = 3;
  const int64_t shard_size = num_heads_ * head_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);
  CHECK_EQ(num_heads_, num_kv_heads_);
  LOAD_MERGED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      options_(options),
      process_group_(process_group) {
  check_quantization_supported(quant_args_);
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (is_block_fp8_quant(quant_args_)) {
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles = (out_features + block_n - 1) / block_n;
    const int64_t k_tiles = (in_features_per_partition + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  const RowParallelReduceMode reduce_mode =
      enable_result_reduction_ ? RowParallelReduceMode::ALL_REDUCE
                               : RowParallelReduceMode::NONE;
  return forward_impl(input, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::forward(
    torch::Tensor input,
    RowParallelReduceMode reduce_mode) {
  const RowParallelReduceMode supported_mode =
      enable_result_reduction_ ? RowParallelReduceMode::ALL_REDUCE
                               : RowParallelReduceMode::NONE;
  CHECK(reduce_mode == supported_mode)
      << "MUSA row-parallel linear supports only NONE and ALL_REDUCE.";
  return forward_impl(input, reduce_mode);
}

torch::Tensor RowParallelLinearImpl::forward_impl(
    torch::Tensor input,
    RowParallelReduceMode reduce_mode) {
  auto bias = bias_.defined() && rank_ == 0
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;

  if (!input_is_parallelized_) {
    input = xllm::parallel_state::scatter(input, process_group_);
  }

  torch::Tensor output;
  if (is_block_fp8_quant(quant_args_)) {
    if (weight_.scalar_type() == torch::kFloat8_e4m3fn) {
      output = block_fp8_matmul(input,
                                weight_,
                                weight_scale_inv_,
                                quant_args_.weight_block_size(),
                                bias,
                                output_buffers_);
    } else {
      output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
    }
  } else {
    output = matmul_with_output_buffer(input, weight_, bias, output_buffers_);
  }

  if (reduce_mode == RowParallelReduceMode::NONE) {
    return output;
  }

  if (enable_result_reduction_ && world_size_ > 1) {
    output = xllm::parallel_state::reduce(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = world_size_ == 1 ? 0 : rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  if (is_block_fp8_quant(quant_args_)) {
    if (!block_fp8_resolved_unquantized_ && !weight_scale_inv_is_loaded_ &&
        state_dict.has("weight") && !state_dict.has("weight_scale_inv")) {
      block_fp8_resolved_unquantized_ = true;
      weight_.set_data(torch::empty(weight_.sizes(), options_));
      weight_is_loaded_ = false;
    }
    if (!block_fp8_resolved_unquantized_) {
      LOAD_SHARDED_WEIGHT(weight_scale_inv, 1);
    }
    LOAD_SHARDED_WEIGHT(weight, 1);
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Replicated linear layer.
ReplicatedLinearImpl::ReplicatedLinearImpl(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           const QuantArgs& quant_args,
                                           const torch::TensorOptions& options)
    : quant_args_(quant_args), options_(options) {
  check_quantization_supported(quant_args_);
  if (!quant_args_.quant_descs().empty() ||
      quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features}, options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features}, options),
                           /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  CHECK(!is_fp8_dtype(weight_.scalar_type()))
      << "MUSA replicated linear does not support per-tensor FP8 weights.";
  return matmul_with_output_buffer(input, weight_, bias, output_buffers_);
}

std::optional<torch::Tensor> ReplicatedLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  prepare_weight_for_linear_load(this,
                                 quant_args_,
                                 options_,
                                 resolved_weight_quant_method_,
                                 weight_,
                                 weight_is_loaded_);

  const torch::Tensor checkpoint_weight = state_dict.get_tensor("weight");
  CHECK(!checkpoint_weight.defined() ||
        !is_fp8_dtype(checkpoint_weight.scalar_type()))
      << "MUSA replicated linear does not support FP8 checkpoint weights.";
  LOAD_WEIGHT(weight);
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace musa
}  // namespace layer
}  // namespace xllm
