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

#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cctype>

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

// ============================================================================
// Persistent Matmul Output Buffer (CUDA-graph capture safety)
// ============================================================================
// On USE_CUDA + USE_MUSA, F::linear inside cuda::matmul calls at::empty
// to allocate the output tensor. During MUSA CUDA-graph capture, that
// allocation surfaces as
//   "MUSA error: operation not permitted when stream is capturing"
// because torch_musa's allocator does not honor c10::cuda::MemPoolContext set
// by xLLM's graph executor (unlike libtorch's c10 CUDA allocator that sglang
// relies on).
//
// We sidestep the problem by pre-allocating the matmul output once during the
// pre-capture warmup pass and re-using the same buffer in subsequent
// captured forwards via at::mm_out / at::addmm_out. This requires:
//   1. The buffer is allocated during the FIRST warmup forward (still on the
//      capture stream but BEFORE graph_.capture_begin()), then freed by the
//      caching allocator on the next stream sync, and re-acquired (same
//      address) for the captured forward.
//   2. Captures occur largest-bucket-first so the buffer never grows after a
//      smaller-bucket graph has been captured. For Qwen3.5-27B with
//      MAX_SEQS_PER_BATCH=4 the only decode bucket is 4, so this is
//      trivially satisfied. For configs with multiple decode buckets the
//      executor must warm up the max bucket first (sglang's
//      capture_one_batch_size does this implicitly via reverse order).
//
// Buffer sizing: we cap each layer at kMatmulOutputBufMaxRows rows. Decode
// bucket sizes coming out of get_bucket_num_tokens() top out at
// max_seqs_per_batch (<=128 in practice). Larger inputs (prefill, eager) skip
// the buffer and fall back to F::linear, which is safe outside graph capture.
constexpr int64_t kMatmulOutputBufMaxRows = 128;

inline void maybe_set_persistent_output_buf(
    xllm::kernel::MatmulParams& params,
    torch::Tensor& output_buf,
    const torch::Tensor& input,
    const torch::Tensor& weight) {
  if (input.dim() != 2 || weight.dim() != 2) {
    return;
  }
  const int64_t M = input.size(0);
  if (M <= 0 || M > kMatmulOutputBufMaxRows) {
    return;
  }
  const int64_t N = weight.size(0);
  const bool needs_realloc =
      !output_buf.defined() || output_buf.size(0) < M ||
      output_buf.size(1) != N ||
      output_buf.scalar_type() != input.scalar_type() ||
      output_buf.device() != input.device();
  if (needs_realloc) {
    // Grow-only: never shrink, so views handed out for already-captured
    // smaller-bucket graphs stay valid.
    const int64_t target_M =
        output_buf.defined() ? std::max(M, output_buf.size(0)) : M;
    output_buf = torch::empty({target_M, N}, input.options());
  }
  params.output_buf = output_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/M);
}

// ============================================================================
// FP8 Fused Weight Utilities
// ============================================================================
// Unlike INT8/SmoothQuant (per-channel), FP8 usually uses per-tensor scaling.
// When fusing separate layers (e.g., gate_proj + up_proj) into one, we cannot
// simply concatenate them if they have different scaling factors. We must
// requantize all partitions to align with a single global max_scale.

struct Fp8PartitionInfo {
  std::vector<float> scales;
  std::vector<int64_t> logical_widths;

  bool empty() const { return scales.empty(); }
  size_t size() const { return scales.size(); }
};

inline float compute_max_scale(const std::vector<float>& scales) {
  if (scales.empty()) {
    return 1.0f;
  }
  return *std::max_element(scales.begin(), scales.end());
}

// Detect if the checkpoint contains valid separate scales for each partition.
// The check on the last element serves as a heuristic to ensure the scales
// are fully populated and not just initialized to a sentinel/minimum value.
inline bool is_unfused_checkpoint(const std::vector<float>& scales) {
  return scales.size() > 1 &&
         scales.back() > std::numeric_limits<float>::lowest();
}

// Realigns FP8 partitions to a unified global scale to enable fusion.
// Logic:
// 1. Recover original values (FP8 -> FP16) using partition-specific scales.
// 2. Re-quantize (FP16 -> FP8) using the new global max_scale.
void requantize_fp8_weight(torch::Tensor& weight,
                           const std::vector<float>& partition_scales,
                           const std::vector<int64_t>& logical_widths,
                           float max_scale) {
  if (partition_scales.size() != logical_widths.size()) {
    return;
  }

  int64_t start = 0;
  for (size_t idx = 0; idx < logical_widths.size(); ++idx) {
    int64_t logical_width = logical_widths[idx];
    if (logical_width == 0) {
      continue;
    }
    int64_t end = start + logical_width;

    // Dequantize: FP8 -> FP16 with original scale
    auto weight_slice = weight.slice(0, start, end);
    auto weight_fp16 = weight_slice.to(torch::kFloat16) * partition_scales[idx];

    // Requantize: FP16 -> FP8 with unified max_scale
    auto scale_tensor = torch::tensor(
        {max_scale}, weight_fp16.options().dtype(torch::kFloat32));
    auto weight_quantized =
        torch::empty_like(weight_slice, torch::kFloat8_e4m3fn);

    xllm::kernel::StaticScaledFp8QuantParams quant_params;
    quant_params.output = weight_quantized;
    quant_params.input = weight_fp16;
    quant_params.scale = scale_tensor;
    xllm::kernel::static_scaled_fp8_quant(quant_params);

    weight.slice(0, start, end).copy_(weight_quantized);
    start = end;
  }
}

// Load max input scale from multiple prefixes
torch::Tensor load_max_input_scale(const StateDict& state_dict,
                                   const std::vector<std::string>& prefixes) {
  torch::Tensor max_scale;
  for (const auto& prefix : prefixes) {
    auto scale_tensor = state_dict.get_tensor(prefix + "input_scale");
    if (scale_tensor.defined()) {
      auto scale_val = scale_tensor.flatten().max();
      if (!max_scale.defined()) {
        max_scale = scale_val;
      } else {
        max_scale = torch::max(max_scale, scale_val);
      }
    }
  }
  return max_scale;
}

// ============================================================================
// FP8 Forward Helper
// ============================================================================
// Performs FP8 W8A8 quantized linear: input quantization + scaled matmul.
// Consolidates repeated logic from Column/QKV/RowParallelLinear forward paths.
//
// Performance Optimization:
// - If input is already FP8 (from fused RMSNorm+FP8 quantization), skip
//   quantization step and use input_scale directly. This avoids redundant
//   memory reads/writes.
// - For non-FP8 inputs, quantization is performed based on input_scale:
//   - input_scale provided: static quantization (faster, no absmax compute)
//   - input_scale not provided: dynamic quantization (computes absmax)

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  // Flatten input to 2D for matmul
  auto input_2d = input.view({-1, input.size(-1)});

  torch::Tensor quantized_input;
  torch::Tensor a_scale;

  // Check if input is already FP8 quantized (from fused RMSNorm+FP8)
  if (input.dtype() == torch::kFloat8_e4m3fn) {
    // Input is already FP8, use directly (skip quantization)
    // This is the fast path when using fused RMSNorm+FP8 quantization
    CHECK(input_scale.has_value())
        << "input_scale must be provided when input is already FP8";
    quantized_input = input_2d;
    a_scale = input_scale.value();
  } else {
    // Input is not FP8, perform quantization
    // (static if input_scale provided, dynamic otherwise)
    xllm::kernel::Fp8ScaledQuantizeParams quantize_params;
    quantize_params.input = input_2d;
    quantize_params.output = std::nullopt;
    quantize_params.scale = input_scale;

    std::tie(quantized_input, a_scale) =
        xllm::kernel::fp8_scaled_quantize(quantize_params);
  }

  // FP8 scaled matmul
  xllm::kernel::Fp8ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = a_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.bias = bias;
  matmul_params.output = std::nullopt;
  matmul_params.output_dtype = output_dtype;
  matmul_params.input_shape = input.sizes().vec();

  return xllm::kernel::fp8_scaled_matmul(matmul_params);
}

std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

// DeepSeek-style block-wise FP8 (W8A8, weight_block_size e.g. [128,128]): the
// weight is stored as float8_e4m3fn [N,K] with a sibling weight_scale_inv
// [ceil(N/bn), ceil(K/bk)] (one inverse scale per bn x bk block). Distinguished
// from per-tensor FP8 by a non-empty weight_block_size.
bool is_block_fp8_quant(const QuantArgs& quant_args) {
  return quant_args.quant_method() == kQuantMethodFp8 &&
         quant_args.weight_block_size().size() == 2 &&
         quant_args.weight_block_size()[0] > 0 &&
         quant_args.weight_block_size()[1] > 0;
}

// Dequantize a block-wise FP8 weight to BF16 using its inverse-scale grid.
// Mirrors the DeepSeek/minimax reference: w_bf16[n,k] = w_fp8[n,k] *
// scale_inv[n/bn, k/bk]. block_n/block_k default [128,128].
torch::Tensor dequantize_fp8_block_weight(const torch::Tensor& fp8_weight,
                                          const torch::Tensor& weight_scale_inv,
                                          int64_t block_n,
                                          int64_t block_k) {
  CHECK_EQ(fp8_weight.dim(), 2)
      << "block-fp8 weight must be 2D, got " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.dim(), 2)
      << "block-fp8 weight_scale_inv must be 2D, got "
      << weight_scale_inv.sizes();
  const int64_t n = fp8_weight.size(0);
  const int64_t k = fp8_weight.size(1);
  const int64_t n_tiles = (n + block_n - 1) / block_n;
  const int64_t k_tiles = (k + block_k - 1) / block_k;
  CHECK_EQ(weight_scale_inv.size(0), n_tiles)
      << "block-fp8 scale rows " << weight_scale_inv.sizes()
      << " mismatch weight " << fp8_weight.sizes();
  CHECK_EQ(weight_scale_inv.size(1), k_tiles)
      << "block-fp8 scale cols " << weight_scale_inv.sizes()
      << " mismatch weight " << fp8_weight.sizes();

  if (n % block_n == 0 && k % block_k == 0) {
    auto w = fp8_weight.to(torch::kBFloat16)
                 .reshape({n_tiles, block_n, k_tiles, block_k});
    auto s = weight_scale_inv.to(torch::kBFloat16)
                 .reshape({n_tiles, 1, k_tiles, 1});
    return (w * s).reshape({n, k});
  }
  auto expanded = weight_scale_inv.repeat_interleave(block_n, /*dim=*/0)
                      .repeat_interleave(block_k, /*dim=*/1)
                      .slice(/*dim=*/0, /*start=*/0, /*end=*/n)
                      .slice(/*dim=*/1, /*start=*/0, /*end=*/k)
                      .to(torch::kBFloat16);
  return fp8_weight.to(torch::kBFloat16) * expanded;
}

// Step-1 (semi-FP8) block-fp8 linear: dequantize the FP8 weight to BF16 on the
// fly and run the standard BF16 matmul. Keeps weights FP8 in device memory
// (memory win) while computing in BF16 (correctness gate before native FP8).
torch::Tensor block_fp8_dequant_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& weight_scale_inv,
    const std::vector<int64_t>& weight_block_size,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buf) {
  const int64_t block_n = weight_block_size[0];
  const int64_t block_k = weight_block_size[1];
  auto weight_bf16 = dequantize_fp8_block_weight(
      weight_fp8, weight_scale_inv, block_n, block_k);
  if (std::getenv("XLLM_DEBUG_FP8") != nullptr) {
    static std::atomic<int> dbg_count{0};
    int idx = dbg_count.fetch_add(1);
    if (idx < 24) {
      auto sf = weight_scale_inv.to(torch::kFloat32);
      auto wf = weight_bf16.to(torch::kFloat32);
      LOG(INFO) << "[FP8 dequant #" << idx << "] w=" << weight_fp8.sizes() << " "
                << weight_fp8.scalar_type() << ", scale=" << weight_scale_inv.sizes()
                << " " << weight_scale_inv.scalar_type() << " defined="
                << weight_scale_inv.defined() << ", scale[min="
                << sf.min().item<float>() << ",max=" << sf.max().item<float>()
                << ",mean=" << sf.mean().item<float>() << "], dequant[min="
                << wf.min().item<float>() << ",max=" << wf.max().item<float>()
                << ",std=" << wf.std().item<float>() << "]";
    }
  }
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_bf16;
  matmul_params.bias = bias;
  maybe_set_persistent_output_buf(matmul_params, output_buf, input, weight_bf16);
  return xllm::kernel::matmul(matmul_params);
}

// Step-2 (native) block-fp8 linear: dynamically quantize the activation to FP8
// per token-group (group = block_k along K) and run the mate/muDNN block-wise
// FP8 GEMM directly against the FP8 weight. No BF16 weight is ever
// materialized, so per-token memory traffic drops from ~135 GB (dequant path)
// to ~27 GB for Qwen3.5-27B -- roughly the FP8 memory-bandwidth roofline.
torch::Tensor block_fp8_native_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& weight_scale_inv,
    const std::vector<int64_t>& weight_block_size,
    const std::optional<torch::Tensor>& bias) {
  const int64_t block_k = weight_block_size[1];

  auto in_shape = input.sizes().vec();
  const int64_t k = input.size(-1);
  CHECK_EQ(k % block_k, 0) << "native block-fp8 GEMM requires K % " << block_k
                           << " == 0, got K=" << k;

  const auto input_2d = input.reshape({-1, k}).contiguous();

  // Fused per-token-group dynamic FP8 activation quantization (absmax / 448) via
  // the MUSA-native kernel: one pass produces a_fp8 [M,K] e4m3 and a_scale
  // [M, K/128] fp32 (K-major), matching the mate groupwise GEMM (1,128,128).
  auto [a_fp8, a_scale] =
      xllm::kernel::per_token_group_quant_fp8(input_2d, block_k);

  xllm::kernel::Fp8BlockMatmulParams params;
  params.a = a_fp8;
  params.b = weight_fp8;
  params.a_scale = a_scale;
  CHECK_EQ(weight_scale_inv.scalar_type(), torch::kFloat32)
      << "native block-fp8 GEMM requires FP32 weight scales";
  CHECK(weight_scale_inv.is_contiguous())
      << "native block-fp8 GEMM requires contiguous weight scales";
  params.b_scale = weight_scale_inv;
  params.output_dtype = (input.scalar_type() == torch::kFloat16)
                            ? torch::kFloat16
                            : torch::kBFloat16;
  auto out = xllm::kernel::fp8_block_matmul(params);  // [m, n]

  if (bias.has_value() && bias.value().defined()) {
    out = out + bias.value().to(out.scalar_type());
  }

  in_shape.back() = weight_fp8.size(0);
  return out.reshape(in_shape);
}

// Dispatch the block-fp8 linear forward. Native mate/muDNN groupwise GEMM is the
// default; set XLLM_FP8_DEQUANT=1 to fall back to the semi-FP8 dequant-to-BF16
// path (slower; kept for debugging / numerical bisection only).
torch::Tensor block_fp8_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& weight_scale_inv,
    const std::vector<int64_t>& weight_block_size,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buf) {
  static const bool use_dequant = std::getenv("XLLM_FP8_DEQUANT") != nullptr;
  if (use_dequant) {
    return block_fp8_dequant_forward(input,
                                     weight_fp8,
                                     weight_scale_inv,
                                     weight_block_size,
                                     bias,
                                     output_buf);
  }
  return block_fp8_native_forward(
      input, weight_fp8, weight_scale_inv, weight_block_size, bias);
}

void resolve_weight_quant_method_for_linear_load(
    const QuantArgs& quant_args,
    const StateDict& state_dict,
    const std::vector<std::string>* local_prefixes,
    std::optional<std::string>& resolved_weight_quant_method) {
  const auto prefixes =
      local_prefixes == nullptr ? std::vector<std::string>{} : *local_prefixes;
  auto resolved =
      quant_args.get_quant_method_from_prefixes(state_dict, prefixes);
  if (resolved.has_value()) {
    resolved_weight_quant_method = to_lower_copy(resolved.value());
    return;
  }
  if (quant_args.is_compressed_tensors_w8a8_dynamic()) {
    bool is_w8a8_dynamic = false;
    if (prefixes.empty()) {
      torch::Tensor weight = state_dict.get_tensor("weight");
      is_w8a8_dynamic = state_dict.has("weight_scale") && weight.defined() &&
                        weight.scalar_type() == torch::kInt8;
    } else {
      is_w8a8_dynamic = true;
      for (const std::string& prefix : prefixes) {
        torch::Tensor weight = state_dict.get_tensor(prefix + "weight");
        if (!state_dict.has(prefix + "weight_scale") || !weight.defined() ||
            weight.scalar_type() != torch::kInt8) {
          is_w8a8_dynamic = false;
          break;
        }
      }
    }
    if (is_w8a8_dynamic) {
      resolved_weight_quant_method = "w8a8_dynamic";
      return;
    }
  }
  if (!quant_args.quant_descs().empty()) {
    LOG(WARNING) << "[LinearLoad][QuantMethod] quant_descs is not empty but "
                    "quant method was not resolved from state_dict prefixes. "
                    "state_dict.prefix="
                 << state_dict.prefix();
  }
  resolved_weight_quant_method = std::nullopt;
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

torch::Dtype get_w8a8_deq_scale_dtype(const torch::TensorOptions& options) {
  const torch::Dtype dtype = c10::typeMetaToScalarType(options.dtype());
  if (dtype == torch::kFloat16) {
    return torch::kInt64;
  }
  if (dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  LOG(WARNING) << "W8A8 deq_scale defaults to float32 for dtype " << dtype;
  return torch::kFloat32;
}

struct W8A8LinearParamRefs {
  torch::Tensor& weight;
  bool& weight_is_loaded;
  torch::Tensor& input_scale;
  bool& input_scale_is_loaded;
  torch::Tensor& input_offset;
  bool& input_offset_is_loaded;
  torch::Tensor& deq_scale;
  bool& deq_scale_is_loaded;
  torch::Tensor& quant_bias;
  bool& quant_bias_is_loaded;
  torch::Tensor& weight_scale;
  bool& weight_scale_is_loaded;
  torch::Tensor& weight_offset;
  bool& weight_offset_is_loaded;
};

void ensure_w8a8_params_for_linear_load(
    torch::nn::Module* module,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const std::optional<std::string>& resolved_weight_quant_method,
    int64_t shared_input_param_size,
    W8A8LinearParamRefs refs) {
  std::vector<weight::LazyParameterSpec> specs;
  auto push = [&](torch::Tensor& tensor,
                  bool& tensor_is_loaded,
                  const char* name,
                  std::vector<int64_t> sizes,
                  const torch::TensorOptions& tensor_options) {
    specs.push_back(weight::LazyParameterSpec{
        &tensor, &tensor_is_loaded, name, std::move(sizes), tensor_options});
  };

  if (!is_w8a8_quant(resolved_weight_quant_method) &&
      !is_w8a8_dynamic_quant(resolved_weight_quant_method)) {
    if (!quant_args.quant_descs().empty() ||
        quant_args.is_compressed_tensors_w8a8_dynamic()) {
      // Quant args indicated a checkpoint that may be quantized, so the
      // constructor initialized weights as kInt8. If the actual checkpoint is
      // not resolved to a W8A8 method, re-register the weight in the original
      // dtype so the subsequent load can copy checkpoint weights correctly.
      CHECK(refs.weight.defined())
          << "weight must be registered before lazy quant fallback";
      const int64_t out_features = refs.weight.size(0);
      const int64_t in_features = refs.weight.size(1);
      specs.reserve(1);
      push(refs.weight,
           refs.weight_is_loaded,
           "weight",
           {out_features, in_features},
           options);
      weight::ensure_parameter_storage(module, specs);
    }
    return;
  }

  CHECK(refs.weight.defined())
      << "weight must be registered before lazy quant init";
  const int64_t out_features = refs.weight.size(0);
  const int64_t in_features = refs.weight.size(1);

  specs.reserve(4);
  if (is_w8a8_quant(resolved_weight_quant_method)) {
    push(refs.input_scale,
         refs.input_scale_is_loaded,
         "input_scale",
         {shared_input_param_size},
         options.dtype(torch::kFloat32));
    push(refs.input_offset,
         refs.input_offset_is_loaded,
         "input_offset",
         {shared_input_param_size},
         options.dtype(torch::kInt8));
    push(refs.deq_scale,
         refs.deq_scale_is_loaded,
         "deq_scale",
         {out_features},
         options.dtype(get_w8a8_deq_scale_dtype(options)));
    push(refs.quant_bias,
         refs.quant_bias_is_loaded,
         "quant_bias",
         {out_features},
         options.dtype(torch::kInt32));
  } else {
    push(refs.weight_scale,
         refs.weight_scale_is_loaded,
         "weight_scale",
         {out_features},
         options.dtype(torch::kFloat32));
    push(refs.weight_offset,
         refs.weight_offset_is_loaded,
         "weight_offset",
         {out_features},
         options.dtype(torch::kFloat32));
  }
  weight::ensure_parameter_storage(module, specs);
}

bool tensors_allclose_as_fp32(const torch::Tensor& lhs,
                              const torch::Tensor& rhs) {
  return torch::allclose(lhs.to(torch::kFloat32), rhs.to(torch::kFloat32));
}

bool load_shared_tensor_from_prefixes_or_fail(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::string& name,
    torch::Tensor& tensor,
    bool& tensor_is_loaded) {
  // W8A8 fused input_scale/offset shoul be same
  if (tensor_is_loaded || !tensor.defined()) {
    return tensor_is_loaded;
  }
  torch::Tensor first_candidate;
  std::string first_prefix;
  for (const auto& prefix : prefixes) {
    auto candidate = state_dict.get_tensor(prefix + name);
    if (!candidate.defined()) {
      continue;
    }
    auto flattened = candidate.flatten();
    if (!first_candidate.defined()) {
      first_candidate = flattened;
      first_prefix = prefix;
      continue;
    }
    CHECK_EQ(flattened.sizes(), first_candidate.sizes())
        << "Shared tensor size for " << name << ": prefix '" << prefix
        << "' has shape " << flattened.sizes() << ", but prefix '"
        << first_prefix << "' has shape " << first_candidate.sizes() << ".";
    CHECK(tensors_allclose_as_fp32(flattened, first_candidate))
        << "Shared tensor value for " << name << ": prefix '" << prefix
        << "' differs from prefix '" << first_prefix << "'.";
  }
  if (!first_candidate.defined()) {
    return false;
  }
  CHECK_EQ(first_candidate.numel(), tensor.numel())
      << "Tensor size mismatch for shared: " << state_dict.prefix() << name;
  tensor.copy_(first_candidate.view(tensor.sizes()));
  tensor_is_loaded = true;
  return true;
}

void collapse_shared_tensor_to_scalar_or_fail(torch::Tensor& tensor,
                                              const char* name) {
  // W8A8 fused input_scale/offset shoul be same
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_GT(tensor.numel(), 0) << name << " must contain at least one element.";
  if (tensor.numel() <= 1) {
    return;
  }
  auto flattened = tensor.flatten();
  auto first = flattened.slice(0, 0, 1).expand_as(flattened);
  CHECK(tensors_allclose_as_fp32(flattened, first))
      << "Shared tensor value for " << name
      << " in fused static W8A8 should be same.";
  tensor = tensor.flatten().slice(0, 0, 1);
}

torch::Tensor npu_w8a8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& input_scale,
    const torch::Tensor& input_offset,
    const torch::Tensor& deq_scale,
    const std::optional<torch::Tensor>& quant_bias,
    at::ScalarType output_dtype) {
  xllm::kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  quant_params.scale = input_scale;
  quant_params.zero_point = input_offset;
  // quant_params.output_dtype = at::ScalarType::QInt8;
  quant_params.axis = -1;

  auto quantized_input = xllm::kernel::quantize(quant_params);

  xllm::kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = deq_scale;
  quant_matmul_params.bias = quant_bias;
  quant_matmul_params.output_dtype = output_dtype;

  return xllm::kernel::quant_matmul(quant_matmul_params);
}

torch::Tensor npu_w8a8_dynamic_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  xllm::kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  // quant_params.dst_type = at::kChar;

  torch::Tensor quantized_input;
  std::optional<torch::Tensor> pertoken_scale;
  std::tie(quantized_input, pertoken_scale) =
      xllm::kernel::dynamic_quant(quant_params);
  CHECK(pertoken_scale.has_value() && pertoken_scale->defined())
      << "dynamic_quant must return per-token scale for w8a8_dynamic.";

  xllm::kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = weight_scale;
  quant_matmul_params.pertoken_scale = pertoken_scale;
  quant_matmul_params.output_dtype = output_dtype;
  if (bias.has_value() && bias->defined()) {
    quant_matmul_params.bias = bias;
  }
  auto output = xllm::kernel::quant_matmul(quant_matmul_params);
  return output;
}

#if defined(USE_DCU)
torch::Tensor dcu_w8a8_dynamic_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  xllm::kernel::ScaledQuantizeParams quantize_params;
  quantize_params.x = input;
  quantize_params.smooth = torch::Tensor();  // no smooth factor

  torch::Tensor quantized_input;
  torch::Tensor input_scale;
  std::tie(quantized_input, input_scale) =
      xllm::kernel::scaled_quantize(quantize_params);

  xllm::kernel::ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = input_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.output_dtype = output_dtype;
  matmul_params.bias = bias;
  matmul_params.beta = 0.0;
  matmul_params.a_quant_bit_size = 8;

  return xllm::kernel::scaled_matmul(matmul_params);
}
#endif  // USE_DCU

}  // namespace

ColumnParallelLinearImpl::ColumnParallelLinearImpl(const ModelContext& context)
    : ColumnParallelLinearImpl(
          context.get_model_args().hidden_size(),
          context.get_model_args().vocab_size(),
          /*bias=*/false,
          /*gather_output=*/true,
          QuantArgs{},  // do not use quantization for lm_head
          context.get_parallel_args().tp_group_,
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
    const LinearExtraArgs& linear_extra_args,
    int32_t output_replicas)
    : gather_output_(gather_output),
      device_(options.device()),
      process_group_(process_group),
      quant_args_(quant_args),
      options_(options),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
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
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({in_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8 (DeepSeek-style): FP8 weight [N,K] + FP32 inverse-scale
    // grid [ceil(N/bn), ceil(K/bk)]. Checkpoint BF16 scales are converted once
    // while loading so each forward can pass the GEMM-ready scale directly.
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles = (out_features_per_partition + block_n - 1) / block_n;
    const int64_t k_tiles = (in_features + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale is per-tensor (scalar)
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({1}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
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

  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (is_block_fp8_quant(quant_args_)) {
    if (weight_.scalar_type() == torch::kFloat8_e4m3fn) {
      // Native block-FP8 GEMM (per-token-group activation quant + mate/muDNN
      // groupwise matmul); XLLM_FP8_DEQUANT=1 forces the slower
      // dequant-to-BF16 fallback.
      output = block_fp8_forward(input,
                                 weight_,
                                 weight_scale_inv_,
                                 quant_args_.weight_block_size(),
                                 bias,
                                 output_buf_);
    } else {
      // Module not actually quantized (no weight_scale_inv in checkpoint):
      // weight was re-registered to BF16 at load; run the standard matmul.
      xllm::kernel::MatmulParams matmul_params;
      matmul_params.a = input;
      matmul_params.b = weight_;
      matmul_params.bias = bias;
      maybe_set_persistent_output_buf(
          matmul_params, output_buf_, input, weight_);
      output = xllm::kernel::matmul(matmul_params);
    }
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    CHECK(!quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    maybe_set_persistent_output_buf(matmul_params, output_buf_, input, weight_);
    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

bool ColumnParallelLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ColumnParallelLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);
    // for input, there is one smooth value
    LOAD_WEIGHT(smooth);
  } else if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8: FP8 weight + BF16 inverse-scale grid. Column parallel
    // shards output (dim 0) for both the weight and the N-block scale grid.
    // See QKVParallelLinearImpl::load_state_dict for why this must be a
    // sticky, per-shard-presence decision rather than a per-call "scale not
    // loaded yet" check (this single-prefix module isn't fused across
    // siblings, but weight_scale_inv_is_loaded_ can still read false on an
    // unrelated call, e.g. one only carrying this module's bias).
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
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    // Find the first available "smooth" tensor in prefixes (e.g.,
    // "gate.smooth", "up_proj.smooth", etc.)
    for (const auto& prefix : prefixes) {
      auto smooth_tensor_candidate = state_dict.get_tensor(prefix + "smooth");
      if (smooth_tensor_candidate.defined()) {
        // Copy the found smooth tensor to the module parameter
        CHECK_EQ(smooth_.sizes(), smooth_tensor_candidate.sizes())
            << "smooth weight size mismatch for " << state_dict.prefix()
            << "smooth";
        smooth_.copy_(smooth_tensor_candidate);
        smooth_is_loaded_ = true;
        break;
      }
    }
    LOAD_FUSED_WEIGHT(qweight, 0);
    LOAD_FUSED_WEIGHT(per_channel_scale, 0);
  } else if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8 fused (e.g. gate_proj+up_proj): concatenate the FP8
    // weights and their N-block inverse-scale grids along dim 0. Each partition
    // N is a multiple of block_n, so the grids concatenate cleanly.
    // See QKVParallelLinearImpl::load_state_dict for why the quantized-vs-BF16
    // decision must be sticky and based on per-shard weight/scale
    // co-presence in a single call, not on "the fused accumulator hasn't
    // finished collecting all sibling shards yet" (siblings may arrive from
    // different shard files).
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
    LOAD_FUSED_WEIGHT(weight, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 fused layer loading: each partition may have its own per-tensor scale
    // (unfused checkpoint). We must requantize all partitions with max_scale.

    // Step 1: Collect partition info BEFORE LOAD_FUSED_WEIGHT (clears list)
    Fp8PartitionInfo partition_info;
    if (!weight_scale_is_loaded_) {
      for (const auto& prefix : prefixes) {
        auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
        if (scale_tensor.defined()) {
          partition_info.scales.push_back(scale_tensor.flatten().item<float>());
        }
        auto weight_tensor = state_dict.get_sharded_tensor(
            prefix + "weight", 0, rank, world_size);
        if (weight_tensor.defined()) {
          partition_info.logical_widths.push_back(weight_tensor.size(0));
        }
      }
    }

    // Step 2: Load fused weight
    LOAD_FUSED_WEIGHT(weight, 0);

    // Step 3: Requantize if needed (unfused checkpoint case)
    if (!weight_scale_is_loaded_ && !partition_info.empty()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales) && weight_.defined() &&
          partition_info.logical_widths.size() ==
              partition_info.scales.size()) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }

      weight_scale_.fill_(max_scale);
      weight_scale_is_loaded_ = true;
    }

    // Step 4: Load input_scale for static activation quantization
    if (!quant_args_.activation_dynamic() && input_scale_.defined() &&
        !input_scale_is_loaded_) {
      auto max_input_scale = load_max_input_scale(state_dict, prefixes);
      if (max_input_scale.defined()) {
        input_scale_.copy_(max_input_scale.view({1}));
        input_scale_is_loaded_ = true;
      }
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    // Fused static W8A8 quantizes the shared input only once, so keep a single
    // input_scale/input_offset slot and pull the first available tensor.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_FUSED_WEIGHT(deq_scale, 0);
    LOAD_FUSED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    LOAD_FUSED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_FUSED_WEIGHT(weight_offset, 0);
    }
  } else {
    LOAD_FUSED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    return input_scale_;
  }
  return std::nullopt;
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0 with variable shard sizes
  if (quant_args_.quant_method() == "smoothquant") {
    // For smoothquant, load quantized weights with variable shard sizes
    LOAD_MERGED_WEIGHT_V2(qweight, 0);
    LOAD_MERGED_WEIGHT_V2(per_channel_scale, 0);
  } else {
    if (is_w8a8_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_WEIGHT(input_scale);
      LOAD_WEIGHT(input_offset);
      LOAD_MERGED_WEIGHT_V2(deq_scale, 0);
      LOAD_MERGED_WEIGHT_V2(quant_bias, 0);
    } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_MERGED_WEIGHT_V2(weight_scale, 0);
      if (weight_offset_.defined()) {
        LOAD_MERGED_WEIGHT_V2(weight_offset, 0);
      }
    } else if (is_block_fp8_quant(quant_args_)) {
      // Merged-variable-shard block-fp8 path is used by GDN in_proj_qkv
      // (quantized: fused FP8 weight + fused block scale) and conv1d (NOT
      // quantized: plain BF16). The checkpoint stores in_proj_qkv already
      // fused, so its block scale grid is a single tensor (LOAD_WEIGHT). At
      // TP=1 it loads whole; TP>1 would need the scale grid split along the
      // N-block dim.
      // See QKVParallelLinearImpl::load_state_dict for why this decision
      // must be sticky and based on per-call weight/scale co-presence.
      if (!block_fp8_resolved_unquantized_ && !weight_scale_inv_is_loaded_ &&
          state_dict.has("weight") && !state_dict.has("weight_scale_inv")) {
        block_fp8_resolved_unquantized_ = true;
        weight_.set_data(torch::empty(weight_.sizes(), options_));
        weight_is_loaded_ = false;
      }
      if (!block_fp8_resolved_unquantized_) {
        LOAD_WEIGHT(weight_scale_inv);
        if (weight_scale_inv_is_loaded_) {
          CHECK_EQ(world_size, 1) << "block-fp8 merged-variable-shard TP>1 "
                                     "scale sharding is a TODO";
        }
      }
      LOAD_MERGED_WEIGHT_V2(weight, 0);
    } else {
      // For regular weights, use the new merged weight loading with variable
      // shard sizes
      LOAD_MERGED_WEIGHT_V2(weight, 0);
    }
  }

  if (bias_.defined()) {
    // For bias, we might need to handle it differently based on the use case
    // For now, we'll use the same approach if bias is also sharded
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
    const QuantArgs& quant_args,
    const LinearExtraArgs& linear_extra_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  (void)linear_extra_args;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8: fused QKV FP8 weight [out,hidden] + FP32 inverse-scale
    // grid [ceil(out/bn), ceil(hidden/bk)]. Checkpoint BF16 scales are
    // converted once while loading.
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles = (out_features_per_partition + block_n - 1) / block_n;
    const int64_t k_tiles = (hidden_size + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale: create {3} for Q/K/V, will use max() after loading
    // load separate scales then merge with max
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({3}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    // Also create {3} for Q/K/V, will use max() after loading
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({3}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
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
      // Native block-FP8 GEMM on the QKV weight (per-token-group activation
      // quant + mate/muDNN groupwise matmul); XLLM_FP8_DEQUANT=1 forces the
      // slower dequant-to-BF16 fallback.
      output = block_fp8_forward(input,
                                 weight_,
                                 weight_scale_inv_,
                                 quant_args_.weight_block_size(),
                                 bias,
                                 output_buf_);
    } else {
      xllm::kernel::MatmulParams matmul_params;
      matmul_params.a = input;
      matmul_params.b = weight_;
      matmul_params.bias = bias;
      maybe_set_persistent_output_buf(
          matmul_params, output_buf_, input, weight_);
      output = xllm::kernel::matmul(matmul_params);
    }
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization
    CHECK(!quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    // Use max of Q/K/V scales as unified scale for fused projection
    // Note: weight_scale_ and input_scale_ are already scalar tensors
    // (replaced with max values in load_state_dict)
    auto a_scale = input_scale_.defined()
                       ? std::optional<torch::Tensor>(input_scale_)
                       : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, a_scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    maybe_set_persistent_output_buf(matmul_params, output_buf_, input, weight_);

    output = xllm::kernel::matmul(matmul_params);
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});
  // Block-wise FP8: the checkpoint may deliver the Q/K/V shards for this
  // fused module across *different* safetensors files (e.g. q_proj in one
  // shard file, k_proj/v_proj in another). weight_/weight_scale_inv_ are
  // accumulated incrementally across load_state_dict calls (see
  // load_fused_weight), so we must decide "is this module actually
  // quantized in this checkpoint" without waiting for all 3 shards to be
  // present simultaneously in one call -- otherwise we'd wrongly conclude
  // "not quantized yet" while e.g. Q's shard is legitimately still pending
  // and prematurely retype weight_ to BF16, corrupting the accumulator: a
  // later call that completes accumulation would then numerically cast the
  // still-quantized (unscaled) FP8 shards into BF16 instead of dequantizing
  // them, producing garbage attention.
  //
  // Instead: each individual shard's own weight_scale_inv (if it exists)
  // always lives in the same shard file as that shard's own weight tensor
  // (HF safetensors never splits a single named tensor's data from a
  // sibling scale it doesn't own). So the first time we observe *any*
  // prefix's weight present without its own co-located scale, this fused
  // module is definitively unquantized in this checkpoint -- decide that
  // once, stickily, and only then retype/reload as BF16.
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
      // Fuse the Q/K/V N-block inverse-scale grids along dim 0 (same
      // KV-replica handling as the weight); accumulates across shard files
      // exactly like the weight above.
      LOAD_QKV_WEIGHT(weight_scale_inv, 0, num_kv_head_replicas_);
    }
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // Build partition info for Q/K/V
    Fp8PartitionInfo partition_info;
    int64_t num_heads_per_partition = num_heads_ / world_size_;
    int64_t num_kv_heads_per_partition = num_kv_heads_ / world_size_;
    partition_info.logical_widths = {
        num_heads_per_partition * head_size_,      // Q
        num_kv_heads_per_partition * head_size_,   // K
        num_kv_heads_per_partition * head_size_};  // V

    for (const auto& prefix : prefixes) {
      auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
      if (scale_tensor.defined()) {
        partition_info.scales.push_back(scale_tensor.flatten().item<float>());
      }
    }

    // Requantize if unfused checkpoint (multiple scales)
    if (partition_info.scales.size() > 1 && weight_.defined()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales)) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }
      weight_scale_.fill_(max_scale);
    } else if (partition_info.scales.size() == 1) {
      weight_scale_.fill_(partition_info.scales[0]);
    } else {
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    }

    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_FUSED_WEIGHT(input_scale, 0);
    }

    // For per-tensor quantization with fused QKV, replace scale tensors with
    // scalar max values to avoid recomputing max() in every forward() call.
    // Only apply for per-tensor quantization.
    // Per-channel/per-block quantization should NOT take max.
    if (weight_scale_.defined() && weight_scale_.numel() > 1) {
      weight_scale_ = weight_scale_.max();
    }
    if (input_scale_.defined() && input_scale_.numel() > 1) {
      input_scale_ = input_scale_.max();
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    // input_scale/input_offset are shared activation-quant params and should
    // not inherit the KV-head replication logic used by output-channel tensors.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_QKV_WEIGHT(deq_scale, 0, num_kv_head_replicas_);
    LOAD_QKV_WEIGHT(quant_bias, 0, num_kv_head_replicas_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_QKV_WEIGHT(weight_scale, 0, num_kv_head_replicas_);
    if (weight_offset_.defined()) {
      LOAD_QKV_WEIGHT(weight_offset, 0, num_kv_head_replicas_);
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});
  CHECK_EQ(num_heads_, num_kv_heads_);
  LOAD_MERGED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    const std::vector<std::string> shared_input_prefixes{""};
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
  }
}

std::optional<torch::Tensor> QKVParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    // input_scale_ is already reduced to per-tensor scale in load_state_dict.
    return input_scale_;
  }
  return std::nullopt;
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
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      options_(options),
      process_group_(process_group),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ = register_parameter(
        "per_channel_scale",
        torch::empty({out_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    smooth_ = register_parameter("smooth",
                                 torch::empty({in_features_per_partition},
                                              options.dtype(torch::kFloat32)),
                                 /*requires_grad=*/false);
    // Output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8: FP8 weight [N, K_pp] + FP32 inverse-scale grid
    // [ceil(N/bn), ceil(K_pp/bk)]. Row parallel shards input (dim 1).
    // Checkpoint BF16 scales are converted once while loading.
    const int64_t block_n = quant_args_.weight_block_size()[0];
    const int64_t block_k = quant_args_.weight_block_size()[1];
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t n_tiles = (out_features + block_n - 1) / block_n;
    const int64_t k_tiles =
        (in_features_per_partition + block_k - 1) / block_k;
    weight_scale_inv_ = register_parameter(
        "weight_scale_inv",
        torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    // Weight scale is per-tensor (scalar)
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({1}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
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
  auto bias = bias_.defined() && rank_ == 0
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;
  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(smooth_.defined()) << "smooth is required for smoothquant.";
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }

    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (is_block_fp8_quant(quant_args_)) {
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    if (weight_.scalar_type() == torch::kFloat8_e4m3fn) {
      // Native block-FP8 GEMM (per-token-group activation quant + mate/muDNN
      // groupwise matmul); XLLM_FP8_DEQUANT=1 forces the slower
      // dequant-to-BF16 fallback.
      output = block_fp8_forward(input,
                                 weight_,
                                 weight_scale_inv_,
                                 quant_args_.weight_block_size(),
                                 bias,
                                 output_buf_);
    } else {
      xllm::kernel::MatmulParams matmul_params;
      matmul_params.a = input;
      matmul_params.b = weight_;
      matmul_params.bias = bias;
      maybe_set_persistent_output_buf(
          matmul_params, output_buf_, input, weight_);
      output = xllm::kernel::matmul(matmul_params);
    }
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization
    CHECK(!quant_args_.activation_dynamic())
        << "FP8 quantization does not support activation_dynamic yet";

    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    maybe_set_persistent_output_buf(matmul_params, output_buf_, input, weight_);
    output = xllm::kernel::matmul(matmul_params);
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_WEIGHT(per_channel_scale);
    LOAD_SHARDED_WEIGHT(smooth, 0);
  } else if (is_block_fp8_quant(quant_args_)) {
    // Block-wise FP8: Row parallel shards input (dim 1) for both the FP8
    // weight and the K-block inverse-scale grid.
    // See QKVParallelLinearImpl::load_state_dict for why this decision must
    // be sticky and based on per-call weight/scale co-presence.
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
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    if (rank_ == 0) {
      LOAD_WEIGHT(quant_bias);
    } else if (quant_bias_.defined()) {
      quant_bias_.zero_();
      quant_bias_is_loaded_ = true;
    }
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : quant_args_(quant_args),
      options_(options),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  (void)linear_extra_args;
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
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    return npu_w8a8_linear_forward(input,
                                   weight_,
                                   input_scale_,
                                   input_offset_,
                                   deq_scale_,
                                   quant_bias,
                                   input.scalar_type());
  }
  if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    return dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, input.scalar_type());
#elif defined(USE_NPU)
    return npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, input.scalar_type());
#endif
  }
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;
  maybe_set_persistent_output_buf(matmul_params, output_buf_, input, weight_);

  auto output = xllm::kernel::matmul(matmul_params);
  return output;
}

bool ReplicatedLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ReplicatedLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
}

at::ScalarType ReplicatedLinearImpl::output_dtype() const {
  return output_dtype_;
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
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});
  LOAD_WEIGHT(weight);
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    LOAD_WEIGHT(quant_bias);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(weight_scale);
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  }
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
