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

#include <glog/logging.h>
#include <musa_runtime.h>
#include <tvm/ffi/extra/stl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr const char* kTopkUri = "xllm_musa_topk_gating";
constexpr const char* kMate025GemmArtifactId =
    "fbd1a6df72047350033cb4c524dd8145b8f18698";

int64_t get_musa_mp_count(int32_t device_id) {
  thread_local int32_t cached_device_id = -1;
  thread_local int64_t cached_mp_count = 0;
  if (cached_device_id == device_id && cached_mp_count > 0) {
    return cached_mp_count;
  }

  musaDeviceProp properties;
  const musaError_t error = musaGetDeviceProperties(&properties, device_id);
  CHECK_EQ(error, musaSuccess)
      << "Failed to get MUSA properties for device " << device_id << ": "
      << musaGetErrorString(error);
  cached_device_id = device_id;
  cached_mp_count = properties.multiProcessorCount;
  return cached_mp_count;
}

struct MateMubinGemmLauncher {
  const char* kernel_name;
  int64_t tile_m;
  int64_t tile_n;
  double base_score;
  bool tme_cache_hint_b;
};

constexpr std::array<MateMubinGemmLauncher, 3> kContiguousFp8MubinLaunchers = {{
    {"e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_512_"
     "256x256B128_epilogue_group_block_128_persis_stage2_stable",
     256,
     256,
     1.0,
     false},
    {"e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_"
     "128x256B128_epilogue_group_block_128_persis_stage4_stable",
     128,
     256,
     0.8,
     false},
    {"e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_"
     "256x128B128_epilogue_group_block_128_persis_stage4_stable",
     256,
     128,
     0.8,
     false},
}};
constexpr std::array<MateMubinGemmLauncher, 3> kContiguousBf16MubinLaunchers = {
    {
        {"bf16bf16bf16bf16ssgemm_gm1_nt_tce_512_"
         "256x256B128_epilogue_persis_stage2_stable",
         256,
         256,
         1.0,
         false},
        {"bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_"
         "128x256B128_epilogue_persis_stage4_stable",
         128,
         256,
         0.8,
         false},
        {"bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_"
         "256x128B128_epilogue_persis_stage4_stable",
         256,
         128,
         0.8,
         false},
    }};

constexpr std::array<MateMubinGemmLauncher, 4> kRaggedFp8MubinLaunchers = {{
    {"e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_"
     "128x256B128_epilogue_group_block_128_persis_stage4_stable",
     128,
     256,
     0.0,
     false},
    {"e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_"
     "128x256B128_epilogue_group_block_128_persis_stage4_btmenc_stable",
     128,
     256,
     0.0,
     true},
    {"e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_"
     "256x256B128_epilogue_group_block_128_persis_stage2_stable",
     256,
     256,
     0.0,
     false},
    {"e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_"
     "256x256B128_epilogue_group_block_128_persis_stage2_btmenc_stable",
     256,
     256,
     0.0,
     true},
}};

constexpr std::array<MateMubinGemmLauncher, 4> kRaggedBf16MubinLaunchers = {{
    {"bf16bf16bf16bf16ssgemm_gm3_nt_tce_256_"
     "128x256B128_epilogue_persis_stage4_stable",
     128,
     256,
     0.0,
     false},
    {"bf16bf16bf16bf16ssgemm_gm3_nt_tce_256_"
     "128x256B128_epilogue_persis_stage4_btmenc_stable",
     128,
     256,
     0.0,
     true},
    {"bf16bf16bf16bf16ssgemm_gm3_nt_tce_512_"
     "256x256B128_epilogue_persis_stage2_stable",
     256,
     256,
     0.0,
     false},
    {"bf16bf16bf16bf16ssgemm_gm3_nt_tce_512_"
     "256x256B128_epilogue_persis_stage2_btmenc_stable",
     256,
     256,
     0.0,
     true},
}};

constexpr std::array<MateMubinGemmLauncher, 3> kMaskedFp8MubinLaunchers = {{
    {"e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_"
     "128x256B128_epilogue_group_block_128_persis_stage4_btmenc_stable",
     128,
     256,
     0.0,
     true},
    {"e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_"
     "256x256B128_epilogue_group_block_128_persis_stage2_btmenc_stable",
     256,
     256,
     0.0,
     true},
    {"e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_"
     "256x256B128_epilogue_group_block_128_persis_stage2_stable",
     256,
     256,
     0.0,
     false},
}};

constexpr std::array<MateMubinGemmLauncher, 3> kMaskedBf16MubinLaunchers = {{
    {"bf16bf16bf16bf16ssgemm_gm4_nt_tce_256_"
     "128x256B128_epilogue_persis_stage4_btmenc_stable",
     128,
     256,
     0.0,
     true},
    {"bf16bf16bf16bf16ssgemm_gm4_nt_tce_512_"
     "256x256B128_epilogue_persis_stage2_btmenc_stable",
     256,
     256,
     0.0,
     true},
    {"bf16bf16bf16bf16ssgemm_gm4_nt_tce_512_"
     "256x256B128_epilogue_persis_stage2_stable",
     256,
     256,
     0.0,
     false},
}};

int64_t ceil_div(int64_t dividend, int64_t divisor) {
  CHECK_GE(dividend, 0);
  CHECK_GT(divisor, 0);
  return (dividend + divisor - 1) / divisor;
}

double estimate_contiguous_mubin_score(const MateMubinGemmLauncher& launcher,
                                       int64_t m,
                                       int64_t n,
                                       int64_t num_experts,
                                       int64_t total_mp_count) {
  const int64_t average_group_m = m / num_experts;
  const int64_t upper_average_group_m =
      static_cast<int64_t>(static_cast<double>(average_group_m) * 1.04);
  const int64_t lower_average_group_m =
      static_cast<int64_t>(static_cast<double>(average_group_m) * 0.96);
  const int64_t deviate_group_count = num_experts / 3;
  const int64_t lower_group_tiles =
      ceil_div(lower_average_group_m, launcher.tile_m);
  const int64_t upper_group_tiles =
      ceil_div(upper_average_group_m, launcher.tile_m);
  const int64_t group_tiles =
      lower_group_tiles * deviate_group_count +
      upper_group_tiles * (num_experts - deviate_group_count);
  const int64_t n_tiles = ceil_div(n, launcher.tile_n);
  const int64_t total_tiles = std::max<int64_t>(group_tiles * n_tiles, 1);
  const int64_t waves =
      std::max<int64_t>(ceil_div(total_tiles, total_mp_count), 1);
  return launcher.base_score * static_cast<double>(m) * static_cast<double>(n) /
         (static_cast<double>(launcher.tile_m) *
          static_cast<double>(launcher.tile_n) * static_cast<double>(waves) *
          static_cast<double>(total_mp_count));
}

template <size_t Size>
size_t select_contiguous_mubin_launcher(
    const std::array<MateMubinGemmLauncher, Size>& launchers,
    int64_t m,
    int64_t n,
    int64_t num_experts,
    int64_t total_mp_count) {
  CHECK_GT(m, 0);
  CHECK_GT(n, 0);
  CHECK_GT(num_experts, 0);
  CHECK_GT(total_mp_count, 0);
  auto best = std::max_element(
      launchers.begin(),
      launchers.end(),
      [=](const MateMubinGemmLauncher& lhs, const MateMubinGemmLauncher& rhs) {
        return estimate_contiguous_mubin_score(
                   lhs, m, n, num_experts, total_mp_count) <
               estimate_contiguous_mubin_score(
                   rhs, m, n, num_experts, total_mp_count);
      });
  CHECK(best != launchers.end());
  return static_cast<size_t>(std::distance(launchers.begin(), best));
}

template <size_t Size>
size_t select_ragged_mubin_launcher(
    const std::array<MateMubinGemmLauncher, Size>& launchers,
    int64_t m,
    int64_t num_experts,
    int64_t alignment) {
  CHECK_GT(num_experts, 0);
  CHECK(alignment == 128 || alignment == 256);
  const bool use_tme_cache_hint = m <= 192 * num_experts;
  auto selected =
      std::find_if(launchers.begin(),
                   launchers.end(),
                   [=](const MateMubinGemmLauncher& launcher) {
                     return launcher.tile_m == alignment &&
                            launcher.tme_cache_hint_b == use_tme_cache_hint;
                   });
  CHECK(selected != launchers.end());
  return static_cast<size_t>(std::distance(launchers.begin(), selected));
}

template <size_t Size>
size_t select_masked_mubin_launcher(
    const std::array<MateMubinGemmLauncher, Size>& launchers,
    int64_t expected_tokens) {
  CHECK_GE(expected_tokens, 0);
  const int64_t selected_m = expected_tokens <= 128 ? 128 : 256;
  const bool use_tme_cache_hint = selected_m == 128 || expected_tokens <= 256;
  auto selected =
      std::find_if(launchers.begin(),
                   launchers.end(),
                   [=](const MateMubinGemmLauncher& launcher) {
                     return launcher.tile_m == selected_m &&
                            launcher.tme_cache_hint_b == use_tme_cache_hint;
                   });
  CHECK(selected != launchers.end());
  return static_cast<size_t>(std::distance(launchers.begin(), selected));
}

template <size_t Size>
std::array<std::string, Size> resolve_mubin_paths(
    const std::array<MateMubinGemmLauncher, Size>& launchers) {
  const char* mubin_root_env = std::getenv("MATE_MUBIN_DIR");
  CHECK(mubin_root_env != nullptr && mubin_root_env[0] != '\0')
      << "MATE_MUBIN_DIR must point to the Mate 0.2.5 MUBIN root.";
  const std::string gemm_root = std::string(mubin_root_env) + "/gemm/" +
                                kMate025GemmArtifactId + "/mubin/";

  std::array<std::string, Size> paths;
  for (size_t index = 0; index < launchers.size(); ++index) {
    paths[index] = gemm_root + launchers[index].kernel_name + ".o";
    CHECK_EQ(::access(paths[index].c_str(), R_OK), 0)
        << "Mate 0.2.5 GEMM MUBIN object is unavailable: " << paths[index];
  }
  return paths;
}

struct MateMubinGemmArguments {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t num_experts;
  int64_t batch;
  int64_t quant_tile;
  int64_t total_mp_count;
  int64_t stride_m_a;
  int64_t stride_k_a;
  int64_t stride_batch_a;
  int64_t stride_n_b;
  int64_t stride_k_b;
  int64_t stride_batch_b;
  int64_t stride_m_out;
  int64_t stride_batch_out;
  int64_t scale_a_m;
  int64_t scale_a_k;
  int64_t scale_a_numel;
  int64_t scale_b_n;
  int64_t scale_b_k;
  int64_t scale_b_numel;
  int64_t group_idx_len;
  int64_t target_mp_count;
};

void launch_mubin_gemm(const MateMubinGemmLauncher& launcher,
                       const std::string& object_path,
                       const torch::Tensor& input,
                       const torch::Tensor& weights,
                       const torch::Tensor& output,
                       const std::optional<torch::Tensor>& input_scale,
                       const std::optional<torch::Tensor>& weight_scale,
                       const torch::Tensor& token_info,
                       const MateMubinGemmArguments& args) {
  const std::string launcher_uri = "run_" + std::string(launcher.kernel_name);
  get_function(launcher_uri, launcher_uri)(
      object_path,
      to_ffi_tensor_view(input),
      to_ffi_tensor_view(weights),
      to_ffi_tensor_view(output),
      to_ffi_optional_tensor(input_scale),
      to_ffi_optional_tensor(weight_scale),
      to_ffi_optional_tensor(std::nullopt),
      to_ffi_optional_tensor(std::optional<torch::Tensor>(token_info)),
      args.m,
      args.n,
      args.k,
      args.num_experts,
      args.batch,
      args.quant_tile,
      args.total_mp_count,
      args.stride_m_a,
      args.stride_k_a,
      args.stride_batch_a,
      args.stride_n_b,
      args.stride_k_b,
      args.stride_batch_b,
      args.stride_m_out,
      args.stride_batch_out,
      args.scale_a_m,
      args.scale_a_k,
      args.scale_a_numel,
      args.scale_b_n,
      args.scale_b_k,
      args.scale_b_numel,
      args.group_idx_len,
      to_ffi_optional_tensor(std::nullopt),
      args.target_mp_count,
      to_ffi_optional_tensor(std::nullopt));
}

void check_masked_moe_inputs(const torch::Tensor& input,
                             const torch::Tensor& weights,
                             const torch::Tensor& token_counts,
                             int64_t expected_tokens) {
  CHECK(input.defined() && weights.defined() && token_counts.defined())
      << "Mate masked MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 3)
      << "Mate masked MoE input must be [experts, tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate masked MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), input.size(0))
      << "Mate masked MoE token_counts must have one entry per expert.";
  CHECK_EQ(input.size(0), weights.size(0));
  CHECK_GT(input.size(1), 0);
  CHECK_EQ(input.size(2), weights.size(2))
      << "Mate masked MoE input/weight shapes are incompatible: input="
      << input.sizes() << " weights=" << weights.sizes();
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate masked MoE tensors must be contiguous.";
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32)
      << "Mate masked MoE token_counts must be int32.";
  CHECK_GE(expected_tokens, 0)
      << "Mate masked MoE expected_tokens must be non-negative.";
}

}  // namespace

torch::Tensor masked_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& token_counts,
                                   torch::ScalarType output_dtype,
                                   int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output_dtype, torch::kBFloat16);
  MusaTvmffiStreamGuard stream_guard(input.device());

  torch::Tensor output =
      torch::empty({input.size(0), input.size(1), weights.size(1)},
                   input.options().dtype(output_dtype));
  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index =
      select_masked_mubin_launcher(kMaskedBf16MubinLaunchers, expected_tokens);
  const MateMubinGemmLauncher& launcher =
      kMaskedBf16MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kMaskedBf16MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0) * input.size(1),
      .n = weights.size(1),
      .k = input.size(2),
      .num_experts = input.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.stride(1),
      .stride_k_a = input.stride(2),
      .stride_batch_a = input.stride(0) * input.size(0),
      .stride_n_b = weights.stride(1),
      .stride_k_b = weights.stride(2),
      .stride_batch_b = weights.stride(0),
      .stride_m_out = output.stride(1),
      .stride_batch_out = output.stride(0) * output.size(0),
      .scale_a_m = 0,
      .scale_a_k = 0,
      .scale_a_numel = 0,
      .scale_b_n = 0,
      .scale_b_k = 0,
      .scale_b_numel = 0,
      .group_idx_len = input.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::nullopt,
                    std::nullopt,
                    token_counts,
                    args);
  return output;
}

torch::Tensor masked_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& token_counts,
                                  torch::ScalarType output_dtype,
                                  int64_t expected_tokens) {
  check_masked_moe_inputs(input, weights, token_counts, expected_tokens);
  TORCH_CHECK(input.scalar_type() == torch::kFloat8_e4m3fn &&
                  weights.scalar_type() == torch::kFloat8_e4m3fn,
              "Mate FP8 MoE GEMM currently requires e4m3 inputs and weights.");
  TORCH_CHECK(input_scale.scalar_type() == torch::kFloat32 &&
                  weight_scale.scalar_type() == torch::kFloat32,
              "Mate FP8 MoE scales must be float32.");
  TORCH_CHECK(input_scale.dim() == 3 && weight_scale.dim() == 3,
              "Mate FP8 MoE scales must be 3-D.");
  TORCH_CHECK(input_scale.size(0) == input.size(0) &&
                  input_scale.size(1) == input.size(1) &&
                  input_scale.size(2) * 128 == input.size(2),
              "Mate FP8 input scale shape does not match input.");
  TORCH_CHECK(weight_scale.size(0) == weights.size(0) &&
                  weight_scale.size(1) * 128 == weights.size(1) &&
                  weight_scale.size(2) * 128 == weights.size(2),
              "Mate FP8 weight scale shape does not match weights.");
  TORCH_CHECK(input_scale.is_contiguous() && weight_scale.is_contiguous(),
              "Mate FP8 MoE scales must be contiguous.");

  CHECK_EQ(output_dtype, torch::kBFloat16);
  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output =
      torch::empty({input.size(0), input.size(1), weights.size(1)},
                   input.options().dtype(output_dtype));
  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index =
      select_masked_mubin_launcher(kMaskedFp8MubinLaunchers, expected_tokens);
  const MateMubinGemmLauncher& launcher =
      kMaskedFp8MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kMaskedFp8MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0) * input.size(1),
      .n = weights.size(1),
      .k = input.size(2),
      .num_experts = input.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.stride(1),
      .stride_k_a = input.stride(2),
      .stride_batch_a = input.stride(0) * input.size(0),
      .stride_n_b = weights.stride(1),
      .stride_k_b = weights.stride(2),
      .stride_batch_b = weights.stride(0),
      .stride_m_out = output.stride(1),
      .stride_batch_out = output.stride(0) * output.size(0),
      .scale_a_m = input_scale.size(0) * input_scale.size(1),
      .scale_a_k = input_scale.size(2),
      .scale_a_numel = input_scale.numel(),
      .scale_b_n = weight_scale.size(1),
      .scale_b_k = weight_scale.size(2),
      .scale_b_numel = weight_scale.numel(),
      .group_idx_len = input.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::optional<torch::Tensor>(input_scale),
                    std::optional<torch::Tensor>(weight_scale),
                    token_counts,
                    args);
  return output;
}

torch::Tensor contiguous_moe_gemm_bf16(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& token_counts,
                                       torch::ScalarType output_dtype) {
  CHECK(input.defined() && weights.defined() && token_counts.defined())
      << "Mate contiguous BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous BF16 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous BF16 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous BF16 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);

  CHECK_EQ(output_dtype, torch::kBFloat16);
  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  if (input.size(0) == 0 || weights.size(1) == 0) {
    return output;
  }
  if (input.size(1) == 0) {
    output.zero_();
    return output;
  }

  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index =
      select_contiguous_mubin_launcher(kContiguousBf16MubinLaunchers,
                                       input.size(0),
                                       weights.size(1),
                                       weights.size(0),
                                       total_mp_count);
  const MateMubinGemmLauncher& launcher =
      kContiguousBf16MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kContiguousBf16MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0),
      .n = weights.size(1),
      .k = input.size(1),
      .num_experts = weights.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.size(1),
      .stride_k_a = input.size(0),
      .stride_batch_a = input.numel(),
      .stride_n_b = weights.size(2),
      .stride_k_b = weights.size(1),
      .stride_batch_b = weights.size(1) * weights.size(2),
      .stride_m_out = output.size(1),
      .stride_batch_out = output.numel(),
      .scale_a_m = 0,
      .scale_a_k = 0,
      .scale_a_numel = 0,
      .scale_b_n = 0,
      .scale_b_k = 0,
      .scale_b_numel = 0,
      .group_idx_len = weights.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::nullopt,
                    std::nullopt,
                    token_counts,
                    args);
  return output;
}

torch::Tensor ragged_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& row_expert_ids,
                                   torch::ScalarType output_dtype,
                                   int64_t alignment) {
  CHECK(input.defined() && weights.defined() && row_expert_ids.defined())
      << "Mate Ragged BF16 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && weights.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged BF16 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged BF16 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  CHECK_EQ(output_dtype, torch::kBFloat16);
  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  if (input.size(0) == 0 || weights.size(1) == 0) {
    return output;
  }
  if (input.size(1) == 0) {
    output.zero_();
    return output;
  }

  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index = select_ragged_mubin_launcher(
      kRaggedBf16MubinLaunchers, input.size(0), weights.size(0), alignment);
  const MateMubinGemmLauncher& launcher =
      kRaggedBf16MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kRaggedBf16MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0),
      .n = weights.size(1),
      .k = input.size(1),
      .num_experts = weights.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.stride(0),
      .stride_k_a = input.stride(1),
      .stride_batch_a = input.stride(0) * input.size(0),
      .stride_n_b = weights.stride(1),
      .stride_k_b = weights.stride(2),
      .stride_batch_b = weights.stride(0),
      .stride_m_out = output.stride(0),
      .stride_batch_out = output.stride(0) * output.size(0),
      .scale_a_m = 0,
      .scale_a_k = 0,
      .scale_a_numel = 0,
      .scale_b_n = 0,
      .scale_b_k = 0,
      .scale_b_numel = 0,
      .group_idx_len = input.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::nullopt,
                    std::nullopt,
                    row_expert_ids,
                    args);
  return output;
}

torch::Tensor contiguous_moe_gemm_fp8(const torch::Tensor& input,
                                      const torch::Tensor& input_scale,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& token_counts,
                                      torch::ScalarType output_dtype) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && token_counts.defined())
      << "Mate contiguous FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2)
      << "Mate contiguous FP8 MoE input must be [tokens, K], got "
      << input.sizes();
  CHECK_EQ(weights.dim(), 3)
      << "Mate contiguous FP8 MoE weights must be [experts, N, K], got "
      << weights.sizes();
  CHECK_EQ(token_counts.dim(), 1);
  CHECK_EQ(token_counts.size(0), weights.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        token_counts.is_contiguous())
      << "Mate contiguous FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(output_dtype, torch::kBFloat16);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(token_counts.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));

  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  if (input.size(0) == 0 || weights.size(1) == 0) {
    return output;
  }
  if (input.size(1) == 0) {
    output.zero_();
    return output;
  }

  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index =
      select_contiguous_mubin_launcher(kContiguousFp8MubinLaunchers,
                                       input.size(0),
                                       weights.size(1),
                                       weights.size(0),
                                       total_mp_count);
  const MateMubinGemmLauncher& launcher =
      kContiguousFp8MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kContiguousFp8MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0),
      .n = weights.size(1),
      .k = input.size(1),
      .num_experts = weights.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.size(1),
      .stride_k_a = input.size(0),
      .stride_batch_a = input.numel(),
      .stride_n_b = weights.size(2),
      .stride_k_b = weights.size(1),
      .stride_batch_b = weights.size(1) * weights.size(2),
      .stride_m_out = output.size(1),
      .stride_batch_out = output.numel(),
      .scale_a_m = input_scale.size(0),
      .scale_a_k = input_scale.size(1),
      .scale_a_numel = input_scale.numel(),
      .scale_b_n = weight_scale.size(1),
      .scale_b_k = weight_scale.size(2),
      .scale_b_numel = weight_scale.numel(),
      .group_idx_len = weights.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::optional<torch::Tensor>(input_scale),
                    std::optional<torch::Tensor>(weight_scale),
                    token_counts,
                    args);
  return output;
}

torch::Tensor ragged_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& row_expert_ids,
                                  torch::ScalarType output_dtype,
                                  int64_t alignment) {
  CHECK(input.defined() && input_scale.defined() && weights.defined() &&
        weight_scale.defined() && row_expert_ids.defined())
      << "Mate Ragged FP8 MoE GEMM received an undefined tensor.";
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weights.dim(), 3);
  CHECK_EQ(row_expert_ids.dim(), 1);
  CHECK_EQ(row_expert_ids.size(0), input.size(0));
  CHECK_EQ(input.size(1), weights.size(2));
  CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
        weights.is_contiguous() && weight_scale.is_contiguous() &&
        row_expert_ids.is_contiguous())
      << "Mate Ragged FP8 MoE tensors must be contiguous.";
  CHECK_EQ(input.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(weights.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(weight_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(row_expert_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(input_scale.dim(), 2);
  CHECK_EQ(input_scale.size(0), input.size(0));
  CHECK_EQ(input_scale.size(1) * 128, input.size(1));
  CHECK_EQ(weight_scale.dim(), 3);
  CHECK_EQ(weight_scale.size(0), weights.size(0));
  CHECK_EQ(weight_scale.size(1) * 128, weights.size(1));
  CHECK_EQ(weight_scale.size(2) * 128, weights.size(2));
  CHECK(alignment == 128 || alignment == 256)
      << "Mate Ragged FP8 MoE alignment must be 128 or 256.";
  CHECK_EQ(input.size(0) % alignment, 0);

  CHECK_EQ(output_dtype, torch::kBFloat16);
  MusaTvmffiStreamGuard stream_guard(input.device());
  torch::Tensor output = torch::empty({input.size(0), weights.size(1)},
                                      input.options().dtype(output_dtype));
  if (input.size(0) == 0 || weights.size(1) == 0) {
    return output;
  }
  if (input.size(1) == 0) {
    output.zero_();
    return output;
  }

  const int64_t total_mp_count =
      get_musa_mp_count(static_cast<int32_t>(input.get_device()));
  const size_t launcher_index = select_ragged_mubin_launcher(
      kRaggedFp8MubinLaunchers, input.size(0), weights.size(0), alignment);
  const MateMubinGemmLauncher& launcher =
      kRaggedFp8MubinLaunchers[launcher_index];
  static const auto object_paths =
      resolve_mubin_paths(kRaggedFp8MubinLaunchers);
  const MateMubinGemmArguments args{
      .m = input.size(0),
      .n = weights.size(1),
      .k = input.size(1),
      .num_experts = weights.size(0),
      .batch = 1,
      .quant_tile = 128,
      .total_mp_count = total_mp_count,
      .stride_m_a = input.stride(0),
      .stride_k_a = input.stride(1),
      .stride_batch_a = input.stride(0) * input.size(0),
      .stride_n_b = weights.stride(1),
      .stride_k_b = weights.stride(2),
      .stride_batch_b = weights.stride(0),
      .stride_m_out = output.stride(0),
      .stride_batch_out = output.stride(0) * output.size(0),
      .scale_a_m = input_scale.size(0),
      .scale_a_k = input_scale.size(1),
      .scale_a_numel = input_scale.numel(),
      .scale_b_n = weight_scale.size(1),
      .scale_b_k = weight_scale.size(2),
      .scale_b_numel = weight_scale.numel(),
      .group_idx_len = input.size(0),
      .target_mp_count = total_mp_count,
  };
  launch_mubin_gemm(launcher,
                    object_paths[launcher_index],
                    input,
                    weights,
                    output,
                    std::optional<torch::Tensor>(input_scale),
                    std::optional<torch::Tensor>(weight_scale),
                    row_expert_ids,
                    args);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> musa_moe_topk_softmax(
    const torch::Tensor& router_logits,
    int64_t topk) {
  CHECK(router_logits.defined());
  CHECK_EQ(router_logits.dim(), 2);
  CHECK(router_logits.is_contiguous());
  CHECK_EQ(router_logits.scalar_type(), torch::kBFloat16);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, router_logits.size(1));

  MusaTvmffiStreamGuard stream_guard(router_logits.device());
  auto topk_weights =
      torch::empty({router_logits.size(0), topk},
                   router_logits.options().dtype(torch::kFloat32));
  auto topk_ids = torch::empty({router_logits.size(0), topk},
                               router_logits.options().dtype(torch::kInt32));
  auto unused_correction_bias = topk_weights.reshape({-1});
  get_function(kTopkUri, "xllm_musa_topk_softmax")(
      to_ffi_tensor_view(topk_weights),
      to_ffi_tensor_view(topk_ids),
      to_ffi_tensor_view(router_logits),
      /*renormalize=*/true,
      /*moe_softcapping=*/0.0,
      to_ffi_tensor_view(unused_correction_bias),
      /*has_correction_bias=*/false);
  return std::make_tuple(topk_weights, topk_ids);
}

bool musa_moe_topk_softmax_available() {
  static const bool available = [] {
    const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
    if (ops_path == nullptr || ops_path[0] == '\0') {
      return false;
    }
    const std::string so_path =
        std::string(ops_path) + "/" + kTopkUri + "/" + kTopkUri + ".so";
    return ::access(so_path.c_str(), R_OK) == 0;
  }();
  return available;
}

}  // namespace xllm::kernel::musa
