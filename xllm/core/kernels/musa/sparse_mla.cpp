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

#include "core/kernels/musa/sparse_mla.h"

#include <c10/musa/MUSAGuard.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {
void check_tensor(const torch::Tensor& tensor,
                  const torch::Device& device,
                  torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.is_contiguous());
}
}  // namespace

torch::Tensor sparse_mla_bf16(const torch::Tensor& query,
                              const torch::Tensor& cache,
                              const torch::Tensor& indices,
                              const torch::Tensor& lengths,
                              double scale) {
  CHECK(query.defined());
  CHECK(query.device().is_privateuseone());
  check_tensor(query, query.device(), torch::kBFloat16);
  check_tensor(cache, query.device(), torch::kBFloat16);
  check_tensor(indices, query.device(), torch::kInt32);
  check_tensor(lengths, query.device(), torch::kInt32);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(cache.dim(), 3);
  CHECK_EQ(indices.dim(), 3);
  CHECK_EQ(lengths.dim(), 1);
  CHECK_EQ(query.size(2), 512);
  CHECK_EQ(cache.size(1), 1);
  CHECK_EQ(cache.size(2), 512);
  CHECK_GT(cache.size(0), 0);
  CHECK_EQ(indices.size(0), query.size(0));
  CHECK_EQ(indices.size(1), 1);
  CHECK_GT(indices.size(2), 0);
  CHECK_EQ(indices.size(2) % 64, 0);
  CHECK_EQ(lengths.size(0), query.size(0));
  for (const auto& tensor : {query, cache, indices}) {
    CHECK_LE(tensor.numel() * tensor.element_size(),
             std::numeric_limits<int32_t>::max())
        << "Sparse MLA v1 export uses 32-bit tensor addressing";
  }
  CHECK(std::isfinite(scale) && scale > 0);
  const int64_t heads = query.size(1);
  CHECK(heads == 8 || heads == 16 || heads == 32 || heads == 64);
  c10::musa::MUSAGuard device_guard(query.device());
  musaDeviceProp properties{};
  CHECK_EQ(musaGetDeviceProperties(&properties, query.device().index()),
           musaSuccess);
  CHECK_EQ(properties.major * 10 + properties.minor, 31);
  auto output = torch::empty_like(query);
  if (query.size(0) == 0) {
    return output;
  }
  CHECK(!is_stream_capturing())
      << "Sparse MLA currently requires eager execution";
  CHECK(ensure_tilelang_loader());
  auto function =
      get_function("sparse_mla_bf16_v1_mp31_h" + std::to_string(heads), "main");
  auto max_logits = torch::empty({query.size(0), heads},
                                 query.options().dtype(torch::kFloat32));
  auto lse = torch::empty_like(max_logits);
  // Keep scratch alive until the FFI stream has handed its work back to Torch.
  TvmffiStreamGuard stream_guard(query.device());
  function(to_ffi_tensor(query),
           to_ffi_tensor(cache),
           to_ffi_tensor(indices),
           to_ffi_tensor(output),
           to_ffi_tensor(max_logits),
           to_ffi_tensor(lse),
           to_ffi_tensor(lengths),
           static_cast<float>(scale * std::log2(std::exp(1.0))),
           /*persistent_blocks=*/0,
           /*kv_span_bytes=*/cache.numel() * cache.element_size());
  return output;
}

torch::Tensor sparse_mla_nope_fp8(const torch::Tensor& query,
                                  const torch::Tensor& cache,
                                  const torch::Tensor& indices,
                                  const torch::Tensor& lengths,
                                  double scale) {
  CHECK(query.device().is_privateuseone());
  check_tensor(query, query.device(), torch::kBFloat16);
  check_tensor(cache, query.device(), torch::kUInt8);
  check_tensor(indices, query.device(), torch::kInt32);
  check_tensor(lengths, query.device(), torch::kInt32);
  CHECK_EQ(query.dim(), 4);
  CHECK_EQ(query.size(1), 1);
  CHECK_EQ(query.size(3), 512);
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(2);
  CHECK(heads == 8 || heads == 16 || heads == 32 || heads == 64);
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), 528);
  CHECK_GT(cache.numel(), 0);
  CHECK_EQ(indices.dim(), 4);
  CHECK_EQ(indices.size(0), batch);
  CHECK_EQ(indices.size(1), 1);
  CHECK_EQ(indices.size(2), 1);
  CHECK_GT(indices.size(3), 0);
  CHECK_EQ(indices.size(3) % 64, 0);
  CHECK_EQ(lengths.sizes(), torch::IntArrayRef({batch}));
  for (const auto& tensor : {query, cache, indices}) {
    CHECK_LE(tensor.numel() * tensor.element_size(),
             std::numeric_limits<int32_t>::max());
  }
  CHECK(std::isfinite(scale) && scale > 0);
  c10::musa::MUSAGuard device_guard(query.device());
  CHECK(!is_stream_capturing());
  musaDeviceProp properties{};
  CHECK_EQ(musaGetDeviceProperties(&properties, query.device().index()),
           musaSuccess);
  CHECK_EQ(properties.major * 10 + properties.minor, 31);
  auto output = torch::empty_like(query);
  if (batch == 0) {
    return output;
  }
  CHECK(ensure_tilelang_loader());
  auto metadata_function = get_function("mla_ops", "get_mla_decoding_metadata");
  const int64_t parts = std::max<int64_t>(properties.multiProcessorCount, 1);
  CHECK_GT(parts, 1);
  CHECK_LE(parts, 64)
      << "Exported scheduled decode supports at most 64 MP parts";
  const int64_t max_splits = parts <= 32 ? 32 : 64;
  auto function =
      get_function("sparse_mla_nope_fp8_v1_mp31_h" + std::to_string(heads) +
                       "_splits" + std::to_string(max_splits),
                   "main");
  auto ints = query.options().dtype(torch::kInt32);
  auto floats = query.options().dtype(torch::kFloat32);
  auto metadata = torch::empty({parts, 8}, ints);
  auto splits = torch::empty({batch + 1}, ints);
  auto lse = torch::empty({batch, heads, 1}, floats);
  auto glse = torch::empty({batch + parts, 1, heads}, floats);
  auto partial = torch::empty({batch + parts, 1, heads, 512}, floats);
  auto flat_cache = cache.view({-1, 1, 528});
  auto fp8_cache = flat_cache.view(torch::kFloat8_e4m3fn);
  auto bf16_cache = flat_cache.view(torch::kBFloat16);
  auto scales = flat_cache.view(torch::kFloat32);
  auto ffi_query = to_ffi_tensor(query);
  auto ffi_lengths = to_ffi_tensor(lengths);
  TvmffiStreamGuard stream_guard(query.device());
  metadata_function(
      ffi::Optional<ffi::Tensor>(),
      heads,
      int64_t{1},
      ffi::Optional<int64_t>(heads),
      true,
      ffi::Optional<int64_t>(indices.size(3)),
      ffi::Optional<ffi::Tensor>(to_ffi_tensor(metadata)),
      ffi::Optional<ffi::Tensor>(to_ffi_tensor(splits)),
      ffi::Optional<ffi::TensorView>(ffi::TensorView(ffi_query)),
      ffi::Optional<int64_t>(batch),
      ffi::Optional<ffi::TensorView>(ffi::TensorView(ffi_lengths)),
      ffi::Optional<ffi::TensorView>(),
      ffi::Optional<int64_t>());
  function(ffi_query,
           to_ffi_tensor(fp8_cache),
           to_ffi_tensor(bf16_cache),
           to_ffi_tensor(scales),
           to_ffi_tensor(indices),
           ffi_lengths,
           to_ffi_tensor(metadata),
           to_ffi_tensor(splits),
           to_ffi_tensor(glse),
           to_ffi_tensor(partial),
           to_ffi_tensor(output),
           to_ffi_tensor(lse),
           static_cast<float>(scale * std::log2(std::exp(1.0))),
           cache.numel() * cache.element_size());
  return output;
}
}  // namespace xllm::kernel::musa
