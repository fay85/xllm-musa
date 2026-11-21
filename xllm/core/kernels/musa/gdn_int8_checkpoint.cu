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

#include "core/kernels/musa/gdn_int8_checkpoint.h"

#ifndef C10_USE_GLOG
#define C10_USE_GLOG 1
#endif
#include <glog/logging.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/kernels/musa/musa_kernel_utils.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kQuantThreads = 128;
constexpr int32_t kConvCopyThreads = 256;

template <typename T>
__device__ __forceinline__ float ssm_to_float(T v);

template <>
__device__ __forceinline__ float ssm_to_float<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float ssm_to_float<__mt_bfloat16>(__mt_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float ssm_to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T float_to_ssm(float v);

template <>
__device__ __forceinline__ float float_to_ssm<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ __mt_bfloat16 float_to_ssm<__mt_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ __half float_to_ssm<__half>(float v) {
  return __float2half_rn(v);
}

template <typename SsmT>
__global__ void quantize_ssm_to_int8_kernel(
    const uint64_t* __restrict__ ssm_ptrs,
    int8_t* __restrict__ qdata,
    __mt_bfloat16* __restrict__ scales,
    int64_t num_ckpt_slots,
    int64_t num_heads,
    int64_t head_v_dim,
    int64_t head_k_dim,
    int64_t active_slot,
    int64_t ckpt_slot) {
  const int64_t layer = static_cast<int64_t>(blockIdx.y);
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t head_elems = head_v_dim * head_k_dim;
  const int64_t slot_elems = num_heads * head_elems;
  const SsmT* __restrict__ src =
      reinterpret_cast<const SsmT*>(ssm_ptrs[layer]) +
      active_slot * slot_elems + head * head_elems;
  int8_t* __restrict__ dst_q =
      qdata + ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) *
                  head_elems;
  __mt_bfloat16* __restrict__ dst_scale =
      scales +
      ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) * head_k_dim;

  for (int64_t k = static_cast<int64_t>(threadIdx.x); k < head_k_dim;
       k += static_cast<int64_t>(blockDim.x)) {
    float amax = 0.0f;
    for (int64_t v = 0; v < head_v_dim; ++v) {
      const float val = fabsf(ssm_to_float(src[v * head_k_dim + k]));
      amax = fmaxf(amax, val);
    }
    const float raw_scale = fmaxf(amax, 1e-8f) / 127.0f;
    const __mt_bfloat16 scale_bf16 = __float2bfloat16_rn(raw_scale);
    float q_scale = __bfloat162float(scale_bf16);
    if (q_scale < 1e-12f) {
      q_scale = 1e-12f;
    }
    const float inv_scale = 1.0f / q_scale;
    dst_scale[k] = scale_bf16;

    for (int64_t v = 0; v < head_v_dim; ++v) {
      const float val = ssm_to_float(src[v * head_k_dim + k]);
      float qf = nearbyintf(val * inv_scale);
      qf = fminf(fmaxf(qf, -127.0f), 127.0f);
      dst_q[v * head_k_dim + k] = static_cast<int8_t>(qf);
    }
  }
}

template <typename SsmT>
__global__ void dequantize_ssm_from_int8_kernel(
    const uint64_t* __restrict__ ssm_ptrs,
    const int8_t* __restrict__ qdata,
    const __mt_bfloat16* __restrict__ scales,
    int64_t num_ckpt_slots,
    int64_t num_heads,
    int64_t head_v_dim,
    int64_t head_k_dim,
    int64_t active_slot,
    int64_t ckpt_slot) {
  const int64_t layer = static_cast<int64_t>(blockIdx.y);
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t head_elems = head_v_dim * head_k_dim;
  const int64_t slot_elems = num_heads * head_elems;
  SsmT* __restrict__ dst =
      reinterpret_cast<SsmT*>(ssm_ptrs[layer]) + active_slot * slot_elems +
      head * head_elems;
  const int8_t* __restrict__ src_q =
      qdata + ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) *
                  head_elems;
  const __mt_bfloat16* __restrict__ src_scale =
      scales +
      ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) * head_k_dim;

  for (int64_t k = static_cast<int64_t>(threadIdx.x); k < head_k_dim;
       k += static_cast<int64_t>(blockDim.x)) {
    const float q_scale = __bfloat162float(src_scale[k]);
    for (int64_t v = 0; v < head_v_dim; ++v) {
      const float val =
          static_cast<float>(src_q[v * head_k_dim + k]) * q_scale;
      dst[v * head_k_dim + k] = float_to_ssm<SsmT>(val);
    }
  }
}

__global__ void copy_conv_slot_kernel(const uint64_t* __restrict__ conv_ptrs,
                                      uint16_t* __restrict__ conv_ckpt,
                                      int64_t num_ckpt_slots,
                                      int64_t conv_elems_per_slot,
                                      int64_t active_slot,
                                      int64_t ckpt_slot,
                                      bool to_ckpt) {
  const int64_t layer = static_cast<int64_t>(blockIdx.y);
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (idx >= conv_elems_per_slot) {
    return;
  }
  uint16_t* __restrict__ active_ptr =
      reinterpret_cast<uint16_t*>(conv_ptrs[layer]) +
      active_slot * conv_elems_per_slot;
  uint16_t* __restrict__ ckpt_ptr =
      conv_ckpt + (layer * num_ckpt_slots + ckpt_slot) * conv_elems_per_slot;
  if (to_ckpt) {
    ckpt_ptr[idx] = active_ptr[idx];
  } else {
    active_ptr[idx] = ckpt_ptr[idx];
  }
}

struct GdnInt8CheckpointPool {
  c10::Device device = c10::Device(c10::kPrivateUse1, 0);
  int64_t num_active_slots = 0;
  int64_t num_ckpt_slots = 0;
  int64_t num_layers = 0;
  int64_t num_heads = 0;
  int64_t head_v_dim = 0;
  int64_t head_k_dim = 0;
  int64_t conv_elems_per_slot = 0;
  torch::ScalarType ssm_dtype = torch::kFloat32;
  torch::ScalarType conv_dtype = torch::kBFloat16;
  torch::Tensor qdata;
  torch::Tensor scale;
  torch::Tensor conv_ckpt;
  torch::Tensor d_ssm_ptrs;
  torch::Tensor d_conv_ptrs;
};

std::mutex& pool_mutex() {
  static std::mutex m;
  return m;
}

template <typename SsmT>
__global__ void quantize_single_layer_ssm_to_int8_kernel(
    const SsmT* __restrict__ src_layer_ssm,
    int8_t* __restrict__ qdata,
    __mt_bfloat16* __restrict__ scales,
    int64_t layer,
    int64_t num_ckpt_slots,
    int64_t num_heads,
    int64_t head_v_dim,
    int64_t head_k_dim,
    int64_t ckpt_slot) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t head_elems = head_v_dim * head_k_dim;
  const SsmT* __restrict__ src = src_layer_ssm + head * head_elems;
  int8_t* __restrict__ dst_q =
      qdata + ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) *
                  head_elems;
  __mt_bfloat16* __restrict__ dst_scale =
      scales +
      ((layer * num_ckpt_slots + ckpt_slot) * num_heads + head) * head_k_dim;

  for (int64_t k = static_cast<int64_t>(threadIdx.x); k < head_k_dim;
       k += static_cast<int64_t>(blockDim.x)) {
    float amax = 0.0f;
    for (int64_t v = 0; v < head_v_dim; ++v) {
      const float val = fabsf(ssm_to_float(src[v * head_k_dim + k]));
      amax = fmaxf(amax, val);
    }
    const float raw_scale = fmaxf(amax, 1e-8f) / 127.0f;
    const __mt_bfloat16 scale_bf16 = __float2bfloat16_rn(raw_scale);
    float q_scale = __bfloat162float(scale_bf16);
    if (q_scale < 1e-12f) {
      q_scale = 1e-12f;
    }
    const float inv_scale = 1.0f / q_scale;
    dst_scale[k] = scale_bf16;

    for (int64_t v = 0; v < head_v_dim; ++v) {
      const float val = ssm_to_float(src[v * head_k_dim + k]);
      float qf = nearbyintf(val * inv_scale);
      qf = fminf(fmaxf(qf, -127.0f), 127.0f);
      dst_q[v * head_k_dim + k] = static_cast<int8_t>(qf);
    }
  }
}

__global__ void copy_single_layer_conv_to_ckpt_kernel(
    const uint16_t* __restrict__ src_conv,
    uint16_t* __restrict__ conv_ckpt,
    int64_t layer,
    int64_t num_ckpt_slots,
    int64_t conv_elems_per_slot,
    int64_t src_cols,
    int64_t dst_cols,
    int64_t ckpt_slot) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= conv_elems_per_slot) {
    return;
  }
  uint16_t* __restrict__ ckpt_ptr =
      conv_ckpt + (layer * num_ckpt_slots + ckpt_slot) * conv_elems_per_slot;
  if (src_cols == dst_cols) {
    ckpt_ptr[idx] = src_conv[idx];
  } else {
    const int64_t row = idx / dst_cols;
    const int64_t col = idx % dst_cols;
    ckpt_ptr[idx] =
        (col < src_cols) ? src_conv[row * src_cols + col] : uint16_t{0};
  }
}

thread_local ScopedGdnSinglepassCheckpointPlan* tls_active_singlepass_plan =
    nullptr;

std::unordered_map<const void*, std::shared_ptr<GdnInt8CheckpointPool>>&
pool_registry() {
  static std::unordered_map<const void*, std::shared_ptr<GdnInt8CheckpointPool>>
      reg;
  return reg;
}

std::unordered_map<const void*,
                   std::pair<std::shared_ptr<GdnInt8CheckpointPool>, int64_t>>&
layer_pool_registry() {
  static std::unordered_map<
      const void*,
      std::pair<std::shared_ptr<GdnInt8CheckpointPool>, int64_t>>
      reg;
  return reg;
}

const void* find_pool_key(const std::vector<::xllm::KVCache>& kv_caches) {
  for (const auto& kv : kv_caches) {
    const torch::Tensor ssm = kv.get_ssm_cache();
    if (ssm.defined() && ssm.numel() > 0) {
      return ssm.data_ptr();
    }
  }
  return nullptr;
}

std::shared_ptr<GdnInt8CheckpointPool> lookup_pool(
    const std::vector<::xllm::KVCache>& kv_caches) {
  const void* key = find_pool_key(kv_caches);
  if (key == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(pool_mutex());
  auto& reg = pool_registry();
  auto it = reg.find(key);
  return it == reg.end() ? nullptr : it->second;
}

}  // namespace

void init_gdn_int8_checkpoint_pool(
    const std::vector<::xllm::KVCache>& kv_caches,
    int64_t num_ckpt_slots) {
  if (num_ckpt_slots <= 0) {
    return;
  }
  std::vector<uint64_t> ssm_ptrs;
  std::vector<uint64_t> conv_ptrs;
  int64_t num_active_slots = 0;
  int64_t num_heads = 0;
  int64_t head_v_dim = 0;
  int64_t head_k_dim = 0;
  int64_t conv_elems = 0;
  torch::ScalarType ssm_dtype = torch::kFloat32;
  torch::ScalarType conv_dtype = torch::kBFloat16;
  std::optional<c10::Device> dev;

  for (const auto& kv : kv_caches) {
    const torch::Tensor ssm = kv.get_ssm_cache();
    const torch::Tensor conv = kv.get_conv_cache();
    if (!ssm.defined() || ssm.numel() == 0) {
      continue;
    }
    CHECK(conv.defined() && conv.numel() > 0)
        << "GDN layer must define both conv and ssm caches";
    CHECK(ssm.is_contiguous() && conv.is_contiguous())
        << "GDN caches must be contiguous for Int8 checkpointing";
    CHECK_EQ(ssm.dim(), 4);
    CHECK_EQ(conv.dim(), 3);
    CHECK_EQ(conv.element_size(), 2)
        << "GDN conv cache must have 2-byte elements";
    CHECK_EQ(ssm.size(0), conv.size(0))
        << "Int8 checkpointing requires checkpoint_stride == 1";
    if (!dev.has_value()) {
      dev = ssm.device();
      num_active_slots = ssm.size(0);
      num_heads = ssm.size(1);
      head_v_dim = ssm.size(2);
      head_k_dim = ssm.size(3);
      conv_elems = conv.size(1) * conv.size(2);
      ssm_dtype = ssm.scalar_type();
      conv_dtype = conv.scalar_type();
      CHECK_EQ(torch::elementSize(conv_dtype), 2u)
          << "conv_cache expected 16-bit dtype, got " << conv_dtype;
    } else {
      CHECK_EQ(ssm.size(0), num_active_slots);
      CHECK_EQ(ssm.size(1), num_heads);
      CHECK_EQ(ssm.size(2), head_v_dim);
      CHECK_EQ(ssm.size(3), head_k_dim);
      CHECK_EQ(conv.size(1) * conv.size(2), conv_elems);
      CHECK_EQ(ssm.scalar_type(), ssm_dtype);
      CHECK_EQ(conv.scalar_type(), conv_dtype);
    }
    ssm_ptrs.push_back(reinterpret_cast<uint64_t>(ssm.data_ptr()));
    conv_ptrs.push_back(reinterpret_cast<uint64_t>(conv.data_ptr()));
  }

  if (ssm_ptrs.empty() || !dev.has_value()) {
    return;
  }

  auto pool = std::make_shared<GdnInt8CheckpointPool>();
  pool->device = *dev;
  pool->num_active_slots = num_active_slots;
  pool->num_ckpt_slots = num_ckpt_slots;
  pool->num_layers = static_cast<int64_t>(ssm_ptrs.size());
  pool->num_heads = num_heads;
  pool->head_v_dim = head_v_dim;
  pool->head_k_dim = head_k_dim;
  pool->conv_elems_per_slot = conv_elems;
  pool->ssm_dtype = ssm_dtype;
  pool->conv_dtype = conv_dtype;

  const XLLMOptionalDeviceGuard device_guard(*dev);
  pool->qdata = torch::empty(
      {pool->num_layers, num_ckpt_slots, num_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().dtype(torch::kInt8).device(*dev));
  pool->scale = torch::empty(
      {pool->num_layers, num_ckpt_slots, num_heads, head_k_dim},
      torch::TensorOptions().dtype(torch::kBFloat16).device(*dev));
  pool->conv_ckpt = torch::empty(
      {pool->num_layers, num_ckpt_slots, conv_elems},
      torch::TensorOptions().dtype(conv_dtype).device(*dev));

  torch::Tensor h_ssm_ptrs =
      torch::from_blob(ssm_ptrs.data(),
                       {pool->num_layers},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();
  torch::Tensor h_conv_ptrs =
      torch::from_blob(conv_ptrs.data(),
                       {pool->num_layers},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();
  pool->d_ssm_ptrs = h_ssm_ptrs.to(*dev);
  pool->d_conv_ptrs = h_conv_ptrs.to(*dev);
  musaDeviceSynchronize();

  const void* key = reinterpret_cast<const void*>(ssm_ptrs[0]);
  {
    std::lock_guard<std::mutex> lock(pool_mutex());
    pool_registry()[key] = pool;
    auto& layer_reg = layer_pool_registry();
    for (size_t layer_i = 0; layer_i < ssm_ptrs.size(); ++layer_i) {
      layer_reg[reinterpret_cast<const void*>(ssm_ptrs[layer_i])] =
          std::make_pair(pool, static_cast<int64_t>(layer_i));
    }
  }

  const int64_t total_bytes = pool->qdata.nbytes() + pool->scale.nbytes() +
                              pool->conv_ckpt.nbytes();
  LOG(INFO) << "Initialized MUSA GDN Int8 Checkpoint Pool: layers="
            << pool->num_layers << ", active_slots=" << pool->num_active_slots
            << ", ckpt_slots=" << pool->num_ckpt_slots
            << ", total_bytes=" << (total_bytes / (1024.0 * 1024.0)) << " MB";
}

int64_t get_gdn_int8_checkpoint_slots(
    const std::vector<::xllm::KVCache>& kv_caches) {
  auto pool = lookup_pool(kv_caches);
  return pool ? pool->num_ckpt_slots : 0;
}

void store_gdn_int8_checkpoint(const std::vector<::xllm::KVCache>& kv_caches,
                               int32_t active_slot_id,
                               int32_t ckpt_logical_id) {
  auto pool = lookup_pool(kv_caches);
  CHECK(pool != nullptr) << "GDN Int8 Checkpoint Pool is not initialized";
  CHECK_GT(active_slot_id, 0);
  CHECK_LT(active_slot_id, pool->num_active_slots);
  const int64_t ckpt_slot =
      static_cast<int64_t>(ckpt_logical_id) - pool->num_active_slots;
  CHECK_GE(ckpt_slot, 0);
  CHECK_LT(ckpt_slot, pool->num_ckpt_slots);

  const XLLMOptionalDeviceGuard device_guard(pool->device);
  musaStream_t stream = current_musa_stream(pool->device);

  const dim3 quant_grid(static_cast<uint32_t>(pool->num_heads),
                        static_cast<uint32_t>(pool->num_layers));
  const uint64_t* ssm_ptrs =
      reinterpret_cast<const uint64_t*>(pool->d_ssm_ptrs.data_ptr<int64_t>());
  int8_t* qdata = pool->qdata.data_ptr<int8_t>();
  __mt_bfloat16* scales =
      reinterpret_cast<__mt_bfloat16*>(pool->scale.data_ptr());

  if (pool->ssm_dtype == torch::kFloat32) {
    quantize_ssm_to_int8_kernel<float>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_slot_id,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kBFloat16) {
    quantize_ssm_to_int8_kernel<__mt_bfloat16>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_slot_id,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kFloat16) {
    quantize_ssm_to_int8_kernel<__half>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_slot_id,
            ckpt_slot);
  } else {
    LOG(FATAL) << "Unsupported GDN SSM dtype for Int8 checkpoint: "
               << pool->ssm_dtype;
  }
  check_musa_kernel_launch("quantize_ssm_to_int8_kernel");

  const int64_t conv_blocks_x =
      (pool->conv_elems_per_slot + kConvCopyThreads - 1) / kConvCopyThreads;
  const dim3 conv_grid(static_cast<uint32_t>(conv_blocks_x),
                       static_cast<uint32_t>(pool->num_layers));
  copy_conv_slot_kernel<<<conv_grid, kConvCopyThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(pool->d_conv_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint16_t*>(pool->conv_ckpt.data_ptr()),
      pool->num_ckpt_slots,
      pool->conv_elems_per_slot,
      active_slot_id,
      ckpt_slot,
      /*to_ckpt=*/true);
  check_musa_kernel_launch("copy_conv_slot_kernel(store)");
  LOG_FIRST_N(INFO, 3) << "MUSA GDN Int8 Checkpoint store: active_slot="
                       << active_slot_id << " -> ckpt_slot=" << ckpt_slot
                       << " (logical_id=" << ckpt_logical_id << ")";
}

ScopedGdnSinglepassCheckpointPlan::ScopedGdnSinglepassCheckpointPlan(
    std::vector<GdnSinglepassInteriorTarget> targets)
    : targets_(std::move(targets)), prev_(tls_active_singlepass_plan) {
  tls_active_singlepass_plan = this;
}

ScopedGdnSinglepassCheckpointPlan::~ScopedGdnSinglepassCheckpointPlan() {
  tls_active_singlepass_plan = prev_;
}

bool has_active_gdn_singlepass_checkpoints() {
  return tls_active_singlepass_plan != nullptr &&
         !tls_active_singlepass_plan->targets().empty();
}

std::vector<GdnSinglepassInteriorTarget>
get_active_gdn_singlepass_checkpoints_for_slot(int32_t live_slot_id) {
  std::vector<GdnSinglepassInteriorTarget> out;
  if (tls_active_singlepass_plan == nullptr || live_slot_id <= 0) {
    return out;
  }
  for (const auto& t : tls_active_singlepass_plan->targets()) {
    if (t.live_slot_id == live_slot_id) {
      out.push_back(t);
    }
  }
  std::sort(out.begin(),
            out.end(),
            [](const GdnSinglepassInteriorTarget& a,
               const GdnSinglepassInteriorTarget& b) {
              return a.offset_tokens < b.offset_tokens;
            });
  return out;
}

void store_gdn_int8_checkpoint_from_tensors(
    const torch::Tensor& layer_ssm_cache,
    const torch::Tensor& layer_ssm_state,
    const torch::Tensor& layer_conv_state,
    int32_t ckpt_logical_id) {
  CHECK(layer_ssm_cache.defined()) << "layer_ssm_cache must be defined";
  std::shared_ptr<GdnInt8CheckpointPool> pool;
  int64_t layer_idx = -1;
  {
    std::lock_guard<std::mutex> lock(pool_mutex());
    auto& layer_reg = layer_pool_registry();
    auto it = layer_reg.find(layer_ssm_cache.data_ptr());
    if (it != layer_reg.end()) {
      pool = it->second.first;
      layer_idx = it->second.second;
    }
  }
  CHECK(pool != nullptr && layer_idx >= 0)
      << "GDN Int8 Checkpoint layer pool not found for layer_ssm_cache";
  const int64_t ckpt_slot =
      static_cast<int64_t>(ckpt_logical_id) - pool->num_active_slots;
  CHECK_GE(ckpt_slot, 0);
  CHECK_LT(ckpt_slot, pool->num_ckpt_slots);

  const XLLMOptionalDeviceGuard device_guard(pool->device);
  musaStream_t stream = current_musa_stream(pool->device);

  torch::Tensor ssm_contig = layer_ssm_state;
  if (ssm_contig.scalar_type() != pool->ssm_dtype) {
    ssm_contig = ssm_contig.to(pool->ssm_dtype);
  }
  if (!ssm_contig.is_contiguous()) {
    ssm_contig = ssm_contig.contiguous();
  }
  CHECK_EQ(ssm_contig.dim(), 3);
  CHECK_EQ(ssm_contig.size(0), pool->num_heads);
  CHECK_EQ(ssm_contig.size(1), pool->head_v_dim);
  CHECK_EQ(ssm_contig.size(2), pool->head_k_dim);

  torch::Tensor conv_contig = layer_conv_state;
  if (conv_contig.scalar_type() != pool->conv_dtype) {
    conv_contig = conv_contig.to(pool->conv_dtype);
  }
  if (!conv_contig.is_contiguous()) {
    conv_contig = conv_contig.contiguous();
  }
  CHECK_EQ(conv_contig.dim(), 2);
  CHECK_GT(conv_contig.size(0), 0);
  CHECK_EQ(pool->conv_elems_per_slot % conv_contig.size(0), 0);
  const int64_t src_conv_cols = conv_contig.size(1);
  const int64_t dst_conv_cols = pool->conv_elems_per_slot / conv_contig.size(0);
  CHECK_GT(src_conv_cols, 0);
  CHECK_LE(src_conv_cols, dst_conv_cols);

  const dim3 quant_grid(static_cast<uint32_t>(pool->num_heads), 1u);
  int8_t* qdata = pool->qdata.data_ptr<int8_t>();
  __mt_bfloat16* scales =
      reinterpret_cast<__mt_bfloat16*>(pool->scale.data_ptr());

  if (pool->ssm_dtype == torch::kFloat32) {
    quantize_single_layer_ssm_to_int8_kernel<float>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_contig.data_ptr<float>(),
            qdata,
            scales,
            layer_idx,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kBFloat16) {
    quantize_single_layer_ssm_to_int8_kernel<__mt_bfloat16>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __mt_bfloat16*>(ssm_contig.data_ptr()),
            qdata,
            scales,
            layer_idx,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kFloat16) {
    quantize_single_layer_ssm_to_int8_kernel<__half>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(ssm_contig.data_ptr()),
            qdata,
            scales,
            layer_idx,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            ckpt_slot);
  } else {
    LOG(FATAL) << "Unsupported GDN SSM dtype for Int8 checkpoint: "
               << pool->ssm_dtype;
  }
  check_musa_kernel_launch("quantize_single_layer_ssm_to_int8_kernel");

  const int64_t conv_blocks_x =
      (pool->conv_elems_per_slot + kConvCopyThreads - 1) / kConvCopyThreads;
  const dim3 conv_grid(static_cast<uint32_t>(conv_blocks_x), 1u);
  copy_single_layer_conv_to_ckpt_kernel<<<conv_grid, kConvCopyThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(conv_contig.data_ptr()),
      reinterpret_cast<uint16_t*>(pool->conv_ckpt.data_ptr()),
      layer_idx,
      pool->num_ckpt_slots,
      pool->conv_elems_per_slot,
      src_conv_cols,
      dst_conv_cols,
      ckpt_slot);
  check_musa_kernel_launch("copy_single_layer_conv_to_ckpt_kernel");

  if (layer_idx == 0 && tls_active_singlepass_plan != nullptr) {
    tls_active_singlepass_plan->mark_written(ckpt_logical_id);
  }
  LOG_FIRST_N(INFO, 6)
      << "MUSA GDN Single-Pass Interior Checkpoint store: ckpt_slot="
      << ckpt_slot << " (logical_id=" << ckpt_logical_id << ")";
}

torch::Tensor allocate_gdn_int8_checkpoint_packet(
    const std::vector<::xllm::KVCache>& kv_caches) {
  auto pool = lookup_pool(kv_caches);
  CHECK(pool != nullptr) << "GDN Int8 Checkpoint Pool is not initialized";
  const XLLMOptionalDeviceGuard device_guard(pool->device);
  const int64_t q_bytes =
      pool->num_layers * pool->num_heads * pool->head_v_dim * pool->head_k_dim;
  const int64_t s_bytes =
      pool->num_layers * pool->num_heads * pool->head_k_dim * 2;
  const int64_t c_bytes = pool->num_layers * pool->conv_elems_per_slot * 2;
  return torch::empty(
      {q_bytes + s_bytes + c_bytes},
      torch::TensorOptions().dtype(torch::kUInt8).device(pool->device));
}

torch::Tensor pack_gdn_int8_checkpoint_slot(
    const std::vector<::xllm::KVCache>& kv_caches,
    int32_t ckpt_logical_id) {
  auto pool = lookup_pool(kv_caches);
  CHECK(pool != nullptr) << "GDN Int8 Checkpoint Pool is not initialized";
  const int64_t ckpt_slot =
      static_cast<int64_t>(ckpt_logical_id) - pool->num_active_slots;
  CHECK_GE(ckpt_slot, 0);
  CHECK_LT(ckpt_slot, pool->num_ckpt_slots);
  const XLLMOptionalDeviceGuard device_guard(pool->device);
  torch::Tensor q_part =
      pool->qdata.select(1, ckpt_slot).contiguous().view(torch::kUInt8).view({-1});
  torch::Tensor s_part =
      pool->scale.select(1, ckpt_slot).contiguous().view(torch::kUInt8).view({-1});
  torch::Tensor c_part =
      pool->conv_ckpt.select(1, ckpt_slot).contiguous().view(torch::kUInt8).view({-1});
  return torch::cat({q_part, s_part, c_part}, /*dim=*/0);
}

void unpack_gdn_int8_checkpoint_slot(
    const std::vector<::xllm::KVCache>& kv_caches,
    int32_t ckpt_logical_id,
    const torch::Tensor& packet) {
  auto pool = lookup_pool(kv_caches);
  CHECK(pool != nullptr) << "GDN Int8 Checkpoint Pool is not initialized";
  const int64_t ckpt_slot =
      static_cast<int64_t>(ckpt_logical_id) - pool->num_active_slots;
  CHECK_GE(ckpt_slot, 0);
  CHECK_LT(ckpt_slot, pool->num_ckpt_slots);
  const XLLMOptionalDeviceGuard device_guard(pool->device);
  const int64_t q_bytes =
      pool->num_layers * pool->num_heads * pool->head_v_dim * pool->head_k_dim;
  const int64_t s_bytes =
      pool->num_layers * pool->num_heads * pool->head_k_dim * 2;
  const int64_t c_bytes = pool->num_layers * pool->conv_elems_per_slot * 2;
  CHECK_EQ(packet.numel(), q_bytes + s_bytes + c_bytes);

  torch::Tensor q_dst = pool->qdata.select(1, ckpt_slot);
  torch::Tensor s_dst = pool->scale.select(1, ckpt_slot);
  torch::Tensor c_dst = pool->conv_ckpt.select(1, ckpt_slot);
  q_dst.copy_(
      packet.narrow(0, 0, q_bytes).view(torch::kInt8).view(q_dst.sizes()));
  s_dst.copy_(
      packet.narrow(0, q_bytes, s_bytes)
          .view(pool->scale.scalar_type())
          .view(s_dst.sizes()));
  c_dst.copy_(
      packet.narrow(0, q_bytes + s_bytes, c_bytes)
          .view(pool->conv_dtype)
          .view(c_dst.sizes()));
}

void restore_gdn_int8_checkpoint(std::vector<::xllm::KVCache>& kv_caches,
                                 int32_t active_write_id,
                                 int32_t ckpt_logical_id) {
  auto pool = lookup_pool(kv_caches);
  CHECK(pool != nullptr) << "GDN Int8 Checkpoint Pool is not initialized";
  CHECK_GT(active_write_id, 0);
  CHECK_LT(active_write_id, pool->num_active_slots);
  const int64_t ckpt_slot =
      static_cast<int64_t>(ckpt_logical_id) - pool->num_active_slots;
  CHECK_GE(ckpt_slot, 0);
  CHECK_LT(ckpt_slot, pool->num_ckpt_slots);

  const XLLMOptionalDeviceGuard device_guard(pool->device);
  musaStream_t stream = current_musa_stream(pool->device);

  const dim3 quant_grid(static_cast<uint32_t>(pool->num_heads),
                        static_cast<uint32_t>(pool->num_layers));
  const uint64_t* ssm_ptrs =
      reinterpret_cast<const uint64_t*>(pool->d_ssm_ptrs.data_ptr<int64_t>());
  const int8_t* qdata = pool->qdata.data_ptr<int8_t>();
  const __mt_bfloat16* scales =
      reinterpret_cast<const __mt_bfloat16*>(pool->scale.data_ptr());

  if (pool->ssm_dtype == torch::kFloat32) {
    dequantize_ssm_from_int8_kernel<float>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_write_id,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kBFloat16) {
    dequantize_ssm_from_int8_kernel<__mt_bfloat16>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_write_id,
            ckpt_slot);
  } else if (pool->ssm_dtype == torch::kFloat16) {
    dequantize_ssm_from_int8_kernel<__half>
        <<<quant_grid, kQuantThreads, 0, stream>>>(
            ssm_ptrs,
            qdata,
            scales,
            pool->num_ckpt_slots,
            pool->num_heads,
            pool->head_v_dim,
            pool->head_k_dim,
            active_write_id,
            ckpt_slot);
  } else {
    LOG(FATAL) << "Unsupported GDN SSM dtype for Int8 checkpoint: "
               << pool->ssm_dtype;
  }
  check_musa_kernel_launch("dequantize_ssm_from_int8_kernel");

  const int64_t conv_blocks_x =
      (pool->conv_elems_per_slot + kConvCopyThreads - 1) / kConvCopyThreads;
  const dim3 conv_grid(static_cast<uint32_t>(conv_blocks_x),
                       static_cast<uint32_t>(pool->num_layers));
  copy_conv_slot_kernel<<<conv_grid, kConvCopyThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(pool->d_conv_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint16_t*>(pool->conv_ckpt.data_ptr()),
      pool->num_ckpt_slots,
      pool->conv_elems_per_slot,
      active_write_id,
      ckpt_slot,
      /*to_ckpt=*/false);
  check_musa_kernel_launch("copy_conv_slot_kernel(restore)");
  LOG_FIRST_N(INFO, 3) << "MUSA GDN Int8 Checkpoint restore: ckpt_slot="
                       << ckpt_slot << " (logical_id=" << ckpt_logical_id
                       << ") -> active_slot=" << active_write_id;
}

}  // namespace xllm::kernel::musa
