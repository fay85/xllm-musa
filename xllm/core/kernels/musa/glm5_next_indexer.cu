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

#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "core/kernels/musa/glm5_next_indexer.h"
#include "core/kernels/musa/musa_fp8_quant_utils.cuh"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {
// Every thread owns one channel. Each butterfly reads the previous stage
// before any lane overwrites shared memory.
__device__ float hadamard(float value, float* shared) {
  const int32_t lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  for (int32_t stride = 1; stride < 128; stride *= 2) {
    const float other = shared[lane ^ stride];
    const float next = (lane & stride) ? other - value : value + other;
    __syncthreads();
    shared[lane] = next;
    value = next;
    __syncthreads();
  }
  return static_cast<float>(c10::BFloat16(value * 0.08838834764831845f));
}

__device__ float quant_scale(float value, float* shared) {
  const int32_t lane = threadIdx.x;
  shared[lane] = fabsf(value);
  __syncthreads();
  for (int32_t offset = 64; offset > 0; offset /= 2) {
    if (lane < offset) {
      shared[lane] = fmaxf(shared[lane], shared[lane + offset]);
    }
    __syncthreads();
  }
  return exp2f(ceilf(log2f(fmaxf(shared[0], 1.0e-4f) / 448.0f)));
}

__global__ void update_pools(const c10::BFloat16* keys,
                             const c10::BFloat16* gates,
                             const float* ape,
                             const int64_t* positions,
                             const int32_t* cu,
                             const int32_t* state_ids,
                             const int32_t* read_ids,
                             const int32_t* table,
                             uint8_t* cache,
                             const c10::BFloat16* read_tail,
                             c10::BFloat16* tail,
                             int32_t* rows,
                             int32_t* pool_lengths,
                             int64_t states,
                             int64_t pages,
                             int64_t table_width) {
  const int32_t request = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  const bool valid = state > 0 && state < states;
  __shared__ float shared[128];
  if (valid && source > 0 && source < states && source != state) {
    for (int32_t field = 0; field < 8; ++field) {
      tail[state * 1024 + field * 128 + lane] =
          read_tail[source * 1024 + field * 128 + lane];
    }
  }
  __syncthreads();
  for (int32_t row = cu[request]; row < cu[request + 1]; ++row) {
    const int64_t position = positions[row];
    if (lane == 0) {
      rows[row] = valid && position >= 0 ? request : -1;
      pool_lengths[row] = valid && position >= 0 ? (position + 1) / 4 : 0;
    }
    if (!valid || position < 0) {
      continue;
    }
    if (position == 0) {
      for (int32_t field = 0; field < 8; ++field) {
        tail[state * 1024 + field * 128 + lane] = c10::BFloat16(0.0f);
      }
    }
    const int64_t ring = position % 4;
    tail[state * 1024 + ring * 128 + lane] = keys[row * 128 + lane];
    tail[state * 1024 + 512 + ring * 128 + lane] = gates[row * 128 + lane];
    __syncthreads();
    if (ring != 3) {
      continue;
    }
    const int64_t pool = position / 4;
    const int64_t column = pool / 64;
    const int64_t page =
        column < table_width ? table[request * table_width + column] : -1;
    if (page < 0 || page >= pages) {
      continue;
    }
    float logits[4];
    float maximum = -INFINITY;
    for (int32_t j = 0; j < 4; ++j) {
      logits[j] =
          static_cast<float>(tail[state * 1024 + 512 + j * 128 + lane]) +
          ape[j * 128 + lane];
      maximum = fmaxf(maximum, logits[j]);
    }
    float numerator = 0;
    float denominator = 0;
    for (int32_t j = 0; j < 4; ++j) {
      const float probability = expf(logits[j] - maximum);
      numerator +=
          probability * static_cast<float>(tail[state * 1024 + j * 128 + lane]);
      denominator += probability;
    }
    float value =
        static_cast<float>(c10::BFloat16(__fdiv_rn(numerator, denominator)));
    value = hadamard(value, shared);
    const float scale = quant_scale(value, shared);
    // MATE stores payload and scales in separate planes within each page.
    uint8_t* page_data = cache + page * 64 * 132;
    reinterpret_cast<c10::Float8_e4m3fn*>(page_data)[(pool % 64) * 128 + lane] =
        QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
    if (lane == 0) {
      reinterpret_cast<float*>(page_data + 64 * 128)[pool % 64] = scale;
    }
    __syncthreads();
  }
}

__global__ void update_pools_parallel(const c10::BFloat16* keys,
                                      const c10::BFloat16* gates,
                                      const float* ape,
                                      const int64_t* positions,
                                      const int32_t* cu,
                                      const int32_t* state_ids,
                                      const int32_t* read_ids,
                                      const int32_t* table,
                                      uint8_t* cache,
                                      const c10::BFloat16* read_tail,
                                      int32_t* rows,
                                      int32_t* pool_lengths,
                                      int64_t states,
                                      int64_t pages,
                                      int64_t table_width) {
  const int32_t row = blockIdx.x;
  const int32_t lane = threadIdx.x;
  int32_t request = 0;
  while (cu[request + 1] <= row) {
    ++request;
  }
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  const int64_t position = positions[row];
  const bool valid = state > 0 && state < states && position >= 0;
  if (lane == 0) {
    rows[row] = valid ? request : -1;
    pool_lengths[row] = valid ? (position + 1) / 4 : 0;
  }
  if (!valid || position % 4 != 3) {
    return;
  }
  __shared__ float shared[128];
  float pool_keys[4];
  float pool_gates[4];
  for (int32_t j = 0; j < 4; ++j) {
    const int64_t read_state = source > 0 && source < states ? source : state;
    pool_keys[j] =
        static_cast<float>(read_tail[read_state * 1024 + j * 128 + lane]);
    pool_gates[j] =
        static_cast<float>(read_tail[read_state * 1024 + 512 + j * 128 + lane]);
    // Find the most recent write to this ring entry. Normally this reads at
    // most four positions; padding is skipped just as in the decode kernel.
    for (int32_t input_row = row; input_row >= cu[request]; --input_row) {
      const int64_t input_position = positions[input_row];
      if (input_position >= 0 && input_position % 4 == j) {
        pool_keys[j] = static_cast<float>(keys[input_row * 128 + lane]);
        pool_gates[j] = static_cast<float>(gates[input_row * 128 + lane]);
        break;
      }
      if (input_position == 0) {
        pool_keys[j] = 0;
        pool_gates[j] = 0;
        break;
      }
    }
  }
  const int64_t pool = position / 4;
  const int64_t column = pool / 64;
  const int64_t page =
      column < table_width ? table[request * table_width + column] : -1;
  if (page < 0 || page >= pages) {
    return;
  }
  float logits[4];
  float maximum = -INFINITY;
  for (int32_t j = 0; j < 4; ++j) {
    logits[j] = pool_gates[j] + ape[j * 128 + lane];
    maximum = fmaxf(maximum, logits[j]);
  }
  float numerator = 0;
  float denominator = 0;
  for (int32_t j = 0; j < 4; ++j) {
    const float probability = expf(logits[j] - maximum);
    numerator += probability * pool_keys[j];
    denominator += probability;
  }
  float value =
      static_cast<float>(c10::BFloat16(__fdiv_rn(numerator, denominator)));
  value = hadamard(value, shared);
  const float scale = quant_scale(value, shared);
  // MATE stores payload and scales in separate planes within each page.
  uint8_t* page_data = cache + page * 64 * 132;
  reinterpret_cast<c10::Float8_e4m3fn*>(page_data)[(pool % 64) * 128 + lane] =
      QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
  if (lane == 0) {
    reinterpret_cast<float*>(page_data + 64 * 128)[pool % 64] = scale;
  }
}

// Finalize only the request's four-entry ring after all pools read its old
// tail.
__global__ void update_pool_tails(const c10::BFloat16* keys,
                                  const c10::BFloat16* gates,
                                  const int64_t* positions,
                                  const int32_t* cu,
                                  const int32_t* state_ids,
                                  const int32_t* read_ids,
                                  const c10::BFloat16* snapshot,
                                  c10::BFloat16* tail,
                                  int64_t states) {
  const int32_t request = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int64_t state = state_ids[request];
  const int64_t source = read_ids[request];
  if (state <= 0 || state >= states) {
    return;
  }
  for (int32_t field = 0; field < 8; ++field) {
    if (source > 0 && source < states && source != state) {
      tail[state * 1024 + field * 128 + lane] =
          snapshot[source * 1024 + field * 128 + lane];
    }
  }
  // Keep the last write to each ring entry, including across padded rows.
  int32_t written = 0;
  for (int32_t row = cu[request + 1] - 1; row >= cu[request]; --row) {
    const int64_t position = positions[row];
    if (position < 0) {
      continue;
    }
    const int64_t ring = position % 4;
    if (!(written & (1 << ring))) {
      tail[state * 1024 + ring * 128 + lane] = keys[row * 128 + lane];
      tail[state * 1024 + 512 + ring * 128 + lane] = gates[row * 128 + lane];
      written |= 1 << ring;
    }
    if (position == 0) {
      for (int32_t j = 0; j < 4; ++j) {
        if (!(written & (1 << j))) {
          tail[state * 1024 + j * 128 + lane] = c10::BFloat16(0.0f);
          tail[state * 1024 + 512 + j * 128 + lane] = c10::BFloat16(0.0f);
        }
      }
      break;
    }
    if (written == 15) {
      break;
    }
  }
}

__global__ void quantize_query(const c10::BFloat16* query,
                               const float* weights,
                               c10::Float8_e4m3fn* output,
                               float* scaled_weights) {
  const int64_t row = blockIdx.x;
  const int32_t lane = threadIdx.x;
  __shared__ float shared[128];
  const float value =
      hadamard(static_cast<float>(query[row * 128 + lane]), shared);
  const float scale = quant_scale(value, shared);
  output[row * 128 + lane] =
      QuantConverter<c10::Float8_e4m3fn>::convert(__fdiv_rn(value, scale));
  if (lane == 0) {
    scaled_weights[row] = weights[row] * scale * (1.0f / 64.0f);
  }
}

__global__ void expand_pools(const int64_t* selected,
                             const int64_t* positions,
                             const int32_t* rows,
                             const int32_t* pool_lengths,
                             int32_t* tokens,
                             int32_t* lengths,
                             int64_t selected_width,
                             int64_t width,
                             int64_t token_topk) {
  const int64_t row = blockIdx.x;
  const int64_t visible = rows[row] >= 0 ? positions[row] + 1 : 0;
  const int64_t tail = visible % 4;
  const int64_t end = visible <= token_topk ? visible : token_topk + tail;
  if (threadIdx.x == 0) {
    lengths[row] = static_cast<int32_t>(end);
  }
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    int32_t token = -1;
    if (col < end && visible <= token_topk) {
      token = static_cast<int32_t>(col);
    } else if (visible > token_topk && col < token_topk &&
               col / 4 < selected_width) {
      const int64_t pool = selected[row * selected_width + col / 4];
      if (pool >= 0 && pool < pool_lengths[row]) {
        token = static_cast<int32_t>(pool * 4 + col % 4);
      }
    } else if (col >= token_topk && col < end) {
      token = static_cast<int32_t>((visible / 4) * 4 + col - token_topk);
    }
    tokens[row * width + col] = token;
  }
}

void check_tensor(const torch::Tensor& value,
                  const torch::Tensor& anchor,
                  torch::ScalarType dtype) {
  CHECK(value.defined());
  CHECK_EQ(value.device(), anchor.device());
  CHECK_EQ(value.scalar_type(), dtype);
  CHECK(value.is_contiguous());
}

// Keep the foreign allocator's tensor alive until the Torch consumer finishes.
// The existing FFI stream guard orders its producer before Torch reads it.
torch::Tensor own_logits(ffi::Tensor tensor,
                         const torch::TensorOptions& options) {
  std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<int64_t> strides;
  for (int32_t dim = 0; dim < tensor.ndim(); ++dim) {
    strides.push_back(tensor.stride(dim));
  }
  void* data = tensor.data_ptr();
  return torch::from_blob(
      data,
      shape,
      strides,
      [owner = std::move(tensor)](void*) {},
      options.dtype(torch::kFloat32));
}
}  // namespace

Glm5NextIndexerSelection glm5_next_indexer(const torch::Tensor& query,
                                           const torch::Tensor& raw_key,
                                           const torch::Tensor& gate,
                                           const torch::Tensor& weights,
                                           const torch::Tensor& ape,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& cu_seqlens,
                                           const torch::Tensor& state_ids,
                                           const torch::Tensor& read_state_ids,
                                           const torch::Tensor& block_table,
                                           torch::Tensor& cache,
                                           torch::Tensor& tail,
                                           int64_t max_context,
                                           int64_t token_topk) {
  CHECK(query.device().is_privateuseone());
  XLLMDeviceGuard device_guard(query.device());
  check_tensor(query, query, torch::kBFloat16);
  check_tensor(raw_key, query, torch::kBFloat16);
  check_tensor(gate, query, torch::kBFloat16);
  check_tensor(weights, query, torch::kFloat32);
  check_tensor(ape, query, torch::kFloat32);
  check_tensor(positions, query, torch::kInt64);
  check_tensor(cu_seqlens, query, torch::kInt32);
  check_tensor(state_ids, query, torch::kInt32);
  check_tensor(read_state_ids, query, torch::kInt32);
  check_tensor(block_table, query, torch::kInt32);
  check_tensor(cache, query, torch::kUInt8);
  check_tensor(tail, query, torch::kBFloat16);
  CHECK_EQ(query.dim(), 3);
  const int64_t tokens = query.size(0);
  CHECK_EQ(query.sizes(), torch::IntArrayRef({tokens, 32, 128}));
  CHECK_EQ(raw_key.sizes(), torch::IntArrayRef({tokens, 128}));
  CHECK_EQ(gate.sizes(), raw_key.sizes());
  CHECK_EQ(weights.sizes(), torch::IntArrayRef({tokens, 32}));
  CHECK_EQ(ape.sizes(), torch::IntArrayRef({4, 128}));
  CHECK_EQ(positions.sizes(), torch::IntArrayRef({tokens}));
  CHECK_EQ(block_table.dim(), 2);
  const int64_t batch = block_table.size(0);
  CHECK_EQ(cu_seqlens.sizes(), torch::IntArrayRef({batch + 1}));
  CHECK_EQ(state_ids.sizes(), torch::IntArrayRef({batch}));
  CHECK_EQ(read_state_ids.sizes(), state_ids.sizes());
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.sizes().slice(1), torch::IntArrayRef({64, 1, 132}));
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.sizes().slice(1), torch::IntArrayRef({2, 4, 128}));
  CHECK_GT(token_topk, 0);
  CHECK_EQ(token_topk % 4, 0);
  CHECK_LE(token_topk, 2048);
  CHECK_GT(max_context, 0);
  CHECK_LE(max_context, block_table.size(1) * 256);
  CHECK_LE(max_context, std::numeric_limits<int32_t>::max());
  const int64_t width = ((token_topk + 3 + 127) / 128) * 128;
  auto ints = query.options().dtype(torch::kInt32);
  Glm5NextIndexerSelection result{torch::empty({tokens, width}, ints),
                                  torch::empty({tokens}, ints),
                                  torch::empty({tokens}, ints)};
  if (tokens == 0) {
    return result;
  }
  musaDeviceProp properties{};
  CHECK_EQ(musaGetDeviceProperties(&properties, query.device().index()),
           musaSuccess);
  CHECK_EQ(properties.major * 10 + properties.minor, 31);
  // Resolve both scoring functions before any persistent state is mutated.
  auto metadata_fn =
      get_function("glm_kpool_metadata_v1", "get_paged_mqa_logits_metadata");
  auto score_fn = get_function("glm_kpool_score_v1", "fp8_paged_mqa_logits");
  const auto stream = current_musa_stream(query.device());
  auto pool_lengths = torch::empty({tokens}, ints);
  auto snapshot =
      read_state_ids.data_ptr() != state_ids.data_ptr() ? tail.clone() : tail;
  if (tokens > batch) {
    update_pools_parallel<<<tokens, 128, 0, stream>>>(
        raw_key.data_ptr<c10::BFloat16>(),
        gate.data_ptr<c10::BFloat16>(),
        ape.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        state_ids.data_ptr<int32_t>(),
        read_state_ids.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        cache.data_ptr<uint8_t>(),
        snapshot.data_ptr<c10::BFloat16>(),
        result.rows.data_ptr<int32_t>(),
        pool_lengths.data_ptr<int32_t>(),
        tail.size(0),
        cache.size(0),
        block_table.size(1));
    update_pool_tails<<<batch, 128, 0, stream>>>(
        raw_key.data_ptr<c10::BFloat16>(),
        gate.data_ptr<c10::BFloat16>(),
        positions.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        state_ids.data_ptr<int32_t>(),
        read_state_ids.data_ptr<int32_t>(),
        snapshot.data_ptr<c10::BFloat16>(),
        tail.data_ptr<c10::BFloat16>(),
        tail.size(0));
  } else {
    update_pools<<<batch, 128, 0, stream>>>(raw_key.data_ptr<c10::BFloat16>(),
                                            gate.data_ptr<c10::BFloat16>(),
                                            ape.data_ptr<float>(),
                                            positions.data_ptr<int64_t>(),
                                            cu_seqlens.data_ptr<int32_t>(),
                                            state_ids.data_ptr<int32_t>(),
                                            read_state_ids.data_ptr<int32_t>(),
                                            block_table.data_ptr<int32_t>(),
                                            cache.data_ptr<uint8_t>(),
                                            snapshot.data_ptr<c10::BFloat16>(),
                                            tail.data_ptr<c10::BFloat16>(),
                                            result.rows.data_ptr<int32_t>(),
                                            pool_lengths.data_ptr<int32_t>(),
                                            tail.size(0),
                                            cache.size(0),
                                            block_table.size(1));
  }
  check_musa_kernel_launch("GLM IndexPool update");
  const int64_t pools = (max_context + 3) / 4;
  const int64_t count = std::min<int64_t>(token_topk / 4, pools);
  auto selected =
      torch::full({tokens, count}, -1, query.options().dtype(torch::kInt64));
  if (max_context > token_topk) {
    auto quant = torch::empty({tokens, 1, 32, 128},
                              query.options().dtype(torch::kFloat8_e4m3fn));
    auto scaled_weights = torch::empty_like(weights);
    quantize_query<<<tokens * 32, 128, 0, stream>>>(
        query.data_ptr<c10::BFloat16>(),
        weights.data_ptr<float>(),
        quant.data_ptr<c10::Float8_e4m3fn>(),
        scaled_weights.data_ptr<float>());
    check_musa_kernel_launch("GLM IndexPool query quantization");
    // At most 16 MiB of FP32 logits per chunk, and <=128 query rows to match
    // the exported scheduler. Main history is never expanded per head.
    const int64_t chunk = std::max<int64_t>(
        1, std::min<int64_t>(128, (16 * 1024 * 1024) / (pools * 4)));
    auto pool_ids = torch::arange(pools, ints).unsqueeze(0);
    for (int64_t first = 0; first < tokens; first += chunk) {
      const int64_t size = std::min<int64_t>(chunk, tokens - first);
      auto lens = pool_lengths.narrow(0, first, size);
      auto table_rows =
          result.rows.narrow(0, first, size).clamp_min(0).to(torch::kInt64);
      auto table = block_table.index_select(0, table_rows).contiguous();
      auto scheduling =
          torch::empty({properties.multiProcessorCount + 1, 2}, ints);
      ffi::Tensor ffi_logits;
      {
        TvmffiStreamGuard guard(query.device());
        metadata_fn(
            to_ffi_tensor(lens), int64_t{64}, to_ffi_tensor(scheduling));
        ffi_logits =
            score_fn(to_ffi_tensor(quant.narrow(0, first, size)),
                     to_ffi_tensor(cache),
                     to_ffi_tensor(scaled_weights.narrow(0, first, size)),
                     to_ffi_tensor(lens),
                     to_ffi_tensor(table),
                     to_ffi_tensor(scheduling),
                     pools,
                     true)
                .cast<ffi::Tensor>();
      }
      auto scores = own_logits(std::move(ffi_logits), query.options())
                        .narrow(1, 0, pools);
      scores.masked_fill_(pool_ids.ge(lens.unsqueeze(1)), -INFINITY);
      selected.narrow(0, first, size)
          .copy_(std::get<1>(scores.topk(count, -1, true, false)));
    }
  }
  expand_pools<<<tokens, 256, 0, stream>>>(selected.data_ptr<int64_t>(),
                                           positions.data_ptr<int64_t>(),
                                           result.rows.data_ptr<int32_t>(),
                                           pool_lengths.data_ptr<int32_t>(),
                                           result.tokens.data_ptr<int32_t>(),
                                           result.lengths.data_ptr<int32_t>(),
                                           count,
                                           width,
                                           token_topk);
  check_musa_kernel_launch("GLM IndexPool candidate expansion");
  return result;
}
}  // namespace xllm::kernel::musa
