/* Copyright 2026 The xLLM Authors.

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

#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_kernel_utils.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kDecodeAttentionBlockSize = 256;
constexpr int32_t kMateCacheTransferBlockSize = 256;
constexpr int32_t kMusaWarpSize = 32;
constexpr int64_t kWarpFusedSafeMaxTotalLen = 512;

template <typename T>
constexpr T ceil_div(T value, T divisor) {
  return (value + divisor - 1) / divisor;
}

musaStream_t current_dsv4_cache_stream() {
  return c10::cuda::getCurrentCUDAStream();
}

void check_musa_runtime(musaError_t status) {
  CHECK(status == musaSuccess)
      << "DeepSeek-V4 MUSA runtime call failed: " << musaGetErrorString(status);
}

// Bridge the worker2 call sites to the target tree's graph-aware stream
// guard. The guard establishes current->FFI ordering on construction and
// FFI->current ordering on destruction when the default MUSA stream needs the
// pooled FFI stream. During graph capture the operator is bound directly to
// the current capture stream.
thread_local std::optional<TvmffiStreamGuard> dsv4_tvmffi_stream_guard;

void bind_musa_tvmffi_current_stream(const torch::Device& device) {
  bind_tvmffi_stream(device);
}

void bind_musa_tvmffi_stream(const torch::Device& device) {
  CHECK(!dsv4_tvmffi_stream_guard.has_value());
  dsv4_tvmffi_stream_guard.emplace(device);
}

void wait_musa_ffi_stream_for_current(const torch::Device&) {}

void wait_current_musa_stream_for_ffi(const torch::Device&) {
  CHECK(dsv4_tvmffi_stream_guard.has_value());
  dsv4_tvmffi_stream_guard.reset();
}

// The local sparse-prefill AOT is validated on the Torch compatibility
// stream used by its metadata and KV producers.  Preserve a null handle here:
// on MUSA it denotes the valid default stream, while the generic graph-aware
// helper intentionally replaces it with a pooled stream and therefore needs
// per-call event handoffs.  This binding is private to local prefill; decode
// paths retain their existing stream policy.
void bind_dsv4_local_prefill_tvmffi_current_stream(
    const torch::Device& device) {
  void* const stream = reinterpret_cast<void*>(current_dsv4_cache_stream());
  constexpr int32_t kDlCuda = 2;
  constexpr int32_t kDlExtDev = 12;
  for (const int32_t device_type : {kDlCuda, kDlExtDev}) {
    void* original_stream = nullptr;
    const int rc = TVMFFIEnvSetStream(
        device_type, device.index(), stream, &original_stream);
    CHECK_EQ(rc, 0) << "failed to bind local prefill FFI stream, dev_type="
                    << device_type << ", device=" << device.index();
  }
}

bool dsv4_paged_decode_sync_enabled() { return false; }

bool dsv4_paged_decode_warp_fused_enabled() { return true; }

bool dsv4_paged_decode_precompute_probs_enabled() { return true; }

bool dsv4_mate_flashmla_enabled() { return true; }

std::string dsv4_mate_flashmla_uri() { return "mla_ops"; }

bool dsv4_mate_fused_dsa_enabled() { return true; }

bool dsv4_mate_fused_dsa_h8_enabled() { return true; }

std::string dsv4_mate_fused_dsa_uri(int64_t heads) {
  return "dsa_model1_h" + std::to_string(heads) + "_topk512_extra128_bs128";
}

std::string dsv4_mate_prefill_uri(int64_t heads,
                                  int64_t compiled_compressed_kv_len) {
  std::string uri = "dsa_model1_prefill_h" + std::to_string(heads) +
                    "_topk512_pack" + std::to_string(64 / heads);
  if (compiled_compressed_kv_len != 512) {
    uri += "_c" + std::to_string(compiled_compressed_kv_len);
  }
  return uri;
}

std::string dsv4_mate_c4_prefill_uri(int64_t heads) {
  return "dsa_model1_c4_prefill_h" + std::to_string(heads) + "_topk640";
}

int64_t dsv4_mate_c4_prefill_h8_max_seq_len() { return 512; }

bool dsv4_mate_c4_prefill_h8_use_h16() { return true; }

int64_t dsv4_mate_prefill_override_compressed_kv_len() { return 512; }

int64_t dsv4_mate_prefill_max_seq_len() { return 8192; }

int dsv4_mate_prefill_redundant_runs() { return 1; }

std::string dsv4_mate_metadata_uri() { return "mla_ops"; }

bool dsv4_c128_device_metadata_enabled() { return true; }

std::string dsv4_tilelang_runtime_library() {
  return "/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtvm.so";
}

std::string dsv4_tilelang_library() {
  return "/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so";
}

ffi::Optional<ffi::Function> try_get_dsv4_mate_flashmla_function(
    bool direct_mla,
    bool native_leftpad,
    std::string* reason) {
  const std::string uri = dsv4_mate_flashmla_uri();
  const std::string so_path = path_to_uri_so_lib(uri);
  std::ifstream so_file(so_path);
  if (!so_file.good()) {
    if (reason != nullptr) {
      *reason = "Mate FlashMLA module is missing: " + so_path;
    }
    return ffi::Optional<ffi::Function>();
  }

  try {
    const char* function_name =
        native_leftpad
            ? "mla_leftpad"
            : (direct_mla ? "mla" : "dispatch_mla_impl_for_fa_interface");
    auto func = get_module(uri)->GetFunction(function_name);
    if (!func.defined()) {
      if (reason != nullptr) {
        *reason = std::string(function_name) + " is not exported by " + so_path;
      }
      return ffi::Optional<ffi::Function>();
    }
    return func;
  } catch (const std::exception& ex) {
    if (reason != nullptr) {
      *reason = "failed to load Mate FlashMLA: " + std::string(ex.what());
    }
    return ffi::Optional<ffi::Function>();
  }
}

bool dsv4_mate_flashmla_direct_enabled() { return true; }

bool dsv4_mate_flashmla_zero_copy_enabled() { return false; }

bool dsv4_mate_flashmla_graph_page_aligned_zero_copy_enabled() { return false; }

bool dsv4_mate_flashmla_graph_native_leftpad_enabled() { return false; }

bool dsv4_mate_flashmla_fuse_compact_query_enabled() { return true; }

int64_t dsv4_mate_flashmla_max_kv_len() { return 4096; }

int64_t dsv4_mate_flashmla_min_kv_len() { return 192; }

struct Dsv4MateFlashMlaWorkspace {
  c10::DeviceIndex device = -1;
  torch::ScalarType dtype = torch::kFloat32;
  int64_t num_heads = 0;
  int64_t page_capacity = 0;
  torch::Tensor page_table;
  torch::Tensor cache_seqlens;
  torch::Tensor cache_leftpad;
  torch::Tensor q_latent;
  torch::Tensor q_rope;
  torch::Tensor compact_cache;
  torch::Tensor output;
  torch::Tensor softmax_lse;
};

Dsv4MateFlashMlaWorkspace& dsv4_mate_flashmla_workspace(const torch::Tensor& q,
                                                        int64_t mate_pages) {
  thread_local Dsv4MateFlashMlaWorkspace workspace;
  const bool shape_changed = workspace.device != q.device().index() ||
                             workspace.dtype != q.scalar_type() ||
                             workspace.num_heads != q.size(1);
  if (shape_changed) {
    workspace.device = q.device().index();
    workspace.dtype = q.scalar_type();
    workspace.num_heads = q.size(1);
    workspace.q_latent = torch::empty({1, 1, q.size(1), 512}, q.options());
    workspace.q_rope = torch::empty({1, 1, q.size(1), 64}, q.options());
    workspace.output = torch::empty_like(workspace.q_latent);
    workspace.softmax_lse = torch::empty(
        {1, 1, q.size(1)},
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    workspace.cache_seqlens = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    workspace.cache_leftpad = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  }
  if (workspace.page_capacity < mate_pages || shape_changed) {
    workspace.page_capacity = mate_pages;
    workspace.page_table = torch::empty(
        {1, mate_pages},
        torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    workspace.page_table.copy_(
        torch::arange(
            mate_pages,
            torch::TensorOptions().dtype(torch::kInt32).device(q.device()))
            .view({1, mate_pages}));
    workspace.compact_cache =
        torch::empty({mate_pages, 64, 1, 576}, q.options());
  }
  return workspace;
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_compact_dynamic_mate_window_kernel(
        const scalar_t* __restrict__ source,
        scalar_t* __restrict__ destination,
        const int32_t* __restrict__ block_table,
        const int32_t* __restrict__ dynamic_kv_seq_lens,
        const scalar_t* __restrict__ q,
        scalar_t* __restrict__ q_latent,
        scalar_t* __restrict__ q_rope,
        int32_t* __restrict__ cache_seqlens,
        int64_t block_table_sequence_index,
        int64_t dynamic_sequence_index,
        int64_t window_capacity,
        int64_t source_page_size,
        int64_t source_num_blocks,
        int64_t max_blocks_per_seq,
        int64_t output_rows,
        int64_t row_width,
        int64_t num_heads,
        int32_t prepare_query) {
  const int64_t output_row = blockIdx.x;
  if (output_row >= output_rows) {
    return;
  }

  const int64_t raw_kv_len =
      static_cast<int64_t>(dynamic_kv_seq_lens[dynamic_sequence_index]);
  const int64_t cache_capacity = max_blocks_per_seq * source_page_size;
  const int64_t kv_len =
      raw_kv_len > 0 ? min(raw_kv_len, cache_capacity) : int64_t{0};
  const int64_t active_rows = min(kv_len, window_capacity);
  const int64_t window_start = kv_len - active_rows;

  // The compact cache is consumed by FlashMLA immediately after this kernel,
  // so the same completion dependency can also publish its query views and
  // active sequence length. This removes one graph node per dense-attention
  // layer without changing the cache materialization semantics.
  if (prepare_query != 0 && output_row < num_heads) {
    for (int64_t dim = threadIdx.x; dim < 512; dim += blockDim.x) {
      q_latent[output_row * 512 + dim] =
          dim < 448 ? q[output_row * 512 + dim] : static_cast<scalar_t>(0);
    }
    for (int64_t dim = threadIdx.x; dim < 64; dim += blockDim.x) {
      q_rope[output_row * 64 + dim] = q[output_row * 512 + 448 + dim];
    }
  }
  if (prepare_query != 0 && output_row == 0 && threadIdx.x == 0) {
    cache_seqlens[0] = static_cast<int32_t>(active_rows);
  }

  int64_t source_row = -1;
  if (output_row < active_rows) {
    const int64_t logical_row = window_start + output_row;
    const int64_t logical_page = logical_row / source_page_size;
    const int64_t page_offset = logical_row % source_page_size;
    const int64_t physical_page = static_cast<int64_t>(
        block_table[block_table_sequence_index * max_blocks_per_seq +
                    logical_page]);
    if (physical_page >= 0 && physical_page < source_num_blocks) {
      source_row = physical_page * source_page_size + page_offset;
    }
  }

  scalar_t* destination_row = destination + output_row * row_width;
  constexpr int64_t kVectorBytes = sizeof(uint4);
  const int64_t row_bytes = row_width * sizeof(scalar_t);
  if (row_bytes % kVectorBytes == 0) {
    auto* destination_vector = reinterpret_cast<uint4*>(destination_row);
    const int64_t vectors_per_row = row_bytes / kVectorBytes;
    if (source_row < 0) {
      for (int64_t vector = threadIdx.x; vector < vectors_per_row;
           vector += blockDim.x) {
        destination_vector[vector] = make_uint4(0, 0, 0, 0);
      }
      return;
    }
    const auto* source_vector =
        reinterpret_cast<const uint4*>(source + source_row * row_width);
    for (int64_t vector = threadIdx.x; vector < vectors_per_row;
         vector += blockDim.x) {
      destination_vector[vector] = source_vector[vector];
    }
    return;
  }
  if (source_row < 0) {
    for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
      destination_row[col] = static_cast<scalar_t>(0);
    }
    return;
  }
  const scalar_t* source_row_ptr = source + source_row * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    destination_row[col] = source_row_ptr[col];
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_prepare_dynamic_flashmla_query_kernel(
        const scalar_t* __restrict__ q,
        scalar_t* __restrict__ q_latent,
        scalar_t* __restrict__ q_rope,
        int32_t* __restrict__ cache_seqlens,
        const int32_t* __restrict__ dynamic_kv_seq_lens,
        int64_t sequence_index,
        int64_t window_capacity,
        int64_t num_heads) {
  const int64_t head = blockIdx.x;
  if (head < num_heads) {
    for (int64_t dim = threadIdx.x; dim < 512; dim += blockDim.x) {
      q_latent[head * 512 + dim] =
          dim < 448 ? q[head * 512 + dim] : static_cast<scalar_t>(0);
    }
    for (int64_t dim = threadIdx.x; dim < 64; dim += blockDim.x) {
      q_rope[head * 64 + dim] = q[head * 512 + 448 + dim];
    }
  }
  if (head == 0 && threadIdx.x == 0) {
    const int64_t raw_kv_len =
        static_cast<int64_t>(dynamic_kv_seq_lens[sequence_index]);
    cache_seqlens[0] = static_cast<int32_t>(
        raw_kv_len > 0 ? min(raw_kv_len, window_capacity) : int64_t{0});
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_prepare_dynamic_flashmla_page_table_kernel(
        scalar_t* __restrict__ source_cache,
        const int32_t* __restrict__ source_block_table,
        const int32_t* __restrict__ dynamic_kv_seq_lens,
        int32_t* __restrict__ mate_page_table,
        int32_t* __restrict__ cache_seqlens,
        int32_t* __restrict__ cache_leftpad,
        int64_t sequence_index,
        int64_t window_capacity,
        int64_t source_page_size,
        int64_t source_num_blocks,
        int64_t max_blocks_per_seq,
        int64_t mate_page_size,
        int64_t mate_page_capacity,
        int64_t row_width) {
  const int64_t raw_kv_len =
      static_cast<int64_t>(dynamic_kv_seq_lens[sequence_index]);
  const int64_t cache_capacity = max_blocks_per_seq * source_page_size;
  const int64_t kv_len =
      raw_kv_len > 0 ? min(raw_kv_len, cache_capacity) : int64_t{0};
  const int64_t active_rows = min(kv_len, window_capacity);
  const int64_t window_start = kv_len - active_rows;
  const int64_t first_mate_page = window_start / mate_page_size;
  const int64_t leading_rows = window_start % mate_page_size;
  const int64_t visible_rows = active_rows + leading_rows;
  const int64_t visible_pages =
      (visible_rows + mate_page_size - 1) / mate_page_size;
  const int64_t source_mate_pages_per_block = source_page_size / mate_page_size;

  for (int64_t output_page = threadIdx.x; output_page < mate_page_capacity;
       output_page += blockDim.x) {
    const int64_t visible_page =
        visible_pages > 0 ? min(output_page, visible_pages - 1) : int64_t{0};
    const int64_t logical_mate_page = first_mate_page + visible_page;
    const int64_t logical_token = logical_mate_page * mate_page_size;
    const int64_t logical_source_page = logical_token / source_page_size;
    const int64_t source_mate_page_offset =
        (logical_token % source_page_size) / mate_page_size;
    int32_t physical_mate_page = 0;
    if (logical_source_page >= 0 && logical_source_page < max_blocks_per_seq) {
      const int32_t physical_source_page =
          source_block_table[logical_source_page];
      if (physical_source_page >= 0 &&
          physical_source_page < source_num_blocks) {
        physical_mate_page =
            static_cast<int32_t>(static_cast<int64_t>(physical_source_page) *
                                     source_mate_pages_per_block +
                                 source_mate_page_offset);
      }
    }
    mate_page_table[output_page] = physical_mate_page;
  }
  if (threadIdx.x == 0) {
    // Expose the page-aligned prefix and publish the number of older rows in
    // its first page.  Legacy MLA ignores cache_leftpad and uses the xLLM-side
    // correction; the native-leftpad Mate entry masks them inside softmax.
    cache_seqlens[0] = static_cast<int32_t>(visible_rows);
    cache_leftpad[0] = static_cast<int32_t>(leading_rows);
  }

  // The compact bridge zero-fills the inactive tail of the last 64-row Mate
  // page.  A direct view must preserve that contract: the non-ASM FlashMLA
  // implementation can prefetch the whole last page before applying the
  // sequence-length mask, so exposing uninitialized xLLM cache rows can
  // contaminate an otherwise valid prefix with NaNs.  Clear only the partial
  // tail (at most 63 rows); future decode steps write their row before the
  // attention read, so this does not alter any live KV entry.
  const int64_t tail_rows =
      visible_rows > 0 && visible_rows % mate_page_size != 0
          ? mate_page_size - visible_rows % mate_page_size
          : int64_t{0};
  const int64_t tail_elements = tail_rows * row_width;
  for (int64_t element = threadIdx.x; element < tail_elements;
       element += blockDim.x) {
    const int64_t logical_token = kv_len + element / row_width;
    const int64_t logical_source_page = logical_token / source_page_size;
    const int64_t source_offset = logical_token % source_page_size;
    if (logical_source_page < 0 || logical_source_page >= max_blocks_per_seq) {
      continue;
    }
    const int32_t physical_source_page =
        source_block_table[logical_source_page];
    if (physical_source_page < 0 || physical_source_page >= source_num_blocks) {
      continue;
    }
    source_cache[(static_cast<int64_t>(physical_source_page) *
                      source_page_size +
                  source_offset) *
                     row_width +
                 element % row_width] = static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_apply_flashmla_attention_sink_kernel(
        scalar_t* __restrict__ output,
        const float* __restrict__ softmax_lse,
        const float* __restrict__ attn_sink,
        float* __restrict__ retained_mass_output,
        int64_t num_heads,
        int64_t head_dim) {
  const int64_t head = blockIdx.x;
  if (head >= num_heads) {
    return;
  }
  const float retained_mass =
      1.0f / (1.0f + expf(attn_sink[head] - softmax_lse[head]));
  if (retained_mass_output != nullptr) {
    if (threadIdx.x == 0) {
      retained_mass_output[head] = retained_mass;
    }
    return;
  }
  scalar_t* head_output = output + head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    head_output[dim] = static_cast<scalar_t>(
        static_cast<float>(head_output[dim]) * retained_mass);
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_compact_mate_cache_kernel(const scalar_t* __restrict__ source,
                                   scalar_t* __restrict__ destination,
                                   const int32_t* __restrict__ block_table,
                                   int64_t sequence_index,
                                   int64_t kv_len,
                                   int64_t source_page_size,
                                   int64_t source_num_blocks,
                                   int64_t max_blocks_per_seq,
                                   int64_t output_rows,
                                   int64_t row_width) {
  const int64_t row = blockIdx.x;
  if (row >= output_rows) {
    return;
  }

  constexpr int64_t kMatePageSize = 64;
  const int64_t used_pages = (kv_len + kMatePageSize - 1) / kMatePageSize;
  int64_t logical_token = row;
  bool copy_row = logical_token < kv_len;
  // Mate's non-ASM kernel may prefetch one page beyond the active prefix. For
  // an exact page boundary, preserve the old bridge behavior by duplicating
  // the last active page; for a partial page, all rows after kv_len stay zero.
  if (!copy_row && kv_len % kMatePageSize == 0 &&
      row / kMatePageSize == used_pages) {
    logical_token = kv_len - kMatePageSize + row % kMatePageSize;
    copy_row = true;
  }

  int64_t source_row = -1;
  if (copy_row) {
    const int64_t source_page = logical_token / source_page_size;
    const int64_t source_offset = logical_token % source_page_size;
    if (source_page < max_blocks_per_seq && source_offset >= 0 &&
        source_offset < source_page_size) {
      const int64_t block_id = static_cast<int64_t>(
          block_table[sequence_index * max_blocks_per_seq + source_page]);
      if (block_id >= 0 && block_id < source_num_blocks) {
        source_row = block_id * source_page_size + source_offset;
      }
    }
  }

  scalar_t* destination_row = destination + row * row_width;
  if (source_row < 0) {
    for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
      destination_row[col] = static_cast<scalar_t>(0);
    }
    return;
  }

  const scalar_t* source_row_ptr = source + source_row * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    destination_row[col] = source_row_ptr[col];
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMateCacheTransferBlockSize, 1)
    dsv4_prepare_zero_copy_mate_cache_kernel(
        scalar_t* __restrict__ source_cache,
        const int32_t* __restrict__ source_block_table,
        int32_t* __restrict__ mate_page_table,
        const scalar_t* __restrict__ q,
        scalar_t* __restrict__ q_latent,
        scalar_t* __restrict__ q_rope,
        int32_t* __restrict__ cache_seqlens,
        int64_t sequence_index,
        int64_t source_max_blocks_per_sequence,
        int64_t source_page_size,
        int64_t source_num_blocks,
        int64_t mate_pages,
        int64_t kv_len,
        int64_t row_width,
        int64_t num_heads) {
  constexpr int64_t kMatePageSize = 64;
  const int64_t mate_page = blockIdx.x;
  if (mate_page < mate_pages && threadIdx.x == 0) {
    const int64_t logical_token = mate_page * kMatePageSize;
    const int64_t source_page = logical_token / source_page_size;
    const int64_t source_subpage =
        (logical_token % source_page_size) / kMatePageSize;
    const int64_t physical_page = static_cast<int64_t>(
        source_block_table[sequence_index * source_max_blocks_per_sequence +
                           source_page]);
    const int64_t subpages_per_source_page = source_page_size / kMatePageSize;
    mate_page_table[mate_page] =
        physical_page < 0
            ? -1
            : static_cast<int32_t>(physical_page * subpages_per_source_page +
                                   source_subpage);
  }

  const int64_t head = blockIdx.x;
  if (head < num_heads) {
    for (int64_t dim = threadIdx.x; dim < 512; dim += blockDim.x) {
      q_latent[head * 512 + dim] =
          dim < 448 ? q[head * 512 + dim] : static_cast<scalar_t>(0);
    }
    for (int64_t dim = threadIdx.x; dim < 64; dim += blockDim.x) {
      q_rope[head * 64 + dim] = q[head * 512 + 448 + dim];
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cache_seqlens[0] = static_cast<int32_t>(kv_len);
  }

  const int64_t tail_rows =
      kv_len % kMatePageSize == 0 ? 0 : kMatePageSize - kv_len % kMatePageSize;
  const int64_t tail_row = blockIdx.x;
  if (tail_row >= tail_rows) {
    return;
  }
  const int64_t logical_token = kv_len + tail_row;
  const int64_t source_page = logical_token / source_page_size;
  const int64_t source_offset = logical_token % source_page_size;
  if (source_page >= source_max_blocks_per_sequence) {
    return;
  }
  const int64_t physical_page = static_cast<int64_t>(
      source_block_table[sequence_index * source_max_blocks_per_sequence +
                         source_page]);
  if (physical_page < 0 || physical_page >= source_num_blocks) {
    return;
  }
  scalar_t* row =
      source_cache +
      (physical_page * source_page_size + source_offset) * row_width;
  for (int64_t col = threadIdx.x; col < row_width; col += blockDim.x) {
    row[col] = static_cast<scalar_t>(0);
  }
}

std::tuple<torch::Tensor, torch::Tensor> dsv4_view_mate_cache(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& q,
    Dsv4MateFlashMlaWorkspace& workspace,
    int64_t sequence_index,
    int64_t kv_len) {
  constexpr int64_t kMatePageSize = 64;
  CHECK(cache.defined() && cache.is_contiguous());
  CHECK(cache.dim() == 3 || cache.dim() == 4);
  CHECK(cache.size(1) == 64 || cache.size(1) == 128);
  CHECK_EQ(cache.size(-1), 576);
  CHECK_EQ(block_table.scalar_type(), torch::kInt32);
  CHECK(block_table.is_contiguous());
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_GT(kv_len, 0);

  const int64_t mate_pages = (kv_len + kMatePageSize - 1) / kMatePageSize;
  const int64_t source_pages = (kv_len + cache.size(1) - 1) / cache.size(1);
  CHECK_LE(source_pages, block_table.size(1));
  const int64_t subpages_per_source_page = cache.size(1) / kMatePageSize;
  auto mate_cache = cache.view({cache.size(0) * subpages_per_source_page,
                                kMatePageSize,
                                1,
                                cache.size(-1)});
  auto page_table = workspace.page_table.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/mate_pages);
  const int64_t tail_rows =
      kv_len % kMatePageSize == 0 ? 0 : kMatePageSize - kv_len % kMatePageSize;
  const dim3 block(kMateCacheTransferBlockSize);
  const dim3 grid(
      std::max<int64_t>({int64_t{1}, mate_pages, tail_rows, q.size(1)}));
  DISPATCH_FLOATING_TYPES(
      cache.scalar_type(), "dsv4_prepare_zero_copy_mate_cache", [&] {
        dsv4_prepare_zero_copy_mate_cache_kernel<scalar_t>
            <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                cache.data_ptr<scalar_t>(),
                block_table.data_ptr<int32_t>(),
                page_table.data_ptr<int32_t>(),
                q.data_ptr<scalar_t>(),
                workspace.q_latent.data_ptr<scalar_t>(),
                workspace.q_rope.data_ptr<scalar_t>(),
                workspace.cache_seqlens.data_ptr<int32_t>(),
                sequence_index,
                block_table.size(1),
                cache.size(1),
                cache.size(0),
                mate_pages,
                kv_len,
                cache.size(-1),
                q.size(1));
      });
  check_musa_kernel_launch("dsv4 kernel");
  return {mate_cache, page_table};
}

torch::Tensor dsv4_compact_mate_cache(const torch::Tensor& cache,
                                      const torch::Tensor& block_table,
                                      int64_t sequence_index,
                                      int64_t kv_len) {
  constexpr int64_t kMatePageSize = 64;
  CHECK(cache.defined() && cache.is_contiguous());
  CHECK(cache.dim() == 3 || cache.dim() == 4);
  if (cache.dim() == 4) {
    CHECK_EQ(cache.size(2), 1);
  }
  CHECK_EQ(cache.size(-1), 576);
  CHECK_EQ(cache.numel(), cache.size(0) * cache.size(1) * cache.size(-1));
  CHECK(block_table.scalar_type() == torch::kInt32);
  CHECK(block_table.is_contiguous());
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_GT(kv_len, 0);
  CHECK(cache.size(1) == 64 || cache.size(1) == 128);

  const int64_t used_pages = (kv_len + kMatePageSize - 1) / kMatePageSize;
  const int64_t source_pages = (kv_len + cache.size(1) - 1) / cache.size(1);
  CHECK_LE(source_pages, block_table.size(1));
  const int64_t output_pages = used_pages + 1;
  const int64_t row_width = cache.size(-1);
  auto output_sizes = cache.sizes().vec();
  output_sizes[0] = output_pages;
  output_sizes[1] = kMatePageSize;
  auto output = torch::empty(output_sizes, cache.options());
  auto source_2d = cache.view({-1, row_width});
  auto output_2d = output.view({-1, row_width});
  const dim3 grid(output_2d.size(0));
  const dim3 block(kMateCacheTransferBlockSize);
  DISPATCH_FLOATING_TYPES(cache.scalar_type(), "dsv4_compact_mate_cache", [&] {
    dsv4_compact_mate_cache_kernel<scalar_t>
        <<<grid, block, 0, current_dsv4_cache_stream()>>>(
            source_2d.data_ptr<scalar_t>(),
            output_2d.data_ptr<scalar_t>(),
            block_table.data_ptr<int32_t>(),
            sequence_index,
            kv_len,
            cache.size(1),
            cache.size(0),
            block_table.size(1),
            output_2d.size(0),
            row_width);
  });
  check_musa_kernel_launch("dsv4 kernel");
  return output;
}

void warn_dsv4_mate_flashmla_fallback(const std::string& reason) {
  static std::once_flag once;
  std::call_once(once, [&reason]() {
    LOG(WARNING) << "DeepSeek-V4 Mate FlashMLA fallback: " << reason;
  });
}

torch::Tensor try_dsv4_mate_flashmla_decode(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& dynamic_kv_seq_lens,
    int64_t dynamic_window_capacity,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& retained_mass_output,
    const torch::Tensor& destination) {
  constexpr int64_t kModelHeadDim = 512;
  constexpr int64_t kNopeHeadDim = 448;
  constexpr int64_t kMateLatentHeadDim = 512;
  constexpr int64_t kMateCacheHeadDim = 576;
  constexpr int64_t kMatePageSize = 64;
  constexpr int64_t kXllmPageSize = 128;
  if (!dsv4_mate_flashmla_enabled()) {
    return torch::Tensor();
  }
  const bool dynamic_graph_window = dynamic_kv_seq_lens.defined();
  // Dynamic Graph FlashMLA uses a different accumulation/layout path from
  // eager scalar-length decode. Keep an opt-out for bit/accuracy isolation;
  // the default remains enabled for the performance path.
  if (dynamic_graph_window) {
    const char* disable_dynamic_mate =
        std::getenv("XLLM_DSV4_DISABLE_DYNAMIC_MATE");
    if (disable_dynamic_mate != nullptr &&
        std::string(disable_dynamic_mate) == "1") {
      return torch::Tensor();
    }
  }
  const int64_t min_kv_len = dsv4_mate_flashmla_min_kv_len();
  const int64_t max_kv_len = dsv4_mate_flashmla_max_kv_len();
  if (max_kv_len <= 0 ||
      (!dynamic_graph_window && (kv_len < min_kv_len || kv_len > max_kv_len))) {
    return torch::Tensor();
  }
  if (q.dim() != 3 || q.size(0) != 1 || q.size(-1) != kModelHeadDim ||
      cache.dim() < 3 ||
      (cache.size(1) != kMatePageSize && cache.size(1) != kXllmPageSize) ||
      cache.size(-1) != kMateCacheHeadDim || block_table.dim() != 2 ||
      sequence_index < 0 || sequence_index >= block_table.size(0) ||
      kv_len <= 0 ||
      (q.scalar_type() != torch::kBFloat16 &&
       q.scalar_type() != torch::kFloat16)) {
    return torch::Tensor();
  }
  if (dynamic_graph_window &&
      (dynamic_kv_seq_lens.scalar_type() != torch::kInt32 ||
       !dynamic_kv_seq_lens.is_contiguous() || dynamic_window_capacity <= 0 ||
       dynamic_window_capacity > max_kv_len)) {
    return torch::Tensor();
  }
  CHECK(destination.defined() && destination.sizes() == q.sizes() &&
        destination.scalar_type() == q.scalar_type() &&
        destination.device() == q.device() && destination.is_contiguous());
  const int64_t num_heads = q.size(1);
  torch::Tensor retained_mass;
  if (retained_mass_output.has_value() && retained_mass_output->defined()) {
    retained_mass = *retained_mass_output;
    CHECK(attn_sink.defined() && attn_sink.numel() == num_heads)
        << "Deferred FlashMLA sink scaling requires per-head attention sink.";
    CHECK(retained_mass.device() == q.device() &&
          retained_mass.scalar_type() == torch::kFloat32 &&
          retained_mass.is_contiguous() && retained_mass.dim() == 1 &&
          retained_mass.size(0) == num_heads)
        << "FlashMLA retained-mass output must be contiguous FP32 [num_heads].";
  }

  std::string reason;
  const bool direct_mla = dsv4_mate_flashmla_direct_enabled();
  const bool native_leftpad =
      dynamic_graph_window && direct_mla &&
      dsv4_mate_flashmla_graph_page_aligned_zero_copy_enabled() &&
      dsv4_mate_flashmla_graph_native_leftpad_enabled();
  auto func_opt =
      try_get_dsv4_mate_flashmla_function(direct_mla, native_leftpad, &reason);
  if (!func_opt.defined()) {
    warn_dsv4_mate_flashmla_fallback(reason);
    return torch::Tensor();
  }

  try {
    auto source_block_table = block_table
                                  .slice(/*dim=*/0,
                                         /*start=*/sequence_index,
                                         /*end=*/sequence_index + 1)
                                  .to(q.device())
                                  .to(torch::kInt32)
                                  .contiguous();
    const int64_t source_page_size = cache.size(1);
    const int64_t source_pages =
        (kv_len + source_page_size - 1) / source_page_size;
    if (!dynamic_graph_window && source_pages > source_block_table.size(1)) {
      warn_dsv4_mate_flashmla_fallback(
          "full-prefix attention exceeds the source page table");
      return torch::Tensor();
    }
    const int64_t active_capacity =
        dynamic_graph_window ? dynamic_window_capacity : kv_len;
    const bool graph_page_aligned_zero_copy =
        dynamic_graph_window && direct_mla &&
        dsv4_mate_flashmla_graph_page_aligned_zero_copy_enabled();
    const bool fuse_compact_query =
        dynamic_graph_window && !graph_page_aligned_zero_copy &&
        dsv4_mate_flashmla_fuse_compact_query_enabled();
    const int64_t mate_pages =
        graph_page_aligned_zero_copy
            ? (active_capacity + 2 * kMatePageSize - 2) / kMatePageSize
            : (active_capacity + kMatePageSize - 1) / kMatePageSize;
    const int64_t output_pages = mate_pages + 1;
    auto& flash_workspace = dsv4_mate_flashmla_workspace(q, output_pages);
    const bool zero_copy = direct_mla && dsv4_mate_flashmla_zero_copy_enabled();
    torch::Tensor mate_cache;
    torch::Tensor page_table;
    if (graph_page_aligned_zero_copy) {
      CHECK_EQ(source_page_size % kMatePageSize, 0)
          << "Graph page-aligned FlashMLA requires the xLLM cache page size "
             "to be divisible by 64.";
      mate_cache =
          cache.view({cache.size(0) * (source_page_size / kMatePageSize),
                      kMatePageSize,
                      1,
                      kMateCacheHeadDim});
      page_table = flash_workspace.page_table.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/output_pages);
    } else if (dynamic_graph_window) {
      mate_cache = flash_workspace.compact_cache.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/output_pages);
      page_table = flash_workspace.page_table.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/output_pages);
      const dim3 block(kMateCacheTransferBlockSize);
      const dim3 cache_grid(output_pages * kMatePageSize);
      DISPATCH_FLOATING_TYPES(
          cache.scalar_type(), "dsv4_compact_dynamic_mate_window", [&] {
            dsv4_compact_dynamic_mate_window_kernel<scalar_t>
                <<<cache_grid, block, 0, current_dsv4_cache_stream()>>>(
                    cache.data_ptr<scalar_t>(),
                    mate_cache.data_ptr<scalar_t>(),
                    source_block_table.data_ptr<int32_t>(),
                    dynamic_kv_seq_lens.data_ptr<int32_t>(),
                    fuse_compact_query ? q.data_ptr<scalar_t>() : nullptr,
                    fuse_compact_query
                        ? flash_workspace.q_latent.data_ptr<scalar_t>()
                        : nullptr,
                    fuse_compact_query
                        ? flash_workspace.q_rope.data_ptr<scalar_t>()
                        : nullptr,
                    fuse_compact_query
                        ? flash_workspace.cache_seqlens.data_ptr<int32_t>()
                        : nullptr,
                    /*block_table_sequence_index=*/0,
                    /*dynamic_sequence_index=*/sequence_index,
                    dynamic_window_capacity,
                    source_page_size,
                    cache.size(0),
                    source_block_table.size(1),
                    mate_cache.size(0) * mate_cache.size(1),
                    cache.size(-1),
                    num_heads,
                    fuse_compact_query ? 1 : 0);
          });
      check_musa_kernel_launch("dsv4 kernel");
    } else if (zero_copy) {
      std::tie(mate_cache, page_table) = dsv4_view_mate_cache(
          cache, source_block_table, q, flash_workspace, 0, kv_len);
    } else {
      // The compatibility bridge materializes the active prefix and clears
      // rows that the non-ASM kernel may prefetch beyond the sequence.
      mate_cache =
          dsv4_compact_mate_cache(cache, source_block_table, 0, kv_len);
      page_table =
          torch::arange(
              mate_cache.size(0),
              torch::TensorOptions().dtype(torch::kInt32).device(q.device()))
              .view({1, -1});
    }

    auto q_4d = q.unsqueeze(1);
    auto q_latent = flash_workspace.q_latent;
    auto q_rope = flash_workspace.q_rope;
    auto cache_seqlens = flash_workspace.cache_seqlens;
    auto cache_leftpad = flash_workspace.cache_leftpad;
    if (dynamic_graph_window) {
      if (!fuse_compact_query) {
        const dim3 grid(std::max<int64_t>(num_heads, 1));
        const dim3 block(kMateCacheTransferBlockSize);
        DISPATCH_FLOATING_TYPES(
            q.scalar_type(), "dsv4_prepare_dynamic_flashmla_query", [&] {
              dsv4_prepare_dynamic_flashmla_query_kernel<scalar_t>
                  <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                      q.data_ptr<scalar_t>(),
                      q_latent.data_ptr<scalar_t>(),
                      q_rope.data_ptr<scalar_t>(),
                      cache_seqlens.data_ptr<int32_t>(),
                      dynamic_kv_seq_lens.data_ptr<int32_t>(),
                      sequence_index,
                      dynamic_window_capacity,
                      num_heads);
            });
        check_musa_kernel_launch("dsv4 kernel");
      }
      if (graph_page_aligned_zero_copy) {
        DISPATCH_FLOATING_TYPES(
            cache.scalar_type(),
            "dsv4_prepare_dynamic_flashmla_page_table",
            [&] {
              dsv4_prepare_dynamic_flashmla_page_table_kernel<scalar_t>
                  <<<1,
                     kMateCacheTransferBlockSize,
                     0,
                     current_dsv4_cache_stream()>>>(
                      cache.data_ptr<scalar_t>(),
                      source_block_table.data_ptr<int32_t>(),
                      dynamic_kv_seq_lens.data_ptr<int32_t>(),
                      page_table.data_ptr<int32_t>(),
                      cache_seqlens.data_ptr<int32_t>(),
                      cache_leftpad.data_ptr<int32_t>(),
                      sequence_index,
                      dynamic_window_capacity,
                      source_page_size,
                      cache.size(0),
                      source_block_table.size(1),
                      kMatePageSize,
                      output_pages,
                      cache.size(-1));
            });
        check_musa_kernel_launch("dsv4 kernel");
      }
    } else if (!zero_copy) {
      q_latent.zero_();
      q_latent.slice(/*dim=*/-1, /*start=*/0, /*end=*/kNopeHeadDim)
          .copy_(q_4d.slice(
              /*dim=*/-1, /*start=*/0, /*end=*/kNopeHeadDim));
      q_rope.copy_(q_4d.slice(/*dim=*/-1,
                              /*start=*/kNopeHeadDim,
                              /*end=*/kModelHeadDim));
      cache_seqlens.fill_(kv_len);
    }
    auto ckv = mate_cache.slice(/*dim=*/-1,
                                /*start=*/0,
                                /*end=*/kMateLatentHeadDim);
    auto kpe = mate_cache.slice(/*dim=*/-1,
                                /*start=*/kMateLatentHeadDim,
                                /*end=*/kMateCacheHeadDim);
    auto output = destination.unsqueeze(1);
    auto softmax_lse = flash_workspace.softmax_lse;
    torch::Tensor workspace;
    if (!direct_mla) {
      constexpr int64_t kWorkspaceBytes = 16 * 1024 * 1024;
      workspace = torch::empty(
          {kWorkspaceBytes},
          torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
    }
    if (dynamic_graph_window) {
      bind_musa_tvmffi_current_stream(q.device());
    } else {
      bind_musa_tvmffi_stream(q.device());
      wait_musa_ffi_stream_for_current(q.device());
    }
    auto func = func_opt.value();
    if (direct_mla) {
      if (native_leftpad) {
        func(to_ffi_tensor(q_latent),
             to_ffi_tensor(q_rope),
             to_ffi_tensor(ckv),
             to_ffi_tensor(kpe),
             to_ffi_tensor(page_table),
             to_ffi_tensor(cache_seqlens),
             to_ffi_tensor(cache_leftpad),
             to_ffi_tensor(output),
             to_ffi_tensor(softmax_lse),
             softmax_scale,
             /*is_causal=*/false);
      } else {
        func(to_ffi_tensor(q_latent),
             to_ffi_tensor(q_rope),
             to_ffi_tensor(ckv),
             to_ffi_tensor(kpe),
             to_ffi_tensor(page_table),
             to_ffi_tensor(cache_seqlens),
             to_ffi_tensor(output),
             to_ffi_tensor(softmax_lse),
             softmax_scale,
             /*is_causal=*/false);
      }
    } else {
      func(to_ffi_tensor(q_latent),
           to_ffi_tensor(q_rope),
           to_ffi_tensor(ckv),
           to_ffi_tensor(kpe),
           to_ffi_tensor(cache_seqlens),
           to_ffi_tensor(page_table),
           softmax_scale,
           /*is_causal=*/false,
           /*cu_seqlens_q=*/ffi::Optional<ffi::Tensor>(),
           /*max_seqlen_q=*/ffi::Optional<int64_t>(),
           to_ffi_tensor(output),
           to_ffi_tensor(softmax_lse),
           to_ffi_tensor(workspace),
           /*is_inited=*/false);
    }
    if (!dynamic_graph_window) {
      wait_current_musa_stream_for_ffi(q.device());
    }

    if (attn_sink.defined() && attn_sink.numel() == num_heads) {
      const dim3 grid(num_heads);
      const dim3 block(kMateCacheTransferBlockSize);
      DISPATCH_FLOATING_TYPES(
          destination.scalar_type(), "dsv4_apply_flashmla_attention_sink", [&] {
            dsv4_apply_flashmla_attention_sink_kernel<scalar_t>
                <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                    destination.data_ptr<scalar_t>(),
                    softmax_lse.data_ptr<float>(),
                    attn_sink.data_ptr<float>(),
                    retained_mass.defined() ? retained_mass.data_ptr<float>()
                                            : nullptr,
                    num_heads,
                    kModelHeadDim);
          });
      check_musa_kernel_launch("dsv4 kernel");
    }
    return destination;
  } catch (const std::exception& ex) {
    warn_dsv4_mate_flashmla_fallback(ex.what());
    return torch::Tensor();
  }
}

bool ensure_dsv4_tilelang_musa_loader(std::string* reason) {
#if defined(USE_MUSA)
  struct LoaderStatus {
    bool loaded = false;
    std::string reason;
  };
  static const LoaderStatus status = []() {
    LoaderStatus result;
    void* handle =
        dlopen(dsv4_tilelang_runtime_library().c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      result.reason = "failed to load TileLang runtime: " +
                      std::string(error == nullptr ? "unknown" : error);
      return result;
    }
    handle = dlopen(dsv4_tilelang_library().c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      result.reason = "failed to load TileLang library: " +
                      std::string(error == nullptr ? "unknown" : error);
      return result;
    }
    result.loaded = true;
    return result;
  }();
  if (!status.loaded && reason != nullptr) {
    *reason = status.reason;
  }
  return status.loaded;
#else
  if (reason != nullptr) {
    *reason = "Mate sparse MLA requires a MUSA build";
  }
  return false;
#endif
}

ffi::Optional<ffi::Function> try_get_dsv4_mate_fused_dsa_function(
    std::string* reason,
    const std::string& uri) {
  static auto* function_cache =
      new std::unordered_map<std::string, ffi::Function>();
  static std::mutex function_cache_mutex;
  {
    std::lock_guard<std::mutex> lock(function_cache_mutex);
    auto cached = function_cache->find(uri);
    if (cached != function_cache->end()) {
      return cached->second;
    }
  }
  if (!ensure_dsv4_tilelang_musa_loader(reason)) {
    return ffi::Optional<ffi::Function>();
  }
  const std::string so_path = path_to_uri_so_lib(uri);
  std::ifstream so_file(so_path);
  if (!so_file.good()) {
    if (reason != nullptr) {
      *reason = "Mate MODEL1 module is missing: " + so_path;
    }
    return ffi::Optional<ffi::Function>();
  }
  try {
    auto function = get_module(uri)->GetFunction("main");
    if (!function.defined()) {
      if (reason != nullptr) {
        *reason = "main is not exported by " + so_path;
      }
      return ffi::Optional<ffi::Function>();
    }
    std::lock_guard<std::mutex> lock(function_cache_mutex);
    auto [it, inserted] = function_cache->emplace(uri, function.value());
    return inserted ? function : ffi::Optional<ffi::Function>(it->second);
  } catch (const std::exception& ex) {
    if (reason != nullptr) {
      *reason = "failed to load Mate MODEL1 module: " + std::string(ex.what());
    }
    return ffi::Optional<ffi::Function>();
  }
}

bool dsv4_model1_metadata(const torch::Tensor& q,
                          int64_t heads,
                          const torch::Tensor& topk_length,
                          const torch::Tensor& extra_topk_length,
                          torch::Tensor* metadata,
                          torch::Tensor* num_splits,
                          std::string* reason) {
  constexpr int64_t kTopk = 512;
  constexpr int64_t kExtraTopk = 128;
  const std::string uri = dsv4_mate_metadata_uri();
  const std::string so_path = path_to_uri_so_lib(uri);
  std::ifstream so_file(so_path);
  if (!so_file.good()) {
    if (reason != nullptr) {
      *reason = "Mate metadata module is missing: " + so_path;
    }
    return false;
  }
  try {
    auto function = get_module(uri)->GetFunction("get_mla_decoding_metadata");
    if (!function.defined()) {
      if (reason != nullptr) {
        *reason = "get_mla_decoding_metadata is not exported by " + so_path;
      }
      return false;
    }
    const int64_t q_tokens_per_head_k = q.size(1) * heads;
    if (!metadata->defined() || metadata->dim() != 2 ||
        metadata->size(1) != 8 || metadata->device() != q.device() ||
        !num_splits->defined() || num_splits->numel() != q.size(0) + 1 ||
        num_splits->device() != q.device()) {
      if (reason != nullptr) {
        *reason = "Mate MODEL1 metadata workspace is not initialized";
      }
      return false;
    }
    auto ffi_q = to_ffi_tensor(q);
    auto ffi_topk_length = to_ffi_tensor(topk_length);
    auto ffi_extra_topk_length = to_ffi_tensor(extra_topk_length);
    function.value()(
        /*seqlens_k=*/ffi::Optional<ffi::Tensor>(),
        /*num_q_tokens_per_head_k=*/q_tokens_per_head_k,
        /*h_k=*/int64_t{1},
        /*h_q=*/ffi::Optional<int64_t>(heads),
        /*is_fp8_kvcache=*/true,
        /*topk=*/ffi::Optional<int64_t>(kTopk),
        /*tile_scheduler_metadata=*/
        ffi::Optional<ffi::Tensor>(to_ffi_tensor(*metadata)),
        /*num_splits=*/
        ffi::Optional<ffi::Tensor>(to_ffi_tensor(*num_splits)),
        /*q=*/ffi::Optional<ffi::TensorView>(ffi::TensorView(ffi_q)),
        /*bs=*/ffi::Optional<int64_t>(q.size(0)),
        /*topk_length=*/
        ffi::Optional<ffi::TensorView>(ffi::TensorView(ffi_topk_length)),
        /*extra_topk_length=*/
        ffi::Optional<ffi::TensorView>(ffi::TensorView(ffi_extra_topk_length)),
        /*extra_topk=*/ffi::Optional<int64_t>(kExtraTopk));
    return true;
  } catch (const std::exception& ex) {
    if (reason != nullptr) {
      *reason =
          "failed to build Mate MODEL1 metadata: " + std::string(ex.what());
    }
    return false;
  }
}

torch::Tensor dsv4_model1_cache_bytes(const torch::Tensor& packed_cache) {
  CHECK(packed_cache.defined());
  CHECK_EQ(packed_cache.scalar_type(), torch::kUInt8);
  CHECK_EQ(packed_cache.dim(), 4);
  CHECK_EQ(packed_cache.size(2), 1);
  CHECK_EQ(packed_cache.size(3), 584);
  const int64_t page_stride = packed_cache.stride(0);
  CHECK_GE(page_stride, packed_cache.size(1) * 584);
  return packed_cache.as_strided({packed_cache.size(0), page_stride},
                                 {page_stride, 1});
}

struct Dsv4MateModel1Workspace {
  int64_t batch = 0;
  int64_t heads = 0;
  c10::DeviceIndex device = -1;
  torch::Tensor indices;
  torch::Tensor extra_indices;
  torch::Tensor topk_length;
  torch::Tensor extra_topk_length;
  torch::Tensor metadata;
  torch::Tensor num_splits;
  torch::Tensor glse;
  torch::Tensor output_partial;
  torch::Tensor output;
  torch::Tensor lse;
  int64_t cached_topk_length = -1;
  int64_t cached_extra_topk_length = -1;
  int64_t cached_metadata_key = std::numeric_limits<int64_t>::min();
  bool metadata_valid = false;
  unsigned long long metadata_capture_id = 0;
  bool metadata_was_captured = false;
};

// All Attention layers on one worker thread observe the same q6 causal
// length pattern for a decode step. The packed cache pointers differ by
// layer, but MODEL1 scheduling metadata does not; share its build across the
// layer loop without changing the external MATE ABI.
thread_local int64_t dsv4_mate_model1_metadata_key =
    std::numeric_limits<int64_t>::min();
thread_local int64_t dsv4_mate_model1_metadata_group = 0;

struct Dsv4MatePrefillWorkspace {
  int64_t seq_len = 0;
  int64_t padded_seq_len = 0;
  int64_t compressed_kv_len = 0;
  int64_t heads = 0;
  int64_t compiled_compressed_kv_len = 0;
  c10::DeviceIndex device = -1;
  torch::Tensor q_padded;
  torch::Tensor indices;
  torch::Tensor topk_length;
  torch::Tensor row_masks;
  torch::Tensor sink;
  torch::Tensor combined_kv;
  torch::Tensor output;
  std::array<torch::Tensor, 3> output_runs;
  torch::Tensor max_logits;
  torch::Tensor lse;
};

// Build the fixed MODEL1 C128 sequence-pack metadata on device. The previous
// path created the same 512-entry rows in a host vector and copied two tensors
// to MUSA whenever the prefill shape changed.
__global__ void __launch_bounds__(256, 1)
    dsv4_c128_prefill_metadata_kernel(int32_t* __restrict__ indices,
                                      int32_t* __restrict__ topk_length,
                                      int64_t packed_seq,
                                      int64_t token_pack,
                                      int64_t compressed_len,
                                      int64_t compiled_compressed_kv_len) {
  constexpr int64_t kTopk = 512;
  constexpr int64_t kCompressRatio = 128;
  const int64_t pack = blockIdx.x;
  if (pack >= packed_seq) {
    return;
  }

  const int64_t start = pack * token_pack;
  const int64_t end = start + token_pack - 1;
  const int64_t num_compressed =
      min((end + 1) / kCompressRatio, compressed_len);
  const int64_t swa_start = max(start - 127, int64_t{0});
  const int64_t num_swa = end - swa_start + 1;
  const int64_t total = num_compressed + num_swa;
  if (threadIdx.x == 0) {
    topk_length[pack] = static_cast<int32_t>(total);
  }
  int32_t* row = indices + pack * kTopk;
  for (int64_t index = threadIdx.x; index < kTopk; index += blockDim.x) {
    int32_t value = -1;
    if (index < num_compressed) {
      value = static_cast<int32_t>(index);
    } else if (index < total) {
      value = static_cast<int32_t>(compiled_compressed_kv_len + swa_start +
                                   index - num_compressed);
    }
    row[index] = value;
  }
}

template <typename scalar_t>
__global__ void dsv4_median3_kernel(const scalar_t* __restrict__ input_a,
                                    const scalar_t* __restrict__ input_b,
                                    const scalar_t* __restrict__ input_c,
                                    scalar_t* __restrict__ output,
                                    int64_t numel) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= numel) {
    return;
  }
  const float a = static_cast<float>(input_a[index]);
  const float b = static_cast<float>(input_b[index]);
  const float c = static_cast<float>(input_c[index]);
  const float low = fminf(a, b);
  const float high = fmaxf(a, b);
  output[index] = static_cast<scalar_t>(fmaxf(low, fminf(high, c)));
}

struct Dsv4MateC4PrefillWorkspace {
  int64_t seq_len = 0;
  int64_t padded_seq_len = 0;
  int64_t heads = 0;
  int64_t compressed_len = 0;
  c10::DeviceIndex device = -1;
  torch::Tensor q_padded;
  torch::Tensor combined_kv;
  torch::Tensor indices;
  torch::Tensor topk_length;
  torch::Tensor empty_indices;
  torch::Tensor empty_lengths;
  torch::Tensor sink;
  torch::Tensor output_padded;
  torch::Tensor max_logits;
  torch::Tensor lse;
};

struct Dsv4MateLocalSparsePrefillWorkspace {
  int64_t q_len = 0;
  int64_t kv_len = 0;
  int64_t compressed_len = 0;
  int64_t topk = 0;
  int64_t main_topk = 0;
  int64_t extra_topk = 0;
  c10::DeviceIndex device = -1;
  torch::Tensor combined_kv;
  torch::Tensor main_kv;
  torch::Tensor extra_kv;
  torch::Tensor indices;
  torch::Tensor extra_indices;
  torch::Tensor topk_length;
  torch::Tensor extra_topk_length;
  torch::Tensor empty_indices;
  torch::Tensor empty_lengths;
  torch::Tensor sink;
  torch::Tensor output;
  torch::Tensor max_logits;
  torch::Tensor lse;
};

// Build the two independent candidate lists consumed by the Mate MODEL1
// has_extra=true ABI.  Keeping compressed and SWA candidates separate avoids
// the device-to-device combined-KV copy that the legacy bridge required.
__global__ void dsv4_local_sparse_prefill_split_metadata_kernel(
    const int32_t* __restrict__ compressed_indices,
    int32_t* __restrict__ main_indices,
    int32_t* __restrict__ main_topk_length,
    int32_t* __restrict__ extra_indices,
    int32_t* __restrict__ extra_topk_length,
    int64_t q_len,
    int64_t kv_len,
    int64_t compressed_len,
    int64_t compressed_topk,
    int64_t main_topk,
    int64_t extra_topk,
    int64_t q_abs_start,
    int64_t compress_ratio) {
  constexpr int64_t kSwaTopk = 128;
  const int64_t token = blockIdx.x;
  if (token >= q_len) {
    return;
  }
  const int64_t absolute_token = q_abs_start + token;
  __shared__ int32_t main_count;
  __shared__ int32_t swa_start;
  __shared__ int32_t swa_count;
  if (threadIdx.x == 0) {
    main_count = 0;
    if (compress_ratio == 1) {
      // C1 attends the uncompressed KV stream directly.
      main_count = static_cast<int32_t>(min(absolute_token + 1, kv_len));
    } else if (compress_ratio == 4 && compressed_indices != nullptr) {
      const int64_t causal_compressed_len =
          min(compressed_len, (absolute_token + 1) / compress_ratio);
      const int32_t* source = compressed_indices + token * compressed_topk;
      while (main_count < compressed_topk && source[main_count] >= 0 &&
             source[main_count] < causal_compressed_len) {
        ++main_count;
      }
    } else if (compress_ratio == 128) {
      main_count = static_cast<int32_t>(
          min(compressed_len, (absolute_token + 1) / compress_ratio));
    }
    swa_start = static_cast<int32_t>(
        compress_ratio == 1 ? 0
                            : max(absolute_token - kSwaTopk + 1, int64_t{0}));
    swa_count = static_cast<int32_t>(
        compress_ratio == 1
            ? 0
            : max(min(absolute_token + 1, kv_len) - swa_start, int64_t{0}));
    main_topk_length[token] = main_count;
    extra_topk_length[token] = swa_count;
  }
  __syncthreads();
  int32_t* main_row = main_indices + token * main_topk;
  int32_t* extra_row = extra_indices + token * extra_topk;
  const int32_t* source = compressed_indices == nullptr
                              ? nullptr
                              : compressed_indices + token * compressed_topk;
  for (int64_t column = threadIdx.x; column < main_topk; column += blockDim.x) {
    int32_t value = -1;
    if (column < main_count) {
      value = compress_ratio == 4
                  ? source[column]
                  : static_cast<int32_t>(compress_ratio == 1 ? column : column);
    }
    main_row[column] = value;
  }
  for (int64_t column = threadIdx.x; column < extra_topk;
       column += blockDim.x) {
    int32_t value = -1;
    if (column < swa_count) {
      value = swa_start + static_cast<int32_t>(column);
    }
    extra_row[column] = value;
  }
}

__global__ void dsv4_local_sparse_prefill_metadata_kernel(
    const int32_t* __restrict__ compressed_indices,
    int32_t* __restrict__ combined_indices,
    int32_t* __restrict__ topk_length,
    int64_t q_len,
    int64_t kv_len,
    int64_t compressed_len,
    int64_t compressed_topk,
    int64_t combined_topk,
    int64_t q_abs_start,
    int64_t compress_ratio) {
  constexpr int64_t kSwaTopk = 128;
  const int64_t token = blockIdx.x;
  if (token >= q_len) {
    return;
  }
  const int64_t absolute_token = q_abs_start + token;
  __shared__ int32_t main_count;
  __shared__ int32_t swa_start;
  __shared__ int32_t swa_count;
  if (threadIdx.x == 0) {
    main_count = 0;
    if (compress_ratio == 4 && compressed_indices != nullptr) {
      const int64_t causal_compressed_len =
          min(compressed_len, (absolute_token + 1) / compress_ratio);
      const int32_t* source = compressed_indices + token * compressed_topk;
      while (main_count < compressed_topk && source[main_count] >= 0 &&
             source[main_count] < causal_compressed_len) {
        ++main_count;
      }
    } else if (compress_ratio == 128) {
      main_count = static_cast<int32_t>(
          min(compressed_len, (absolute_token + 1) / compress_ratio));
    }
    if (compress_ratio == 1) {
      swa_start = 0;
      swa_count = static_cast<int32_t>(min(absolute_token + 1, kv_len));
    } else {
      swa_start =
          static_cast<int32_t>(max(absolute_token - kSwaTopk + 1, int64_t{0}));
      swa_count = static_cast<int32_t>(
          max(min(absolute_token + 1, kv_len) - swa_start, int64_t{0}));
    }
    topk_length[token] = main_count + swa_count;
  }
  __syncthreads();
  int32_t* destination = combined_indices + token * combined_topk;
  const int32_t* source = compressed_indices == nullptr
                              ? nullptr
                              : compressed_indices + token * compressed_topk;
  for (int64_t column = threadIdx.x; column < combined_topk;
       column += blockDim.x) {
    int32_t value = -1;
    if (column < main_count) {
      value =
          compress_ratio == 4 ? source[column] : static_cast<int32_t>(column);
    } else if (column < main_count + swa_count) {
      value = static_cast<int32_t>(compressed_len) + swa_start +
              static_cast<int32_t>(column) - main_count;
    }
    destination[column] = value;
  }
}

Dsv4MateLocalSparsePrefillWorkspace& dsv4_mate_local_sparse_prefill_workspace(
    const torch::Tensor& q,
    int64_t kv_len,
    int64_t compressed_len,
    int64_t main_topk,
    int64_t extra_topk) {
  constexpr int64_t kHeads = 64;
  constexpr int64_t kHeadDim = 512;
  thread_local Dsv4MateLocalSparsePrefillWorkspace workspace;
  const int64_t q_len = q.size(0);
  if (workspace.q_len == q_len && workspace.kv_len == kv_len &&
      workspace.compressed_len == compressed_len &&
      workspace.main_topk == main_topk && workspace.extra_topk == extra_topk &&
      workspace.device == q.device().index()) {
    return workspace;
  }
  workspace.q_len = q_len;
  workspace.kv_len = kv_len;
  workspace.compressed_len = compressed_len;
  workspace.topk = main_topk;
  workspace.main_topk = main_topk;
  workspace.extra_topk = extra_topk;
  workspace.device = q.device().index();
  auto i32_options = q.options().dtype(torch::kInt32);
  auto f32_options = q.options().dtype(torch::kFloat32);
  // Keep separate streams for the main compressed candidates and the SWA
  // candidates.  The combined_kv allocation remains for the legacy fallback.
  workspace.combined_kv =
      torch::empty({compressed_len + kv_len, 1, kHeadDim}, q.options());
  workspace.main_kv =
      compressed_len > 0
          ? torch::empty({compressed_len, 1, kHeadDim}, q.options())
          : torch::empty({kv_len, 1, kHeadDim}, q.options());
  workspace.extra_kv = torch::empty({kv_len, 1, kHeadDim}, q.options());
  workspace.indices = torch::empty({q_len, 1, main_topk}, i32_options);
  workspace.extra_indices = torch::empty({q_len, 1, extra_topk}, i32_options);
  workspace.topk_length = torch::empty({q_len}, i32_options);
  workspace.extra_topk_length = torch::empty({q_len}, i32_options);
  workspace.empty_indices = torch::empty({q_len, 1, 0}, i32_options);
  workspace.empty_lengths = torch::zeros({q_len}, i32_options);
  workspace.sink = torch::empty({kHeads}, f32_options);
  workspace.output = torch::empty_like(q);
  workspace.max_logits = torch::empty({q_len, kHeads}, f32_options);
  workspace.lse = torch::empty_like(workspace.max_logits);
  return workspace;
}

__global__ void dsv4_c4_prefill_metadata_kernel(
    const int32_t* __restrict__ compressed_indices,
    int32_t* __restrict__ combined_indices,
    int32_t* __restrict__ topk_length,
    int64_t real_seq_len,
    int64_t padded_seq_len,
    int64_t compressed_len) {
  constexpr int64_t kCompressedTopk = 512;
  constexpr int64_t kSwaTopk = 128;
  constexpr int64_t kCombinedTopk = kCompressedTopk + kSwaTopk;
  const int64_t token = blockIdx.x;
  if (token >= padded_seq_len) {
    return;
  }
  __shared__ int32_t main_count;
  __shared__ int32_t swa_start;
  __shared__ int32_t swa_count;
  if (threadIdx.x == 0) {
    main_count = 0;
    if (token < real_seq_len) {
      const int32_t* source = compressed_indices + token * kCompressedTopk;
      while (main_count < kCompressedTopk && source[main_count] >= 0 &&
             source[main_count] < compressed_len) {
        ++main_count;
      }
    }
    swa_start =
        static_cast<int32_t>(token >= kSwaTopk ? token - kSwaTopk + 1 : 0);
    swa_count = static_cast<int32_t>(token - swa_start + 1);
    topk_length[token] = main_count + swa_count;
  }
  __syncthreads();
  int32_t* destination = combined_indices + token * kCombinedTopk;
  const int32_t* source = token < real_seq_len
                              ? compressed_indices + token * kCompressedTopk
                              : nullptr;
  for (int64_t column = threadIdx.x; column < kCombinedTopk;
       column += blockDim.x) {
    int32_t value = -1;
    if (column < main_count) {
      value = source[column];
    } else if (column < main_count + swa_count) {
      value = static_cast<int32_t>(compressed_len) + swa_start +
              static_cast<int32_t>(column) - main_count;
    }
    destination[column] = value;
  }
}

Dsv4MateC4PrefillWorkspace& dsv4_mate_c4_prefill_workspace(
    const torch::Tensor& q,
    int64_t padded_seq_len,
    int64_t compressed_len) {
  constexpr int64_t kCompiledHeads = 64;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kCombinedTopk = 640;
  thread_local Dsv4MateC4PrefillWorkspace workspace;
  const int64_t seq_len = q.size(0);
  const int64_t heads = q.size(1);
  if (workspace.seq_len == seq_len &&
      workspace.padded_seq_len == padded_seq_len && workspace.heads == heads &&
      workspace.compressed_len == compressed_len &&
      workspace.device == q.device().index()) {
    return workspace;
  }
  workspace.seq_len = seq_len;
  workspace.padded_seq_len = padded_seq_len;
  workspace.heads = heads;
  workspace.compressed_len = compressed_len;
  workspace.device = q.device().index();
  auto i32_options = q.options().dtype(torch::kInt32);
  auto f32_options = q.options().dtype(torch::kFloat32);
  workspace.q_padded =
      torch::zeros({padded_seq_len, kCompiledHeads, kHeadDim}, q.options());
  workspace.combined_kv =
      torch::empty({compressed_len + padded_seq_len, 1, kHeadDim}, q.options());
  workspace.indices =
      torch::empty({padded_seq_len, 1, kCombinedTopk}, i32_options);
  workspace.topk_length = torch::empty({padded_seq_len}, i32_options);
  workspace.empty_indices = torch::empty({padded_seq_len, 1, 0}, i32_options);
  workspace.empty_lengths = torch::zeros({padded_seq_len}, i32_options);
  workspace.sink = torch::zeros({kCompiledHeads}, f32_options);
  workspace.output_padded = torch::empty_like(workspace.q_padded);
  workspace.max_logits =
      torch::empty({padded_seq_len, kCompiledHeads}, f32_options);
  workspace.lse = torch::empty_like(workspace.max_logits);
  return workspace;
}

Dsv4MatePrefillWorkspace& dsv4_mate_prefill_workspace(
    const torch::Tensor& q,
    int64_t padded_seq_len,
    int64_t compressed_kv_len,
    int64_t compiled_compressed_kv_len) {
  constexpr int64_t kTopk = 512;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kCompressRatio = 128;
  thread_local Dsv4MatePrefillWorkspace workspace;
  const int64_t seq_len = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t token_pack = 64 / heads;
  const int64_t packed_seq = padded_seq_len / token_pack;
  if (workspace.seq_len == seq_len &&
      workspace.padded_seq_len == padded_seq_len &&
      workspace.compressed_kv_len == compressed_kv_len &&
      workspace.heads == heads &&
      workspace.compiled_compressed_kv_len == compiled_compressed_kv_len &&
      workspace.device == q.device().index()) {
    return workspace;
  }

  workspace.seq_len = seq_len;
  workspace.padded_seq_len = padded_seq_len;
  workspace.compressed_kv_len = compressed_kv_len;
  workspace.heads = heads;
  workspace.compiled_compressed_kv_len = compiled_compressed_kv_len;
  workspace.device = q.device().index();
  auto i32_options = q.options().dtype(torch::kInt32);
  workspace.indices = torch::empty({packed_seq, 1, kTopk}, i32_options);
  workspace.topk_length = torch::empty({packed_seq}, i32_options);
  if (dsv4_c128_device_metadata_enabled()) {
    deepseek_v4_c128_prefill_metadata(workspace.indices,
                                      workspace.topk_length,
                                      token_pack,
                                      compressed_kv_len,
                                      compiled_compressed_kv_len);
  } else {
    std::vector<int32_t> host_indices(packed_seq * kTopk, -1);
    std::vector<int32_t> host_topk_length(packed_seq, 0);
    for (int64_t pack = 0; pack < packed_seq; ++pack) {
      const int64_t start = pack * token_pack;
      const int64_t end = start + token_pack - 1;
      const int64_t num_compressed =
          std::min((end + 1) / kCompressRatio, compressed_kv_len);
      const int64_t swa_start = std::max(start - 127, int64_t{0});
      const int64_t num_swa = end - swa_start + 1;
      const int64_t total = num_compressed + num_swa;
      host_topk_length[pack] = static_cast<int32_t>(total);
      int32_t* row = host_indices.data() + pack * kTopk;
      for (int64_t index = 0; index < num_compressed; ++index) {
        row[index] = static_cast<int32_t>(index);
      }
      for (int64_t index = 0; index < num_swa; ++index) {
        row[num_compressed + index] = static_cast<int32_t>(
            compiled_compressed_kv_len + swa_start + index);
      }
    }
    auto host_indices_tensor = torch::from_blob(
        host_indices.data(), {packed_seq, 1, kTopk}, torch::kInt32);
    auto host_topk_length_tensor =
        torch::from_blob(host_topk_length.data(), {packed_seq}, torch::kInt32);
    workspace.indices.copy_(host_indices_tensor);
    workspace.topk_length.copy_(host_topk_length_tensor);
  }
  workspace.row_masks =
      torch::empty({packed_seq, 1, kTopk}, q.options().dtype(torch::kInt32));
  workspace.sink =
      torch::empty({heads * token_pack}, q.options().dtype(torch::kFloat32));
  workspace.combined_kv = torch::empty(
      {compiled_compressed_kv_len + padded_seq_len, 1, kHeadDim}, q.options());
  workspace.q_padded =
      torch::zeros({padded_seq_len, heads, kHeadDim}, q.options());
  workspace.output = torch::empty_like(workspace.q_padded);
  for (auto& output_run : workspace.output_runs) {
    output_run = torch::empty_like(workspace.q_padded);
  }
  workspace.max_logits = torch::empty({packed_seq, heads * token_pack},
                                      q.options().dtype(torch::kFloat32));
  workspace.lse = torch::empty_like(workspace.max_logits);
  return workspace;
}

Dsv4MateModel1Workspace& dsv4_model1_workspace(const torch::Tensor& q,
                                             int64_t metadata_group) {
  constexpr int64_t kTopk = 512;
  constexpr int64_t kExtraTopk = 128;
  constexpr int64_t kHeadDim = 512;
  thread_local std::unordered_map<std::string, Dsv4MateModel1Workspace> workspaces;
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const std::string key = std::to_string(q.device().index()) + ":" +
                          std::to_string(batch) + ":" + std::to_string(heads) +
                          ":" + std::to_string(metadata_group);
  auto& workspace = workspaces[key];
  if (workspace.batch != batch || workspace.heads != heads ||
      workspace.device != q.device().index() || !workspace.metadata.defined()) {
    musaDeviceProp properties{};
    check_musa_runtime(
        musaGetDeviceProperties(&properties, q.device().index()));
    const int64_t metadata_parts =
        std::max<int64_t>(properties.multiProcessorCount /
                              std::max<int64_t>((heads + 63) / 64, 1),
                          1);
    workspace.batch = batch;
    workspace.heads = heads;
    workspace.device = q.device().index();
    auto i32_options = q.options().dtype(torch::kInt32);
    auto f32_options = q.options().dtype(torch::kFloat32);
    workspace.indices = torch::empty({batch, 1, 1, kTopk}, i32_options);
    workspace.extra_indices =
        torch::empty({batch, 1, 1, kExtraTopk}, i32_options);
    workspace.topk_length = torch::empty({batch}, i32_options);
    workspace.extra_topk_length = torch::empty({batch}, i32_options);
    workspace.metadata = torch::empty({metadata_parts, 8}, i32_options);
    workspace.num_splits = torch::empty({batch + 1}, i32_options);
    workspace.glse =
        torch::empty({batch + metadata_parts, 1, heads}, f32_options);
    workspace.output_partial =
        torch::empty({batch + metadata_parts, 1, heads, kHeadDim}, f32_options);
    workspace.output = torch::empty({batch, 1, heads, kHeadDim}, q.options());
    workspace.lse = torch::empty({batch, heads, 1}, f32_options);
    workspace.cached_topk_length = -1;
    workspace.cached_extra_topk_length = -1;
    workspace.cached_metadata_key = std::numeric_limits<int64_t>::min();
    workspace.metadata_valid = false;
  }
  return workspace;
}

torch::Tensor try_dsv4_mate_model1_sparse_decode(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    int64_t topk_length,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    int64_t extra_topk_length,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& batched_topk_lengths = std::nullopt,
    const std::optional<torch::Tensor>& batched_extra_topk_lengths =
        std::nullopt) {
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kTopk = 512;
  constexpr int64_t kExtraTopk = 128;
  constexpr int64_t kPageSize = 128;
  constexpr double kCompiledSoftmaxScale = 0.044194173824159216;

  if (!dsv4_mate_fused_dsa_enabled() ||
      std::abs(softmax_scale - kCompiledSoftmaxScale) > 1.0e-8 ||
      q.scalar_type() != torch::kBFloat16 || q.dim() != 3 ||
      (q.size(1) != 8 && q.size(1) != 16 && q.size(1) != 64) ||
      q.size(2) != kHeadDim || !q.is_contiguous() ||
      (q.size(1) == 8 && !dsv4_mate_fused_dsa_h8_enabled())) {
    return torch::Tensor();
  }
  if (!packed_cache.defined() || packed_cache.size(1) != kPageSize ||
      !extra_packed_cache.defined() ||
      extra_packed_cache.size(1) != kPageSize || !physical_indices.defined() ||
      !extra_physical_indices.defined()) {
    return torch::Tensor();
  }
  CHECK_EQ(physical_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(extra_physical_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(physical_indices.device(), q.device());
  CHECK_EQ(extra_physical_indices.device(), q.device());
  CHECK_GE(topk_length, 0);
  CHECK_LE(topk_length, kTopk);
  CHECK_GE(extra_topk_length, 0);
  CHECK_LE(extra_topk_length, kExtraTopk);

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const bool has_batched_lengths = batched_topk_lengths.has_value() ||
                                   batched_extra_topk_lengths.has_value();
  if (has_batched_lengths) {
    if (!batched_topk_lengths.has_value() ||
        !batched_extra_topk_lengths.has_value() ||
        !batched_topk_lengths->defined() ||
        !batched_extra_topk_lengths->defined() ||
        batched_topk_lengths->dim() != 1 ||
        batched_extra_topk_lengths->dim() != 1 ||
        batched_topk_lengths->size(0) != batch ||
        batched_extra_topk_lengths->size(0) != batch ||
        batched_topk_lengths->device() != q.device() ||
        batched_extra_topk_lengths->device() != q.device()) {
      return torch::Tensor();
    }
  }
  std::string reason;
  const std::string uri = dsv4_mate_fused_dsa_uri(heads);
  auto function = try_get_dsv4_mate_fused_dsa_function(&reason, uri);
  if (!function.defined()) {
    static std::once_flag once;
    std::call_once(once, [&reason]() {
      LOG(WARNING) << "DeepSeek-V4 Mate MODEL1 fallback: " << reason;
    });
    return torch::Tensor();
  }

  try {
    // Cache packing is issued on the compatibility CUDA stream, while the
    // MATE FFI path consumes the native MUSA stream.  Queue an event wait so
    // the packed C4/SWA bytes are visible without a host synchronize.
    deepseek_v4_wait_model1_pack_event();
    auto main_bytes = dsv4_model1_cache_bytes(packed_cache);
    auto extra_bytes = dsv4_model1_cache_bytes(extra_packed_cache);
    auto q4d = q.unsqueeze(1);
    auto flat_indices = physical_indices.reshape({batch, -1});
    auto flat_extra_indices = extra_physical_indices.reshape({batch, -1});
    CHECK_GE(flat_indices.size(1), topk_length);
    CHECK_GE(flat_extra_indices.size(1), extra_topk_length);
    const int64_t metadata_group = has_batched_lengths ? dsv4_mate_model1_metadata_group : 0;
    auto& workspace = dsv4_model1_workspace(q, metadata_group);
    torch::Tensor indices_4d;
    if (flat_indices.size(1) == kTopk) {
      indices_4d = flat_indices.view({batch, 1, 1, kTopk});
    } else {
      workspace.indices.fill_(-1);
      workspace.indices.view({batch, kTopk})
          .slice(/*dim=*/1, 0, flat_indices.size(1))
          .copy_(flat_indices);
      indices_4d = workspace.indices;
    }
    torch::Tensor extra_indices_4d;
    if (flat_extra_indices.size(1) == kExtraTopk) {
      extra_indices_4d = flat_extra_indices.view({batch, 1, 1, kExtraTopk});
    } else {
      workspace.extra_indices.fill_(-1);
      workspace.extra_indices.view({batch, kExtraTopk})
          .slice(/*dim=*/1, 0, flat_extra_indices.size(1))
          .copy_(flat_extra_indices);
      extra_indices_4d = workspace.extra_indices;
    }
    const int64_t effective_topk_length =
        std::min<int64_t>(topk_length, flat_indices.size(1));
    const int64_t effective_extra_topk_length =
        std::min<int64_t>(extra_topk_length, flat_extra_indices.size(1));
    // Scalar decode can reuse metadata until the visible length crosses a
    // boundary. Compact MTP has one dynamic prefix per row, so use the
    // device-provided vectors and refresh once for the whole q batch. This
    // preserves the Mate batch ABI while removing the old per-row host loop.
    musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (metadata_group != 0) {
      check_musa_runtime(musaStreamGetCaptureInfo(
          current_dsv4_cache_stream(), &capture_status, &capture_id));
    }
    const bool capturing = capture_status == musaStreamCaptureStatusActive;
    const bool new_capture = metadata_group != 0 &&
        (workspace.metadata_was_captured != capturing ||
         (capturing && workspace.metadata_capture_id != capture_id));
    const bool refresh_metadata = new_capture ||
        (has_batched_lengths
             ? (!workspace.metadata_valid ||
                dsv4_mate_model1_metadata_key ==
                    std::numeric_limits<int64_t>::min() ||
                workspace.cached_metadata_key !=
                    dsv4_mate_model1_metadata_key)
             : (!workspace.metadata_valid ||
                workspace.cached_topk_length != effective_topk_length ||
                workspace.cached_extra_topk_length !=
                    effective_extra_topk_length));
    if (has_batched_lengths && (metadata_group == 0 || refresh_metadata)) {
      auto topk_lengths = batched_topk_lengths.value();
      auto extra_topk_lengths = batched_extra_topk_lengths.value();
      if (topk_lengths.scalar_type() != torch::kInt32) {
        topk_lengths = topk_lengths.to(torch::kInt32);
      }
      if (extra_topk_lengths.scalar_type() != torch::kInt32) {
        extra_topk_lengths = extra_topk_lengths.to(torch::kInt32);
      }
      workspace.topk_length.copy_(topk_lengths.contiguous());
      workspace.extra_topk_length.copy_(extra_topk_lengths.contiguous());
    } else if (refresh_metadata) {
      workspace.topk_length.fill_(effective_topk_length);
      workspace.extra_topk_length.fill_(effective_extra_topk_length);
    }
    auto sink = attn_sink.defined()
                    ? attn_sink
                    : torch::full({heads},
                                  -std::numeric_limits<float>::infinity(),
                                  q.options().dtype(torch::kFloat32));
    // Keep replicated h64 on the event-ordered FFI stream. Moving h64 to the
    // current Torch stream caused a long-decode token duplication followed by
    // a subsequent-request hang, while the FFI-stream path is the last
    // configuration that passed the complete 2.5K+1.5K gate. The h8 kernel
    // keeps its separately validated current-stream fast path.
    const bool use_current_stream = heads == 8;
    if (use_current_stream) {
      bind_musa_tvmffi_current_stream(q.device());
    } else {
      bind_musa_tvmffi_stream(q.device());
      wait_musa_ffi_stream_for_current(q.device());
    }
    if (refresh_metadata && !dsv4_model1_metadata(q4d,
                                                  heads,
                                                  workspace.topk_length,
                                                  workspace.extra_topk_length,
                                                  &workspace.metadata,
                                                  &workspace.num_splits,
                                                  &reason)) {
      static std::once_flag once;
      std::call_once(once, [&reason]() {
        LOG(WARNING) << "DeepSeek-V4 Mate MODEL1 metadata fallback: " << reason;
      });
      return torch::Tensor();
    }
    if (refresh_metadata) {
      workspace.metadata_capture_id = capture_id;
      workspace.metadata_was_captured = capturing;
      workspace.cached_topk_length = effective_topk_length;
      workspace.cached_extra_topk_length = effective_extra_topk_length;
      if (has_batched_lengths) {
        workspace.cached_metadata_key = dsv4_mate_model1_metadata_key;
      }
      workspace.metadata_valid = true;
    }
    function.value()(to_ffi_tensor(q4d),
                     to_ffi_tensor(main_bytes.view(torch::kFloat8_e4m3fn)),
                     to_ffi_tensor(main_bytes.view(torch::kBFloat16)),
                     to_ffi_tensor(main_bytes),
                     to_ffi_tensor(indices_4d),
                     to_ffi_tensor(workspace.topk_length),
                     to_ffi_tensor(extra_bytes.view(torch::kFloat8_e4m3fn)),
                     to_ffi_tensor(extra_bytes.view(torch::kBFloat16)),
                     to_ffi_tensor(extra_bytes),
                     to_ffi_tensor(extra_indices_4d),
                     to_ffi_tensor(workspace.extra_topk_length),
                     to_ffi_tensor(sink),
                     to_ffi_tensor(workspace.metadata),
                     to_ffi_tensor(workspace.num_splits),
                     to_ffi_tensor(workspace.glse),
                     to_ffi_tensor(workspace.output_partial),
                     to_ffi_tensor(workspace.output),
                     to_ffi_tensor(workspace.lse));
    if (!use_current_stream) {
      wait_current_musa_stream_for_ffi(q.device());
    }
    auto result = workspace.output.squeeze(1);
    if (output.has_value()) {
      output.value().copy_(result);
      return output.value();
    }
    return result;
  } catch (const std::exception& ex) {
    static std::once_flag once;
    std::call_once(once, [&ex]() {
      LOG(WARNING) << "DeepSeek-V4 Mate MODEL1 exception: " << ex.what();
    });
    return torch::Tensor();
  }
}

torch::Tensor try_dsv4_mate_model1_sparse_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kCompressRatio = 128;
  constexpr double kCompiledSoftmaxScale = 0.044194173824159216;

  auto reject = [](const char*) { return torch::Tensor(); };
  if (!true) {
    return reject("disabled");
  }
  if (std::abs(softmax_scale - kCompiledSoftmaxScale) > 1.0e-8) {
    return reject("softmax_scale");
  }
  if (q.scalar_type() != torch::kBFloat16 || q.dim() != 3) {
    return reject("q_dtype_or_rank");
  }
  if (kv.scalar_type() != torch::kBFloat16 || kv.dim() != 2) {
    return reject("kv_dtype_or_rank");
  }
  if (compressed_kv.scalar_type() != torch::kBFloat16 ||
      compressed_kv.dim() != 2) {
    return reject("compressed_kv_dtype_or_rank");
  }
  if (!q.is_contiguous() || !kv.is_contiguous() ||
      !compressed_kv.is_contiguous()) {
    return reject("noncontiguous_input");
  }
  const int64_t seq_len = q.size(0);
  const int64_t heads = q.size(1);
  // The MP31 packed prefill kernel is stable for TP4/TP8. The heads=32
  // configuration still has a nondeterministic first-PV tile and must fall
  // back until its TileLang schedule is fixed.
  if (heads != 8 && heads != 16) {
    return reject("unsupported_heads");
  }
  const int64_t max_seq_len = dsv4_mate_prefill_max_seq_len();
  if (max_seq_len > 0 && seq_len > max_seq_len) {
    return reject("seq_len_guard");
  }
  const int64_t token_pack = heads == 8 ? 8 : 4;
  const int64_t padded_seq_len =
      token_pack > 0 ? ((seq_len + token_pack - 1) / token_pack) * token_pack
                     : 0;
  if (token_pack == 0 || q.size(2) != kHeadDim || kv.size(0) != seq_len ||
      kv.size(1) != kHeadDim || compressed_kv.size(1) != kHeadDim ||
      seq_len <= 0 || compressed_kv.size(0) != seq_len / kCompressRatio) {
    return torch::Tensor();
  }
  if (attn_sink.defined() &&
      (attn_sink.scalar_type() != torch::kFloat32 || attn_sink.dim() != 1 ||
       attn_sink.size(0) != heads)) {
    return reject("attn_sink_shape_or_dtype");
  }

  std::string reason;
  const int64_t compressed_len = compressed_kv.size(0);
  int64_t compiled_compressed_kv_len = compressed_len;
  auto function = try_get_dsv4_mate_fused_dsa_function(
      &reason, dsv4_mate_prefill_uri(heads, compressed_len));
  if (!function.defined()) {
    compiled_compressed_kv_len = 512;
    function = try_get_dsv4_mate_fused_dsa_function(
        &reason, dsv4_mate_prefill_uri(heads, compiled_compressed_kv_len));
  }
  if (!function.defined()) {
    static std::once_flag once;
    std::call_once(once, [&reason]() {
      LOG(WARNING) << "DeepSeek-V4 Mate prefill fallback: " << reason;
    });
    return torch::Tensor();
  }
  if (compiled_compressed_kv_len < compressed_len) {
    return reject("compiled_compressed_kv_len");
  }

  try {
    const int64_t packed_seq = padded_seq_len / token_pack;
    auto& workspace = dsv4_mate_prefill_workspace(
        q, padded_seq_len, compressed_len, compiled_compressed_kv_len);
    workspace.q_padded.slice(/*dim=*/0, 0, seq_len).copy_(q);
    if (padded_seq_len != seq_len) {
      workspace.q_padded.slice(/*dim=*/0, seq_len, padded_seq_len).zero_();
    }
    if (attn_sink.defined()) {
      workspace.sink.view({token_pack, heads})
          .copy_(attn_sink.unsqueeze(0).expand({token_pack, heads}));
    } else {
      workspace.sink.zero_();
    }
    auto q_pack =
        workspace.q_padded.view({packed_seq, heads * token_pack, kHeadDim});
    workspace.combined_kv.slice(/*dim=*/0, 0, compressed_len)
        .copy_(compressed_kv.view({compressed_len, 1, kHeadDim}));
    workspace.combined_kv
        .slice(/*dim=*/0,
               compiled_compressed_kv_len,
               compiled_compressed_kv_len + seq_len)
        .copy_(kv.view({seq_len, 1, kHeadDim}));
    if (padded_seq_len != seq_len) {
      workspace.combined_kv
          .slice(/*dim=*/0,
                 compiled_compressed_kv_len + seq_len,
                 compiled_compressed_kv_len + padded_seq_len)
          .zero_();
    }
    torch::Tensor output_padded = workspace.output;

    bind_musa_tvmffi_stream(q.device());
    wait_musa_ffi_stream_for_current(q.device());
    auto invoke_prefill = [&](torch::Tensor& run_output) {
      auto output_pack =
          run_output.view({packed_seq, heads * token_pack, kHeadDim});
      function.value()(to_ffi_tensor(q_pack),
                       to_ffi_tensor(workspace.combined_kv),
                       to_ffi_tensor(workspace.indices),
                       to_ffi_tensor(workspace.topk_length),
                       to_ffi_tensor(workspace.row_masks),
                       to_ffi_tensor(workspace.sink),
                       to_ffi_tensor(output_pack),
                       to_ffi_tensor(workspace.max_logits),
                       to_ffi_tensor(workspace.lse));
    };
    const int redundant_runs = dsv4_mate_prefill_redundant_runs();
    if (redundant_runs == 1) {
      invoke_prefill(output_padded);
    } else {
      for (auto& run_output : workspace.output_runs) {
        invoke_prefill(run_output);
      }
    }
    wait_current_musa_stream_for_ffi(q.device());
    if (redundant_runs == 3) {
      constexpr int64_t kBlockSize = 256;
      const int64_t numel = output_padded.numel();
      const dim3 grid((numel + kBlockSize - 1) / kBlockSize);
      const dim3 block(kBlockSize);
      DISPATCH_FLOATING_TYPES(
          output_padded.scalar_type(), "dsv4_mate_prefill_median3", [&] {
            dsv4_median3_kernel<scalar_t>
                <<<grid, block, 0, current_dsv4_cache_stream()>>>(
                    workspace.output_runs[0].data_ptr<scalar_t>(),
                    workspace.output_runs[1].data_ptr<scalar_t>(),
                    workspace.output_runs[2].data_ptr<scalar_t>(),
                    output_padded.data_ptr<scalar_t>(),
                    numel);
          });
      check_musa_kernel_launch("dsv4 kernel");
    }
    auto result_view = output_padded.slice(/*dim=*/0, 0, seq_len);
    if (output.has_value()) {
      CHECK_EQ(output.value().sizes(), q.sizes());
      CHECK(output.value().is_contiguous());
      output.value().copy_(result_view);
      return output.value();
    }
    return result_view;
  } catch (const std::exception& ex) {
    static std::once_flag once;
    std::call_once(once, [&ex]() {
      LOG(WARNING) << "DeepSeek-V4 Mate prefill exception: " << ex.what();
    });
    return torch::Tensor();
  }
}

torch::Tensor try_dsv4_mate_model1_c4_sparse_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kCompressedTopk = 512;
  constexpr int64_t kCompressRatio = 4;
  constexpr double kCompiledSoftmaxScale = 0.044194173824159216;
  auto reject = [](const std::string&) { return torch::Tensor(); };
  if (!true) {
    return reject("disabled");
  }
  if (std::abs(softmax_scale - kCompiledSoftmaxScale) > 1.0e-8) {
    return reject("softmax_scale");
  }
  if (q.scalar_type() != torch::kBFloat16 || q.dim() != 3) {
    return reject("q_dtype_or_rank");
  }
  if (kv.scalar_type() != torch::kBFloat16 || kv.dim() != 2) {
    return reject("kv_dtype_or_rank");
  }
  if (compressed_kv.scalar_type() != torch::kBFloat16 ||
      compressed_kv.dim() != 2) {
    return reject("compressed_kv_dtype_or_rank");
  }
  if (compressed_indices.scalar_type() != torch::kInt32 ||
      compressed_indices.dim() != 2) {
    return reject("compressed_indices_dtype_or_rank");
  }
  if (!q.is_contiguous() || !kv.is_contiguous() ||
      !compressed_kv.is_contiguous() || !compressed_indices.is_contiguous()) {
    return reject("noncontiguous_input");
  }
  const int64_t seq_len = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t compressed_len = compressed_kv.size(0);
  // The h8 artifact is unstable for long sequences on the current MP31
  // TileLang schedule. The h16 artifact uses the same 64-head workspace and
  // is accurate for long C4 prefills, so pad h8 inputs to the h16 ABI and
  // discard the extra heads after the FFI call.
  const bool use_h16_abi = heads == 8 && dsv4_mate_c4_prefill_h8_use_h16();
  const int64_t abi_heads = use_h16_abi ? 16 : heads;
  const int64_t max_seq_len = heads == 8 && !use_h16_abi
                                  ? dsv4_mate_c4_prefill_h8_max_seq_len()
                                  : int64_t{0};
  const int64_t padded_seq_len =
      ((seq_len + kCompressRatio - 1) / kCompressRatio) * kCompressRatio;
  if ((heads != 8 && heads != 16 && heads != 32) || q.size(2) != kHeadDim ||
      kv.size(0) != seq_len || kv.size(1) != kHeadDim ||
      compressed_kv.size(1) != kHeadDim || seq_len <= 0 ||
      compressed_len != seq_len / kCompressRatio ||
      compressed_indices.sizes() !=
          torch::IntArrayRef({seq_len, kCompressedTopk}) ||
      (heads == 8 && !use_h16_abi &&
       (max_seq_len <= 0 || seq_len > max_seq_len))) {
    return reject("unsupported_shape");
  }
  if (attn_sink.defined() &&
      (attn_sink.scalar_type() != torch::kFloat32 || attn_sink.dim() != 1 ||
       attn_sink.size(0) != heads)) {
    return reject("attn_sink_shape_or_dtype");
  }

  std::string reason;
  auto function = try_get_dsv4_mate_fused_dsa_function(
      &reason, dsv4_mate_c4_prefill_uri(abi_heads));
  if (!function.defined()) {
    static std::once_flag once;
    std::call_once(once, [&reason]() {
      LOG(WARNING) << "DeepSeek-V4 Mate C4 prefill fallback: " << reason;
    });
    return torch::Tensor();
  }

  try {
    auto& workspace =
        dsv4_mate_c4_prefill_workspace(q, padded_seq_len, compressed_len);
    workspace.q_padded.slice(/*dim=*/0, 0, seq_len)
        .slice(/*dim=*/1, 0, heads)
        .copy_(q);
    if (use_h16_abi) {
      workspace.q_padded.slice(/*dim=*/0, 0, seq_len)
          .slice(/*dim=*/1, heads, abi_heads)
          .zero_();
    }
    if (padded_seq_len != seq_len) {
      workspace.q_padded.slice(/*dim=*/0, seq_len, padded_seq_len).zero_();
    }
    workspace.combined_kv.slice(/*dim=*/0, 0, compressed_len)
        .copy_(compressed_kv.view({compressed_len, 1, kHeadDim}));
    workspace.combined_kv
        .slice(/*dim=*/0, compressed_len, compressed_len + seq_len)
        .copy_(kv.view({seq_len, 1, kHeadDim}));
    if (padded_seq_len != seq_len) {
      workspace.combined_kv
          .slice(/*dim=*/0,
                 compressed_len + seq_len,
                 compressed_len + padded_seq_len)
          .zero_();
    }
    workspace.sink.slice(/*dim=*/0, 0, abi_heads).zero_();
    if (attn_sink.defined()) {
      workspace.sink.slice(/*dim=*/0, 0, heads).copy_(attn_sink);
    }
    dsv4_c4_prefill_metadata_kernel<<<dim3(padded_seq_len),
                                      dim3(kMateCacheTransferBlockSize),
                                      0,
                                      current_dsv4_cache_stream()>>>(
        compressed_indices.data_ptr<int32_t>(),
        workspace.indices.data_ptr<int32_t>(),
        workspace.topk_length.data_ptr<int32_t>(),
        seq_len,
        padded_seq_len,
        compressed_len);
    check_musa_kernel_launch("dsv4 kernel");

    bind_musa_tvmffi_stream(q.device());
    wait_musa_ffi_stream_for_current(q.device());
    function.value()(to_ffi_tensor(workspace.q_padded),
                     to_ffi_tensor(workspace.combined_kv),
                     to_ffi_tensor(workspace.indices),
                     to_ffi_tensor(workspace.topk_length),
                     to_ffi_tensor(workspace.combined_kv),
                     to_ffi_tensor(workspace.empty_indices),
                     to_ffi_tensor(workspace.empty_lengths),
                     to_ffi_tensor(workspace.sink),
                     to_ffi_tensor(workspace.output_padded),
                     to_ffi_tensor(workspace.max_logits),
                     to_ffi_tensor(workspace.lse));
    wait_current_musa_stream_for_ffi(q.device());
    auto result_view = workspace.output_padded.slice(/*dim=*/0, 0, seq_len)
                           .slice(/*dim=*/1, 0, heads);
    if (output.has_value()) {
      CHECK_EQ(output.value().sizes(), q.sizes());
      output.value().copy_(result_view);
    }
    return output.has_value() ? output.value() : result_view;
  } catch (const std::exception& ex) {
    static std::once_flag once;
    std::call_once(once, [&ex]() {
      LOG(WARNING) << "DeepSeek-V4 Mate C4 prefill exception: " << ex.what();
    });
    return torch::Tensor();
  }
}

torch::Tensor try_dsv4_mate_model1_local_sparse_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    int64_t q_abs_start,
    int64_t compress_ratio,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  constexpr int64_t kHeads = 64;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kCompressedTopk = 512;
  constexpr int64_t kSwaTopk = 128;
  constexpr double kCompiledSoftmaxScale = 0.044194173824159216;
  auto reject = [](const std::string&) { return torch::Tensor(); };
  if (std::abs(softmax_scale - kCompiledSoftmaxScale) > 1.0e-8) {
    return reject("softmax_scale");
  }
  if (q.scalar_type() != torch::kBFloat16 || q.dim() != 3 ||
      q.size(1) != kHeads || q.size(2) != kHeadDim || !q.is_contiguous()) {
    return reject("q_layout");
  }
  if (kv.scalar_type() != torch::kBFloat16 || kv.dim() != 2 ||
      kv.size(1) != kHeadDim || !kv.is_contiguous()) {
    return reject("kv_layout");
  }
  if (q.size(0) <= 0 || kv.size(0) <= 0 || q_abs_start < 0 ||
      q_abs_start + q.size(0) > kv.size(0)) {
    return reject("query_range");
  }
  if (compress_ratio != 1 && compress_ratio != 4 && compress_ratio != 128) {
    return reject("compress_ratio");
  }
  const int64_t compressed_len =
      compress_ratio == 1 ? 0 : compressed_kv.size(0);
  if (compress_ratio != 1 &&
      (compressed_kv.scalar_type() != torch::kBFloat16 ||
       compressed_kv.dim() != 2 || compressed_kv.size(1) != kHeadDim ||
       !compressed_kv.is_contiguous())) {
    return reject("compressed_kv_layout");
  }
  if (compress_ratio == 4 &&
      (compressed_indices.scalar_type() != torch::kInt32 ||
       compressed_indices.dim() != 2 ||
       compressed_indices.sizes() !=
           torch::IntArrayRef({q.size(0), kCompressedTopk}) ||
       !compressed_indices.is_contiguous())) {
    return reject("compressed_indices_layout");
  }
  if (attn_sink.scalar_type() != torch::kFloat32 || attn_sink.dim() != 1 ||
      attn_sink.size(0) != kHeads || !attn_sink.is_contiguous()) {
    return reject("attn_sink_layout");
  }

  const int64_t last_absolute_token = q_abs_start + q.size(0) - 1;
  int64_t max_candidates = 0;
  if (compress_ratio == 1) {
    max_candidates = std::min<int64_t>(last_absolute_token + 1, kv.size(0));
  } else if (compress_ratio == 4) {
    max_candidates = kCompressedTopk + kSwaTopk;
  } else {
    max_candidates =
        std::min<int64_t>(compressed_len,
                          (last_absolute_token + 1) / compress_ratio) +
        std::min<int64_t>(kSwaTopk, last_absolute_token + 1);
  }
  const int64_t topk =
      std::max<int64_t>(((max_candidates + 63) / 64) * 64, int64_t{64});

  std::string reason;
  // Prefer the new has_extra=true artifact.  It receives compressed KV and
  // the SWA KV stream independently, eliminating the legacy combined-KV
  // staging copy.  If the artifact is not installed or throws, continue to
  // the validated combined-KV path below.
  auto split_function = try_get_dsv4_mate_fused_dsa_function(
      &reason, "dsa_model1_local_prefill_h64_extra");
  if (split_function.defined()) {
    try {
      const int64_t main_topk =
          compress_ratio == 1
              ? std::max<int64_t>(
                    ((std::min<int64_t>(last_absolute_token + 1, kv.size(0)) +
                      63) /
                     64) *
                        64,
                    64)
          : compress_ratio == 4
              ? kCompressedTopk
              : std::max<int64_t>(
                    ((std::min<int64_t>(
                          compressed_len,
                          (last_absolute_token + 1) / compress_ratio) +
                      63) /
                     64) *
                        64,
                    64);
      // Keep a fixed 128-wide extra descriptor even for C1.  Its runtime
      // length is zero for C1, which avoids zero-sized robust descriptors in
      // the has_extra=true TileLang ABI.
      constexpr int64_t extra_topk = kSwaTopk;
      auto& workspace = dsv4_mate_local_sparse_prefill_workspace(
          q, kv.size(0), compressed_len, main_topk, extra_topk);
      if (compress_ratio == 1) {
        workspace.main_kv.slice(/*dim=*/0, 0, kv.size(0))
            .copy_(kv.view({kv.size(0), 1, kHeadDim}));
      } else {
        workspace.main_kv.slice(/*dim=*/0, 0, compressed_len)
            .copy_(compressed_kv.view({compressed_len, 1, kHeadDim}));
      }
      workspace.extra_kv.slice(/*dim=*/0, 0, kv.size(0))
          .copy_(kv.view({kv.size(0), 1, kHeadDim}));
      workspace.sink.copy_(attn_sink);
      const int32_t* compressed_indices_ptr =
          compress_ratio == 4 ? compressed_indices.data_ptr<int32_t>()
                              : nullptr;
      dsv4_local_sparse_prefill_split_metadata_kernel<<<
          dim3(q.size(0)),
          dim3(kMateCacheTransferBlockSize),
          0,
          current_dsv4_cache_stream()>>>(
          compressed_indices_ptr,
          workspace.indices.data_ptr<int32_t>(),
          workspace.topk_length.data_ptr<int32_t>(),
          workspace.extra_indices.data_ptr<int32_t>(),
          workspace.extra_topk_length.data_ptr<int32_t>(),
          q.size(0),
          kv.size(0),
          compressed_len,
          compress_ratio == 4 ? compressed_indices.size(1) : 0,
          main_topk,
          extra_topk,
          q_abs_start,
          compress_ratio);
      check_musa_kernel_launch("dsv4 kernel");

      bind_dsv4_local_prefill_tvmffi_current_stream(q.device());
      split_function.value()(to_ffi_tensor(q),
                             to_ffi_tensor(workspace.main_kv),
                             to_ffi_tensor(workspace.indices),
                             to_ffi_tensor(workspace.topk_length),
                             to_ffi_tensor(workspace.extra_kv),
                             to_ffi_tensor(workspace.extra_indices),
                             to_ffi_tensor(workspace.extra_topk_length),
                             to_ffi_tensor(workspace.sink),
                             to_ffi_tensor(workspace.output),
                             to_ffi_tensor(workspace.max_logits),
                             to_ffi_tensor(workspace.lse));
      if (output.has_value()) {
        CHECK_EQ(output.value().sizes(), q.sizes());
        output.value().copy_(workspace.output);
        return output.value();
      }
      return workspace.output;
    } catch (const std::exception& ex) {
    }
  }

  // The legacy h16 URI was compiled with 64 heads. Keep using the validated
  // binary until a correctly named H64 artifact can be rebuilt in the
  // portable cache.
  auto function = try_get_dsv4_mate_fused_dsa_function(
      &reason, dsv4_mate_c4_prefill_uri(/*heads=*/16));
  if (!function.defined()) {
    static std::once_flag once;
    std::call_once(once, [&reason]() {
      LOG(WARNING) << "DeepSeek-V4 Mate local sparse prefill fallback: "
                   << reason;
    });
    return torch::Tensor();
  }

  try {
    auto& workspace = dsv4_mate_local_sparse_prefill_workspace(
        q, kv.size(0), compressed_len, topk, 0);
    if (compressed_len > 0) {
      workspace.combined_kv.slice(/*dim=*/0, 0, compressed_len)
          .copy_(compressed_kv.view({compressed_len, 1, kHeadDim}));
    }
    workspace.combined_kv
        .slice(/*dim=*/0, compressed_len, compressed_len + kv.size(0))
        .copy_(kv.view({kv.size(0), 1, kHeadDim}));
    workspace.sink.copy_(attn_sink);
    const int32_t* compressed_indices_ptr =
        compress_ratio == 4 ? compressed_indices.data_ptr<int32_t>() : nullptr;
    dsv4_local_sparse_prefill_metadata_kernel<<<
        dim3(q.size(0)),
        dim3(kMateCacheTransferBlockSize),
        0,
        current_dsv4_cache_stream()>>>(
        compressed_indices_ptr,
        workspace.indices.data_ptr<int32_t>(),
        workspace.topk_length.data_ptr<int32_t>(),
        q.size(0),
        kv.size(0),
        compressed_len,
        compress_ratio == 4 ? compressed_indices.size(1) : 0,
        topk,
        q_abs_start,
        compress_ratio);
    check_musa_kernel_launch("dsv4 kernel");

    bind_dsv4_local_prefill_tvmffi_current_stream(q.device());
    function.value()(to_ffi_tensor(q),
                     to_ffi_tensor(workspace.combined_kv),
                     to_ffi_tensor(workspace.indices),
                     to_ffi_tensor(workspace.topk_length),
                     to_ffi_tensor(workspace.combined_kv),
                     to_ffi_tensor(workspace.empty_indices),
                     to_ffi_tensor(workspace.empty_lengths),
                     to_ffi_tensor(workspace.sink),
                     to_ffi_tensor(workspace.output),
                     to_ffi_tensor(workspace.max_logits),
                     to_ffi_tensor(workspace.lse));
    if (output.has_value()) {
      CHECK_EQ(output.value().sizes(), q.sizes());
      output.value().copy_(workspace.output);
      return output.value();
    }
    return workspace.output;
  } catch (const std::exception& ex) {
    static std::once_flag once;
    std::call_once(once, [&ex]() {
      LOG(WARNING) << "DeepSeek-V4 Mate local sparse prefill exception: "
                   << ex.what();
    });
    return torch::Tensor();
  }
}

int64_t dsv4_paged_decode_warp_fused_max_total_len() {
  return kWarpFusedSafeMaxTotalLen;
}

template <typename scalar_t>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_dense_decode_scores_kernel(const scalar_t* __restrict__ q,
                                    const scalar_t* __restrict__ kv,
                                    const scalar_t* __restrict__ compressed_kv,
                                    float* __restrict__ scores,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    int64_t kv_len,
                                    int64_t compressed_kv_len,
                                    float softmax_scale) {
  const int64_t head = blockIdx.x;
  const int64_t token = blockIdx.y;
  const int64_t total_len = kv_len + compressed_kv_len;
  if (head >= num_heads || token >= total_len) {
    return;
  }

  const scalar_t* row = token < kv_len
                            ? kv + token * head_dim
                            : compressed_kv + (token - kv_len) * head_dim;
  const scalar_t* q_row = q + head * head_dim;

  float sum = 0.0f;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    sum += musa_convert_to_float(q_row[dim]) * musa_convert_to_float(row[dim]);
  }

  __shared__ float shared[kDecodeAttentionBlockSize];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[head * total_len + token] = shared[0] * softmax_scale;
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_dense_decode_context_kernel(const float* __restrict__ scores,
                                     const scalar_t* __restrict__ kv,
                                     const scalar_t* __restrict__ compressed_kv,
                                     const float* __restrict__ attn_sink,
                                     scalar_t* __restrict__ output,
                                     int64_t num_heads,
                                     int64_t head_dim,
                                     int64_t kv_len,
                                     int64_t compressed_kv_len,
                                     bool has_attn_sink) {
  const int64_t head = blockIdx.x;
  const int64_t total_len = kv_len + compressed_kv_len;
  if (head >= num_heads || total_len <= 0) {
    return;
  }

  const float* head_scores = scores + head * total_len;

  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_max = fmaxf(local_max, head_scores[token]);
  }

  __shared__ float shared[kDecodeAttentionBlockSize];
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = shared[0];

  float local_denom = 0.0f;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_denom += expf(head_scores[token] - max_score);
  }
  shared[threadIdx.x] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = shared[0];
  if (has_attn_sink) {
    denom += expf(attn_sink[head] - max_score);
  }
  denom = fmaxf(denom, 1.0e-20f);

  scalar_t* out_head = output + head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < total_len; ++token) {
      const scalar_t* row = token < kv_len
                                ? kv + token * head_dim
                                : compressed_kv + (token - kv_len) * head_dim;
      const float prob = expf(head_scores[token] - max_score) / denom;
      acc += prob * musa_convert_to_float(row[dim]);
    }
    out_head[dim] = static_cast<scalar_t>(acc);
  }
}

template <typename index_t>
__device__ __forceinline__ int64_t
dsv4_paged_flat_row_from_table(const index_t* __restrict__ block_table,
                               int64_t sequence_index,
                               int64_t logical_pos,
                               int64_t block_size,
                               int64_t num_blocks,
                               int64_t max_blocks_per_seq) {
  const int64_t table_idx = logical_pos / block_size;
  const int64_t block_offset = logical_pos % block_size;
  if (table_idx < 0 || table_idx >= max_blocks_per_seq) {
    return int64_t{-1};
  }

  int64_t block_id = static_cast<int64_t>(
      block_table[sequence_index * max_blocks_per_seq + table_idx]);
  int64_t flat = block_id * block_size + block_offset;
  if ((flat < 0 || flat >= num_blocks * block_size) && block_id > 0) {
    flat = (block_id - 1) * block_size + block_offset;
  }
  return (flat >= 0 && flat < num_blocks * block_size) ? flat : int64_t{-1};
}

template <typename index_t>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_paged_row_indices_kernel(const index_t* __restrict__ block_table,
                                  int64_t* __restrict__ row_indices,
                                  int64_t sequence_index,
                                  int64_t kv_start,
                                  int64_t kv_window_len,
                                  int64_t block_size,
                                  int64_t num_blocks,
                                  int64_t max_blocks_per_seq) {
  const int64_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= kv_window_len) {
    return;
  }

  const int64_t logical_pos = kv_start + token;
  row_indices[token] = dsv4_paged_flat_row_from_table(block_table,
                                                      sequence_index,
                                                      logical_pos,
                                                      block_size,
                                                      num_blocks,
                                                      max_blocks_per_seq);
}

#if !defined(USE_DCU)
__device__ __forceinline__ float dsv4_warp_sum(float value) {
  for (int offset = kMusaWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float dsv4_block_max(float value,
                                                float* __restrict__ scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ float dsv4_block_sum(float value,
                                                float* __restrict__ scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t, typename index_t, int32_t kContextSplits = 1,
          int32_t kContextUnroll = 1>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_paged_decode_warp_fused_kernel(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ cache,
        const index_t* __restrict__ block_table,
        const int32_t* __restrict__ dynamic_kv_seq_lens,
        const scalar_t* __restrict__ dense_compressed_kv,
        const scalar_t* __restrict__ compressed_cache,
        const index_t* __restrict__ compressed_block_table,
        const float* __restrict__ attn_sink,
        scalar_t* __restrict__ output,
        int64_t sequence_index,
        int64_t kv_start,
        int64_t kv_window_len,
        int64_t block_size,
        int64_t num_blocks,
        int64_t max_blocks_per_seq,
        int64_t compressed_sequence_index,
        int64_t paged_compressed_kv_len,
        int64_t compressed_block_size,
        int64_t compressed_num_blocks,
        int64_t compressed_max_blocks_per_seq,
        int64_t num_heads,
        int64_t head_dim,
        int64_t dense_compressed_kv_len,
        int64_t row_width,
        int64_t compressed_row_width,
        float softmax_scale,
        bool has_attn_sink,
        bool precompute_probs,
        bool batched_verify,
        int64_t dynamic_compress_ratio) {
  const int64_t head = blockIdx.x;
  const int64_t query = blockIdx.y;
  const int64_t total_len =
      kv_window_len + paged_compressed_kv_len + dense_compressed_kv_len;
  if (head >= num_heads || total_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + total_len;
  // Resolve each paged row once, then reuse its address for all head
  // dimensions. Repeating 64-bit block/page division in the context loop
  // costs far more than the value multiply/add. Keep pointer storage aligned
  // even for an odd scalar-decode history length.
  const int64_t pointer_offset =
      ((total_len + kDecodeAttentionBlockSize + 1) / 2) * 2;
  const scalar_t** row_addresses =
      reinterpret_cast<const scalar_t**>(shared + pointer_offset);

  int64_t runtime_kv_start = kv_start;
  int64_t runtime_kv_window_len = kv_window_len;
  int64_t runtime_compressed_len = paged_compressed_kv_len;
  if (dynamic_kv_seq_lens != nullptr) {
    const int64_t raw_runtime_kv_len =
        static_cast<int64_t>(dynamic_kv_seq_lens[
            batched_verify ? query : sequence_index]);
    const int64_t runtime_kv_len =
        raw_runtime_kv_len > 0 ? raw_runtime_kv_len : 0;
    const int64_t cache_capacity = max_blocks_per_seq * block_size;
    const int64_t effective_runtime_kv_len =
        runtime_kv_len < cache_capacity ? runtime_kv_len : cache_capacity;
    runtime_kv_window_len = effective_runtime_kv_len < kv_window_len
                                ? effective_runtime_kv_len
                                : kv_window_len;
    runtime_kv_start = effective_runtime_kv_len - runtime_kv_window_len;
    if (dynamic_compress_ratio > 0) {
      runtime_compressed_len = min(paged_compressed_kv_len,
                                   runtime_kv_len / dynamic_compress_ratio);
    }
  }

  constexpr int kWarpsPerBlock = kDecodeAttentionBlockSize / kMusaWarpSize;
  const int warp_id = threadIdx.x / kMusaWarpSize;
  const int lane = threadIdx.x % kMusaWarpSize;
  const scalar_t* q_row = q + (query * num_heads + head) * head_dim;

  for (int64_t token = warp_id; token < total_len; token += kWarpsPerBlock) {
    bool row_valid = true;
    const scalar_t* row = nullptr;
    if (token < kv_window_len) {
      row_valid = token < runtime_kv_window_len;
      const int64_t flat =
          row_valid ? dsv4_paged_flat_row_from_table(block_table,
                                                     sequence_index,
                                                     runtime_kv_start + token,
                                                     block_size,
                                                     num_blocks,
                                                     max_blocks_per_seq)
                    : int64_t{-1};
      row_valid = row_valid && flat >= 0;
      row = row_valid ? cache + flat * row_width : cache;
    } else if (token < kv_window_len + paged_compressed_kv_len) {
      const int64_t compressed_token = token - kv_window_len;
      row_valid = compressed_token < runtime_compressed_len;
      const int64_t flat =
          row_valid ? dsv4_paged_flat_row_from_table(compressed_block_table,
                                         compressed_sequence_index,
                                         compressed_token,
                                         compressed_block_size,
                                         compressed_num_blocks,
                                         compressed_max_blocks_per_seq)
                    : int64_t{-1};
      row_valid = row_valid && flat >= 0;
      row = row_valid ? compressed_cache + flat * compressed_row_width
                      : compressed_cache;
    } else {
      row = dense_compressed_kv +
            (token - kv_window_len - paged_compressed_kv_len) * head_dim;
    }

    float sum = 0.0f;
    if (row_valid) {
      for (int64_t dim = lane; dim < head_dim; dim += kMusaWarpSize) {
        sum +=
            musa_convert_to_float(q_row[dim]) * musa_convert_to_float(row[dim]);
      }
      sum = dsv4_warp_sum(sum);
    }
    if (lane == 0) {
      scores[token] = row_valid ? sum * softmax_scale : -INFINITY;
      row_addresses[token] = row_valid ? row : nullptr;
    }
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_max = fmaxf(local_max, scores[token]);
  }
  const float max_score = dsv4_block_max(local_max, reduce);

  float local_denom = 0.0f;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_denom += expf(scores[token] - max_score);
  }
  float denom = dsv4_block_sum(local_denom, reduce);
  if (threadIdx.x == 0 && has_attn_sink) {
    reduce[0] = denom + expf(attn_sink[head] - max_score);
  } else if (threadIdx.x == 0) {
    reduce[0] = denom;
  }
  __syncthreads();
  denom = fmaxf(reduce[0], 1.0e-20f);

  // Every output dimension uses the same softmax probability. Computing it
  // inside the dimension-major context loop repeats total_len exponentials
  // for every head dimension. Reuse the score workspace after the reduction
  // so each token pays for expf and division only once.
  if (precompute_probs) {
    for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
      scores[token] = expf(scores[token] - max_score) / denom;
    }
    __syncthreads();
  }

  scalar_t* out_head = output + (query * num_heads + head) * head_dim;
  // Duplicate the identical score/softmax work, but partition output columns
  // across CTAs. Every output retains the original token accumulation order.
  // The default specialization folds back to the original unsplit loop.
  const int64_t context_tile = (head_dim + kContextSplits - 1) / kContextSplits;
  const int64_t context_begin = kContextSplits == 1
      ? int64_t{0} : static_cast<int64_t>(blockIdx.z) * context_tile;
  const int64_t context_end = kContextSplits == 1
      ? head_dim : min(context_begin + context_tile, head_dim);
  for (int64_t dim = context_begin + threadIdx.x; dim < context_end;
       dim += blockDim.x) {
    float acc = 0.0f;
#pragma unroll(kContextUnroll)
    for (int64_t token = 0; token < total_len; ++token) {
      const scalar_t* row = row_addresses[token];
      if (row == nullptr) {
        continue;
      }
      const float prob = precompute_probs
                             ? scores[token]
                             : expf(scores[token] - max_score) / denom;
      acc += prob * musa_convert_to_float(row[dim]);
    }
    out_head[dim] = static_cast<scalar_t>(acc);
  }
}
#endif  // !defined(USE_DCU)

template <typename scalar_t>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_paged_decode_scores_kernel(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ cache,
        const int64_t* __restrict__ row_indices,
        const scalar_t* __restrict__ dense_compressed_kv,
        const scalar_t* __restrict__ compressed_cache,
        const int64_t* __restrict__ compressed_row_indices,
        float* __restrict__ scores,
        int64_t num_heads,
        int64_t head_dim,
        int64_t kv_window_len,
        int64_t paged_compressed_kv_len,
        int64_t dense_compressed_kv_len,
        int64_t row_width,
        int64_t compressed_row_width,
        float softmax_scale) {
  const int64_t head = blockIdx.x;
  const int64_t token = blockIdx.y;
  const int64_t total_len =
      kv_window_len + paged_compressed_kv_len + dense_compressed_kv_len;
  if (head >= num_heads || token >= total_len) {
    return;
  }

  bool row_valid = true;
  const scalar_t* row = nullptr;
  if (token < kv_window_len) {
    const int64_t flat = row_indices[token];
    row_valid = flat >= 0;
    row = row_valid ? cache + flat * row_width : cache;
  } else if (token < kv_window_len + paged_compressed_kv_len) {
    const int64_t flat = compressed_row_indices[token - kv_window_len];
    row_valid = flat >= 0;
    row = row_valid ? compressed_cache + flat * compressed_row_width
                    : compressed_cache;
  } else {
    row = dense_compressed_kv +
          (token - kv_window_len - paged_compressed_kv_len) * head_dim;
  }
  if (!row_valid) {
    if (threadIdx.x == 0) {
      scores[head * total_len + token] = -INFINITY;
    }
    return;
  }

  const scalar_t* q_row = q + head * head_dim;
  float sum = 0.0f;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    sum += musa_convert_to_float(q_row[dim]) * musa_convert_to_float(row[dim]);
  }

  __shared__ float shared[kDecodeAttentionBlockSize];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[head * total_len + token] = shared[0] * softmax_scale;
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kDecodeAttentionBlockSize, 1)
    dsv4_paged_decode_context_kernel(
        const float* __restrict__ scores,
        const scalar_t* __restrict__ cache,
        const int64_t* __restrict__ row_indices,
        const scalar_t* __restrict__ dense_compressed_kv,
        const scalar_t* __restrict__ compressed_cache,
        const int64_t* __restrict__ compressed_row_indices,
        const float* __restrict__ attn_sink,
        scalar_t* __restrict__ output,
        int64_t num_heads,
        int64_t head_dim,
        int64_t kv_window_len,
        int64_t paged_compressed_kv_len,
        int64_t dense_compressed_kv_len,
        int64_t row_width,
        int64_t compressed_row_width,
        bool has_attn_sink) {
  const int64_t head = blockIdx.x;
  const int64_t total_len =
      kv_window_len + paged_compressed_kv_len + dense_compressed_kv_len;
  if (head >= num_heads || total_len <= 0) {
    return;
  }

  const float* head_scores = scores + head * total_len;

  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_max = fmaxf(local_max, head_scores[token]);
  }

  __shared__ float shared[kDecodeAttentionBlockSize];
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = shared[0];

  float local_denom = 0.0f;
  for (int64_t token = threadIdx.x; token < total_len; token += blockDim.x) {
    local_denom += expf(head_scores[token] - max_score);
  }
  shared[threadIdx.x] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = shared[0];
  if (has_attn_sink) {
    denom += expf(attn_sink[head] - max_score);
  }
  denom = fmaxf(denom, 1.0e-20f);

  scalar_t* out_head = output + head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < total_len; ++token) {
      bool row_valid = true;
      const scalar_t* row = nullptr;
      if (token < kv_window_len) {
        const int64_t flat = row_indices[token];
        row_valid = flat >= 0;
        row = row_valid ? cache + flat * row_width : cache;
      } else if (token < kv_window_len + paged_compressed_kv_len) {
        const int64_t flat = compressed_row_indices[token - kv_window_len];
        row_valid = flat >= 0;
        row = row_valid ? compressed_cache + flat * compressed_row_width
                        : compressed_cache;
      } else {
        row = dense_compressed_kv +
              (token - kv_window_len - paged_compressed_kv_len) * head_dim;
      }
      if (!row_valid) {
        continue;
      }
      const float prob = expf(head_scores[token] - max_score) / denom;
      acc += prob * musa_convert_to_float(row[dim]);
    }
    out_head[dim] = static_cast<scalar_t>(acc);
  }
}

}  // namespace

void deepseek_v4_c128_prefill_metadata(torch::Tensor indices,
                                       torch::Tensor topk_length,
                                       int64_t token_pack,
                                       int64_t compressed_len,
                                       int64_t compiled_compressed_kv_len) {
  constexpr int64_t kTopk = 512;
  CHECK(indices.defined() && topk_length.defined());
  CHECK_EQ(indices.scalar_type(), torch::kInt32);
  CHECK_EQ(topk_length.scalar_type(), torch::kInt32);
  CHECK_EQ(indices.dim(), 3);
  CHECK_EQ(indices.size(1), 1);
  CHECK_EQ(indices.size(2), kTopk);
  CHECK_EQ(topk_length.dim(), 1);
  CHECK_EQ(topk_length.size(0), indices.size(0));
  CHECK(indices.is_contiguous() && topk_length.is_contiguous());
  CHECK_EQ(indices.device(), topk_length.device());
  CHECK_GT(token_pack, 0);
  CHECK_GE(compressed_len, 0);
  CHECK_GE(compiled_compressed_kv_len, compressed_len);

  const int64_t packed_seq = indices.size(0);
  if (packed_seq == 0) {
    return;
  }
  dsv4_c128_prefill_metadata_kernel<<<packed_seq,
                                      256,
                                      0,
                                      current_dsv4_cache_stream()>>>(
      indices.data_ptr<int32_t>(),
      topk_length.data_ptr<int32_t>(),
      packed_seq,
      token_pack,
      compressed_len,
      compiled_compressed_kv_len);
  check_musa_kernel_launch("dsv4 kernel");
}

torch::Tensor deepseek_v4_dense_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  CHECK(q.defined() && kv.defined())
      << "deepseek_v4_dense_attention_decode expects q and kv.";
  CHECK_EQ(q.dim(), 3) << "q must be [1, num_heads, head_dim].";
  CHECK_EQ(q.size(0), 1) << "q must contain exactly one decode token.";
  CHECK_EQ(kv.dim(), 2) << "kv must be [kv_len, head_dim].";
  CHECK_EQ(q.size(2), kv.size(1)) << "q/kv head_dim mismatch.";
  CHECK_EQ(q.scalar_type(), kv.scalar_type()) << "q/kv dtype mismatch.";
  CHECK(q.is_contiguous()) << "q must be contiguous.";
  CHECK(kv.is_contiguous()) << "kv must be contiguous.";

  const bool has_compressed =
      compressed_kv.defined() && compressed_kv.numel() > 0;
  if (has_compressed) {
    CHECK_EQ(compressed_kv.dim(), 2)
        << "compressed_kv must be [compressed_kv_len, head_dim].";
    CHECK_EQ(compressed_kv.size(1), q.size(2))
        << "compressed_kv head_dim mismatch.";
    CHECK_EQ(compressed_kv.scalar_type(), q.scalar_type())
        << "compressed_kv dtype mismatch.";
    CHECK(compressed_kv.is_contiguous()) << "compressed_kv must be contiguous.";
  }

  const bool has_sink = attn_sink.defined() && attn_sink.numel() > 0;
  if (has_sink) {
    CHECK_EQ(attn_sink.dim(), 1) << "attn_sink must be [num_heads].";
    CHECK_EQ(attn_sink.size(0), q.size(1)) << "attn_sink num_heads mismatch.";
    CHECK_EQ(attn_sink.scalar_type(), torch::kFloat32)
        << "attn_sink must be float32.";
    CHECK(attn_sink.is_contiguous()) << "attn_sink must be contiguous.";
  }

  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t kv_len = kv.size(0);
  const int64_t compressed_kv_len = has_compressed ? compressed_kv.size(0) : 0;
  const int64_t total_len = kv_len + compressed_kv_len;

  torch::Tensor out =
      output.has_value() ? output.value() : torch::empty_like(q);
  CHECK(out.defined());
  CHECK(out.sizes() == q.sizes()) << "output must have shape " << q.sizes();
  CHECK_EQ(out.scalar_type(), q.scalar_type()) << "output dtype mismatch.";
  CHECK(out.is_contiguous()) << "output must be contiguous.";

  if (total_len <= 0) {
    out.zero_();
    return out;
  }

  auto scores = torch::empty(
      {num_heads, total_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
  const dim3 score_grid(num_heads, total_len);
  const dim3 block(kDecodeAttentionBlockSize);
  DISPATCH_FLOATING_TYPES(
      q.scalar_type(), "dsv4_dense_decode_scores_kernel", [&] {
        dsv4_dense_decode_scores_kernel<scalar_t>
            <<<score_grid, block, 0, current_dsv4_cache_stream()>>>(
                q.data_ptr<scalar_t>(),
                kv.data_ptr<scalar_t>(),
                has_compressed ? compressed_kv.data_ptr<scalar_t>() : nullptr,
                scores.data_ptr<float>(),
                num_heads,
                head_dim,
                kv_len,
                compressed_kv_len,
                static_cast<float>(softmax_scale));
      });

  const dim3 context_grid(num_heads);
  DISPATCH_FLOATING_TYPES(
      q.scalar_type(), "dsv4_dense_decode_context_kernel", [&] {
        dsv4_dense_decode_context_kernel<scalar_t>
            <<<context_grid, block, 0, current_dsv4_cache_stream()>>>(
                scores.data_ptr<float>(),
                kv.data_ptr<scalar_t>(),
                has_compressed ? compressed_kv.data_ptr<scalar_t>() : nullptr,
                has_sink ? attn_sink.data_ptr<float>() : nullptr,
                out.data_ptr<scalar_t>(),
                num_heads,
                head_dim,
                kv_len,
                compressed_kv_len,
                has_sink);
      });

  return out;
}

static torch::Tensor
deepseek_v4_paged_attention_decode_with_paged_compressed_kv_impl(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& kv_seq_lens,
    int64_t window_size,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_sequence_index,
    int64_t requested_compressed_kv_len,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& retained_mass_output,
    const std::optional<torch::Tensor>& output) {
  CHECK(q.defined() && cache.defined() && block_table.defined())
      << "deepseek_v4_paged_attention_decode expects q/cache/block_table.";
  CHECK_EQ(q.dim(), 3) << "q must be [1, num_heads, head_dim].";
  CHECK_EQ(q.size(0), 1) << "q must contain exactly one decode token.";
  CHECK_GE(cache.dim(), 3)
      << "cache must be [num_blocks, block_size, ..., head_dim].";
  CHECK_EQ(block_table.dim(), 2) << "block_table must be [batch, max_blocks].";
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK_GT(kv_len, 0);
  CHECK(q.is_contiguous()) << "q must be contiguous.";
  CHECK(cache.is_contiguous()) << "cache must be contiguous.";
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64)
      << "block_table must be int32 or int64.";
  CHECK_EQ(q.scalar_type(), cache.scalar_type()) << "q/cache dtype mismatch.";
  CHECK_EQ(q.device(), cache.device()) << "q/cache device mismatch.";

  const bool has_dynamic_kv_len =
      kv_seq_lens.defined() && kv_seq_lens.numel() > 0;
  if (has_dynamic_kv_len) {
    CHECK_EQ(kv_seq_lens.dim(), 1)
        << "graph replay kv_seq_lens must be one-dimensional.";
    CHECK_EQ(kv_seq_lens.scalar_type(), torch::kInt32)
        << "graph replay kv_seq_lens must use int32.";
    CHECK_EQ(kv_seq_lens.device(), q.device())
        << "graph replay kv_seq_lens must be on the query device.";
    CHECK(kv_seq_lens.is_contiguous())
        << "graph replay kv_seq_lens must be contiguous.";
    CHECK_LT(sequence_index, kv_seq_lens.numel());
  }

  const int64_t num_blocks = cache.size(0);
  const int64_t block_size = cache.size(1);
  const int64_t max_blocks_per_seq = block_table.size(1);
  CHECK_GT(num_blocks, 0);
  CHECK_GT(block_size, 0);
  CHECK_GT(max_blocks_per_seq, 0);

  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rows = num_blocks * block_size;
  CHECK_EQ(cache.numel() % rows, 0)
      << "cache cannot be flattened as [blocks * block_size, row_width].";
  const int64_t row_width = cache.numel() / rows;
  CHECK_GE(row_width, head_dim)
      << "cache row width must cover q head_dim, got row_width=" << row_width
      << " head_dim=" << head_dim;

  const bool has_dense_compressed =
      compressed_kv.defined() && compressed_kv.numel() > 0;
  if (has_dense_compressed) {
    CHECK_EQ(compressed_kv.dim(), 2)
        << "compressed_kv must be [compressed_kv_len, head_dim].";
    CHECK_EQ(compressed_kv.size(1), head_dim)
        << "compressed_kv head_dim mismatch.";
    CHECK_EQ(compressed_kv.scalar_type(), q.scalar_type())
        << "compressed_kv dtype mismatch.";
    CHECK_EQ(compressed_kv.device(), q.device())
        << "compressed_kv device mismatch.";
    CHECK(compressed_kv.is_contiguous()) << "compressed_kv must be contiguous.";
  }

  const bool has_paged_compressed = requested_compressed_kv_len > 0 &&
                                    compressed_cache.defined() &&
                                    compressed_block_table.defined();
  int64_t compressed_num_blocks = 0;
  int64_t compressed_block_size = 0;
  int64_t compressed_max_blocks_per_seq = 0;
  int64_t compressed_row_width = head_dim;
  if (has_paged_compressed) {
    CHECK_GE(compressed_cache.dim(), 3)
        << "compressed_cache must be [num_blocks, block_size, ..., head_dim].";
    CHECK_EQ(compressed_block_table.dim(), 2)
        << "compressed_block_table must be [batch, max_blocks].";
    CHECK_GE(compressed_sequence_index, 0);
    CHECK_LT(compressed_sequence_index, compressed_block_table.size(0));
    CHECK(compressed_cache.is_contiguous())
        << "compressed_cache must be contiguous.";
    CHECK(compressed_block_table.scalar_type() == torch::kInt32 ||
          compressed_block_table.scalar_type() == torch::kInt64)
        << "compressed_block_table must be int32 or int64.";
    CHECK_EQ(compressed_cache.scalar_type(), q.scalar_type())
        << "compressed_cache dtype mismatch.";
    CHECK_EQ(compressed_cache.device(), q.device())
        << "compressed_cache device mismatch.";

    compressed_num_blocks = compressed_cache.size(0);
    compressed_block_size = compressed_cache.size(1);
    compressed_max_blocks_per_seq = compressed_block_table.size(1);
    CHECK_GT(compressed_num_blocks, 0);
    CHECK_GT(compressed_block_size, 0);
    CHECK_GT(compressed_max_blocks_per_seq, 0);
    const int64_t compressed_rows =
        compressed_num_blocks * compressed_block_size;
    CHECK_EQ(compressed_cache.numel() % compressed_rows, 0)
        << "compressed_cache cannot be flattened as paged rows.";
    compressed_row_width = compressed_cache.numel() / compressed_rows;
    CHECK_GE(compressed_row_width, head_dim)
        << "compressed cache row width must cover q head_dim.";
  }

  const bool has_sink = attn_sink.defined() && attn_sink.numel() > 0;
  if (has_sink) {
    CHECK_EQ(attn_sink.dim(), 1) << "attn_sink must be [num_heads].";
    CHECK_EQ(attn_sink.size(0), num_heads) << "attn_sink num_heads mismatch.";
    CHECK_EQ(attn_sink.scalar_type(), torch::kFloat32)
        << "attn_sink must be float32.";
    CHECK_EQ(attn_sink.device(), q.device()) << "attn_sink device mismatch.";
    CHECK(attn_sink.is_contiguous()) << "attn_sink must be contiguous.";
  }

  const int64_t cache_capacity = max_blocks_per_seq * block_size;
  const int64_t effective_kv_len = std::min<int64_t>(kv_len, cache_capacity);
  // A captured launch needs fixed tensor and grid shapes. Reserve the whole
  // SWA window and let the fused kernel read the replay-time length from
  // kv_seq_lens, masking unused rows on device.
  const int64_t kv_window_len =
      has_dynamic_kv_len
          ? (window_size > 0 ? std::min<int64_t>(cache_capacity, window_size)
                             : cache_capacity)
          : (window_size > 0 ? std::min<int64_t>(effective_kv_len, window_size)
                             : effective_kv_len);
  const int64_t kv_start =
      has_dynamic_kv_len ? 0 : effective_kv_len - kv_window_len;
  const int64_t dense_compressed_kv_len =
      has_dense_compressed ? compressed_kv.size(0) : int64_t{0};
  const int64_t paged_compressed_kv_len =
      has_paged_compressed ? std::min<int64_t>(requested_compressed_kv_len,
                                               compressed_max_blocks_per_seq *
                                                   compressed_block_size)
                           : int64_t{0};
  const int64_t total_len =
      kv_window_len + paged_compressed_kv_len + dense_compressed_kv_len;
  if (retained_mass_output.has_value() && retained_mass_output->defined()) {
    CHECK(has_dynamic_kv_len && paged_compressed_kv_len == 0 &&
          dense_compressed_kv_len == 0 && true && false)
        << "Deferred sink scaling requires strict dynamic Graph FlashMLA.";
  }

  torch::Tensor out =
      output.has_value() ? output.value() : torch::empty_like(q);
  CHECK(out.defined());
  CHECK(out.sizes() == q.sizes()) << "output must have shape " << q.sizes();
  CHECK_EQ(out.scalar_type(), q.scalar_type()) << "output dtype mismatch.";
  CHECK(out.is_contiguous()) << "output must be contiguous.";

  if (total_len <= 0) {
    out.zero_();
    return out;
  }

  auto block_table_contig = block_table.to(cache.device()).contiguous();
  auto compressed_block_table_contig =
      has_paged_compressed ? compressed_block_table.to(cache.device())
                                 .to(block_table.scalar_type())
                                 .contiguous()
                           : block_table_contig;

#if !defined(USE_DCU)
  if ((has_dynamic_kv_len || kv_start == 0) && paged_compressed_kv_len == 0 &&
      dense_compressed_kv_len == 0) {
    auto mate_output = try_dsv4_mate_flashmla_decode(
        q,
        cache,
        block_table_contig,
        sequence_index,
        effective_kv_len,
        has_dynamic_kv_len ? kv_seq_lens : torch::Tensor(),
        has_dynamic_kv_len ? kv_window_len : int64_t{0},
        attn_sink,
        softmax_scale,
        retained_mass_output,
        out);
    if (mate_output.defined()) {
      return out;
    }
  }
#endif

#if !defined(USE_DCU)
  const int64_t warp_fused_max_total_len =
      dsv4_paged_decode_warp_fused_max_total_len();
  // Dynamic Graph replay cannot fall through to the allocation-based
  // multi-kernel path below.  Keep the proven 512-row default, but honor the
  // existing explicit unsafe-long opt-in for targeted validation of larger
  // fixed-capacity Graphs (for example C4 at 2.5K: 128 SWA + 625 compressed).
  const bool use_graph_dynamic_fused = has_dynamic_kv_len &&
                                       warp_fused_max_total_len > 0 &&
                                       total_len <= warp_fused_max_total_len;
  CHECK(!has_dynamic_kv_len || use_graph_dynamic_fused)
      << "graph replay paged decode exceeds the safe fixed fused capacity: "
      << total_len;
  if (use_graph_dynamic_fused ||
      (dsv4_paged_decode_warp_fused_enabled() && warp_fused_max_total_len > 0 &&
       total_len <= warp_fused_max_total_len)) {
    const dim3 fused_grid(num_heads);
    const dim3 fused_block(kDecodeAttentionBlockSize);
    const size_t shared_bytes =
        static_cast<size_t>((total_len + kDecodeAttentionBlockSize + 1) / 2) *
            2 * sizeof(float) +
        static_cast<size_t>(total_len) * sizeof(void*);
    const auto stream = current_dsv4_cache_stream();

    if (block_table.scalar_type() == torch::kInt32) {
      DISPATCH_FLOATING_TYPES(
          q.scalar_type(), "dsv4_paged_decode_warp_fused_i32", [&] {
            dsv4_paged_decode_warp_fused_kernel<scalar_t, int32_t>
                <<<fused_grid, fused_block, shared_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    cache.data_ptr<scalar_t>(),
                    block_table_contig.data_ptr<int32_t>(),
                    has_dynamic_kv_len ? kv_seq_lens.data_ptr<int32_t>()
                                       : nullptr,
                    has_dense_compressed ? compressed_kv.data_ptr<scalar_t>()
                                         : nullptr,
                    has_paged_compressed ? compressed_cache.data_ptr<scalar_t>()
                                         : nullptr,
                    compressed_block_table_contig.data_ptr<int32_t>(),
                    has_sink ? attn_sink.data_ptr<float>() : nullptr,
                    out.data_ptr<scalar_t>(),
                    sequence_index,
                    kv_start,
                    kv_window_len,
                    block_size,
                    num_blocks,
                    max_blocks_per_seq,
                    compressed_sequence_index,
                    paged_compressed_kv_len,
                    compressed_block_size,
                    compressed_num_blocks,
                    compressed_max_blocks_per_seq,
                    num_heads,
                    head_dim,
                    dense_compressed_kv_len,
                    row_width,
                    compressed_row_width,
                    static_cast<float>(softmax_scale),
                    has_sink,
                    dsv4_paged_decode_precompute_probs_enabled(),
                    /*batched_verify=*/false,
                    /*dynamic_compress_ratio=*/0);
          });
    } else {
      DISPATCH_FLOATING_TYPES(
          q.scalar_type(), "dsv4_paged_decode_warp_fused_i64", [&] {
            dsv4_paged_decode_warp_fused_kernel<scalar_t, int64_t>
                <<<fused_grid, fused_block, shared_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    cache.data_ptr<scalar_t>(),
                    block_table_contig.data_ptr<int64_t>(),
                    has_dynamic_kv_len ? kv_seq_lens.data_ptr<int32_t>()
                                       : nullptr,
                    has_dense_compressed ? compressed_kv.data_ptr<scalar_t>()
                                         : nullptr,
                    has_paged_compressed ? compressed_cache.data_ptr<scalar_t>()
                                         : nullptr,
                    compressed_block_table_contig.data_ptr<int64_t>(),
                    has_sink ? attn_sink.data_ptr<float>() : nullptr,
                    out.data_ptr<scalar_t>(),
                    sequence_index,
                    kv_start,
                    kv_window_len,
                    block_size,
                    num_blocks,
                    max_blocks_per_seq,
                    compressed_sequence_index,
                    paged_compressed_kv_len,
                    compressed_block_size,
                    compressed_num_blocks,
                    compressed_max_blocks_per_seq,
                    num_heads,
                    head_dim,
                    dense_compressed_kv_len,
                    row_width,
                    compressed_row_width,
                    static_cast<float>(softmax_scale),
                    has_sink,
                    dsv4_paged_decode_precompute_probs_enabled(),
                    /*batched_verify=*/false,
                    /*dynamic_compress_ratio=*/0);
          });
    }
    check_musa_kernel_launch("dsv4 kernel");
    if (dsv4_paged_decode_sync_enabled()) {
      check_musa_runtime(musaStreamSynchronize(stream));
    }
    return out;
  }
#endif  // !defined(USE_DCU)

  auto row_indices = torch::empty(
      {kv_window_len},
      torch::TensorOptions().dtype(torch::kLong).device(q.device()));
  auto compressed_row_indices = torch::empty(
      {paged_compressed_kv_len},
      torch::TensorOptions().dtype(torch::kLong).device(q.device()));
  auto scores = torch::empty(
      {num_heads, total_len},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
  const dim3 score_grid(num_heads, total_len);
  const dim3 context_grid(num_heads);
  const dim3 block(kDecodeAttentionBlockSize);
  const auto stream = current_dsv4_cache_stream();

  if (kv_window_len > 0) {
    const dim3 index_grid(
        ceil_div<int64_t>(kv_window_len, kDecodeAttentionBlockSize));
    if (block_table_contig.scalar_type() == torch::kInt32) {
      dsv4_paged_row_indices_kernel<int32_t><<<index_grid, block, 0, stream>>>(
          block_table_contig.data_ptr<int32_t>(),
          row_indices.data_ptr<int64_t>(),
          sequence_index,
          kv_start,
          kv_window_len,
          block_size,
          num_blocks,
          max_blocks_per_seq);
    } else {
      dsv4_paged_row_indices_kernel<int64_t><<<index_grid, block, 0, stream>>>(
          block_table_contig.data_ptr<int64_t>(),
          row_indices.data_ptr<int64_t>(),
          sequence_index,
          kv_start,
          kv_window_len,
          block_size,
          num_blocks,
          max_blocks_per_seq);
    }
    check_musa_kernel_launch("dsv4 kernel");
    if (dsv4_paged_decode_sync_enabled()) {
      check_musa_runtime(musaStreamSynchronize(stream));
    }
  }

  if (paged_compressed_kv_len > 0) {
    const dim3 compressed_index_grid(
        ceil_div<int64_t>(paged_compressed_kv_len, kDecodeAttentionBlockSize));
    if (compressed_block_table_contig.scalar_type() == torch::kInt32) {
      dsv4_paged_row_indices_kernel<int32_t>
          <<<compressed_index_grid, block, 0, stream>>>(
              compressed_block_table_contig.data_ptr<int32_t>(),
              compressed_row_indices.data_ptr<int64_t>(),
              compressed_sequence_index,
              /*kv_start=*/0,
              paged_compressed_kv_len,
              compressed_block_size,
              compressed_num_blocks,
              compressed_max_blocks_per_seq);
    } else {
      dsv4_paged_row_indices_kernel<int64_t>
          <<<compressed_index_grid, block, 0, stream>>>(
              compressed_block_table_contig.data_ptr<int64_t>(),
              compressed_row_indices.data_ptr<int64_t>(),
              compressed_sequence_index,
              /*kv_start=*/0,
              paged_compressed_kv_len,
              compressed_block_size,
              compressed_num_blocks,
              compressed_max_blocks_per_seq);
    }
    check_musa_kernel_launch("dsv4 kernel");
  }

  DISPATCH_FLOATING_TYPES(q.scalar_type(), "dsv4_paged_decode_attention", [&] {
    dsv4_paged_decode_scores_kernel<scalar_t><<<score_grid, block, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        row_indices.data_ptr<int64_t>(),
        has_dense_compressed ? compressed_kv.data_ptr<scalar_t>() : nullptr,
        has_paged_compressed ? compressed_cache.data_ptr<scalar_t>() : nullptr,
        compressed_row_indices.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        num_heads,
        head_dim,
        kv_window_len,
        paged_compressed_kv_len,
        dense_compressed_kv_len,
        row_width,
        compressed_row_width,
        static_cast<float>(softmax_scale));
    check_musa_kernel_launch("dsv4 kernel");
    if (dsv4_paged_decode_sync_enabled()) {
      check_musa_runtime(musaStreamSynchronize(stream));
    }
    dsv4_paged_decode_context_kernel<scalar_t>
        <<<context_grid, block, 0, stream>>>(
            scores.data_ptr<float>(),
            cache.data_ptr<scalar_t>(),
            row_indices.data_ptr<int64_t>(),
            has_dense_compressed ? compressed_kv.data_ptr<scalar_t>() : nullptr,
            has_paged_compressed ? compressed_cache.data_ptr<scalar_t>()
                                 : nullptr,
            compressed_row_indices.data_ptr<int64_t>(),
            has_sink ? attn_sink.data_ptr<float>() : nullptr,
            out.data_ptr<scalar_t>(),
            num_heads,
            head_dim,
            kv_window_len,
            paged_compressed_kv_len,
            dense_compressed_kv_len,
            row_width,
            compressed_row_width,
            has_sink);
    check_musa_kernel_launch("dsv4 kernel");
    if (dsv4_paged_decode_sync_enabled()) {
      check_musa_runtime(musaStreamSynchronize(stream));
    }
  });
  return out;
}

torch::Tensor deepseek_v4_paged_attention_sas(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output) {
#if defined(USE_DCU)
  return torch::Tensor();
#else
  CHECK_EQ(q.dim(), 3);
  CHECK_EQ(q.size(2), 512);
  CHECK_GT(q.size(0), 0);
  CHECK_GT(q.size(1), 0);
  CHECK_EQ(cache.scalar_type(), q.scalar_type());
  CHECK_GE(cache.dim(), 3);
  CHECK_GE(cache.size(-1), q.size(2));
  CHECK_EQ(cache.numel(), cache.size(0) * cache.size(1) * cache.size(-1));
  CHECK_EQ(block_table.dim(), 2);
  CHECK_GE(sequence_index, 0);
  CHECK_LT(sequence_index, block_table.size(0));
  CHECK(block_table.scalar_type() == torch::kInt32 || block_table.scalar_type() == torch::kInt64);
  CHECK_EQ(output.sizes(), q.sizes());
  CHECK_EQ(output.scalar_type(), q.scalar_type());
  CHECK(!q.is_alias_of(output));
  CHECK_GT(kv_len, 0);
  CHECK_LE(kv_len, block_table.size(1) * cache.size(1));
  CHECK_GT(window_size, 0);
  CHECK_LE(window_size, 512);
  CHECK_EQ(attn_sink.scalar_type(), torch::kFloat32);
  CHECK_EQ(attn_sink.numel(), q.size(1));
  for (const torch::Tensor& tensor : {q, cache, block_table, attn_sink, output}) {
    CHECK(tensor.is_contiguous());
    CHECK_EQ(tensor.device(), q.device());
  }
  const int64_t visible = std::min(kv_len, window_size);
  const size_t shared_bytes =
      ((visible + kDecodeAttentionBlockSize + 1) / 2) * 2 * sizeof(float) +
      visible * sizeof(void*);
  const musaStream_t stream = current_dsv4_cache_stream();
  DISPATCH_FLOATING_TYPES(q.scalar_type(), "dsv4_paged_sas", [&] {
    auto launch = [&](auto index_tag) {
      using Index = decltype(index_tag);
      dsv4_paged_decode_warp_fused_kernel<scalar_t, Index, 2>
          <<<dim3(q.size(1), q.size(0), 2), kDecodeAttentionBlockSize, shared_bytes, stream>>>(
              q.data_ptr<scalar_t>(), cache.data_ptr<scalar_t>(),
              block_table.data_ptr<Index>(), nullptr, nullptr, nullptr, nullptr,
              attn_sink.data_ptr<float>(), output.data_ptr<scalar_t>(),
              sequence_index, kv_len - visible, visible,
              cache.size(1), cache.size(0), block_table.size(1),
              /*compressed_sequence_index=*/0, /*paged_compressed_kv_len=*/0,
              /*compressed_block_size=*/0, /*compressed_num_blocks=*/0,
              /*compressed_max_blocks_per_seq=*/0, q.size(1), q.size(2),
              /*dense_compressed_kv_len=*/0, cache.size(-1), /*compressed_row_width=*/0,
              static_cast<float>(softmax_scale), /*has_attn_sink=*/true,
              /*precompute_probs=*/true, /*batched_verify=*/true,
              /*dynamic_compress_ratio=*/0);
    };
    if (block_table.scalar_type() == torch::kInt32) {
      launch(int32_t{0});
    } else {
      launch(int64_t{0});
    }
  });
  check_musa_kernel_launch("dsv4 paged noncausal SAS");
  return output;
#endif
}

torch::Tensor deepseek_v4_paged_attention_verify_compressed(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_capacity,
    int64_t compression_ratio,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output) {
#if defined(USE_DCU)
  return torch::Tensor();
#else
  CHECK_EQ(q.dim(), 3);
  CHECK_GT(q.size(0), 0);
  CHECK_GT(q.size(1), 0);
  CHECK_GT(q.size(2), 0);
  CHECK_EQ(kv_seq_lens.dim(), 1);
  CHECK_EQ(kv_seq_lens.size(0), q.size(0));
  CHECK_EQ(kv_seq_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(output.sizes(), q.sizes());
  CHECK_EQ(output.scalar_type(), q.scalar_type());
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(compressed_block_table.dim(), 2);
  CHECK_EQ(block_table.size(0), 1);
  CHECK_EQ(compressed_block_table.size(0), 1);
  CHECK(block_table.scalar_type() == torch::kInt32 ||
        block_table.scalar_type() == torch::kInt64);
  CHECK_EQ(compressed_block_table.scalar_type(), block_table.scalar_type());
  CHECK_GE(compressed_capacity, 0);
  CHECK_LE(compressed_capacity, 384);
  CHECK(compression_ratio == 4 || compression_ratio == 128);
  for (const torch::Tensor& tensor : {q, cache, block_table, kv_seq_lens,
                                    compressed_cache, compressed_block_table,
                                    attn_sink, output}) {
    CHECK(tensor.is_contiguous());
    CHECK_EQ(tensor.device(), q.device());
  }
  for (const torch::Tensor& values : {cache, compressed_cache}) {
    CHECK_GE(values.dim(), 3);
    CHECK_EQ(values.scalar_type(), q.scalar_type());
    CHECK_GE(values.size(-1), q.size(2));
    CHECK_GT(values.size(0), 0);
    CHECK_GT(values.size(1), 0);
    CHECK_EQ(values.numel(), values.size(0) * values.size(1) * values.size(-1));
  }
  CHECK_LE(compressed_capacity,
           compressed_block_table.size(1) * compressed_cache.size(1));
  CHECK_EQ(attn_sink.scalar_type(), torch::kFloat32);
  CHECK_EQ(attn_sink.numel(), q.size(1));
  constexpr int64_t kWindow = 128;
  int32_t context_splits = 1;
  const char* splits_env = std::getenv("XLLM_DSV4_C128_CONTEXT_SPLITS");
  if (splits_env != nullptr) {
    const std::string requested(splits_env);
    CHECK(requested == "1" || requested == "2" || requested == "4" || requested == "8")
        << "C128 context splits must be 1, 2, 4 or 8";
    context_splits = static_cast<int32_t>(requested[0] - '0');
  }
  // Multiple CTAs must not overwrite Q while another split still reads it.
  if (q.is_alias_of(output)) {
    context_splits = 1;
  }
  const dim3 block(kDecodeAttentionBlockSize);
  const size_t shared_bytes =
      ((kWindow + compressed_capacity + kDecodeAttentionBlockSize + 1) / 2) *
          2 * sizeof(float) +
      (kWindow + compressed_capacity) * sizeof(void*);
  const musaStream_t stream = current_dsv4_cache_stream();
  DISPATCH_FLOATING_TYPES(q.scalar_type(), "dsv4_paged_verify_c128", [&] {
    auto launch = [&](auto index_tag, auto split_tag, auto unroll_tag) {
      using Index = decltype(index_tag);
      constexpr int32_t kSplits = decltype(split_tag)::value;
      constexpr int32_t kUnroll = decltype(unroll_tag)::value;
      const dim3 grid(q.size(1), q.size(0), kSplits);
      dsv4_paged_decode_warp_fused_kernel<scalar_t, Index, kSplits, kUnroll>
          <<<grid, block, shared_bytes, stream>>>(
              q.data_ptr<scalar_t>(), cache.data_ptr<scalar_t>(),
              block_table.data_ptr<Index>(), kv_seq_lens.data_ptr<int32_t>(),
              nullptr, compressed_cache.data_ptr<scalar_t>(),
              compressed_block_table.data_ptr<Index>(),
              attn_sink.data_ptr<float>(), output.data_ptr<scalar_t>(),
              /*sequence_index=*/0, /*kv_start=*/0, kWindow,
              cache.size(1), cache.size(0), block_table.size(1),
              /*compressed_sequence_index=*/0, compressed_capacity,
              compressed_cache.size(1), compressed_cache.size(0),
              compressed_block_table.size(1), q.size(1), q.size(2),
              /*dense_compressed_kv_len=*/0, cache.size(-1),
              compressed_cache.size(-1), static_cast<float>(softmax_scale),
              /*has_attn_sink=*/true, /*precompute_probs=*/true,
              /*batched_verify=*/true, compression_ratio);
    };
    auto dispatch_splits = [&](auto index_tag, auto unroll_tag) {
      switch (context_splits) {
        case 2: launch(index_tag, std::integral_constant<int32_t, 2>{}, unroll_tag); break;
        case 4: launch(index_tag, std::integral_constant<int32_t, 4>{}, unroll_tag); break;
        case 8: launch(index_tag, std::integral_constant<int32_t, 8>{}, unroll_tag); break;
        default: launch(index_tag, std::integral_constant<int32_t, 1>{}, unroll_tag); break;
      }
    };
    const char* unroll_env = std::getenv("XLLM_DSV4_C128_CONTEXT_UNROLL");
    const std::string unroll = unroll_env == nullptr ? "1" : unroll_env;
    CHECK(unroll == "1" || unroll == "2" || unroll == "4" || unroll == "8");
    auto dispatch_unroll = [&](auto index_tag) {
      if (unroll == "2") {
        dispatch_splits(index_tag, std::integral_constant<int32_t, 2>{});
      } else if (unroll == "4") {
        dispatch_splits(index_tag, std::integral_constant<int32_t, 4>{});
      } else if (unroll == "8") {
        dispatch_splits(index_tag, std::integral_constant<int32_t, 8>{});
      } else {
        dispatch_splits(index_tag, std::integral_constant<int32_t, 1>{});
      }
    };
    if (block_table.scalar_type() == torch::kInt32) {
      dispatch_unroll(int32_t{0});
    } else {
      dispatch_unroll(int64_t{0});
    }
  });
  check_musa_kernel_launch("dsv4 paged verify compressed");
  return output;
#endif
}

torch::Tensor deepseek_v4_paged_attention_verify_c128(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_capacity,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const torch::Tensor& output) {
  return deepseek_v4_paged_attention_verify_compressed(
      q, cache, block_table, kv_seq_lens, compressed_cache,
      compressed_block_table, compressed_capacity, /*compression_ratio=*/128,
      attn_sink, softmax_scale, output);
}

torch::Tensor deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    const torch::Tensor& kv_seq_lens,
    int64_t window_size,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_cache,
    const torch::Tensor& compressed_block_table,
    int64_t compressed_sequence_index,
    int64_t compressed_kv_len,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& retained_mass_output,
    const std::optional<torch::Tensor>& output) {
  return deepseek_v4_paged_attention_decode_with_paged_compressed_kv_impl(
      q,
      cache,
      block_table,
      sequence_index,
      kv_len,
      kv_seq_lens,
      window_size,
      compressed_kv,
      compressed_cache,
      compressed_block_table,
      compressed_sequence_index,
      compressed_kv_len,
      attn_sink,
      softmax_scale,
      retained_mass_output,
      output);
}

torch::Tensor deepseek_v4_paged_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t sequence_index,
    int64_t kv_len,
    int64_t window_size,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return deepseek_v4_paged_attention_decode_with_paged_compressed_kv(
      q,
      cache,
      block_table,
      sequence_index,
      kv_len,
      torch::Tensor(),
      window_size,
      compressed_kv,
      torch::Tensor(),
      torch::Tensor(),
      /*compressed_sequence_index=*/0,
      /*compressed_kv_len=*/0,
      attn_sink,
      softmax_scale,
      std::nullopt,
      output);
}

torch::Tensor deepseek_v4_mate_sparse_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    int64_t topk_length,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    int64_t extra_topk_length,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return try_dsv4_mate_model1_sparse_decode(q,
                                            packed_cache,
                                            physical_indices,
                                            topk_length,
                                            extra_packed_cache,
                                            extra_physical_indices,
                                            extra_topk_length,
                                            attn_sink,
                                            softmax_scale,
                                            output);
}

torch::Tensor deepseek_v4_mate_sparse_attention_decode_batched(
    const torch::Tensor& q,
    const torch::Tensor& packed_cache,
    const torch::Tensor& physical_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& extra_packed_cache,
    const torch::Tensor& extra_physical_indices,
    const torch::Tensor& extra_topk_lengths,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return try_dsv4_mate_model1_sparse_decode(q,
                                            packed_cache,
                                            physical_indices,
                                            /*topk_length=*/0,
                                            extra_packed_cache,
                                            extra_physical_indices,
                                            /*extra_topk_length=*/0,
                                            attn_sink,
                                            softmax_scale,
                                            output,
                                            topk_lengths,
                                            extra_topk_lengths);
}

void deepseek_v4_mate_sparse_attention_set_metadata_key(int64_t metadata_key) {
  dsv4_mate_model1_metadata_key = metadata_key;
}
void deepseek_v4_mate_sparse_attention_set_metadata_group(int64_t metadata_group) {
  dsv4_mate_model1_metadata_group = metadata_group;
}

torch::Tensor deepseek_v4_mate_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return try_dsv4_mate_model1_sparse_prefill(
      q, kv, compressed_kv, attn_sink, softmax_scale, output);
}

torch::Tensor deepseek_v4_mate_c4_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return try_dsv4_mate_model1_c4_sparse_prefill(q,
                                                kv,
                                                compressed_kv,
                                                compressed_indices,
                                                attn_sink,
                                                softmax_scale,
                                                output);
}

torch::Tensor deepseek_v4_mate_local_sparse_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& compressed_kv,
    const torch::Tensor& compressed_indices,
    const torch::Tensor& attn_sink,
    int64_t q_abs_start,
    int64_t compress_ratio,
    double softmax_scale,
    const std::optional<torch::Tensor>& output) {
  return try_dsv4_mate_model1_local_sparse_prefill(q,
                                                   kv,
                                                   compressed_kv,
                                                   compressed_indices,
                                                   attn_sink,
                                                   q_abs_start,
                                                   compress_ratio,
                                                   softmax_scale,
                                                   output);
}

}  // namespace xllm::kernel::musa
