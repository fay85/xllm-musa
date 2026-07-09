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

#include "cuda_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#if defined(XLLM_TORCH_MUSA)
#include <musa_runtime_api.h>
#endif
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#if defined(XLLM_TORCH_MUSA)
#include "core/layers/musa/flashinfer_planinfo.h"
#else
#include "core/layers/cuda/flashinfer_planinfo.h"
#endif
#if !defined(XLLM_TORCH_MUSA)
#include "core/layers/cuda/xattention_planinfo.h"
#endif
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/device.h"
#if !defined(XLLM_TORCH_MUSA)
#include "core/platform/shared_vmm_allocator.h"
#include "core/platform/vmm_torch_allocator.h"
#endif
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"
#include "kernels/cuda/global_capture_instance.h"
#if defined(XLLM_TORCH_MUSA)
#include "kernels/musa/musa_tvmffi_stream.h"
#else
#include "kernels/cuda/utils.h"
#endif

namespace xllm::runtime::cuda {

namespace {

const bool s_enable_graph_timing = [] {
  const char* env = std::getenv("XLLM_GRAPH_TIMING");
  return env != nullptr && std::string(env) == "1";
}();

struct GraphPoolMemoryUsage {
  size_t reserved_bytes = 0;
  size_t allocated_bytes = 0;
  size_t active_bytes = 0;
};

GraphPoolMemoryUsage get_graph_pool_memory_usage(
    c10::DeviceIndex device_index,
    const at::cuda::MempoolId_t& pool_id) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index ||
        segment.owner_private_pool_id != pool_id) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

GraphPoolMemoryUsage get_private_pools_memory_usage(
    c10::DeviceIndex device_index) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index) {
      continue;
    }
    if (segment.owner_private_pool_id == at::cuda::MempoolId_t{0, 0}) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

size_t get_allocator_reserved_bytes(c10::DeviceIndex device_index) {
  const auto device_stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
  const size_t stat_index =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return static_cast<size_t>(device_stats.reserved_bytes[stat_index].current);
}

// Accepts CUDA / torch_musa tensors of either int32 OR int64 dtype. Returns
// true iff the tensor is suitable for the LLM-decode fast path (we'll cast
// any int64 input down to int32 inside the fast-path host wrapper before the
// kernel call). int64 acceptance is necessary because PyTorch's default
// integer dtype is int64, and torch_musa returns positions / tokens as int64
// without an explicit cast at the upstream metadata-builder boundary -- the
// values are tiny (positions <= max_position_embeddings, vocab ids, block
// ids) and always fit in int32, but the containers come in 8-byte.
//
// Without this relaxation the fast path was always rejected in graph mode on
// torch_musa, forcing the slow path's per-call `.slice(0, 0, N).copy_(...)`
// chain for the persistent paged-KV mirrors (correct functionally but adds
// a per-layer-per-step host overhead).
bool is_cuda_contiguous_int_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_contiguous()) {
    return false;
  }
  const auto sc = tensor.scalar_type();
  if (sc != torch::kInt32 && sc != torch::kInt64) {
    return false;
  }
  // Accept CUDA tensors as well as torch_musa's MUSA tensors. On torch_musa
  // 2.7/2.9 the MUSA backend uses device_type=PrivateUse1, so `is_cuda()`
  // returns false and the legacy check would unconditionally fall back to
  // the slow path.
  const auto dt = tensor.device().type();
  return dt == c10::DeviceType::CUDA || dt == c10::DeviceType::PrivateUse1;
}


bool is_cpu_int_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.device().is_cpu() || tensor.numel() == 0) {
    return false;
  }
  const auto sc = tensor.scalar_type();
  return sc == torch::kInt32 || sc == torch::kInt64;
}

const int32_t* host_int32_data_ptr(const torch::Tensor& tensor,
                                   std::vector<int32_t>* cast_buf) {
  if (!is_cpu_int_tensor(tensor)) {
    return nullptr;
  }
  if (tensor.scalar_type() == torch::kInt32) {
    return tensor.contiguous().data_ptr<int32_t>();
  }
  CHECK(cast_buf != nullptr) << "host int64 tensor requires cast buffer";
  const torch::Tensor i32 = tensor.to(torch::kInt32).contiguous();
  cast_buf->assign(i32.data_ptr<int32_t>(),
                   i32.data_ptr<int32_t>() + i32.numel());
  return cast_buf->data();
}

bool has_llm_decode_host_metadata(const AttentionHostInput& host) {
  return !host.kv_seq_lens.empty() && is_cpu_int_tensor(host.paged_kv_indptr) &&
         is_cpu_int_tensor(host.paged_kv_indices) &&
         is_cpu_int_tensor(host.paged_kv_last_page_len);
}

}  // namespace

// CudaGraphPersistentParam implementation
CudaGraphPersistentParam::CudaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = options.max_tokens_per_batch();
  // num_sequences
  int64_t max_seqs_per_batch;
  if (is_rec_multi_round_mode()) {
    // max_seqs_per_batch is the max sequence count per Batch in a scheduler
    // group.
    // When is_rec_multi_round_mode() == true, multiply by beam_width to account
    // for beam search.
    max_seqs_per_batch = options.max_seqs_per_batch() * options_.beam_width();
  } else {
    max_seqs_per_batch = options.max_seqs_per_batch();
  }
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_linear_state_indices_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_num_accepted_tokens_ = torch::ones(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::full({max_seqs_per_batch, max_block_table_len},
                  -1,
                  torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Cuda graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks
  // Estimate max blocks: max_seqs_per_batch * max_block_table_len
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...,
  // max_seqs_per_batch]
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt).device(device));
  persistent_kv_seq_lens_delta_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  // will be updated by q_cu_seq_lens, q_cu_seq_lens is the cumulative sum of
  // q_seq_lens
  persistent_chunked_prefill_qo_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  // aux_hidden_states will be lazily initialized when needed

  expanded_kv_seq_lens_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_expanded_block_tables_ =
      torch::full({max_tokens_per_batch, max_block_table_len},
                  -1,
                  torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_indptr_ = torch::zeros(
      {max_tokens_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));
  persistent_expanded_paged_kv_last_page_len_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
}

bool CudaGraphPersistentParam::can_use_llm_decode_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) const {
  if (!params.meta.batch_forward_type.is_decode() ||
      is_rec_multi_round_mode() || params.has_llmrec_params()) {
    return false;
  }
  const bool device_token_metadata_ok =
      is_cuda_contiguous_int_tensor(tokens) &&
      is_cuda_contiguous_int_tensor(positions) &&
      is_cuda_contiguous_int_tensor(params.attention.device.new_cache_slots);
  if (!device_token_metadata_ok) {
    return false;
  }
  if (has_llm_decode_host_metadata(params.attention.host)) {
    return true;
  }
  return is_cuda_contiguous_int_tensor(params.attention.device.kv_seq_lens) &&
         is_cuda_contiguous_int_tensor(
             params.attention.device.paged_kv_indptr) &&
         is_cuda_contiguous_int_tensor(
             params.attention.device.paged_kv_indices) &&
         is_cuda_contiguous_int_tensor(
             params.attention.device.paged_kv_last_page_len);
}

void CudaGraphPersistentParam::update_llm_decode_metadata_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    int64_t actual_batch_size,
    int64_t actual_num_tokens) {
  CHECK_GE(actual_batch_size, 0) << "actual_batch_size must be >= 0";
  CHECK_GE(actual_num_tokens, 0) << "actual_num_tokens must be >= 0";

  // The fast-path kernel reads all metadata as `int32_t*` (small indices --
  // block ids, positions, vocab ids, etc. always fit in int32, and int32
  // halves the metadata bandwidth + register pressure in the captured
  // attention loop). On torch_musa positions / tokens / paged_kv_* may arrive
  // as int64 (PyTorch's default integer dtype). Cast eagerly here, before
  // graph capture/replay, so the captured forward sees a stable int32
  // device pointer with valid data. The cast tensors are small (worst case
  // `bucket_num_tokens` ints, ~tens of bytes per step for decode bucket=1)
  // and allocated outside any captured region.
  auto to_int32 = [](const torch::Tensor& t) -> torch::Tensor {
    if (!t.defined() || t.scalar_type() == torch::kInt32) {
      return t;
    }
    return t.to(torch::kInt32);
  };
  const torch::Tensor tokens_i32 = to_int32(tokens);
  const torch::Tensor positions_i32 = to_int32(positions);
  const torch::Tensor new_cache_slots_i32 =
      to_int32(params.attention.device.new_cache_slots);
  const torch::Tensor kv_seq_lens_i32 =
      to_int32(params.attention.device.kv_seq_lens);

  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_.index());
  if (has_llm_decode_host_metadata(params.attention.host)) {
    const auto& host = params.attention.host;
    std::vector<int32_t> indptr_cast;
    std::vector<int32_t> indices_cast;
    std::vector<int32_t> last_page_cast;
    const int32_t* host_indptr =
        host_int32_data_ptr(host.paged_kv_indptr, &indptr_cast);
    const int32_t* host_indices =
        host_int32_data_ptr(host.paged_kv_indices, &indices_cast);
    const int32_t* host_last_page =
        host_int32_data_ptr(host.paged_kv_last_page_len, &last_page_cast);
    CHECK(host_indptr != nullptr && host_indices != nullptr &&
          host_last_page != nullptr)
        << "host paged-KV mirrors must be int32/int64 CPU tensors";
    CHECK_GE(static_cast<int64_t>(host.kv_seq_lens.size()),
             actual_batch_size + 1)
        << "host kv_seq_lens too small for batch";
    const int64_t actual_indices_size =
        host.paged_kv_indices.defined() ? host.paged_kv_indices.numel() : 0;
    xllm::kernel::cuda::LlmDecodeMetadataHostUpdateParams host_update_params{
        .src_tokens = tokens_i32.data_ptr<int32_t>(),
        .src_positions = positions_i32.data_ptr<int32_t>(),
        .src_new_cache_slots = new_cache_slots_i32.data_ptr<int32_t>(),
        .host_kv_seq_lens = host.kv_seq_lens.data(),
        .host_paged_kv_indptr = host_indptr,
        .host_paged_kv_indices = host_indices,
        .host_paged_kv_last_page_len = host_last_page,
        .dst_tokens = persistent_tokens_.data_ptr<int32_t>(),
        .dst_positions = persistent_positions_.data_ptr<int32_t>(),
        .dst_new_cache_slots = persistent_new_cache_slots_.data_ptr<int32_t>(),
        .dst_kv_seq_lens = kv_seq_lens_.data_ptr<int32_t>(),
        .dst_kv_seq_lens_delta =
            persistent_kv_seq_lens_delta_.data_ptr<int32_t>(),
        .dst_paged_kv_indptr = persistent_paged_kv_indptr_.data_ptr<int32_t>(),
        .dst_paged_kv_indices = persistent_paged_kv_indices_.data_ptr<int32_t>(),
        .dst_paged_kv_last_page_len =
            persistent_paged_kv_last_page_len_.data_ptr<int32_t>(),
        .actual_num_tokens = actual_num_tokens,
        .padded_num_tokens = static_cast<int64_t>(padded_num_tokens),
        .actual_batch_size = actual_batch_size,
        .actual_indices_size = actual_indices_size,
    };
    xllm::kernel::cuda::update_llm_decode_metadata_from_host(host_update_params,
                                                             stream);
    return;
  }

  const torch::Tensor paged_kv_indptr_i32 =
      to_int32(params.attention.device.paged_kv_indptr);
  const torch::Tensor paged_kv_indices_i32 =
      to_int32(params.attention.device.paged_kv_indices);
  const torch::Tensor paged_kv_last_page_len_i32 =
      to_int32(params.attention.device.paged_kv_last_page_len);
  const int64_t actual_indices_size = paged_kv_indices_i32.size(0);
  xllm::kernel::cuda::LlmDecodeMetadataUpdateParams update_params{
      .src_tokens = tokens_i32.data_ptr<int32_t>(),
      .src_positions = positions_i32.data_ptr<int32_t>(),
      .src_new_cache_slots = new_cache_slots_i32.data_ptr<int32_t>(),
      .src_kv_seq_lens = kv_seq_lens_i32.data_ptr<int32_t>(),
      .src_paged_kv_indptr = paged_kv_indptr_i32.data_ptr<int32_t>(),
      .src_paged_kv_indices = paged_kv_indices_i32.data_ptr<int32_t>(),
      .src_paged_kv_last_page_len =
          paged_kv_last_page_len_i32.data_ptr<int32_t>(),
      .dst_tokens = persistent_tokens_.data_ptr<int32_t>(),
      .dst_positions = persistent_positions_.data_ptr<int32_t>(),
      .dst_new_cache_slots = persistent_new_cache_slots_.data_ptr<int32_t>(),
      .dst_kv_seq_lens = kv_seq_lens_.data_ptr<int32_t>(),
      .dst_kv_seq_lens_delta =
          persistent_kv_seq_lens_delta_.data_ptr<int32_t>(),
      .dst_paged_kv_indptr = persistent_paged_kv_indptr_.data_ptr<int32_t>(),
      .dst_paged_kv_indices = persistent_paged_kv_indices_.data_ptr<int32_t>(),
      .dst_paged_kv_last_page_len =
          persistent_paged_kv_last_page_len_.data_ptr<int32_t>(),
      .actual_num_tokens = actual_num_tokens,
      .padded_num_tokens = static_cast<int64_t>(padded_num_tokens),
      .actual_batch_size = actual_batch_size,
      .actual_indices_size = actual_indices_size,
      .max_indices_size_for_graph_capacity =
          persistent_paged_kv_indices_.numel(),
  };
  xllm::kernel::cuda::update_llm_decode_metadata(update_params, stream);
}

void CudaGraphPersistentParam::set_aux_hidden_states(
    const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

size_t CudaGraphPersistentParam::get_persistent_tensor_bytes() const {
  auto bytes = [](const torch::Tensor& t) {
    return t.defined() ? static_cast<size_t>(t.numel()) * t.element_size() : 0;
  };
  size_t total = 0;
  total += bytes(persistent_tokens_);
  total += bytes(persistent_positions_);
  total += bytes(persistent_new_cache_slots_);
  total += bytes(persistent_linear_state_indices_);
  total += bytes(persistent_num_accepted_tokens_);
  total += bytes(persistent_block_tables_);
  total += bytes(hidden_states_);
  total += bytes(q_seq_lens_);
  total += bytes(kv_seq_lens_);
  total += bytes(persistent_embedding_);
  total += bytes(aux_hidden_states_);
  total += bytes(persistent_paged_kv_indptr_);
  total += bytes(persistent_paged_kv_indices_);
  total += bytes(persistent_paged_kv_last_page_len_);
  total += bytes(persistent_decode_qo_indptr_);
  total += bytes(persistent_kv_seq_lens_delta_);
  total += bytes(persistent_chunked_prefill_qo_indptr_);
  total += bytes(expanded_kv_seq_lens_);
  total += bytes(persistent_expanded_block_tables_);
  total += bytes(persistent_expanded_paged_kv_indptr_);
  total += bytes(persistent_expanded_paged_kv_indices_);
  total += bytes(persistent_expanded_paged_kv_last_page_len_);
  return total;
}

std::vector<int32_t> CudaGraphPersistentParam::update_expanded_spec_decode_attention(
    const ModelInputParams& input_params,
    uint32_t actual_num_tokens,
    uint32_t padded_num_tokens) {
  CHECK(input_params.is_spec_verify)
      << "expanded spec decode attention is only for spec verify";
  CHECK(input_params.meta.batch_forward_type.is_chunked_prefill())
      << "expanded spec decode attention expects chunked prefill";
  CHECK(input_params.graph.use_expanded_decode_for_spec_verify_attention)
      << "MTP worker must prepare expanded spec-verify graph input";
  CHECK(input_params.graph.expanded_kv_seq_lens.defined())
      << "expanded spec-verify kv seq lens must be defined";
  CHECK(input_params.graph.expanded_block_tables.defined())
      << "expanded spec-verify block tables must be defined";
  CHECK_EQ(input_params.graph.expanded_kv_seq_lens_vec.size(),
           static_cast<size_t>(actual_num_tokens))
      << "expanded kv seq lens size must match validate tokens";
  CHECK_EQ(input_params.graph.expanded_block_tables.size(0),
           static_cast<int64_t>(actual_num_tokens))
      << "expanded block table rows must match validate tokens";

  std::vector<int32_t> expanded_kv_seq_lens_vec =
      input_params.graph.expanded_kv_seq_lens_vec;
  expanded_kv_seq_lens_vec.reserve(padded_num_tokens);
  if (padded_num_tokens > actual_num_tokens) {
    const int64_t pad_count = padded_num_tokens - actual_num_tokens;
    for (int64_t i = 0; i < pad_count; ++i) {
      expanded_kv_seq_lens_vec.emplace_back(1);
    }
  }

  torch::Tensor expanded_kv_tensor =
      torch::tensor(expanded_kv_seq_lens_vec, torch::kInt).to(device_);
  expanded_kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .copy_(expanded_kv_tensor, /*non_blocking=*/true);

  const int64_t block_table_len =
      input_params.graph.expanded_block_tables.size(1);
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .zero_();
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len)
      .copy_(input_params.graph.expanded_block_tables, /*non_blocking=*/true);

  CHECK(input_params.graph.expanded_paged_kv_indptr.defined())
      << "expanded spec-verify paged_kv_indptr must be defined";
  CHECK(input_params.graph.expanded_paged_kv_indices.defined())
      << "expanded spec-verify paged_kv_indices must be defined";
  CHECK(input_params.graph.expanded_paged_kv_last_page_len.defined())
      << "expanded spec-verify paged_kv_last_page_len must be defined";

  const int64_t actual_indptr_size =
      input_params.graph.expanded_paged_kv_indptr.size(0);
  persistent_expanded_paged_kv_indptr_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_indptr_size)
      .copy_(input_params.graph.expanded_paged_kv_indptr,
             /*non_blocking=*/true);
  if (padded_num_tokens + 1 > static_cast<uint32_t>(actual_indptr_size)) {
    persistent_expanded_paged_kv_indptr_
        .slice(/*dim=*/0,
               /*start=*/actual_indptr_size,
               /*end=*/static_cast<int64_t>(padded_num_tokens + 1))
        .fill_(persistent_expanded_paged_kv_indptr_
                   .slice(/*dim=*/0,
                          /*start=*/actual_indptr_size - 1,
                          /*end=*/actual_indptr_size)
                   .item<int32_t>());
  }

  const int64_t actual_indices_size =
      input_params.graph.expanded_paged_kv_indices.size(0);
  persistent_expanded_paged_kv_indices_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_indices_size)
      .copy_(input_params.graph.expanded_paged_kv_indices,
             /*non_blocking=*/true);

  persistent_expanded_paged_kv_last_page_len_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(input_params.graph.expanded_paged_kv_last_page_len,
             /*non_blocking=*/true);
  if (padded_num_tokens > actual_num_tokens) {
    persistent_expanded_paged_kv_last_page_len_
        .slice(/*dim=*/0,
               /*start=*/actual_num_tokens,
               /*end=*/static_cast<int64_t>(padded_num_tokens))
        .fill_(1);
  }

  return expanded_kv_seq_lens_vec;
}

std::optional<ModelInputParams> CudaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params, args_.enable_mla()));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.meta.num_sequences;

  // Piecewise prefill graph capture/replay: when padding is needed, all
  // layers must process padded_num_tokens tokens so tensor sizes are
  // consistent across GDN (which uses max_query_len to reshape) and
  // full-attention (which uses positions/persistent_tokens). To avoid
  // corrupting the KV cache, padding tokens are filled with the last
  // actual token's values (same token_id, position, and cache slot), and
  // q_cu_seq_lens/kv_cu_seq_lens are overridden to [0, padded] so
  // FlashInfer treats all padded tokens as one sequence.
  const bool piecewise_prefill_pad =
      return_capture_params && attn_metadata->is_prefill &&
      padded_num_tokens > actual_num_tokens;
  if (piecewise_prefill_pad) {
    attn_metadata->max_query_len = padded_num_tokens;
    if (attn_metadata->q_seq_lens_vec.size() == 1) {
      attn_metadata->q_seq_lens_vec[0] =
          static_cast<int32_t>(padded_num_tokens);
    }
    if (attn_metadata->kv_seq_lens_vec.size() == 1) {
      attn_metadata->kv_seq_lens_vec[0] =
          static_cast<int32_t>(padded_num_tokens);
    }
  }
  // Match ACL graph persistent param: the expanded-decode graph input is
  // authoritative once MTP worker prepared it; do not additionally require
  // is_spec_verify here (capture/replay params may arrive without it set).
  const bool use_expanded_spec_decode_attention =
      params.graph.use_expanded_decode_for_spec_verify_attention &&
      (params.meta.batch_forward_type.is_chunked_prefill() ||
       attn_metadata->is_chunked_prefill);
  std::vector<int32_t> expanded_kv_seq_lens_vec;
  if (use_expanded_spec_decode_attention) {
    expanded_kv_seq_lens_vec = update_expanded_spec_decode_attention(
        params, actual_num_tokens, padded_num_tokens);
  }
  const bool use_llm_decode_fast_path =
      !use_expanded_spec_decode_attention &&
      can_use_llm_decode_fast_path(tokens, positions, params);

#if defined(USE_CUDA) || defined(USE_MUSA)
  // SGLang-style host-mirror staging for the Mate FFI decode kernel under
  // CUDA/MUSA stream capture.
  //
  // The Mate FFI batch_decode wrapper requires kDLCPU pointers for the three
  // paged-KV index tensors (indptr / indices / last_page_len). When the input
  // builder pre-stages them in `params.attention.host.paged_kv_*`, the
  // AttentionMetadataBuilder forwards them verbatim and batch_decode reuses
  // the CPU storage at zero cost (this is the common LLM-engine path; see
  // batch_input_builder.cpp lines 928-937).
  //
  // The warmup/profile path (profile_manager -> run_graph_decode_request)
  // currently lands in batch_decode with all three host mirrors undefined --
  // batch_decode's lazy `.to(kCPU)` fallback is fatal during stream capture
  // ("operation not permitted when stream is capturing"). Rather than
  // depending on which input builder populated the host mirrors, we stage
  // them here unconditionally for the decode path, *outside* the captured
  // region: the host copy lives across capture begin / replay because we
  // assign back into `attn_metadata->paged_kv_*_host` and the same shared
  // AttentionMetadata is what every full-attention layer's decoder_forward
  // reads. Mirrors sglang's pattern of building CPU-resident kv_indptr from
  // CPU-resident seq_lens before calling FlashInfer's plan/run (see
  // flashinfer_backend.py:1128-1182 `global_override_indptr_cpu`).
  //
  // Cheap when the input builder already pre-staged (just a shared_ptr ref);
  // a single per-step D2H per index tensor in the fallback case (3 D2H total,
  // batch-sized, runs once before capture begin).
  const bool decode_path = !attn_metadata->is_prefill &&
                           !attn_metadata->is_chunked_prefill;
  const bool expanded_decode_attention_path =
      use_expanded_spec_decode_attention;
  if (decode_path || expanded_decode_attention_path) {
    auto ensure_host_mirror = [](torch::Tensor& host_field,
                                 const torch::Tensor& device_field) {
      if (host_field.defined()) {
        return;
      }
      if (!device_field.defined()) {
        return;
      }
      host_field = device_field.to(torch::kCPU);
    };
    if (s_enable_graph_timing) {
      LOG(INFO) << "GRAPH_TIMING ensure_host_mirror: indptr_host_defined="
                << attn_metadata->paged_kv_indptr_host.defined()
                << " indices_host_defined="
                << attn_metadata->paged_kv_indices_host.defined()
                << " last_page_len_host_defined="
                << attn_metadata->paged_kv_last_page_len_host.defined();
    }
    ensure_host_mirror(attn_metadata->paged_kv_indptr_host,
                       attn_metadata->paged_kv_indptr);
    ensure_host_mirror(attn_metadata->paged_kv_indices_host,
                       attn_metadata->paged_kv_indices);
    ensure_host_mirror(attn_metadata->paged_kv_last_page_len_host,
                       attn_metadata->paged_kv_last_page_len);
  }
#endif
  auto build_capture_params_if_needed =
      [&]() -> std::optional<ModelInputParams> {
    if (!return_capture_params) {
      return std::nullopt;
    }
    CHECK(params_for_capture.has_value())
        << "params_for_capture should be initialized when "
           "return_capture_params "
           "is true";
    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    if (!params.embedding.linear_state_ids.empty()) {
      params_for_capture->embedding.linear_state_ids =
          params.embedding.linear_state_ids;
      params_for_capture->embedding.linear_state_indices =
          persistent_linear_state_indices(params.meta.num_sequences);
    }
    if (params.num_accepted_tokens.defined()) {
      params_for_capture->num_accepted_tokens = persistent_num_accepted_tokens(
          static_cast<uint32_t>(actual_batch_size));
      torch::Tensor nat_host = params.num_accepted_tokens.to(torch::kCPU)
                                   .to(torch::kLong)
                                   .contiguous();
      const int64_t copy_size =
          std::min<int64_t>(actual_batch_size, nat_host.numel());
      const int64_t* data = nat_host.data_ptr<int64_t>();
      params_for_capture->num_accepted_tokens_host.assign(data,
                                                           data + copy_size);
    }
    params_for_capture->attn_metadata = attn_metadata;
    params_for_capture->is_spec_verify = params.is_spec_verify;
    if (use_expanded_spec_decode_attention) {
      params_for_capture->graph.use_expanded_decode_for_spec_verify_attention =
          true;
      params_for_capture->graph.expanded_kv_seq_lens =
          expanded_kv_seq_lens(padded_num_tokens);
      params_for_capture->graph.expanded_block_tables =
          persistent_expanded_block_tables(padded_num_tokens);
      params_for_capture->graph.expanded_kv_seq_lens_vec =
          expanded_kv_seq_lens_vec;
      params_for_capture->graph.expanded_paged_kv_indptr =
          persistent_expanded_paged_kv_indptr(padded_num_tokens);
      params_for_capture->graph.expanded_paged_kv_indices =
          persistent_expanded_paged_kv_indices_;
      params_for_capture->graph.expanded_paged_kv_last_page_len =
          persistent_expanded_paged_kv_last_page_len(padded_num_tokens);
    }
    return params_for_capture;
  };

  // Copy data from input parameters to persistent graph tensors
  if (use_llm_decode_fast_path) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use fast path for LLM decode metadata update";
    update_llm_decode_metadata_fast_path(tokens,
                                         positions,
                                         params,
                                         padded_num_tokens,
                                         actual_batch_size,
                                         actual_num_tokens);
  } else {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
        << actual_num_tokens << "]";
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(tokens, /*non_blocking=*/true);

    if (padded_num_tokens > actual_num_tokens) {
      if (piecewise_prefill_pad) {
        // Fill padding tokens with last actual token's value so their
        // embedding/QKV matches — prevents KV cache corruption when
        // FlashInfer processes all padded tokens as one sequence.
        persistent_tokens_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .copy_(persistent_tokens_
                       .slice(/*dim=*/0,
                              /*start=*/actual_num_tokens - 1,
                              /*end=*/actual_num_tokens)
                       .expand({static_cast<int64_t>(padded_num_tokens -
                                                     actual_num_tokens)}));
      } else {
        persistent_tokens_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .fill_(0);
      }
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ positions: src shape=" << positions.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_positions_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(positions, /*non_blocking=*/true);

    if (piecewise_prefill_pad) {
      // Fill padding positions with last actual token's position so RoPE
      // produces the same output for padding tokens as the last actual
      // token — consistent with the token padding fill above.
      persistent_positions_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .copy_(persistent_positions_
                     .slice(/*dim=*/0,
                            /*start=*/actual_num_tokens - 1,
                            /*end=*/actual_num_tokens)
                     .expand({static_cast<int64_t>(padded_num_tokens -
                                                   actual_num_tokens)}));
    }
  }

  if (!is_rec_multi_round_mode() && !use_llm_decode_fast_path) {
    // q_seq_lens is q_cu_seq_lens in GPU Model.
    // kv_seq_lens is kv_cu_seq_lens in GPU Model.
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ q_seq_lens: src shape="
        << params.attention.device.q_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ kv_seq_lens: src shape="
        << params.attention.device.kv_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ new_cache_slots: src shape="
        << params.attention.device.new_cache_slots.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_new_cache_slots_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(params.attention.device.new_cache_slots, /*non_blocking=*/true);
    if (padded_num_tokens > actual_num_tokens) {
      if (piecewise_prefill_pad) {
        // Fill padding cache slots with last actual token's slot so the
        // KV cache write for padding tokens overwrites the last actual
        // token's slot with the same value (no corruption).
        persistent_new_cache_slots_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .copy_(persistent_new_cache_slots_
                       .slice(/*dim=*/0,
                              /*start=*/actual_num_tokens - 1,
                              /*end=*/actual_num_tokens)
                       .expand({static_cast<int64_t>(padded_num_tokens -
                                                     actual_num_tokens)}));
      } else {
        persistent_new_cache_slots_
            .slice(/*dim=*/0,
                   /*start=*/actual_num_tokens,
                   /*end=*/padded_num_tokens)
            .fill_(0);
      }
    }

    // Keep metadata tensors pointing to persistent buffers used by graph
    // capture/replay so their addresses are stable and shapes match padded
    // tensors in capture path.
    attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/
                                              actual_batch_size + 1);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(/*actual_batch_size=*/
                                                actual_batch_size + 1);

    // For piecewise prefill: override q_cu_seq_lens and kv_cu_seq_lens to
    // [0, padded_num_tokens] so FlashInfer processes all padded tokens and
    // outputs [padded, dim] — matching GDN output size. The persistent
    // buffers are updated in-place so the eager FlashInfer plan/attention
    // sees the padded values.
    if (piecewise_prefill_pad) {
      q_seq_lens_.slice(/*dim=*/0, /*start=*/1, /*end=*/2)
          .fill_(static_cast<int32_t>(padded_num_tokens));
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/1, /*end=*/2)
          .fill_(static_cast<int32_t>(padded_num_tokens));
    }

    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
  }

  if (!is_rec_multi_round_mode() &&
      !params.embedding.linear_state_ids.empty()) {
    if (params.embedding.linear_state_indices.defined()) {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(params.embedding.linear_state_indices, /*non_blocking=*/true);
    } else {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(torch::tensor(params.embedding.linear_state_ids, torch::kInt)
                     .to(device_),
                 /*non_blocking=*/true);
    }
  }

  if (params.num_accepted_tokens.defined()) {
    persistent_num_accepted_tokens_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
        .copy_(params.num_accepted_tokens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size),
               /*non_blocking=*/true);
  }

  // Copy block table data. In rec multi-round, block_tables may already be
  // expanded to batch_size * beam_width rows while num_sequences still tracks
  // the logical request count. Use the tensor's real row count here.
  const int64_t actual_block_table_batch =
      is_rec_multi_round_mode() ? params.attention.device.block_tables.size(0)
                                : actual_batch_size;
  const int64_t actual_block_table_len =
      params.attention.device.block_tables.size(1);
  torch::Tensor slice_persistent_block_tables =
      persistent_block_tables_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_block_table_batch)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ block_tables: src shape="
      << params.attention.device.block_tables.sizes()
      << ", dst slice shape=" << slice_persistent_block_tables.sizes();
  slice_persistent_block_tables.copy_(params.attention.device.block_tables,
                                      /*non_blocking=*/true);
  if (!attn_metadata->is_prefill || args_.enable_mla()) {
    attn_metadata->block_table = slice_persistent_block_tables;
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = embedding.size(1);
      torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape=[" << embedding_tokens << ", "
        << embedding.size(1) << "]";
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  const bool is_decode_with_llmrec =
      params.meta.batch_forward_type.is_decode() && params.has_llmrec_params();
  const bool use_two_stage_decode =
      !::xllm::RecConfig::get_instance().enable_xattention_one_stage() &&
      is_decode_with_llmrec;
  const int32_t head_dim = args_.head_dim();
  const int64_t tp_size =
      options_.world_size() / std::max(options_.dp_size(), 1);
  const int64_t n_heads = args_.n_heads() / std::max(tp_size, int64_t{1});
  const int64_t total_kv_heads = args_.n_kv_heads().value_or(args_.n_heads());
  const int64_t n_kv_heads =
      (total_kv_heads >= tp_size)
          ? (total_kv_heads / std::max(tp_size, int64_t{1}))
          : 1;
  const int64_t block_size = options_.block_size();

  // Get sliding_window from ModelArgs (default to -1 if not available)
  // Note: sliding_window in ModelArgs is the actual window size, but in
  // attention it's used as window_size_left which is typically sliding_window
  // - 1. This matches the behavior in attention.cpp where sliding_window_ is
  // initialized as sliding_window - 1 regardless of the value.
  int32_t sliding_window = args_.sliding_window();
  sliding_window =
      sliding_window - 1;  // Convert to window_size_left (always subtract 1)

  // Get dtype from k_cache
  const auto dtype = k_cache.scalar_type();
  // Determine backend
  const std::string backend = xllm::kernel::cuda::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      /*use_custom_mask=*/false);

  bool use_tensor_core =
      xllm::kernel::cuda::should_use_tensor_core(dtype, n_heads, n_kv_heads);
#ifdef XLLM_TORCH_MUSA
  // Keep in sync with BaseAttentionImpl::decode_use_tensor_core_ on MUSA:
  // the Mate FFI ships the dedicated `batch_decode_*` kernel (exporting the
  // "run" symbol) for our paged-KV layouts, while the chunked-prefill
  // `batch_prefill_*` kernel only exports "paged_run". When this graph-mode
  // planner picks the chunked-prefill URI but `FlashInferAttentionImpl::
  // decoder_forward` calls `batch_decode(..., decode_use_tensor_core_=false)`,
  // `batch_decode` falls into the else branch and tries to look up
  // get_function(prefill_uri, "run") which doesn't exist in the .so. Force
  // the same value here so the planner and the runtime caller agree on the
  // URI scheme.
  use_tensor_core = false;
#endif
#if defined(XLLM_TORCH_MUSA)
  if (use_two_stage_decode) {
    LOG(FATAL) << "two-stage xattention decode is not supported in "
                  "XLLM_TORCH_MUSA builds.";
  }
#else
  if (use_two_stage_decode) {
    if (params.attention.device.q_seq_lens.defined() &&
        params.attention.device.q_seq_lens.numel() > 0) {
      const int64_t q_numel = params.attention.device.q_seq_lens.numel();
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_numel)
          .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);
      attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/q_numel);
    }

    if (params.attention.device.kv_seq_lens.defined() &&
        params.attention.device.kv_seq_lens.numel() > 0) {
      const int64_t kv_numel = params.attention.device.kv_seq_lens.numel();
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_numel)
          .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);
      attn_metadata->kv_cu_seq_lens =
          kv_seq_lens(/*actual_batch_size=*/kv_numel);
      if (kv_numel > 1) {
        attn_metadata->kv_seq_lens = torch::diff(attn_metadata->kv_cu_seq_lens);
      }
    }

    attn_metadata->paged_kv_indptr = torch::Tensor();
    attn_metadata->paged_kv_indices = torch::Tensor();
    attn_metadata->paged_kv_last_page_len = torch::Tensor();
    attn_metadata->qo_indptr = torch::Tensor();

    // Update plan_info if attn_metadata exists and enable_cuda_graph is true.
    const auto& llmrec_params = *params.llmrec_params();
    auto cache = attn_metadata->xattention_two_stage_decode_cache.value();
    CHECK(cache.q_cu_seq_lens_shared.defined())
        << "q_cu_seq_lens_shared must be initialized in rec worker";
    CHECK(cache.qo_indptr_expanded.defined())
        << "qo_indptr_expanded must be initialized in rec worker";
    CHECK(cache.paged_kv_indptr_expanded.defined() &&
          cache.paged_kv_indices_expanded.defined() &&
          cache.paged_kv_last_page_len_expanded.defined())
        << "paged_kv_* expanded tensors must be initialized in rec worker";
    CHECK(cache.shared_lse.defined() && cache.shared_o.defined() &&
          cache.unshared_lse.defined() && cache.unshared_o.defined())
        << "two-stage shared/unshared output cache tensors must be "
           "initialized in rec worker";

    const int64_t request_batch_size =
        cache.q_cu_seq_lens_shared.numel() > 0
            ? (cache.q_cu_seq_lens_shared.numel() - 1)
            : 0;
    CHECK_GT(request_batch_size, 0)
        << "request_batch_size must be > 0 for two-stage xattention";
    const int64_t beam_width = std::max<int64_t>(llmrec_params.beam_width, 1);
    const int64_t total_beam = request_batch_size * beam_width;

    CHECK_EQ(cache.shared_o.size(0), total_beam)
        << "shared_o first dim mismatch: expected total_beam=" << total_beam
        << ", got " << cache.shared_o.size(0);
    CHECK_EQ(cache.unshared_o.size(0), total_beam)
        << "unshared_o first dim mismatch: expected total_beam=" << total_beam
        << ", got " << cache.unshared_o.size(0);
    CHECK_EQ(cache.qo_indptr_expanded.numel(), total_beam + 1)
        << "qo_indptr_expanded size mismatch: expected " << (total_beam + 1)
        << ", got " << cache.qo_indptr_expanded.numel();
    CHECK_EQ(cache.paged_kv_indptr_expanded.numel(), total_beam + 1)
        << "paged_kv_indptr_expanded size mismatch";
    CHECK_EQ(cache.paged_kv_indices_expanded.numel(), total_beam)
        << "paged_kv_indices_expanded size mismatch";
    CHECK_EQ(cache.paged_kv_last_page_len_expanded.numel(), total_beam)
        << "paged_kv_last_page_len_expanded size mismatch";

    const int64_t max_decode_step =
        !llmrec_params.unshared_k_caches.empty()
            ? llmrec_params.unshared_k_caches[0].size(2)
            : std::max<int64_t>(
                  ::xllm::RecConfig::get_instance().max_decode_rounds() - 1, 1);
    CHECK_GT(max_decode_step, 0)
        << "max_decode_step must be > 0 for two-stage decode";

    cache.cached_batch_size = static_cast<int32_t>(request_batch_size);
    cache.cached_beam_size = static_cast<int32_t>(beam_width);
    cache.cached_num_heads = static_cast<int32_t>(n_heads);
    cache.cached_head_size = static_cast<int32_t>(head_dim);
    cache.cached_max_decode_step = static_cast<int32_t>(max_decode_step);
    cache.cached_step = (llmrec_params.current_round_tensor.defined() &&
                         llmrec_params.current_round_tensor.numel() > 0)
                            ? llmrec_params.current_round_tensor.item<int32_t>()
                            : 0;

    attn_metadata->xattention_two_stage_decode_cache = cache;

    layer::AttentionMetadata shared_attn_meta = *attn_metadata;
    shared_attn_meta.plan_info = attn_metadata->shared_plan_info;
    shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;
    shared_attn_meta.is_causal = false;

    attn_metadata->shared_plan_info->layer_id = 0;
    layer::xattention::update_xattention_plan_info(
        attn_metadata->shared_plan_info,
        backend,
        shared_attn_meta,
        dtype,
        dtype,
        dtype,
        head_dim,
        head_dim,
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        /*block_size=*/1,
        /*window_size_left=*/-1,
        /*enable_cuda_graph=*/true,
        /*causal=*/false,
        /*use_tensor_core=*/true,
        /*is_shared_stage_plan*/ true);

    layer::AttentionMetadata unshared_attn_meta = *attn_metadata;
    unshared_attn_meta.plan_info = attn_metadata->unshared_plan_info;
    unshared_attn_meta.qo_indptr = cache.qo_indptr_expanded;
    unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
    unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
    unshared_attn_meta.paged_kv_last_page_len =
        cache.paged_kv_last_page_len_expanded;
    unshared_attn_meta.is_causal = false;

    attn_metadata->unshared_plan_info->layer_id = 0;
    layer::xattention::update_xattention_plan_info(
        attn_metadata->unshared_plan_info,
        backend,
        unshared_attn_meta,
        dtype,
        dtype,
        dtype,
        head_dim,
        head_dim,
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        static_cast<int32_t>(max_decode_step),
        sliding_window,
        /*enable_cuda_graph=*/true,
        /*causal=*/false,
        use_tensor_core,
        /*is_shared_stage_plan*/ false);
    return build_capture_params_if_needed();
  }
#endif
  if (use_expanded_spec_decode_attention) {
    const int64_t expanded_batch = static_cast<int64_t>(padded_num_tokens);
    attn_metadata->use_expanded_decode_for_spec_verify_attention = true;
    attn_metadata->expanded_kv_seq_lens =
        expanded_kv_seq_lens(static_cast<uint32_t>(expanded_batch));
    attn_metadata->expanded_block_table =
        persistent_expanded_block_tables(static_cast<uint32_t>(expanded_batch));
    if (!expanded_kv_seq_lens_vec.empty()) {
      attn_metadata->expanded_kv_seq_lens_host =
          torch::tensor(expanded_kv_seq_lens_vec, torch::kInt);
    }
    attn_metadata->paged_kv_indptr =
        persistent_expanded_paged_kv_indptr(
            static_cast<uint32_t>(expanded_batch));
    attn_metadata->paged_kv_indices = persistent_expanded_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_expanded_paged_kv_last_page_len(
            static_cast<uint32_t>(expanded_batch));
    attn_metadata->qo_indptr =
        persistent_decode_qo_indptr(static_cast<uint32_t>(expanded_batch));
    attn_metadata->expanded_paged_kv_indptr =
        persistent_expanded_paged_kv_indptr(
            static_cast<uint32_t>(expanded_batch));
    attn_metadata->expanded_paged_kv_indices =
        persistent_expanded_paged_kv_indices_;
    attn_metadata->expanded_paged_kv_last_page_len =
        persistent_expanded_paged_kv_last_page_len(
            static_cast<uint32_t>(expanded_batch));
#if defined(USE_CUDA) || defined(USE_MUSA)
    auto ensure_host_mirror = [](torch::Tensor& host_field,
                                 const torch::Tensor& device_field) {
      if (host_field.defined()) {
        return;
      }
      if (!device_field.defined()) {
        return;
      }
      host_field = device_field.to(torch::kCPU);
    };
    ensure_host_mirror(attn_metadata->paged_kv_indptr_host,
                       attn_metadata->paged_kv_indptr);
    ensure_host_mirror(attn_metadata->paged_kv_indices_host,
                       attn_metadata->paged_kv_indices);
    ensure_host_mirror(attn_metadata->paged_kv_last_page_len_host,
                       attn_metadata->paged_kv_last_page_len);
#endif
  } else if (use_llm_decode_fast_path) {
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->q_cu_seq_lens =
        persistent_decode_qo_indptr(static_cast<uint32_t>(actual_batch_size));
    attn_metadata->kv_cu_seq_lens =
        kv_seq_lens(static_cast<uint32_t>(actual_batch_size + 1));
    attn_metadata->kv_seq_lens =
        persistent_kv_seq_lens_delta(static_cast<uint32_t>(actual_batch_size));
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
    attn_metadata->paged_kv_indptr =
        persistent_paged_kv_indptr(static_cast<uint32_t>(actual_batch_size));
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len = persistent_paged_kv_last_page_len(
        static_cast<uint32_t>(actual_batch_size));
    attn_metadata->qo_indptr =
        persistent_decode_qo_indptr(static_cast<uint32_t>(actual_batch_size));
  } else {
    CHECK(params.attention.device.paged_kv_indptr.defined())
        << "paged_kv_indptr should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indptr: src shape="
        << params.attention.device.paged_kv_indptr.sizes()
        << ", dst slice shape=[" << (actual_batch_size + 1) << "]";
    if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
      torch::Tensor paged_kv_indptr_cpu =
          params.attention.device.paged_kv_indptr.to(torch::kCPU);
      VLOG(kGraphExecutorLogVerboseLevel)
          << "copy_ paged_kv_indptr: src values=" << paged_kv_indptr_cpu;
    }
    persistent_paged_kv_indptr_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.paged_kv_indptr, /*non_blocking=*/true);
    CHECK(params.attention.device.paged_kv_indices.defined())
        << "paged_kv_indices should not be null";
    const int64_t actual_indices_size =
        params.attention.device.paged_kv_indices.size(0);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indices: src shape="
        << params.attention.device.paged_kv_indices.sizes()
        << ", dst slice shape=[" << actual_indices_size << "]";
    persistent_paged_kv_indices_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_indices_size)
        .copy_(params.attention.device.paged_kv_indices, /*non_blocking=*/true);
    CHECK(params.attention.device.paged_kv_last_page_len.defined())
        << "paged_kv_last_page_len should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_last_page_len: src shape="
        << params.attention.device.paged_kv_last_page_len.sizes()
        << ", dst slice shape=[" << actual_batch_size << "]";
    persistent_paged_kv_last_page_len_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size)
        .copy_(params.attention.device.paged_kv_last_page_len,
               /*non_blocking=*/true);
    attn_metadata->kv_seq_lens =
        torch::diff(kv_seq_lens(/*actual_batch_size=*/actual_batch_size + 1));
    attn_metadata->paged_kv_indptr =
        persistent_paged_kv_indptr(actual_batch_size);
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_paged_kv_last_page_len(actual_batch_size);
    attn_metadata->qo_indptr = persistent_decode_qo_indptr(actual_batch_size);
  }
  // Update plan_info if attn_metadata exists and enable_cuda_graph is true
  // This ensures plan_info is updated before CUDA graph capture/replay
  {
    // Determine if causal (prefill mode)
    const bool causal =
        attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;
    CHECK_EQ(dtype, torch::ScalarType::BFloat16)
        << "only support bf16 kvcache for now";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", causal=" << causal << ", backend=" << backend
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

    if (attn_metadata->is_prefill) {
      layer::flashinfer::update_prefill_plan_info(
          attn_metadata->plan_info,
          backend,
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          /*enable_cuda_graph=*/true);
    } else if (use_expanded_spec_decode_attention ||
               !attn_metadata->is_chunked_prefill) {
      // Spec-verify validate uses per-token batch_decode for full-attention
      // layers; regular decode uses the same decode plan path.
      const int32_t max_kv_blocks_per_seq_for_capture =
          block_size > 0
              ? static_cast<int32_t>(
                    (args_.max_position_embeddings() + block_size - 1) /
                    block_size)
              : 0;
      layer::flashinfer::update_decode_plan_info(
          attn_metadata->plan_info,
          /*backend=*/"fa2",  // flashinfer paged fa3 is slow, use fa2 instead
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          static_cast<int32_t>(block_size),  // block_size
          sliding_window,                    // window_size_left
          /*enable_cuda_graph=*/true,
          use_tensor_core,
          max_kv_blocks_per_seq_for_capture);
    } else if (attn_metadata->is_chunked_prefill) {
      // Worst-case KV blocks per sequence for graph capture: plan_info is
      // computed once (cached on PlanInfo) and reused for all replays. Make
      // sure the cached plan covers any future block count by computing it
      // against ceil(max_position_embeddings / block_size) blocks per
      // sequence. See update_chunked_prefill_plan_info comment.
      const int32_t max_kv_blocks_per_seq_for_capture =
          block_size > 0
              ? static_cast<int32_t>(
                    (args_.max_position_embeddings() + block_size - 1) /
                    block_size)
              : 0;
      layer::flashinfer::update_chunked_prefill_plan_info(
          attn_metadata->plan_info,
          /*backend=*/"fa2",  // flashinfer paged fa3 is slow, use fa2 instead
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          static_cast<int32_t>(block_size),  // block_size
          sliding_window,                    // window_size_left
          /*enable_cuda_graph=*/true,
          /*causal=*/true,
          max_kv_blocks_per_seq_for_capture);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() plan_info updated: uri="
        << attn_metadata->plan_info->uri << ", plan_info.defined="
        << attn_metadata->plan_info->plan_info.defined() << ", plan_info.size="
        << (attn_metadata->plan_info->plan_info.defined()
                ? attn_metadata->plan_info->plan_info.size()
                : 0);
  }

  // Return ModelInputParams with persistent buffer references if requested
  return build_capture_params_if_needed();
}

void CudaGraph::refresh_persistent_paged_kv_host_mirrors(
    const std::shared_ptr<layer::AttentionMetadata>& attn_metadata,
    const AttentionHostInput& host_src) {
#if defined(USE_CUDA) || defined(USE_MUSA)
  // Only applies to the Mate FFI decode path. Prefill/chunked-prefill and MLA
  // attention do not pass host pointers through the FFI run() boundary, so
  // there is nothing to stabilize there.
  if (!attn_metadata) {
    return;
  }
  if (attn_metadata->is_prefill ||
      (attn_metadata->is_chunked_prefill &&
       !attn_metadata->use_expanded_decode_for_spec_verify_attention)) {
    return;
  }

  // Helper: lazily allocate / grow a pinned host buffer, copy fresh metadata
  // into it, then re-point `host_field` at the owning slice. The buffer's
  // underlying storage pointer must be STABLE across the lifetime of this
  // CudaGraph (captured FFI run() bakes it into the graph).
  //
  // Prefer copying from attention.host CPU mirrors (batch_input_builder path):
  // the values are already on host, so a CPU->pinned memcpy avoids the extra
  // device round-trip and the musaStreamSync that blocking D2H would insert.
  // Fall back to blocking D2H from persistent device tensors for callers that
  // did not pre-stage host mirrors (profile / warmup paths).
  //
  // CRITICAL: when allocating for the first time, size to max(numel,
  // min_alloc_numel). The captured graph cannot tolerate a later realloc:
  // if the KV cache crosses a block boundary mid-replay (e.g., decode 38 of
  // a question with prefill=27, block_size=64), the device-side
  // paged_kv_indices numel grows from 1 to 2 entries. With min_alloc_numel
  // set to the worst case at capture time, the first allocation is already
  // large enough and subsequent refresh calls hit the same storage.
  // Otherwise the realloc returns fresh memory and the captured kernel
  // dereferences a stale (freed) pointer, producing silently-wrong
  // attention outputs from L3 onward. See refresh-call-site comment.
  auto refresh_one = [](torch::Tensor& host_buf,
                        torch::Tensor& host_field,
                        const torch::Tensor& device_src,
                        const torch::Tensor& cpu_src,
                        int64_t min_alloc_numel) {
    if (!device_src.defined()) {
      return;
    }
    const int64_t numel = device_src.numel();
    const torch::ScalarType src_dtype = device_src.scalar_type();
    const bool needs_alloc =
        !host_buf.defined() || host_buf.scalar_type() != src_dtype ||
        host_buf.numel() < numel ||
        host_buf.numel() < min_alloc_numel;
    if (needs_alloc) {
      // Pinned so cudaMemcpyAsync from device is a real async copy that can
      // be captured into the graph (the Mate FFI submits H2D internally on
      // some shapes; pinning ensures the captured operation refreshes the
      // device buffer from our stable host pointer on every replay). Sized
      // to max(numel, min_alloc_numel) so the first allocation already
      // covers the worst-case KV cache layout for this CudaGraph instance.
      auto opts = torch::TensorOptions()
                      .dtype(src_dtype)
                      .device(torch::kCPU)
                      .pinned_memory(true);
      const int64_t alloc_numel = std::max<int64_t>(numel, min_alloc_numel);
      host_buf = torch::empty({alloc_numel}, opts);
    }
    auto dst = host_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/numel);
    // device_src may have a non-1D shape (e.g., [bs+1]); view as flat for the
    // copy and let host_field carry the original shape via a view-back.
    //
    // The Mate FFI batch_decode wrapper reads paged_kv_* host tensors on the
    // CPU at submit time, so destination bytes must be valid before replay.
    // CPU->pinned and blocking D2H both satisfy that; async D2H would not.
    const bool use_cpu_src = cpu_src.defined() && cpu_src.device().is_cpu() &&
                           cpu_src.numel() == numel;
    if (use_cpu_src) {
      torch::Tensor cpu_flat = cpu_src.contiguous().view({numel});
      if (cpu_flat.scalar_type() != src_dtype) {
        cpu_flat = cpu_flat.to(src_dtype);
      }
      dst.copy_(cpu_flat);
    } else {
      dst.copy_(device_src.contiguous().view({numel}), /*non_blocking=*/false);
    }
    // Re-point host_field at the persistent storage; preserve the original
    // logical shape so downstream callers that interrogate sizes still see
    // the same view they did before this rewrite.
    host_field = dst.view(device_src.sizes());
  };

  refresh_one(paged_kv_indptr_host_buf_,
              attn_metadata->paged_kv_indptr_host,
              attn_metadata->paged_kv_indptr,
              host_src.paged_kv_indptr,
              paged_kv_indptr_host_max_numel_);
  refresh_one(paged_kv_indices_host_buf_,
              attn_metadata->paged_kv_indices_host,
              attn_metadata->paged_kv_indices,
              host_src.paged_kv_indices,
              paged_kv_indices_host_max_numel_);
  refresh_one(paged_kv_last_page_len_host_buf_,
              attn_metadata->paged_kv_last_page_len_host,
              attn_metadata->paged_kv_last_page_len,
              host_src.paged_kv_last_page_len,
              paged_kv_last_page_len_host_max_numel_);
#endif
}

// CudaGraph implementation
bool CudaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const at::cuda::MempoolId_t& pool,
                        TorchMemPool* pool_ptr) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // Compute worst-case pinned-host-mirror sizes for paged-KV metadata so the
  // FIRST allocation inside refresh_persistent_paged_kv_host_mirrors already
  // covers the largest layout this CudaGraph instance can ever see. Without
  // this, the captured graph bakes in a host pointer whose underlying buffer
  // gets reallocated mid-replay when the KV cache crosses a block boundary
  // (e.g., decode step 38 of a 27-token-prefill question with block_size=64
  // grows paged_kv_indices from 1 -> 2 entries), causing the captured Mate
  // FFI batch_decode kernel to dereference stale memory and silently corrupt
  // attention output. See refresh_persistent_paged_kv_host_mirrors comment.
  //
  // For a decode bucket of N input tokens, at most N sequences are active and
  // each can hold up to ceil(max_position_embeddings / block_size) blocks.
  {
    const int64_t block_size = options.block_size();
    const int64_t max_pos = args.max_position_embeddings();
    const int64_t max_blocks_per_seq =
        block_size > 0 ? (max_pos + block_size - 1) / block_size : 0;
    const int64_t max_seqs = static_cast<int64_t>(bucket_num_tokens);
    paged_kv_indptr_host_max_numel_ = max_seqs + 1;
    paged_kv_indices_host_max_numel_ = max_seqs * max_blocks_per_seq;
    paged_kv_last_page_len_host_max_numel_ = max_seqs;
  }

  // Guard CUDA graph capture region with a device-level exclusive lock to
  // prevent conflicting GPU work from other streams (e.g., prepare streams) on
  // the same device when using cudaStreamCaptureModeGlobal. Capture requires
  // exclusive access, so we use write lock.
  std::optional<std::unique_lock<std::shared_mutex>> capture_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& capture_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_write_lock(
            device_index_);
    capture_lock_guard.emplace(capture_lock);
  }
  // Use the returned ModelInputParams for graph capture
  // Always use capture stream for plan/update + capture + forward.
  at::cuda::CUDAStream original_stream =
      at::cuda::getCurrentCUDAStream(device_index_);
  at::cuda::CUDAStream capture_stream = capture_stream_;
  if (original_stream != capture_stream) {
    original_stream.synchronize();
    capture_stream.synchronize();
  }
  std::optional<at::cuda::CUDAStreamGuard> stream_guard;
  stream_guard.emplace(capture_stream);

  // auto& tensor_options = model->options();

  // Update persistent parameters with input data before capture (includes
  // FlashInfer plan/update).
  auto full_attention_cache =
      CudaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "CUDA graph capture requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;
  auto graph_params_opt =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               padded_num_tokens_,
                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  // SGLang-pattern persistent host mirrors for the Mate FFI batch_decode
  // run() call. CudaGraphPersistentParam::update() built the attn_metadata's
  // host fields as fresh `.to(kCPU)` tensors -- their .data_ptr() would
  // change on every step, but the captured graph bakes whatever pointer it
  // saw at capture time. Swap them out for views into per-CudaGraph stable
  // pinned host buffers so the captured pointer remains valid forever and
  // every replay sees fresh values via in-place `copy_`. Mirrors SGLang's
  // FlashInfer MLA backend treatment of `cuda_graph_kv_indptr_cpu` (see
  // python/sglang/srt/layers/attention/flashinfer_mla_backend.py:365-477).
  refresh_persistent_paged_kv_host_mirrors(
      graph_params_opt.value().attn_metadata, params.attention.host);

  LOG(INFO) << "CUDA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens << ", actual_num_tokens: " << actual_num_tokens
            << ", is_piecewise: " << is_piecewise_;

  if (is_piecewise_) {
    // Piecewise capture mode (for prefill)
    // Warmup: execute forward once without capture to initialize cuBLAS handles
    // and other CUDA resources. This is necessary because these resources
    // cannot be created during CUDA graph capture mode.
    model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                   persistent_param_.persistent_positions(padded_num_tokens_),
                   kv_cache,
                   graph_params_opt.value());

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::cuda::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin piecewise capture via GlobalCaptureInstance.
    GlobalCaptureInstance::get_instance().begin_capture(pool);

    // Execute forward pass - attention operations will be captured separately
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    // Only capture aux_hidden_states when enable_graph_aux_hidden_states is on
    // (e.g. main worker in EAGLE-3); draft worker has this option false.
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "Piecewise capture forward_result shape: "
        << forward_result.hidden_states.sizes();

    // End capture and get piecewise graphs
    auto piecewise_graphs = GlobalCaptureInstance::get_instance().end_capture();

    if (!piecewise_graphs || piecewise_graphs->empty()) {
      LOG(WARNING) << "Failed to capture piecewise graph: no graphs captured";
      return false;
    }

    // Move piecewise graphs to member
    piecewise_graph_ = std::move(*piecewise_graphs);

    LOG(INFO) << "Piecewise graph capture end, bucket_num_tokens: "
              << bucket_num_tokens
              << ", num_graphs: " << piecewise_graph_.size()
              << ", num_runners: " << piecewise_graph_.num_runners();
  } else {
    // Normal capture mode (for decode)

    // Two pre-capture warmup passes, each preceded by a stream sync. This
    // mirrors sglang/python/sglang/srt/model_executor/cuda_graph_runner.py
    // (capture_one_batch_size: `for _ in range(2): synchronize(); barrier();
    // run_once()`). Goals:
    //
    //   1. Allocate every per-op output buffer (embedding/index_select,
    //      cuBLAS workspace, attention scratch, ...) into the caching pool
    //      before capture_begin(). MUSA's stream-capture rejects any allocator
    //      growth call (musaMalloc / VMM map) with "operation not permitted
    //      when stream is capturing", so these must be pre-warmed.
    //
    //   2. Synchronize BEFORE the capture so warmup buffers are returned to
    //      the pool with no pending stream dependency. Without the sync, when
    //      the captured forward later reuses one of those buffers, the caching
    //      allocator inserts a stream-wait (cudaStreamWaitEvent) to honor the
    //      cross-stream dep -- which itself is illegal during capture and
    //      surfaces as the same MUSA "operation not permitted" error inside
    //      at::musa::IndexSelect even though the buffer was already mapped.
    //
    //   3. The second warmup catches any allocations the first pass left in
    //      a transient state (e.g. tvm-ffi workspace expansion on first call)
    //      so the pool reaches a steady size before capture_begin().
    //
    // Reuses the outer `capture_stream` (set up at the top of this function
    // and active via stream_guard) so the warmup runs on the exact stream
    // about to be captured -- syncing a different stream would leave the
    // capture stream with stale pending work.
    static const bool capture_logits_enabled =
        std::getenv("XLLM_GRAPH_CAPTURE_LOGITS") != nullptr;
    capture_logits_ =
        capture_logits_enabled && bucket_num_tokens == 1 &&
        !options.enable_speculative_decode();
    if (capture_logits_) {
      persistent_param_.ensure_logits_buffer(
          args.vocab_size(), torch::kBFloat16, persistent_param_.device());
      LOG(INFO) << "D1: capturing lm_head into decode graph (bucket_num_tokens="
                << bucket_num_tokens << ")";
    }
    for (int warmup_iter = 0; warmup_iter < 2; ++warmup_iter) {
      capture_stream.synchronize();
      auto warmup_result = model->forward(
          persistent_param_.persistent_tokens(padded_num_tokens_),
          persistent_param_.persistent_positions(padded_num_tokens_),
          kv_cache,
          graph_params_opt.value());
      if (capture_logits_) {
        // Pre-warm lm_head output_buf_ so no torch::empty fires under capture.
        auto warmup_logits =
            model->logits(warmup_result.hidden_states, torch::Tensor());
        persistent_param_.set_logits(warmup_logits);
      }
    }
    capture_stream.synchronize();

    // Record Mate FFI internal scratch allocations on one extra eager forward.
    // The Mate decode .so allocates via TVM-FFI's DLPackManagedTensorAllocator
    // hook (torch::empty), which MUSA rejects under stream capture. We capture
    // the exact sequence of tensors here and replay them during capture_begin.
    recorded_ffi_allocs_.clear();
    xllm::kernel::cuda::begin_ffi_alloc_record();
    {
      auto ffi_forward = model->forward(
          persistent_param_.persistent_tokens(padded_num_tokens_),
          persistent_param_.persistent_positions(padded_num_tokens_),
          kv_cache,
          graph_params_opt.value());
      if (capture_logits_) {
        auto ffi_logits =
            model->logits(ffi_forward.hidden_states, torch::Tensor());
        persistent_param_.set_logits(ffi_logits);
      }
    }
    recorded_ffi_allocs_ = xllm::kernel::cuda::end_ffi_alloc_record();
    capture_stream.synchronize();
    LOG(INFO) << "Recorded " << recorded_ffi_allocs_.size()
              << " Mate FFI scratch tensors for decode graph capture, "
                 "bucket_num_tokens="
              << bucket_num_tokens;

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::cuda::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin graph capture (capture_mode defaults to
    // cudaStreamCaptureModeGlobal)
    // graph_.capture_begin(pool);
    graph_.capture_begin(pool, cudaStreamCaptureModeThreadLocal);

    xllm::kernel::cuda::begin_ffi_alloc_replay(&recorded_ffi_allocs_);
    // Execute forward pass - CUDA graph will capture this
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    // D1: capture lm_head GEMM inside the graph. For B=1 decode we skip
    // index_select (the only known MUSA capture blocker) by passing an
    // undefined selected_token_idxes - hidden_states is already [1, hidden].
    if (capture_logits_) {
      auto captured_logits =
          model->logits(forward_result.hidden_states, torch::Tensor());
      persistent_param_.set_logits(captured_logits);
    }

    // End graph capture
    graph_.capture_end();
    xllm::kernel::cuda::end_ffi_alloc_replay();
  }

  // Synchronize to ensure graph capture is completed.
  capture_stream.synchronize();

  // Explicitly restore stream after capture before replay logic.
  stream_guard.reset();

  // Replay is unified in CudaGraphExecutorImpl::run() after capture success
  // for both prefill and decode.

  LOG(INFO) << "CUDA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

ModelOutput CudaGraph::replay(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_cache,
                              const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Guard CUDA graph replay with a device-level shared lock to allow multiple
  // replay operations to run concurrently while preventing conflicts with
  // capture operations. Replay can share the lock with other replay/prepare
  // operations.
  std::optional<std::shared_lock<std::shared_mutex>> replay_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& replay_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
            device_index_);
    replay_lock_guard.emplace(replay_lock);
  }

  // Update persistent parameters with new input data
  auto full_attention_cache =
      CudaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "CUDA graph replay requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;

  if (is_piecewise_) {
    // Piecewise replay mode (for prefill)
    // Need to get updated params with attn_metadata for attention replay
    auto updated_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true);
    CHECK(updated_params_opt.has_value())
        << "update() should return ModelInputParams for piecewise replay";

    const auto& updated_params = updated_params_opt.value();
    CHECK(piecewise_graph_.num_runners() > 0)
        << "Piecewise graph must have attention runners";
    CHECK(updated_params.attn_metadata)
        << "attn_metadata is required for piecewise replay";
    CHECK(updated_params.attn_metadata->plan_info)
        << "plan_info is required for piecewise replay";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraph::replay() piecewise replay with uri="
        << updated_params.attn_metadata->plan_info->uri
        << ", plan_info.defined="
        << updated_params.attn_metadata->plan_info->plan_info.defined();

    // Build AttentionReplayParams from updated attn_metadata
    ::xllm::kernel::cuda::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    replay_params.plan_info =
        updated_params.attn_metadata->plan_info->plan_info;
    replay_params.q_cu_seq_lens = updated_params.attn_metadata->q_cu_seq_lens;
    replay_params.kv_cu_seq_lens = updated_params.attn_metadata->kv_cu_seq_lens;

    // Replay piecewise graphs and attention runners
    piecewise_graph_.replay(replay_params);
  } else {
    // Normal replay mode (for decode).
    //
    // Request the metadata back from update() so we can refresh the
    // per-CudaGraph persistent host mirrors of paged_kv_* before replaying.
    // The captured graph holds (stable) pointers to our pinned host buffers
    // from capture time; the data inside those buffers must reflect the
    // current step's paged-KV layout, which is what update() just materialized
    // on the corresponding persistent *device* tensors. The returned
    // ModelInputParams is otherwise unused in the replay branch -- it is a
    // single small per-step allocation that pays for itself by avoiding the
    // page fault inside the captured Mate decode kernel (see .mudmp under
    // repro logs and refresh_persistent_paged_kv_host_mirrors() for the
    // pointer-stability rationale).
    auto replay_params_opt = persistent_param_.update(tokens,
                                                      k_cache,
                                                      v_cache,
                                                      positions,
                                                      params,
                                                      padded_num_tokens_,
                                                      /*return_capture_params=*/true);
    CHECK(replay_params_opt.has_value())
        << "update() should return ModelInputParams for decode replay";

    // During graph replay, the captured graph reads paged-KV metadata from
    // persistent *device* tensors (updated by update_llm_decode_metadata_fast
    // _path() above). The pinned host mirrors were set up during capture
    // (see capture() path) and their pointers are baked into the graph, but
    // the graph's FFI batch_decode() call casts paged_kv_*_host to (void) —
    // they are NOT read during replay. The plan_info that consumed the host
    // mirror is cached after first creation and never recomputed on replay.
    //
    // Skipping refresh_persistent_paged_kv_host_mirrors() here avoids 3
    // blocking D2H copies (paged_kv_indptr/indices/last_page_len) that would
    // otherwise stall the CPU for ~48 ms waiting for the previous graph to
    // complete on the same stream. Set XLLM_KEEP_HOST_MIRROR_REFRESH=1 to
    // re-enable for debugging.
    static const bool s_keep_host_mirror_refresh = [] {
      const char* env = std::getenv("XLLM_KEEP_HOST_MIRROR_REFRESH");
      return env && std::string(env) == "1";
    }();
    if (s_keep_host_mirror_refresh) {
      refresh_persistent_paged_kv_host_mirrors(
          replay_params_opt.value().attn_metadata, params.attention.host);
    }

    if (s_enable_graph_timing) {
      static auto s_last_launch = std::chrono::steady_clock::now();
      static int s_step_count = 0;
      c10::cuda::getCurrentCUDAStream(device_index_).synchronize();
      const auto t_sync_end = std::chrono::steady_clock::now();
      const auto step_ms = std::chrono::duration<double, std::milli>(
                               t_sync_end - s_last_launch)
                               .count();
      LOG(INFO) << "GRAPH_TIMING step=" << s_step_count
                << " total=" << step_ms;
      s_step_count++;
      s_last_launch = t_sync_end;
    }

    graph_.replay();
  }

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in CudaGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  ModelOutput output(get_hidden_states(actual_num_tokens));
  if (capture_logits_) {
    output.logits = persistent_param_.logits(actual_num_tokens);
  }
  return output;
}

// CudaGraphExecutorImpl implementation
CudaGraphExecutorImpl::CudaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      enable_prefill_piecewise_graph_(::xllm::ExecutionConfig::get_instance()
                                          .enable_prefill_piecewise_graph()) {
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
  // Keep one pool per executor instance so all captured graphs can reuse it,
  // while avoiding cross-instance stale-handle reuse.
  graph_pool_ = at::cuda::graph_pool_handle();
  // Create single persistent parameter object shared by all CudaGraph instances
  persistent_param_ =
      std::make_unique<CudaGraphPersistentParam>(args_, device_, options_);
  const size_t persistent_bytes =
      persistent_param_->get_persistent_tensor_bytes();
  LOG(INFO) << "Persistent input tensor total size: " << persistent_bytes
            << " bytes (" << (persistent_bytes / (1024 * 1024)) << " MB)";

  const auto private_pool_usage =
      get_private_pools_memory_usage(device_.index());
  baseline_private_pool_reserved_bytes_ = private_pool_usage.reserved_bytes;
  baseline_private_pool_allocated_bytes_ = private_pool_usage.allocated_bytes;
  baseline_private_pool_active_bytes_ = private_pool_usage.active_bytes;
  baseline_allocator_reserved_bytes_ =
      get_allocator_reserved_bytes(device_.index());
}

std::optional<std::pair<torch::Tensor, torch::Tensor>>
CudaGraphExecutorImpl::find_first_full_attention_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    if (cache.empty()) {
      continue;
    }
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return std::make_pair(std::move(k_cache), std::move(v_cache));
    }
  }
  return std::nullopt;
}


namespace {
// Physical pool id: same id => reuse across different shapes (prefill vs decode
// are different physical pools).
constexpr uint32_t kPhysicalPoolIdPrefill = 0;
constexpr uint32_t kPhysicalPoolIdDecode = 1;
}  // namespace

#if !defined(XLLM_TORCH_MUSA)
// ============== VMM Allocator Support ==============
// These functions provide VMM-based memory pool for CUDA Graph capture,
// enabling memory reuse across different shape captures.

struct CudaGraphExecutorImpl::VmmPoolState {
  std::unique_ptr<xllm::SharedVMMAllocator> allocator;
  std::unique_ptr<xllm::VMMTorchAllocator> torch_allocator;
  std::unordered_map<uint32_t, std::unique_ptr<TorchMemPool>> mempools_by_shape;
};

CudaGraphExecutorImpl::~CudaGraphExecutorImpl() {
  // Release captured graphs before MemPool objects to avoid PyTorch MemPool
  // use_count assertion during destruction.
  prefill_graphs_.clear();
  graphs_.clear();
  vmm_pools_.clear();
}

CudaGraphExecutorImpl::VmmPoolState&
CudaGraphExecutorImpl::get_or_create_vmm_pool_state(uint32_t physical_pool_id) {
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto& slot = vmm_pools_[physical_pool_id];
  if (!slot) {
    auto state = std::make_unique<VmmPoolState>();
    state->allocator = std::make_unique<xllm::SharedVMMAllocator>();
    state->allocator->init(device_.index());
    state->torch_allocator =
        std::make_unique<xllm::VMMTorchAllocator>(state->allocator.get());
    slot = std::move(state);
    LOG(INFO) << "Created VMM pool state for executor " << this << ", device "
              << device_.index() << ", physical_pool_id: " << physical_pool_id;
  }
  return *slot;
}

TorchMemPool* CudaGraphExecutorImpl::get_or_create_vmm_mempool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  VmmPoolState& state = get_or_create_vmm_pool_state(physical_pool_id);
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto& mempools = state.mempools_by_shape;
  auto it = mempools.find(shape_id);
  if (it != mempools.end()) {
    return it->second.get();
  }
  auto pool = std::make_unique<TorchMemPool>(state.torch_allocator.get(),
                                             /*is_user_created=*/true);
  TorchMemPool* ptr = pool.get();
  mempools[shape_id] = std::move(pool);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Created per-shape VMM MemPool for executor " << this << ", device "
      << device_.index() << ", physical_pool_id: " << physical_pool_id
      << ", shape_id: " << shape_id << ", pool_id: {" << ptr->id().first << ", "
      << ptr->id().second << "}";
  return ptr;
}

TorchMemPool* CudaGraphExecutorImpl::get_vmm_mempool(uint32_t physical_pool_id,
                                                     uint32_t shape_id) {
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto it = vmm_pools_.find(physical_pool_id);
  if (it == vmm_pools_.end() || !it->second) {
    return nullptr;
  }
  auto& mempools = it->second->mempools_by_shape;
  auto it_pool = mempools.find(shape_id);
  if (it_pool == mempools.end()) {
    return nullptr;
  }
  return it_pool->second.get();
}

CudaGraphExecutorImpl::GraphMemoryUsageStats
CudaGraphExecutorImpl::get_graph_memory_usage_stats() {
  GraphMemoryUsageStats stats;

  if (!::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    const auto pool = get_mem_pool();
    const auto usage = get_graph_pool_memory_usage(device_.index(), pool);
    stats.executor_total_bytes = usage.reserved_bytes;
    stats.allocated_pool_bytes = usage.allocated_bytes;
    stats.active_pool_bytes = usage.active_bytes;
    stats.pool_high_water_mark_bytes = usage.allocated_bytes;

    if (stats.executor_total_bytes == 0) {
      const auto private_pool_usage =
          get_private_pools_memory_usage(device_.index());
      if (private_pool_usage.reserved_bytes >
          baseline_private_pool_reserved_bytes_) {
        stats.executor_total_bytes = private_pool_usage.reserved_bytes -
                                     baseline_private_pool_reserved_bytes_;
      }
      if (private_pool_usage.allocated_bytes >
          baseline_private_pool_allocated_bytes_) {
        stats.allocated_pool_bytes = private_pool_usage.allocated_bytes -
                                     baseline_private_pool_allocated_bytes_;
      }
      if (private_pool_usage.active_bytes >
          baseline_private_pool_active_bytes_) {
        stats.active_pool_bytes = private_pool_usage.active_bytes -
                                  baseline_private_pool_active_bytes_;
      }
      stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
    }

    if (stats.executor_total_bytes == 0) {
      const size_t allocator_reserved_bytes =
          get_allocator_reserved_bytes(device_.index());
      if (allocator_reserved_bytes > baseline_allocator_reserved_bytes_) {
        stats.executor_total_bytes =
            allocator_reserved_bytes - baseline_allocator_reserved_bytes_;
      }
    }
  } else {
#if !defined(XLLM_TORCH_MUSA)
    std::lock_guard<std::mutex> lock(vmm_mutex_);
    for (const auto& kv : vmm_pools_) {
      const VmmPoolState& pool_state = *kv.second;
      stats.executor_total_bytes += pool_state.allocator->mapped_size();
      stats.allocated_pool_bytes += pool_state.allocator->high_water_mark();
      stats.active_pool_bytes += pool_state.allocator->current_offset();
    }
    stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
#endif
  }

  stats.persistent_param_bytes =
      persistent_param_ ? persistent_param_->get_persistent_tensor_bytes() : 0;
  stats.executor_total_bytes += stats.persistent_param_bytes;

  return stats;
}

size_t CudaGraphExecutorImpl::get_graph_memory_usage_bytes() {
  return get_graph_memory_usage_stats().executor_total_bytes;
}

void CudaGraphExecutorImpl::log_graph_memory_after_capture() {
  const auto stats = get_graph_memory_usage_stats();
  const size_t executor_total_bytes = stats.executor_total_bytes;
  const size_t persistent_param_bytes = stats.persistent_param_bytes;
  const size_t allocated_bytes = stats.allocated_pool_bytes;
  const size_t active_bytes = stats.active_pool_bytes;
  const size_t high_water_mark_bytes = stats.pool_high_water_mark_bytes;

  if (executor_total_bytes <= last_logged_executor_total_bytes_) {
    return;
  }
  last_logged_executor_total_bytes_ = executor_total_bytes;

  const bool vmm_enabled =
      ::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool();
  auto format_size = [](size_t bytes) {
    return c10::CachingDeviceAllocator::format_size(bytes);
  };
  LOG(INFO) << "GraphExecutorMemory Usage:"
            << " enable_graph_vmm_pool=" << vmm_enabled
            << " executor_total_memory=" << format_size(executor_total_bytes)
            << " persistent_param=" << format_size(persistent_param_bytes)
            << " allocated_pool_memory=" << format_size(allocated_bytes)
            << " active_pool_memory=" << format_size(active_bytes)
            << " pool_high_water_mark=" << format_size(high_water_mark_bytes);
}

// Switch to new virtual address space before capture for the given physical
// pool. Enables physical memory reuse within that physical pool across shapes.
void CudaGraphExecutorImpl::reset_vmm_allocator_offset(
    uint32_t physical_pool_id) {
  auto& state = get_or_create_vmm_pool_state(physical_pool_id);
  state.allocator->switch_to_new_virtual_space();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Reset VMM allocator for device " << device_.index()
      << ", physical_pool_id: " << physical_pool_id;
}


#else
struct CudaGraphExecutorImpl::VmmPoolState {};

CudaGraphExecutorImpl::~CudaGraphExecutorImpl() {
  prefill_graphs_.clear();
  graphs_.clear();
}

CudaGraphExecutorImpl::VmmPoolState&
CudaGraphExecutorImpl::get_or_create_vmm_pool_state(uint32_t physical_pool_id) {
  LOG(FATAL) << "Graph VMM pool is not enabled for XLLM_TORCH_MUSA builds";
}

TorchMemPool* CudaGraphExecutorImpl::get_or_create_vmm_mempool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  (void)physical_pool_id;
  (void)shape_id;
  LOG(FATAL) << "Graph VMM pool is not enabled for XLLM_TORCH_MUSA builds";
  return nullptr;
}

TorchMemPool* CudaGraphExecutorImpl::get_vmm_mempool(uint32_t physical_pool_id,
                                                     uint32_t shape_id) {
  (void)physical_pool_id;
  (void)shape_id;
  return nullptr;
}

void CudaGraphExecutorImpl::reset_vmm_allocator_offset(
    uint32_t physical_pool_id) {
  (void)physical_pool_id;
}

// Stubs for the graph memory usage reporters. The implementations in the
// !XLLM_TORCH_MUSA branch read torch's private/active pool counters via
// c10::cuda::CUDACachingAllocator, which is not available on torch_musa. The
// call sites (run / run_piecewise) still invoke these unconditionally after a
// successful capture, so we provide zero-returning stubs to keep the link
// satisfied without changing call-site behavior. log_graph_memory_after_capture
// is a no-op here; the captured-bytes diagnostic logging is skipped on MUSA.
CudaGraphExecutorImpl::GraphMemoryUsageStats
CudaGraphExecutorImpl::get_graph_memory_usage_stats() {
  return GraphMemoryUsageStats{};
}

size_t CudaGraphExecutorImpl::get_graph_memory_usage_bytes() {
  return 0;
}

void CudaGraphExecutorImpl::log_graph_memory_after_capture() {}

#endif  // !XLLM_TORCH_MUSA

// Get graph memory pool id for capture. When VMM is enabled, uses per-shape
// MemPool under (physical_pool_id, shape_id).
at::cuda::MempoolId_t CudaGraphExecutorImpl::get_mem_pool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  if (!::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    // Non-VMM mode intentionally uses one pool per executor instance.
    // Rationale: this executor is designed for single-threaded invocation, and
    // concurrent run() on the same executor instance is not allowed.
    // Under this contract, a per-instance pool is safe and keeps graph memory
    // ownership tied to the executor lifecycle.
    return graph_pool_;
  }
  // Per-shape VMM MemPool: look up pool for (physical_pool_id, shape_id).
  TorchMemPool* pool = get_vmm_mempool(physical_pool_id, shape_id);
  CHECK(pool != nullptr)
      << "VMM MemPool for shape_id=" << shape_id
      << ", physical_pool_id=" << physical_pool_id
      << " not found; get_or_create_vmm_mempool must be called before capture";
  return pool->id();
}

// Static method to get CUDA capture stream for current thread
// Each thread gets its own high-priority capture stream
c10::cuda::CUDAStream CudaGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  // Use thread_local to ensure each thread has its own capture stream
  // This is required because CUDA graphs must be captured on a non-default
  // stream. We use high-priority streams for better performance.
  thread_local c10::cuda::CUDAStream thread_capture_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true, device_index);

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized capture_stream for thread: "
              << std::this_thread::get_id()
              << ", stream: " << thread_capture_stream
              << ", device_index: " << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ForwardInput CudaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput CudaGraphExecutorImpl::attach_aux_hidden_states_if_needed(
    const torch::Tensor& hidden_states,
    uint32_t n_tokens) const {
  if (options_.enable_graph_aux_hidden_states()) {
    auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
    if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
  }
  return ModelOutput(hidden_states);
}

ModelInputParams CudaGraphExecutorImpl::maybe_precompute_embedding_for_graph(
    const torch::Tensor& tokens,
    const ModelInputParams& params) const {
#ifdef XLLM_TORCH_MUSA
  // Only intervene on decode or MTP spec-verify validate graph paths.
  // Piecewise prefill already breaks the graph at attention boundaries.
  const bool in_spec_verify_embedding_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if (!params.meta.batch_forward_type.is_decode() &&
      !in_spec_verify_embedding_phase) {
    return params;
  }
  // Caller already provided pre-computed embeddings (multimodal encoder path).
  // The existing CudaGraphPersistentParam::update() will copy this through to
  // persistent_embedding_ exactly as it does today; do not double-compute.
  if (params.embedding.input_embedding.defined()) {
    return params;
  }
  // Skip when the model doesn't expose a generic WordEmbedding layer. Some
  // backends (e.g. raw NPU) use a custom embedding op that lives behind a
  // different interface, in which case we have nothing to hoist out of the
  // graph and fall through to existing behavior.
  auto embed_layer = model_->get_word_embedding();
  if (embed_layer.is_empty()) {
    return params;
  }

  // Compute the embedding eagerly while still outside the captured stream
  // region. On torch_musa 2.7.1 the underlying at::musa::IndexSelect calls
  // EmptyMUSA -> musaMemMap to allocate its output, which is illegal during
  // stream capture ("operation not permitted when stream is capturing"); by
  // running the lookup here the allocation goes through the normal caching
  // allocator path and stays out of the capture region.
  //
  // The downstream wiring is already in place:
  //   * CudaGraphPersistentParam::update() copies `params.embedding
  //     .input_embedding` into `persistent_embedding_` (see the update path
  //     under "Update persistent embedding from input_embedding if
  //     available").
  //   * For capture, `build_capture_params_if_needed` rewrites
  //     `params_for_capture->embedding.input_embedding` to a view of
  //     `persistent_embedding_`, so the captured forward references the
  //     persistent buffer's stable address.
  //   * For replay, the captured graph already references that same
  //     persistent address; refreshing the buffer contents here is sufficient
  //     to feed each step with the correct per-token embeddings.
  //
  // Qwen3Next-family models (including Qwen3.5) already honour
  // `input_params.embedding.input_embedding` in their forward, branching
  // around the in-graph `embed_tokens_(tokens)` call when the field is
  // defined (see xllm/models/llm/qwen3_next_hybrid_base.h).
  ModelInputParams new_params = params;
  new_params.embedding.input_embedding = embed_layer(tokens);
  return new_params;
#else
  (void)tokens;
  return params;
#endif
}

ModelOutput CudaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                       const torch::Tensor& positions,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  const bool is_prefill = params.meta.batch_forward_type.is_prefill();
  const bool is_decode = params.meta.batch_forward_type.is_decode();

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const uint32_t bucket_num_tokens =
      get_bucket_num_tokens(n_tokens, is_prefill);

  // Prefill phase with piecewise graph
  if (is_prefill && enable_prefill_piecewise_graph_) {
    // Check if token count is within limit
    const bool graph_mode_supported = n_tokens <= max_tokens_for_graph_mode_;

    if (!graph_mode_supported) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Token count " << n_tokens
          << " exceeds max_tokens_for_graph_mode ("
          << max_tokens_for_graph_mode_ << "), falling back to eager mode";
      COUNTER_INC(num_model_execution_total_eager);
      size_t free_b = 0, total_b = 0;
      musaMemGetInfo(&free_b, &total_b);
      LOG(INFO) << "PREFILL_EAGER mem_before_forward: free=" << (double)free_b / 1e9
                << " GB, n_tokens=" << n_tokens;
      auto result = model_->forward(tokens, positions, kv_caches, params);
      size_t free_a = 0, total_a = 0;
      musaMemGetInfo(&free_a, &total_a);
      LOG(INFO) << "PREFILL_EAGER mem_after_forward: free=" << (double)free_a / 1e9
                << " GB (delta=" << (double)(free_a - free_b) / 1e9 << " GB)";
      Device::empty_cache(/*device_index=*/-1);
      size_t free_c = 0, total_c = 0;
      musaMemGetInfo(&free_c, &total_c);
      LOG(INFO) << "PREFILL_EAGER mem_after_empty_cache: free=" << (double)free_c / 1e9
                << " GB (delta_from_forward=" << (double)(free_c - free_a) / 1e9 << " GB)";
      return result;
    }

    // Check if piecewise graph exists for this bucket
    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      // Replay existing piecewise graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in prefill piecewise replay mode";
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist, try to create it lazily with piecewise capture
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in prefill piecewise capture mode";

    TorchMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdPrefill);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdPrefill, shape_id);
    }
    const at::cuda::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdPrefill, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing piecewise CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      prefill_graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = prefill_graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Fail fast intentionally: after entering graph mode, silently falling back
    // to eager can hide allocator/capture regressions and make latency behavior
    // non-deterministic in production. Operators can disable graph mode via
    // ::xllm::ExecutionConfig::get_instance().enable_graph() or
    // ::xllm::ExecutionConfig::get_instance().enable_prefill_piecewise_graph()
    // when fallback behavior is preferred over strict graph-mode enforcement.
    LOG(FATAL)
        << "Failed to capture piecewise CUDA graph for bucket num_tokens: "
        << bucket_num_tokens << " (actual num_tokens: " << n_tokens << ")";
  }

  // Prefill without piecewise graph: use eager mode
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    auto result = model_->forward(tokens, positions, kv_caches, params);
    Device::empty_cache(/*device_index=*/-1);
    return result;
  }

  // Decode phase with full graph
  if (is_decode) {
    // Check if conditions are suitable for graph execution (replay or capture)
    const auto max_seq_len = args_.max_position_embeddings();
    const bool seq_len_supported = params.meta.kv_max_seq_len <= max_seq_len;

    // Early return if conditions are not suitable for graph operations
    if (!seq_len_supported) {
      LOG(WARNING) << "Not suitable for CUDA graph operations, falling back to "
                      "eager mode.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // On MUSA the in-graph IndexSelect (embedding lookup) is the only known
    // capture blocker. Compute the embedding here, outside the captured
    // stream region, and pass it through `params.embedding.input_embedding`
    // so both capture and replay paths read from the persistent embedding
    // buffer. No-op on other platforms / non-decode forwards. See
    // maybe_precompute_embedding_for_graph() for the full rationale.
    const ModelInputParams graph_params =
        maybe_precompute_embedding_for_graph(tokens, params);

    // Check if captured graph exists for this bucket num_tokens
    auto it = graphs_.find(bucket_num_tokens);
    if (it != graphs_.end()) {
      // Replay the existing graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in decode replay mode";
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      if (result.logits.defined()) {
        output.logits = result.logits;
      }
      return output;
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in decode capture mode";

    TorchMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdDecode);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdDecode, shape_id);
    }
    const at::cuda::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdDecode, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          graph_params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests. Recompute the embedding so the persistent buffer
      // reflects the current token batch (the capture-time embedding above
      // would otherwise be reused unchanged, which is only correct in the
      // unlikely case the post-capture request happens to match exactly).
      const ModelInputParams replay_params =
          maybe_precompute_embedding_for_graph(tokens, params);
      auto result = graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, replay_params);
      auto output =
          attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
      if (result.logits.defined()) {
        output.logits = result.logits;
      }
      return output;
    }

    // Same fail-fast policy as prefill capture above: keep graph-mode behavior
    // explicit and avoid silently switching execution semantics after a capture
    // failure. Use ::xllm::ExecutionConfig::get_instance().enable_graph() to
    // turn off graph mode if eager fallback is desired for resiliency.
    LOG(FATAL) << "Failed to capture CUDA graph for bucket num_tokens: "
               << bucket_num_tokens << " (actual num_tokens: " << n_tokens
               << ")";
  }

  // MTP spec-verify validate phase (Qwen3.5 hybrid linear attention).
  const bool in_spec_verify_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if (in_spec_verify_phase) {
    static const bool force_spec_verify_eager =
        std::getenv("XLLM_SPEC_VERIFY_EAGER") != nullptr;
    if (force_spec_verify_eager) {
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    if (!model_->is_hybrid_linear_attention()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode for spec verify because the "
             "chunked-prefill validate graph path is currently only adapted "
             "for hybrid linear attention models.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    if (!params.graph.use_expanded_decode_for_spec_verify_attention) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode for spec verify because expanded "
             "decode attention graph input was not prepared.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    const auto max_seq_len = args_.max_position_embeddings();
    if (params.meta.kv_max_seq_len > max_seq_len) {
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    const ModelInputParams graph_params =
        maybe_precompute_embedding_for_graph(tokens, params);
    const uint64_t graph_key = get_graph_key(bucket_num_tokens, graph_params);

    auto it = spec_verify_graphs_.find(graph_key);
    if (it != spec_verify_graphs_.end()) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in spec-verify replay mode";
      auto result =
          it->second->replay(tokens, positions, kv_caches, graph_params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in spec-verify capture mode";

    TorchMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdDecode);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdDecode, shape_id);
    }
    const at::cuda::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdDecode, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          graph_params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing CUDA spec-verify graph for bucket "
                   "num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ", q_max_seq_len: " << graph_params.meta.q_max_seq_len
                << ") done";
      log_graph_memory_after_capture();
      spec_verify_graphs_[graph_key] = std::move(graph);
      const ModelInputParams replay_params =
          maybe_precompute_embedding_for_graph(tokens, params);
      auto result = spec_verify_graphs_[graph_key]->replay(
          tokens, positions, kv_caches, replay_params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    LOG_FIRST_N(WARNING, 1)
        << "Failed to capture CUDA spec-verify graph for bucket num_tokens: "
        << bucket_num_tokens << ", falling back to eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Chunked prefill without a spec-verify graph path (e.g. draft extend) runs
  // eager: only MTP validate expanded-decode has a captured graph today.
  if (params.meta.batch_forward_type.is_chunked_prefill() ||
      params.meta.batch_forward_type.is_mixed()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for chunked prefill/mixed batch "
           "without a CUDA graph path (bucket num_tokens="
        << bucket_num_tokens << ", type="
        << params.meta.batch_forward_type.to_string() << ").";
    COUNTER_INC(num_model_execution_total_eager);
    auto result = model_->forward(tokens, positions, kv_caches, params);
    Device::empty_cache(/*device_index=*/-1);
    return result;
  }

  // Defensive fallback for unsupported forward types (should be unreachable for
  // normal prefill/decode paths).
  LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

uint64_t CudaGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill()) {
    const uint64_t q_max_seq_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t CudaGraphExecutorImpl::get_bucket_num_tokens(uint32_t num_tokens,
                                                      bool is_prefill) const {
  // no_padding only works for decode, prefill requires padding for graph reuse
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding() &&
      !is_prefill) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 8, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

// NOTE: REGISTER_EXECUTOR for CudaGraphExecutorImpl lives in
// cuda_graph_executor_impl.h. Keeping it in this .cpp meant the static
// initializer's TU was referenced only via runtime factory lookup, and the
// linker dropped libcuda_graph_executor.a's only .o as unused. Putting the
// macro in the header matches base/vlm/acl/mlu/dcu graph executors.

}  // namespace xllm::runtime::cuda
