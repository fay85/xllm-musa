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

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <absl/container/flat_hash_map.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
#include <ATen/cuda/MemPool.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#if defined(XLLM_TORCH_MUSA)
#include "core/kernels/musa/llm_decode_metadata_update.h"
#include "core/kernels/musa/piecewise_graphs.h"
#else
#include "core/kernels/cuda/llm_decode_metadata_update.h"
#include "core/kernels/cuda/piecewise_graphs.h"
#endif
#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "options.h"

namespace xllm::runtime::cuda {

constexpr uint64_t kSpecVerifyGraphKeyMask = 1ull << 63;
constexpr uint64_t kSpecVerifyQMaxSeqLenShift = 32;

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
using TorchMemPool = at::cuda::MemPool;
#else
using TorchMemPool = c10::cuda::MemPool;
#endif

// Helper class to hold persistent parameters for CUDA graph execution
// Multiple CudaGraph instances can share the same CudaGraphPersistentParam
// object
class CudaGraphPersistentParam {
 public:
  CudaGraphPersistentParam(const ModelArgs& args,
                           const torch::Device& device,
                           const runtime::Options& options);

  ~CudaGraphPersistentParam() = default;

  // Update persistent tensors with new input data
  // If return_capture_params is true, returns a ModelInputParams with
  // persistent buffer references. padded_num_tokens must be > 0 when
  // return_capture_params is true, used for build new ModelInputParams for
  // capture. If return_capture_params is false, only updates persistent buffers
  // and returns std::nullopt.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens = 0,
                                         bool return_capture_params = false);

  // Getter methods for persistent tensors
  torch::Tensor persistent_tokens(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_tokens_;
  }
  torch::Tensor persistent_positions(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_positions_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_positions_;
  }
  torch::Tensor persistent_new_cache_slots(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_new_cache_slots_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_new_cache_slots_;
  }
  torch::Tensor persistent_block_tables(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_block_tables_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_block_tables_;
  }
  torch::Tensor hidden_states(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return hidden_states_;
  }
  // Setter for hidden_states (for assignment)
  void set_hidden_states(const torch::Tensor& value) {
    const uint32_t result_tokens = value.size(0);
    hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens)
        .copy_(value, /*non_blocking=*/true);
  }
  // Logits captured inside the graph (D1). [num_seqs, vocab_size].
  torch::Tensor logits(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return logits_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return logits_;
  }
  void set_logits(const torch::Tensor& value) {
    const uint32_t result_tokens = value.size(0);
    logits_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens)
        .copy_(value, /*non_blocking=*/true);
  }
  bool has_logits_buffer() const { return logits_.defined(); }
  const torch::Device& device() const { return device_; }
  void ensure_logits_buffer(int64_t vocab_size, torch::ScalarType dtype,
                            const torch::Device& device) {
    if (!logits_.defined()) {
      logits_ = torch::empty({options_.max_tokens_per_batch(), vocab_size},
                             torch::TensorOptions().dtype(dtype).device(device));
    }
  }
  torch::Tensor q_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return q_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return q_seq_lens_;
  }
  torch::Tensor kv_seq_lens(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return kv_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return kv_seq_lens_;
  }
  torch::Tensor persistent_embedding(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_embedding_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_embedding_;
  }
  torch::Tensor persistent_linear_state_indices(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_linear_state_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_linear_state_indices_;
  }
  torch::Tensor persistent_num_accepted_tokens(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_num_accepted_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_num_accepted_tokens_;
  }
  torch::Tensor aux_hidden_states(uint32_t actual_tokens) const {
    if (!aux_hidden_states_.defined() || aux_hidden_states_.numel() == 0) {
      return aux_hidden_states_;
    }
    if (actual_tokens > 0) {
      return aux_hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return aux_hidden_states_;
  }
  // Setter for aux_hidden_states (for assignment)
  void set_aux_hidden_states(const torch::Tensor& value);
  size_t get_persistent_tensor_bytes() const;
  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_paged_kv_indptr_;
  }
  torch::Tensor persistent_paged_kv_indices(uint32_t actual_size) const {
    if (actual_size > 0) {
      return persistent_paged_kv_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_size);
    }
    return persistent_paged_kv_indices_;
  }
  torch::Tensor persistent_paged_kv_last_page_len(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_paged_kv_last_page_len_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_paged_kv_last_page_len_;
  }
  torch::Tensor persistent_decode_qo_indptr(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_decode_qo_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1);
    }
    return persistent_decode_qo_indptr_;
  }
  torch::Tensor persistent_kv_seq_lens_delta(uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_kv_seq_lens_delta_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_kv_seq_lens_delta_;
  }
  torch::Tensor persistent_expanded_block_tables(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_expanded_block_tables_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_expanded_block_tables_;
  }
  torch::Tensor expanded_kv_seq_lens(uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return expanded_kv_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return expanded_kv_seq_lens_;
  }
  torch::Tensor persistent_expanded_paged_kv_indptr(
      uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_expanded_paged_kv_indptr_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens + 1);
    }
    return persistent_expanded_paged_kv_indptr_;
  }
  torch::Tensor persistent_expanded_paged_kv_indices(
      uint32_t actual_size) const {
    if (actual_size > 0) {
      return persistent_expanded_paged_kv_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_size);
    }
    return persistent_expanded_paged_kv_indices_;
  }
  torch::Tensor persistent_expanded_paged_kv_last_page_len(
      uint32_t actual_tokens) const {
    if (actual_tokens > 0) {
      return persistent_expanded_paged_kv_last_page_len_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_expanded_paged_kv_last_page_len_;
  }

 private:
  std::vector<int32_t> update_expanded_spec_decode_attention(
      const ModelInputParams& input_params,
      uint32_t actual_num_tokens,
      uint32_t padded_num_tokens);
  bool can_use_llm_decode_fast_path(const torch::Tensor& tokens,
                                    const torch::Tensor& positions,
                                    const ModelInputParams& params) const;
  void update_llm_decode_metadata_fast_path(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            const ModelInputParams& params,
                                            uint32_t padded_num_tokens,
                                            int64_t actual_batch_size,
                                            int64_t actual_num_tokens);

  const ModelArgs& args_;
  const torch::Device& device_;
  const runtime::Options& options_;

  // Persistent tensors - basic parameters
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_new_cache_slots_;
  torch::Tensor persistent_block_tables_;
  torch::Tensor hidden_states_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor persistent_embedding_;
  torch::Tensor persistent_linear_state_indices_;
  torch::Tensor persistent_num_accepted_tokens_;
  torch::Tensor aux_hidden_states_;
  // [max_tokens_per_batch, vocab_size] - persistent logits output for D1.
  torch::Tensor logits_;

  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr_;
  torch::Tensor persistent_paged_kv_indices_;
  torch::Tensor persistent_paged_kv_last_page_len_;
  torch::Tensor persistent_decode_qo_indptr_;
  torch::Tensor persistent_kv_seq_lens_delta_;

  // TODO maybe not used. or use q_cu_seq_lens instead.
  torch::Tensor persistent_chunked_prefill_qo_indptr_;

  // Qwen3.5 MTP spec-verify expanded decode attention (per validate token).
  torch::Tensor persistent_expanded_block_tables_;
  torch::Tensor expanded_kv_seq_lens_;
  torch::Tensor persistent_expanded_paged_kv_indptr_;
  torch::Tensor persistent_expanded_paged_kv_indices_;
  torch::Tensor persistent_expanded_paged_kv_last_page_len_;
};

// CUDA graph executor using libtorch CUDAGraph for memory management
class CudaGraph {
 public:
  // is_piecewise: if true, use piecewise graph capture for prefill
  // capture_stream: the stream to use for CUDA graph capture
  explicit CudaGraph(CudaGraphPersistentParam& persistent_param,
                     at::DeviceIndex device_index,
                     at::cuda::CUDAStream capture_stream,
                     bool is_piecewise = false)
      : persistent_param_(persistent_param),
        device_index_(device_index),
        capture_stream_(capture_stream),
        is_piecewise_(is_piecewise) {}

  // Capture computation graph for given bucket num_tokens
  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens,
               const at::cuda::MempoolId_t& pool,
               TorchMemPool* pool_ptr = nullptr);

  // Replay captured graph with new input data
  ModelOutput replay(const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     std::vector<KVCache>& kv_cache,
                     const ModelInputParams& params);

  // Get the hidden states from the last capture
  torch::Tensor get_hidden_states(uint32_t actual_num_tokens) const {
    return persistent_param_.hidden_states(actual_num_tokens);
  }

 private:
  // Print graph held tensors for debugging
  void print_graph_tensors() const;

  // Refresh the persistent host mirrors used by the Mate FFI batch_decode
  // run() call. Lazily allocates pinned CPU buffers sized to worst case (set
  // by CudaGraphPersistentParam at executor construction), then copies the
  // current device-tensor contents into them and overwrites
  // `attn_metadata->paged_kv_*_host` to reference those persistent buffers.
  //
  // Called once after persistent_param_.update() returns, for the warmup
  // forward, the FFI record pass, AND the captured pass -- they all reuse the
  // same shared_ptr<AttentionMetadata>.
  //
  // On replay (graph_.replay() path), the captured graph already references
  // the persistent host buffer pointers from capture time; this method
  // refreshes their *contents* so the captured H2D copy sees fresh values.
  // When attention.host paged-KV mirrors are populated (normal LLM-engine
  // path), copies CPU->pinned host directly and avoids per-step D2H sync.
  void refresh_persistent_paged_kv_host_mirrors(
      const std::shared_ptr<layer::AttentionMetadata>& attn_metadata,
      const AttentionHostInput& host_src);

  // CUDA graph for capturing and replaying (decode mode)
  at::cuda::CUDAGraph graph_;
  // Piecewise graphs for prefill mode
  PiecewiseGraphs piecewise_graph_;
  // Whether this graph uses piecewise capture
  bool is_piecewise_ = false;

  uint32_t padded_num_tokens_;

  // D1: when true, lm_head GEMM was captured inside the graph. On replay,
  // persistent_param_.logits() holds the computed logits.
  bool capture_logits_ = false;

  // Reference to persistent parameters (shared across multiple CudaGraph
  // instances)
  CudaGraphPersistentParam& persistent_param_;

  // CUDA stream for graph capture (reference, owned by CudaGraphExecutorImpl)
  at::cuda::CUDAStream capture_stream_;
  at::DeviceIndex device_index_;

  // Mate FFI scratch tensors recorded during an eager warmup pass and replayed
  // during graph capture so the hook never calls torch::empty under capture.
  // Must outlive graph_ (destruction order: graph_ first, then this vector).
  std::vector<torch::Tensor> recorded_ffi_allocs_;

  // Persistent host (CPU) mirrors of paged_kv_* tensors, owned by this graph.
  //
  // Why these exist: the Mate FFI batch_decode `run` function takes
  // kDLCPU pointers for paged_kv_indptr / paged_kv_indices /
  // paged_kv_last_page_len. Inside the FFI those host buffers are read at
  // submit time *and* their pointers may be baked into captured device
  // operations (e.g., for the FmhaFwdKernelWarpSpecialized parameter
  // struct). If we let `.to(kCPU)` create a fresh per-call tensor, then on
  // every replay the captured graph holds a dangling pointer to the
  // previous-step host buffer (already freed). On torch_musa 2.7.1 this
  // surfaces as a GPU page fault inside the captured Mate decode kernel
  // ("ExceptionType: IllegalAddress ... Reading from 0x... Fault (Page
  // Directory)"; see the .mudmp under repro logs).

  // Grow-only across captures so smaller-bucket graphs keep referencing
  // the same storage even when a larger bucket later expands the buffer.
  //
  // PRE-CAPTURE PRE-ALLOCATION (set in capture(), enforced inside
  // refresh_persistent_paged_kv_host_mirrors):
  //   The first allocation MUST size the buffer to the maximum possible
  //   numel for this CudaGraph instance, not the warmup-time numel. If
  //   we sized to warmup-time (typically 1 block per sequence), then
  //   when the KV cache crosses a block boundary (e.g., decode step 38
  //   of a 27-token-prefill question with block_size=64), the helper's
  //   `host_buf.numel() < numel` check would trigger a realloc to a new
  //   storage. The captured graph still references the OLD storage's
  //   data_ptr (baked into the FmhaFwdKernelWarpSpecialized param
  //   struct), so it reads stale/freed memory and produces a small but
  //   nonzero divergence at L3 (first full-attention layer). That
  //   divergence cascades through all downstream layers and surfaces as
  //   silently-wrong arithmetic in the generated text. Pre-allocating
  //   to the worst-case size makes subsequent refresh_one() calls a
  //   no-op for the alloc branch and keeps the captured pointer stable.
  torch::Tensor paged_kv_indptr_host_buf_;
  torch::Tensor paged_kv_indices_host_buf_;
  torch::Tensor paged_kv_last_page_len_host_buf_;

  // Pre-computed max numel for each host buf (set in capture()).
  // 0 means "no pre-allocation hint", and refresh_one() falls back to its
  // legacy "alloc to current device numel" behavior. Non-zero means the
  // first allocation will be max(device_numel, hint).
  int64_t paged_kv_indptr_host_max_numel_{0};
  int64_t paged_kv_indices_host_max_numel_{0};
  int64_t paged_kv_last_page_len_host_max_numel_{0};
};

// Executor implementation using CUDA graph optimization
class CudaGraphExecutorImpl : public ExecutorImpl {
 public:
  CudaGraphExecutorImpl(CausalLM* model,
                        const ModelArgs& args,
                        const torch::Device& device,
                        const runtime::Options& options);

  ~CudaGraphExecutorImpl() override;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

  // Return current graph executor memory usage in bytes (including persistent
  // parameters). Exposed for tests and diagnostics.
  size_t get_graph_memory_usage_bytes();

  static std::optional<std::pair<torch::Tensor, torch::Tensor>>
  find_first_full_attention_cache(const std::vector<KVCache>& kv_caches);

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  // Lazy-loaded CUDA graphs for decode phase (by bucket_num_tokens)
  absl::flat_hash_map<uint32_t, std::unique_ptr<CudaGraph>> graphs_;

  // Lazy-loaded CUDA graphs for MTP spec-verify validate (composite key)
  absl::flat_hash_map<uint64_t, std::unique_ptr<CudaGraph>> spec_verify_graphs_;

  // Lazy-loaded CUDA graphs for prefill phase with piecewise capture
  // (by bucket_num_tokens)
  absl::flat_hash_map<uint32_t, std::unique_ptr<CudaGraph>> prefill_graphs_;

  // Chunked-prefill piecewise graphs require an exact host-shape key because
  // Qwen3.5 GDN control flow depends on per-sequence query lengths.
  absl::flat_hash_map<uint64_t, std::unique_ptr<CudaGraph>>
      chunked_prefill_graphs_;

  // Persistent parameters shared across all CudaGraph instances
  std::unique_ptr<CudaGraphPersistentParam> persistent_param_;

  // CUDA graph memory pool shared across all CudaGraph instances.
  // This executor is expected to be called from a single worker thread (no
  // concurrent run() on the same executor instance), so sharing one pool per
  // executor is intentional. If concurrent calls are introduced in the future,
  // this assumption must be revisited.
  at::cuda::MempoolId_t graph_pool_;
  // Whether to enable prefill piecewise graph
  bool enable_prefill_piecewise_graph_;
  int64_t max_tokens_for_graph_mode_ = 0;

  // Get bucket num_tokens for given num_tokens
  // For num_tokens < 8: use 1, 2, 4, 8
  // For num_tokens >= 8: use multiples of 8
  // When is_prefill=true, no_padding is disabled (prefill requires padding)
  uint32_t get_bucket_num_tokens(uint32_t num_tokens,
                                 bool is_prefill = false) const;

  uint64_t get_graph_key(uint32_t bucket_num_tokens,
                         const ModelInputParams& params) const;

  uint64_t get_chunked_prefill_graph_key(
      uint32_t bucket_num_tokens,
      const ModelInputParams& params) const;

  ModelOutput attach_aux_hidden_states_if_needed(
      const torch::Tensor& hidden_states,
      uint32_t n_tokens) const;

  ModelInputParams maybe_precompute_embedding_for_graph(
      const torch::Tensor& tokens,
      const ModelInputParams& params) const;

  // Get CUDA graph memory pool id for capture. When VMM is enabled, uses
  // per-shape MemPool under (physical_pool_id, shape_id). Same physical_pool_id
  // => reuse across different shapes (e.g. prefill vs decode are different
  // pools).
  at::cuda::MempoolId_t get_mem_pool(uint32_t physical_pool_id = 0,
                                     uint32_t shape_id = 0);

  // Switch VMM allocator to a new virtual address space before capture for the
  // given physical pool. Enables physical memory reuse within that pool across
  // shapes (max(shape) instead of sum(shape)).
  void reset_vmm_allocator_offset(uint32_t physical_pool_id);

  struct VmmPoolState;

  struct GraphMemoryUsageStats {
    size_t executor_total_bytes = 0;
    size_t persistent_param_bytes = 0;
    size_t allocated_pool_bytes = 0;
    size_t active_pool_bytes = 0;
    size_t pool_high_water_mark_bytes = 0;
  };

  VmmPoolState& get_or_create_vmm_pool_state(uint32_t physical_pool_id);
  TorchMemPool* get_or_create_vmm_mempool(uint32_t physical_pool_id,
                                          uint32_t shape_id);
  TorchMemPool* get_vmm_mempool(uint32_t physical_pool_id, uint32_t shape_id);
  GraphMemoryUsageStats get_graph_memory_usage_stats();
  void log_graph_memory_after_capture();

  std::mutex vmm_mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<VmmPoolState>> vmm_pools_;

  size_t baseline_private_pool_reserved_bytes_ = 0;
  size_t baseline_private_pool_allocated_bytes_ = 0;
  size_t baseline_private_pool_active_bytes_ = 0;
  size_t baseline_allocator_reserved_bytes_ = 0;

  size_t last_logged_executor_total_bytes_ = 0;

  // Get CUDA capture stream for current thread
  // Each thread automatically gets its own high-priority capture stream
  // Returns the stream and device index
  static c10::cuda::CUDAStream get_capture_stream(
      c10::DeviceIndex device_index);
};

// REGISTER_EXECUTOR generates a static initializer in an anonymous namespace.
// Putting it in the header (matching base/vlm/acl/mlu/dcu graph executors)
// means each TU that includes this header emits its own initializer copy, so
// the static initializer is guaranteed to run from at least one .o file that
// IS linked into the final executable (the cuda_graph_executor_impl.cpp .o is
// otherwise referenced only via runtime factory lookup, and the linker drops
// the whole TU as unused). At runtime the factory's emplace() dedupes the
// duplicates so only the first registration takes effect. Picking the backend
// key at compile time avoids the macro's class##_registered symbol collision.
#if defined(XLLM_TORCH_MUSA)
REGISTER_EXECUTOR("musa", CudaGraphExecutorImpl);
#else
REGISTER_EXECUTOR("cuda", CudaGraphExecutorImpl);
#endif

}  // namespace xllm::runtime::cuda
