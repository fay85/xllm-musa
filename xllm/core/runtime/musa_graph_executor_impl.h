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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <torch/torch.h>

#include "torch_musa/csrc/core/MUSAStream.h"

// torch_musa's native MUSAGraph header expects this stream alias.
namespace at::musa {
using c10::musa::MUSAStream;
}  // namespace at::musa

#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

// MUSA host_defines.h defines __noinline__ as a macro. Folly uses that token
// inside __attribute__((__noinline__)), so leaving it defined creates a nested
// attribute during the MUSA device parse.
#ifdef __noinline__
#undef __noinline__
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/kernels/musa/llm_decode_metadata_update.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/options.h"

namespace xllm::runtime::musa {

using MusaMemPool = c10::musa::MemPool;

// Helper class to hold persistent parameters for MUSA graph execution
// Multiple MusaGraph instances can share the same MusaGraphPersistentParam
// object
class MusaGraphPersistentParam final {
 public:
  MusaGraphPersistentParam(const ModelArgs& args,
                           const torch::Device& device,
                           const runtime::Options& options);

  ~MusaGraphPersistentParam() = default;

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
  const torch::Device& device() const { return device_; }
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
  torch::Tensor persistent_kv_cache_tokens_nums(
      uint32_t actual_batch_size) const {
    if (actual_batch_size > 0) {
      return persistent_kv_cache_tokens_nums_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_kv_cache_tokens_nums_;
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

 private:
  bool can_use_llm_decode_fast_path(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      const ModelInputParams& params,
      const layer::Fa3AttentionMetadata& fa3_metadata) const;
  void update_llm_decode_metadata_fast_path(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      const ModelInputParams& params,
      const layer::Fa3AttentionMetadata& fa3_metadata,
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
  torch::Tensor persistent_kv_cache_tokens_nums_;
  torch::Tensor persistent_num_accepted_tokens_;
  torch::Tensor aux_hidden_states_;

  // FlashInfer decode mode parameters
  torch::Tensor persistent_paged_kv_indptr_;
  torch::Tensor persistent_paged_kv_indices_;
  torch::Tensor persistent_paged_kv_last_page_len_;
  torch::Tensor persistent_decode_qo_indptr_;
  torch::Tensor persistent_kv_seq_lens_delta_;
};

// MUSA graph executor using torch_musa MUSAGraph for memory management.
class MusaGraph final {
 public:
  // capture_stream: the stream to use for MUSA graph capture.
  explicit MusaGraph(MusaGraphPersistentParam& persistent_param,
                     at::DeviceIndex device_index,
                     c10::musa::MUSAStream capture_stream)
      : persistent_param_(persistent_param),
        device_index_(device_index),
        capture_stream_(capture_stream) {}

  // Capture computation graph for given bucket num_tokens
  bool capture(CausalLM* model,
               const ModelArgs& args,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens,
               const c10::musa::MempoolId_t& pool,
               MusaMemPool* pool_ptr = nullptr);

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

  // Reference to persistent parameters (shared across multiple MusaGraph
  // instances).
  MusaGraphPersistentParam& persistent_param_;

  at::DeviceIndex device_index_;
  // Native MUSA capture stream owned by MusaGraphExecutorImpl.
  c10::musa::MUSAStream capture_stream_;

  uint32_t padded_num_tokens_ = 0;

  // FA3 scheduler metadata consumed by the captured full-attention kernels.
  // Its address is fixed at capture time; replay copies freshly generated
  // values into the same storage before graph launch.
  torch::Tensor captured_fa3_scheduler_metadata_;

  // Mate FFI scratch tensors recorded during an eager warmup pass and replayed
  // during graph capture so the hook never calls torch::empty under capture.
  // Must outlive the graph holders declared last below.
  std::vector<torch::Tensor> recorded_ffi_allocs_;

  // Declare graph holders last so they are destroyed before the tensors and
  // scratch buffers whose addresses were retained during capture.
  at::musa::MUSAGraph graph_;
};

// Executor implementation using MUSA graph optimization
class MusaGraphExecutorImpl final : public ExecutorImpl {
 public:
  MusaGraphExecutorImpl(CausalLM* model,
                        const ModelArgs& args,
                        const torch::Device& device,
                        const runtime::Options& options);

  ~MusaGraphExecutorImpl() override;

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

  // Lazy-loaded MUSA graphs for decode phase (by bucket_num_tokens).
  absl::flat_hash_map<uint32_t, std::unique_ptr<MusaGraph>> graphs_;

  // Persistent parameters shared across all MusaGraph instances
  std::unique_ptr<MusaGraphPersistentParam> persistent_param_;

  // MUSA graph memory pool shared across all MusaGraph instances.
  // This executor is expected to be called from a single worker thread (no
  // concurrent run() on the same executor instance), so sharing one pool per
  // executor is intentional. If concurrent calls are introduced in the future,
  // this assumption must be revisited.
  c10::musa::MempoolId_t graph_pool_;

  // Get bucket num_tokens for given num_tokens.
  // Decode: 1/2/4/8 then multiples of 16, or exact when no_padding is enabled.
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;

  ModelOutput attach_aux_hidden_states_if_needed(
      const torch::Tensor& hidden_states,
      uint32_t n_tokens) const;

  ModelInputParams maybe_precompute_embedding_for_graph(
      const torch::Tensor& tokens,
      const ModelInputParams& params) const;

  // Get MUSA graph memory pool id for capture. When VMM is enabled, uses
  // per-shape MemPool under (physical_pool_id, shape_id). Same physical_pool_id
  // => reuse across different shapes (e.g. prefill vs decode are different
  // pools).
  c10::musa::MempoolId_t get_mem_pool(uint32_t physical_pool_id = 0,
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
  MusaMemPool* get_or_create_vmm_mempool(uint32_t physical_pool_id,
                                         uint32_t shape_id);
  MusaMemPool* get_vmm_mempool(uint32_t physical_pool_id, uint32_t shape_id);
  GraphMemoryUsageStats get_graph_memory_usage_stats();
  void log_graph_memory_after_capture();

  std::mutex vmm_mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<VmmPoolState>> vmm_pools_;

  size_t baseline_private_pool_reserved_bytes_ = 0;
  size_t baseline_private_pool_allocated_bytes_ = 0;
  size_t baseline_private_pool_active_bytes_ = 0;
  size_t baseline_allocator_reserved_bytes_ = 0;

  size_t last_logged_executor_total_bytes_ = 0;

  // Get the MUSA-compatible capture stream for the current thread.
  // Each thread automatically gets its own high-priority capture stream
  // Returns the stream and device index
  static c10::musa::MUSAStream get_capture_stream(
      c10::DeviceIndex device_index);
};

// REGISTER_EXECUTOR generates a static initializer in an anonymous namespace.
// Putting it in the header (matching base/vlm/acl/mlu/dcu graph executors)
// means each TU that includes this header emits its own initializer copy, so
// the static initializer is guaranteed to run from at least one .o file that
// IS linked into the final executable (the musa_graph_executor_impl.cpp .o is
// otherwise referenced only via runtime factory lookup, and the linker drops
// the whole TU as unused). At runtime the factory's emplace() dedupes the
// duplicates so only the first registration takes effect.
REGISTER_EXECUTOR("musa", MusaGraphExecutorImpl);

}  // namespace xllm::runtime::musa
