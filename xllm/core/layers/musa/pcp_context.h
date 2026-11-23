/* Copyright 2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer::musa_pcp {

int32_t single_prefill_front_rows(int32_t query_rows, int32_t history_rows);

class SingleCheckpointSlotsScope {
 public:
  explicit SingleCheckpointSlotsScope(const std::vector<int32_t>* slots);
  ~SingleCheckpointSlotsScope();

  SingleCheckpointSlotsScope(const SingleCheckpointSlotsScope&) = delete;
  SingleCheckpointSlotsScope& operator=(const SingleCheckpointSlotsScope&) = delete;

 private:
  const std::vector<int32_t>* previous_;
};

const std::vector<int32_t>* single_checkpoint_slots();

class SingleCheckpointWarmupScope {
 public:
  SingleCheckpointWarmupScope();
  ~SingleCheckpointWarmupScope();

  SingleCheckpointWarmupScope(const SingleCheckpointWarmupScope&) = delete;
  SingleCheckpointWarmupScope& operator=(const SingleCheckpointWarmupScope&) = delete;

 private:
  bool previous_;
};

bool in_single_checkpoint_warmup();

// Rows owned by the last rank that earlier ranks write into their KV cache
// after the layers.
struct DeferredKv final {
  torch::Tensor rows;
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
};

// Recurrent-state rows that earlier ranks copy from the last rank after the
// layers.
struct DeferredState final {
  torch::Tensor cache;
  torch::Tensor indices;
};

// Rows from the preceding rank whose transfer was posted ahead of their use.
struct PendingReceive final {
  c10::intrusive_ptr<c10d::Work> work;
  torch::Tensor rows;
};

// Prefill context parallelism splits one fresh prompt into contiguous row
// ranges, one per CP rank. Inside the layers data only flows from earlier to
// later ranks, so the last rank ends with the full KV cache and the final
// recurrent state. Earlier ranks catch up after the layers only when they
// decode as well.
struct Context final {
  ProcessGroup* cp_group = nullptr;
  int32_t cp_rank = 0;
  int32_t batch_size = 1;
  bool is_batch = false;
  // Rank r owns packed local rows of count (row_offsets[r + 1] - row_offsets[r]).
  // For single-sequence (!is_batch), rank r owns contiguous global rows
  // [row_offsets[r], row_offsets[r + 1]). For multi-sequence (is_batch), rank 0
  // owns all sequences' front halves [f_0, ..., f_{B-1}] and rank 1 owns all
  // sequences' back halves [r_0, ..., r_{B-1}].
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> rank0_seq_ends;
  int32_t max_rows = 0;
  bool replicate_final_state = true;
  torch::Tensor global_slot_mapping;
  torch::Tensor rank0_row_indices;
  torch::Tensor rank1_row_indices;
  torch::Tensor unpermute_indices;
  torch::Tensor rank0_slot_mapping;
  torch::Tensor rank1_slot_mapping;
  torch::Tensor conv_prefix_row_indices;
  AttentionMetadata local_metadata;
  std::vector<DeferredKv> deferred_kv;
  std::vector<DeferredState> deferred_states;
  std::deque<PendingReceive> pending_receives;
  // Buffers a root sent without waiting. A captured graph would otherwise
  // reuse their memory before the transfer stream reads them, so they live
  // until the final gather joins that stream.
  std::vector<torch::Tensor> in_flight_sends;
  std::vector<c10::intrusive_ptr<c10d::Work>> in_flight_send_works;
};

std::vector<int32_t> multibatch_front_lens(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    bool is_prefill,
    bool lengths_are_cumulative = false);

struct BucketInputs final {
  int32_t batch_size = 0;
  int32_t rank0_capacity = 0;
  int32_t rank1_capacity = 0;
  std::vector<int32_t> front_lens;
  std::vector<int32_t> back_lens;
  std::vector<int32_t> kv_lens;
  bool any_history = false;
  torch::Tensor indices;
  torch::Tensor metadata;
  bool gdn_checkpoint_runner = false;
  std::vector<int32_t> linear_state_ids;
};

std::vector<int32_t> multibatch_bucket_capacities(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    bool is_prefill,
    int32_t stride,
    int64_t max_tokens,
    bool lengths_are_cumulative = false);

std::shared_ptr<BucketInputs> make_bucket_inputs(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    bool is_prefill,
    const std::vector<int32_t>& capacities,
    const torch::Device& device);

void update_bucket_inputs(BucketInputs& inputs,
                          const std::vector<int32_t>& q_seq_lens,
                          const std::vector<int32_t>& kv_seq_lens,
                          bool is_prefill);

class BucketInputsScope final {
 public:
  explicit BucketInputsScope(BucketInputs* inputs);
  ~BucketInputsScope();
  BucketInputsScope(const BucketInputsScope&) = delete;
  BucketInputsScope& operator=(const BucketInputsScope&) = delete;

 private:
  BucketInputs* previous_ = nullptr;
};

bool has_bucket_inputs();

const BucketInputs* get_bucket_inputs();

// Emplaces a context when the batch is one fresh prefill long enough to split
// across the CP group, and leaves `context` empty otherwise. `num_rows` is the
// row count of the forward, including graph padding.
void build_context(const AttentionMetadata& global_metadata,
                   const ParallelArgs& parallel_args,
                   int32_t num_rows,
                   std::optional<Context>* context);

// Joins pending async MCCL send works onto the active compute/capture stream
// while keeping `in_flight_sends` buffers alive until the PCP scope finishes.
void join_pending_sends(Context& context);

// Whether this rank continues rows of a preceding rank and therefore takes its
// recurrent state from the preceding rank rather than from the cache.
bool receives_prefix_state(const Context& context);

// Rows of `global` owned by this rank along `dim`.
torch::Tensor shard_rows(const torch::Tensor& global,
                         const Context& context,
                         int64_t dim);

// Concatenates the rows of every rank in rank order.
torch::Tensor gather_rows(const torch::Tensor& local, const Context& context);

// Data produced on the preceding rank flows to the rank that continues its
// rows, one transfer per call in layer order on every rank. Receivers post a
// transfer before the rows are needed so it overlaps their own work, then take
// the oldest posted rows. Each call is a no-op on ranks it does not concern.
void expect_prefix_rows(torch::IntArrayRef shape,
                        const torch::TensorOptions& options,
                        Context& context);
// Posts `batch_size` rows of `cache`, the per-sequence state slots.
void expect_prefix_cache_row(const torch::Tensor& cache, Context& context);
// Posts the causal-conv1d boundary prefix (`[conv_state_len, qkv_dim]` for a
// single sequence or `[batch_size, conv_state_len, qkv_dim]` for multi-batch).
void expect_conv_prefix(int64_t conv_state_len,
                        int64_t qkv_dim,
                        const torch::TensorOptions& options,
                        Context& context);
// Extracts the trailing `conv_state_len` rows per sequence on rank 0.
torch::Tensor extract_conv_prefix(const torch::Tensor& local_rows,
                                  int64_t conv_state_len,
                                  int64_t qkv_dim,
                                  const Context& context);
torch::Tensor take_prefix_rows(Context& context);
void receive_prefix_cache_row(torch::Tensor& cache,
                              const torch::Tensor& indices,
                              Context& context);
void send_prefix_rows(const torch::Tensor& rows, Context& context);
void send_prefix_cache_row(const torch::Tensor& cache,
                           const torch::Tensor& indices,
                           Context& context);

// Posts the transfer of projected key/value rows from the preceding rank.
// projected_options must describe the projection's compute dtype and device,
// not the paged cache storage: an FP8 cache still receives BF16 projected rows.
void expect_prefix_kv(int64_t num_kv_heads,
                      int64_t head_dim,
                      const torch::TensorOptions& projected_options,
                      Context& context);

// Writes this rank's key/value rows and every earlier rank's rows into this
// rank's paged cache.
void exchange_kv(const torch::Tensor& key,
                 const torch::Tensor& value,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 KVCache& kv_cache,
                 Context& context);

// Marks final recurrent-state rows that earlier ranks copy from the last rank
// after the layers.
void defer_final_state(const torch::Tensor& cache,
                       const torch::Tensor& indices,
                       Context& context);

// Completes the deferred KV and recurrent-state copies and returns the hidden
// states of every row in rank order.
torch::Tensor finish(const torch::Tensor& local_hidden_states,
                     Context& context);

// Completes the deferred KV and recurrent-state copies without gathering hidden
// states across ranks (for intermediate chunks where logits/bootstrap are absent).
torch::Tensor finish_state_only(const torch::Tensor& local_hidden_states,
                                Context& context);

}  // namespace xllm::layer::musa_pcp
