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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_pcp_context.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_sp_context.h"

namespace xllm {
namespace layer {

using KVState = std::tuple<torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor>;

class DeepseekV4AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV4AttentionImpl() = default;
  explicit DeepseekV4AttentionImpl(const ModelContext& context,
                                   int32_t layer_id = -1);
  DeepseekV4AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          int32_t layer_id = -1);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const DSAMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      KVCache& kv_cache,
      KVState& kv_state,
      bool is_prefill,
      bool is_chunked_prefill,
      const std::
          tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&
              compress_metadata);

  void load_state_dict(const StateDict& state_dict);
  void write_context_kv(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        const torch::Tensor& cache_slots,
                        KVCache& kv_cache);
  int64_t non_registered_weight_bytes() const { return 0; }

  void set_sequence_parallel_context(
      const dsv4_musa_sp::DeepseekV4SPContext* context) {
    sequence_parallel_context_ = context;
  }

  // Prefill context parallelism (PCP): the earlier CP group streams its
  // per-layer K/V rows and compressor recurrent state to the later group so
  // the later group's cache and recurrence see the full causal history.
  void set_pcp_context(dsv4_musa_pcp::Context* context) {
    pcp_context_ = context;
  }

 private:
  // Current accuracy-protected implementation.  A separately linked legacy
  // implementation may override only the Graph target-q6 route; keeping the
  // entry point split makes that performance A/B explicit and reversible.
  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward_current_impl(
      const DSAMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      KVCache& kv_cache,
      KVState& kv_state,
      bool is_prefill,
      bool is_chunked_prefill,
      const std::
          tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&
              compress_metadata);

  std::vector<std::pair<int64_t, int64_t>> build_query_ranges(
      const DSAMetadata& attn_metadata,
      int64_t total_tokens) const;
  std::vector<int64_t> build_kv_lens(
      const DSAMetadata& attn_metadata,
      size_t num_sequences,
      const std::vector<std::pair<int64_t, int64_t>>& query_ranges) const;
  torch::Tensor gather_cache(const torch::Tensor& cache,
                             const torch::Tensor& block_table,
                             int64_t sequence_index,
                             int64_t kv_len) const;
  torch::Tensor paged_dense_attention(
      const torch::Tensor& q,
      const torch::Tensor& cache,
      const torch::Tensor& block_table,
      int64_t sequence_index,
      int64_t kv_len,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& compressed_kv,
      const torch::Tensor& compressed_cache,
      const torch::Tensor& compressed_block_table,
      int64_t compressed_kv_len) const;
  torch::Tensor mate_sparse_attention(
      const torch::Tensor& q,
      const torch::Tensor& packed_cache,
      const torch::Tensor& logical_indices,
      const torch::Tensor& compressed_block_table,
      const torch::Tensor& extra_packed_cache,
      const torch::Tensor& swa_block_table,
      int64_t sequence_index,
      int64_t kv_len,
      const torch::Tensor& output = torch::Tensor(),
      const torch::Tensor& physical_indices = torch::Tensor(),
      const torch::Tensor& extra_physical_indices = torch::Tensor()) const;
  // Compact q6 verification uses one MATE launch with per-row causal lengths
  // instead of a host loop over q rows. The MATE ABI accepts vectorized
  // top-k/extra lengths and masks each row independently.
  torch::Tensor mate_sparse_attention_batched(
      const torch::Tensor& q,
      const torch::Tensor& packed_cache,
      const torch::Tensor& logical_indices,
      const torch::Tensor& compressed_block_table,
      const torch::Tensor& extra_packed_cache,
      const torch::Tensor& swa_block_table,
      int64_t sequence_index,
      int64_t scalar_kv_len,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& output = torch::Tensor(),
      const torch::Tensor& physical_indices = torch::Tensor(),
      const torch::Tensor& extra_physical_indices = torch::Tensor());
  torch::Tensor dense_attention(const torch::Tensor& q,
                                const torch::Tensor& kv,
                                const torch::Tensor& compressed_kv,
                                const torch::Tensor& compressed_indices,
                                bool causal,
                                int64_t q_abs_start,
                                bool use_full_kv) const;
  torch::Tensor compress_prefill_sequence(
      const torch::Tensor& hidden_states,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      size_t sequence_index,
      int64_t prefix_snapshot_tokens = 0,
      int64_t cached_prefix_tokens = 0,
      const torch::Tensor& projected_kv = {},
      const torch::Tensor& projected_score = {});
  torch::Tensor compress_decode_step(const torch::Tensor& hidden_step,
                                     int64_t start_pos,
                                     const torch::Tensor& cos,
                                     const torch::Tensor& sin,
                                     size_t sequence_index,
                                     const torch::Tensor& projected_kv = {},
                                     const torch::Tensor& projected_score = {});
  torch::Tensor compress_indexer_prefill_sequence(
      const torch::Tensor& hidden_states,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      size_t sequence_index,
      int64_t prefix_snapshot_tokens = 0,
      int64_t cached_prefix_tokens = 0,
      const torch::Tensor& projected_kv = {},
      const torch::Tensor& projected_score = {});
  torch::Tensor compress_indexer_decode_step(
      const torch::Tensor& hidden_step,
      int64_t start_pos,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      size_t sequence_index,
      const torch::Tensor& projected_kv = {},
      const torch::Tensor& projected_score = {});
  torch::Tensor build_indexer_topk(
      const torch::Tensor& qr,
      const torch::Tensor& hidden_states,
      const torch::Tensor& precomputed_head_weights,
      const torch::Tensor& precomputed_query,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      const torch::Tensor& index_kv,
      const torch::Tensor& index_cache,
      const torch::Tensor& index_scale_cache,
      const torch::Tensor& index_block_table,
      int64_t sequence_index,
      int64_t index_len,
      int64_t q_abs_start,
      const torch::Tensor& output = torch::Tensor(),
      const torch::Tensor& mate_compressed_block_table = torch::Tensor(),
      const torch::Tensor& mate_swa_block_table = torch::Tensor(),
      int64_t mate_kv_len = 0,
      int64_t mate_block_size = 0,
      torch::Tensor* mate_physical_indices = nullptr,
      torch::Tensor* mate_extra_physical_indices = nullptr,
      const torch::Tensor& dynamic_kv_len = torch::Tensor(),
      int64_t batched_query_rows_per_sequence = 0,
      int64_t batched_sequence_count = 1);

  int64_t num_heads_ = 0;
  int64_t hidden_dim_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_lora_rank_ = 0;
  int64_t o_lora_rank_ = 0;
  int64_t o_groups_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t nope_head_dim_ = 0;
  int64_t window_size_ = 0;
  int64_t dspark_block_size_ = 0;
  int64_t compress_ratio_ = 1;
  int64_t compressor_coff_ = 1;
  int64_t n_local_heads_ = 0;
  int64_t n_local_groups_ = 0;
  int64_t tp_rank_ = 0;
  int64_t tp_size_ = 1;
  int64_t decode_workspace_tokens_ = 1;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 0;
  double softmax_scale_ = 1.0;
  double indexer_score_scale_ = 1.0;
  double eps_ = 1e-6;
  int32_t layer_id_ = 0;

  ReplicatedLinear q_a_proj_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  RMSNorm q_layernorm_{nullptr};
  ReplicatedLinear kv_proj_{nullptr};
  RMSNorm kv_layernorm_{nullptr};
  ColumnParallelLinear o_a_proj_{nullptr};
  RowParallelLinear o_b_proj_{nullptr};
  ReplicatedLinear compressor_wkv_{nullptr};
  ReplicatedLinear compressor_wgate_{nullptr};
  RMSNorm compressor_norm_{nullptr};
  ReplicatedLinear indexer_wq_b_{nullptr};
  ReplicatedLinear indexer_weights_proj_{nullptr};
  ReplicatedLinear indexer_compressor_wkv_{nullptr};
  ReplicatedLinear indexer_compressor_wgate_{nullptr};
  RMSNorm indexer_compressor_norm_{nullptr};
  // SP prefill projects the main and indexer compressor KV/score values with
  // one private GEMM before issuing one packed sequence collective.
  torch::Tensor sp_joint_compressor_projection_weight_;

  torch::Tensor q_rms_gamma_;
  // wo_a is consumed outside the generic Linear forward. Materialize its FP8
  // checkpoint weight once during loading instead of dequantizing every step.
  torch::Tensor o_a_projection_weight_;
  torch::Tensor decode_o_b_reduce_output_;
  // Decode packs wq_a and wkv into one materialized BF16 projection.  The
  // q branch then runs on a private stream while the main stream finishes KV
  // normalization/cache writes.  Prefill keeps the original row shapes.
  torch::Tensor decode_q_a_norm_output_;
  torch::Tensor qkv_a_weight_;
  torch::Tensor decode_qkv_a_output_;
  torch::Tensor decode_q_down_;
  torch::Tensor decode_kv_down_;
  bool qkv_a_weight_loaded_ = false;
  bool qkv_a_indexer_weight_loaded_ = false;
  // C4 layers consume main and indexer Q-B projections from the same
  // normalized low-rank query. Decode packs both weights into one GEMM while
  // retaining row-sliced main-Q projection for larger prefill shapes.
  torch::Tensor q_indexer_b_weight_;
  torch::Tensor decode_q_indexer_b_output_;
  torch::Tensor decode_q_main_output_;
  bool q_indexer_b_weight_loaded_ = false;
  // Graph-owned decode outputs prevent consecutive q2 token launches from
  // aliasing a backend workspace during capture/replay.
  torch::Tensor decode_paged_attention_output_;
  torch::Tensor decode_spec_attention_output_;
  torch::Tensor decode_inverse_rope_output_;
  torch::Tensor decode_wo_a_output_;
  // Replay-time valid KV lengths are derived from persistent input positions.
  torch::Tensor decode_kv_seq_lens_;
  torch::Tensor decode_spec_indices_output_;
  torch::Tensor decode_spec_mate_physical_indices_;
  torch::Tensor decode_spec_mate_extra_physical_indices_;
  // Single-token graph replay owns all compressor/indexer state and output
  // buffers. Multi-token/MTP paths keep their existing per-request tensors.
  torch::Tensor decode_compressor_kv_slots_;
  torch::Tensor decode_compressor_score_slots_;
  torch::Tensor decode_indexer_kv_slots_;
  torch::Tensor decode_indexer_score_slots_;
  torch::Tensor decode_compressor_kv_state_;
  torch::Tensor decode_compressor_score_state_;
  torch::Tensor decode_indexer_kv_state_;
  torch::Tensor decode_indexer_score_state_;
  torch::Tensor decode_compressor_kv_projection_;
  torch::Tensor decode_compressor_score_projection_;
  torch::Tensor decode_compressor_output_;
  torch::Tensor decode_compressor_norm_output_;
  torch::Tensor decode_compressor_rope_output_;
  torch::Tensor decode_indexer_kv_projection_;
  torch::Tensor decode_indexer_score_projection_;
  torch::Tensor decode_indexer_output_;
  torch::Tensor decode_indexer_norm_output_;
  torch::Tensor decode_indexer_rope_output_;
  torch::Tensor decode_indexer_head_weight_projection_;
  torch::Tensor decode_indexer_head_weights_;
  torch::Tensor decode_indexer_topk_output_;
  torch::Tensor attn_sink_;
  torch::Tensor compressor_ape_;
  torch::Tensor indexer_compressor_ape_;
  torch::Tensor indexer_hadamard_;
  std::vector<torch::Tensor> compressor_kv_history_;
  std::vector<torch::Tensor> compressor_kv_state_;
  std::vector<torch::Tensor> compressor_score_state_;
  // Prefix snapshots are captured in the private state cache by the MUSA
  // metadata path. They are intentionally separate from the live suffix
  // state, because the same layer may persist a newly advanced state while a
  // later request still needs the immutable prefix boundary.
  std::vector<torch::Tensor> prefix_compressor_kv_state_;
  std::vector<torch::Tensor> prefix_compressor_score_state_;
  std::vector<torch::Tensor> indexer_kv_history_;
  std::vector<torch::Tensor> indexer_kv_state_;
  std::vector<torch::Tensor> indexer_score_state_;
  std::vector<torch::Tensor> prefix_indexer_kv_state_;
  std::vector<torch::Tensor> prefix_indexer_score_state_;
  bool attn_sink_loaded_ = false;
  bool compressor_ape_loaded_ = false;
  bool indexer_compressor_ape_loaded_ = false;
  const dsv4_musa_sp::DeepseekV4SPContext* sequence_parallel_context_ = nullptr;
  dsv4_musa_pcp::Context* pcp_context_ = nullptr;
};
TORCH_MODULE(DeepseekV4Attention);

}  // namespace layer
}  // namespace xllm
