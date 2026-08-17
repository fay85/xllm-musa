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
  int64_t non_registered_weight_bytes() const { return 0; }

 private:
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
  torch::Tensor dense_attention(const torch::Tensor& q,
                                const torch::Tensor& kv,
                                const torch::Tensor& compressed_kv,
                                const torch::Tensor& compressed_indices,
                                bool causal,
                                int64_t q_abs_start,
                                bool use_full_kv) const;
  torch::Tensor compress_prefill_sequence(const torch::Tensor& hidden_states,
                                          const torch::Tensor& cos,
                                          const torch::Tensor& sin,
                                          size_t sequence_index);
  torch::Tensor compress_decode_step(const torch::Tensor& hidden_step,
                                     int64_t start_pos,
                                     const torch::Tensor& cos,
                                     const torch::Tensor& sin,
                                     size_t sequence_index);

  int64_t num_heads_ = 0;
  int64_t hidden_dim_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_lora_rank_ = 0;
  int64_t o_lora_rank_ = 0;
  int64_t o_groups_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t nope_head_dim_ = 0;
  int64_t window_size_ = 0;
  int64_t compress_ratio_ = 1;
  int64_t compressor_coff_ = 1;
  int64_t n_local_heads_ = 0;
  int64_t n_local_groups_ = 0;
  int64_t tp_rank_ = 0;
  int64_t tp_size_ = 1;
  double softmax_scale_ = 1.0;
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

  torch::Tensor q_rms_gamma_;
  torch::Tensor attn_sink_;
  torch::Tensor compressor_ape_;
  std::vector<torch::Tensor> compressor_hidden_history_;
  std::vector<torch::Tensor> compressor_kv_history_;
  std::vector<torch::Tensor> compressor_kv_state_;
  std::vector<torch::Tensor> compressor_score_state_;
  bool attn_sink_loaded_ = false;
  bool compressor_ape_loaded_ = false;
};
TORCH_MODULE(DeepseekV4Attention);

}  // namespace layer
}  // namespace xllm
