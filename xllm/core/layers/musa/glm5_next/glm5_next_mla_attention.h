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

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/musa/flash_mla_sparse.h"
#include "core/kernels/musa/glm5_next_indexer.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"

namespace xllm::layer {

// Complete NoPE sparse-attention layer, including replicated rate-4 IndexPool.
// Main head projections are TP sharded; indexer and latent cache are
// replicated. Uses canonical self_attn.* checkpoint names after the caller
// strips its layer prefix. BF16/FP32 weights and E4M3 weights with FP32 128x128
// block scales are accepted; projections run BF16, head-weight scoring runs
// FP32 (as in vLLM).
class Glm5NextMLAAttentionImpl final : public torch::nn::Module {
 public:
  Glm5NextMLAAttentionImpl(const ModelArgs& args,
                           const ParallelArgs& parallel,
                           torch::TensorOptions options,
                           bool enable_indexer = true);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  std::pair<torch::Tensor, kernel::musa::Glm5NextIndexerSelection> forward(
      const torch::Tensor& positions,
      const torch::Tensor& hidden_states,
      const AttentionMetadata& metadata,
      KVCache& cache,
      const kernel::musa::Glm5NextIndexerSelection* shared_selection = nullptr);

  // Exact layer cache layout, using the existing IndexedKVCache ownership and
  // transfer interface. Slot zero of the tail pool is reserved for padding.
  static KVCache create_cache(int64_t pages,
                              int64_t state_slots,
                              const torch::TensorOptions& options,
                              bool fp8_main_cache = false);

 private:
  std::vector<std::string> required_weights() const;
  bool weights_ready() const;
  torch::Tensor dense_weight(const std::string& name,
                             torch::ScalarType dtype) const;
  void materialize_weights();
  int64_t hidden_size_;
  int64_t q_rank_;
  int64_t heads_;
  int64_t token_topk_;
  bool enable_indexer_;
  bool loaded_ = false;
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> pending_;
  ReplicatedLinear fused_qkv_a_proj_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  ColumnParallelLinear kv_b_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RMSNorm q_norm_{nullptr};
  RMSNorm kv_norm_{nullptr};
  ReplicatedLinear index_q_{nullptr};
  ReplicatedLinear index_kw_{nullptr};
  ReplicatedLinear index_weights_{nullptr};
  ReplicatedLinear index_gate_{nullptr};
  RMSNorm index_norm_{nullptr};
  torch::Tensor ape_;
  torch::Tensor w_k_;
  torch::Tensor w_v_;
};
TORCH_MODULE(Glm5NextMLAAttention);

using Glm5NextSparseAttentionImpl = Glm5NextMLAAttentionImpl;
using Glm5NextSparseAttention = Glm5NextMLAAttention;

// GLM NoPE attention core after projections/norm and IndexPool selection.
// q [T,H,256], latent [T,512], w_k/w_v [H,512,256] are BF16; heads are
// already sharded by the caller. Returns [T,H,256] before row-parallel o_proj.
// candidates [T,K] contains logical token IDs with -1 padding. lengths [T]
// bounds each candidate prefix; rows [T] maps queries to block_table requests.
// Cache is BF16 [P,page_size,1,512] or inline FP8 uint8 [P,page_size,1,528].
// For FP8 each row becomes an independent single-query decode item. The caller
// supplies valid unique write slots [T] int64, or -1 for padded input rows.
// All inputs are device tensors. Supports eager and graph execution on MP31.
inline torch::Tensor glm5_next_mla_attention(const torch::Tensor& q,
                                             const torch::Tensor& latent,
                                             const torch::Tensor& w_k,
                                             const torch::Tensor& w_v,
                                             const torch::Tensor& candidates,
                                             const torch::Tensor& lengths,
                                             const torch::Tensor& rows,
                                             const torch::Tensor& positions,
                                             const torch::Tensor& block_table,
                                             const torch::Tensor& slots,
                                             torch::Tensor& cache) {
  CHECK_EQ(q.dim(), 3);
  CHECK_EQ(q.size(2), 256);
  CHECK_EQ(q.scalar_type(), torch::kBFloat16);
  CHECK(q.device().is_privateuseone());
  c10::DeviceGuard device_guard(q.device());
  CHECK(q.size(1) == 8 || q.size(1) == 16 || q.size(1) == 32 ||
        q.size(1) == 64);
  CHECK_EQ(latent.sizes(), torch::IntArrayRef({q.size(0), 512}));
  CHECK_EQ(w_k.sizes(), torch::IntArrayRef({q.size(1), 512, 256}));
  CHECK_EQ(w_v.sizes(), w_k.sizes());
  for (const auto& tensor : {latent, w_k, w_v}) {
    CHECK_EQ(tensor.device(), q.device());
    CHECK_EQ(tensor.scalar_type(), q.scalar_type());
  }
  CHECK_EQ(cache.dim(), 4);
  CHECK(cache.is_contiguous());
  CHECK_EQ(cache.device(), q.device());
  const bool fp8 = cache.scalar_type() == torch::kUInt8;
  CHECK(fp8 || cache.scalar_type() == q.scalar_type());
  CHECK_EQ(cache.size(2), 1);
  CHECK_EQ(cache.size(3), fp8 ? 528 : 512);
  CHECK_EQ(candidates.size(0), q.size(0));
  CHECK_EQ(lengths.sizes(), torch::IntArrayRef({q.size(0)}));
  CHECK_EQ(lengths.device(), q.device());
  CHECK_EQ(lengths.scalar_type(), torch::kInt32);
  CHECK(lengths.is_contiguous());
  // Map before cache mutation so unsupported metadata fails before writes.
  auto indices = kernel::musa::flash_mla_sparse_map_indices(
      candidates, rows, positions, block_table, cache.size(1), cache.size(0));
  auto q_latent = torch::bmm(q.transpose(0, 1), w_k.transpose(1, 2))
                      .transpose(0, 1)
                      .contiguous();
  torch::Tensor z;
  if (fp8) {
    kernel::musa::flash_mla_sparse_fp8_cache_write(
        latent.contiguous(), slots, cache);
    z = kernel::musa::flash_mla_sparse_nope_fp8(
            q_latent.unsqueeze(1),
            cache,
            indices.unsqueeze(1).unsqueeze(1),
            lengths,
            0.0625)
            .squeeze(1);
  } else {
    kernel::musa::flash_mla_sparse_cache_write(
        latent.contiguous(), slots, cache);
    z = kernel::musa::flash_mla_sparse_bf16(q_latent,
                                            cache.view({-1, 1, 512}),
                                            indices.unsqueeze(1),
                                            lengths,
                                            0.0625);
  }
  return torch::bmm(z.transpose(0, 1), w_v).transpose(0, 1).contiguous();
}
}  // namespace xllm::layer
