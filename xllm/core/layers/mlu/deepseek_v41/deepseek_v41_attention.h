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
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "layers/common/linear.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"

namespace xllm::layer {
class DeepseekV41AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV41AttentionImpl(const ModelContext& context, int32_t layer_id);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  void write_context_kv(const torch::Tensor& hidden,
                        const torch::Tensor& positions,
                        const torch::Tensor& rope,
                        const torch::Tensor& cache_slots,
                        KVCache& cache);
  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& positions,
      const torch::Tensor& rope,
      std::vector<KVCache>& caches,
      const ModelInputParams& input,
      Csa2Context& shared,
      const mlu_v4_cp::DeepseekV4CpContext* cp_context = nullptr);

 private:
  void decode_attention(const torch::Tensor& q,
                        const torch::Tensor& selected,
                        const torch::Tensor& lengths,
                        torch::Tensor output);
  void forward_cached_attention(
      const torch::Tensor& q,
      const torch::Tensor& kv,
      const torch::Tensor& raw,
      const torch::Tensor& scores,
      const torch::Tensor& iq,
      const torch::Tensor& iw,
      const torch::Tensor& positions,
      const torch::Tensor& rope,
      std::vector<KVCache>& caches,
      const ModelInputParams& input,
      Csa2Context& shared,
      torch::Tensor output,
      torch::Tensor selected,
      torch::Tensor lengths,
      const mlu_v4_cp::DeepseekV4CpContext* cp_context);
  void update_compressed_cache(const torch::Tensor& raw,
                               const torch::Tensor& scores,
                               int64_t begin,
                               int32_t seq,
                               const torch::Tensor& rope,
                               KVCache& cache,
                               const std::vector<torch::Tensor>& tables);
  void compute_attention(const torch::Tensor& q,
                         const torch::Tensor& window,
                         const torch::Tensor& main,
                         const torch::Tensor& indices,
                         int64_t begin,
                         torch::Tensor output);
  torch::Tensor project_output(torch::Tensor output,
                               const torch::Tensor& positions,
                               const torch::Tensor& rope);
  ModelArgs args_;
  int32_t layer_id_;
  int32_t source_ = -1;
  int32_t ratio_ = 0;
  bool index_source_ = false;
  ProcessGroup* tp_group_;
  int64_t rank_;
  int64_t tp_size_;
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  ColumnParallelLinear wq_b_{nullptr};
  RowParallelLinear wo_b_{nullptr};
  DeepseekV41Indexer indexer_{nullptr};
};
TORCH_MODULE(DeepseekV41Attention);
}  // namespace xllm::layer
