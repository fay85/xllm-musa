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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::layer {
// One forward's cross-layer selection. KV itself remains framework-owned.
struct Csa2Context {
  // Each defined entry has shape {query_rows, k}: one top-k list per query.
  // A selection with no keys keeps the two-dimensional shape {query_rows, 0}.
  std::vector<torch::Tensor> topk;
  std::vector<torch::Tensor> candidates;
  std::vector<torch::Tensor> block_tables;
  // Graph-owned scratch. Piecewise segments consume then overwrite this on
  // the same replay stream; addresses remain fixed for the graph lifetime.
  torch::Tensor selected_kv;
  torch::Tensor selected_lengths;
};

class DeepseekV41IndexerImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV41IndexerImpl(const ModelContext& context,
                                  int32_t layer_id);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  std::pair<torch::Tensor, torch::Tensor> project_query(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const torch::Tensor& positions,
      const torch::Tensor& rope);
  torch::Tensor project_key(const torch::Tensor& latent,
                            const torch::Tensor& positions,
                            const torch::Tensor& rope);
  void select_candidate_indices(const torch::Tensor& iq,
                                const torch::Tensor& iw,
                                const torch::Tensor& cache,
                                const torch::Tensor& table,
                                int64_t length,
                                int32_t seq,
                                Csa2Context& shared);
  void select_decode_indices(const torch::Tensor& iq,
                             const torch::Tensor& iw,
                             const torch::Tensor& cache,
                             const torch::Tensor& table,
                             int64_t length,
                             int32_t seq,
                             Csa2Context& shared);
  void select_indices(const torch::Tensor& iq,
                      const torch::Tensor& iw,
                      const torch::Tensor& keys,
                      int64_t begin,
                      int32_t seq,
                      Csa2Context& shared);

 private:
  ModelArgs args_;
  int32_t layer_id_;
  int32_t ratio_;
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weights_;
};
TORCH_MODULE(DeepseekV41Indexer);
}  // namespace xllm::layer
