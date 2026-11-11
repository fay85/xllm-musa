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
#include "layers/mlu/deepseek_v41/deepseek_v41_metadata.h"

namespace xllm::layer {
class DeepseekV41IndexerImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV41IndexerImpl(const ModelContext& context,
                                  int32_t layer_id);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  torch::Tensor project_query(const torch::Tensor& qr,
                              const torch::Tensor& positions,
                              const torch::Tensor& rope);
  torch::Tensor project_indexer(
      const torch::Tensor& x,
      const torch::Tensor& latent,
      const torch::Tensor& positions,
      const torch::Tensor& slots,
      const torch::Tensor& rope,
      const torch::Tensor& cache,
      const std::optional<torch::Tensor>& num_valid_input_rows = std::nullopt,
      const std::optional<torch::Tensor>& num_valid_compressed_rows =
          std::nullopt);
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
