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
#include <utility>

#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::layer {
StateDict build_deepseek_v41_hash_state(const ModelArgs& args,
                                        const std::string& tokenizer_path);

class DeepseekV41EngramImpl final : public torch::nn::Module {
 public:
  class HashState final {
   public:
    HashState(const DeepseekV41EngramImpl& owner,
              int64_t rows,
              int64_t query_width = 1);
    void prepare(const ModelInputParams& input);
    torch::Tensor forward(const torch::Tensor& tokens);
    std::pair<torch::Tensor, torch::Tensor> lookup_rows() const;

   private:
    static void compute(void* state);
    const DeepseekV41EngramImpl& owner_;
    torch::Tensor host_tokens_, host_history_, host_hashes_;
    torch::Tensor history_, hashes_;
    torch::Tensor host_weight_rows_, host_scale_rows_;
    std::vector<int32_t> lengths_;
  };
  std::unique_ptr<HashState> create_hash_state(int64_t rows,
                                               int64_t query_width = 1) const;
  DeepseekV41EngramImpl(const ModelContext& context, int32_t engram_id);
  void load_state_dict(const StateDict& state);
  void load_hash_state(const StateDict& state);
  void verify_loaded_weights() const;
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& tokens,
                        const ModelInputParams& input,
                        HashState* hash_state = nullptr,
                        const torch::Tensor& row_indices = {});

 private:
  void compute_hashes(const int64_t* tokens,
                      const int64_t* history,
                      const std::vector<int32_t>& lengths,
                      int64_t* output) const;
  void lookup_host_rows(const torch::Tensor& hashes,
                        torch::Tensor& weight_rows,
                        torch::Tensor& scale_rows) const;
  ModelArgs args_;
  int32_t engram_id_;
  ProcessGroup* lookup_group_;
  ProcessGroup* world_group_;
  torch::TensorOptions options_;
  int64_t table_start_;
  int64_t table_end_;
  bool cpu_offload_;
  bool node_shared_memory_ = false;
  std::unordered_map<std::string, torch::Tensor> weights_;
  torch::Tensor token_map_, multipliers_, primes_, offsets_, fp8_values_;
};
TORCH_MODULE(DeepseekV41Engram);
}  // namespace xllm::layer
