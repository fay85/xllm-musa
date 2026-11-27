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

#include <limits>

#include "core/kernels/mlu/dflash2_sampling.h"
#include "core/runtime/dflash2_worker_impl.h"

namespace xllm {

std::optional<std::pair<torch::Tensor, torch::Tensor>>
DFlash2WorkerImpl::fused_sample(const torch::Tensor& candidate_ids,
                                const torch::Tensor& edge_log_probs,
                                const torch::Tensor& gumbel_noise,
                                const SamplingParameters& sampling_params,
                                int64_t vocab_size) {
  if (!kernel::mlu::supports_dflash2_sampling(
          candidate_ids,
          edge_log_probs,
          gumbel_noise,
          sampling_params.do_sample,
          sampling_params.sampling_mode == SamplingMode::RANDOM,
          sampling_params.sampling_mode == SamplingMode::GREEDY)) {
    return std::nullopt;
  }
  auto [draft_token_ids, candidate_logits] = kernel::mlu::sample_dflash2_paths(
      candidate_ids,
      edge_log_probs,
      gumbel_noise,
      sampling_params.do_sample,
      sampling_params.sampling_mode == SamplingMode::RANDOM);
  torch::Tensor draft_logits =
      torch::full({candidate_ids.size(/*dim=*/0),
                   candidate_ids.size(/*dim=*/1),
                   vocab_size},
                  -std::numeric_limits<float>::infinity(),
                  edge_log_probs.options());
  draft_logits.scatter_(/*dim=*/-1, candidate_ids, candidate_logits);
  return std::make_pair(std::move(draft_token_ids), std::move(draft_logits));
}

}  // namespace xllm
