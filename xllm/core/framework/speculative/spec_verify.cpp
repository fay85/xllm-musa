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

#include "core/framework/speculative/spec_verify.h"

#include <glog/logging.h>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/runtime/forward_params.h"

namespace xllm {

SampleOutput run_rejection_sampling(const SamplingParameters& sampling_params,
                                    const torch::Tensor& draft_token_ids,
                                    const torch::Tensor& draft_logits,
                                    const torch::Tensor& target_logits,
                                    const torch::Tensor& target_hidden_states) {
  SampleOutput sample_output = rejection_sample(
      sampling_params, draft_token_ids, draft_logits, target_logits);

  if (target_hidden_states.defined()) {
    const torch::Tensor& next_tokens = sample_output.next_tokens;
    sample_output.embeddings =
        target_hidden_states.view({next_tokens.size(/*dim=*/0),
                                   next_tokens.size(/*dim=*/1),
                                   target_hidden_states.size(/*dim=*/-1)});
  }
  return sample_output;
}

namespace spec_verify {

void expand_verify_sampling_params(SamplingParameters& sampling_params,
                                   const int32_t num_val_tokens,
                                   const int32_t total_num_val_tokens,
                                   const torch::Device& device,
                                   bool repeat_unique_history) {
  // Build indices on-device to avoid a CPU construction and host-to-device
  // copy on every validation step.
  sampling_params.selected_token_idxes =
      torch::arange(total_num_val_tokens,
                    torch::dtype(/*dtype=*/torch::kLong).device(device));
  sampling_params.sample_idxes = sampling_params.selected_token_idxes;

  const auto repeat = [num_val_tokens](torch::Tensor& tensor) {
    if (tensor.defined()) {
      tensor = tensor.repeat_interleave(num_val_tokens, /*dim=*/0);
    }
  };
  repeat(sampling_params.frequency_penalties);
  repeat(sampling_params.presence_penalties);
  repeat(sampling_params.repetition_penalties);
  repeat(sampling_params.temperatures);
  repeat(sampling_params.top_p);
  repeat(sampling_params.top_k);
  if (repeat_unique_history) {
    repeat(sampling_params.unique_token_ids);
    repeat(sampling_params.unique_token_counts);
    repeat(sampling_params.unique_token_ids_lens);
  }
  repeat(sampling_params.filter_mask);
  repeat(sampling_params.filter_bitmask);
}

}  // namespace spec_verify

}  // namespace xllm
