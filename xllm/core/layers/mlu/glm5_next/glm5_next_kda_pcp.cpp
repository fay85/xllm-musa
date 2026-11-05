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

#include "layers/mlu/glm5_next/glm5_next_kda.h"

namespace xllm::layer {

torch::Tensor Glm5NextKDAImpl::forward_pcp(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const glm5_next_pcp::Context& context) {
  // mHC/DSA retain CP-local rows. KDA consumes complete request sequences
  // with heads sharded over CP x TP, including its persistent conv/SSM state.
  const torch::Tensor global_hidden =
      glm5_next_pcp::gather_restore(hidden_states, context);
  const torch::Tensor global_output =
      forward(global_hidden, attn_metadata, kv_cache, input_params);
  return glm5_next_pcp::shard_rows(global_output, context);
}

}  // namespace xllm::layer
