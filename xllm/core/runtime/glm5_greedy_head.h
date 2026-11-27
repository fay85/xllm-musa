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

#include <limits>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/parallel_state/musa_process_group.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"
#include "core/kernels/musa/sharded_argmax.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/options.h"
#include "core/util/env_var.h"

namespace xllm {
inline bool is_glm5_width_audit(const ModelInputParams& params) {
  const auto& ids = params.embedding.request_ids;
  return ids.size() == 1 && ids.front().starts_with("width-audit-");
}
inline torch::Tensor glm5_greedy_tokens(CausalLM& model,
                                        const runtime::Options& options,
                                        const std::string& model_type,
                                        const ForwardInput& input,
                                        const torch::Tensor& hidden,
                                        const torch::Tensor& indices,
                                        torch::Tensor& selected_hidden) {
  static const bool enabled =
      util::get_bool_env("XLLM_GLM5_SHARDED_GREEDY", true);
  const auto& params = input.sampling_params;
  const bool glm_mtp =
      model_type == "glm5_next_mtp" ||
      (model_type == "glm5_next" && input.input_params.is_spec_verify);
  if (!enabled || !glm_mtp || indices.numel() == 0 ||
      options.instance_role() != InstanceRole::DECODE ||
      options.cp_size() != 1 ||
      (options.speculative_algorithm() != "mtp" &&
       options.speculative_algorithm() != "MTP") ||
      options.enable_adaptive_speculative_decode() ||
      input.skip_sampling_for_logits_only ||
      !input.json_object_states.empty() || !params.all_greedy_sample ||
      params.logprobs || params.return_probs || params.use_beam_search ||
      params.filter_mask.defined() || params.filter_bitmask.defined() ||
      params.repetition_penalties.defined() ||
      params.frequency_penalties.defined() ||
      params.presence_penalties.defined() || !params.sample_idxes.defined() ||
      params.sample_idxes.numel() != indices.numel()) {
    return {};
  }
  auto head = model.get_lm_head();
  const torch::Tensor weight = head->weight();
  ProcessGroup* group = head->process_group();
  if (!weight.defined() || weight.dim() != 2 || group == nullptr) {
    return {};
  }
  const int64_t cols = weight.size(0);
  if (group->world_size() <= 1 || weight.scalar_type() != torch::kBFloat16 ||
      hidden.scalar_type() != torch::kBFloat16 || cols < 16384 ||
      cols % 8 != 0 ||
      cols * group->world_size() > std::numeric_limits<int32_t>::max()) {
    return {};
  }
  selected_hidden = hidden.index_select(/*dim=*/0, indices);
  const torch::Tensor local_scores =
      torch::empty({selected_hidden.size(0), cols}, selected_hidden.options());
  const torch::Tensor local_logits =
      kernel::musa::matmul(selected_hidden, weight, head->bias(), local_scores);
  auto pairs = kernel::musa::shard_argmax_candidates(
      local_logits, static_cast<int32_t>(cols * group->rank()));
  const bool eager_draft = model_type == "glm5_next_mtp" &&
                           !ExecutionConfig::get_instance().enable_graph();
  torch::Tensor gathered;
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  auto* musa_group = dynamic_cast<ProcessGroupImpl*>(group);
  if (musa_group != nullptr && eager_draft) {
    gathered = musa_group->gather_rows_on_current_stream(pairs);
  } else
#endif
  {
    gathered = group->allgather_base_sync(pairs);
  }
  const torch::Tensor tokens = kernel::musa::merge_shard_argmax(
      gathered,
      local_logits.size(0),
      eager_draft && input.token_ids.scalar_type() == torch::kInt
          ? torch::kInt
          : torch::kLong);
  static const bool validate =
      util::get_bool_env("XLLM_GLM5_VALIDATE_SHARDED_GREEDY", false);
  if (validate && is_glm5_width_audit(input.input_params) &&
      !kernel::musa::is_stream_capturing()) {
    const torch::Tensor full_logits = model.logits(hidden, indices);
    CHECK(torch::equal(tokens, full_logits.argmax(/*dim=*/-1)))
        << "GLM5 sharded greedy token mismatch";
    LOG_FIRST_N(INFO, 8) << "Validated GLM5 sharded greedy tokens, model="
                         << model_type << " rows=" << tokens.numel();
  }
  return tokens;
}
}  // namespace xllm
