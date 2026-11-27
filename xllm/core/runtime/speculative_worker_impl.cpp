/* Copyright 2025-2026 The xLLM Authors.

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

#include "speculative_worker_impl.h"

#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/speculative/spec_verify.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

bool should_run_speculative_decode(const ModelInputParams& params) {
  if (!params.meta.batch_forward_type.is_decode()) {
    return false;
  }

  const auto& dp_token_nums = params.parallel.dp_global_token_nums;
  const auto& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_is_decode.empty()) {
    return dp_token_nums.size() <= 1;
  }
  if (dp_is_decode.size() != dp_token_nums.size()) {
    return false;
  }

  // Idle DP ranks (no scheduled tokens this step) must not veto speculative
  // decode for the active ranks. Under enable_graph=False these idle ranks keep
  // dp_is_decode=0 (the backfill in llm_engine only fires when enable_graph=
  // True), which made an all-of-ones check fail for any bs<dp_size batch and
  // silently fell back to the non-speculative path (validate never ran). Only
  // ranks that actually carry tokens gate the decision; require every such rank
  // to be in decode.
  bool any_active = false;
  for (size_t i = 0; i < dp_is_decode.size(); ++i) {
    if (dp_token_nums[i] == 0) {
      continue;  // idle rank: does not participate in the vote
    }
    any_active = true;
    if (dp_is_decode[i] != 1) {
      return false;
    }
  }
  return any_active;
}

void scale_speculative_parallel_token_counts(ModelInputParams& params,
                                             int32_t multiplier) {
  for (int32_t& token_num : params.parallel.dp_global_token_nums) {
    token_num *= multiplier;
  }
  for (int32_t& token_num : params.parallel.raw_dp_global_token_nums) {
    token_num *= multiplier;
  }
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, multiplier);
}

SpeculativeWorkerImpl::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type)
    : WorkerImpl(parallel_args, device, options) {
  if (worker_type == WorkerType::LLM) {
    impl_ =
        std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ =
        std::make_unique<VLMWorkerImpl>(parallel_args, device, target_options);
  } else {
    LOG(FATAL) << "Unsupported speculative worker type: "
               << worker_type.to_string();
  }
}

SpeculativeWorkerImpl::~SpeculativeWorkerImpl() {
  // Draft-model subclasses shut down the shared transfer's threadpool first
  // (see ~DraftModelSpecWorkerImpl); this base only releases refs.
  if (impl_ != nullptr) {
    impl_->clear_hierarchy_kv_cache_transfer();
  }
  clear_hierarchy_kv_cache_transfer();
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed,
                                       MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  CHECK(impl_ != nullptr);
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->init_model_from_path(model_weights_path,
                                         random_seed,
                                         master_status,
                                         [this](ModelArgs& args, ModelLoader&) {
                                           apply_target_model_overrides(args);
                                         });
    if (result) {
      dtype_ = impl_->dtype();
    }
  }
  return result;
}

void SpeculativeWorkerImpl::apply_target_model_overrides(
    [[maybe_unused]] ModelArgs& args) const {
#if !defined(USE_NPU)
  args.num_speculative_tokens(options_.num_speculative_tokens());
#endif
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU) || defined(USE_MUSA)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_shape);
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const ForwardInput& input) {
  ModelInputParams& mutable_params =
      const_cast<ModelInputParams&>(input.input_params);
  // Cannot observe the missing-host-restore-handle throw: the step_async
  // lambda consumed this batch's handle (and cleared requires_host_restore)
  // before dispatching here, or it failed the promise and this step never
  // ran. Kept so a direct step() caller still installs the synchronizer.
  set_hierarchy_layer_synchronizer(mutable_params);
  const bool run_speculative_decode =
      should_run_speculative_decode(input.input_params);
  if (input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0) {
    if (input.input_params.meta.batch_forward_type.is_decode() &&
        !run_speculative_decode) {
      ForwardInput aligned_input = input;
      aligned_input.input_params.meta.batch_forward_type =
          BatchForwardType::EMPTY;
      return step_empty(aligned_input);
    }
    return step_empty(input);
  }

  if (run_speculative_decode) {
    return step_decode(input);
  }
  return step_prefill(input);
}

void SpeculativeWorkerImpl::finalize_verify_output(
    ForwardOutput& target_output,
    SampleOutput sample_output,
    const ForwardInput& input) const {
  const torch::Tensor& tokens = sample_output.next_tokens;
  CHECK(tokens.defined() && tokens.dim() == 2);
  CHECK_EQ(tokens.size(/*dim=*/1), options_.num_speculative_tokens() + 1);
  const auto& states = input.json_object_states;
  CHECK(states.empty() ||
        states.size() == static_cast<size_t>(tokens.size(/*dim=*/0)));
  sample_output.embeddings = torch::Tensor();
  target_output.sample_output = std::move(sample_output);
  target_output.is_speculative_verify = true;
  target_output.speculative_constrained_mask.reserve(states.size());
  for (const auto& state : states) {
    target_output.speculative_constrained_mask.emplace_back(
        static_cast<char>(state.initialized()));
  }
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    bool repeat_unique_history) {
  validate_input = input.to(device_, dtype_);
  validate_input.skip_sampling_for_logits_only = true;
  validate_input.device_tensors_ready = false;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t total_num_val_tokens =
      input.input_params.meta.num_sequences * num_val_tokens;
  // Recurrent targets keep the verification tokens in one query per request.
  DecodeExpansionOptions decode_options;
  decode_options.num_speculative_tokens = options_.num_speculative_tokens();
  decode_options.block_size = block_size();
  decode_options.per_request = input.input_params.is_spec_verify;
  auto rows =
      expand_decode_inputs(make_decode_batch_context(input), decode_options);
  build_decode_input(validate_input, rows, decode_options);
  auto& params = validate_input.input_params;
  if (!decode_options.per_request) {
    params.meta.num_speculative_tokens = options_.num_speculative_tokens();
  }
  params.attention.rebuild_device_buffer(device_);
  spec_verify::expand_verify_sampling_params(validate_input.sampling_params,
                                             num_val_tokens,
                                             total_num_val_tokens,
                                             device_,
                                             repeat_unique_history);
  scale_speculative_parallel_token_counts(params, num_val_tokens);
  validate_input.device_tensors_ready = true;
}

void SpeculativeWorkerImpl::restore_json_object_states(ForwardInput& input) {
  impl_->restore_json_object_states(input);
}
}  // namespace xllm
