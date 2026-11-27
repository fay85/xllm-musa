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

#include <optional>

#include "core/common/types.h"

namespace xllm {

// Instance-role predicates shared by every options struct that carries
// `enable_disagg_pd()` and `instance_role()` (xllm::Options,
// runtime::Options and ContinuousScheduler::Options). They exist because a bare
// `instance_role() == InstanceRole::PREFILL` comparison does not mean "this
// instance only runs prefill": the role flag and the deployment-shape flags are
// independent (`--instance_role` and `--enable_disagg_pd` are separate flags in
// disagg_pd_config.cpp, and instance_role is deliberately not read from the
// json config), so the role alone says nothing about which batches the
// scheduler will hand to this instance.
//
// `instance_role() == InstanceRole::PREFILL` works for both accessor shapes:
// std::optional's heterogeneous comparison yields false for an unset role,
// which is the wanted answer.

// True for the PREFILL role of a disaggregated-PD deployment. This is the
// repository-wide condition for "the PD prefill side", and it is what guards
// the prefill-only admissions in Master::validate_options and the exclusive
// batch mode in resolve_batch_mode(). With enable_disagg_pd=false the
// continuous scheduler runs prefill and decode on the same instance whatever
// the role says (PrefillFirstPolicy::schedule falls through to
// schedule_decode_from_queue), so the enable_disagg_pd() term is required.
template <typename OptionsType>
inline bool is_pd_prefill_role(const OptionsType& options) {
  return options.enable_disagg_pd() &&
         options.instance_role() == InstanceRole::PREFILL;
}

}  // namespace xllm
