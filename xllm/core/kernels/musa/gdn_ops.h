/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "core/kernels/musa/ops_api.h"

namespace xllm {
namespace kernel {

struct CausalConv1dUpdateParams;
struct ChunkGatedDeltaRuleParams;
struct FusedGdnGatingParams;
struct FusedQkvzbaSplitReshapeParams;
struct FusedRecurrentGatedDeltaRuleParams;
struct GatedLayerNormParams;
struct PartialRotaryEmbeddingParams;
struct FusedSigmoidGatingDeltaRuleUpdateParams;

namespace musa {

torch::Tensor l2_norm(torch::Tensor& x, double eps);

std::pair<torch::Tensor, torch::Tensor> l2_norm_pair_fused(
    const torch::Tensor& query,
    const torch::Tensor& key,
    double eps);

// Normalizes Q/K in place. The fused H=128 kernel loads each row before any
// stores, so input/output aliasing is safe.
void l2_norm_pair_fused_inplace(torch::Tensor& query,
                                torch::Tensor& key,
                                double eps);

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params);

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params);

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params);

torch::Tensor gated_layer_norm(GatedLayerNormParams& params);

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params);

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params);

torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk);

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params);

torch::Tensor mate_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

torch::Tensor fused_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

void causal_conv1d_fwd(const torch::Tensor& x,
                       const torch::Tensor& weight,
                       torch::Tensor& out,
                       const std::optional<torch::Tensor>& bias,
                       const std::optional<torch::Tensor>& conv_states,
                       const std::optional<torch::Tensor>& query_start_loc,
                       const std::optional<torch::Tensor>& cache_indices,
                       const std::optional<torch::Tensor>& has_initial_state,
                       bool silu_activation,
                       int64_t pad_slot_id);

void causal_conv1d_fwd_token_major(const torch::Tensor& x,
                                   const torch::Tensor& weight,
                                   torch::Tensor& out,
                                   const std::optional<torch::Tensor>& bias,
                                   const torch::Tensor& conv_states,
                                   const torch::Tensor& query_start_loc,
                                   const torch::Tensor& cache_indices,
                                   const torch::Tensor& has_initial_state,
                                   bool silu_activation,
                                   int64_t pad_slot_id);

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode);

torch::Tensor causal_conv1d_prefill(const torch::Tensor& x,
                                    const torch::Tensor& weight,
                                    const torch::Tensor& conv_state,
                                    const std::optional<torch::Tensor>& bias,
                                    const torch::Tensor& query_start_loc,
                                    const torch::Tensor& cache_indices,
                                    const torch::Tensor& has_initial_state,
                                    bool silu_activation);

torch::Tensor fused_sigmoid_gating_delta_rule_update(
    FusedSigmoidGatingDeltaRuleUpdateParams& params);

}  // namespace musa
}  // namespace kernel
}  // namespace xllm
