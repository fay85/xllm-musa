/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "rejection_sampler.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/stack.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>

#include "kernels/ops_api.h"
#include "sampler.h"

#if defined(USE_MUSA)
namespace xllm::kernel::musa {
torch::Tensor lookup_device_compact_vocab_column_ids(int32_t device_index,
                                                     int64_t compact_cols,
                                                     int64_t full_vocab_size);
}  // namespace xllm::kernel::musa
#endif

namespace xllm {

namespace {
// index_select that supports multiple dimensions index
torch::Tensor index_select_2d(const torch::Tensor& input,
                              int64_t dim,
                              const torch::Tensor& index) {
  return input.gather(dim, index.unsqueeze(dim)).squeeze(dim);
}

torch::Tensor normalize_recovered_probs(torch::Tensor recovered_probs) {
  const auto sums = recovered_probs.sum(/*dim=*/-1, /*keepdim=*/true);
  // A zero residual has no valid normalization; fall back to uniform.
  const double uniform_prob =
      1.0 / static_cast<double>(recovered_probs.size(-1));
  recovered_probs.div_(sums.clamp_min(1e-6));
  recovered_probs.masked_fill_(sums <= 0, uniform_prob);
  return recovered_probs;
}

bool is_packed_sparse_draft_probs(const torch::Tensor& draft_probs,
                                  int64_t vocab_size) {
  return draft_probs.defined() && draft_probs.dim() == 3 &&
         draft_probs.size(-1) >= 3 && draft_probs.size(-1) <= 2049 &&
         draft_probs.size(-1) < vocab_size &&
         ((draft_probs.size(-1) - 1) % 2 == 0);
}

torch::Tensor lookup_compact_draft_column_ids(const torch::Tensor& draft_probs,
                                              int64_t vocab_size) {
#if defined(USE_MUSA)
  if (draft_probs.defined() && !draft_probs.is_cpu() &&
      draft_probs.dim() == 3 && draft_probs.size(-1) > 2049 &&
      draft_probs.size(-1) < vocab_size) {
    return xllm::kernel::musa::lookup_device_compact_vocab_column_ids(
        draft_probs.device().index(), draft_probs.size(-1) - 1, vocab_size);
  }
#else
  (void)draft_probs;
  (void)vocab_size;
#endif
  return torch::Tensor();
}

torch::Tensor sample_recovered_tokens(const DraftProposal& draft_proposal,
                                      const torch::Tensor& target_probs,
                                      bool allow_inplace_target = false) {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const std::optional<torch::Tensor>& draft_probs =
      draft_proposal.draft_probs();
  const int64_t vocab_size = target_probs.size(-1);
  torch::Tensor recovered_probs;
  if (draft_probs.has_value()) {
    torch::Tensor compact_col_ids =
        lookup_compact_draft_column_ids(draft_probs.value(), vocab_size);
    if (compact_col_ids.defined()) {
      const torch::Tensor& packed = draft_probs.value();
      const int64_t compact_cols = compact_col_ids.size(0);
      torch::Tensor compact_probs =
          packed.narrow(/*dim=*/-1, /*start=*/1, /*length=*/compact_cols)
              .reshape({-1, compact_cols});
      recovered_probs =
          allow_inplace_target ? target_probs : target_probs.clone();
      torch::Tensor flat_rec = recovered_probs.reshape({-1, vocab_size});
      torch::Tensor updated =
          flat_rec.index_select(/*dim=*/1, compact_col_ids)
              .sub_(compact_probs)
              .clamp_min_(0.0f);
      flat_rec.index_copy_(/*dim=*/1, compact_col_ids, updated);
#if defined(USE_MUSA)
      return Sampler::random_sample(recovered_probs);
#else
      return Sampler::random_sample(normalize_recovered_probs(recovered_probs));
#endif
    }
  }
  if (draft_probs.has_value() &&
      is_packed_sparse_draft_probs(draft_probs.value(), vocab_size)) {
    const torch::Tensor& packed = draft_probs.value();
    const int64_t top_k = (packed.size(-1) - 1) / 2;
    torch::Tensor candidate_ids =
        packed.narrow(/*dim=*/-1, /*start=*/1, /*length=*/top_k)
            .to(torch::kLong);
    torch::Tensor candidate_probs =
        packed.narrow(/*dim=*/-1, /*start=*/1 + top_k, /*length=*/top_k);
    recovered_probs = allow_inplace_target ? target_probs : target_probs.clone();
    torch::Tensor updated =
        recovered_probs.gather(/*dim=*/-1, candidate_ids)
            .sub_(candidate_probs)
            .clamp_min_(0.0f);
    recovered_probs.scatter_(/*dim=*/-1, candidate_ids, updated);
#if defined(USE_MUSA)
    return Sampler::random_sample(recovered_probs);
#else
    return Sampler::random_sample(normalize_recovered_probs(recovered_probs));
#endif
  }
  if (draft_probs.has_value()) {
    if (allow_inplace_target) {
      recovered_probs = target_probs;
      recovered_probs.sub_(draft_probs.value()).clamp_min_(0);
    } else {
      recovered_probs = target_probs.sub(draft_probs.value()).clamp_min_(0);
    }
  } else {
    recovered_probs = allow_inplace_target ? target_probs : target_probs.clone();
    recovered_probs.scatter_(
        /*dim=*/-1, draft_token_ids.unsqueeze(-1), /*value=*/0);
  }
#if defined(USE_MUSA)
  return Sampler::random_sample(recovered_probs);
#else
  return Sampler::random_sample(normalize_recovered_probs(recovered_probs));
#endif
}

torch::Tensor mask_out_after_first_reject(
    const torch::Tensor& accepted_bool,
    const torch::Tensor& accepted_token_ids) {
  const auto accepted_mask =
      RejectionSampler::build_accepted_mask(accepted_bool);
  return torch::where(accepted_mask,
                      accepted_token_ids,
                      torch::full_like(accepted_token_ids, -1));
}

std::tuple<torch::Tensor, torch::Tensor> finalize_random_sample(
    const torch::Tensor& accepted,
    const torch::Tensor& recovered_token_ids,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto combined = torch::where(accepted, draft_token_ids, recovered_token_ids);
  auto accepted_token_ids = torch::cat({combined, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    masked_accepted_token_ids =
        mask_out_after_first_reject(accepted, accepted_token_ids);
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

std::tuple<torch::Tensor, torch::Tensor> random_sample_with_ownership(
    const DraftProposal& draft_proposal,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens,
    bool allow_inplace_target) {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const std::optional<torch::Tensor>& draft_probs =
      draft_proposal.draft_probs();
  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  const int64_t vocab_size = target_probs.size(-1);
  torch::Tensor acceptance_probs;
  if (draft_probs.has_value()) {
    const bool has_leading_selected_prob =
        is_packed_sparse_draft_probs(draft_probs.value(), vocab_size) ||
        lookup_compact_draft_column_ids(draft_probs.value(), vocab_size)
            .defined();
    torch::Tensor selected_draft_probs =
        has_leading_selected_prob
            ? draft_probs->select(/*dim=*/-1, /*index=*/0)
            : index_select_2d(draft_probs.value(), /*dim=*/-1, draft_token_ids);
    acceptance_probs =
        selected_target_probs / selected_draft_probs.clamp_min(1e-12f);
  } else {
    acceptance_probs = selected_target_probs;
  }
  auto accepted = (uniform_rand < acceptance_probs);

  auto recovered_token_ids = sample_recovered_tokens(
      draft_proposal, target_probs, allow_inplace_target);

  return finalize_random_sample(accepted,
                                recovered_token_ids,
                                draft_token_ids,
                                bonus_token_ids,
                                mask_out_rejected_tokens);
}

}  // namespace

RejectionSampler::RejectionSampler(const torch::Tensor& do_sample,
                                   bool all_random_sample,
                                   bool all_greedy_sample,
                                   bool logprobs,
                                   int64_t max_top_logprobs,
                                   bool enable_fused_kernel)
    : logprobs_(logprobs),
      max_top_logprobs_(max_top_logprobs),
      all_random_sample_(all_random_sample),
      all_greedy_sample_(all_greedy_sample),
      enable_fused_kernel_(enable_fused_kernel) {
  CHECK(do_sample.defined());
  // Keep a private expanded view and do not mutate the caller-owned tensor.
  // The same SamplingParameters object is reused later by MTP draft extend.
  // An in-place unsqueeze here corrupts Sampler::forward() mixed-mode shape
  // assumptions and can broadcast sampled token ids into 2D.
  do_sample_ = do_sample.unsqueeze(/*dim=*/-1);
}

SampleOutput RejectionSampler::forward(const DraftProposal& draft_proposal,
                                       const torch::Tensor& target_logits,
                                       const torch::Tensor& bonus_token_ids,
                                       bool mask_out_rejected_tokens) const {
  const int64_t expected_batch_size = do_sample_.size(0);
  const int64_t expected_vocab_size = target_logits.size(-1);
  const int64_t expected_n_spec = target_logits.size(1) - 1;
  const bool has_sparse_draft_probs =
      draft_proposal.draft_probs().has_value() &&
      (is_packed_sparse_draft_probs(*draft_proposal.draft_probs(),
                                    expected_vocab_size) ||
       lookup_compact_draft_column_ids(*draft_proposal.draft_probs(),
                                       expected_vocab_size)
           .defined());
  if (has_sparse_draft_probs) {
    const torch::Tensor& ids = draft_proposal.token_ids();
    const torch::Tensor& sp = *draft_proposal.draft_probs();
    CHECK(ids.defined() && ids.dim() == 2 &&
          ids.size(0) == expected_batch_size && ids.size(1) == expected_n_spec);
    CHECK(sp.size(0) == expected_batch_size && sp.size(1) == expected_n_spec);
  } else {
    draft_proposal.validate(
        /*expected_batch_size=*/expected_batch_size,
        /*expected_vocab_size=*/expected_vocab_size,
        /*expected_num_speculative_tokens=*/expected_n_spec);
  }
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();

  // The bonus row is sampled by the target worker and does not participate in
  // random rejection sampling, so drop it before scoring draft positions.
  const torch::Tensor target_draft_logits = target_logits.slice(
      /*dim=*/1, /*start=*/0, /*end=*/target_logits.size(1) - 1);

  torch::Tensor target_probs;
  if (!all_greedy_sample_) {
    target_probs = torch::softmax(
        target_draft_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  }

  // The fused kernel needs dense draft_probs; greedy/sparse proposals skip it.
  bool use_fused_kernel = draft_proposal.draft_probs().has_value() &&
                          !has_sparse_draft_probs && enable_fused_kernel_ &&
                          (!logprobs_ && mask_out_rejected_tokens);

  torch::Tensor accepted_token_ids;
  torch::Tensor masked_accepted_token_ids;

  auto random_sample_dispatch =
      [&]() -> std::tuple<torch::Tensor, torch::Tensor> {
    const torch::Tensor uniform_rand =
        torch::rand(draft_token_ids.sizes(), target_probs.options());
    if (use_fused_kernel) {
      return random_sample_fused(draft_token_ids,
                                 draft_proposal.draft_probs().value(),
                                 target_probs,
                                 uniform_rand,
                                 bonus_token_ids,
                                 mask_out_rejected_tokens);
    }
    return random_sample_with_ownership(draft_proposal,
                                        target_probs,
                                        uniform_rand,
                                        bonus_token_ids,
                                        mask_out_rejected_tokens,
                                        /*allow_inplace_target=*/all_random_sample_);
  };

  if (all_greedy_sample_) {
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        greedy_sample(draft_token_ids,
                      target_draft_logits,
                      bonus_token_ids,
                      mask_out_rejected_tokens);
  } else if (all_random_sample_) {
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        random_sample_dispatch();
  } else {
    // mixed sample, sample both then choose based on do_sample_
    auto [random, masked_random] = random_sample_dispatch();
    auto [greedy, masked_greedy] = greedy_sample(draft_token_ids,
                                                 target_probs,
                                                 bonus_token_ids,
                                                 mask_out_rejected_tokens);
    accepted_token_ids = torch::where(do_sample_, random, greedy);
    if (mask_out_rejected_tokens) {
      masked_accepted_token_ids =
          torch::where(do_sample_, masked_random, masked_greedy);
    }
  }

  SampleOutput output;
  output.next_tokens =
      mask_out_rejected_tokens ? masked_accepted_token_ids : accepted_token_ids;

  if (logprobs_) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    // [batch_size, n_speculative_tokens + 1, vocab_size]
    auto target_logprobs = torch::log_softmax(
        target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // select the logprobs for each sequence
    const auto selected_logprobs =
        index_select_2d(target_logprobs, /*dim=*/-1, accepted_token_ids);
    // output.probs = selected_probs;
    output.logprobs = selected_logprobs;

    if (max_top_logprobs_ > 0) {
      auto [values, indices] =
          target_logprobs.topk(max_top_logprobs_, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

// build mask from accepted matrix
// for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
//               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
torch::Tensor RejectionSampler::build_accepted_mask(
    const torch::Tensor& accepted) {
  // build the mask for the first rejected token
  const auto batch_size = accepted.size(0);
  const auto n_tokens = accepted.size(1);

  // use LongTensor since argmax does not support bool
  auto accepted_int64 = accepted.to(torch::kInt64);
  auto bonus_mask = torch::zeros({batch_size, 1}, accepted_int64.options());
  auto combined_mask = torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1);
  // [batch_size, 1]
  auto first_rejected_mask =
      (1 - combined_mask).argmax(/*dim=*/1, /*keepdim=*/true);

  // [1, n_speculative_tokens + 1]
  auto indices =
      torch::arange(n_tokens + 1, accepted.device()).unsqueeze(/*dim=*/0);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_mask = indices <= first_rejected_mask;
  return accepted_mask;
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::random_sample(
    const DraftProposal& draft_proposal,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  return random_sample_with_ownership(draft_proposal,
                                      target_probs,
                                      uniform_rand,
                                      bonus_token_ids,
                                      mask_out_rejected_tokens,
                                      /*allow_inplace_target=*/false);
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::random_sample_fused(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  CHECK_EQ(draft_probs.dim(), 3)
      << "Fused rejection sampler requires dense draft_probs [batch, n_spec, "
         "vocab].";

  const auto device = draft_token_ids.device();
  const int64_t batch_size = draft_token_ids.size(0);
  const int64_t n_spec = draft_token_ids.size(1);
  const int64_t vocab_size = target_probs.size(2);

  // Strictly check device consistency for bonus_token_ids and draft_token_ids
  CHECK_EQ(bonus_token_ids.device().type(), device.type())
      << "bonus_token_ids must be on the same device as draft_token_ids";

  // Check that bonus_token_ids has at least batch_size elements
  CHECK_GE(bonus_token_ids.numel(), batch_size)
      << "bonus_token_ids numel (" << bonus_token_ids.numel()
      << ") is smaller than batch_size (" << batch_size << ")";

  // Prepare input Tensors and ensure they are contiguous where needed
  // If draft_token_ids is already int32 and contiguous, no copy occurs
  torch::Tensor draft_token_ids_int32 =
      draft_token_ids.reshape({-1}).to(torch::kInt32).contiguous();
  torch::Tensor bonus_token_ids_int32 =
      bonus_token_ids.reshape({-1}).to(torch::kInt32).contiguous();

  // Ensure large probability matrices are in the correct shape and contiguous
  torch::Tensor draft_probs_flat =
      draft_probs.reshape({-1, vocab_size}).contiguous();
  torch::Tensor target_probs_flat =
      target_probs.reshape({-1, vocab_size}).contiguous();
  torch::Tensor uniform_rand_flat =
      uniform_rand.to(torch::kFloat32).flatten().contiguous();

  // Create auxiliary tensors directly on the target device to avoid unnecessary
  // copies
  torch::TensorOptions options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor num_draft_tokens =
      torch::full({batch_size}, n_spec, options_int32);
  torch::Tensor cu_num_draft_tokens =
      torch::arange(n_spec, (batch_size + 1) * n_spec, n_spec, options_int32);

  // Always create recovery probability matrix here, as kernel requires it
  torch::Tensor uniform_probs =
      torch::empty({batch_size * n_spec, vocab_size},
                   target_probs.options().dtype(torch::kFloat32))
          .exponential_();

  // Call the fused kernel
  kernel::RejectionSampleParams params;
  params.draft_token_ids = draft_token_ids_int32;
  params.num_draft_tokens = num_draft_tokens;
  params.cu_num_draft_tokens = cu_num_draft_tokens;
  params.draft_probs = draft_probs_flat;
  params.target_probs = target_probs_flat;
  params.bonus_token_ids = bonus_token_ids_int32;
  params.uniform_rand = uniform_rand_flat;
  params.uniform_probs = uniform_probs;
  params.max_spec_len = n_spec;

  // The result is flattened, and positions of rejected tokens are set to -1
  torch::Tensor output_token_ids = kernel::rejection_sample(params);

  // Reshape result to [batch, n_spec + 1]
  torch::Tensor masked_result =
      output_token_ids.reshape({batch_size, n_spec + 1}).to(torch::kInt64);

  // When mask_out_rejected_tokens=true and logprobs_=false,
  // we can safely return masked_result for both outputs.
  return {masked_result, masked_result};
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::greedy_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_scores,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  torch::Tensor target_token_ids = Sampler::greedy_sample(target_scores);
  return greedy_sample_from_token_ids(draft_token_ids,
                                      target_token_ids,
                                      bonus_token_ids,
                                      mask_out_rejected_tokens);
}

std::tuple<torch::Tensor, torch::Tensor>
RejectionSampler::greedy_sample_from_token_ids(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_token_ids,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  CHECK_EQ(target_token_ids.sizes(), draft_token_ids.sizes())
      << "target and draft token shapes must match";
  // [batch_size, n_speculative_tokens + 1]
  torch::Tensor accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    auto accepted = (target_token_ids == draft_token_ids);
    masked_accepted_token_ids =
        mask_out_after_first_reject(accepted, accepted_token_ids);
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

}  // namespace xllm
