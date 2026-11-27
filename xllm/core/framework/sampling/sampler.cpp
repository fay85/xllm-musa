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

#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>

#include "common/global_flags.h"
#include "core/framework/config/model_config.h"
#include "core/framework/sampling/logits_processor.h"
#include "sampling_params.h"

namespace xllm {

torch::Tensor gumbel_noise(const SamplingParameters& params,
                           torch::IntArrayRef sizes,
                           const torch::Device& device) {
  const auto options =
      torch::TensorOptions().dtype(/*dtype=*/torch::kFloat32).device(device);
  if (params.sampling_mode == SamplingMode::GREEDY) {
    return torch::zeros(sizes, options);
  }
  auto noise = torch::empty(sizes, options);
  noise.exponential_(/*lambd=*/1.0)
      .clamp_min_(std::numeric_limits<float>::min())
      .log_();
  if (params.sampling_mode == SamplingMode::MIXED) {
    c10::SmallVector<int64_t, 4> shape(sizes.size(), 1);
    shape.front() = sizes.front();
    noise.masked_fill_(params.do_sample.to(device).logical_not().view(shape),
                       /*value=*/0.0);
  }
  return noise;
}

torch::Tensor gumbel_argmax(const torch::Tensor& logits,
                            torch::Tensor gumbel_noise) {
  torch::sub_out(gumbel_noise, logits, gumbel_noise);
  return gumbel_noise.argmax(/*dim=*/-1);
}

torch::Tensor gumbel_sample(const torch::Tensor& logits) {
  auto noise = torch::empty_like(
      logits, logits.options().dtype(/*dtype=*/torch::kFloat32));
  noise.exponential_(/*lambd=*/1.0)
      .clamp_min_(std::numeric_limits<float>::min())
      .log_();
  return gumbel_argmax(logits, std::move(noise));
}

void publish_logits(torch::Tensor& logits,
                    const torch::Tensor& sample_logits,
                    const SamplingParameters& params) {
  if (params.selected_token_idxes.numel() == params.sample_idxes.numel()) {
    logits = sample_logits;
    return;
  }
  logits = logits.to(sample_logits.scalar_type());
  logits.index_copy_(
      /*dim=*/0, params.sample_idxes.to(/*dtype=*/torch::kLong), sample_logits);
}

SampleOutput sample(torch::Tensor& logits,
                    const SamplingParameters& params,
                    const torch::Tensor& filter_mask) {
  SampleOutput output;
  torch::Tensor queried_logprobs;
  torch::Tensor query_ids;
  torch::Tensor query_valid;
  torch::Tensor query_rows;
  if (params.logprob_token_ids.defined()) {
    query_valid = params.logprob_token_ids.ge(0);
    query_rows = query_valid.any(/*dim=*/1).nonzero().view({-1});
    if (query_rows.numel() > 0) {
      query_ids = params.logprob_token_ids.index_select(/*dim=*/0, query_rows);
      query_valid = query_valid.index_select(/*dim=*/0, query_rows);
      const torch::Tensor query_sample_idxes =
          params.sample_idxes.index_select(/*dim=*/0, query_rows);
      const torch::Tensor original_rows =
          logits.index_select(/*dim=*/0, query_sample_idxes);
      queried_logprobs = torch::log_softmax(
          original_rows, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    }
  }
  torch::Tensor sample_logits = process_logits(logits, params, filter_mask);

  torch::Tensor sample_temperatures = params.temperatures;
  torch::Tensor sample_top_k = params.top_k;
  torch::Tensor sample_top_p = params.top_p;
  const bool use_sample_indices =
      params.selected_token_idxes.numel() != params.sample_idxes.numel();
  if (use_sample_indices) {
    if (params.temperatures.defined()) {
      sample_temperatures =
          params.temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_k.defined()) {
      sample_top_k = params.top_k.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_p.defined()) {
      sample_top_p = params.top_p.index_select(/*dim=*/0, params.sample_idxes);
    }
  }

  if (params.sampling_mode == SamplingMode::GREEDY && !params.logprobs &&
      !use_sample_indices && !filter_mask.defined()) {
    output.next_tokens = greedy_sample(sample_logits).to(torch::kLong);
    publish_logits(logits, sample_logits, params);
    return output;
  }

  // apply temperatures, top-k and top-p
  apply_logits_transforms(
      sample_logits, sample_temperatures, sample_top_k, sample_top_p);
  publish_logits(logits, sample_logits, params);

  torch::Tensor sample_indices;
  if (params.sampling_mode == SamplingMode::GREEDY) {
    sample_indices = greedy_sample(sample_logits);
  } else {
    sample_indices = params.sampling_mode == SamplingMode::RANDOM
                         ? gumbel_sample(sample_logits)
                         : gumbel_argmax(sample_logits,
                                         gumbel_noise(params,
                                                      sample_logits.sizes(),
                                                      sample_logits.device()));
  }
  output.next_tokens = sample_indices;

  if (params.logprobs) {
    if (::xllm::ModelConfig::get_instance().enable_qwen3_reranker()) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, /*options=*/torch::kLong)
              .to(sample_indices.device());
      sample_logits = sample_logits.index_select(/*dim=*/1, indices);
      auto logprobs = torch::log_softmax(
          sample_logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs =
        logprobs.gather(/*dim=*/-1, sample_indices.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
      // max_top_logprobs is the batch-wide max, so one request asking for more
      // than the vocabulary would make topk throw and fail every request in the
      // batch. The factories reject such requests up front
      // (RequestSamplingParam::top_logprobs_vocab_error); clamp here as the
      // last line of defense so the sampler can never be the failure point.
      // Every consumer sizes by the returned row width, so a shorter row is
      // safe.
      const int64_t k =
          std::min<int64_t>(params.max_top_logprobs, logprobs.size(-1));
      auto [values, indices] = logprobs.topk(k, /*dim=*/-1);
      if (params.logprob_token_ids.defined()) {
        CHECK_LE(params.logprob_token_ids.size(1), k);
      }
      if (queried_logprobs.defined()) {
        torch::Tensor requested = queried_logprobs.gather(
            /*dim=*/1, query_ids.clamp_min(0).to(torch::kLong));
        torch::Tensor query_values =
            torch::full({query_rows.numel(), k},
                        -std::numeric_limits<float>::infinity(),
                        values.options());
        torch::Tensor query_tokens =
            torch::full({query_rows.numel(), k}, -1, indices.options());
        query_values.slice(/*dim=*/1, /*start=*/0, /*end=*/query_ids.size(1))
            .copy_(
                requested.masked_fill(query_valid.logical_not(),
                                      -std::numeric_limits<float>::infinity()));
        query_tokens.slice(/*dim=*/1, /*start=*/0, /*end=*/query_ids.size(1))
            .copy_(query_ids.to(torch::kLong));
        values.index_copy_(/*dim=*/0, query_rows, query_values);
        indices.index_copy_(/*dim=*/0, query_rows, query_tokens);
      }
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor greedy_sample(const torch::Tensor& logits) {
  return logits.argmax(/*dim=*/-1);
}

}  // namespace xllm
