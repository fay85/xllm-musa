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

#include "core/framework/sampling/logits_processor.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>

#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif
#include "core/kernels/ops_api.h"

namespace xllm {

void apply_sampling_penalties(torch::Tensor& logits,
                              const SamplingParameters& params) {
  if (params.frequency_penalties.defined()) {
    const auto counts = params.unique_token_counts;
    const auto penalty =
        counts * params.frequency_penalties.unsqueeze(/*dim=*/-1) +
        counts.gt(/*other=*/0) *
            params.presence_penalties.unsqueeze(/*dim=*/-1);
    logits.scatter_add_(/*dim=*/-1,
                        params.unique_token_ids,
                        (-penalty).to(logits.scalar_type()));
  }
  if (params.repetition_penalties.defined()) {
    const auto scores = logits.gather(/*dim=*/-1, params.unique_token_ids);
    const auto factors = params.repetition_penalties.unsqueeze(/*dim=*/-1);
    auto delta =
        torch::where(scores < 0, scores * factors, scores / factors) - scores;
    delta.masked_fill_(~torch::isfinite(scores), /*value=*/0);
    if (params.unique_token_ids_lens.defined()) {
      const auto positions =
          torch::arange(params.unique_token_ids.size(/*dim=*/-1),
                        params.unique_token_ids.options());
      delta.masked_fill_(
          positions >= params.unique_token_ids_lens.unsqueeze(/*dim=*/-1),
          /*value=*/0);
    }
    logits.scatter_add_(
        /*dim=*/-1, params.unique_token_ids, delta.to(logits.scalar_type()));
  }
}

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures) {
  c10::SmallVector<int64_t, 4> shape(logits.dim(), 1);
  shape.front() = temperatures.numel();
  auto unsqueezed_temperatures = temperatures.reshape(shape);
  // Cache device-side scalar to avoid synchronous H2D copy that forces
  // aclrtSynchronizeStream per forward.
  static thread_local torch::Tensor one_scalar;
  const torch::Device& target_device = unsqueezed_temperatures.device();
  if (!one_scalar.defined() || one_scalar.device() != target_device) {
    one_scalar = torch::full(
        {}, /*fill_value=*/1.0, torch::TensorOptions().device(target_device));
  }
  unsqueezed_temperatures = torch::where(
      unsqueezed_temperatures == 0, one_scalar, unsqueezed_temperatures);

  logits.div_(unsqueezed_temperatures);
}

namespace {
void apply_top_k_top_p_torch_impl(torch::Tensor& logits,
                                  const torch::Tensor& top_k,
                                  const torch::Tensor& top_p) {
  auto [sorted_logits, logits_idx] =
      logits.sort(/*dim=*/-1, /*descending=*/true);
  const float filter_value = -std::numeric_limits<float>::infinity();

  if (top_k.defined()) {
    auto processed_top_k = top_k.unsqueeze(/*dim=*/1);
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    static thread_local torch::Tensor max_value_scalar;
    const torch::Device& top_k_device = processed_top_k.device();
    if (!max_value_scalar.defined() ||
        max_value_scalar.device() != top_k_device) {
      max_value_scalar = torch::full(
          {}, kMaxInt64, torch::TensorOptions().device(top_k_device));
    }
    processed_top_k =
        torch::where(processed_top_k <= 0, max_value_scalar, processed_top_k);

    const int64_t vocab_size = logits.size(/*dim=*/-1);
    auto top_k_mask = torch::arange(vocab_size, sorted_logits.device())
                          .expand_as(sorted_logits);
    top_k_mask = top_k_mask >= processed_top_k;
    sorted_logits.masked_fill_(top_k_mask, filter_value);
  }

  if (top_p.defined()) {
    const torch::Tensor processed_top_p = top_p.unsqueeze(/*dim=*/1);
    const torch::Tensor probs =
        sorted_logits.softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
    const torch::Tensor probs_sum = probs.cumsum(/*dim=*/-1);
    const torch::Tensor mask = (probs_sum - probs) > processed_top_p;
    sorted_logits.masked_fill_(mask, filter_value);
  }
  // Speculative verification retains an alias of the target logits before
  // sampling. Preserve that storage so it sees the filtered distribution.
  logits.scatter_(/*dim=*/-1,
                  /*index=*/logits_idx,
                  /*core=*/sorted_logits.to(logits.scalar_type()));
}
}  // namespace

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p) {
  if (!top_k.defined() && !top_p.defined()) {
    return;
  }

#if defined(USE_MLU)
  if (!logits.device().is_cpu()) {
    kernel::TopKPParams params{
        .logits = logits, .top_k = top_k, .top_p = top_p};
    logits = xllm::kernel::apply_top_k_top_p(params);
    return;
  }
#endif

#if defined(USE_NPU)
  if (!logits.device().is_cpu() && top_k.defined() && top_p.defined()) {
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    static thread_local torch::Tensor max_value_scalar;
    const torch::Device& top_k_device = top_k.device();
    if (!max_value_scalar.defined() ||
        max_value_scalar.device() != top_k_device) {
      max_value_scalar = torch::full(
          {}, kMaxInt64, torch::TensorOptions().device(top_k_device));
    }

    auto processed_top_k = torch::where(top_k <= 0, max_value_scalar, top_k)
                               .to(/*dtype=*/torch::kInt32);
    kernel::TopKPParams params{
        .logits = logits, .top_k = std::move(processed_top_k), .top_p = top_p};
    kernel::apply_top_k_top_p(params);
    return;
  }
#endif
  apply_top_k_top_p_torch_impl(logits, top_k, top_p);
}

void apply_logits_transforms(torch::Tensor& logits,
                             const torch::Tensor& temperatures,
                             const torch::Tensor& top_k,
                             const torch::Tensor& top_p) {
  if (temperatures.defined()) {
    apply_temperatures(logits, temperatures);
  }
  apply_top_k_top_p(logits, top_k, top_p);
}

void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask) {
  CHECK(logits.defined()) << "logits must be defined";
  CHECK(bitmask.defined()) << "bitmask must be defined";
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D [batch, vocab]";
  CHECK_EQ(bitmask.dim(), 2) << "bitmask must be 2-D [batch, words]";
  CHECK_EQ(logits.size(/*dim=*/0), bitmask.size(/*dim=*/0))
      << "bitmask batch mismatch, logits.size(0)=" << logits.size(/*dim=*/0)
      << ", bitmask.size(0)=" << bitmask.size(/*dim=*/0);

  const int64_t vocab_size = logits.size(/*dim=*/1);
  const int64_t expected_words = (vocab_size + 31) / 32;
  CHECK_EQ(bitmask.size(/*dim=*/1), expected_words)
      << "bitmask word count mismatch, bitmask.size(1)="
      << bitmask.size(/*dim=*/1) << ", expected=" << expected_words;
  CHECK(bitmask.scalar_type() == torch::kInt32 ||
        bitmask.scalar_type() == torch::kInt64)
      << "bitmask must be int32/int64";

#if defined(USE_NPU)
  if (kernel::npu::tilelang::can_apply_token_bitmask_inplace(logits, bitmask)) {
    kernel::npu::tilelang::apply_token_bitmask_inplace(logits, bitmask);
    return;
  }
#endif

  static thread_local torch::Tensor word_indices;
  static thread_local torch::Tensor bit_indices;
  static thread_local int64_t cached_vocab_size = -1;
  static thread_local torch::Device cached_device(/*type=*/torch::kCPU);
  const auto device = logits.device();
  if (vocab_size != cached_vocab_size || device != cached_device) {
    const auto options =
        torch::TensorOptions().dtype(/*dtype=*/torch::kInt64).device(device);
    const auto token_ids = torch::arange(vocab_size, options);
    word_indices = torch::floor_divide(token_ids, /*other=*/32);
    bit_indices = torch::remainder(token_ids, /*other=*/32);
    cached_vocab_size = vocab_size;
    cached_device = device;
  }
  torch::Tensor words = bitmask.to(/*dtype=*/torch::kInt64)
                            .bitwise_and(/*other=*/0xffffffffLL)
                            .index_select(/*dim=*/1, word_indices);
  const torch::Tensor allowed =
      torch::bitwise_and(torch::bitwise_right_shift(words, bit_indices),
                         /*other=*/1);
  const auto mask_options =
      torch::TensorOptions().dtype(logits.dtype()).device(logits.device());
  const torch::Tensor additive = torch::where(
      allowed.to(/*dtype=*/torch::kBool),
      torch::zeros(/*size=*/{1}, mask_options),
      torch::full(/*size=*/{1}, /*fill_value=*/-1.0e9F, mask_options));
  logits.add_(additive);
}

torch::Tensor process_logits(torch::Tensor& logits,
                             const SamplingParameters& params,
                             const torch::Tensor& filter_mask) {
  const torch::Tensor& effective_filter_mask =
      filter_mask.defined() ? filter_mask : params.filter_mask;
  apply_sampling_penalties(logits, params);

  torch::Tensor sample_logits = logits;
  torch::Tensor sample_filter_bitmask = params.filter_bitmask;
  const bool use_sample_indices =
      params.selected_token_idxes.numel() != params.sample_idxes.numel();
  if (use_sample_indices) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
    if (sample_filter_bitmask.defined()) {
      sample_filter_bitmask =
          sample_filter_bitmask.index_select(/*dim=*/0, params.sample_idxes);
    }
  }

  if (sample_filter_bitmask.defined()) {
    apply_token_bitmask_inplace(sample_logits, sample_filter_bitmask);
  } else if (effective_filter_mask.defined()) {
    CHECK_EQ(effective_filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << effective_filter_mask.dim();
    CHECK_EQ(effective_filter_mask.size(/*dim=*/0),
             sample_logits.size(/*dim=*/0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << effective_filter_mask.size(/*dim=*/0)
        << ", sample_logits.size(0)=" << sample_logits.size(/*dim=*/0);
    CHECK_EQ(effective_filter_mask.size(/*dim=*/1),
             sample_logits.size(/*dim=*/1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << effective_filter_mask.size(/*dim=*/1)
        << ", sample_logits.size(1)=" << sample_logits.size(/*dim=*/1);
    sample_logits = sample_logits + effective_filter_mask;
  }
  return sample_logits;
}

}  // namespace xllm
