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

#include "suffix_worker_impl.h"

#include <algorithm>
#include <unordered_map>

#include "common/metrics.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/runtime/json_object_decode_adapter.h"
#include "core/runtime/speculative_worker_utils.h"
#include "framework/sampling/sampling_params.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

void extend_validation_history(const SamplingParameters& original,
                               const std::vector<int32_t>& proposals,
                               int32_t num_sequences,
                               int32_t draft_width,
                               SamplingParameters& expanded) {
  if (!original.unique_token_ids.defined() ||
      (!original.frequency_penalties.defined() &&
       !original.repetition_penalties.defined())) {
    return;
  }
  const torch::Device device = expanded.unique_token_ids.device();
  torch::Tensor ids =
      original.unique_token_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  torch::Tensor counts =
      original.unique_token_counts.to(torch::kCPU).to(torch::kInt).contiguous();
  torch::Tensor lengths = original.unique_token_ids_lens.to(torch::kCPU)
                              .to(torch::kInt)
                              .contiguous();
  CHECK_EQ(ids.size(0), num_sequences);
  const int32_t width = draft_width + 1;
  const int64_t capacity = ids.size(1) + draft_width;
  torch::Tensor new_ids =
      torch::zeros({num_sequences * width, capacity}, torch::kLong);
  torch::Tensor new_counts =
      torch::zeros({num_sequences * width, capacity}, torch::kInt);
  torch::Tensor new_lengths =
      torch::zeros({num_sequences * width}, torch::kInt);
  const auto old_ids = ids.accessor<int64_t, 2>();
  const auto old_counts = counts.accessor<int32_t, 2>();
  const auto old_lengths = lengths.accessor<int32_t, 1>();
  auto out_ids = new_ids.accessor<int64_t, 2>();
  auto out_counts = new_counts.accessor<int32_t, 2>();
  auto out_lengths = new_lengths.accessor<int32_t, 1>();
  for (int32_t seq = 0; seq < num_sequences; ++seq) {
    std::unordered_map<int64_t, int32_t> history;
    history.reserve(static_cast<size_t>(capacity));
    for (int32_t i = 0; i < old_lengths[seq]; ++i) {
      history[old_ids[seq][i]] = old_counts[seq][i];
    }
    for (int32_t pos = 0; pos < width; ++pos) {
      const int32_t row = seq * width + pos;
      int32_t column = 0;
      for (const auto& [token, count] : history) {
        out_ids[row][column] = token;
        out_counts[row][column] = count;
        ++column;
      }
      out_lengths[row] = column;
      if (pos < draft_width) {
        ++history[proposals[seq * draft_width + pos]];
      }
    }
  }
  expanded.unique_token_ids = new_ids.to(device);
  expanded.unique_token_counts = new_counts.to(device);
  expanded.unique_token_ids_lens = new_lengths.to(device);
}

// The target may emit one punctuation token that the CTC hypothesis omits
// (e.g. a pause marker); a single-token history gap bridges it. Gap 0
// measured strictly worse acceptance; gaps >= 2 measured no extra gain.
constexpr int32_t kMaxHistoryGap = 1;
// Longest prompt suffix considered when anchoring the hint tail.
constexpr size_t kMaxPromptMatchWidth = 16;

std::vector<int32_t> find_prompt_lookup_draft(std::span<const int32_t> history,
                                              std::span<const int32_t> hint,
                                              int32_t max_tokens) {
  if (history.empty() || hint.empty() || max_tokens <= 0) {
    return {};
  }
  const size_t max_gap =
      std::min<size_t>(static_cast<size_t>(kMaxHistoryGap), history.size() - 1);
  // Exact suffixes take priority. A bounded gap can bridge a punctuation
  // token emitted by the target but absent from the CTC hypothesis.
  for (size_t gap = 0; gap <= max_gap; ++gap) {
    const auto prefix = history.first(history.size() - gap);
    const size_t max_match = std::min(kMaxPromptMatchWidth, prefix.size());
    for (size_t width = max_match; width > 0; --width) {
      for (size_t end = width; end < hint.size(); ++end) {
        if (!std::equal(prefix.end() - width,
                        prefix.end(),
                        hint.begin() + end - width)) {
          continue;
        }
        const size_t count = std::min<size_t>(static_cast<size_t>(max_tokens),
                                              hint.size() - end);
        return std::vector<int32_t>(hint.begin() + end,
                                    hint.begin() + end + count);
      }
    }
  }
  return {};
}

}  // namespace

namespace {
runtime::Options SuffixTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}
}  // namespace

SuffixWorkerImpl::SuffixWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   WorkerType worker_type)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            SuffixTargetOptions(options),
                            worker_type),
      suffix_cache_(options_.speculative_suffix_cache_max_depth(),
                    options_.speculative_suffix_max_cached_requests()),
      vlm_target_(worker_type == WorkerType::VLM) {}

ForwardInput SuffixWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  auto& input_params = inputs.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = this->block_size();

  const auto& next_tokens = last_step_output_.sample_output.next_tokens;
  const auto last_token_ids = safe_to(next_tokens,
                                      torch::TensorOptions()
                                          .dtype(/*dtype=*/torch::kInt64)
                                          .device(/*device=*/torch::kCPU))
                                  .contiguous();
  const Slice<int64_t> last_tokens_ids_slice = {
      last_token_ids.const_data_ptr<int64_t>(),
      static_cast<size_t>(last_token_ids.numel())};

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num =
        last_step_output_.sample_output.next_tokens.size(/*dim=*/1);
  }

  DecodeBuildBuffers buf;
  const auto row_ctx = make_decode_batch_context(inputs);
  prepare_decode_buffers(buf,
                         row_ctx,
                         num_sequences,
                         /*emit_block_tables=*/false);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const auto [token_id, position_offset] = resolve_token_with_position_offset(
        row_ctx.token_ids[seq_id], last_tokens_ids_slice, last_step_decode_num);
    append_decode_token(row_ctx,
                        {.req_id = seq_id,
                         .token_id = token_id,
                         .position_offset = position_offset},
                        block_size,
                        buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "step-update positions/tokens mismatch";

  set_token_position_tensors(inputs,
                             buf.out_token_ids,
                             buf.out_positions,
                             inputs.token_ids.options(),
                             inputs.positions.options());
  // update the input_params
  input_params.meta.kv_max_seq_len = buf.meta.kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(buf.out_kv_seq_lens);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  input_params.attention.rebuild_device_buffer(device_);
  inputs.device_tensors_ready = true;

  return inputs;
}

void SuffixWorkerImpl::store_prompt_hint(const std::string& request_id,
                                         std::vector<int32_t> hint) {
  // Hints are advisory; the bound only stops unbounded growth from requests
  // that finish inside prefill and never reach decode cleanup. Eviction
  // follows insertion order and only degrades acceptance, never correctness.
  const size_t limit =
      static_cast<size_t>(std::max(1, options_.max_seqs_per_batch())) * 2;
  const bool is_new =
      model_prompt_hints_.find(request_id) == model_prompt_hints_.end();
  if (is_new && model_prompt_hints_.size() >= limit) {
    model_prompt_hints_.erase(model_prompt_hints_order_.front());
    model_prompt_hints_order_.pop_front();
  }
  if (is_new) {
    model_prompt_hints_order_.push_back(request_id);
  }
  model_prompt_hints_.insert_or_assign(request_id, std::move(hint));
}

void SuffixWorkerImpl::drop_prompt_hint(const std::string& request_id) {
  if (model_prompt_hints_.erase(request_id) > 0) {
    const auto it = std::find(model_prompt_hints_order_.begin(),
                              model_prompt_hints_order_.end(),
                              request_id);
    if (it != model_prompt_hints_order_.end()) {
      model_prompt_hints_order_.erase(it);
    }
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.parallel.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }
    new_input.input_params.expert.eplb_decode_token_mask =
        eplb::expand_decode_token_mask(
            new_input.input_params.expert.eplb_decode_token_mask,
            options_.num_speculative_tokens() + 1);

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const auto& input_params = input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const auto& request_ids = input_params.embedding.request_ids;

  if (request_ids.size() == static_cast<size_t>(num_sequences)) {
    const torch::Tensor& token_ids = input.token_ids_host;
    Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                       static_cast<size_t>(token_ids.numel())};

    int32_t start_idx = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t q_len = input_params.get_q_seq_len(seq_id);
      Slice<int32_t> seq_tokens =
          tokens_ids_slice.slice(start_idx, start_idx + q_len);
      start_idx += q_len;

      const std::string req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      suffix_cache_.start_request(req_id, seq_tokens);
    }

    std::vector<int32_t> audio_sequences;
    audio_sequences.reserve(static_cast<size_t>(num_sequences));
    for (const auto& data : input_params.multimodal.mm_data.mm_data_vec()) {
      if (!data.hold<MMItemVec>()) {
        continue;
      }
      for (const auto& item : data.items<MMItemVec>()) {
        if (item.type() != MMType::AUDIO) {
          continue;
        }
        audio_sequences.emplace_back(item.state().seq_index());
      }
    }
    if (!audio_sequences.empty()) {
      auto hints = impl_->get_prompt_lookup_hints();
      if (!hints.empty()) {
        CHECK_EQ(hints.size(), audio_sequences.size());
        for (size_t i = 0; i < hints.size(); ++i) {
          const int32_t seq = audio_sequences[i];
          CHECK_GE(seq, 0);
          CHECK_LT(seq, num_sequences);
          store_prompt_hint(request_ids[seq], std::move(hints[i]));
        }
      }
    }

    torch::Tensor next_tokens =
        safe_to(output.sample_output.next_tokens, torch::kCPU);
    if (next_tokens.defined() &&
        next_tokens.numel() == static_cast<int64_t>(num_sequences)) {
      next_tokens = next_tokens.view({-1}).to(torch::kInt);
      Slice<int32_t> next_tokens_slice = {
          next_tokens.data_ptr<int32_t>(),
          static_cast<size_t>(next_tokens.numel())};
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        int32_t token = next_tokens_slice[seq_id];
        if (token < 0) {
          continue;
        }
        const std::string req_id = request_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        suffix_cache_.add_active_response(
            req_id, std::span<const int32_t>(&token, /*count=*/1));
      }
    }
  }

  output.sample_output.embeddings = torch::Tensor();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_decode(
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const auto& request_ids = input.input_params.embedding.request_ids;

  const bool has_request_ids =
      request_ids.size() == static_cast<size_t>(num_sequences);
  if (has_request_ids) {
    std::unordered_set<std::string> current_req_ids;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      if (!request_ids[seq_id].empty()) {
        current_req_ids.insert(request_ids[seq_id]);
      }
    }

    for (const auto& req_id : suffix_active_decode_req_ids_) {
      if (current_req_ids.find(req_id) == current_req_ids.end()) {
        suffix_cache_.stop_request(req_id);
        drop_prompt_hint(req_id);
      }
    }
    suffix_active_decode_req_ids_ = std::move(current_req_ids);
  }

  const torch::Tensor& input_token_ids = input.token_ids_host;
  Slice<int32_t> input_tokens_slice = {
      input_token_ids.data_ptr<int32_t>(),
      static_cast<size_t>(input_token_ids.numel())};

  Timer timer;

  const bool use_ctc_proposal =
      vlm_target_ &&
      input.sampling_params.sampling_mode == SamplingMode::GREEDY;

  std::vector<int32_t> draft_tokens_flat;
  draft_tokens_flat.reserve(num_sequences * num_speculative_tokens);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t fallback_token = input_tokens_slice[seq_id];
    for (int32_t i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens_flat.emplace_back(fallback_token);
    }

    if (!has_request_ids) {
      continue;
    }

    const std::string req_id = request_ids[seq_id];
    if (req_id.empty()) {
      continue;
    }
    const std::span<const int32_t> seed_tokens(&fallback_token, /*count=*/1);
    const auto hint = use_ctc_proposal ? model_prompt_hints_.find(req_id)
                                       : model_prompt_hints_.end();
    if (hint != model_prompt_hints_.end()) {
      const auto proposed = find_prompt_lookup_draft(
          suffix_cache_.request_tail(req_id, seed_tokens),
          hint->second,
          num_speculative_tokens);
      if (!proposed.empty()) {
        for (size_t i = 0; i < proposed.size(); ++i) {
          draft_tokens_flat[seq_id * num_speculative_tokens + i] = proposed[i];
        }
        const int32_t remaining =
            num_speculative_tokens - static_cast<int32_t>(proposed.size());
        if (remaining > 0) {
          const Draft tail =
              suffix_cache_.speculate(req_id, seed_tokens, remaining);
          const int32_t fill =
              std::min<int32_t>(remaining, tail.token_ids.size());
          for (int32_t i = 0; i < fill; ++i) {
            draft_tokens_flat[seq_id * num_speculative_tokens +
                              static_cast<int32_t>(proposed.size()) + i] =
                tail.token_ids[i];
          }
        }
        continue;
      }
    }

    const Draft draft =
        suffix_cache_.speculate(req_id, seed_tokens, num_speculative_tokens);

    const int32_t fill_count =
        std::min<int32_t>(num_speculative_tokens, draft.token_ids.size());
    for (int32_t i = 0; i < fill_count; ++i) {
      draft_tokens_flat[seq_id * num_speculative_tokens + i] =
          draft.token_ids[i];
    }
  }

  ForwardInput validate_input;
  prepare_validate_inputs(
      input, validate_input, /*repeat_unique_history=*/false);

  const torch::Tensor draft_token_ids_host =
      torch::tensor(draft_tokens_flat, /*options=*/torch::kInt32)
          .view({num_sequences, num_speculative_tokens});
  torch::Tensor draft_token_ids = draft_token_ids_host.to(
      validate_input.token_ids.device(), /*dtype=*/torch::kInt64);
  validate_input.token_ids.view({num_sequences, num_val_tokens})
      .narrow(/*dim=*/1, /*start=*/1, num_speculative_tokens)
      .copy_(draft_token_ids);
  extend_validation_history(input.sampling_params,
                            draft_tokens_flat,
                            num_sequences,
                            num_speculative_tokens,
                            validate_input.sampling_params);
  if (!input.json_object_states.empty()) {
    JsonObjectDecodeAdapter::prepare_block_verify_mask(
        input, draft_token_ids_host, validate_input, device_);
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  timer.reset();
  auto future = impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  timer.reset();
  const auto verify_draft_token_ids =
      JsonObjectDecodeAdapter::mask_invalid_draft_tokens(
          draft_token_ids, validate_input.json_object_invalid_draft);
  SampleOutput val_output =
      run_rejection_sampling(validate_input.sampling_params,
                             verify_draft_token_ids,
                             /*draft_logits=*/torch::Tensor(),
                             target_output.logits,
                             target_output.sample_output.embeddings);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if (input.sampling_params.sampling_mode != SamplingMode::GREEDY) {
    speculative_worker::broadcast_speculative_token_ids(val_output.next_tokens,
                                                        parallel_args_);
  }
  val_output.next_tokens = safe_to(val_output.next_tokens,
                                   torch::TensorOptions()
                                       .dtype(/*dtype=*/torch::kInt32)
                                       .device(/*device=*/torch::kCPU))
                               .contiguous();

  std::vector<size_t> failed_sequences;
  if (!input.json_object_states.empty()) {
    failed_sequences = JsonObjectDecodeAdapter::check_verified_tokens(
        input, val_output.next_tokens, target_output);
  }
  if (has_request_ids) {
    const int32_t* accepted_token_ids =
        val_output.next_tokens.const_data_ptr<int32_t>();
    for (size_t seq = 0; seq < request_ids.size(); ++seq) {
      const std::string& req_id = request_ids[seq];
      if (req_id.empty() ||
          std::binary_search(
              failed_sequences.begin(), failed_sequences.end(), seq)) {
        continue;
      }

      const int32_t* tokens = accepted_token_ids + seq * num_val_tokens;
      const auto* accepted_end =
          std::find_if(tokens, tokens + num_val_tokens, [](int32_t token) {
            return token < 0;
          });
      const std::span<const int32_t> accepted(tokens, accepted_end);
      if (!accepted.empty()) {
        suffix_cache_.add_active_response(req_id, accepted);
      }
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  finalize_verify_output(target_output, std::move(val_output), input);
  return target_output;
}

}  // namespace xllm
