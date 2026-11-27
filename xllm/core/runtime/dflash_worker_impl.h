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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache_transfer/kv_cache_transfer.h"
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/model/model_args.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/runtime/draft_model_spec_worker_impl.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/util/utils.h"

namespace xllm {

namespace dflash_detail {

inline int32_t decode_draft_width(int32_t num_speculative_tokens,
                                  bool sample_from_anchor) {
  return sample_from_anchor ? num_speculative_tokens
                            : num_speculative_tokens + 1;
}

inline void reset_draft_model_metadata(ModelInputParams& input_params) {
  input_params.clear_linear_attention_state();
  input_params.num_accepted_tokens = torch::Tensor();
  input_params.num_accepted_tokens_host.clear();
  // Attention metadata is model-owned: DeepSeek-V4 bakes DSA group layout and
  // sparse tiling values such as ori_win_left into opaque tensors. A draft
  // input copied from the target must rebuild those tensors for draft geometry.
  input_params.attn_metadata.reset();
}

// Draft sampling params: clear the sequence-level constraint masks (drafts
// sample unconstrained; the per-position verify rows enforce the grammar)
// while keeping every other sampling setting (do_sample, penalties, greedy)
// intact. Do NOT reuse prepare_query_inputs' params - those also force greedy
// and repeat the parameter rows for the expanded draft positions.
inline SamplingParameters make_draft_sampling_params(
    const SamplingParameters& params) {
  SamplingParameters out = params;
  out.filter_bitmask = torch::Tensor();
  out.filter_mask = torch::Tensor();
  return out;
}

}  // namespace dflash_detail

// DFlash/DFlash2 proposal XFIA platform and execution-mode eligibility.
// Final prelaunch admission also checks KV layout and model capabilities.
bool enable_dflash_proposal_xfia(const torch::Device& device,
                                 const runtime::Options& options);

class DFlashWorkerImpl : public DraftModelSpecWorkerImpl {
 public:
  // VLM deployments must pass their real worker type so the target builds
  // the multimodal pipeline; see the constructor definition for the silent
  // failure it prevents. The argument is mandatory: omitting it is a compile
  // error, so call sites cannot silently fall back to a text-only target.
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   WorkerType worker_type);

  ~DFlashWorkerImpl() override;

 protected:
  void apply_target_model_overrides(ModelArgs& args) const override;
  void apply_draft_model_overrides(ModelArgs& args) const override;
  void on_target_model_loaded() override;
  void on_draft_model_loaded(const std::string& model_weights_path) override;

  bool can_prepare_without_compute_stream_wait(
      const ForwardInput& input) const override;
  bool can_prepare_device_draft_inputs(const ForwardInput& input) const;
  void prepare_target_metadata(ForwardInput& validate_input);
  std::pair<ForwardInput, torch::Tensor> prepare_pending_draft_query(
      const ForwardInput& input,
      const std::vector<int64_t>& rows);
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;

  // The default idle path retires both forwards before returning. A subclass
  // that defers this wait must retain their inputs until the target is ready.
  virtual void finish_empty_decode(const ForwardInput& input,
                                   ForwardOutput& output,
                                   std::optional<ForwardOutput>& draft_output);

  virtual ForwardOutput run_decode_draft(const ForwardInput& input,
                                         ForwardInput& validate_input);
  virtual ForwardOutput run_decode_draft_from_pending(
      const ForwardInput& input,
      const std::vector<int64_t>& rows);

  virtual void prepare_validate_output_async(const ForwardInput& input);
  virtual void finalize_validate_output_async(const ForwardInput& input,
                                              SampleOutput output,
                                              ForwardOutput& target_output);
  std::vector<int64_t> build_batch_to_pending_row_mapping(
      const ForwardInput& input) const;
  void commit_pending_context();
  void retire_async_output();

  // Block layout hook: false (DFlash) -> query_width N+1, slot 0 is the
  // un-selected anchor; true (DSpark) -> query_width N, every position predicts
  // and slot 0 predicts the first draft token. prepare_query_inputs and the
  // query row builder read this so the whole (helper-heavy) query-build logic
  // stays here and a subclass flips one bit.
  virtual bool sample_from_anchor() const { return false; }

  // DFlash/DFlash2 drafts are standalone full-attention models, so they cannot
  // follow the target's kv_split sharding: every rank keeps a full replica of
  // the context and one logical block owns kv_split physical pages. Drafters
  // that share the target's architecture (MTP, DSpark) keep the default 1.
  int64_t draft_replicated_block_pages() const override {
    return SpeculativeConfig::draft_replicated_block_pages(
        options_.speculative_algorithm(),
        parallel_args_.kv_split_size_effective());
  }

  // Shared with subclasses (DSpark): build the N/N+1-wide draft query block and
  // the target validate input. A DSpark override of run_decode_draft calls both
  // before its draft forward.
  void prepare_decode_input(ForwardInput& input);
  void prepare_query_inputs(const ForwardInput& input,
                            ForwardInput& query_input);
  void prepare_validate_inputs(const ForwardInput& input,
                               ForwardInput& validate_input);
  std::optional<ForwardOutput> run_validate(const ForwardInput& input,
                                            ForwardInput& validate_input,
                                            ForwardOutput& draft_output,
                                            bool async_output);

  // A replicated draft pool stores kv_split physical pages per logical block,
  // so the draft's own block table has to be expanded to address them. The
  // decode input builder uses the logical slot arithmetic against the
  // expanded table with the physical dim it still resolves to the
  // draft's flat cache index (logical_block * kv_split + page_in_block).
  // No-op when the draft follows the target's kv_split sharding.
  void expand_draft_block_tables(ModelInputParams& input_params) const;

  struct PendingContext {
    ForwardInput identity;
    torch::Tensor accepted_device;
    torch::Tensor accepted_host;
    torch::Tensor embeddings;
    torch::Tensor base_positions;
    StreamEventPtr d2h_ready;
    StreamEventPtr context_ready;
  };

  bool async_output_enabled_ = false;
  bool proposal_xfia_enabled_ = false;
  bool proposal_attention_enabled_ = false;
  bool draft_prelaunch_enabled_for_init_ = false;
  std::optional<PendingContext> pending_context_;
  // One idle step's inputs; retired only after the following step is queued.
  ForwardOutput idle_decode_output_;
  torch::Tensor prepared_base_positions_;
  dflash_async::DraftCacheBlockTable prepared_context_block_table_;
  StreamEventPtr prepared_context_ready_;
  std::vector<torch::Tensor> consumed_context_tensors_;

 private:
  friend class DFlashWorkerImplTestAccess;

  ForwardOutput execute_greedy_draft_query(ForwardInput query, bool prepared);
  BatchForwardType draft_batch_forward_type() const {
    return draft_impl_->context_.get_model_args().draft_query_uses_decode()
               ? BatchForwardType::DECODE
               : BatchForwardType::CHUNKED_PREFILL;
  }

  // Mirrors sampled tokens to rank 0 under schedule-overlap so every rank
  // commits the same prefix. No-op for a single rank.
  void maybe_broadcast_spec_tokens(torch::Tensor& tokens);

  void update_decode_step_input(
      ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  KvPushFailureIds write_context_kv(
      const ForwardInput& input,
      const torch::Tensor& context_hidden,
      const torch::Tensor& positions_device,
      const torch::Tensor& new_cache_slots_device);

  KvPushFailureIds write_target_context_to_cache(
      const ForwardInput& input,
      const SampleOutput& validate_output);

  // Builds per-position JSON grammar validation states for the target verify
  // rows from the sequence-level states and the drafted block: advances each
  // state across the block (freezing at the first illegal draft token) and
  // attaches the resulting TARGET-phase bitmask to `validate_input`.
  // `draft_tokens_host` is a CPU copy of the [batch, N] drafted block;
  // draft_tokens_ready fences its D2H copy without draining later work.
  void prepare_json_validation_states(const ForwardInput& input,
                                      const torch::Tensor& draft_tokens_host,
                                      const StreamEventPtr& draft_tokens_ready,
                                      ForwardInput& validate_input);

 protected:
  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
  // Whether the target model owns recurrent (linear attention) layers.
  bool target_is_hybrid_recurrent_ = false;
  bool uses_block_parallel_draft_decode() const;
};

}  // namespace xllm
