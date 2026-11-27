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

#pragma once

#include <boost/algorithm/string/predicate.hpp>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <string_view>

#include "core/common/macros.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/option_category.h"
#include "core/framework/model/model_args.h"

namespace xllm {

class JsonReader;

class SpeculativeConfig final {
 public:
  inline static constexpr std::string_view kDFlash2Algorithm = "DFlash2";

  SpeculativeConfig() = default;
  ~SpeculativeConfig() = default;

  static SpeculativeConfig& get_instance();

  static bool is_eagle3_algorithm(std::string_view algorithm);

  static bool is_suffix_algorithm(std::string_view algorithm);

  static bool uses_ssm_engine(std::string_view draft_model_path,
                              std::string_view algorithm,
                              int32_t num_speculative_tokens);

  // Whether a speculative algorithm requires the target model to capture
  // intermediate-layer aux hidden states to drive the draft (Eagle3 and
  // DFlash). Centralizes the algorithm-string classification in one place. The
  // worker consults it to decide whether to populate the target's
  // layers_to_capture; the model then keys off that list alone, never the
  // algorithm string. Takes the algorithm explicitly so a spawned worker
  // process, which reads its own Options rather than this global config, can
  // classify without an initialized singleton.
  static bool requires_aux_hidden_capture(std::string_view algorithm);

  // Whether the speculative worker builds the draft paged cache from the
  // draft ModelArgs instead of reusing the target KVCacheShape. DFlash,
  // DFlash2, and DSpark all dispatch to DraftModelSpecWorkerImpl descendants
  // whose draft_kv_cache_shape unconditionally routes through
  // build_draft_kv_cache_shape (the draft's own slot geometry) whenever the
  // target does not use a grouped cache layout; grouped-layout families take
  // the shared-kpool plan before the engine's calculate_kv_cache.
  // Master-side capacity estimation must size such drafts by their own
  // per-slot geometry; every other worker allocates draft KV tensors with
  // the target KVCacheShape, so its draft slot must not exceed the target
  // slot. Keep this mapping in sync with the algorithm-to-worker dispatch
  // in runtime/worker.cpp.
  static bool draft_manages_own_kv_shape(std::string_view algorithm) {
    return is_block_diffusion_algorithm(algorithm);
  }

  static bool is_dflash2_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, kDFlash2Algorithm);
  }

  static bool is_dflash_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "DFlash");
  }

  // The glm5_next + DFlash2 pairing this whitelist admits is proven correct
  // (aux-hidden capture restores global rows under CP, unit-tested in
  // test_glm5_next_cp.py). Both the master admission gate and the draft
  // worker gate consume this predicate so the whitelist cannot drift when
  // further algorithms or models are cleared.
  static bool is_glm5_next_dflash2(std::string_view model_type,
                                   std::string_view algorithm) {
    return model_type == "glm5_next" && is_dflash2_algorithm(algorithm);
  }

  // The glm_moe_dsa (GLM-5.2/5.3) + DFlash2 pairing, the mirror of
  // is_glm5_next_dflash2: glm5_2.py restores the aux-hidden buffer to
  // global row order under CP by cp_merge_rows on the concatenated
  // (hidden, aux_hidden_buffer) pair, unit-tested in
  // test_glm5_2_cp_ragged.py::
  // test_cp_aux_hidden_is_merged_with_target_in_capture_order. Consumed by
  // the same master admission and draft worker gates. Unlike
  // is_glm5_next_dflash2 the admission is python-impl only (the native/ATB
  // glm_moe_dsa has no aux capture and keeps the startup-time rejection),
  // so the impl restriction is folded into the predicate itself: leaving it
  // in the gates would let the two whitelists drift apart. The impl stays
  // an explicit argument rather than a singleton read so a spawned worker
  // can classify from its own Options, matching the convention of
  // requires_aux_hidden_capture.
  static bool is_glm_moe_dsa_dflash2(std::string_view model_type,
                                     std::string_view algorithm,
                                     std::string_view model_impl) {
    return model_type == "glm_moe_dsa" && is_dflash2_algorithm(algorithm) &&
           ModelConfig::is_python_model_impl(model_impl);
  }

  // The deepseek_v41 + DSpark pairing. The drafter is the checkpoint's own
  // mtp.0/1/2 block, so it inherits the target's kv_split sharding
  // (owns_replicated_kv_cache() is false) and reuses the DFlash worker's PD
  // bootstrap-embedding handoff, which makes the pairing usable under
  // disaggregated PD. Kept next to is_glm5_next_dflash2 so the VLM PD
  // whitelist in Master::Master reads one place.
  static bool is_deepseek_v41_dspark(std::string_view model_type,
                                     std::string_view algorithm) {
    return model_type == "deepseek_v41" && is_dspark_algorithm(algorithm);
  }

  static bool is_mtp_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "MTP");
  }

  // Structured output requires one grammar state per target verify position.
  static bool drives_constraint_channel(std::string_view algorithm);

  // Qwen3.8-27B reports qwen3_5 as its target model_type.
  static bool is_qwen3_5_mtp_or_dflash2(std::string_view model_type,
                                        std::string_view algorithm) {
    return model_type == "qwen3_5" &&
           (is_mtp_algorithm(algorithm) || is_dflash2_algorithm(algorithm));
  }

  static bool is_qwen3_5_dspark(std::string_view model_type,
                                std::string_view algorithm) {
    return is_qwen3_5_target_model_type(model_type) &&
           is_dspark_algorithm(algorithm);
  }

  // DFlashDraftModel is ATB-only; the dense qwen3_5 VLM target is TORCH-only.
  // Admit the combination only on the Python executor, whose qwen3_dflash
  // implementation runs on TORCH.
  static bool is_qwen3_5_python_dflash(std::string_view model_type,
                                       std::string_view algorithm,
                                       std::string_view model_impl) {
    return model_type == "qwen3_5" && is_dflash_algorithm(algorithm) &&
           ModelConfig::is_python_model_impl(model_impl);
  }

  static bool is_qwen3_vl_eagle3(std::string_view model_type,
                                 std::string_view algorithm) {
    return model_type == "qwen3_vl" && algorithm == "Eagle3";
  }

  // JoyaiASR uses the CTC prompt-lookup Suffix draft on the VLM path. The
  // draft is built from audio CTC hypotheses and token history, so the
  // combination needs no separate draft checkpoint. It is admitted for
  // aggregated service routing only; the gated PD path is unchanged.
  static bool is_joyai_asr_suffix(std::string_view model_type,
                                  std::string_view algorithm) {
    return model_type == "joyai_asr" && algorithm == "Suffix";
  }

  // Every OxygenVLM-prefixed target shares the same speculative treatment as
  // Qwen3.5: MTP and DFlash2 use the target's own draft path, DSpark uses the
  // standalone block-diffusion drafter.
  static bool is_oxygenvlm_target_model_type(std::string_view model_type) {
    return model_type.rfind("oxygenvlm", 0) == 0;
  }

  static bool is_oxygenvlm_mtp_or_dflash2(std::string_view model_type,
                                          std::string_view algorithm) {
    return is_oxygenvlm_target_model_type(model_type) &&
           (is_mtp_algorithm(algorithm) || is_dflash2_algorithm(algorithm));
  }

  static bool is_oxygenvlm_dspark(std::string_view model_type,
                                  std::string_view algorithm) {
    return is_oxygenvlm_target_model_type(model_type) &&
           is_dspark_algorithm(algorithm);
  }

  // True for the DSpark block-diffusion draft.
  static bool is_dspark_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "DSpark");
  }

  // True for the block-diffusion draft algorithms (DFlash, DSpark) that record
  // validate metrics inline per-seq and drive the adaptive per-seq varlen
  // prune. Case-insensitive so it matches however the flag was cased. MTP is
  // classified separately via is_mtp_algorithm; callers that also accept MTP
  // must OR the two.
  static bool is_block_diffusion_algorithm(std::string_view algorithm);

  // Standalone drafters own a full-attention KV pool built from their own
  // architecture instead of running inside the target's KV. GLM-5.3-Flash's
  // DFlash2 draft is a Qwen3-style block-diffusion model, so it cannot follow
  // the target's kv_split sharding and keeps a complete replica of the context
  // on every kv_split rank (one logical block = kv_split physical pages).
  //
  // Drafters that share the target's architecture (MTP, Eagle3, DeepSeek-V4
  // DSpark) inherit the target's sharding and must return false here.
  static bool owns_replicated_kv_cache(std::string_view algorithm);

  // Physical pages one logical block occupies in the drafter's KV pool. 1
  // means the draft follows the target's kv_split sharding.
  static int32_t draft_replicated_block_pages(std::string_view algorithm,
                                              int32_t kv_split_size) {
    return kv_split_size > 1 && owns_replicated_kv_cache(algorithm)
               ? kv_split_size
               : 1;
  }

  // MTP and Eagle3 resume their draft from the target hidden state.
  static bool requires_bootstrap_hidden(std::string_view algorithm);
  static bool supports_host_kv_cache(std::string_view algorithm);

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void validate() const;

  [[nodiscard]] static const OptionCategory& option_category();

  PROPERTY(std::string, draft_model);

  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "MTP";

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, enable_probabilistic_draft) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;
};

}  // namespace xllm
