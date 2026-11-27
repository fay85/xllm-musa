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

#include "core/runtime/draft_model_config.h"

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/model_args.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"
#include "core/util/utils.h"
#if defined(USE_NPU)
#include "core/framework/config/kernel_config.h"
#endif

namespace xllm {
namespace {

void configure_deepseek_v4_dspark_args(ModelArgs& args) {
  const bool v41_draft =
      util::is_deepseek_v41_dspark_model_type(args.model_type());
  CHECK_GT(args.dspark_num_layers(), 0)
      << "DeepSeek-V4 DSpark requires at least one draft layer.";
  // The checkpoint lists the decoder layers first and appends the draft stages,
  // so keep the decoder count before n_layers is swapped for the draft count.
  const int32_t decoder_layers = static_cast<int32_t>(args.n_layers());
  args.n_layers(args.dspark_num_layers());
  args.n_hash_layers(0);
  if (v41_draft && args.dspark_n_routed_experts() > 0) {
    args.n_routed_experts(args.dspark_n_routed_experts());
  }
  if (v41_draft && args.dspark_num_experts_per_tok() > 0) {
    args.n_activated_experts(args.dspark_num_experts_per_tok());
    args.num_experts_per_tok(args.dspark_num_experts_per_tok());
  }

  // Draft stages use the checkpoint tail. V4 and V4.1 encode SWA differently.
  const size_t draft_layer_count =
      static_cast<size_t>(args.dspark_num_layers());
  std::vector<int32_t> draft_compress_ratios(draft_layer_count,
                                             /*value=*/v41_draft ? 0 : 1);
  const std::vector<int32_t>& ckpt_compress_ratios = args.compress_ratios();
  bool compress_ratios_from_ckpt = false;
  if (ckpt_compress_ratios.size() ==
      static_cast<size_t>(decoder_layers) + draft_layer_count) {
    std::copy(ckpt_compress_ratios.end() - draft_layer_count,
              ckpt_compress_ratios.end(),
              draft_compress_ratios.begin());
    compress_ratios_from_ckpt = true;
    if (!v41_draft) {
      std::transform(draft_compress_ratios.begin(),
                     draft_compress_ratios.end(),
                     draft_compress_ratios.begin(),
                     util::deepseek_v4_normalize_compress_ratio);
    }
  } else if (!ckpt_compress_ratios.empty()) {
    LOG(WARNING) << "DSpark draft compress_ratios size "
                 << ckpt_compress_ratios.size() << " != decoder_layers("
                 << decoder_layers << ") + draft_layers(" << draft_layer_count
                 << "); falling back to the family default ("
                 << (v41_draft ? 0 : 1) << ") for all draft layers.";
  }
  LOG(INFO) << "DSpark draft compress_ratios=["
            << absl::StrJoin(draft_compress_ratios, ",") << "] over "
            << draft_layer_count << " draft layers (v41=" << v41_draft
            << ", from "
            << (compress_ratios_from_ckpt ? "checkpoint tail"
                                          : "fallback default")
            << ")";
  args.compress_ratios(std::move(draft_compress_ratios));

  // The draft stages own no compressed-KV source (source ids address decoder
  // layers), no indexer, no engram and no candidate layer. Clearing them keeps
  // the draft's cache plan limited to its own SWA stages instead of inheriting
  // decoder-layer ids that fall outside the draft's layer range.
  args.kv_source_layer_ids({});
  args.index_source_layer_ids({});
  args.candidate_source_layer_id(-1);
  args.candidate_topk_blocks(0);
  args.candidate_block_size(0);
  args.engram_layer_ids({});
  args.engram_num_embeddings({});

  args.draft_query_uses_decode(/*draft_query_uses_decode=*/true);
}

// The caller has already classified the algorithm as block diffusion.
bool supports_block_diffusion_draft(std::string_view algorithm,
                                    std::string_view model_type) {
  if (Platform::is_npu()) {
    return true;
  }
  if (Platform::is_mlu()) {
    return SpeculativeConfig::is_dflash2_algorithm(algorithm) ||
           (SpeculativeConfig::is_dspark_algorithm(algorithm) &&
            util::is_deepseek_v41_model_type(model_type));
  }
  return false;
}

}  // namespace

// Reads a draft config's ordered target-side aux hidden capture layer ids.
// Legacy target-layer keys keep their model-specific boundary semantics; the
// speculators boundary-index keys are converted to post-layer ids. With
// `required` an empty result is fatal; otherwise the caller may default it.
std::vector<int32_t> read_capture_layer_ids(
    const std::string& model_weights_path,
    bool required) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  if (!reader.parse(config_path)) {
    CHECK(!required) << "Failed to parse draft config: " << config_path;
    return {};
  }

  std::vector<int32_t> capture_layer_ids =
      AuxHiddenCapture::read_capture_layer_ids(reader);
  CHECK(!required || !capture_layer_ids.empty())
      << "Block-diffusion draft config requires dspark_target_layer_ids, "
         "target_layer_ids, dflash_config.target_layer_ids, "
         "text_config.dspark_target_layer_ids, "
         "text_config.target_layer_ids, or aux_hidden_state_layer_ids: "
      << config_path;
  return capture_layer_ids;
}

// Reads the block-diffusion draft's mask token id. The checkpoint nests the
// draft keys under ``text_config`` while the deployment side may patch them at
// the config root. The root-level aliases are listed first so an existing
// patch keeps winning (idempotent with the text_config fallback).
int32_t read_mask_token_id(const JsonReader& reader) {
  const std::vector<std::string> root_mask_keys{
      "dflash_config.mask_token_id",
      "mask_token_id",
      "dspark_noise_token_id",
      "text_config.dspark_noise_token_id",
      "text_config.mask_token_id"};
  const int32_t mask_token_id =
      reader.value_or<int32_t>(root_mask_keys, /*default=*/-1);
  // Config-drift probe: a root-level patch and the checkpoint's text_config
  // disagree. The root value wins above, so surface the other side instead of
  // silently ignoring it. Same alias order as the main list.
  const std::vector<std::string> text_mask_keys{
      "text_config.dspark_noise_token_id", "text_config.mask_token_id"};
  const int32_t text_mask_token_id =
      reader.value_or<int32_t>(text_mask_keys, mask_token_id);
  if (mask_token_id >= 0 && text_mask_token_id != mask_token_id) {
    // Name the two keys that disagree so a cross-alias value clash is not
    // mistaken for a same-key override (mirrors value_or's alias order).
    std::string root_key = "<none>";
    for (const std::string& key : root_mask_keys) {
      if (reader.value<int32_t>(key).has_value()) {
        root_key = key;
        break;
      }
    }
    std::string text_key = "<none>";
    for (const std::string& key : text_mask_keys) {
      if (reader.value<int32_t>(key).has_value()) {
        text_key = key;
        break;
      }
    }
    LOG(WARNING) << "Block-diffusion config drift: '" << root_key
                 << "' resolved mask_token_id=" << mask_token_id << " but '"
                 << text_key << "' carries " << text_mask_token_id
                 << "; the root-level value wins.";
  }
  return mask_token_id;
}

void configure_block_diffusion_target(ModelArgs& args,
                                      const runtime::Options& options) {
  CHECK(options.draft_model_path().has_value())
      << "block-diffusion speculative decoding requires --draft_model.";
  std::vector<int32_t> capture_ids =
      read_capture_layer_ids(*options.draft_model_path());
  std::unordered_set<int32_t> unique_ids;
  for (const int32_t layer_id : capture_ids) {
    CHECK_GE(layer_id, 0) << "Invalid target capture layer.";
    CHECK_LT(layer_id, args.n_layers())
        << "Target capture layer is out of range.";
    CHECK(unique_ids.insert(layer_id).second)
        << "Duplicate target capture layer.";
  }
  args.layers_to_capture(std::move(capture_ids));
}

void configure_block_diffusion_draft(ModelArgs& args,
                                     const runtime::Options& options) {
  const std::string& algorithm = options.speculative_algorithm();
  CHECK(supports_block_diffusion_draft(algorithm, args.model_type()))
      << algorithm << " block-diffusion draft is not supported on "
      << Platform::type_str() << ".";
  const int32_t num_speculative_tokens = options.num_speculative_tokens();
  const bool is_dspark = SpeculativeConfig::is_dspark_algorithm(algorithm);
  const bool is_deepseek_v4_dspark =
      is_dspark && util::is_deepseek_v4_model_type(args.model_type());
  const bool is_deepseek_v41_dspark =
      is_deepseek_v4_dspark &&
      util::is_deepseek_v41_model_type(args.model_type());
  const bool python_model_impl = ModelConfig::is_python_model_impl(
      ModelConfig::get_instance().model_impl());
  const bool v41_draft =
      is_deepseek_v41_dspark && (python_model_impl || Platform::is_mlu());
  const bool is_python_qwen3_pipeline =
      Platform::is_npu() && python_model_impl && args.model_type() == "qwen3";
  std::string draft_model_type = std::string(kDFlashDraftModelType);
  if (is_dspark) {
    draft_model_type = std::string(kDSparkDraftModelType);
  } else if (SpeculativeConfig::is_dflash2_algorithm(algorithm)) {
    const int64_t trained_num_speculative_tokens =
        args.num_speculative_tokens();
    CHECK_GT(trained_num_speculative_tokens, 0)
        << "DFlash2 requires a positive checkpoint token budget.";
    if (Platform::is_mlu()) {
      CHECK_LE(num_speculative_tokens, trained_num_speculative_tokens)
          << "DFlash2 runtime token count must not exceed the trained budget.";
    } else {
      CHECK_EQ(num_speculative_tokens, trained_num_speculative_tokens)
          << "DFlash2 runtime token count must match dflash_config.block_size "
             "excluding the anchor.";
    }
    CHECK_LE(args.dflash2_conv_kernel_size(), num_speculative_tokens + 1)
        << "DFlash2 draft queries must cover the convolution kernel.";
    draft_model_type = std::string(kDFlash2DraftModelType);
    args.dummy_token_count(num_speculative_tokens + 1);
  }
  if (is_deepseek_v4_dspark) {
    // Native MLU and Python select the V4.1 draft's SWA/cache semantics.
    draft_model_type = v41_draft
                           ? std::string(util::kDeepseekV41DSparkModelType)
                           : std::string(util::kDeepseekV4DSparkModelType);
  }
  LOG(INFO) << "Overriding draft model_type from " << args.model_type()
            << " to " << draft_model_type
            << " for block-diffusion speculative decoding";
  args.layers_to_capture({});
  args.model_type(draft_model_type);
  args.num_speculative_tokens(num_speculative_tokens);
  args.draft_query_uses_decode(false);
  args.requires_eager_execution(!is_deepseek_v4_dspark &&
                                !is_python_qwen3_pipeline);
  if (SpeculativeConfig::is_dflash_algorithm(algorithm) &&
      is_python_qwen3_pipeline) {
    args.dummy_token_count(num_speculative_tokens + 1);
  }
  if (is_dspark && (python_model_impl || v41_draft)) {
    args.dummy_token_count(num_speculative_tokens);
  }
  if (is_deepseek_v4_dspark) {
    configure_deepseek_v4_dspark_args(args);
  }
}

void configure_mtp_family_draft(ModelArgs& args,
                                const runtime::Options& options) {
#if !defined(USE_NPU)
  args.num_speculative_tokens(options.num_speculative_tokens());
#endif
  if (options.num_speculative_tokens() != 0 ||
      args.num_nextn_predict_layers() == 0) {
    return;
  }
  const std::string& current_type = args.model_type();
#if defined(USE_NPU)
  const char* mtp_model_type = nullptr;
  if (current_type == "qwen3_5_text" || current_type == "qwen3_5") {
    mtp_model_type = "qwen3_5_mtp";
  } else if (current_type == "qwen3_5_moe_text" ||
             current_type == "qwen3_5_moe") {
    mtp_model_type = "qwen3_5_moe_mtp";
  }
  if (mtp_model_type != nullptr) {
    LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
              << mtp_model_type << " for speculative decoding";
    args.model_type(mtp_model_type);
  }
#else
  static const std::unordered_map<std::string, std::string>
      kModelTypeToMtpType = {
          {"deepseek_v3", "deepseek_v3_mtp"},
          {"deepseek_v32", "deepseek_v32_mtp"},
          {"deepseek_v4", "deepseek_v4_mtp"},
          {"glm_moe_dsa", "glm_moe_dsa_mtp"},
          {"joyai_llm_flash", "joyai_llm_flash_mtp"},
          {"mimo", "mimo_mtp"},
          {"qwen3_5", "qwen3_5_mtp"},
          {"qwen3_5_moe", "qwen3_5_moe_mtp"},
          {"qwen3_5_text", "qwen3_5_mtp"},
          {"qwen3_5_moe_text", "qwen3_5_moe_mtp"},
      };
  auto it = kModelTypeToMtpType.find(current_type);
  if (it != kModelTypeToMtpType.end()) {
    LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
              << it->second << " for speculative decoding";
    args.model_type(it->second);
  }
#endif
}

}  // namespace xllm
