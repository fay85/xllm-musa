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

// The caller checks backend support before applying the runtime geometry.
// Call before constructing layers so attention, convolution and dummy inputs
// share the same block size, which includes the anchor token.
void apply_dflash2_runtime_block_size(ModelArgs& args,
                                      int32_t runtime_block_size) {
  const int32_t trained_block_size = args.dflash2_block_size();
  CHECK_GT(trained_block_size, 0);
  CHECK_GT(runtime_block_size, 1) << "DFlash2 requires candidate tokens.";
  CHECK_LE(runtime_block_size, trained_block_size)
      << "DFlash2 runtime block must not exceed the checkpoint's "
         "dflash_config.block_size (which includes the anchor).";
  CHECK_LE(args.dflash2_conv_kernel_size(), runtime_block_size)
      << "DFlash2 runtime block must cover the convolution kernel.";
  args.dflash2_block_size(runtime_block_size);
  args.num_speculative_tokens(runtime_block_size - 1);
  LOG(INFO) << "DFlash2 trained block size: " << trained_block_size
            << ", runtime block size: " << runtime_block_size
            << ", candidate tokens: " << runtime_block_size - 1;
}

// Apply the selected draft implementation's cache and query geometry.
void configure_deepseek_v4_dspark_args(ModelArgs& args,
                                       const runtime::Options& options,
                                       bool v41_draft,
                                       bool native_sas) {
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

  // Default to the checkpoint's block_size; --num_speculative_tokens overrides.
  const int32_t ckpt_block_size = read_block_size(options.model_path());
  const int32_t user_num_spec = options.num_speculative_tokens();
  if (user_num_spec > 0 && user_num_spec != ckpt_block_size) {
    LOG(WARNING) << "--num_speculative_tokens=" << user_num_spec
                 << " overrides DSpark checkpoint dspark_block_size="
                 << ckpt_block_size << ".";
  }
  args.dspark_block_size(user_num_spec > 0 ? user_num_spec : ckpt_block_size);

  // DSpark stages are all standard SWA layers. Their stage ids are not target
  // model layer ids, so the target compress_ratios[0..N) must not be reused.
  // V4 encoded pure SWA as compress ratio 1; V4.1 encodes it as 0.
  // Checkpoints append one ratio per draft stage. V4.1 uses zero for SWA;
  // the V4 attention implementation uses the normalized family value one.
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

  args.dspark_use_native_sas(native_sas);
  args.draft_query_uses_decode(args.dspark_use_native_sas() || v41_draft);
  args.enable_confidence_head(options.enable_adaptive_speculative_decode());
  args.confidence_head_with_markov(true);
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

// Reads the DSpark draft's block size (which includes the anchor token). The
// checkpoint nests the key under ``text_config`` while the deployment side
// may patch it at the config root. The root key is listed first so an
// existing patch keeps winning (idempotent with the fallback).
int32_t read_block_size(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse DSpark draft config: " << config_path;
  const int32_t block_size = reader.value_or<int32_t>(
      std::vector<std::string>{"dspark_block_size",
                               "text_config.dspark_block_size"},
      /*default=*/0);
  // Config-drift probe: the root-level patch and the checkpoint's text_config
  // disagree. The root value wins above, so surface the other side instead of
  // silently ignoring it.
  const int32_t text_block_size =
      reader.value_or<int32_t>("text_config.dspark_block_size", block_size);
  if (text_block_size != block_size) {
    LOG(WARNING) << "DSpark config drift: dspark_block_size=" << block_size
                 << " but text_config.dspark_block_size=" << text_block_size
                 << "; the root-level value wins.";
  }
  return block_size;
}

// Configure both target capture and draft identity in one place. Backend
// capability checks precede changes to model arguments.
void configure_block_diffusion_model(ModelArgs& args,
                                     const runtime::Options& options) {
  if (!options.is_draft_engine()) {
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
    return;
  }

  const std::string& algorithm = options.speculative_algorithm();
  CHECK(supports_block_diffusion_draft(algorithm, args.model_type()))
      << algorithm << " block-diffusion draft is not supported on "
      << Platform::type_str() << ".";
  if (SpeculativeConfig::is_dflash2_algorithm(algorithm) &&
      Platform::supports_dflash2_runtime_block_size()) {
    apply_dflash2_runtime_block_size(args, options.draft_graph_query_width());
  }
  const bool is_dspark = SpeculativeConfig::is_dspark_algorithm(algorithm);
  const bool is_deepseek_v4_dspark =
      is_dspark && util::is_deepseek_v4_model_type(args.model_type());
  const bool is_deepseek_v41_dspark =
      is_deepseek_v4_dspark &&
      util::is_deepseek_v41_model_type(args.model_type());
  // Model selection and compress-ratio encoding must use the same predicate.
  const bool python_model_impl = ModelConfig::is_python_model_impl(
      ModelConfig::get_instance().model_impl());
  const bool v41_draft =
      is_deepseek_v41_dspark && (python_model_impl || Platform::is_mlu());
  const bool is_python_qwen3_dspark =
      is_dspark && args.model_type() == "qwen3" && Platform::is_npu() &&
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl());
  std::string draft_model_type = std::string(kDFlashDraftModelType);
  if (is_dspark) {
    draft_model_type = std::string(kDSparkDraftModelType);
  } else if (SpeculativeConfig::is_dflash2_algorithm(algorithm)) {
    draft_model_type = std::string(kDFlash2DraftModelType);
    CHECK_GT(args.dflash2_block_size(), 0);
    args.dummy_token_count(args.dflash2_block_size());
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
  args.draft_query_uses_decode(false);
  args.requires_eager_execution(
      (Platform::is_mlu() && v41_draft) ||
      (!is_deepseek_v4_dspark && !is_python_qwen3_dspark));
  if (is_dspark && (python_model_impl || v41_draft)) {
    args.dummy_token_count(
        std::max<int32_t>(1, options.num_speculative_tokens()));
  }
  if (is_deepseek_v4_dspark) {
    bool native_sas = Platform::is_mlu();
#if defined(USE_NPU)
    native_sas = KernelConfig::get_instance().enable_dspark_native_sas() &&
                 !(python_model_impl && is_deepseek_v41_dspark);
#endif
    configure_deepseek_v4_dspark_args(args, options, v41_draft, native_sas);
  }
}

}  // namespace xllm
