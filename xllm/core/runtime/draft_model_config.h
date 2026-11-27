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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

struct ModelArgs;
namespace runtime {
class Options;
}

class JsonReader;

// Reads 0-based post-layer capture indices. Missing capture configuration is
// fatal when required; Eagle3 may omit it and use its default capture layers.
std::vector<int32_t> read_capture_layer_ids(
    const std::string& model_weights_path,
    bool required = true);

// Reads the block-diffusion draft's mask token id from a parsed config. The
// checkpoint nests the key under ``text_config`` while the deployment side may
// patch it at the config root through any alias; the root-level aliases are
// listed first so an existing patch keeps winning. Returns -1 when no alias is
// present, for the caller to validate.
int32_t read_mask_token_id(const JsonReader& reader);

void configure_block_diffusion_target(ModelArgs& args,
                                      const runtime::Options& options);

// Apply before model construction so draft layers and dummy inputs agree.
void configure_block_diffusion_draft(ModelArgs& args,
                                     const runtime::Options& options);

void configure_mtp_family_draft(ModelArgs& args,
                                const runtime::Options& options);

}  // namespace xllm
