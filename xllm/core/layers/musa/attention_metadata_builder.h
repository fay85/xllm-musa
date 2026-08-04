/* Copyright 2025-2026 The xLLM Authors.

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

#include <optional>

namespace xllm {

struct ModelInputParams;

namespace layer {

struct AttentionMetadata;

namespace musa {

// Populate metadata needed only by MUSA attention and hybrid GDN layers.
void populate_attention_metadata(AttentionMetadata& attn_metadata,
                                 const ModelInputParams& params,
                                 const std::optional<torch::Tensor>& attn_mask);

// Finalize fields that depend on common causal/graph flags.
void finalize_attention_metadata(AttentionMetadata& attn_metadata);

}  // namespace musa
}  // namespace layer
}  // namespace xllm
