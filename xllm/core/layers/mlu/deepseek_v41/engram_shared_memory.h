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

#include <torch/torch.h>

#include <string>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {

// Collective over the model's world group at weight-load time. Ranks on each
// host publish disjoint source row ranges into one full byte table, independent
// of the attention DP/TP layout. The returned tensor owns its mapping and file
// descriptor, independently of the checkpoint.
torch::Tensor load_shared_engram_tensor(const torch::Tensor& source,
                                        const std::string& name,
                                        ProcessGroup& group);

}  // namespace xllm::layer
