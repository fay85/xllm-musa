/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace xllm::kernel::musa {

// Read-only q1 construction. Returns chronological rows, sparse picks, and
// the physical commit slot; the caller must run attention before writing KV.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deepseek_v41_swa_decode_read(const torch::Tensor& cache,
                             const torch::Tensor& current_kv,
                             const torch::Tensor& table,
                             const torch::Tensor& positions,
                             int64_t window,
                             const std::optional<torch::Tensor>& compressed,
                             const std::optional<torch::Tensor>& extra);

}  // namespace xllm::kernel::musa
