/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <optional>

namespace xllm::kernel::musa {
torch::Tensor deepseek_v41_engram_gate(
    const torch::Tensor& input,
    const torch::Tensor& kv,
    const torch::Tensor& qw,
    const torch::Tensor& kw,
    const std::optional<torch::Tensor>& mask = std::nullopt);
}  // namespace xllm::kernel::musa
