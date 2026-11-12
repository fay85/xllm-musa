/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <string>

namespace xllm::kernel::musa {

// Launch a pinned offline-built MUSA binary on the current Torch stream.
// The manifest supplies the exported symbol, block size and shared memory.
void launch_musa_aot_kernel(const torch::Device& device,
                            const std::string& directory,
                            const std::string& key,
                            const std::array<uint32_t, 3>& grid,
                            void** arguments);

}  // namespace xllm::kernel::musa
