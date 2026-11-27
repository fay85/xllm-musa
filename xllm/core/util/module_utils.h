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

namespace xllm {

// nn::Module::to(cpu) often leaves NPU CachingAllocator blocks alive because
// Parameter still holds the old Device storage. Copy to host then set_data so
// the NPU TensorImpl is dropped, then empty_cache.
inline void force_module_storage_to_cpu(torch::nn::Module& module) {
  torch::NoGradGuard no_grad;
  auto move_one = [](torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device().is_cpu()) {
      return;
    }
    torch::Tensor host = tensor.to(torch::kCPU).contiguous();
    tensor.set_data(host);
  };
  for (auto& kv : module.named_parameters(/*recurse=*/true)) {
    move_one(kv.value());
  }
  for (auto& kv : module.named_buffers(/*recurse=*/true)) {
    move_one(kv.value());
  }
}

}  // namespace xllm
