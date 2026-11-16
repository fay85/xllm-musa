/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include <torch/extension.h>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/layers/musa/deepseek_v41/markov_head.h"

PYBIND11_MODULE(v41_markov_test_module, module) {
  module.def("bias", &xllm::layer::deepseek_v41_markov_bias);
  module.def("argmax", [](const torch::Tensor& logits) {
    // Match Sampler::greedy_sample: its specialized kernel currently supports
    // large BF16 rows only. FP32 and small rows use the supported Torch path.
    if (xllm::kernel::musa::row_argmax_supported(logits)) {
      return xllm::kernel::musa::row_argmax(logits, torch::Tensor());
    }
    return logits.argmax(/*dim=*/-1);
  });
}
