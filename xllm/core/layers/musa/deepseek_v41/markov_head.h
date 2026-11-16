/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

namespace xllm::layer {

// Keep the checkpoint's BF16 embedding/projection. Promoting the result before
// the caller adds base logits follows SGLang's V4 DSpark project_bias and
// BuildStepLocal contract; promoting after a BF16 add loses near-tie precision.
inline torch::Tensor deepseek_v41_markov_bias(const torch::Tensor& previous,
                                              const torch::Tensor& embedding,
                                              const torch::Tensor& head,
                                              bool fp32_logits) {
  CHECK_EQ(previous.dim(), 1);
  CHECK_EQ(embedding.dim(), 2);
  CHECK_EQ(head.sizes(), embedding.sizes());
  CHECK_EQ(embedding.scalar_type(), torch::kBFloat16);
  CHECK_EQ(head.scalar_type(), torch::kBFloat16);
  CHECK_EQ(previous.device(), embedding.device());
  CHECK_EQ(head.device(), embedding.device());
  const torch::Tensor bias = torch::nn::functional::linear(
      torch::embedding(embedding, previous.to(torch::kInt64)), head);
  return fp32_logits ? bias.to(torch::kFloat32) : bias;
}

}  // namespace xllm::layer
