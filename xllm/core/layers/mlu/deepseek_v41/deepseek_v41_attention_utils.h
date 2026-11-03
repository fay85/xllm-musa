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

namespace xllm::layer::deepseek_v41 {
inline torch::Tensor norm(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          double eps) {
  torch::Tensor value = x.to(torch::kFloat32);
  return (value * torch::rsqrt(value.square().mean(-1, true) + eps) *
          weight.to(torch::kFloat32))
      .to(x.scalar_type());
}

inline torch::Tensor dense_projection(const torch::Tensor& x,
                                      const torch::Tensor& w) {
  return torch::matmul(x, w.t());
}

// The cache contains repeated cos and sin values for interleaved pairs.
inline torch::Tensor rotate(torch::Tensor x,
                            const torch::Tensor& positions,
                            const torch::Tensor& cache,
                            int64_t dim,
                            bool inverse = false) {
  torch::Tensor cs = cache.index_select(0, positions.to(torch::kInt64));
  torch::Tensor cos = cs.slice(-1, 0, dim);
  torch::Tensor sin = cs.slice(-1, dim, 2 * dim);
  if (x.dim() == 3) {
    cos = cos.unsqueeze(1);
    sin = sin.unsqueeze(1);
  }
  torch::Tensor tail = x.slice(-1, x.size(-1) - dim).to(torch::kFloat32);
  torch::Tensor pairs = tail.reshape({-1, dim / 2, 2});
  torch::Tensor swapped =
      torch::stack({-pairs.select(-1, 1), pairs.select(-1, 0)}, -1)
          .reshape_as(tail);
  tail = tail * cos + swapped * (inverse ? -sin : sin);
  x.slice(-1, x.size(-1) - dim).copy_(tail.to(x.scalar_type()));
  return x;
}
}  // namespace xllm::layer::deepseek_v41
