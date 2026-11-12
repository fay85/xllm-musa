/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/

#include "core/kernels/musa/deepseek_v41_sparse_attention.h"

#include <glog/logging.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v41_mate_sparse_attention(
    const torch::Tensor& query,
    const torch::Tensor& kv,
    const torch::Tensor& indices,
    const torch::Tensor& sink,
    double scale,
    const std::string& library_path) {
  CHECK(query.defined() && kv.defined() && indices.defined() && sink.defined());
  CHECK(query.device().type() == torch::kPrivateUse1);
  for (const torch::Tensor& value : {kv, indices, sink}) {
    CHECK_EQ(value.device(), query.device());
  }
  CHECK_EQ(query.scalar_type(), torch::kBFloat16);
  CHECK_EQ(kv.scalar_type(), torch::kBFloat16);
  CHECK_EQ(sink.scalar_type(), torch::kFloat32);
  CHECK(indices.scalar_type() == torch::kInt32 ||
        indices.scalar_type() == torch::kInt64);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(kv.dim(), 2);
  CHECK_EQ(indices.dim(), 2);
  CHECK_EQ(sink.dim(), 1);
  CHECK_EQ(query.size(1), 8);
  CHECK_EQ(query.size(2), 512);
  CHECK_EQ(kv.size(1), 512);
  CHECK_EQ(sink.size(0), 8);
  CHECK_EQ(indices.size(0), query.size(0));
  CHECK_GT(kv.size(0), 0);
  CHECK_LE(kv.size(0), std::numeric_limits<int32_t>::max());
  CHECK_GT(indices.size(1), 0);
  CHECK_LE(indices.size(1), 4096);
  CHECK_LE(query.size(0), 512);
  CHECK(std::isfinite(scale) && scale > 0.0);
  CHECK(!library_path.empty());
  torch::DeviceGuard guard(query.device());
  torch::Tensor output = torch::empty(query.sizes(), query.options());
  if (query.size(0) == 0) {
    return output;
  }
  // Normalize all invalid picks before narrowing to int32, including large
  // int64 sentinels. MATE consumes -1 entries; they never read padding NaNs.
  const torch::Tensor valid = indices.ge(0).logical_and(indices.lt(kv.size(0)));
  torch::Tensor picks = torch::where(valid, indices, -1).to(torch::kInt32);
  const int64_t padding = (64 - indices.size(1) % 64) % 64;
  if (padding != 0) {
    picks = torch::cat(
        {picks, torch::full({query.size(0), padding}, -1, picks.options())},
        -1);
  }
  const torch::Tensor q = query.contiguous();
  const torch::Tensor keys = kv.contiguous().unsqueeze(1);
  picks = picks.contiguous().unsqueeze(1);
  const torch::Tensor bias = sink.contiguous();
  const torch::Tensor max_logits =
      torch::empty({query.size(0), 8}, query.options().dtype(torch::kFloat32));
  const torch::Tensor lse = torch::empty_like(max_logits);
  const ffi::Function run = get_function_from_file(library_path, "main");
  // The kernel uses exp2 internally. The sink remains the original logit;
  // its log2 conversion is inside the donor kernel, as in its Python API.
  constexpr double kLog2E = 1.44269504;
  TvmffiStreamGuard stream_guard(query.device());
  run(to_ffi_tensor(q),
      to_ffi_tensor(keys),
      to_ffi_tensor(picks),
      to_ffi_tensor(output),
      to_ffi_tensor(max_logits),
      to_ffi_tensor(lse),
      to_ffi_tensor(bias),
      scale * kLog2E,
      /*persistent_blocks=*/int64_t{0});
  return output;
}

}  // namespace xllm::kernel::musa
