/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <tuple>
#include <glog/logging.h>

#include "core/kernels/musa/deepseek_v4_hc_policy.h"
#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace xllm::layer::dsv4_musa_hc {

inline bool tiled_prefill_enabled() {
  const char* value = std::getenv("XLLM_DSV4_HC_TILED_PREFILL");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool tiled_prefill_shape(int64_t rows,
                                int64_t hidden_dim,
                                int64_t row_limit,
                                int64_t hc_mult,
                                int64_t sinkhorn_iters,
                                bool enabled) {
  // Only the two local SP chunks of a 16K Prefill. Decode and draft keep
  // their existing routes, as do models outside this qualified HC shape.
  return enabled && rows > 1024 && rows <= 2048 && hidden_dim == 4096 &&
         row_limit == 1024 && hc_mult == 4 && sinkhorn_iters == 20;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tiled_prefill(
    const torch::Tensor& x,
    const torch::Tensor& hc_fn,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double norm_eps,
    double hc_eps) {
  const int64_t rows = x.size(0);
  const int64_t hidden_dim = x.size(2);
  const int64_t row_limit =
      kernel::musa::deepseek_v4_hc_prefill_row_limit(hidden_dim);
  CHECK(tiled_prefill_shape(rows,
                           hidden_dim,
                           row_limit,
                           hc_mult,
                           sinkhorn_iters,
                           /*enabled=*/true));
  auto output = torch::empty({rows, hidden_dim}, x.options());
  auto post = torch::empty({rows, hc_mult}, hc_scale.options());
  auto comb = torch::empty({rows, hc_mult, hc_mult}, hc_scale.options());
  // Keep the original <=1024-row kernel, weight dtype and arithmetic.
  // Views write directly into final storage. Avoid a one-row tail: the
  // single-token HC finish uses a different reduction order than Prefill.
  // Norm is deliberately separate, matching the original long Prefill path.
  const int64_t tile_rows = rows == row_limit + 1 ? row_limit - 1 : row_limit;
  for (int64_t start = 0; start < rows; start += tile_rows) {
    const int64_t count = std::min(tile_rows, rows - start);
    kernel::musa::deepseek_v4_hc_pre_decode(
        x.narrow(0, start, count),
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        output.narrow(0, start, count),
        post.narrow(0, start, count),
        comb.narrow(0, start, count),
        /*norm_weight=*/std::nullopt,
        /*output_is_normalized=*/nullptr);
  }
  return std::make_tuple(output, post, comb);
}

}  // namespace xllm::layer::dsv4_musa_hc
