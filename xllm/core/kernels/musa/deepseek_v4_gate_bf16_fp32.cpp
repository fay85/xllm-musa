/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <tvm/ffi/container/tuple.h>

#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

torch::Tensor deepseek_v4_gate_bf16_fp32(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& output,
    const std::string& uri) {
  if (!input.defined() || !weight.defined() || input.dim() != 2 ||
      weight.dim() != 2 || input.scalar_type() != torch::kBFloat16 ||
      weight.scalar_type() != torch::kBFloat16 ||
      input.size(1) != weight.size(1) || input.device() != weight.device()) {
    return torch::Tensor();
  }

  const int64_t m = input.size(0);
  const int64_t n = weight.size(0);
  const int64_t k = input.size(1);
  torch::Tensor out;
  if (output.has_value() && output.value().defined()) {
    out = output.value();
    if (out.dim() != 2 || out.size(0) != m || out.size(1) != n ||
        out.scalar_type() != torch::kFloat32 ||
        out.device() != input.device()) {
      return torch::Tensor();
    }
  } else {
    out = torch::empty({m, n}, input.options().dtype(torch::kFloat32));
  }
  if (m == 0 || n == 0 || k == 0) {
    return out;
  }

  // Avoid get_function's fatal missing-symbol diagnostic for installations
  // that only carry the generic FlashInfer bundle. Missing Mate artifacts are
  // a normal fallback condition for the model layer.
  const std::string so_path = path_to_uri_so_lib(uri);
  if (!std::ifstream(so_path).good()) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_GATE] Mate BF16->FP32 GEMM unavailable: " << so_path;
    return torch::Tensor();
  }

  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  auto out_contig = out.contiguous();
  auto input_batched = input_contig.unsqueeze(0);
  auto weight_batched = weight_contig.unsqueeze(0);
  auto output_batched = out_contig.unsqueeze(0);

  try {
    // The model uses the process-global gemm_ops bundle on every one of its
    // routed-MoE layers.  Resolve the default symbol once per worker process
    // instead of taking the global module-cache mutex and searching the
    // exported function table on every layer invocation.  Preserve the
    // dynamic-URI path for standalone callers and keep a missing symbol as a
    // normal model fallback condition.
    const tvm::ffi::Function run = [&]() {
      if (uri == "gemm_ops") {
        static const tvm::ffi::Optional<tvm::ffi::Function> default_run = [] {
          return get_module("gemm_ops")->GetFunction("bmm");
        }();
        if (!default_run.defined()) {
          throw std::runtime_error("gemm_ops does not export MATE 0.2.7 bmm");
        }
        return default_run.value();
      }
      auto run_opt = get_module(uri)->GetFunction("bmm");
      if (!run_opt.defined()) {
        throw std::runtime_error(uri + " does not export MATE 0.2.7 bmm");
      }
      return run_opt.value();
    }();
    const auto invoke = [&]() {
      run(to_ffi_tensor(input_batched),
          to_ffi_tensor(weight_batched),
          to_ffi_tensor(output_batched),
          to_ffi_optional_tensor(std::nullopt),
          to_ffi_optional_tensor(std::nullopt),
          to_ffi_optional_tensor(std::nullopt),
          tvm::ffi::Tuple<int64_t, int64_t>(-1, -1),
          tvm::ffi::Tuple<int64_t, int64_t>(-1, -1),
          /*trans_a=*/false,
          /*trans_b=*/true,
          /*fixed_scale_layout=*/false);
    };
    // The router projection is immediately consumed by fused TopK and has no
    // independent work to overlap.  Run it on Torch's current MUSA stream so
    // every routed-MoE layer avoids a current->pooled->current pair of events.
    // bind_musa_tvmffi_current_stream handles the valid null handle used by
    // MUSA's default stream.
    bind_tvmffi_stream_to_current_torch_stream(input.device());
    try {
      invoke();
    } catch (...) {
      sync_current_stream(input.device());
      throw;
    }
  } catch (const c10::Error& error) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_GATE] Mate BF16->FP32 GEMM fallback: "
        << error.what_without_backtrace();
    return torch::Tensor();
  } catch (const std::exception& error) {
    LOG_FIRST_N(WARNING, 1)
        << "[DSV4_MUSA_GATE] Mate BF16->FP32 GEMM fallback: " << error.what();
    return torch::Tensor();
  }

  if (!out.is_contiguous()) {
    out.copy_(out_contig);
  }
  LOG_FIRST_N(INFO, 1) << "[DSV4_MUSA_GATE] Mate 0.2.7 bmm active: M=" << m
                       << " N=" << n << " K=" << k << " output=fp32";
  return out;
}

}  // namespace xllm::kernel::musa
