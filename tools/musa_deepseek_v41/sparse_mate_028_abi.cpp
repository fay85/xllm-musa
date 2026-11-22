/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapt the nine-argument xLLM V4.1 sparse ABI to MATE v0.2.8.
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

namespace {

namespace ffi = tvm::ffi;

int64_t kv_byte_span(ffi::TensorView kv) {
  const DLDataType dtype = kv.dtype();
  if (dtype.code != kDLBfloat || dtype.bits != 16 || dtype.lanes != 1) {
    TVM_FFI_THROW(ValueError) << "MATE V4.1 sparse KV must contain BF16 values";
  }
  if (kv.ndim() != 3 || kv.size(1) != 1 || kv.size(2) != 512 ||
      kv.size(0) <= 0 || kv.stride(2) != 1 || kv.stride(0) < kv.size(2)) {
    TVM_FFI_THROW(ValueError)
        << "MATE V4.1 sparse KV must be a nonempty positive-strided MQA view";
  }
  const int64_t rows = kv.size(0);
  const int64_t stride = kv.stride(0);
  constexpr int64_t kBf16Bytes = 2;
  constexpr int64_t kMaxElements =
      std::numeric_limits<int64_t>::max() / kBf16Bytes;
  if (rows - 1 > (kMaxElements - kv.size(2)) / stride) {
    TVM_FFI_THROW(ValueError) << "MATE V4.1 sparse KV span exceeds int64 bytes";
  }
  return ((rows - 1) * stride + kv.size(2)) * kBf16Bytes;
}

const ffi::Function& get_provider() {
  static const int32_t kModuleAnchor = 0;
  static const ffi::Module module = [] {
    Dl_info info{};
    if (dladdr(&kModuleAnchor, &info) == 0 || info.dli_fname == nullptr) {
      TVM_FFI_THROW(RuntimeError)
          << "Cannot resolve the installed MATE sparse ABI adapter";
    }
    const auto path = std::filesystem::path(info.dli_fname).parent_path() /
                      "libmate028_sparse_provider.so";
    return ffi::Module::LoadFromFile(path.string());
  }();
  static const ffi::Function run = module->GetFunction("main").value();
  return run;
}

void sparse_v41(ffi::TensorView query,
                ffi::TensorView kv,
                ffi::TensorView indices,
                ffi::TensorView output,
                ffi::TensorView max_logits,
                ffi::TensorView lse,
                ffi::TensorView sink,
                double scale_log2,
                int64_t persistent_blocks) {
  get_provider()(query,
                 kv,
                 indices,
                 output,
                 max_logits,
                 lse,
                 sink,
                 scale_log2,
                 persistent_blocks,
                 kv_byte_span(kv));
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(main, sparse_v41);
