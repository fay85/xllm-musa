/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/kernels/scalar_type_dispatch.h"

namespace ffi = tvm::ffi;

namespace xllm::kernel::musa {

// MATE 0.2.7 metadata consists of four arrays, each padded to a multiple of 4.
inline int64_t fa3_scheduler_metadata_size(int64_t batch_size) {
  return ((batch_size + 3) / 4) * 16;
}

// Resolved once; set FLASHINFER_OPS_PATH before the first operator lookup.
std::string get_musa_ops_root();

// Verify the complete bundle once per root, before loading any device code.
// Deployed bundles must remain immutable for the lifetime of the process.
void validate_musa_ops_bundle(const std::string& root);

// Query the compiled inventory without loading or executing device code.
bool is_musa_operator_built(const std::string& uri);

inline bool is_torch_device(const torch::Device& device) {
  return device.is_privateuseone() || device.is_cuda();
}

void bind_tvmffi_stream(const torch::Device& device);

bool is_stream_capturing();

void sync_current_stream(const torch::Device& device);

void sync_ffi_stream(const torch::Device& device);

void sync_graph_preparation_stage(const torch::Device& device);

class TvmffiPreparationSyncGuard final {
 public:
  TvmffiPreparationSyncGuard();
  ~TvmffiPreparationSyncGuard();

  TvmffiPreparationSyncGuard(const TvmffiPreparationSyncGuard&) = delete;
  TvmffiPreparationSyncGuard& operator=(const TvmffiPreparationSyncGuard&) =
      delete;

 private:
  bool previous_ = false;
};

// During graph capture, replaces only an invalid current MUSA stream with the
// capture stream. Individual FFI operators still rebind their stream handle.
class TvmffiStreamOverrideGuard final {
 public:
  TvmffiStreamOverrideGuard(const torch::Device& device, void* stream);
  ~TvmffiStreamOverrideGuard();

  TvmffiStreamOverrideGuard(const TvmffiStreamOverrideGuard&) = delete;
  TvmffiStreamOverrideGuard& operator=(const TvmffiStreamOverrideGuard&) =
      delete;

 private:
  torch::Device device_;
  bool active_ = false;
  void* previous_forced_stream_ = nullptr;
};

class TvmffiStreamGuard final {
 public:
  explicit TvmffiStreamGuard(const torch::Device& device);
  ~TvmffiStreamGuard();

  TvmffiStreamGuard(const TvmffiStreamGuard&) = delete;
  TvmffiStreamGuard& operator=(const TvmffiStreamGuard&) = delete;

 private:
  torch::Device device_;
  bool active_ = false;
  // True when the FFI kernel was bound to the pool stream (eager-mode
  // fallback). In that case the destructor must establish ordering from the
  // FFI stream to the compute stream before subsequent PyTorch operations.
  bool needs_sync_ = false;
  bool uses_event_handoff_ = false;
};

std::string path_to_uri_so_lib(const std::string& uri);

DLDataType to_dl_data_type(torch::ScalarType scalar_type);

ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor);

// Creates an FFI tensor that owns only its DLPack metadata.  The caller must
// keep the Torch storage alive until the launched device work completes.
ffi::Tensor to_ffi_borrowed_tensor(const torch::Tensor& torch_tensor);

ffi::TensorView to_ffi_tensor_view(const torch::Tensor& torch_tensor);

ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional);

ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors);

ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional);

ffi::Module get_module(const std::string& uri);

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name);

// Registers TileLang's embedded MUSA-module loader with TVM FFI. Returns false
// when libtilelang is unavailable so callers can use a non-TileLang fallback.
bool ensure_tilelang_loader();

enum class FfiAllocMode { kPassthrough, kRecord, kReplay };

void begin_ffi_alloc_record();

std::vector<torch::Tensor> end_ffi_alloc_record();

void begin_ffi_alloc_replay(const std::vector<torch::Tensor>* recorded);

void end_ffi_alloc_replay();

FfiAllocMode get_ffi_alloc_mode();

void bind_tvmffi_stream_to_current_torch_stream(const torch::Device& device);
}  // namespace xllm::kernel::musa
