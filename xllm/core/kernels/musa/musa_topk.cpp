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
#include <mudnn.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "core/kernels/musa/musa_ops_api.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

void check_mudnn(mudnnStatus_t status, const char* what) {
  CHECK(status == MUDNN_STATUS_SUCCESS)
      << what << " status=" << static_cast<int32_t>(status);
}

void check_musa(musaError_t status, const char* what) {
  CHECK(status == musaSuccess)
      << what << " status=" << static_cast<int32_t>(status);
}

class DeviceBuffer final {
 public:
  DeviceBuffer() = default;

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    if (bytes <= capacity_) {
      return ptr_;
    }
    release();
    check_musa(musaMalloc(&ptr_, bytes), "musaMalloc TopK buffer");
    capacity_ = bytes;
    return ptr_;
  }

  void release() {
    if (ptr_ == nullptr) {
      return;
    }
    check_musa(musaFree(ptr_), "musaFree TopK buffer");
    ptr_ = nullptr;
    capacity_ = 0;
  }

 private:
  void* ptr_ = nullptr;
  size_t capacity_ = 0;
};

class TopKDeviceResources final {
 public:
  explicit TopKDeviceResources(int32_t device_index)
      : device_index_(device_index) {
    c10::musa::MUSAGuard device_guard(device_index_);
    check_mudnn(mudnnCreate(&handle_), "mudnnCreate");
  }

  ~TopKDeviceResources() {
    c10::musa::MUSAGuard device_guard(device_index_);
    input_buffer_.release();
    values_buffer_.release();
    indices_buffer_.release();
    workspace_buffer_.release();
    if (handle_ != nullptr) {
      check_mudnn(mudnnDestroy(handle_), "mudnnDestroy");
      handle_ = nullptr;
    }
  }

  TopKDeviceResources(const TopKDeviceResources&) = delete;
  TopKDeviceResources& operator=(const TopKDeviceResources&) = delete;

  mudnnHandle_t handle_on_current_stream() {
    const musaStream_t stream =
        c10::musa::getCurrentMUSAStream(device_index_).stream();
    check_mudnn(mudnnSetStream(handle_, stream), "mudnnSetStream");
    return handle_;
  }

  DeviceBuffer& input_buffer() { return input_buffer_; }

  DeviceBuffer& values_buffer() { return values_buffer_; }

  DeviceBuffer& indices_buffer() { return indices_buffer_; }

  DeviceBuffer& workspace_buffer() { return workspace_buffer_; }

 private:
  int32_t device_index_;
  mudnnHandle_t handle_ = nullptr;
  DeviceBuffer input_buffer_;
  DeviceBuffer values_buffer_;
  DeviceBuffer indices_buffer_;
  DeviceBuffer workspace_buffer_;
};

TopKDeviceResources& resources_for_device(int32_t device_index) {
  int device_count = 0;
  check_musa(musaGetDeviceCount(&device_count), "musaGetDeviceCount");
  CHECK_GE(device_index, 0) << "MUSA device index must be non-negative";
  CHECK_LT(device_index, device_count)
      << "MUSA device index " << device_index << " exceeds device count "
      << device_count;

  static thread_local std::vector<std::unique_ptr<TopKDeviceResources>>
      resources_by_device;
  if (resources_by_device.size() < static_cast<size_t>(device_count)) {
    resources_by_device.resize(static_cast<size_t>(device_count));
  }

  std::unique_ptr<TopKDeviceResources>& resources =
      resources_by_device[static_cast<size_t>(device_index)];
  if (resources == nullptr) {
    resources = std::make_unique<TopKDeviceResources>(device_index);
  }
  return *resources;
}

void check_topk_indices(const torch::Tensor& input,
                        const torch::Tensor& values,
                        const torch::Tensor& indices) {
  CHECK(indices.scalar_type() == torch::kLong)
      << "TopK indices must be int64, got " << indices.scalar_type();
  CHECK_EQ(indices.dim(), 2) << "TopK indices must be 2-D";
  const int64_t vocab_size = input.size(-1);
  const int64_t k = indices.size(-1);
  const torch::Tensor host_indices = indices.to(torch::kCPU).contiguous();
  const int64_t* data = host_indices.const_data_ptr<int64_t>();
  const int64_t numel = host_indices.numel();
  for (int64_t flat_index = 0; flat_index < numel; ++flat_index) {
    const int64_t token_index = data[flat_index];
    if (token_index >= 0 && token_index < vocab_size) {
      continue;
    }
    float low32_as_float = 0.0f;
    const uint32_t low32 = static_cast<uint32_t>(
        static_cast<uint64_t>(token_index) & 0xffffffffULL);
    std::memcpy(&low32_as_float, &low32, sizeof(low32_as_float));
    const float topk_value =
        values.flatten()[flat_index].to(torch::kCPU).item<float>();
    LOG(FATAL) << "MUSA_TOPK_INDEX_CORRUPTION"
               << " flat_index=" << flat_index << " row=" << (flat_index / k)
               << " column=" << (flat_index % k)
               << " token_index=" << token_index
               << " low32_as_float=" << low32_as_float
               << " topk_value=" << topk_value << " vocab_size=" << vocab_size;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> topk(const torch::Tensor& input,
                                              int64_t k) {
  CHECK(input.defined()) << "TopK input must be defined";
  CHECK(input.device().type() == torch::kMUSA)
      << "software-beam TopK expects a MUSA tensor, got " << input.device();
  const int32_t device_index = static_cast<int32_t>(input.device().index());
  CHECK_GE(device_index, 0) << "software-beam TopK requires a device index";
  c10::musa::MUSAGuard device_guard(device_index);

  int current_device = -1;
  check_musa(musaGetDevice(&current_device), "musaGetDevice");
  CHECK_EQ(current_device, device_index)
      << "software-beam TopK input/current device mismatch";
  CHECK_EQ(input.dim(), 2) << "software-beam TopK expects 2-D [rows, vocab]";
  CHECK(input.scalar_type() == torch::kFloat32)
      << "software-beam TopK expects FP32 logprobs, got "
      << input.scalar_type();
  CHECK_GT(k, 0) << "TopK k must be positive";
  CHECK_LE(k, input.size(1)) << "TopK k exceeds vocab";

  const torch::Tensor contiguous_input = input.contiguous();
  const int64_t rows = contiguous_input.size(0);
  const int64_t cols = contiguous_input.size(1);
  CHECK_LE(rows, std::numeric_limits<int32_t>::max())
      << "software-beam TopK row count exceeds muDNN int32 dimensions";
  CHECK_LE(cols, std::numeric_limits<int32_t>::max())
      << "software-beam TopK vocabulary exceeds muDNN int32 dimensions";
  CHECK_LE(k, std::numeric_limits<int32_t>::max())
      << "software-beam TopK k exceeds muDNN int32 dimensions";
  const size_t input_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
  const size_t values_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(k) * sizeof(float);
  const size_t indices_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(k) * sizeof(int64_t);

  // Wait for every stream before reusing or growing the per-device raw
  // buffers. This makes freeing a previous allocation safe while keeping the
  // buffers outside the graph-aware caching allocator.
  check_musa(musaDeviceSynchronize(), "synchronize before TopK");
  TopKDeviceResources& resources = resources_for_device(device_index);
  void* input_ptr = resources.input_buffer().get(input_bytes);
  void* values_ptr = resources.values_buffer().get(values_bytes);
  void* indices_ptr = resources.indices_buffer().get(indices_bytes);
  check_musa(musaMemcpy(input_ptr,
                        contiguous_input.data_ptr(),
                        input_bytes,
                        musaMemcpyDeviceToDevice),
             "copy TopK input");

  const int32_t input_dims[2] = {static_cast<int32_t>(rows),
                                 static_cast<int32_t>(cols)};
  const int32_t input_strides[2] = {static_cast<int32_t>(cols), 1};
  const int32_t output_dims[2] = {static_cast<int32_t>(rows),
                                  static_cast<int32_t>(k)};
  const int32_t output_strides[2] = {static_cast<int32_t>(k), 1};

  mudnnHandle_t handle = resources.handle_on_current_stream();
  mudnnTensorDescriptor_t input_desc = nullptr;
  mudnnTensorDescriptor_t values_desc = nullptr;
  mudnnTensorDescriptor_t indices_desc = nullptr;
  mudnnTopKDescriptor_t topk_desc = nullptr;
  check_mudnn(mudnnCreateTensorDescriptor(&input_desc), "create input desc");
  check_mudnn(mudnnCreateTensorDescriptor(&values_desc), "create values desc");
  check_mudnn(mudnnCreateTensorDescriptor(&indices_desc),
              "create indices desc");
  check_mudnn(mudnnCreateTopKDescriptor(&topk_desc), "create topk desc");
  check_mudnn(mudnnSetTensorNdDescriptor(input_desc,
                                         MUDNN_DATA_FLOAT,
                                         /*nbDims=*/2,
                                         input_dims,
                                         input_strides),
              "set input desc");
  check_mudnn(mudnnSetTensorNdDescriptor(values_desc,
                                         MUDNN_DATA_FLOAT,
                                         /*nbDims=*/2,
                                         output_dims,
                                         output_strides),
              "set values desc");
  check_mudnn(mudnnSetTensorNdDescriptor(indices_desc,
                                         MUDNN_DATA_INT64,
                                         /*nbDims=*/2,
                                         output_dims,
                                         output_strides),
              "set indices desc");
  check_mudnn(mudnnSetTopKDescriptor(topk_desc,
                                     static_cast<int32_t>(k),
                                     /*dim=*/1,
                                     /*largest=*/true,
                                     /*sorted=*/true),
              "set topk desc");

  size_t workspace_bytes = 0;
  check_mudnn(mudnnGetTopKWorkspaceSize(handle,
                                        topk_desc,
                                        input_desc,
                                        values_desc,
                                        indices_desc,
                                        &workspace_bytes),
              "mudnnGetTopKWorkspaceSize");
  void* workspace = resources.workspace_buffer().get(workspace_bytes);
  const mudnnStatus_t run_status = mudnnTopK(handle,
                                             topk_desc,
                                             input_desc,
                                             input_ptr,
                                             workspace,
                                             workspace_bytes,
                                             values_desc,
                                             values_ptr,
                                             indices_desc,
                                             indices_ptr);
  check_mudnn(run_status, "mudnnTopK");
  check_musa(musaDeviceSynchronize(), "synchronize after TopK");
  mudnnDestroyTopKDescriptor(topk_desc);
  mudnnDestroyTensorDescriptor(indices_desc);
  mudnnDestroyTensorDescriptor(values_desc);
  mudnnDestroyTensorDescriptor(input_desc);

  torch::Tensor values = torch::empty({rows, k},
                                      contiguous_input.options().memory_format(
                                          torch::MemoryFormat::Contiguous));
  torch::Tensor indices =
      torch::empty({rows, k},
                   contiguous_input.options()
                       .dtype(torch::kLong)
                       .memory_format(torch::MemoryFormat::Contiguous));
  check_musa(musaMemcpy(values.data_ptr(),
                        values_ptr,
                        values_bytes,
                        musaMemcpyDeviceToDevice),
             "copy TopK values");
  check_musa(musaMemcpy(indices.data_ptr(),
                        indices_ptr,
                        indices_bytes,
                        musaMemcpyDeviceToDevice),
             "copy TopK indices");
  check_musa(musaDeviceSynchronize(), "synchronize TopK outputs");

  check_topk_indices(contiguous_input, values, indices);
  return {values, indices};
}

}  // namespace xllm::kernel::musa
