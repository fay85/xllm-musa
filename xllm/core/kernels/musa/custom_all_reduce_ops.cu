/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/musa/custom_all_reduce.cuh"

namespace {

int64_t meta_size() { return static_cast<int64_t>(sizeof(vllm::Signal)); }

std::tuple<int64_t, at::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  int32_t device_index = 0;
  CUDACHECK(cudaGetDevice(&device_index));
  const c10::cuda::CUDAGuard guard(device_index);
  void* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, static_cast<size_t>(size)));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  CUDACHECK(cudaMemsetAsync(buffer, 0, static_cast<size_t>(size), stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  at::Tensor handle =
      at::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  CUDACHECK(cudaIpcGetMemHandle(
      reinterpret_cast<cudaIpcMemHandle_t*>(handle.data_ptr()), buffer));
  return {reinterpret_cast<int64_t>(buffer), handle};
}

int64_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr = nullptr;
  CUDACHECK(cudaIpcOpenMemHandle(
      &ipc_ptr,
      *reinterpret_cast<const cudaIpcMemHandle_t*>(mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ipc_ptr);
}

void free_shared_buffer(int64_t ptr) {
  CUDACHECK(cudaFree(reinterpret_cast<void*>(ptr)));
}

int64_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                       torch::Tensor& rank_data,
                       int64_t rank,
                       bool fully_connected) {
  int32_t world_size = static_cast<int32_t>(fake_ipc_ptrs.size());
  TORCH_CHECK(world_size <= vllm::kMaxCustomCollectiveRanks);
  TORCH_CHECK(world_size % 2 == 0);
  TORCH_CHECK(rank >= 0 && rank < world_size);
  vllm::Signal* ipc_ptrs[vllm::kMaxCustomCollectiveRanks];
  for (int32_t i = 0; i < world_size; ++i) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  auto* collective = new vllm::CustomAllreduce(
      ipc_ptrs,
      rank_data.data_ptr(),
      static_cast<size_t>(rank_data.numel() * rank_data.element_size()),
      static_cast<int>(rank),
      world_size,
      fully_connected);
  return reinterpret_cast<int64_t>(collective);
}

void set_algorithm(int64_t handle, int64_t algorithm) {
  reinterpret_cast<vllm::CustomAllreduce*>(handle)->set_algorithm(
      static_cast<int>(algorithm));
}

void register_buffer(int64_t handle,
                     const std::vector<int64_t>& fake_ipc_ptrs) {
  auto* collective = reinterpret_cast<vllm::CustomAllreduce*>(handle);
  TORCH_CHECK(static_cast<int32_t>(fake_ipc_ptrs.size()) ==
              collective->world_size_);
  void* ipc_ptrs[vllm::kMaxCustomCollectiveRanks];
  for (int32_t i = 0; i < static_cast<int32_t>(fake_ipc_ptrs.size()); ++i) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  collective->register_buffer(ipc_ptrs);
}

void all_reduce(int64_t handle,
                torch::Tensor& input,
                torch::Tensor& output,
                int64_t registered_buffer,
                int64_t registered_buffer_size) {
  auto* collective = reinterpret_cast<vllm::CustomAllreduce*>(handle);
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(input.scalar_type() == output.scalar_type());
  TORCH_CHECK(input.numel() == output.numel());
  void* buffer = reinterpret_cast<void*>(registered_buffer);
  const int64_t input_bytes = input.numel() * input.element_size();
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CUDACHECK(cudaStreamIsCapturing(stream, &capture_status));
  const bool graph_capture = capture_status == cudaStreamCaptureStatusActive;
  if (buffer != nullptr && !graph_capture) {
    TORCH_CHECK(input_bytes <= registered_buffer_size);
    CUDACHECK(cudaMemcpyAsync(buffer,
                              input.data_ptr(),
                              static_cast<size_t>(input_bytes),
                              cudaMemcpyDeviceToDevice,
                              stream));
  } else if (buffer == nullptr) {
    buffer = input.data_ptr();
  }
  // During graph capture, register and reduce the real graph input directly.
  // The graph-buffer exchange resolves its peer IPC addresses after capture, so
  // staging the input would only add a device-to-device copy node.
  void* collective_input = graph_capture ? input.data_ptr() : buffer;
  const int32_t numel = static_cast<int32_t>(output.numel());
  switch (output.scalar_type()) {
    case at::kFloat:
      collective->allreduce<float>(stream,
                                   reinterpret_cast<float*>(collective_input),
                                   output.data_ptr<float>(),
                                   numel);
      break;
    case at::kHalf:
      collective->allreduce<half>(stream,
                                  reinterpret_cast<half*>(collective_input),
                                  reinterpret_cast<half*>(output.data_ptr()),
                                  numel);
      break;
    case at::kBFloat16:
      collective->allreduce<nv_bfloat16>(
          stream,
          reinterpret_cast<nv_bfloat16*>(collective_input),
          reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
          numel);
      break;
    default:
      TORCH_CHECK(false, "custom all-reduce dtype is not supported");
  }
}

void dispose(int64_t handle) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(handle);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(int64_t handle) {
  auto* collective = reinterpret_cast<vllm::CustomAllreduce*>(handle);
  auto [ipc_handles, offsets] = collective->get_graph_buffer_ipc_meta();
  std::vector<int64_t> handle_bytes(ipc_handles.begin(), ipc_handles.end());
  return {std::move(handle_bytes), std::move(offsets)};
}

void register_graph_buffers(int64_t handle,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto* collective = reinterpret_cast<vllm::CustomAllreduce*>(handle);
  std::vector<std::string> handle_bytes;
  handle_bytes.reserve(handles.size());
  for (const std::vector<int64_t>& one_rank : handles) {
    std::string bytes(one_rank.size(), '\0');
    for (size_t i = 0; i < one_rank.size(); ++i) {
      bytes[i] = static_cast<char>(one_rank[i]);
    }
    handle_bytes.push_back(std::move(bytes));
  }
  collective->register_graph_buffers(handle_bytes, offsets);
}

}  // namespace

namespace xllm {
void register_musa_custom_all_reduce_kernel() {}
}  // namespace xllm

TORCH_LIBRARY(_C_custom_ar, m) {
  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.def("dispose(int fa) -> ()");
  m.def("meta_size() -> int");
  m.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
  m.def("get_graph_buffer_ipc_meta(int fa) -> (int[], int[])");
  m.def(
      "register_graph_buffers(int fa, int[][] handles, int[][] offsets) -> ()");
  m.def("allocate_shared_buffer_and_handle(int size) -> (int, Tensor)");
  m.def("open_mem_handle(Tensor mem_handle) -> int");
  m.def("free_shared_buffer(int ptr) -> ()");
  m.def("set_algorithm(int fa, int algorithm) -> ()");
}

TORCH_LIBRARY_IMPL(_C_custom_ar, PrivateUse1, m) {
  m.impl("init_custom_ar", init_custom_ar);
  m.impl("all_reduce", all_reduce);
}

TORCH_LIBRARY_IMPL(_C_custom_ar, CPU, m) {
  m.impl("open_mem_handle", open_mem_handle);
}

TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, m) {
  m.impl("dispose", dispose);
  m.impl("meta_size", meta_size);
  m.impl("register_buffer", register_buffer);
  m.impl("get_graph_buffer_ipc_meta", get_graph_buffer_ipc_meta);
  m.impl("register_graph_buffers", register_graph_buffers);
  m.impl("allocate_shared_buffer_and_handle",
         allocate_shared_buffer_and_handle);
  m.impl("free_shared_buffer", free_shared_buffer);
  m.impl("set_algorithm", set_algorithm);
}
