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

#pragma once

#include <glog/logging.h>
#include <mccl.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <type_traits>
#include <vector>

namespace xllm {

// Like SGLang's PyNccl graph path, enqueue on the caller's stream without a
// ProcessGroupMCCL Work/event/watchdog. The communicator is initialized and
// warmed before capture. This preserves stream ordering, not host completion.
// The owner must keep this object alive until its captured graphs are released.
class MusaNativeCollectives final {
 public:
  MusaNativeCollectives(int32_t rank,
                        int32_t world_size,
                        const std::string& group_name,
                        const torch::Device& device,
                        const c10::intrusive_ptr<c10d::Store>& store)
      : world_size_(world_size), device_(device) {
    CHECK_GT(world_size_, 0);
    CHECK_GE(rank, 0);
    CHECK_LT(rank, world_size_);
    CHECK_EQ(device_.type(), torch::kPrivateUse1);
    torch::DeviceGuard guard(device_);
    musaStreamCaptureStatus capture_status;
    CHECK_EQ(musaStreamIsCapturing(stream(), &capture_status), musaSuccess);
    CHECK_EQ(capture_status, musaStreamCaptureStatusNone)
        << "Native MCCL must be initialized outside graph capture";
    if (world_size_ == 1) {
      return;
    }
    mcclUniqueId id{};
    const std::string key = "xllm/native_mccl/" + group_name + "/unique_id";
    if (rank == 0) {
      check(mcclGetUniqueId(&id));
      std::vector<uint8_t> bytes(sizeof(id));
      std::memcpy(bytes.data(), &id, sizeof(id));
      store->set(key, bytes);
    } else {
      const auto bytes = store->get(key);
      CHECK_EQ(bytes.size(), sizeof(id));
      std::memcpy(&id, bytes.data(), sizeof(id));
    }
    mcclComm_t comm = nullptr;
    check(mcclCommInitRank(&comm, world_size_, id, rank));
    comm_.reset(comm);
    torch::Tensor warmup = torch::zeros(
        {1}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    allreduce(warmup);
    c10::musa::getCurrentMUSAStream(device_.index()).synchronize();
    LOG(INFO) << "Native synchronous MCCL initialized for " << group_name
              << ", local rank " << rank << "/" << world_size_;
  }

  ~MusaNativeCollectives() {
    torch::DeviceGuard guard(device_);
    comm_.reset();
  }

  MusaNativeCollectives(const MusaNativeCollectives&) = delete;
  MusaNativeCollectives& operator=(const MusaNativeCollectives&) = delete;

  void allreduce(torch::Tensor& input) const {
    validate(input);
    if (world_size_ == 1 || input.numel() == 0) {
      return;
    }
    torch::DeviceGuard guard(device_);
    check(mcclAllReduce(input.data_ptr(),
                        input.data_ptr(),
                        input.numel(),
                        datatype(input),
                        mcclSum,
                        comm_.get(),
                        stream()));
  }

  torch::Tensor allgather(const torch::Tensor& input) const {
    torch::DeviceGuard guard(device_);
    torch::Tensor contiguous = input.contiguous();
    validate(contiguous);
    std::vector<int64_t> shape;
    shape.reserve(input.dim() + 1);
    shape.emplace_back(world_size_);
    shape.insert(shape.end(), input.sizes().begin(), input.sizes().end());
    torch::Tensor output = torch::empty(shape, input.options());
    if (world_size_ == 1) {
      output[0].copy_(contiguous);
      return output;
    }
    if (input.numel() != 0) {
      check(mcclAllGather(contiguous.data_ptr(),
                          output.data_ptr(),
                          contiguous.numel(),
                          datatype(contiguous),
                          comm_.get(),
                          stream()));
    }
    return output;
  }

  void broadcast(torch::Tensor& input, int32_t root_rank) const {
    validate(input);
    CHECK_GE(root_rank, 0);
    CHECK_LT(root_rank, world_size_);
    if (world_size_ == 1 || input.numel() == 0) {
      return;
    }
    torch::DeviceGuard guard(device_);
    check(mcclBroadcast(input.data_ptr(),
                        input.data_ptr(),
                        input.numel(),
                        datatype(input),
                        root_rank,
                        comm_.get(),
                        stream()));
  }

  void reduce_scatter(const torch::Tensor& input, torch::Tensor& output) const {
    validate(input);
    validate(output);
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_EQ(input.numel(), world_size_ * output.numel());
    torch::DeviceGuard guard(device_);
    if (world_size_ == 1) {
      output.copy_(input.view(output.sizes()));
      return;
    }
    if (output.numel() != 0) {
      check(mcclReduceScatter(input.data_ptr(),
                              output.data_ptr(),
                              output.numel(),
                              datatype(input),
                              mcclSum,
                              comm_.get(),
                              stream()));
    }
  }

 private:
  class CommDeleter final {
   public:
    void operator()(mcclComm_t comm) const { check(mcclCommDestroy(comm)); }
  };

  static void check(mcclResult_t result) {
    CHECK_EQ(result, mcclSuccess) << mcclGetErrorString(result);
  }

  static mcclDataType_t datatype(const torch::Tensor& input) {
    switch (input.scalar_type()) {
      case torch::kFloat32:
        return mcclFloat32;
      case torch::kFloat16:
        return mcclFloat16;
      case torch::kBFloat16:
        return mcclBfloat16;
      case torch::kFloat64:
        return mcclFloat64;
      case torch::kInt32:
        return mcclInt32;
      case torch::kInt64:
        return mcclInt64;
      case torch::kInt8:
        return mcclInt8;
      case torch::kUInt8:
        return mcclUint8;
      default:
        LOG(FATAL) << "Unsupported native MCCL datatype "
                   << input.scalar_type();
    }
    return mcclFloat32;
  }

  void validate(const torch::Tensor& tensor) const {
    CHECK(tensor.defined());
    CHECK_EQ(tensor.device(), device_);
    CHECK(tensor.is_contiguous());
  }

  musaStream_t stream() const {
    return c10::musa::getCurrentMUSAStream(device_.index()).stream();
  }

  const int32_t world_size_;
  const torch::Device device_;
  std::unique_ptr<std::remove_pointer_t<mcclComm_t>, CommDeleter> comm_;
};

}  // namespace xllm
