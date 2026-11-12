/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND. See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <c10/core/StreamGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSACachingAllocator.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace xllm {
// Explicit C1 model-forward boundary, not an exception fallback. Every Torch
// and FFI operator sees the same valid non-null compute stream. Input/output
// fences preserve the caller's stream contract; record output storage on its
// future consumer stream before publishing the Tensor.
class MusaModelComputeStream final {
 public:
  explicit MusaModelComputeStream(const torch::Device& device)
      : original_(c10::musa::getCurrentMUSAStream(device.index())),
        compute_(get_compute_stream(device.index())) {
    CHECK(device.type() == torch::kPrivateUse1);
    CHECK_NE(compute_.stream(), nullptr);
    original_.synchronize();
    guard_.emplace(compute_.unwrap());
  }
  MusaModelComputeStream(const MusaModelComputeStream&) = delete;
  MusaModelComputeStream& operator=(const MusaModelComputeStream&) = delete;

  void record_output(const torch::Tensor& output) const {
    CHECK(output.defined());
    CHECK_EQ(output.device(), original_.device());
    c10::musa::MUSACachingAllocator::recordStream(output.storage().data_ptr(),
                                                  original_);
  }

  void finish() {
    CHECK(guard_.has_value());
    compute_.synchronize();
    guard_.reset();
  }

 private:
  static c10::musa::MUSAStream get_compute_stream(c10::DeviceIndex device) {
    static thread_local std::unordered_map<int32_t, c10::musa::MUSAStream>
        streams;
    auto it = streams.find(device);
    if (it == streams.end()) {
      it = streams
               .emplace(device,
                        c10::musa::getStreamFromPool(/*isHighPriority=*/false,
                                                     device))
               .first;
    }
    return it->second;
  }
  c10::musa::MUSAStream original_;
  c10::musa::MUSAStream compute_;
  // If execution throws, only restore the current-stream metadata. The
  // exception remains fatal to model execution; never continue serving from
  // a partially mutated cache or fall back to another implementation.
  std::optional<c10::StreamGuard> guard_;
};
}  // namespace xllm
