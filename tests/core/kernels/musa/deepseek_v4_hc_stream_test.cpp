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

#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAGuard.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <chrono>
#include <fstream>
#include <thread>
#include <tuple>

#include "core/kernels/musa/musa_dsv4_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

void delay_ffi_producer(void* /*data*/) {
  // Make the producer/consumer ordering deterministic without device resets
  // or timing-dependent assertions. No MUSA API is called in this callback.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST(DeepseekV4HcStreamTest, DefaultStreamWaitsForPrenormBeforeNativeFinish) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    GTEST_SKIP() << "MUSA device unavailable";
  }
  if (!std::ifstream(path_to_uri_so_lib("mhc_prenorm_ops")).good()) {
    GTEST_SKIP() << "Mate HC prenorm artifact unavailable";
  }
  at::detail::getMUSAHooks().initMUSA();
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const c10::musa::MUSAStream reference_stream =
      c10::musa::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  const c10::musa::MUSAStream default_stream =
      c10::musa::getDefaultMUSAStream(/*device_index=*/0);
  c10::StreamGuard reference_guard(reference_stream.unwrap());
  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor x = (torch::arange(4 * 4096, cpu_options) * 0.0031)
                        .sin()
                        .reshape({1, 4, 4096})
                        .to(device, torch::kBFloat16);
  torch::Tensor weight =
      ((torch::arange(24 * 4 * 4096, cpu_options) * 0.00037).cos() * 0.005)
          .reshape({24, 4 * 4096})
          .to(device);
  torch::Tensor scale = torch::full({3}, 0.1, cpu_options).to(device);
  torch::Tensor base = torch::zeros({24}, cpu_options).to(device);
  torch::Tensor norm = torch::ones({4096}, x.options());
  auto run_hc = [&](const torch::Tensor& input) {
    bool normalized = false;
    auto result = deepseek_v4_hc_pre_decode(input,
                                            weight,
                                            scale,
                                            base,
                                            /*hc_mult=*/4,
                                            /*sinkhorn_iters=*/20,
                                            /*norm_eps=*/1e-6,
                                            /*hc_eps=*/1e-6,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            norm,
                                            &normalized);
    EXPECT_TRUE(normalized) << "The Mate fused-normalization route must run";
    return result;
  };
  auto reference = run_hc(x);
  reference_stream.synchronize();
  // Leave different prenorm partials in the shared workspace. The tested
  // call must wait for its own producer, not consume these stale partials.
  run_hc((x * 0.25 + 0.75).contiguous());
  reference_stream.synchronize();

  c10::StreamGuard default_guard(default_stream.unwrap());
  bind_tvmffi_stream(device);
  musaStream_t ffi_stream =
      reinterpret_cast<musaStream_t>(TVMFFIEnvGetStream(/*device_type=*/12,
                                                        /*device_id=*/0));
  ASSERT_NE(ffi_stream, default_stream.stream());
  ASSERT_EQ(musaLaunchHostFunc(ffi_stream, delay_ffi_producer, nullptr),
            musaSuccess);
  auto actual = run_hc(x);
  default_stream.synchronize();
  EXPECT_TRUE(torch::allclose(std::get<0>(actual).to(torch::kCPU),
                              std::get<0>(reference).to(torch::kCPU),
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(std::get<1>(actual).to(torch::kCPU),
                              std::get<1>(reference).to(torch::kCPU),
                              /*rtol=*/1e-5,
                              /*atol=*/1e-5));
  EXPECT_TRUE(torch::allclose(std::get<2>(actual).to(torch::kCPU),
                              std::get<2>(reference).to(torch::kCPU),
                              /*rtol=*/1e-5,
                              /*atol=*/1e-5));
}

}  // namespace
}  // namespace xllm::kernel::musa
