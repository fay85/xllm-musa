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
#include <gtest/gtest.h>
#include <musa_runtime_api.h>
#include <torch/torch.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int64_t kDefaultBeamWidth = 128;
constexpr int64_t kDefaultIterations = 3;
constexpr int64_t kGrowthProbeRows = 32;
constexpr int64_t kVocabSize = 151936;
constexpr int64_t kCandidatesPerBeam = 2;
constexpr int64_t kSeed = 20260828;

int64_t read_positive_env(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  char* end = nullptr;
  errno = 0;
  const long long parsed_value = std::strtoll(value, &end, /*base=*/10);
  CHECK_EQ(errno, 0) << "Invalid integer in " << name << ": " << value;
  CHECK(end != value && end != nullptr && end[0] == '\0')
      << "Invalid integer in " << name << ": " << value;
  CHECK_GT(parsed_value, 0) << name << " must be positive";
  return static_cast<int64_t>(parsed_value);
}

bool is_musa_available() {
  int device_count = 0;
  const musaError_t status = musaGetDeviceCount(&device_count);
  return status == musaSuccess && device_count > 0;
}

void expect_candidate_sets_equal(const torch::Tensor& actual_indices,
                                 const torch::Tensor& reference_indices) {
  const torch::Tensor actual_sorted =
      std::get<0>(torch::sort(actual_indices, /*dim=*/-1));
  const torch::Tensor reference_sorted =
      std::get<0>(torch::sort(reference_indices, /*dim=*/-1));
  EXPECT_TRUE(torch::equal(actual_sorted, reference_sorted));
}

TEST(MusaTopkTest, ConfiguredBeamMatchesCpuCandidateSet) {
  if (!is_musa_available()) {
    GTEST_SKIP() << "MUSA device is unavailable";
  }

  const musaError_t set_device_status = musaSetDevice(/*device=*/0);
  ASSERT_EQ(set_device_status, musaSuccess)
      << "musaSetDevice failed with status="
      << static_cast<int32_t>(set_device_status);

  const int64_t beam_width = read_positive_env(
      /*name=*/"XLLM_MUSA_TOPK_TEST_BEAM_WIDTH",
      /*default_value=*/kDefaultBeamWidth);
  const int64_t iterations = read_positive_env(
      /*name=*/"XLLM_MUSA_TOPK_TEST_ITERATIONS",
      /*default_value=*/kDefaultIterations);
  const int64_t k = kCandidatesPerBeam * beam_width;
  ASSERT_LE(k, kVocabSize);

  torch::manual_seed(kSeed);
  const torch::Tensor source_cpu = torch::randn(
      {beam_width, kVocabSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const torch::Tensor source =
      source_cpu.to(torch::Device(/*type=*/torch::kPrivateUse1, /*index=*/0));
  const std::tuple<torch::Tensor, torch::Tensor> reference =
      torch::topk(source_cpu, k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  const torch::Tensor reference_indices = std::get<1>(reference);

  if (beam_width > kGrowthProbeRows) {
    const torch::Tensor growth_probe =
        source.narrow(/*dim=*/0, /*start=*/0, /*length=*/kGrowthProbeRows);
    const std::tuple<torch::Tensor, torch::Tensor> probe_result =
        xllm::kernel::musa::topk(growth_probe, k);
    EXPECT_EQ(std::get<0>(probe_result).size(0), kGrowthProbeRows);
    EXPECT_EQ(std::get<1>(probe_result).size(0), kGrowthProbeRows);
  }

  for (int64_t iteration = 0; iteration < iterations; ++iteration) {
    const std::tuple<torch::Tensor, torch::Tensor> result =
        xllm::kernel::musa::topk(source, k);
    const torch::Tensor values = std::get<0>(result).to(torch::kCPU);
    const torch::Tensor indices = std::get<1>(result).to(torch::kCPU);

    ASSERT_EQ(indices.scalar_type(), torch::kLong);
    ASSERT_EQ(indices.dim(), 2);
    ASSERT_EQ(indices.size(0), beam_width);
    ASSERT_EQ(indices.size(1), k);
    EXPECT_GE(indices.min().item<int64_t>(), 0);
    EXPECT_LT(indices.max().item<int64_t>(), kVocabSize);

    const torch::Tensor gathered = source_cpu.gather(/*dim=*/-1, indices);
    EXPECT_TRUE(torch::equal(values, gathered))
        << "TopK values do not match their token indices at iteration "
        << iteration;
    expect_candidate_sets_equal(indices, reference_indices);
  }
}

TEST(MusaTopkTest, EqualScoresPreferSmallerTokenId) {
  if (!is_musa_available()) {
    GTEST_SKIP() << "MUSA device is unavailable";
  }

  const musaError_t set_device_status = musaSetDevice(/*device=*/0);
  ASSERT_EQ(set_device_status, musaSuccess)
      << "musaSetDevice failed with status="
      << static_cast<int32_t>(set_device_status);

  const int64_t rows = 2;
  const int64_t vocab_size = 16;
  const int64_t k = 4;
  torch::Tensor source_cpu = torch::zeros(
      {rows, vocab_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  source_cpu[0][5] = 1.0f;
  source_cpu[0][9] = 1.0f;
  source_cpu[0][2] = 1.0f;
  source_cpu[0][14] = 1.0f;
  source_cpu[1][3] = 2.0f;
  source_cpu[1][11] = 2.0f;
  source_cpu[1][7] = 2.0f;
  source_cpu[1][1] = 2.0f;
  const torch::Tensor source =
      source_cpu.to(torch::Device(/*type=*/torch::kPrivateUse1, /*index=*/0));

  const std::tuple<torch::Tensor, torch::Tensor> result =
      xllm::kernel::musa::topk(source, k);
  const torch::Tensor indices = std::get<1>(result).to(torch::kCPU);
  ASSERT_EQ(indices.size(0), rows);
  ASSERT_EQ(indices.size(1), k);
  EXPECT_EQ(indices[0][0].item<int64_t>(), 2);
  EXPECT_EQ(indices[0][1].item<int64_t>(), 5);
  EXPECT_EQ(indices[0][2].item<int64_t>(), 9);
  EXPECT_EQ(indices[0][3].item<int64_t>(), 14);
  EXPECT_EQ(indices[1][0].item<int64_t>(), 1);
  EXPECT_EQ(indices[1][1].item<int64_t>(), 3);
  EXPECT_EQ(indices[1][2].item<int64_t>(), 7);
  EXPECT_EQ(indices[1][3].item<int64_t>(), 11);
}

}  // namespace
}  // namespace xllm::kernel::musa
