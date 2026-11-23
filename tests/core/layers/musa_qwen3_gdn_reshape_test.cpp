/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/kernels/musa/gdn_ops.h"
#include "core/layers/musa/qwen3_gated_delta_net_base.h"

namespace at {
struct MUSAHooksInterface;
namespace detail {
const MUSAHooksInterface& getMUSAHooks();
}  // namespace detail
}  // namespace at

namespace xllm::layer {
namespace {

class GdnReshapeProbe final : public Qwen3GatedDeltaNetBaseImpl {
 public:
  using Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad;
  using Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_with_pad;

  void load_state_dict(const StateDict& /*state_dict*/) override {}
  void verify_loaded_weights(const std::string& /*prefix*/) const override {}
  bool supports_fp8_input() const override { return false; }

 protected:
  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const NormalizedHiddenStates& /*hidden_states*/) override {
    return {};
  }
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const NormalizedHiddenStates& /*hidden_states*/) override {
    return {};
  }
};

AttentionMetadata make_metadata(const std::vector<int32_t>& lengths,
                                bool chunked) {
  AttentionMetadata metadata;
  metadata.q_seq_lens_vec = lengths;
  metadata.q_seq_lens = torch::tensor(lengths, torch::kInt32);
  metadata.max_query_len = *std::max_element(lengths.begin(), lengths.end());
  metadata.is_prefill = !chunked;
  metadata.is_chunked_prefill = chunked;
  return metadata;
}

TEST(MusaGdnReshapeTest, UniformPaddedInputsAlwaysReturnTokenMajorRows) {
  const GdnReshapeProbe probe;
  for (const bool chunked : {false, true}) {
    for (const int64_t batch_size : {1, 3}) {
      for (const int32_t length : {1, 7}) {
        SCOPED_TRACE(::testing::Message()
                     << "chunked=" << chunked << " batch=" << batch_size
                     << " length=" << length);
        const AttentionMetadata metadata =
            make_metadata(std::vector<int32_t>(batch_size, length), chunked);
        constexpr int64_t kDim = 32;
        const torch::Tensor flat =
            torch::arange(batch_size * length * kDim, torch::kFloat32)
                .view({batch_size * length, kDim});
        const torch::Tensor padded = flat.view({batch_size, length, kDim});
        const torch::Tensor output = probe.reshape_qkvz_unpad(metadata, padded);
        EXPECT_EQ(output.sizes().vec(), flat.sizes().vec());
        EXPECT_TRUE(torch::equal(output, flat));
        EXPECT_TRUE(
            torch::equal(probe.reshape_qkvz_unpad(metadata, flat), flat));
        if (batch_size == 1) {
          EXPECT_EQ(output.data_ptr(), flat.data_ptr());
        }
      }
    }
  }
}

TEST(MusaGdnReshapeTest, NonuniformAndStridedInputsRoundTripWithoutPadding) {
  const GdnReshapeProbe probe;
  const std::vector<int32_t> lengths = {3, 1, 2};
  for (const bool host_lengths : {false, true}) {
    for (const bool chunked : {false, true}) {
      AttentionMetadata metadata = make_metadata(lengths, chunked);
      if (!host_lengths) {
        metadata.q_seq_lens_vec.clear();
      }
      const torch::Tensor storage =
          torch::arange(6 * 16, torch::kFloat32).view({6, 16});
      const torch::Tensor flat = storage.narrow(/*dim=*/1,
                                                /*start=*/0,
                                                /*length=*/8);
      const torch::Tensor padded = probe.reshape_qkvz_with_pad(metadata, flat);
      const torch::Tensor output = probe.reshape_qkvz_unpad(metadata, padded);
      EXPECT_EQ(output.sizes().vec(), flat.sizes().vec());
      EXPECT_TRUE(torch::equal(output, flat));
    }
  }
}

TEST(MusaGdnReshapeTest, DecodeKeepsItsOriginalLayout) {
  const GdnReshapeProbe probe;
  AttentionMetadata metadata = make_metadata({1}, /*chunked=*/false);
  metadata.is_prefill = false;
  const torch::Tensor input = torch::ones({1, 1, 32});
  const torch::Tensor output = probe.reshape_qkvz_unpad(metadata, input);
  EXPECT_EQ(output.sizes().vec(), input.sizes().vec());
  EXPECT_EQ(output.data_ptr(), input.data_ptr());
}

TEST(MusaGdnReshapeGpuTest, SingleTokenCp2PrefixMatchesConvolutionReference) {
  // Exercise the production unpadding method before the real convolution:
  // rank 1 can own one row at a chunk/prefix boundary in a CP2 prefill.
  const GdnReshapeProbe probe;
  if (!c10::is_privateuse1_backend_registered()) {
    c10::register_privateuse1_backend("musa");
  }
  (void)at::detail::getMUSAHooks();
  constexpr int64_t kDim = 10240;
  const torch::Device device(c10::DeviceType::PrivateUse1, 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  for (const bool chunked : {false, true}) {
    AttentionMetadata metadata = make_metadata({1}, chunked);
    const torch::Tensor padded = torch::randn({1, 1, kDim}, options) * 0.2f;
    const torch::Tensor input = probe.reshape_qkvz_unpad(metadata, padded);
    ASSERT_EQ(input.sizes().vec(), (std::vector<int64_t>{1, kDim}));
    const torch::Tensor prefix = torch::randn({3, kDim}, options) * 0.2f;
    const torch::Tensor weight = torch::randn({kDim, 4}, options) * 0.2f;
    const torch::Tensor state = torch::full({4, kDim, 6}, -5.0f, options);
    const torch::Tensor starts =
        torch::tensor({0, 1}, options.dtype(torch::kInt32));
    const torch::Tensor slots =
        torch::tensor({1}, options.dtype(torch::kInt32));
    const torch::Tensor read_slots =
        torch::tensor({2}, options.dtype(torch::kInt32));
    const torch::Tensor initial = torch::ones({1}, options.dtype(torch::kBool));
    const torch::Tensor sequence =
        torch::cat({prefix.cpu(), input.cpu()}, /*dim=*/0).to(torch::kFloat32);
    torch::Tensor expected =
        (sequence * weight.cpu().to(torch::kFloat32).t()).sum(0, true);
    expected *= torch::sigmoid(expected);
    torch::Tensor expected_state = state.cpu().clone();
    expected_state[1]
        .narrow(/*dim=*/1, /*start=*/0, /*length=*/3)
        .copy_(sequence.narrow(/*dim=*/0, /*start=*/1, /*length=*/3)
                   .t()
                   .to(torch::kBFloat16));
    const torch::Tensor output =
        kernel::musa::causal_conv1d_prefill(input,
                                            weight,
                                            state,
                                            std::nullopt,
                                            starts,
                                            slots,
                                            initial,
                                            /*silu_activation=*/true,
                                            prefix,
                                            read_slots);
    EXPECT_TRUE(torch::allclose(output.cpu(),
                                expected.to(torch::kBFloat16),
                                /*rtol=*/1e-2,
                                /*atol=*/1e-3));
    EXPECT_TRUE(torch::equal(state.cpu(), expected_state));
  }
}

}  // namespace
}  // namespace xllm::layer
