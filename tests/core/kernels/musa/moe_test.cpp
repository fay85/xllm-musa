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
#include <torch/torch.h>

#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace xllm::kernel::musa {
namespace {

class MusaMoeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "No MUSA device available";
    }
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
    torch::manual_seed(17);
  }

  torch::TensorOptions options_ = torch::TensorOptions()
                                      .device(torch::kPrivateUse1, 0)
                                      .dtype(torch::kBFloat16);
};

TEST_F(MusaMoeTest, GroupedGemmMatchesReferenceWithEmptyExpert) {
  const torch::Tensor counts = torch::tensor({2, 0, 3}, torch::kInt32);
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat16}) {
    for (const bool fp8 : {false, true}) {
      const torch::ScalarType input_dtype = fp8 ? torch::kFloat8_e4m3fn : dtype;
      const torch::Tensor weights =
          (torch::randn({3, 256, 256}) * 0.1).to(input_dtype);
      const torch::Tensor device_weights = weights.to(options_.device());
      const torch::Tensor weight_scale =
          (torch::arange(12, torch::kFloat32).view({3, 2, 2}) / 16.0 + 0.5);
      const torch::Tensor dequant_weights =
          weights.to(torch::kFloat32) *
          (fp8 ? weight_scale.repeat_interleave(128, 1).repeat_interleave(128,
                                                                          2)
               : torch::ones_like(weights));
      const torch::Tensor device_weight_scale =
          weight_scale.to(options_.device());

      for (const int64_t alignment : {128, 256}) {
        SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " fp8=" << fp8
                                          << " alignment=" << alignment);
        const torch::Tensor input =
            (torch::randn({3, alignment, 256}) * 0.1).to(input_dtype);
        const torch::Tensor device_input = input.to(options_.device());
        const torch::Tensor input_scale =
            (torch::arange(3 * alignment * 2, torch::kFloat32)
                 .remainder(11)
                 .view({3, alignment, 2}) +
             4.0) /
            16.0;
        const torch::Tensor output =
            fp8 ? masked_moe_gemm_fp8(device_input,
                                      input_scale.to(options_.device()),
                                      device_weights,
                                      device_weight_scale,
                                      counts.to(options_.device()),
                                      dtype,
                                      alignment)
                : masked_moe_gemm_bf16(device_input,
                                       device_weights,
                                       counts.to(options_.device()),
                                       dtype,
                                       alignment);
        const torch::Tensor actual = output.cpu().to(torch::kFloat32);
        const torch::Tensor dequant_input =
            input.to(torch::kFloat32) *
            (fp8 ? input_scale.repeat_interleave(128, 2)
                 : torch::ones_like(input));
        for (const int64_t expert : {0, 2}) {
          const int64_t rows = counts[expert].item<int32_t>();
          const torch::Tensor reference =
              torch::matmul(dequant_input[expert].narrow(0, 0, rows),
                            dequant_weights[expert].transpose(0, 1));
          EXPECT_TRUE(torch::allclose(
              actual[expert].narrow(0, 0, rows), reference, 0.02, 0.002));
        }

        const torch::Tensor ragged_input =
            input.index_select(0, torch::tensor({0, 2}, torch::kInt64))
                .view({2 * alignment, 256})
                .to(options_.device());
        const torch::Tensor row_ids = torch::cat(
            {torch::full({alignment}, 0, options_.dtype(torch::kInt32)),
             torch::full({alignment}, 2, options_.dtype(torch::kInt32))});
        const torch::Tensor ragged_scale =
            input_scale.index_select(0, torch::tensor({0, 2}, torch::kInt64))
                .view({2 * alignment, 2})
                .to(options_.device());
        const torch::Tensor ragged_output =
            fp8 ? ragged_moe_gemm_fp8(ragged_input,
                                      ragged_scale,
                                      device_weights,
                                      device_weight_scale,
                                      row_ids,
                                      dtype,
                                      alignment)
                : ragged_moe_gemm_bf16(
                      ragged_input, device_weights, row_ids, dtype, alignment);
        const torch::Tensor reference =
            torch::cat({torch::matmul(dequant_input[0],
                                      dequant_weights[0].transpose(0, 1)),
                        torch::matmul(dequant_input[2],
                                      dequant_weights[2].transpose(0, 1))});
        EXPECT_TRUE(torch::allclose(
            ragged_output.cpu().to(torch::kFloat32), reference, 0.02, 0.002));
      }

      const torch::Tensor compact_input =
          (torch::randn({5, 256}) * 0.1).to(input_dtype);
      const torch::Tensor compact_scale =
          torch::arange(10, torch::kFloat32).view({5, 2}) / 32.0 + 0.25;
      const torch::Tensor compact_output =
          fp8 ? contiguous_moe_gemm_fp8(compact_input.to(options_.device()),
                                        compact_scale.to(options_.device()),
                                        device_weights,
                                        device_weight_scale,
                                        counts.to(options_.device()),
                                        dtype)
              : contiguous_moe_gemm_bf16(compact_input.to(options_.device()),
                                         device_weights,
                                         counts.to(options_.device()),
                                         dtype);
      const torch::Tensor dequant_input =
          compact_input.to(torch::kFloat32) *
          (fp8 ? compact_scale.repeat_interleave(128, 1)
               : torch::ones_like(compact_input));
      const torch::Tensor reference =
          torch::cat({torch::matmul(dequant_input.narrow(0, 0, 2),
                                    dequant_weights[0].transpose(0, 1)),
                      torch::matmul(dequant_input.narrow(0, 2, 3),
                                    dequant_weights[2].transpose(0, 1))});
      EXPECT_TRUE(torch::allclose(
          compact_output.cpu().to(torch::kFloat32), reference, 0.02, 0.002));
    }
  }
}

torch::Tensor quantized_reference(const torch::Tensor& input) {
  const torch::Tensor scales =
      input.abs().amax(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-10) / 448.0;
  return (input / scales)
             .clamp(-448, 448)
             .to(torch::kFloat8_e4m3fn)
             .to(torch::kFloat32) *
         scales;
}

TEST_F(MusaMoeTest, DecodeAotMatchesReferenceForEveryBuiltBatch) {
  const c10::musa::MUSAStream stream = c10::musa::getStreamFromPool();
  const c10::musa::MUSAStreamGuard stream_guard(stream);
  EXPECT_FALSE(fused_moe_aot_available(0));
  EXPECT_FALSE(fused_moe_bf16_aot_available(9));
  constexpr int64_t kExperts = 16;
  constexpr int64_t kHidden = 256;
  constexpr int64_t kIntermediate = 128;
  for (const bool fp8 : {false, true}) {
    const torch::ScalarType dtype =
        fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
    // Dyadic weights avoid CPU/device reduction order crossing FP8 rounding
    // ties after SwiGLU, keeping the pipeline reference deterministic.
    const torch::Tensor w13 =
        (torch::randint(-1, 2, {kExperts, 2 * kIntermediate, kHidden})
             .to(torch::kFloat32) /
         32.0)
            .to(dtype);
    const torch::Tensor w2 =
        (torch::randint(-1, 2, {kExperts, kHidden, kIntermediate})
             .to(torch::kFloat32) /
         32.0)
            .to(dtype);
    const torch::Tensor w13_scale = torch::tensor({0.5f, 1.5f, 0.75f, 1.25f})
                                        .view({1, 2, 2})
                                        .repeat({kExperts, 1, 1});
    const torch::Tensor w2_scale =
        torch::tensor({0.75f, 1.25f}).view({1, 2, 1}).repeat({kExperts, 1, 1});
    const torch::Tensor device_w13 = w13.to(options_.device());
    const torch::Tensor device_w2 = w2.to(options_.device());
    const torch::Tensor device_w13_scale = w13_scale.to(options_.device());
    const torch::Tensor device_w2_scale = w2_scale.to(options_.device());
    const torch::Tensor reference_w13 =
        w13.to(torch::kFloat32) *
        (fp8 ? w13_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
             : torch::ones_like(w13));
    const torch::Tensor reference_w2 =
        w2.to(torch::kFloat32) *
        (fp8 ? w2_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
             : torch::ones_like(w2));
    for (int64_t batch = 1; batch <= 8; ++batch) {
      SCOPED_TRACE(::testing::Message() << "batch=" << batch << " fp8=" << fp8);
      ASSERT_TRUE(fp8 ? fused_moe_aot_available(batch)
                      : fused_moe_bf16_aot_available(batch));
      const torch::Tensor hidden =
          (torch::randint(0, 2, {batch, kHidden}) * 2 - 1).to(torch::kBFloat16);
      const torch::Tensor ids =
          torch::arange(8, torch::kInt32).repeat({batch, 1});
      const torch::Tensor scores =
          (torch::arange(1, 9).to(torch::kFloat32) / 36.0).repeat({batch, 1});
      torch::Tensor device_hidden = hidden.to(options_.device());
      const torch::Tensor device_scores = scores.to(options_.device());
      const torch::Tensor device_ids = ids.to(options_.device());
      const auto forward = [&]() {
        return fp8 ? fused_moe_aot_fp8(device_hidden,
                                       device_w13,
                                       device_w13_scale,
                                       device_w2,
                                       device_w2_scale,
                                       device_scores,
                                       device_ids)
                   : fused_moe_aot_bf16(device_hidden,
                                        device_w13,
                                        device_w2,
                                        device_scores,
                                        device_ids);
      };
      const torch::Tensor actual = forward();
      const torch::Tensor reference_hidden =
          fp8 ? quantized_reference(
                    hidden.to(torch::kFloat32).view({batch * 2, 128}))
                    .view({batch, kHidden})
              : hidden.to(torch::kFloat32);
      torch::Tensor reference = torch::zeros({batch, kHidden});
      for (int64_t expert = 0; expert < 8; ++expert) {
        const torch::Tensor projected =
            torch::matmul(reference_hidden,
                          reference_w13[expert].transpose(0, 1))
                .to(torch::kBFloat16)
                .to(torch::kFloat32);
        const torch::Tensor gate =
            torch::silu(projected.narrow(1, 0, kIntermediate));
        const torch::Tensor activation =
            (gate * projected.narrow(1, kIntermediate, kIntermediate))
                .to(torch::kBFloat16)
                .to(torch::kFloat32);
        const torch::Tensor down =
            torch::matmul(fp8 ? quantized_reference(activation) : activation,
                          reference_w2[expert].transpose(0, 1));
        reference += (down * scores.narrow(1, expert, 1))
                         .to(torch::kBFloat16)
                         .to(torch::kFloat32);
      }
      reference = reference.to(torch::kBFloat16).to(torch::kFloat32);
      const torch::Tensor actual_cpu = actual.cpu().to(torch::kFloat32);
      EXPECT_TRUE(torch::allclose(actual_cpu, reference, 0.03, 0.001))
          << "max_abs=" << (actual_cpu - reference).abs().max().item<float>();
      if (batch != 1 && batch != 8) {
        continue;
      }
      ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
      at::musa::MUSAGraph graph;
      graph.capture_begin();
      const torch::Tensor captured = forward();
      graph.capture_end();
      for (int32_t replay = 0; replay < 2; ++replay) {
        device_hidden.mul_(0.5);
        graph.replay();
        EXPECT_TRUE(torch::equal(captured.cpu(), forward().cpu()));
      }
    }
  }
}

TEST_F(MusaMoeTest, RoutedSwiGluAndCombineMatchReference) {
  for (const int64_t tokens : {1, 129}) {
    for (const bool fp8 : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "tokens=" << tokens << " fp8=" << fp8);
      const torch::Tensor hidden =
          (torch::randn({tokens, 128}) * 0.1).to(torch::kBFloat16);
      const torch::ScalarType weight_dtype =
          fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
      const torch::Tensor w13 =
          (torch::randn({3, 256, 128}) * 0.1).to(weight_dtype);
      const torch::Tensor w2 =
          (torch::randn({3, 128, 128}) * 0.1).to(weight_dtype);
      const torch::Tensor ids = torch::tensor({0, 2}, torch::kInt32)
                                    .repeat({tokens, 1})
                                    .to(options_.device());
      const torch::Tensor scores = torch::tensor({0.25f, 0.75f})
                                       .repeat({tokens, 1})
                                       .to(options_.device());
      torch::Tensor gate_up;
      torch::Tensor row_ids;
      torch::Tensor mapping;
      torch::Tensor counts;
      if (fp8) {
        const auto preprocessed =
            fused_moe_preprocess_fp8(hidden.to(options_.device()),
                                     ids,
                                     /*num_experts=*/3,
                                     /*group_size=*/128);
        counts = std::get<3>(preprocessed);
        mapping = std::get<2>(preprocessed);
        gate_up = contiguous_moe_gemm_fp8(
            std::get<0>(preprocessed),
            std::get<1>(preprocessed),
            w13.to(options_.device()),
            torch::ones({3, 2, 1}, options_.dtype(torch::kFloat32)),
            counts,
            torch::kBFloat16);
      } else {
        const auto preprocessed =
            fused_moe_preprocess_bf16(hidden.to(options_.device()),
                                      ids,
                                      /*num_experts=*/3,
                                      /*alignment=*/128);
        row_ids = std::get<1>(preprocessed);
        mapping = std::get<2>(preprocessed);
        gate_up = ragged_moe_gemm_bf16(std::get<0>(preprocessed),
                                       w13.to(options_.device()),
                                       row_ids,
                                       torch::kBFloat16,
                                       /*alignment=*/128);
      }
      const torch::Tensor gate = gate_up.narrow(1, 0, 128);
      const torch::Tensor activated =
          (torch::silu(gate.to(torch::kFloat32)) *
           gate_up.narrow(1, 128, 128).to(torch::kFloat32))
              .to(torch::kBFloat16);
      torch::Tensor down;
      if (fp8) {
        const auto quantized = per_token_group_quant_fp8(activated.contiguous(),
                                                         /*group_size=*/128);
        down = contiguous_moe_gemm_fp8(
            std::get<0>(quantized),
            std::get<1>(quantized),
            w2.to(options_.device()),
            torch::ones({3, 1, 1}, options_.dtype(torch::kFloat32)),
            counts,
            torch::kBFloat16);
      } else {
        const torch::Tensor indexed =
            fused_moe_indexed_swiglu_bf16(gate_up, mapping);
        down = ragged_moe_gemm_bf16(indexed,
                                    w2.to(options_.device()),
                                    row_ids,
                                    torch::kBFloat16,
                                    /*alignment=*/128);
      }
      const torch::Tensor output =
          moe_combine_result_indexed(down, mapping, scores, tokens, /*topk=*/2);
      const torch::Tensor reference_hidden =
          fp8 ? quantized_reference(hidden.to(torch::kFloat32))
              : hidden.to(torch::kFloat32);
      torch::Tensor reference = torch::zeros({tokens, 128});
      for (const int64_t expert : {0, 2}) {
        const torch::Tensor projected =
            torch::matmul(reference_hidden,
                          w13[expert].to(torch::kFloat32).transpose(0, 1))
                .to(torch::kBFloat16)
                .to(torch::kFloat32);
        const torch::Tensor gate = projected.narrow(1, 0, 128);
        const torch::Tensor activation =
            (torch::silu(gate) * projected.narrow(1, 128, 128))
                .to(torch::kBFloat16)
                .to(torch::kFloat32);
        const torch::Tensor reference_activation =
            fp8 ? quantized_reference(activation) : activation;
        reference +=
            torch::matmul(reference_activation,
                          w2[expert].to(torch::kFloat32).transpose(0, 1))
                .to(torch::kBFloat16)
                .to(torch::kFloat32) *
            (expert == 0 ? 0.25 : 0.75);
      }
      EXPECT_TRUE(torch::allclose(
          output.cpu().to(torch::kFloat32), reference, 0.03, 0.002));
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::musa
