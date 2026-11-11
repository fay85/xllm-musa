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

#include <limits>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/layers/musa/flashinfer_attention.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::layer {
namespace {

TEST(MusaPrefillTest, CachedPrefixesMatchCausalReference) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(758);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  for (const int64_t ratio : {6, 8}) {
    SCOPED_TRACE(ratio);
    const int64_t heads = 2 * ratio;
    const torch::Tensor query =
        torch::randn({5, heads, 256}).to(torch::kBFloat16);
    const torch::Tensor full_key =
        torch::randn({13, 2, 256}).to(torch::kBFloat16);
    const torch::Tensor full_value =
        torch::randn({13, 2, 256}).to(torch::kBFloat16);
    torch::Tensor k_cache = torch::zeros({2, 64, 2, 256}, options);
    torch::Tensor v_cache = torch::zeros_like(k_cache);
    k_cache[0].narrow(0, 0, 3).copy_(full_key.narrow(0, 0, 3));
    k_cache[1].narrow(0, 0, 5).copy_(full_key.narrow(0, 5, 5));
    v_cache[0].narrow(0, 0, 3).copy_(full_value.narrow(0, 0, 3));
    v_cache[1].narrow(0, 0, 5).copy_(full_value.narrow(0, 5, 5));
    KVCache cache(KVCacheTensors{k_cache, v_cache});
    const torch::Tensor new_key =
        torch::cat({full_key.narrow(0, 3, 2), full_key.narrow(0, 10, 3)}, 0);
    const torch::Tensor new_value = torch::cat(
        {full_value.narrow(0, 3, 2), full_value.narrow(0, 10, 3)}, 0);
    torch::Tensor device_query = query.to(device.unwrap());
    torch::Tensor device_key = new_key.to(device.unwrap());
    torch::Tensor device_value = new_value.to(device.unwrap());
    torch::Tensor output = torch::empty_like(device_query);
    AttentionMetadata metadata;
    metadata.is_prefill = true;
    metadata.is_causal = true;
    metadata.max_query_len = 3;
    metadata.max_seq_len = 8;
    metadata.q_seq_lens_vec = {2, 3};
    metadata.kv_seq_lens_vec = {5, 8};
    const auto int_options = options.dtype(torch::kInt32);
    metadata.q_cu_seq_lens = torch::tensor({0, 2, 5}, int_options);
    metadata.kv_seq_lens = torch::tensor({5, 8}, int_options);
    metadata.block_table = torch::tensor({{0}, {1}}, int_options);
    metadata.slot_mapping = torch::tensor({3, 4, 69, 70, 71}, int_options);
    FlashInferAttentionImpl attention(heads, 256, 0.0625f, 2, -1);
    for (const bool chunked : {false, true}) {
      SCOPED_TRACE(chunked);
      metadata.is_prefill = !chunked;
      metadata.is_chunked_prefill = chunked;
      attention.forward(
          metadata, device_query, device_key, device_value, output, cache);
      const torch::Tensor actual = output.cpu().view({5, heads, 256});
      int64_t query_offset = 0;
      int64_t kv_offset = 0;
      for (const int64_t row : {0, 1}) {
        const int64_t q_len = metadata.q_seq_lens_vec[row];
        const int64_t kv_len = metadata.kv_seq_lens_vec[row];
        const torch::Tensor q = query.narrow(0, query_offset, q_len)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor k = full_key.narrow(0, kv_offset, kv_len)
                                    .repeat_interleave(ratio, 1)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor v = full_value.narrow(0, kv_offset, kv_len)
                                    .repeat_interleave(ratio, 1)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor mask =
            torch::arange(kv_len).unsqueeze(0) >
            torch::arange(q_len).unsqueeze(1) + kv_len - q_len;
        const torch::Tensor scores =
            (torch::matmul(q, k.transpose(1, 2)) * 0.0625)
                .masked_fill(mask, -std::numeric_limits<float>::infinity());
        const torch::Tensor expected =
            torch::matmul(torch::softmax(scores, -1), v).transpose(0, 1);
        EXPECT_TRUE(torch::allclose(
            actual.narrow(0, query_offset, q_len).to(torch::kFloat32),
            expected,
            /*rtol=*/0.015,
            /*atol=*/0.015));
        query_offset += q_len;
        kv_offset += kv_len;
      }
    }
  }
}

}  // namespace
}  // namespace xllm::layer
