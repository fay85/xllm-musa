/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/layers/musa/qwen3_5/qwen3_5_vision_attention.h"

#include <algorithm>

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

torch::Tensor Qwen3_5VisionAttentionImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec) {
  auto qkv = qkv_proj_->forward(hidden_states);
  auto qkv_split = split_qkv(qkv);
  for (auto& tensor : qkv_split) {
    tensor = tensor.transpose(0, 1).contiguous();
  }

  auto q = qkv_split[0];
  auto k = qkv_split[1];
  auto v = qkv_split[2];
  const int64_t batch = q.size(0);
  const int64_t sequence = q.size(1);
  const int64_t head_dim = q.size(3);
  CHECK_EQ(head_dim, hidden_size_per_attention_head_)
      << "Qwen3.5 vision head_dim mismatch";

  const int32_t max_seqlen = get_max_sequence_length(cu_seq_len_vec);
  q = q.reshape({batch * sequence,
                 num_attention_heads_per_partition_,
                 hidden_size_per_attention_head_});
  k = k.reshape({batch * sequence,
                 num_attention_heads_per_partition_,
                 hidden_size_per_attention_head_});

  // MUSA rotary kernels use FP32 tables/intermediates for the vision head.
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q.to(torch::kFloat32);
  rotary_params.k = k.to(torch::kFloat32);
  rotary_params.sin = m_sin_pos.to(torch::kFloat32);
  rotary_params.cos = m_cos_pos.to(torch::kFloat32);
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
  rotary_params.position_ids =
      torch::arange(
          q.size(0),
          torch::TensorOptions().dtype(torch::kInt64).device(q.device()))
          .contiguous();
  rotary_params.cu_query_lens = std::nullopt;
  rotary_params.max_query_len = max_seqlen;
  xllm::kernel::apply_rotary(rotary_params);
  q = rotary_params.q.to(hidden_states.scalar_type());
  k = rotary_params.k.to(hidden_states.scalar_type());

  v = v.view({batch * sequence, v.size(2), v.size(3)});
  torch::Tensor output = torch::zeros_like(q);
  const int32_t num_sequences = static_cast<int32_t>(cu_seq_len_vec.size()) - 1;
  for (int32_t index = 0; index < num_sequences; ++index) {
    const int32_t start = cu_seq_len_vec[index];
    const int32_t end = cu_seq_len_vec[index + 1];
    if (end <= start) {
      continue;
    }

    // Fused non-causal attention per image, like the NPU and MLU vision
    // paths. The MUSA SDPA kernel never materializes the [heads, n, n]
    // scores, which for a 1344x672 image alone take 760 MiB in fp32.
    const torch::Tensor q_i =
        q.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor k_i =
        k.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor v_i =
        v.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor out_i =
        torch::scaled_dot_product_attention(q_i,
                                            k_i,
                                            v_i,
                                            /*attn_mask=*/std::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/false,
                                            /*scale=*/scale_);
    output.slice(/*dim=*/0, start, end)
        .copy_(out_i.squeeze(/*dim=*/0).permute({1, 0, 2}));
  }

  output = output.view({batch, sequence, -1});
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  return proj_->forward(output);
}

}  // namespace layer
}  // namespace xllm
