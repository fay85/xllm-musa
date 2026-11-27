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

#include "qwen2_vision_attention.h"

#include <algorithm>

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "kernels/npu/npu_ops_api.h"
#endif
#if defined(USE_MLU)
#include "kernels/mlu/mlu_ops_api.h"
#endif
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"
#include "platform/platform.h"
namespace xllm {
namespace layer {

Qwen2VisionAttentionImpl::Qwen2VisionAttentionImpl(const ModelContext& context,
                                                   bool has_bias) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t hidden_size = args.mm_hidden_size();
  const int64_t num_heads = args.mm_num_attention_heads();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  CHECK(num_heads % tp_size == 0);

  tp_group_ = parallel_args.tp_group_;
  hidden_size_per_attention_head_ = args.mm_head_dim();
  num_attention_heads_per_partition_ = num_heads / tp_size;
  scale_ =
      1.0 / std::sqrt(static_cast<double>(hidden_size_per_attention_head_));

  qkv_proj_ =
      register_module("qkv_proj",
                      QKVParallelLinear(hidden_size,
                                        num_attention_heads_per_partition_,
                                        num_attention_heads_per_partition_,
                                        hidden_size_per_attention_head_,
                                        /*num_kv_head_replicas=*/1,
                                        /*bias=*/has_bias,
                                        /*gather_output=*/false,
                                        parallel_args,
                                        options,
                                        quant_args));

  proj_ = register_module("proj",
                          RowParallelLinear(hidden_size,
                                            hidden_size,
                                            /*bias=*/has_bias,
                                            /*input_is_parallelized=*/true,
                                            /*if_reduce_results=*/true,
                                            quant_args,
                                            parallel_args.tp_group_,
                                            options));
#if defined(USE_NPU)
  // RotaryMul supports Qwen3's 72-wide heads without the ATB setup path.
  if (hidden_size_per_attention_head_ != 72) {
    rope_layer_ = register_module("rope", NpuRopeLayer(context));
  }
#endif
}

std::vector<torch::Tensor> Qwen2VisionAttentionImpl::split_qkv(
    const torch::Tensor& qkv) {
  // [s, b, 3 * head * head_dim]
  const int64_t seq_len = qkv.size(0);
  const int64_t bs = qkv.dim() == 3 ? qkv.size(1) : 1;
  // QKVParallelLinear already stores [Q_rank, K_rank, V_rank]. Gathering
  // every rank only to scatter each component returns these same shards.
  auto qkv_chunks = qkv.chunk(3, /*dim=*/-1);
  auto q = qkv_chunks[0];
  auto k = qkv_chunks[1];
  auto v = qkv_chunks[2];

  // 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
  std::vector<int64_t> new_shape = {seq_len,
                                    bs,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_};
  q = q.reshape(new_shape);
  k = k.reshape(new_shape);
  v = v.reshape(new_shape);

  return {q, k, v};
}

int32_t Qwen2VisionAttentionImpl::get_max_sequence_length(
    const std::vector<int32_t>& cu_seq_len_vec) {
  if (cu_seq_len_vec.size() < static_cast<size_t>(2)) {
    return 0;
  }
  int32_t max_seq_len = 0;
  for (size_t i = 1; i < cu_seq_len_vec.size(); ++i) {
    CHECK_GE(cu_seq_len_vec[i], cu_seq_len_vec[i - 1]);
    max_seq_len =
        std::max(max_seq_len, cu_seq_len_vec[i] - cu_seq_len_vec[i - 1]);
  }
  return max_seq_len;
}

namespace {

#if defined(USE_MUSA) || defined(USE_CUDA) || defined(USE_DCU)
void compute_qwen2_vision_attention_torch(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& output,
    const std::vector<int32_t>& cu_seq_len_vec,
    float scale) {
  if (cu_seq_len_vec.size() < 2) {
    return;
  }

  const int32_t num_seqs = static_cast<int32_t>(cu_seq_len_vec.size()) - 1;
  for (int32_t i = 0; i < num_seqs; ++i) {
    int32_t start = cu_seq_len_vec[i];
    int32_t end = cu_seq_len_vec[i + 1];
    int32_t len = end - start;
    if (len <= 0) {
      continue;
    }

    // Current sequence: [len, H, D] -> [H, len, D]
    const torch::Tensor q_i =
        q.slice(/*dim=*/0, /*start=*/start, /*end=*/end).permute({1, 0, 2});
    const torch::Tensor k_i =
        k.slice(/*dim=*/0, /*start=*/start, /*end=*/end).permute({1, 0, 2});
    const torch::Tensor v_i =
        v.slice(/*dim=*/0, /*start=*/start, /*end=*/end).permute({1, 0, 2});

    torch::Tensor out_i;
    if constexpr (Platform::is_musa()) {
      // Avoid materializing the quadratic score matrix on long images.
      out_i = torch::scaled_dot_product_attention(q_i.unsqueeze(0),
                                                  k_i.unsqueeze(0),
                                                  v_i.unsqueeze(0),
                                                  /*attn_mask=*/std::nullopt,
                                                  /*dropout_p=*/0.0,
                                                  /*is_causal=*/false,
                                                  /*scale=*/scale)
                  .squeeze(/*dim=*/0);
    } else {
      const torch::Tensor scores =
          torch::matmul(q_i * scale, k_i.transpose(1, 2));
      out_i = torch::matmul(torch::softmax(scores, /*dim=*/-1), v_i);
    }

    // Back to [len, H, D] and write into output.
    output.slice(/*dim=*/0, /*start=*/start, /*end=*/end)
        .copy_(out_i.permute({1, 0, 2}).contiguous());
  }
}
#endif

#if defined(USE_NPU)
void compute_qwen_vision_attention_fused(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& output,
    const std::vector<int32_t>& cu_seq_len_vec,
    double scale) {
  CHECK_GE(cu_seq_len_vec.size(), static_cast<size_t>(2));
  CHECK_EQ(cu_seq_len_vec.front(), 0);

  std::vector<int64_t> actual_seq_lengths;
  actual_seq_lengths.reserve(cu_seq_len_vec.size() - 1);
  for (size_t index = 1; index < cu_seq_len_vec.size(); ++index) {
    CHECK_GE(cu_seq_len_vec[index], cu_seq_len_vec[index - 1]);
    actual_seq_lengths.emplace_back(cu_seq_len_vec[index]);
  }

  // Ascend FIA runs faster with a 128-wide head than with an unaligned
  // head between 64 and 128. Preserve the original attention scale and
  // discard the padded output channels, as in the VA encoder.
  const int64_t head_dim = query.size(-1);
  const int64_t padded_head_dim = 128;
  const bool pad_head = head_dim > 64 && head_dim < padded_head_dim;
  auto query_input = query;
  auto key_input = key;
  auto value_input = value;
  if (pad_head) {
    const std::vector<int64_t> padding{0, padded_head_dim - head_dim};
    query_input = torch::constant_pad_nd(query, padding, 0);
    key_input = torch::constant_pad_nd(key, padding, 0);
    value_input = torch::constant_pad_nd(value, padding, 0);
  }

  auto attention_result =
      xllm::kernel::npu::npu_fused_infer_attention(query_input,
                                                   key_input,
                                                   value_input,
                                                   /*atten_mask=*/std::nullopt,
                                                   /*block_table=*/std::nullopt,
                                                   actual_seq_lengths,
                                                   actual_seq_lengths,
                                                   query.size(1),
                                                   key.size(1),
                                                   scale,
                                                   /*block_size=*/128,
                                                   /*sparse_mode=*/0,
                                                   "TND",
                                                   /*softmax_lse_flag=*/false,
                                                   /*is_causal=*/false);
  output.copy_(std::get<0>(attention_result)
                   .slice(/*dim=*/-1, /*start=*/0, /*end=*/head_dim)
                   .view_as(output));
}
#endif  // defined(USE_NPU)

}  // namespace

torch::Tensor Qwen2VisionAttentionImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  // 2. split qkv
  auto qkv_split = split_qkv(qkv);
  // 3. transpose [s, b, h, d] -> [b, s, h, d]
  for (auto& tensor : qkv_split) {
    tensor = tensor.transpose(0, 1).contiguous();
  }
  auto q = qkv_split[0];
  auto k = qkv_split[1];
  auto v = qkv_split[2];
  int64_t B = q.size(0);
  int64_t S = q.size(1);
  int64_t head_dim = q.size(3);
  CHECK_EQ(head_dim, hidden_size_per_attention_head_) << "head_dim mismatch";
  int32_t max_seqlen = get_max_sequence_length(cu_seq_len_vec);

  // 4. rope
  // Reshape q, k from [B, S, H, D] to [B*S, H, D] before applying RoPE so
  // that the RoPE kernel sees the correct total token count (B*S = seq_len),
  // not just the batch dimension (B=1).
  q = q.reshape({B * S, num_attention_heads_per_partition_, head_dim});
  k = k.reshape({B * S, num_attention_heads_per_partition_, head_dim});

  // Apply rotary position embedding to both q and k in a single call.
#if defined(USE_NPU)
  if (head_dim == 72) {
    auto qk = torch::stack({q, k}, /*dim=*/0);
    auto rotated = at_npu::native::custom_ops::npu_rotary_mul(
        qk,
        m_cos_pos.reshape({1, B * S, 1, head_dim}),
        m_sin_pos.reshape({1, B * S, 1, head_dim}));
    q = rotated.select(/*dim=*/0, /*index=*/0);
    k = rotated.select(/*dim=*/0, /*index=*/1);
  } else {
    auto q_atb = q.reshape({B * S, -1});
    auto k_atb = k.reshape({B * S, -1});
    auto rotary_result = rope_layer_->forward(
        q_atb, k_atb, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec);
    q = std::get<0>(rotary_result)
            .reshape({B * S, num_attention_heads_per_partition_, head_dim});
    k = std::get<1>(rotary_result)
            .reshape({B * S, num_attention_heads_per_partition_, head_dim});
  }
#else
  // NOTE: Do NOT call apply_rotary twice; the first call already handles both
  // q and k. A second call would incorrectly apply RoPE to k a second time.
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = m_sin_pos;
  rotary_params.cos = m_cos_pos;
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
#if defined(USE_CUDA) || defined(USE_DCU) || defined(USE_MUSA)
  rotary_params.position_ids =
      torch::arange(
          q.size(0),
          torch::TensorOptions().dtype(torch::kInt64).device(q.device()))
          .contiguous();
  rotary_params.cu_query_lens = std::nullopt;
#else
  rotary_params.cu_query_lens = cu_seq_len;
#endif
  rotary_params.max_query_len = max_seqlen;
  xllm::kernel::apply_rotary(rotary_params);
  q = rotary_params.q.reshape(
      {B * S, num_attention_heads_per_partition_, head_dim});
  k = rotary_params.k.reshape(
      {B * S, num_attention_heads_per_partition_, head_dim});
#endif

  // q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
  // q and k are already [B*S, H, D] after the reshape above; just
  // flatten v to the same shape.
  v = v.view({B * S, v.size(2), v.size(3)});
  torch::Tensor output = torch::zeros_like(q);

  // 5. store k/v cache and do attention
#if defined(USE_MLU)
  std::optional<torch::Tensor> output_lse = std::nullopt;

  xllm::kernel::mlu::batch_prefill(q,
                                   k,
                                   v,
                                   output,
                                   output_lse,
                                   cu_seq_len,
                                   cu_seq_len,
                                   /*alibi_slope=*/std::nullopt,
                                   /*alibi_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   /*out_quant_scale=*/std::nullopt,
                                   /*block_table=*/std::nullopt,
                                   max_seqlen,
                                   max_seqlen,
                                   scale_,
                                   /*is_causal=*/false,
                                   /*window_size_left=*/-1,
                                   /*window_size_right=*/-1,
                                   /*compute_dtype=*/"half",
                                   /*return_lse=*/false);
#elif defined(USE_CUDA) || defined(USE_DCU) || defined(USE_MUSA)
  // Use a pure PyTorch vision attention implementation that
  // matches Transformers Qwen2.5-VL VisionAttention. FlashInfer's precompiled
  // AOT kernels in this project do not support head_dim=80, so we intentionally
  // do not call FlashInfer here and run attention entirely in PyTorch instead.
  compute_qwen2_vision_attention_torch(q, k, v, output, cu_seq_len_vec, scale_);
#elif defined(USE_NPU)
  compute_qwen_vision_attention_fused(q, k, v, output, cu_seq_len_vec, scale_);
#endif

  // context_layer = rearrange(output, "(b s) h d -> s b (h d)", b=batch_size)
  output = output.view({B, S, -1});
  // [B, S, ...] -> [S, B, ...]
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  // 6. output projection
  return proj_->forward(output);
}

void Qwen2VisionAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
  proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
}

}  // namespace layer
}  // namespace xllm
