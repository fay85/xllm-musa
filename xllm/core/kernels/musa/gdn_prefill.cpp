/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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


#include "core/kernels/musa/musa_ops_api.h"

#include <glog/logging.h>
#include <unistd.h>
#include <algorithm>

#include <cstdlib>
#include <sstream>
#include <vector>

#include "core/common/macros.h"
#include "core/kernels/param.h"
#include "core/util/env_var.h"
#include "core/util/xllm_kineto_profiler.h"

namespace xllm {
namespace kernel {
namespace cuda {

namespace {

inline torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

}

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  const auto initial_dtype = query.scalar_type();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  auto to_f32_bhtd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_bhtd(query);
  key = to_f32_bhtd(key);
  value = to_f32_bhtd(value);
  g = to_f32_bhtd(g);
  torch::Tensor beta_f32;
  if (params.beta.has_value() && params.beta.value().defined()) {
    beta_f32 = to_f32_bhtd(params.beta.value());
  } else {
    beta_f32 = torch::ones_like(g);
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);
  const float scale_val =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale_val;

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state =
        params.initial_state.value().to(torch::kFloat32).transpose(-1, -2);
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i);
    auto beta_t = beta_f32.select(2, i);
    auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
    last_recurrent_state.mul_(g_exp);
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
    last_recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
    core_attn_out.select(2, i) =
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  last_recurrent_state = last_recurrent_state.transpose(-1, -2);
  return {core_attn_out, last_recurrent_state};
}

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  auto beta = params.beta;
  const int64_t chunk_size = 64;
  const auto initial_dtype = query.dtype();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  const int64_t Hqk = query.size(2);
  const int64_t Hv = value.size(2);
  if (Hqk != Hv) {
    CHECK(Hv % Hqk == 0) << "chunk_gated_delta_rule: Hv (" << Hv
                         << ") must be a multiple of Hqk (" << Hqk
                         << ") for GQA expansion";
    const int64_t repeat = Hv / Hqk;
    query = query.repeat_interleave(repeat, /*dim=*/2);
    key = key.repeat_interleave(repeat, /*dim=*/2);
  }

  auto to_f32_thd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_thd(query);
  key = to_f32_thd(key);
  value = to_f32_thd(value);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);

  const int64_t pad_size = (chunk_size - sequence_length % chunk_size) %
                           chunk_size;
  using PadOpts = torch::nn::functional::PadFuncOptions;
  if (pad_size != 0) {
    query = torch::nn::functional::pad(query, PadOpts({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, PadOpts({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, PadOpts({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, PadOpts({0, pad_size}));
    g = torch::nn::functional::pad(g, PadOpts({0, pad_size}));
  }
  const int64_t total_sequence_length = sequence_length + pad_size;
  const float scale =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);

  auto reshape_to_chunks = [chunk_size](const torch::Tensor& x) {
    return x.reshape({x.size(0), x.size(1), x.size(2) / chunk_size, chunk_size,
                      x.size(3)});
  };
  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);
  g = g.reshape({g.size(0), g.size(1), g.size(2) / chunk_size, chunk_size});

  auto mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);
  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32).tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }
  attn = attn + torch::eye(chunk_size, torch::TensorOptions()
                                           .dtype(attn.dtype())
                                           .device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, k_beta * g.exp().unsqueeze(-1));

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state = params.initial_state.value().to(value.dtype());
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  }
  auto core_attn_out = torch::zeros_like(value);
  const int64_t num_chunks = total_sequence_length / chunk_size;

  auto upper_mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions()
                      .dtype(torch::kBool)
                      .device(query.device())),
      1);
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(upper_mask, 0.0);
    auto v_prime =
        torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  const auto s = core_attn_out.sizes();
  core_attn_out = core_attn_out.reshape({s[0], s[1], s[2] * s[3], s[4]});
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}


namespace {

constexpr int64_t kGdnChunkSize = 64;

int64_t chunk_pad_size(int64_t seq_len, int64_t chunk_size) {
  return (chunk_size - seq_len % chunk_size) % chunk_size;
}

torch::Tensor pad_time_dim_4d(const torch::Tensor& tensor, int64_t pad_size) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, pad_size}));
}

torch::Tensor pad_time_dim_3d(const torch::Tensor& tensor,
                              int64_t pad_size,
                              double pad_value) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size})
          .mode(torch::kConstant)
          .value(pad_value));
}

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  LOG(FATAL) << "mate GDN prefill expects bfloat16 or float16 q/k/v";
}

void l2norm_last_dim(torch::Tensor& tensor) {
  const auto orig_dtype = tensor.scalar_type();
  tensor = torch::nn::functional::normalize(
      tensor.to(torch::kFloat32),
      torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
  tensor = tensor.to(orig_dtype);
}

// Host cu_seqlens helpers — never D2H. Callers must pass CPU lengths.
bool cu_seqlens_all_full(c10::ArrayRef<int32_t> cu, int64_t max_len) {
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    if (static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1] -
                             cu[static_cast<size_t>(seq_idx)]) != max_len) {
      return false;
    }
  }
  return true;
}

std::vector<int32_t> materialize_cu_seqlens_host(
    const MateGatedDeltaRulePrefillParams& params) {
  if (params.cu_seqlens_host.has_value() &&
      !params.cu_seqlens_host->empty()) {
    return *params.cu_seqlens_host;
  }
  CHECK(params.cu_seqlens.has_value() && params.cu_seqlens->defined())
      << "mate GDN prefill needs cu_seqlens_host or device cu_seqlens";
  // Fallback: one D2H per forward (still far better than 8x/layer). Eager
  // multi-seq only — capture path uses the B=1 zeros+fill_ branch instead.
  const auto cu_cpu =
      params.cu_seqlens->to(torch::kCPU).to(torch::kInt32).contiguous();
  const int32_t* ptr = cu_cpu.data_ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + cu_cpu.numel());
}

// Simple Mate GDN prefill expects per-chunk cumulative log-decay, matching
// mate.gdn_kernels.tilelang.gdn_chunk_local_cumsum for already-log-space g
// (xLLM fused_gdn_gating emits log(alpha) = -A_log * softplus(...)).
//
// Capture-safe for the common single-sequence case (no D2H). Multi-sequence
// path uses host lengths and must not run under graph capture.
torch::Tensor chunk_local_cumsum_log_space(const torch::Tensor& g_log,
                                           int64_t num_seqs,
                                           c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(g_log.dim(), 3) << "g_log must be [1, T, H]";
  CHECK_EQ(g_log.size(0), 1) << "g_log must be packed as batch=1";
  const int64_t total_tokens = g_log.size(1);
  CHECK_EQ(total_tokens % kGdnChunkSize, 0)
      << "g_log T must be padded to a multiple of " << kGdnChunkSize;
  CHECK_EQ(static_cast<int64_t>(cu.size()) - 1, num_seqs);

  if (num_seqs == 1) {
    // Single sequence spanning the full packed tensor: pure torch ops, no D2H.
    const int64_t num_chunks = total_tokens / kGdnChunkSize;
    const int64_t num_heads = g_log.size(2);
    return g_log.view({1, num_chunks, kGdnChunkSize, num_heads})
        .cumsum(/*dim=*/2)
        .reshape({1, total_tokens, num_heads})
        .contiguous();
  }

  // Per-seq vectorized cumsum: 1-2 launches/seq instead of ~ceil(L/64).
  auto out = torch::empty_like(g_log);
  const int64_t num_heads = g_log.size(2);
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t start =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx)]);
    const int64_t end =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1]);
    CHECK_GE(end, start);
    CHECK_LE(end, total_tokens);
    const int64_t len = end - start;
    if (len == 0) {
      continue;
    }
    const int64_t n_full = len / kGdnChunkSize;
    const int64_t rem = len % kGdnChunkSize;
    if (n_full > 0) {
      const int64_t full_end = start + n_full * kGdnChunkSize;
      out.slice(/*dim=*/1, start, full_end) =
          g_log.slice(/*dim=*/1, start, full_end)
              .view({1, n_full, kGdnChunkSize, num_heads})
              .cumsum(/*dim=*/2)
              .reshape({1, n_full * kGdnChunkSize, num_heads});
    }
    if (rem > 0) {
      const int64_t rem_start = start + n_full * kGdnChunkSize;
      out.slice(/*dim=*/1, rem_start, end) =
          g_log.slice(/*dim=*/1, rem_start, end).cumsum(/*dim=*/1);
    }
  }
  return out;
}

torch::Tensor pack_time_dim_4d(const torch::Tensor& padded,
                               c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(padded.dim(), 4);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  CHECK_EQ(padded.size(0), num_seqs);
  const int64_t max_len = padded.size(1);
  // Homogeneous batch: [B, T, H, D] -> [1, B*T, H, D] is a view (no cat).
  if (cu_seqlens_all_full(cu, max_len)) {
    return padded.reshape(
        {1, num_seqs * max_len, padded.size(2), padded.size(3)});
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<size_t>(num_seqs));
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t len = static_cast<int64_t>(
        cu[static_cast<size_t>(seq_idx) + 1] - cu[static_cast<size_t>(seq_idx)]);
    parts.push_back(padded.select(/*dim=*/0, seq_idx).narrow(
        /*dim=*/0, /*start=*/0, /*length=*/len));
  }
  return torch::cat(parts, /*dim=*/0).unsqueeze(0).contiguous();
}

torch::Tensor pack_time_dim_3d(const torch::Tensor& padded,
                               c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(padded.dim(), 3);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  CHECK_EQ(padded.size(0), num_seqs);
  const int64_t max_len = padded.size(1);
  if (cu_seqlens_all_full(cu, max_len)) {
    return padded.reshape({1, num_seqs * max_len, padded.size(2)});
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<size_t>(num_seqs));
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t len = static_cast<int64_t>(
        cu[static_cast<size_t>(seq_idx) + 1] - cu[static_cast<size_t>(seq_idx)]);
    parts.push_back(padded.select(/*dim=*/0, seq_idx).narrow(
        /*dim=*/0, /*start=*/0, /*length=*/len));
  }
  return torch::cat(parts, /*dim=*/0).unsqueeze(0).contiguous();
}

torch::Tensor unpack_time_dim_4d(const torch::Tensor& packed,
                                 c10::ArrayRef<int32_t> cu,
                                 int64_t max_len) {
  CHECK_EQ(packed.dim(), 4);
  CHECK_EQ(packed.size(0), 1);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  if (cu_seqlens_all_full(cu, max_len)) {
    CHECK_EQ(packed.size(1), num_seqs * max_len);
    return packed.reshape(
        {num_seqs, max_len, packed.size(2), packed.size(3)});
  }
  auto out = torch::zeros(
      {num_seqs, max_len, packed.size(2), packed.size(3)}, packed.options());
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t start =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx)]);
    const int64_t end =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1]);
    const int64_t len = end - start;
    out.select(/*dim=*/0, seq_idx)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/len)
        .copy_(packed.select(/*dim=*/0, 0).narrow(
            /*dim=*/0, /*start=*/start, /*length=*/len));
  }
  return out;
}

bool mate_gdn_debug_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_DEBUG");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

bool mate_gdn_validation_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_VALIDATE");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

bool mate_kkt_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_KKT");
    // Default on: the host PyTorch KKT path is a major TTFT hotspot.
    if (env == nullptr) {
      return true;
    }
    return env[0] != '0';
  }();
  return enabled;
}

std::string get_mate_kkt_solve_uri(int64_t num_q_heads,
                                   int64_t num_v_heads,
                                   torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_kkt_solve_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

bool mate_kkt_module_available(const std::string& uri) {
  const std::string ops_path = util::get_string_env("FLASHINFER_OPS_PATH");
  if (ops_path.empty()) {
    return false;
  }
  const std::string so_path = ops_path + "/" + uri + "/" + uri + ".so";
  return ::access(so_path.c_str(), R_OK) == 0;
}

// Host fallback matching SGLang mate.kkt_solve numerically. Prefer the Mate
// TileLang FFI path below: the per-row PyTorch loop is a major TTFT hotspot
// across 48 GDN layers.
//
// key:  [B, T, Hqk, D]  (already L2-normalized, post-padding)
// beta: [B, T, Hv]      (float32, sigmoid output)
// returns: [B, T, Hv, chunk_size]  (same dtype as key)
torch::Tensor kkt_solve_torch(const torch::Tensor& key,
                              const torch::Tensor& beta,
                              int64_t chunk_size) {
  XLLM_KINETO_USER_SCOPE("xllm/kkt_solve_torch");
  const int64_t batch_size = key.size(0);
  const int64_t num_tokens = key.size(1);
  const int64_t Hqk = key.size(2);
  const int64_t DK = key.size(3);
  const int64_t Hv = beta.size(2);

  auto k_f32 = key.to(torch::kFloat32).contiguous();
  auto beta_f32 = beta.to(torch::kFloat32).contiguous();

  if (Hv != Hqk) {
    CHECK(Hv % Hqk == 0) << "kkt_solve: Hv (" << Hv
                         << ") must be a multiple of Hqk (" << Hqk << ")";
    const int64_t repeat = Hv / Hqk;
    k_f32 = k_f32.repeat_interleave(repeat, /*dim=*/2);
  }

  auto k_beta = k_f32 * beta_f32.unsqueeze(-1);

  const int64_t num_chunks = num_tokens / chunk_size;

  auto k_chunks = k_f32.reshape({batch_size, num_chunks, chunk_size, Hv, DK})
                       .permute({0, 3, 1, 2, 4})
                       .contiguous();
  auto kb_chunks = k_beta.reshape({batch_size, num_chunks, chunk_size, Hv, DK})
                         .permute({0, 3, 1, 2, 4})
                         .contiguous();

  auto gram = torch::matmul(kb_chunks, k_chunks.transpose(-1, -2));

  auto mask = torch::triu(
      torch::ones({chunk_size, chunk_size},
                  torch::TensorOptions().dtype(torch::kBool).device(key.device())),
      0);
  auto attn = (-gram).masked_fill(mask, 0.0);

  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }

  attn = attn + torch::eye(chunk_size, torch::TensorOptions()
                                            .dtype(attn.dtype())
                                            .device(attn.device()));

  attn = attn.permute({0, 2, 3, 1, 4}).contiguous();
  attn = attn.reshape({batch_size, num_tokens, Hv, chunk_size});

  return attn.to(key.scalar_type());
}

// Mate TileLang KKT solve via TVM FFI (same kernel SGLang uses).
// ABI: main(k, b, a, num_chunks) with
//   k: [B, T, Hq, K], b: [B, T, Hv], a: [B, T, Hv, 64],
//   num_chunks: B * ceil_div(T, 64)
torch::Tensor kkt_solve_mate_ffi(const torch::Tensor& key,
                                 const torch::Tensor& beta,
                                 int64_t chunk_size) {
  XLLM_KINETO_USER_SCOPE("xllm/kkt_solve_mate");
  CHECK_EQ(chunk_size, kGdnChunkSize)
      << "mate KKT solve currently requires chunk_size=" << kGdnChunkSize;
  CHECK_EQ(key.size(3), 128) << "mate KKT solve currently requires K=128";
  CHECK_EQ(key.size(1) % chunk_size, 0)
      << "mate KKT solve expects T padded to chunk_size";

  const int64_t batch_size = key.size(0);
  const int64_t num_tokens = key.size(1);
  const int64_t num_q_heads = key.size(2);
  const int64_t num_v_heads = beta.size(2);
  const std::string uri =
      get_mate_kkt_solve_uri(num_q_heads, num_v_heads, key.scalar_type());

  auto key_contig = key.contiguous();
  auto beta_contig = beta.to(torch::kFloat32).contiguous();
  auto a = torch::empty({batch_size, num_tokens, num_v_heads, chunk_size},
                        key.options());
  const int32_t num_chunks = static_cast<int32_t>(
      batch_size * (num_tokens / chunk_size));

  auto main = get_function(uri, "main");
  main(to_ffi_tensor(key_contig),
       to_ffi_tensor(beta_contig),
       to_ffi_tensor(a),
       num_chunks);
  return a;
}

torch::Tensor kkt_solve(const torch::Tensor& key,
                        const torch::Tensor& beta,
                        int64_t chunk_size) {
  if (mate_kkt_enabled()) {
    const std::string uri = get_mate_kkt_solve_uri(
        key.size(2), beta.size(2), key.scalar_type());
    if (mate_kkt_module_available(uri)) {
      if (ensure_tilelang_musa_loader()) {
        return kkt_solve_mate_ffi(key, beta, chunk_size);
      }
      LOG_FIRST_N(WARNING, 1)
          << "[MateKktSolve] TileLang MUSA module loader unavailable; "
             "falling back to torch KKT";
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "[MateKktSolve] module not found for uri=" << uri
          << " under FLASHINFER_OPS_PATH; falling back to torch KKT";
    }
  }
  return kkt_solve_torch(key, beta, chunk_size);
}

}

std::string get_mate_gdn_prefill_simple_uri(int64_t num_q_heads,
                                            int64_t num_v_heads,
                                            torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::string get_mate_gdn_prefill_full_uri(int64_t num_q_heads,
                                          int64_t num_v_heads,
                                          torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_full_hq" << num_q_heads << "_hv" << num_v_heads
      << "_" << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

bool mate_gdn_force_simple_kernel() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_PREFILL_SIMPLE");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

bool mate_gdn_module_available(const std::string& uri) {
  const std::string ops_path = util::get_string_env("FLASHINFER_OPS_PATH");
  if (ops_path.empty()) {
    return false;
  }
  const std::string so_path = ops_path + "/" + uri + "/" + uri + ".so";
  return access(so_path.c_str(), R_OK) == 0;
}

std::string get_mate_gdn_prefill_uri(int64_t num_q_heads,
                                     int64_t num_v_heads,
                                     torch::ScalarType dtype) {
  const std::string full_uri =
      get_mate_gdn_prefill_full_uri(num_q_heads, num_v_heads, dtype);
  if (!mate_gdn_force_simple_kernel() && mate_gdn_module_available(full_uri)) {
    return full_uri;
  }
  return get_mate_gdn_prefill_simple_uri(num_q_heads, num_v_heads, dtype);
}

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params) {
  XLLM_KINETO_USER_SCOPE("xllm/mate_gdn_prefill");
  auto query = params.q.contiguous();
  auto key = params.k.contiguous();
  auto value = params.v.contiguous();
  CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4)
      << "mate GDN prefill expects q/k/v shaped [B, T, H, D]";
  CHECK(query.scalar_type() == key.scalar_type() &&
        query.scalar_type() == value.scalar_type())
      << "mate GDN prefill expects q/k/v to share dtype";

  const int64_t input_batch = query.size(0);
  const int64_t input_seq_len = query.size(1);
  const int64_t num_q_heads = query.size(2);
  const int64_t num_v_heads = value.size(2);
  const int64_t head_k_dim = query.size(3);
  const int64_t head_v_dim = value.size(3);
  CHECK(head_k_dim == head_v_dim)
      << "mate GDN prefill currently requires K == V, got K=" << head_k_dim
      << " V=" << head_v_dim;
  CHECK(num_v_heads % num_q_heads == 0)
      << "mate GDN prefill expects Hv divisible by Hqk";

  const std::string uri =
      get_mate_gdn_prefill_uri(num_q_heads, num_v_heads, query.scalar_type());
  const bool use_full_warp =
      uri.find("mate_gdn_prefill_full_") != std::string::npos;

  // ------------------------------------------------------------------
  // Warp-specialized padded path (SGLang/installed Mate).
  // Layout: [B, T, H, D], log-space g (no host cumsum), KKT-filled `a`.
  // State is mate k-last [B, Hv, V, K] — same as the simple path and the
  // installed Mate Python API (kernel indexes h0[..., V, K]).
  // ------------------------------------------------------------------
  if (use_full_warp) {
    const int64_t batch_size = input_batch;
    const int64_t pad_size = chunk_pad_size(input_seq_len, kGdnChunkSize);
    if (pad_size > 0) {
      query = pad_time_dim_4d(query, pad_size);
      key = pad_time_dim_4d(key, pad_size);
      value = pad_time_dim_4d(value, pad_size);
    }
    const int64_t num_tokens = query.size(1);

    if (params.use_qk_l2norm_in_kernel) {
      l2norm_last_dim(query);
      l2norm_last_dim(key);
    }
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();

    auto beta = params.beta.to(torch::kFloat32).contiguous();
    auto g_log = params.g.to(torch::kFloat32).contiguous();
    if (pad_size > 0) {
      beta = pad_time_dim_3d(beta, pad_size, 0.0);
      g_log = pad_time_dim_3d(g_log, pad_size, 0.0);
    }
    // Full warp kernel (is_log_space=true) cumsums log-g internally — do not
    // pre-apply chunk_local_cumsum (that is the simple-kernel ABI).

    // Dummy cu_seqlens required by the C ABI; unused when is_varlen=false.
    auto cu_seqlens = torch::zeros(
        {batch_size + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

    torch::Tensor a;
    torch::Tensor h0;
    torch::Tensor output;
    torch::Tensor final_state;
    {
      MusaTvmffiStreamGuard stream_guard(query.device());
      a = kkt_solve(key, beta, kGdnChunkSize);
      if (params.initial_state.has_value() && params.initial_state->defined()) {
        h0 = params.initial_state->to(torch::kFloat32).contiguous();
      } else {
        h0 = torch::zeros({batch_size, num_v_heads, head_v_dim, head_k_dim},
                          torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(query.device()));
      }
      CHECK_EQ(h0.size(0), batch_size);
      CHECK_EQ(h0.size(1), num_v_heads);
      CHECK_EQ(h0.size(2), head_v_dim);
      CHECK_EQ(h0.size(3), head_k_dim);
      output = torch::empty({batch_size, num_tokens, num_v_heads, head_v_dim},
                            value.options());
      final_state = torch::empty(
          {batch_size, num_v_heads, head_v_dim, head_k_dim},
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

      if (mate_gdn_debug_enabled()) {
        LOG(INFO) << "[MateGdnPrefillFull] q=" << query.sizes()
                  << " a=" << a.sizes() << " uri=" << uri
                  << " pad_size=" << pad_size;
      }
      auto run = get_function(uri, "run");
      run(to_ffi_tensor(query),
          to_ffi_tensor(key),
          to_ffi_tensor(value),
          to_ffi_tensor(a),
          to_ffi_tensor(g_log),
          to_ffi_tensor(beta),
          to_ffi_tensor(h0),
          to_ffi_tensor(cu_seqlens),
          to_ffi_tensor(output),
          to_ffi_tensor(final_state));
    }

    if (pad_size > 0) {
      output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/input_seq_len)
                   .contiguous();
    }
    return {output, final_state};
  }

  // ------------------------------------------------------------------
  // Legacy simple varlen path (fallback).
  // ------------------------------------------------------------------
  torch::Tensor cu_seqlens;
  bool packed_input = false;
  int64_t num_seqs = 1;
  std::vector<int32_t> cu_host;
  std::vector<int32_t> cu_host_unpadded;
  const bool has_device_cu =
      params.cu_seqlens.has_value() && params.cu_seqlens->defined();
  const bool has_host_cu =
      params.cu_seqlens_host.has_value() && !params.cu_seqlens_host->empty();

  if (has_host_cu) {
    cu_host = *params.cu_seqlens_host;
    CHECK_GE(cu_host.size(), 2u);
    num_seqs = static_cast<int64_t>(cu_host.size()) - 1;
    cu_host_unpadded = cu_host;
    if (input_batch == 1) {
      packed_input = true;
    } else {
      CHECK_EQ(input_batch, num_seqs)
          << "padded mate prefill batch must match cu_seqlens sequences";
      query = pack_time_dim_4d(query, cu_host);
      key = pack_time_dim_4d(key, cu_host);
      value = pack_time_dim_4d(value, cu_host);
      packed_input = true;
    }
  } else if (has_device_cu) {
    CHECK_EQ(params.cu_seqlens->dim(), 1);
    CHECK_GE(params.cu_seqlens->size(0), 2);
    num_seqs = params.cu_seqlens->size(0) - 1;
    if (input_batch == 1) {
      packed_input = true;
      if (num_seqs > 1) {
        cu_host = materialize_cu_seqlens_host(params);
        cu_host_unpadded = cu_host;
      }
    } else {
      CHECK_EQ(input_batch, num_seqs)
          << "padded mate prefill batch must match cu_seqlens sequences";
      cu_host = materialize_cu_seqlens_host(params);
      cu_host_unpadded = cu_host;
      query = pack_time_dim_4d(query, cu_host);
      key = pack_time_dim_4d(key, cu_host);
      value = pack_time_dim_4d(value, cu_host);
      packed_input = true;
    }
  } else {
    CHECK_EQ(input_batch, 1)
        << "mate GDN varlen prefill requires cu_seqlens when batch > 1";
    num_seqs = 1;
  }

  const int64_t packed_seq_len = query.size(1);
  const int64_t pad_size = chunk_pad_size(packed_seq_len, kGdnChunkSize);
  if (pad_size > 0) {
    query = pad_time_dim_4d(query, pad_size);
    key = pad_time_dim_4d(key, pad_size);
    value = pad_time_dim_4d(value, pad_size);
  }
  const int64_t num_tokens = query.size(1);

  if (num_seqs == 1) {
    cu_seqlens = torch::zeros(
        {2}, torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    cu_seqlens.select(0, 1).fill_(static_cast<int64_t>(num_tokens));
    cu_host = {0, static_cast<int32_t>(num_tokens)};
  } else {
    CHECK(!cu_host.empty())
        << "multi-seq mate GDN prefill requires host cu_seqlens";
    if (pad_size > 0) {
      cu_host.back() += static_cast<int32_t>(pad_size);
    }
    cu_seqlens = torch::tensor(
        cu_host,
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
  }

  if (params.use_qk_l2norm_in_kernel) {
    l2norm_last_dim(query);
    l2norm_last_dim(key);
  }
  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();

  auto beta = params.beta.to(torch::kFloat32).contiguous();
  auto g_log = params.g.to(torch::kFloat32).contiguous();
  if (packed_input && input_batch > 1) {
    beta = pack_time_dim_3d(beta, cu_host_unpadded);
    g_log = pack_time_dim_3d(g_log, cu_host_unpadded);
  }
  if (pad_size > 0) {
    beta = pad_time_dim_3d(beta, pad_size, 0.0);
    g_log = pad_time_dim_3d(g_log, pad_size, 0.0);
  }
  g_log = chunk_local_cumsum_log_space(g_log, num_seqs, cu_host);

  if (mate_gdn_debug_enabled()) {
    LOG(INFO) << "[MateGdnPrefillSimple] q=" << query.sizes()
              << " uri=" << uri << " pad_size=" << pad_size
              << " num_tokens=" << num_tokens << " num_seqs=" << num_seqs;
  }

  torch::Tensor a;
  torch::Tensor h0;
  torch::Tensor output;
  torch::Tensor final_state;
  {
    MusaTvmffiStreamGuard stream_guard(query.device());
    auto run = get_function(uri, "run");
    a = torch::empty({1, num_tokens, num_v_heads, kGdnChunkSize},
                     query.options());
    if (params.initial_state.has_value() && params.initial_state->defined()) {
      h0 = params.initial_state->to(torch::kFloat32).contiguous();
      CHECK_EQ(h0.dim(), 4) << "mate GDN prefill initial_state must be 4D";
      CHECK_EQ(h0.size(0), num_seqs)
          << "mate GDN prefill initial_state batch mismatch";
      CHECK_EQ(h0.size(1), num_v_heads)
          << "mate GDN prefill initial_state head mismatch";
      CHECK_EQ(h0.size(2), head_v_dim)
          << "mate GDN prefill initial_state V dim mismatch";
      CHECK_EQ(h0.size(3), head_k_dim)
          << "mate GDN prefill initial_state K dim mismatch";
    } else {
      h0 = torch::zeros({num_seqs, num_v_heads, head_v_dim, head_k_dim},
                        torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(query.device()));
    }
    output = torch::empty({1, num_tokens, num_v_heads, head_v_dim},
                          value.options());
    final_state = torch::empty(
        {num_seqs, num_v_heads, head_v_dim, head_k_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    run(to_ffi_tensor(query),
        to_ffi_tensor(key),
        to_ffi_tensor(value),
        to_ffi_tensor(a),
        to_ffi_tensor(g_log),
        to_ffi_tensor(beta),
        to_ffi_tensor(h0),
        to_ffi_tensor(cu_seqlens),
        to_ffi_tensor(output),
        to_ffi_tensor(final_state));
  }

  if (pad_size > 0) {
    output =
        output.slice(/*dim=*/1, /*start=*/0, /*end=*/packed_seq_len).contiguous();
  }

  if (input_batch > 1) {
    output = unpack_time_dim_4d(output, cu_host_unpadded, input_seq_len);
  }
  return {output, final_state};
}

torch::Tensor causal_conv1d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    const torch::IntArrayRef /*num_accepted_tokens_opt*/,
    int64_t activation_mode,
    int64_t /*pad_slot_id*/,
    int64_t /*run_mode*/) {
  const auto device = x.device();

  auto x_t = x.t().contiguous();
  auto out_t = torch::empty_like(x_t);

  auto qsl_cpu = torch::empty(
      {static_cast<int64_t>(query_start_loc_opt.size())}, torch::kInt32);
  for (size_t i = 0; i < query_start_loc_opt.size(); ++i)
    qsl_cpu.data_ptr<int32_t>()[i] =
        static_cast<int32_t>(query_start_loc_opt[i]);
  auto query_start_loc = qsl_cpu.to(device);

  torch::Tensor cache_indices;
  if (!cache_indices_opt.empty()) {
    auto ci_cpu = torch::empty(
        {static_cast<int64_t>(cache_indices_opt.size())}, torch::kInt32);
    for (size_t i = 0; i < cache_indices_opt.size(); ++i)
      ci_cpu.data_ptr<int32_t>()[i] =
          static_cast<int32_t>(cache_indices_opt[i]);
    cache_indices = ci_cpu.to(device);
  }

  torch::Tensor has_initial_state;
  if (!initial_state_mode_opt.empty()) {
    auto his_cpu = torch::empty(
        {static_cast<int64_t>(initial_state_mode_opt.size())}, torch::kBool);
    for (size_t i = 0; i < initial_state_mode_opt.size(); ++i)
      his_cpu.data_ptr<bool>()[i] = (initial_state_mode_opt[i] != 0);
    has_initial_state = his_cpu.to(device);
  }

  const bool silu_activation = (activation_mode != 0);
  const int64_t pad_slot_id = -1;

  causal_conv1d_fwd(
      x_t, weight, out_t, bias_opt, conv_state,
      query_start_loc, cache_indices, has_initial_state,
      silu_activation, pad_slot_id);

  return out_t.t().contiguous().to(x.dtype());
}

torch::Tensor causal_conv1d_prefill(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const torch::Tensor& has_initial_state,
    bool silu_activation) {
  CHECK(query_start_loc.defined() &&
        is_torch_musa_device(query_start_loc.device()))
      << "causal_conv1d_prefill requires device query_start_loc";
  CHECK(cache_indices.defined() &&
        is_torch_musa_device(cache_indices.device()))
      << "causal_conv1d_prefill requires device cache_indices";
  CHECK(has_initial_state.defined() &&
        is_torch_musa_device(has_initial_state.device()))
      << "causal_conv1d_prefill requires device has_initial_state";

  auto x_t = x.t().contiguous();
  auto out_t = torch::empty_like(x_t);
  causal_conv1d_fwd(x_t,
                    weight,
                    out_t,
                    bias,
                    conv_state,
                    query_start_loc,
                    cache_indices,
                    has_initial_state,
                    silu_activation,
                    /*pad_slot_id=*/-1);
  return out_t.t().contiguous();
}

}
}
}
