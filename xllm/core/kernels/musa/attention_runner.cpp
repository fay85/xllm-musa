/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kernels/musa/attention_runner.h"

#include <algorithm>
#include <glog/logging.h>

#include <cstdlib>
#include <string>

namespace {

constexpr int64_t kGdnChunkSize = 64;

torch::Tensor canonical_time_prefix(const torch::Tensor& tensor,
                                    int64_t num_tokens) {
  CHECK_EQ(tensor.size(0), 1);
  CHECK_LE(num_tokens, tensor.size(1));
  if (tensor.dim() == 3) {
    const int64_t dim = tensor.size(2);
    return tensor.as_strided({1, num_tokens, dim}, {num_tokens * dim, dim, 1});
  }
  if (tensor.dim() == 4) {
    const int64_t heads = tensor.size(2);
    const int64_t dim = tensor.size(3);
    return tensor.as_strided({1, num_tokens, heads, dim},
                             {num_tokens * heads * dim, heads * dim, dim, 1});
  }
  LOG(FATAL) << "GDN prefix tensor must be 3D or 4D";
}

bool capture_fa3_in_piecewise_graph() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_PIECEWISE_CAPTURE_FA3");
    return env == nullptr || std::string(env) != "0";
  }();
  return enabled;
}

int64_t capture_fa3_max_padding_tokens() {
  static const int64_t max_padding = [] {
    const char* env =
        std::getenv("XLLM_PIECEWISE_CAPTURE_FA3_MAX_PADDING_TOKENS");
    return env == nullptr
               ? int64_t{32}
               : static_cast<int64_t>(std::strtoll(env, nullptr, 10));
  }();
  return max_padding;
}

}  // namespace

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/kernels/musa/gdn_ops.h"
#include "core/kernels/musa/global_capture_instance.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/ops_api.h"

namespace xllm {
namespace kernel {
namespace musa {

void AttentionRunner::run_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    uint32_t padded_num_tokens) {
  // plan_info is supplied per replay via AttentionReplayParams; not stored
  // here.
  (void)plan_info;

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  uri_ = uri;

  float_workspace_buffer_ = float_workspace_buffer;
  int_workspace_buffer_ = int_workspace_buffer;
  page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
  query_ = query;
  key_ = key;
  value_ = value;
  output_ = output;
  window_size_left_ = window_left;
  scale_ = sm_scale;
  padded_num_tokens_ = padded_num_tokens;
  runner_type_ = RunnerType::PREFILL;

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_chunked_prefill_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr,
    bool causal,
    const torch::Tensor& paged_kv_indptr_host,
    const torch::Tensor& paged_kv_indices_host,
    const torch::Tensor& paged_kv_last_page_len_host,
    uint32_t padded_num_tokens) {
  (void)plan_info;
  (void)paged_kv_indptr;
  (void)paged_kv_indices;
  (void)paged_kv_last_page_len;
  (void)output_lse;
  (void)qo_indptr;
  (void)paged_kv_indptr_host;
  (void)paged_kv_indices_host;
  (void)paged_kv_last_page_len_host;

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  uri_ = uri;
  float_workspace_buffer_ = float_workspace_buffer;
  int_workspace_buffer_ = int_workspace_buffer;
  page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
  query_ = query;
  k_cache_ = k_cache;
  v_cache_ = v_cache;
  output_ = output;
  window_size_left_ = window_left;
  scale_ = sm_scale;
  padded_num_tokens_ = padded_num_tokens;
  runner_type_ = RunnerType::CHUNKED_PREFILL;
  causal_ = causal;

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}


namespace {
// Same ladder as generate_piecewise_prefill_graph_tokens, then Qwen3.5-style
// 64-align used by MusaGraphExecutorImpl when selecting the graph bucket.
int64_t piecewise_graph_bucket_capacity(int64_t n) {
  static const int64_t kBuckets[] = {
      4,  8,  12, 16, 20, 24, 28, 32,  48,  64,  80,   96,   112,  128,
      144, 160, 176, 192, 208, 224, 240, 256, 288,  320,  352,  384,  416,
      448, 480, 512, 576, 640, 704, 768, 832,  896,  960,  1024, 1280,
      1536, 1792, 2048};
  int64_t b = n;
  for (int64_t cand : kBuckets) {
    if (cand >= n) {
      b = cand;
      break;
    }
  }
  if (b % 64 != 0) {
    b = ((b + 63) / 64) * 64;
  }
  return b;
}

void expand_lse_to_token_capacity(torch::Tensor& output_lse, int64_t q_cap) {
  if (!output_lse.defined() || q_cap <= 0) {
    return;
  }
  if (output_lse.dim() == 2) {
    const int64_t heads = output_lse.size(0);
    if (output_lse.size(1) < q_cap) {
      auto expanded = torch::empty({heads, q_cap}, output_lse.options());
      expanded.slice(/*dim=*/1, /*start=*/0, /*end=*/output_lse.size(1))
          .copy_(output_lse);
      output_lse = std::move(expanded);
    }
  } else if (output_lse.dim() == 1) {
    // Infer heads from numel / current trailing size if divisible; else grow
    // by treating the buffer as heads*tokens with tokens inferred from q_cap
    // only when we already know heads from a 2D producer.
    const int64_t cur = output_lse.numel();
    // Prefer growing to heads*q_cap when cur is divisible by a plausible
    // token count <= q_cap.
    int64_t heads = 0;
    for (int64_t tokens = std::min(cur, q_cap); tokens >= 1; --tokens) {
      if (cur % tokens == 0) {
        const int64_t h = cur / tokens;
        if (h > 0 && h <= 256 && (tokens == q_cap || h * q_cap > cur)) {
          heads = h;
          break;
        }
      }
    }
    if (heads > 0) {
      const int64_t need = heads * q_cap;
      if (cur < need) {
        auto expanded = torch::empty({need}, output_lse.options());
        expanded.narrow(/*dim=*/0, /*start=*/0, /*length=*/cur).copy_(output_lse);
        output_lse = std::move(expanded);
      }
    }
  }
}
}  // namespace

void AttentionRunner::run_fa3_paged_prefill_capture(torch::Tensor query,
                                                    torch::Tensor k_cache,
                                                    torch::Tensor v_cache,
                                                    int64_t window_left,
                                                    int64_t window_right,
                                                    double sm_scale,
                                                    torch::Tensor output,
                                                    torch::Tensor output_lse) {
  // Paged FA3 reads the live page table, so it stays between graph segments.
  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  query_ = std::move(query);
  k_cache_ = std::move(k_cache);
  v_cache_ = std::move(v_cache);
  output_ = std::move(output);
  output_lse_ = std::move(output_lse);
  window_size_left_ = window_left;
  window_size_right_ = window_right;
  scale_ = sm_scale;
  runner_type_ = RunnerType::FA3_PAGED_PREFILL;

  // Root cause: capture may observe actual_num_tokens (e.g. 166) while the
  // piecewise graph bucket is larger (e.g. 192 after ladder+64-align). Replay
  // of another request in the same bucket (e.g. 187) then fails
  // output_lse_.numel() >= heads * actual (2656 vs 2992). LSE is a runner
  // workspace (not a graph-captured address), so expanding it here is safe.
  // query_/output_ stay as the tensors passed from the model so they keep
  // pointing at graph-persistent activations.
  const int64_t q_rows = query_.defined() ? query_.size(0) : int64_t{0};
  const int64_t out_rows = output_.defined() ? output_.size(0) : int64_t{0};
  const int64_t q_cap =
      piecewise_graph_bucket_capacity(std::max({q_rows, out_rows, int64_t{1}}));
  expand_lse_to_token_capacity(output_lse_, q_cap);
  LOG(WARNING) << "FA3_PAGED_CAPTURE_SIZES query=" << query_.sizes()
               << " output=" << output_.sizes()
               << " lse=" << output_lse_.sizes()
               << " q_cap=" << q_cap;

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_fa3_prefill_capture(torch::Tensor query,
                                              torch::Tensor key,
                                              torch::Tensor value,
                                              int64_t max_seqlen_q,
                                              int64_t max_seqlen_k,
                                              int64_t window_left,
                                              int64_t window_right,
                                              double sm_scale,
                                              torch::Tensor output,
                                              torch::Tensor output_lse) {
  // FA3 is launched outside the graph during replay. End the current graph
  // segment before storing its tensors, then resume capture for downstream
  // layers; this is the same sequencing used by the FlashInfer runner.
  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  query_ = std::move(query);
  key_ = std::move(key);
  value_ = std::move(value);
  output_ = std::move(output);
  output_lse_ = std::move(output_lse);
  window_size_left_ = window_left;
  window_size_right_ = window_right;
  scale_ = sm_scale;
  max_seqlen_q_ = max_seqlen_q;
  max_seqlen_k_ = max_seqlen_k;
  runner_type_ = RunnerType::FA3_PREFILL;

  // LSE / IO buffers must cover the full piecewise bucket capacity (query
  // rows), not just capture-time actual/max_seqlen. Otherwise a later replay
  // with more tokens in the same bucket fails
  // output_lse_.numel() >= heads * actual_num_tokens (seen 16*166 vs 16*187).
  // Prefer the piecewise bucket capacity: same token list as
  // generate_piecewise_prefill_graph_tokens (at least cover max_seqlen_q).
  auto piecewise_bucket_for = [](int64_t n) -> int64_t {
    static const int64_t kBuckets[] = {
        4,  8,  12, 16, 20, 24, 28, 32,  48,  64,  80,   96,   112,  128,
        144, 160, 176, 192, 208, 224, 240, 256, 288,  320,  352,  384,  416,
        448, 480, 512, 576, 640, 704, 768, 832,  896,  960,  1024, 1280,
        1536, 1792, 2048};
    for (int64_t b : kBuckets) {
      if (b >= n) return b;
    }
    return n;
  };
  const int64_t q_cap = std::max(
      {query_.defined() ? query_.size(0) : int64_t{0},
       key_.defined() ? key_.size(0) : int64_t{0},
       output_.defined() ? output_.size(0) : int64_t{0},
       piecewise_bucket_for(std::max(max_seqlen_q, int64_t{1}))});
  if (output_lse_.defined() && q_cap > 0) {
    if (output_lse_.dim() == 2) {
      const int64_t heads = output_lse_.size(0);
      if (output_lse_.size(1) < q_cap) {
        auto expanded = torch::empty({heads, q_cap}, output_lse_.options());
        expanded.slice(/*dim=*/1, /*start=*/0, /*end=*/output_lse_.size(1))
            .copy_(output_lse_);
        output_lse_ = std::move(expanded);
      }
    } else if (output_lse_.dim() == 1) {
      // Contiguous heads*tokens packing used by some callers.
      // Infer heads from numel / current tokens if possible; else grow by
      // assuming rows match query.
      int64_t cur_tokens = max_seqlen_q > 0 ? max_seqlen_q : int64_t{1};
      if (output_lse_.numel() % cur_tokens == 0) {
        const int64_t heads = output_lse_.numel() / cur_tokens;
        const int64_t need = heads * q_cap;
        if (output_lse_.numel() < need) {
          auto expanded = torch::empty({need}, output_lse_.options());
          expanded.narrow(/*dim=*/0, /*start=*/0, /*length=*/output_lse_.numel())
              .copy_(output_lse_);
          output_lse_ = std::move(expanded);
        }
      }
    }
  }

  ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_reshape_paged_cache_capture(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping) {
  auto& capture = ::xllm::runtime::musa::GlobalCaptureInstance::get_instance();
  capture.temporarily_end_graph();
  key_ = std::move(key);
  value_ = std::move(value);
  k_cache_ = std::move(k_cache);
  v_cache_ = std::move(v_cache);
  slot_mapping_ = std::move(slot_mapping);
  runner_type_ = RunnerType::RESHAPE_PAGED_CACHE;
  LOG(WARNING) << "RESHAPE_CAPTURE_SIZES key=" << key_.sizes()
               << " value=" << value_.sizes()
               << " slot=" << slot_mapping_.sizes();
  xllm::kernel::ReshapePagedCacheParams params;
  params.key = key_;
  params.value = value_;
  params.k_cache = k_cache_;
  params.v_cache = v_cache_;
  params.slot_mapping = slot_mapping_;
  xllm::kernel::reshape_paged_cache(params);
  capture.temporarily_begin_graph();
}


void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  if (runner_type_ == RunnerType::RESHAPE_PAGED_CACHE) {
    const torch::Tensor& slots = params.slot_mapping.defined()
                                     ? params.slot_mapping
                                     : slot_mapping_;
    CHECK(slots.defined()) << "reshape replay requires slot_mapping";
    CHECK(key_.defined() && value_.defined())
        << "reshape replay requires key/value buffers";
    CHECK(k_cache_.defined() && v_cache_.defined())
        << "reshape replay requires KV cache";
    xllm::kernel::ReshapePagedCacheParams rp;
    rp.key = key_;
    rp.value = value_;
    rp.k_cache = k_cache_;
    rp.v_cache = v_cache_;
    rp.slot_mapping = slots;
    xllm::kernel::reshape_paged_cache(rp);
    return;
  }

  if (runner_type_ == RunnerType::GDN_PREFILL) {
    run_gdn_prefill_replay(params);
    return;
  }
  torch::Tensor query_slice =
      query_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor output_slice =
      output_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);

  if (runner_type_ == RunnerType::FA3_PAGED_PREFILL) {
    CHECK(output_lse_.defined())
        << "FA3 paged prefill replay requires output LSE";
    CHECK(params.q_cu_seq_lens.defined())
        << "FA3 paged prefill replay requires query CU lengths";
    CHECK(params.kv_cu_seq_lens.defined())
        << "FA3 paged prefill replay requires KV CU lengths";
    CHECK(params.block_table.defined())
        << "FA3 paged prefill replay requires a block table";
    torch::Tensor seqused_k = params.kv_seq_lens.defined()
                                  ? params.kv_seq_lens
                                  : torch::diff(params.kv_cu_seq_lens);
    if (seqused_k.scalar_type() != torch::kInt32) {
      seqused_k = seqused_k.to(torch::kInt32);
    }
    seqused_k = seqused_k.contiguous();
    torch::Tensor query_slice =
        query_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens)
            .contiguous();
    torch::Tensor output_slice =
        output_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens)
            .contiguous();
    const int64_t num_heads = query_slice.size(1);
    const int64_t num_kv_heads = k_cache_.size(2);
    const int64_t head_dim = query_slice.size(2);
    const int64_t required_lse_elements = num_heads * params.actual_num_tokens;
    CHECK_GE(output_lse_.numel(), required_lse_elements);
    torch::Tensor output_lse = output_lse_.view({-1})
                                   .narrow(/*dim=*/0,
                                           /*start=*/0,
                                           /*length=*/required_lse_elements)
                                   .view({num_heads, params.actual_num_tokens})
                                   .contiguous();
    const int64_t max_seqlen_q =
        params.max_seqlen_q > 0 ? params.max_seqlen_q : query_slice.size(0);
    const int64_t max_seqlen_k =
        params.max_seqlen_k > 0 ? params.max_seqlen_k : 1;
    const int32_t batch_size = static_cast<int32_t>(seqused_k.numel());
    torch::Tensor scheduler_metadata =
        fa3_prefill_scheduler_metadata(query_slice.device(),
                                       batch_size,
                                       static_cast<int32_t>(num_heads),
                                       static_cast<int32_t>(num_kv_heads),
                                       static_cast<int32_t>(head_dim),
                                       static_cast<int32_t>(head_dim),
                                       static_cast<int32_t>(max_seqlen_q),
                                       static_cast<int32_t>(max_seqlen_k),
                                       static_cast<int32_t>(window_size_left_),
                                       static_cast<int32_t>(window_size_right_),
                                       params.q_cu_seq_lens.contiguous(),
                                       params.kv_cu_seq_lens.contiguous(),
                                       seqused_k);
    fa3_prefill_paged(query_slice,
                      k_cache_,
                      v_cache_,
                      params.q_cu_seq_lens.contiguous(),
                      params.kv_cu_seq_lens.contiguous(),
                      seqused_k,
                      params.block_table.contiguous(),
                      scheduler_metadata,
                      max_seqlen_q,
                      window_size_left_,
                      window_size_right_,
                      scale_,
                      output_slice,
                      output_lse);
    return;
  }

  if (runner_type_ == RunnerType::FA3_PREFILL) {
    torch::Tensor key_slice =
        key_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
    torch::Tensor value_slice =
        value_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
    CHECK(output_lse_.defined()) << "FA3 prefill replay requires output LSE";
    CHECK_EQ(output_lse_.dim(), 2)
        << "FA3 prefill replay requires a 2D output LSE buffer";
    CHECK(output_lse_.is_contiguous())
        << "FA3 prefill replay requires a contiguous output LSE buffer";
    const int64_t num_heads = output_lse_.size(0);
    const int64_t required_lse_elements = num_heads * params.actual_num_tokens;
    if (output_lse_.numel() < required_lse_elements) {
      // Capture-time LSE was sized to the capture request, not the bucket.
      // FA3 runs eagerly between graph segments, so grow to bucket capacity.
      output_lse_ = torch::empty(
          {num_heads, std::max<int64_t>(query_.size(0), params.actual_num_tokens)},
          output_lse_.options());
    }
    torch::Tensor output_lse = output_lse_.view({-1})
                                   .narrow(/*dim=*/0,
                                           /*start=*/0,
                                           /*length=*/required_lse_elements)
                                   .view({num_heads, params.actual_num_tokens});

    const int64_t max_seqlen_q =
        params.max_seqlen_q > 0 ? params.max_seqlen_q : max_seqlen_q_;
    const int64_t max_seqlen_k =
        params.max_seqlen_k > 0 ? params.max_seqlen_k : max_seqlen_k_;
    CHECK_GT(max_seqlen_q, 0);
    CHECK_GT(max_seqlen_k, 0);

    // The model normally produces contiguous projection views. Preserve the
    // existing eager FA3 behavior for the uncommon non-contiguous case.
    torch::Tensor query_contiguous = query_slice.contiguous();
    torch::Tensor key_contiguous = key_slice.contiguous();
    torch::Tensor value_contiguous = value_slice.contiguous();
    const torch::Tensor& live_cu_seq_lens = params.gdn_cu_seq_lens.defined()
                                                ? params.gdn_cu_seq_lens
                                                : params.q_cu_seq_lens;
    torch::Tensor q_cu_seq_lens = live_cu_seq_lens.contiguous();
    torch::Tensor kv_cu_seq_lens = params.gdn_cu_seq_lens.defined()
                                       ? params.gdn_cu_seq_lens.contiguous()
                                       : params.kv_cu_seq_lens.contiguous();
    fa3_prefill(query_contiguous,
                key_contiguous,
                value_contiguous,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                max_seqlen_q,
                max_seqlen_k,
                window_size_left_,
                window_size_right_,
                scale_,
                output_slice,
                output_lse);
    return;
  }

  // TODO: support output_lse for replay
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (runner_type_ == RunnerType::CHUNKED_PREFILL) {
    batch_chunked_prefill(uri_,
                          params.plan_info,
                          float_workspace_buffer_,
                          int_workspace_buffer_,
                          page_locked_int_workspace_buffer_,
                          query_slice,
                          k_cache_,
                          v_cache_,
                          params.paged_kv_indptr,
                          params.paged_kv_indices,
                          params.paged_kv_last_page_len,
                          window_size_left_,
                          scale_,
                          output_slice,
                          output_lse,
                          params.qo_indptr,
                          causal_,
                          params.paged_kv_indptr_host,
                          params.paged_kv_indices_host,
                          params.paged_kv_last_page_len_host);
    return;
  }

  torch::Tensor key_slice =
      key_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor value_slice =
      value_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  batch_prefill(uri_,
                params.plan_info,
                float_workspace_buffer_,
                int_workspace_buffer_,
                page_locked_int_workspace_buffer_,
                query_slice,
                key_slice,
                value_slice,
                params.q_cu_seq_lens,
                params.kv_cu_seq_lens,
                window_size_left_,
                scale_,
                output_slice,
                output_lse);
}

bool AttentionRunner::requires_plan_info() const {
  return runner_type_ == RunnerType::PREFILL ||
         runner_type_ == RunnerType::CHUNKED_PREFILL;
}

bool AttentionRunner::is_gdn_prefill() const {
  return runner_type_ == RunnerType::GDN_PREFILL;
}

void AttentionRunner::run_gdn_prefill_capture(torch::Tensor query,
                                              torch::Tensor key,
                                              torch::Tensor value,
                                              torch::Tensor gate,
                                              torch::Tensor beta,
                                              torch::Tensor initial_state,
                                              torch::Tensor cu_seqlens,
                                              torch::Tensor output,
                                              torch::Tensor final_state,
                                              torch::Tensor kkt_output,
                                              torch::Tensor ssm_cache,
                                              torch::Tensor state_indices,
                                              float scale) {
  auto& capture = ::xllm::runtime::musa::GlobalCaptureInstance::get_instance();
  capture.temporarily_end_graph();
  query_ = std::move(query);
  key_ = std::move(key);
  value_ = std::move(value);
  gdn_gate_ = std::move(gate);
  gdn_beta_ = std::move(beta);
  gdn_initial_state_ = std::move(initial_state);
  gdn_cu_seq_lens_ = std::move(cu_seqlens);
  gdn_output_ = std::move(output);
  gdn_final_state_ = std::move(final_state);
  gdn_kkt_output_ = std::move(kkt_output);
  gdn_ssm_cache_ = std::move(ssm_cache);
  gdn_state_indices_ = std::move(state_indices);
  scale_ = scale;
  runner_type_ = RunnerType::GDN_PREFILL;
  capture.temporarily_begin_graph();
}

void AttentionRunner::run_gdn_prefill_replay(
    const AttentionReplayParams& params) {
  const torch::Tensor& live_cu = params.gdn_cu_seq_lens.defined()
                                     ? params.gdn_cu_seq_lens
                                     : params.q_cu_seq_lens;
  const torch::Tensor& kkt_cu = params.gdn_kkt_cu_seq_lens.defined()
                                    ? params.gdn_kkt_cu_seq_lens
                                    : live_cu;
  CHECK(live_cu.defined()) << "GDN runner requires live device cu_seqlens";
  CHECK_EQ(live_cu.scalar_type(), torch::kInt32);
  CHECK(live_cu.is_contiguous());
  CHECK(kkt_cu.defined() && kkt_cu.is_contiguous());
  CHECK_EQ(kkt_cu.scalar_type(), torch::kInt32);
  CHECK_EQ(kkt_cu.size(0), live_cu.size(0));
  CHECK(!params.q_cu_seq_lens_host.empty())
      << "GDN runner requires host cu_seqlens";
  CHECK_EQ(params.q_cu_seq_lens_host.front(), 0);
  CHECK_EQ(static_cast<uint32_t>(params.q_cu_seq_lens_host.back()),
           params.actual_num_tokens)
      << "GDN host CU endpoint must match actual token count";
  CHECK_EQ(live_cu.size(0),
           static_cast<int64_t>(params.q_cu_seq_lens_host.size()));
  CHECK_EQ(query_.size(1), key_.size(1));
  CHECK_EQ(query_.size(1), value_.size(1));
  CHECK_EQ(query_.size(1), gdn_gate_.size(1));
  CHECK_EQ(query_.size(1), gdn_beta_.size(1));
  CHECK_EQ(gdn_output_.size(1), query_.size(1));
  CHECK_EQ(gdn_final_state_.size(0),
           static_cast<int64_t>(params.q_cu_seq_lens_host.size()) - 1);

  CHECK_GT(params.gdn_kkt_num_tokens, 0);
  const int64_t mate_num_tokens = params.gdn_kkt_num_tokens;
  CHECK_GE(mate_num_tokens, params.actual_num_tokens);
  CHECK_EQ(mate_num_tokens % kGdnChunkSize, 0);
  CHECK_LE(mate_num_tokens, gdn_output_.size(1));
  if (params.actual_num_tokens < mate_num_tokens) {
    const int64_t tail_start = params.actual_num_tokens;
    const int64_t tail_length = mate_num_tokens - tail_start;
    query_.narrow(/*dim=*/1, tail_start, tail_length).zero_();
    key_.narrow(/*dim=*/1, tail_start, tail_length).zero_();
    value_.narrow(/*dim=*/1, tail_start, tail_length).zero_();
    gdn_gate_.narrow(/*dim=*/1, tail_start, tail_length).zero_();
    gdn_beta_.narrow(/*dim=*/1, tail_start, tail_length).zero_();
  }
  if (params.actual_num_tokens < gdn_output_.size(1)) {
    gdn_output_
        .narrow(/*dim=*/1,
                /*start=*/params.actual_num_tokens,
                /*length=*/gdn_output_.size(1) - params.actual_num_tokens)
        .zero_();
  }

  MateGatedDeltaRulePrefillParams mate_params;
  mate_params.q = canonical_time_prefix(query_, mate_num_tokens);
  mate_params.k = canonical_time_prefix(key_, mate_num_tokens);
  mate_params.v = canonical_time_prefix(value_, mate_num_tokens);
  mate_params.g = canonical_time_prefix(gdn_gate_, mate_num_tokens);
  mate_params.beta = canonical_time_prefix(gdn_beta_, mate_num_tokens);
  mate_params.scale = static_cast<float>(scale_);
  mate_params.initial_state = gdn_initial_state_;
  mate_params.cu_seqlens = live_cu;
  // Merge MateGatedDeltaRulePrefillParams has no cu_seqlens_kkt /
  // output_final_state; kkt goes via kkt_output when defined.
  (void)kkt_cu;
  mate_params.cu_seqlens_host = params.q_cu_seq_lens_host;
  mate_params.output = canonical_time_prefix(gdn_output_, mate_num_tokens);
  mate_params.final_state = gdn_final_state_;
  mate_params.use_qk_l2norm_in_kernel = true;
  mate_params.allow_inplace_qk_l2norm = true;
  if (gdn_kkt_output_.defined()) {
    mate_params.kkt_output =
        canonical_time_prefix(gdn_kkt_output_, mate_num_tokens);
  }
  auto result = mate_gated_delta_rule_prefill(mate_params);
  CHECK(result.first.defined()) << "GDN runner output is undefined";
  CHECK(result.second.defined()) << "GDN runner final state is undefined";
  CHECK_EQ(result.first.data_ptr(), gdn_output_.data_ptr());
  CHECK_EQ(result.second.data_ptr(), gdn_final_state_.data_ptr());
  if (gdn_ssm_cache_.defined() && gdn_state_indices_.defined()) {
    gdn_ssm_cache_.index_put_({gdn_state_indices_},
                              gdn_final_state_.to(gdn_ssm_cache_.dtype()));
  }
}

void batch_chunked_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr,
    bool causal,
    const torch::Tensor& paged_kv_indptr_host,
    const torch::Tensor& paged_kv_indices_host,
    const torch::Tensor& paged_kv_last_page_len_host) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    AttentionRunner runner;
    const uint32_t padded_num_tokens =
        static_cast<uint32_t>(query.size(/*dim=*/0));
    runner.run_chunked_prefill_capture(uri,
                                       plan_info,
                                       float_workspace_buffer,
                                       int_workspace_buffer,
                                       page_locked_int_workspace_buffer,
                                       query,
                                       k_cache,
                                       v_cache,
                                       paged_kv_indptr,
                                       paged_kv_indices,
                                       paged_kv_last_page_len,
                                       window_left,
                                       sm_scale,
                                       output,
                                       output_lse,
                                       qo_indptr,
                                       causal,
                                       paged_kv_indptr_host,
                                       paged_kv_indices_host,
                                       paged_kv_last_page_len_host,
                                       padded_num_tokens);
    ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }

  batch_chunked_prefill(uri,
                        plan_info,
                        float_workspace_buffer,
                        int_workspace_buffer,
                        page_locked_int_workspace_buffer,
                        query,
                        k_cache,
                        v_cache,
                        paged_kv_indptr,
                        paged_kv_indices,
                        paged_kv_last_page_len,
                        window_left,
                        sm_scale,
                        output,
                        output_lse,
                        qo_indptr,
                        causal,
                        paged_kv_indptr_host,
                        paged_kv_indices_host,
                        paged_kv_last_page_len_host);
}

void fa3_prefill_with_optional_piecewise_capture(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t window_left,
    int64_t window_right,
    double sm_scale,
    torch::Tensor& output,
    torch::Tensor& output_lse) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    // For C1 with bounded bucket padding, record dense FA3 in the surrounding
    // graph instead of splitting at every full-attention layer. Larger padding
    // deltas keep the dynamic runner: captured FA3 launch geometry was verified
    // only within the same <=32-token safety bound used by Qwen3.5 C1 routing.
    const int64_t capture_padding = query.size(0) - max_seqlen_q;
    const int64_t max_capture_padding = capture_fa3_max_padding_tokens();
    const bool safe_c1_shape =
        cu_seqlens_q.numel() == 2 && cu_seqlens_k.numel() == 2 &&
        max_seqlen_q == max_seqlen_k && capture_padding >= 0 &&
        (max_capture_padding < 0 || capture_padding <= max_capture_padding);
    // Do NOT inline FA3 into the surrounding MUSA graph. Inlining freezes
    // cu_seqlens / launch geometry (and any reshape_paged_cache that ran in
    // the same capture segment) at capture-time actual_num_tokens. Replay of
    // a larger request in the same bucket then OOBs in
    // reshape_paged_cache_kernel. Always use an AttentionRunner so live
    // metadata is rebound every replay. (safe_c1_shape / capture_fa3_* kept
    // for logging / future use.)
    (void)safe_c1_shape;
    (void)capture_fa3_in_piecewise_graph();
    (void)capture_fa3_max_padding_tokens();

    AttentionRunner runner;
    runner.run_fa3_prefill_capture(query,
                                   key,
                                   value,
                                   max_seqlen_q,
                                   max_seqlen_k,
                                   window_left,
                                   window_right,
                                   sm_scale,
                                   output,
                                   output_lse);
    ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }

  torch::Tensor query_contiguous = query.contiguous();
  torch::Tensor key_contiguous = key.contiguous();
  torch::Tensor value_contiguous = value.contiguous();
  torch::Tensor q_cu_seq_lens = cu_seqlens_q.contiguous();
  torch::Tensor kv_cu_seq_lens = cu_seqlens_k.contiguous();
  fa3_prefill(query_contiguous,
              key_contiguous,
              value_contiguous,
              q_cu_seq_lens,
              kv_cu_seq_lens,
              max_seqlen_q,
              max_seqlen_k,
              window_left,
              window_right,
              sm_scale,
              output,
              output_lse);
}

bool piecewise_graph_is_capturing() {
  return ::xllm::ExecutionConfig::get_instance().enable_graph() &&
         ::xllm::ExecutionConfig::get_instance()
             .enable_prefill_piecewise_graph() &&
         ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
             .is_capturing();
}

void fa3_prefill_paged_with_optional_piecewise_capture(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k_new,
    const torch::Tensor& seqused_k,
    const torch::Tensor& page_table,
    const torch::Tensor& scheduler_metadata,
    int64_t max_seqlen_q,
    int64_t window_left,
    int64_t window_right,
    double sm_scale,
    torch::Tensor& output,
    torch::Tensor& output_lse) {
  if (piecewise_graph_is_capturing()) {
    AttentionRunner runner;
    runner.run_fa3_paged_prefill_capture(query,
                                         k_cache,
                                         v_cache,
                                         window_left,
                                         window_right,
                                         sm_scale,
                                         output,
                                         output_lse);
    ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }

  fa3_prefill_paged(query,
                    k_cache,
                    v_cache,
                    cu_seqlens_q,
                    cu_seqlens_k_new,
                    seqused_k,
                    page_table,
                    scheduler_metadata,
                    max_seqlen_q,
                    window_left,
                    window_right,
                    sm_scale,
                    output,
                    output_lse);
}

void batch_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    AttentionRunner runner;

    uint32_t padded_num_tokens = static_cast<uint32_t>(query.size(0));

    runner.run_capture(uri,
                       plan_info,
                       float_workspace_buffer,
                       int_workspace_buffer,
                       page_locked_int_workspace_buffer,
                       query,
                       key,
                       value,
                       q_cu_seq_lens,
                       kv_cu_seq_lens,
                       window_left,
                       sm_scale,
                       output,
                       output_lse,
                       padded_num_tokens);

    ::xllm::runtime::musa::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }
  batch_prefill(uri,
                plan_info,
                float_workspace_buffer,
                int_workspace_buffer,
                page_locked_int_workspace_buffer,
                query,
                key,
                value,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                window_left,
                sm_scale,
                output,
                output_lse);
}

}  // namespace musa
}  // namespace kernel
}  // namespace xllm
