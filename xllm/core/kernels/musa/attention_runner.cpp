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

#include <glog/logging.h>

#include <cstdlib>
#include <string>

namespace {

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

namespace xllm {
namespace kernel {
namespace cuda {

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

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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
  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  if (runner_type_ == RunnerType::GDN_PREFILL) {
    run_gdn_prefill_replay(params);
    return;
  }
  torch::Tensor query_slice =
      query_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor output_slice =
      output_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);

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
    CHECK_GE(output_lse_.numel(), required_lse_elements);
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
    torch::Tensor q_cu_seq_lens = params.q_cu_seq_lens.contiguous();
    torch::Tensor kv_cu_seq_lens = params.kv_cu_seq_lens.contiguous();
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
                                              float scale) {
  auto& capture = ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance();
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

  if (params.actual_num_tokens < gdn_output_.size(1)) {
    gdn_output_
        .narrow(/*dim=*/1,
                /*start=*/params.actual_num_tokens,
                /*length=*/gdn_output_.size(1) - params.actual_num_tokens)
        .zero_();
  }

  MateGatedDeltaRulePrefillParams mate_params;
  mate_params.q = query_;
  mate_params.k = key_;
  mate_params.v = value_;
  mate_params.g = gdn_gate_;
  mate_params.beta = gdn_beta_;
  mate_params.scale = static_cast<float>(scale_);
  mate_params.initial_state = gdn_initial_state_;
  mate_params.cu_seqlens = live_cu;
  mate_params.cu_seqlens_kkt = kkt_cu;
  mate_params.cu_seqlens_host = params.q_cu_seq_lens_host;
  mate_params.output = gdn_output_;
  mate_params.final_state = gdn_final_state_;
  mate_params.output_final_state = true;
  mate_params.use_qk_l2norm_in_kernel = true;
  mate_params.allow_inplace_qk_l2norm = true;
  // KKT writes its own temporary `a` when no external buffer is supplied.
  // The graph-owned output/final-state buffers are the critical stable outputs;
  // kkt_output is populated by the layer when the pool supports it.
  if (gdn_kkt_output_.defined()) {
    mate_params.kkt_output = gdn_kkt_output_;
  }
  auto result = mate_gated_delta_rule_prefill(mate_params);
  CHECK(result.first.defined()) << "GDN runner output is undefined";
  CHECK(result.second.defined()) << "GDN runner final state is undefined";
  CHECK_EQ(result.first.data_ptr(), gdn_output_.data_ptr());
  CHECK_EQ(result.second.data_ptr(), gdn_final_state_.data_ptr());
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
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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
    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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
    if (capture_fa3_in_piecewise_graph() && safe_c1_shape) {
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
      return;
    }

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
    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
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

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
