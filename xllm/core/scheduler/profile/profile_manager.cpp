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

#include "profile_manager.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request_state.h"
#include "platform/platform.h"
#include "runtime/acl_graph_bucket_policy.h"
#include "scheduler/profile/graph_warmup.h"
#include "framework/config/parallel_config.h"
#if defined(USE_MUSA)
#include "core/framework/block/musa/deepseek_v4_prefix_transfer.h"
#include "layers/musa/pcp_context.h"
#endif
#include "util/env_var.h"
#include "util/rec_model_utils.h"
#include "util/utils.h"

namespace xllm {

std::vector<int32_t> build_step_time_profile_batch_sizes(
    int32_t max_seqs_per_batch) {
  CHECK_GT(max_seqs_per_batch, 0);
  constexpr int32_t kLegacyMaxProfileBatchSize = 23;
  const int32_t max_profile_batch_size =
      std::min(max_seqs_per_batch, kLegacyMaxProfileBatchSize);
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(max_profile_batch_size / 2 + 2);
  for (int32_t batch_size = 1; batch_size <= max_profile_batch_size;
       batch_size += 2) {
    batch_sizes.push_back(batch_size);
  }
  if (batch_sizes.back() != max_profile_batch_size) {
    batch_sizes.push_back(max_profile_batch_size);
  }
  return batch_sizes;
}

int32_t warmup_decode_seq_len(bool fia_enabled, int32_t max_context_len) {
  // FIA decode graphs refresh dynamic host parameters before every replay and
  // run captures beyond the default 16 decode sequence lengths, so warmup
  // measures capacity with the FIA warmup bound to keep measured memory and
  // the final warmup batch capacity consistent with real captures.
  constexpr int32_t kDefaultWarmupDecodeSeqLen = 16;
  constexpr int32_t kFiaWarmupDecodeSeqLen = 2048;
  return std::min(
      fia_enabled ? kFiaWarmupDecodeSeqLen : kDefaultWarmupDecodeSeqLen,
      max_context_len);
}

namespace {

int32_t decode_warmup_token_bucket(const DecodeGraphWarmupPlan& plan,
                                   int32_t global_batch_size,
                                   int32_t dp_size) {
  CHECK_GT(global_batch_size, 0);
  CHECK_GT(dp_size, 0);
  CHECK_GT(plan.warmup_config.num_decoding_tokens, 0);
  const int32_t local_sequence_batch_size =
      (global_batch_size + dp_size - 1) / dp_size;
  const int64_t num_token_rows =
      static_cast<int64_t>(local_sequence_batch_size) *
      plan.warmup_config.num_decoding_tokens;
  return static_cast<int32_t>(runtime::get_decode_graph_token_bucket(
      num_token_rows, plan.warmup_config.enable_graph_mode_decode_no_padding));
}

}  // namespace

ProfileManager::ProfileManager(Engine* engine, const Options& options)
    : options_(options), engine_(engine) {
  CHECK(engine_ != nullptr);
  int32_t max_decode_batch_size = options_.max_seqs_per_batch();
  const int32_t max_concurrent_requests =
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests();
  if (max_concurrent_requests > 0) {
    max_decode_batch_size =
        std::min(max_decode_batch_size, max_concurrent_requests);
  }
  if (Platform::is_npu()) {
    max_decode_batch_size =
        static_cast<int32_t>(npu::acl_graph_max_global_batch_size(
            static_cast<uint32_t>(std::max<int32_t>(1, max_decode_batch_size)),
            static_cast<uint32_t>(
                std::max<int32_t>(1,
                                  ::xllm::ExecutionConfig::get_instance()
                                      .acl_graph_decode_batch_size_limit())),
            static_cast<uint32_t>(std::max<int32_t>(1, options_.dp_size()))));
  }
  decode_graph_warmup_plan_ =
      build_decode_graph_warmup_plan(engine_->decode_graph_warmup_config(),
                                     max_decode_batch_size,
                                     options_.dp_size());
  block_manager_pool_ = engine_->block_manager_pool();
  CHECK(block_manager_pool_ != nullptr);
  prefill_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), true /*is_prefill*/);
  decode_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), false /*is_prefill*/);
  speculative_validate_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), false /*is_prefill*/);
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(false);
    // The validate-time predictor is only consumed by the adaptive
    // speculative controller, so skip the whole prefix/query/batch sweep
    // unless adaptive speculative decode is actually enabled.
    if (should_profile_speculative_validate()) {
      profile_speculative_validate_time();
    }
    // test accuracy
    // eval_sequence_latency_prediction();
    // eval_batch_latency_prediction("only_prefill");
    // eval_batch_latency_prediction("only_decode");
    // eval_batch_latency_prediction("mix");
  }
  if (options.enable_profile_token_budget()) {
    LOG(INFO) << "Starting profiliing token budget.";
    profile_token_budget();
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  if (!is_rec_multi_round_mode()) {
    const auto& execution_config = ::xllm::ExecutionConfig::get_instance();
    if (execution_config.enable_graph()) {
      if (execution_config.disable_graph_warmup()) {
        LOG(INFO) << "Graph warmup disabled by execution config; graphs will "
                     "be captured lazily from real requests";
      } else {
        warmup_for_graph();
      }
#if defined(USE_NPU)
    } else if (options_.instance_role() == InstanceRole::DECODE) {
      LOG(INFO) << "Skipping eager warmup for decode-only instance";
    } else {
      warmup_for_eager();
#endif
    }
  }
#elif defined(USE_MUSA)
  if (!is_rec_multi_round_mode()) {
    const auto& execution_config = ::xllm::ExecutionConfig::get_instance();
    if (execution_config.enable_graph() &&
        !execution_config.disable_graph_warmup()) {
      const GraphWarmupPlan plan = graph_warmup_plan(options_.instance_role());
      if (plan == GraphWarmupPlan::PREFILL_ONLY) {
        if (execution_config.enable_prefill_piecewise_graph()) {
          warmup_prefill_for_graph();
        }
      } else if (plan == GraphWarmupPlan::DECODE_ONLY) {
        warmup_decode_for_graph();
      } else {
        if (execution_config.enable_prefill_piecewise_graph()) {
          warmup_prefill_for_graph();
        }
        warmup_decode_for_graph();
      }
    }
  }
#endif
}

// --------------------- for test only ---------------------------
void ProfileManager::eval_sequence_latency_prediction() {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;
  int32_t token_step = 500;
  int32_t prefix_step = 500;
  int32_t upper_bound = 4000;

  LOG(INFO) << "Starting testing sequence latency prediction";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(run_request(token_length, prefix_length));
      pred_vec.emplace_back(predict_step_time(token_length, prefix_length));
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
void ProfileManager::eval_batch_latency_prediction(const std::string mode) {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;

  LOG(INFO) << "Starting testing batch latency prediction for " << mode;
  if (mode == "only_prefill") {
    int32_t max_batch_size = 10;
    int32_t token_step = 500;
    int32_t prefix_step = 500;
    int32_t upper_bound = 4000;
    for (int32_t token_length = token_step; token_length < upper_bound;
         token_length += token_step) {
      for (int32_t prefix_length = 0; prefix_length < token_length;
           prefix_length += prefix_step) {
        target_vec.emplace_back(
            run_request(token_length, prefix_length, max_batch_size));
        pred_vec.emplace_back(
            predict_step_time(token_length, prefix_length, max_batch_size));
      }
    }
  }
  if (mode == "only_decode") {
    int32_t max_batch_size = 200;
    int32_t token_length = 500;
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size++) {
      target_vec.emplace_back(
          run_request(token_length, token_length - 1, batch_size));
      pred_vec.emplace_back(
          predict_step_time(token_length, token_length - 1, batch_size));
    }
  }
  if (mode == "mix") {
    if (!::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      LOG(WARNING) << "When chunked prefill is disabled, mixed prefill and "
                      "decode scenarios will not be tested.";
      return;
    }
    int32_t max_batch_size = 100;
    int32_t max_prefill_cnt = 5;
    int32_t token_length = 500;
    for (int32_t batch_size = 50; batch_size <= max_batch_size;
         batch_size += 10) {
      for (int32_t prefill_cnt = 0; prefill_cnt <= max_prefill_cnt;
           prefill_cnt++) {
        std::vector<int32_t> token_length_vec;
        std::vector<int32_t> prefix_length_vec;
        token_length_vec.insert(
            token_length_vec.end(), prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(), prefill_cnt, 0);
        // token_length_vec.insert(token_length_vec.end(), batch_size/5,
        // token_length); prefix_length_vec.insert(prefix_length_vec.end(),
        // batch_size/5, token_length-1);
        token_length_vec.insert(
            token_length_vec.end(), batch_size - prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(),
                                 batch_size - prefill_cnt,
                                 token_length - 1);
        target_vec.emplace_back(
            run_request(token_length_vec, prefix_length_vec));
        pred_vec.emplace_back(
            predict_step_time(token_length_vec, prefix_length_vec));
      }
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
// -------------------------------------------------------------

// ---------------------- dump to file-----------------------
std::string ProfileManager::generate_filename(const std::string& file_suffix) {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

  std::string filename;
  filename = ss.str() + "_" + file_suffix + ".txt";

  return filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::pair<int32_t, double>>& time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << data.first << "," << data.second << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << std::get<0>(data) << "," << std::get<1>(data) << ","
            << std::get<2>(data) << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}
// -------------------------------------------------------------

void ProfileManager::profile_step_time(bool if_dump_to_file) {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // TODO: support length for decode request profile
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  auto block_size = block_manager_pool_->options().block_size();
  bool enable_profile_kv_blocks = options_.enable_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0);

  // prefill time profile
  if (options_.enable_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      // increase prefix length according to block size
      auto block_step = (profile_length_step_ + block_size - 1) / block_size;
      for (int32_t prefix_length = 0;
           prefix_length < token_length - 1 + (block_step * block_size);
           prefix_length += (block_step * block_size)) {
        if (prefix_length > token_length - 1) {
          // avoid kv_cache_token_num == token_length
          prefix_length = token_length - 1;
        }
        double latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step_; k++) {
          latency_mean += run_request(token_length, prefix_length);
        }
        latency_mean /= profile_count_per_step_;
        // use token_length and prefix_length to predict
        time_profiling_data.emplace_back(
            token_length, prefix_length, latency_mean);
      }
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  } else {
    // not consider kv cache
    std::vector<std::pair<int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length *= 0.8) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, 0);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, latency_mean);
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  }
  if (::xllm::DisaggPDConfig::get_instance().enable_disagg_pd()) {
    LOG(INFO) << "Disagg PD enabled, skip decode time profile.";
    return;
  }
  // decode time profile

  std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
  // for (int32_t token_length = profile_max_prompt_length; token_length >
  // 1;token_length >>= 1)
  const std::vector<int32_t> profile_batch_sizes =
      build_step_time_profile_batch_sizes(options_.max_seqs_per_batch());
  for (int32_t token_length = 2; token_length < profile_max_prompt_length;
       token_length += profile_length_step_) {
    for (int32_t batch_size : profile_batch_sizes) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, token_length - 1, batch_size);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, batch_size, latency_mean);
    }
  }
  if (if_dump_to_file) {
    dump_step_time_profile_to_file(time_profiling_data, false /*is_prefill*/);
  }
  train_decode_time_predictor(time_profiling_data);
}

void ProfileManager::train_prefill_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_prefill_time_predictor(
    std::vector<std::pair<int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_decode_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  decode_time_predictor_->fit_for_decode(time_profiling_data);
}

void ProfileManager::train_speculative_validate_time_predictor(
    const std::vector<std::tuple<int32_t, int32_t, int32_t, double>>&
        time_profiling_data) {
  if (time_profiling_data.empty()) {
    return;
  }

  // The least-squares fit + error metrics + coefficient sanitization live in
  // TimePredictor (shared with the prefill/decode predictors). Here we only
  // map the fitted coefficients into the registry struct and publish them.
  speculative_validate_time_predictor_->fit_for_speculative_validate(
      time_profiling_data);
  const std::vector<double> coefficients =
      speculative_validate_time_predictor_->get_coefficients();
  CHECK_EQ(coefficients.size(), 3u)
      << "speculative validate predictor must have 3 coefficients";

  SpeculativeProfileRegistry::ValidateTimePredictor predictor;
  predictor.intercept_ms = coefficients[0];
  predictor.query_token_ms = coefficients[1];
  predictor.query_prefix_ms = coefficients[2];
  // Broadcast to workers FIRST, then commit locally. Workers gate the
  // adaptive path on their own SpeculativeProfileRegistry, so any rank
  // that misses the predictor will diverge from ranks that received it:
  // one side runs adaptive (per-seq variable validate width), the other
  // runs static, which corrupts collectives and shape assumptions.
  // Treat broadcast failure as fatal for the adaptive path and leave the
  // registry unset so every rank consistently falls back to static.
  if (!engine_->set_speculative_validate_time_predictor(predictor)) {
    LOG(ERROR)
        << "Failed to broadcast speculative validate predictor to workers. "
        << "Disabling adaptive speculative decode on all ranks to avoid "
        << "cross-rank divergence.";
    SpeculativeProfileRegistry::get_instance().reset_validate_time_predictor();
    return;
  }
  SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
      predictor);
}

bool ProfileManager::should_profile_speculative_validate() const {
  // The validate-time predictor is only consumed by the adaptive speculative
  // controller: it needs adaptive explicitly enabled, more than one
  // speculative token to prune, and a supported algorithm
  // (MTP / DFlash / DSpark). Otherwise the prefix/query/batch sweep is wasted
  // startup time.
  const SpeculativeConfig& speculative_config =
      ::xllm::SpeculativeConfig::get_instance();
  const std::string& speculative_algorithm =
      speculative_config.speculative_algorithm();
  const bool is_supported_algo =
      SpeculativeConfig::is_mtp_algorithm(speculative_algorithm) ||
      SpeculativeConfig::is_block_diffusion_algorithm(speculative_algorithm);
  return speculative_config.enable_adaptive_speculative_decode() &&
         speculative_config.num_speculative_tokens() > 1 && is_supported_algo;
}

void ProfileManager::profile_speculative_validate_time() {
  // The adaptive gate lives at the call site (should_profile_speculative_
  // validate); guard here too so a direct call cannot profile a config the
  // adaptive controller would never consume.
  if (!should_profile_speculative_validate()) {
    return;
  }
  const SpeculativeConfig& speculative_config =
      ::xllm::SpeculativeConfig::get_instance();
  const std::string& speculative_algorithm =
      speculative_config.speculative_algorithm();
  LOG(INFO) << "Starting speculative validate profile for "
            << speculative_algorithm << ", "
            << "adaptive_enabled="
            << speculative_config.enable_adaptive_speculative_decode();

  auto& model_args = engine_->model_args();
  const int32_t max_context_len = model_args.max_position_embeddings();
  const int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  if (profile_max_prompt_length <= 16) {
    LOG(WARNING)
        << "Skip speculative validate profile because prompt length is too "
        << "small: " << profile_max_prompt_length;
    return;
  }

  const int32_t max_query_len =
      std::min<int32_t>(speculative_config.num_speculative_tokens() + 1, 10);
  const int32_t max_batch_size =
      std::min<int32_t>(options_.max_seqs_per_batch(), 256);
  std::vector<int32_t> query_lens;
  query_lens.push_back(1);
  if (max_query_len > 2) {
    query_lens.push_back((max_query_len + 1) / 2);
  }
  if (max_query_len > 1) {
    query_lens.push_back(max_query_len);
  }
  query_lens.erase(std::unique(query_lens.begin(), query_lens.end()),
                   query_lens.end());

  constexpr int32_t kMaxProfileBatchSize = 32;
  std::vector<int32_t> candidate_batch_sizes = {1, 16, 32};
  std::vector<int32_t> batch_sizes;
  for (const int32_t batch_size : candidate_batch_sizes) {
    if (batch_size <= max_batch_size && batch_size <= kMaxProfileBatchSize) {
      batch_sizes.push_back(batch_size);
    }
  }
  if (batch_sizes.empty()) {
    batch_sizes.push_back(std::min<int32_t>(
        std::max<int32_t>(max_batch_size, 1), kMaxProfileBatchSize));
  }

  std::vector<int32_t> prefix_lens;
  prefix_lens.push_back(profile_max_prompt_length);
  if (profile_max_prompt_length >= 64) {
    prefix_lens.push_back(profile_max_prompt_length / 4);
  }
  prefix_lens.push_back(16);
  for (int32_t& prefix_len : prefix_lens) {
    prefix_len = std::clamp(prefix_len, 16, profile_max_prompt_length);
  }
  std::sort(prefix_lens.begin(), prefix_lens.end());
  prefix_lens.erase(std::unique(prefix_lens.begin(), prefix_lens.end()),
                    prefix_lens.end());

  std::vector<std::tuple<int32_t, int32_t, int32_t, double>>
      time_profiling_data;
  const int32_t total_blocks =
      static_cast<int32_t>(block_manager_pool_->num_blocks());
  const auto block_size = block_manager_pool_->options().block_size();
  for (const int32_t prefix_len : prefix_lens) {
    for (const int32_t query_len : query_lens) {
      for (const int32_t batch_size : batch_sizes) {
        const int32_t token_length = prefix_len + query_len;
        const int32_t blocks_per_seq =
            (prefix_len + block_size - 1) / block_size +
            (token_length + block_size - 1) / block_size;
        if (batch_size * blocks_per_seq > total_blocks * 9 / 10) {
          continue;
        }
        double latency_mean = 0.0;
        for (int32_t k = 0; k < profile_count_per_step_; ++k) {
          latency_mean += run_request(token_length, prefix_len, batch_size);
        }
        latency_mean /= static_cast<double>(profile_count_per_step_);
        LOG(INFO) << "[spec_validate_profile] batch=" << batch_size
                  << " query_len=" << query_len << " prefix_len=" << prefix_len
                  << " latency_ms=" << latency_mean;
        time_profiling_data.emplace_back(
            batch_size, query_len, prefix_len, latency_mean);
      }
    }
  }
  train_speculative_validate_time_predictor(time_profiling_data);
}

// ----------------------predict step time-----------------------
std::vector<double> ProfileManager::get_coefficients(bool is_prefill) {
  if (is_prefill) {
    return prefill_time_predictor_->get_coefficients();
  } else {
    return decode_time_predictor_->get_coefficients();
  }
}

double ProfileManager::get_constant_overhead() {
  if (prefill_time_predictor_->is_trained() &&
      decode_time_predictor_->is_trained()) {
    return (prefill_time_predictor_->get_constant_overhead() +
            decode_time_predictor_->get_constant_overhead()) /
           2;
  } else if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_constant_overhead();
  } else if (decode_time_predictor_->is_trained()) {
    return decode_time_predictor_->get_constant_overhead();
  }
  return 0.0;
}

int32_t ProfileManager::get_quadratic_root(Sequence* sequence, double budget) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_quadratic_root(prefix_length, budget);
  }
  LOG(ERROR) << "Prefill time predictor is not trained yet.";
  return 0;
}

// for single sequence
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  CHECK(length > prefix_length)
      << "Token length (" << length << ") must be greater than prefix length "
      << " (" << prefix_length << ").";
  double ratio = 1.0;
  if (force_use_prefill_predictor) {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
  if (length - 1 == prefix_length) {
    return ratio * decode_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  } else {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
}

double ProfileManager::predict_step_time(Sequence* sequence,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_cache_tokens_num();
  double latency = predict_step_time(length,
                                     prefix_length,
                                     if_need_add_constant_term,
                                     force_use_prefill_predictor);
  return latency;
}
// for single batch or sequences
double ProfileManager::predict_step_time(
    const std::vector<int32_t>& length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(length_vec.size() == prefix_length_vec.size());
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < length_vec.size(); i++) {
    // predict for each sequence
    int32_t length = length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}

// for seq in batch with the same token and prefix length
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         int32_t batch_size) {
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < batch_size; i++) {
    // predict for each sequence
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}
// ---------------------------------------------

// ----------------------for profile token budget-----------------------
void ProfileManager::profile_token_budget() {
  // use token budget means defaultly ignoring prefix cache and decode request's
  // kv cache load overhead
  // warm up
  run_request(options_.profile_max_prompt_length(), 0);
  profile_token_budget_ =
      binary_search_max_tokens(options_.max_global_tpot_ms(), 1, 4096);
  LOG(INFO) << "Profile token budget: " << profile_token_budget_
            << "for TPOT SLO: " << options_.max_global_tpot_ms();
}

bool ProfileManager::check_if_satisfy_slo(int32_t num_tokens,
                                          int32_t tpot_slo_ms) {
  // int32_t prompt_tokens_per_batch = 1024;
  // auto batch_size = num_tokens / prompt_tokens_per_batch;
  // int32_t extra_token_length = num_tokens % prompt_tokens_per_batch;
  // double batch_latency = 0;
  // for (int32_t k = 0; k < profile_count_per_step_; k++) {
  //   batch_latency +=
  //       run_request(prompt_tokens_per_batch, 0, batch_size,
  //       extra_token_length);
  // }
  double batch_latency = 0;
  for (int32_t k = 0; k < profile_count_per_step_; k++) {
    batch_latency += run_request(num_tokens, 0, 1, 0);
  }
  batch_latency /= profile_count_per_step_;
  if (batch_latency <= tpot_slo_ms) {
    return true;
  } else {
    return false;
  }
}

int32_t ProfileManager::binary_search_max_tokens(int32_t tpot_slo_ms,
                                                 int32_t lower_bound,
                                                 int32_t upper_bound) {
  int32_t left = lower_bound;
  int32_t right = upper_bound;
  // [left, right)
  while (left < right) {
    int32_t mid = left + (right - left) / 2;
    if (check_if_satisfy_slo(mid, tpot_slo_ms)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

int32_t ProfileManager::get_token_budget() { return profile_token_budget_; }

// ---------------------------------------------

const std::vector<ProfileManager::CopyBlockProfile>&
ProfileManager::get_copy_block_profile() {
  // NOTE: Add more model profiles here
  static const std::vector<CopyBlockProfile> profiles = {
      // offline copy block profile
      {"Qwen2-7B", 128, 0.48, 0.24, "Qwen2-7B, block_size=128"},
      {"Qwen2-7B", 64, 0.20, 0.25, "Qwen2-7B, block_size=128"},
  };

  return profiles;
}

const ProfileManager::CopyBlockProfile* ProfileManager::find_profile(
    const std::string& model_name,
    int32_t block_size) const {
  const auto& profiles = get_copy_block_profile();
  for (const auto& profile : profiles) {
    if ((profile.model_name == model_name ||
         model_name.find(profile.model_name) != std::string::npos) &&
        profile.block_size == block_size) {
      return &profile;
    }
  }
  LOG(ERROR) << "No profile found for " << model_name
             << " with block_size=" << block_size << ", using default values";
  return nullptr;
}

int32_t ProfileManager::get_max_copy_block_num(double latency_budget) {
  auto block_size = block_manager_pool_->options().block_size();
  const CopyBlockProfile* profile =
      find_profile(::xllm::ModelConfig::get_instance().model_id(), block_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }

  double max_blocks = std::max((latency_budget - b) / a, 0.0);
  return static_cast<int32_t>(max_blocks);
}

double ProfileManager::predict_copy_blocks_time(
    size_t num_copy_blocks,
    bool if_need_add_constant_term) {
  auto block_size = block_manager_pool_->options().block_size();
  const CopyBlockProfile* profile =
      find_profile(::xllm::ModelConfig::get_instance().model_id(), block_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }
  return if_need_add_constant_term ? a * num_copy_blocks + b
                                   : a * num_copy_blocks;
}

std::shared_ptr<Request> ProfileManager::generate_single_request(
    int32_t token_length,
    int32_t prefix_length,
    bool is_graph_warmup,
    bool tolerate_allocation_failure) {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it. For graph warmup, restrict to low token IDs [1, 1000] so
  // VLM mRoPE generators never see unaligned vision_start/image token IDs.
  const int32_t max_token_id =
      is_graph_warmup ? std::min<int32_t>(vocab_size - 2, 1000)
                      : (vocab_size - 2);
  std::uniform_int_distribution<int32_t> dis(
      1, std::max<int32_t>(1, max_token_id));

  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

  RequestState req_state(token_ids);
  req_state.enable_schedule_overlap = options_.enable_schedule_overlap();
  req_state.is_graph_warmup = is_graph_warmup;
  auto request = std::make_shared<Request>(
      /*request_id=*/next_warmup_request_id(),
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      std::move(req_state));

  // TODO: better disable prefix cache
  if (prefix_length > 0) {
    if (!block_manager_pool_->BlockManagerPool::allocate(
            request->sequences()[0].get(), prefix_length)) {
      if (tolerate_allocation_failure) {
        block_manager_pool_->deallocate_without_cache(
            request->sequences()[0].get());
        return nullptr;
      }
      LOG(FATAL) << "Profiling time failed! Not enough blocks, prefix length : "
                 << prefix_length;
    }
    request->sequences()[0]->kv_state().incr_kv_cache_tokens_num(prefix_length);
  }

  if (!block_manager_pool_->BlockManagerPool::allocate(
          request->sequences()[0].get(), token_length)) {
    if (tolerate_allocation_failure) {
      block_manager_pool_->deallocate_without_cache(
          request->sequences()[0].get());
      return nullptr;
    }
    LOG(FATAL) << "Profiling time failed! Not enough blocks, token length : "
               << token_length;
  }

  return request;
}

std::shared_ptr<Request> ProfileManager::generate_single_decode_request(
    int32_t total_length,
    std::optional<int32_t> dp_rank,
    bool is_graph_warmup) {
  std::shared_ptr<Request> request = try_generate_single_decode_request(
      total_length, dp_rank, is_graph_warmup);
  if (request == nullptr) {
    LOG(FATAL) << "Profiling decode step time failed! Not enough blocks, total "
                  "length: "
               << total_length;
  }
  return request;
}

std::shared_ptr<Request> ProfileManager::try_generate_single_decode_request(
    int32_t total_length,
    std::optional<int32_t> dp_rank,
    bool is_graph_warmup) {
  CHECK_GT(total_length, 1) << "Decode profiling requires total_length > 1.";

  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it. For graph warmup, restrict to low token IDs [1, 1000] so
  // VLM mRoPE generators never see unaligned vision_start/image token IDs.
  const int32_t max_token_id =
      is_graph_warmup ? std::min<int32_t>(vocab_size - 2, 1000)
                      : (vocab_size - 2);
  std::uniform_int_distribution<int32_t> dis(
      1, std::max<int32_t>(1, max_token_id));

  const int32_t prompt_length = total_length - 1;
  std::vector<int32_t> prompt_token_ids(prompt_length);
  std::generate(prompt_token_ids.begin(), prompt_token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

  RequestState req_state(prompt_token_ids);
  req_state.enable_schedule_overlap = options_.enable_schedule_overlap();
  req_state.is_graph_warmup = is_graph_warmup;
  const int32_t num_speculative_tokens =
      decode_graph_warmup_plan_.warmup_config.num_speculative_tokens;
  const int64_t num_decoding_tokens =
      decode_graph_warmup_plan_.warmup_config.num_decoding_tokens;
  CHECK_GT(num_decoding_tokens, 0);
  size_t seq_capacity = static_cast<size_t>(total_length) +
                        static_cast<size_t>(num_decoding_tokens);
  if (options_.enable_schedule_overlap()) {
    seq_capacity += static_cast<size_t>(num_decoding_tokens);
  }
  req_state.seq_capacity = seq_capacity;
  auto request = std::make_shared<Request>(
      /*request_id=*/next_warmup_request_id(),
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      std::move(req_state));

  auto* sequence = request->sequences()[0].get();
  if (dp_rank.has_value()) {
    CHECK_GE(dp_rank.value(), 0);
    CHECK_LT(dp_rank.value(), options_.dp_size());
    sequence->set_dp_rank(dp_rank.value());
  }
  if (!block_manager_pool_->BlockManagerPool::allocate(sequence,
                                                       seq_capacity)) {
    return nullptr;
  }
  sequence->kv_state().incr_kv_cache_tokens_num(prompt_length);

  int32_t generated_token = dis(gen);
  generated_token =
      generated_token == eos_token_id ? generated_token + 1 : generated_token;
  sequence->append_token(generated_token);

  // With MTP speculative decoding the worker's decode path requires a valid
  // decode state written via the MTP bootstrap channel before validating the
  // per-token decode state. Inject a placeholder bootstrap embedding so the
  // synthetic warmup/profile request takes the same bootstrap path as a real
  // disagg PD decode request instead of reading stale recycled decode state.
  const int64_t bootstrap_width = mtp_hidden_state_width(model_args);
  prepare_warmup_decode_sequence(
      sequence, bootstrap_width, num_speculative_tokens);

  CHECK(sequence->stage() == SequenceStage::DECODE)
      << "Decode profiling request is not in DECODE stage. total_length: "
      << total_length << ", prompt_length: " << prompt_length
      << ", kv_cache_tokens_num: " << sequence->kv_state().kv_cache_tokens_num()
      << ", num_tokens: " << sequence->num_tokens();
  CHECK_EQ(sequence->num_generated_tokens(), 1)
      << "Decode profiling request should start with one generated token.";

  return request;
}

int32_t ProfileManager::measure_graph_decode_capacity(
    int32_t configured_max_seqs,
    int32_t total_length) {
  CHECK_GT(configured_max_seqs, 0);
  CHECK_GT(total_length, 1);
  CHECK_GT(options_.dp_size(), 0);

  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(static_cast<size_t>(configured_max_seqs));
  for (int32_t index = 0; index < configured_max_seqs; ++index) {
    const int32_t dp_rank = index % options_.dp_size();
    std::shared_ptr<Request> request =
        try_generate_single_decode_request(total_length, dp_rank);
    if (request == nullptr) {
      break;
    }
    requests.emplace_back(std::move(request));
  }

  const int32_t allocatable_sequences = static_cast<int32_t>(requests.size());
  for (const std::shared_ptr<Request>& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }
  return allocatable_sequences;
}

// collect the latency of each step
double ProfileManager::run_request(int32_t token_length,
                                   int32_t prefix_length,
                                   int32_t batch_size,
                                   int32_t extra_token_length,
                                   bool is_graph_warmup) {
  CHECK(token_length >= prefix_length);
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(batch_size);
  sequences_budget.reserve(batch_size);
  requests.reserve(batch_size);

  // batch sequences with the same kv cache and token length
  for (int32_t i = 0; i < batch_size; i++) {
    // generate random token ids and request
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length, is_graph_warmup);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // maybe another sequence for extra token length (< token_length) for token
  // budget profiling
  if (extra_token_length > 0) {
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length, is_graph_warmup);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(requests, sequences, sequences_budget);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// currently for test only
double ProfileManager::run_request(
    const std::vector<int32_t>& token_length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(token_length_vec.size() == prefix_length_vec.size());
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(token_length_vec.size());
  sequences_budget.reserve(token_length_vec.size());
  requests.reserve(token_length_vec.size());

  // batch sequences with the same kv cache and token length
  for (int32_t i = 0; i < token_length_vec.size(); i++) {
    // generate random token ids and request
    int32_t token_length = token_length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];

    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

double ProfileManager::run_decode_request(
    const std::vector<int32_t>& total_length_vec) {
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  for (int32_t total_length : total_length_vec) {
    std::shared_ptr<Request> request =
        generate_single_decode_request(total_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(1);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

double ProfileManager::run_graph_decode_request(
    const std::vector<int32_t>& total_length_vec) {
  CHECK_GT(options_.dp_size(), 0);

  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(total_length_vec.size());
  sequences_budget.reserve(total_length_vec.size());
  requests.reserve(total_length_vec.size());

  for (size_t i = 0; i < total_length_vec.size(); ++i) {
    int32_t dp_rank = static_cast<int32_t>(i % options_.dp_size());
    std::shared_ptr<Request> request = generate_single_decode_request(
        total_length_vec[i], dp_rank, /*is_graph_warmup=*/true);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(1);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// Generate a batch of decode requests in DECODE stage and execute one decode
// step, then return the step latency.
double ProfileManager::profile_decode_step_time(int32_t token_length,
                                                int32_t batch_size,
                                                int32_t min_context_len,
                                                int32_t max_context_len) {
  double total_latency = 0.0;
  for (int32_t i = 0; i < profile_count_per_step_; ++i) {
    std::vector<int32_t> token_length_vec;
    generate_random_decode_batch(batch_size * token_length,
                                 batch_size,
                                 min_context_len,
                                 max_context_len,
                                 token_length_vec);
    total_latency += run_decode_request(token_length_vec);
  }
  return total_latency / profile_count_per_step_;
}

// Generate a batch of random decode requests with an average total sequence
// length of token_length.
void ProfileManager::generate_random_decode_batch(
    int32_t total_length,
    int32_t batch_size,
    int32_t min_context_len,
    int32_t max_context_len,
    std::vector<int32_t>& token_length_vec) {
  CHECK(total_length >= batch_size * min_context_len);
  CHECK(total_length <= batch_size * max_context_len);

  token_length_vec.resize(batch_size, min_context_len);
  int remain = total_length - batch_size * min_context_len;

  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (int i = 0; i < batch_size; ++i) {
    if (remain == 0) break;

    int max = remain > (max_context_len - min_context_len)
                  ? (max_context_len - min_context_len)
                  : remain;

    std::uniform_int_distribution<int> dis(0, max);
    int add = dis(gen);
    token_length_vec[i] += add;
    remain -= add;
  }

  int idx = 0;
  while (remain > 0) {
    if (token_length_vec[idx % batch_size] < max_context_len) {
      token_length_vec[idx % batch_size] += 1;
      --remain;
    }
    ++idx;
  }
}

void ProfileManager::warmup_for_eager() {
  constexpr int32_t kMaxEagerWarmupTokens = 256;
  const int32_t max_context_len =
      engine_->model_args().max_position_embeddings();
  const int32_t prefill_tokens = std::min({options_.max_tokens_per_batch(),
                                           max_context_len,
                                           kMaxEagerWarmupTokens});
  if (prefill_tokens <= 0 || engine_->model_args().vocab_size() <= 2) {
    LOG(INFO) << "Skipping eager warmup because model metadata is incomplete: "
              << "tokens=" << prefill_tokens
              << ", vocab_size=" << engine_->model_args().vocab_size();
    return;
  }
  const double prefill_latency = run_request(prefill_tokens,
                                             0,
                                             1,
                                             /*extra_token_length=*/0,
                                             /*is_graph_warmup=*/true);
  LOG(INFO) << "Eager warmup completed: tokens=" << prefill_tokens
            << ", latency=" << prefill_latency << " ms";
}

void ProfileManager::warmup_for_graph() {
  const GraphWarmupPlan plan = graph_warmup_plan(options_.instance_role());
  if (plan == GraphWarmupPlan::PREFILL_ONLY) {
    LOG(INFO) << "PREFILL graph warmup: prefill only";
    warmup_prefill_for_graph();
    return;
  }
  if (plan == GraphWarmupPlan::DECODE_ONLY) {
    LOG(INFO) << "DECODE graph warmup: decode buckets only";
    warmup_decode_for_graph();
    return;
  }

  warmup_unified_for_graph();
}

void ProfileManager::warmup_prefill_for_graph() {
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  int32_t prefill_tokens =
      std::min(options_.max_tokens_per_batch(), max_context_len);
  if (::xllm::SchedulerConfig::get_instance().enable_dp_fair_token_budget() &&
      options_.dp_size() > 1 &&
      options_.instance_role() == InstanceRole::PREFILL) {
    // The fair per-group budget caps any single sequence's prefill at
    // max_tokens_per_batch / dp_size tokens per scheduling round (floored
    // at one prefill chunk), so warming up at the per-group cap covers the
    // largest real prefill shape.
    const int32_t dp_size = options_.dp_size();
    int32_t per_group_cap =
        (options_.max_tokens_per_batch() + dp_size - 1) / dp_size;
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      const int32_t max_chunk_tokens = ::xllm::SchedulerConfig::get_instance()
                                           .max_tokens_per_chunk_for_prefill();
      per_group_cap =
          std::max(per_group_cap,
                   std::min(max_chunk_tokens, options_.max_tokens_per_batch()));
    } else {
      per_group_cap = options_.max_tokens_per_batch();
    }
    prefill_tokens = std::min(prefill_tokens, per_group_cap);
  }
  if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
    const int32_t max_chunk_tokens =
        ::xllm::SchedulerConfig::get_instance()
            .max_tokens_per_chunk_for_prefill();
    if (max_chunk_tokens > 0) {
      prefill_tokens = std::min(prefill_tokens, max_chunk_tokens);
    }
  }
  const int32_t max_graph_tokens =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_graph_tokens > 0) {
    prefill_tokens = std::min(prefill_tokens, max_graph_tokens);
  }
  double prefill_latency = run_request(prefill_tokens,
                                       /*prefix_length=*/0,
                                       /*batch_size=*/1,
                                       /*extra_token_length=*/0,
                                       /*is_graph_warmup=*/true);
  LOG(INFO) << "Prefill warmup completed: tokens=" << prefill_tokens
            << ", latency=" << prefill_latency << " ms";
#if defined(USE_MUSA)
  const auto& exec_cfg = ::xllm::ExecutionConfig::get_instance();
  if (exec_cfg.enable_prefill_piecewise_graph() &&
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
      options_.dp_size() == 1) {
    const int32_t stride =
        std::max<int32_t>(256, exec_cfg.musa_prefill_graph_token_stride());
    const int32_t prefix_step = std::max<int32_t>(
        128, static_cast<int32_t>(block_manager_pool_->options().block_size()) * 2);
    std::vector<int32_t> chunk_buckets;
    for (int32_t b = (prefill_tokens / stride) * stride; b >= stride;
         b -= stride) {
      chunk_buckets.push_back(b);
    }
    if (stride > 256 &&
        (chunk_buckets.empty() || chunk_buckets.back() > 256)) {
      chunk_buckets.push_back(256);
    }
    double chunk_warmup_ms = 0.0;
    // Under CP>1 + MTP, non-final full chunks (n_tokens < seq_len, numel ==
    // max_chunk_tokens) execute with skip_pcp_gather=true (and sharded draft
    // input), which is part of get_prefill_graph_key(). Pre-capture both the
    // initial (prefix=0, chunked=0) and continuation (prefix>0, chunked=1)
    // non-final chunk keys at startup so the first >8K cold request never pays
    // online lazy graph capture.
    auto run_nonfinal_chunk_warmup = [&](int32_t chunk_tokens,
                                         int32_t prefix_len) -> double {
      const int32_t total_len = prefix_len + chunk_tokens + prefix_step;
      std::shared_ptr<Request> request = generate_single_request(
          total_len, prefix_len, /*is_graph_warmup=*/true);
      std::vector<std::shared_ptr<Request>> requests{request};
      std::vector<Sequence*> sequences{request->sequences()[0].get()};
      std::vector<size_t> sequences_budget{
          static_cast<size_t>(chunk_tokens)};
      auto batches = BatchFactory::get_instance(options_.dp_size())
                         ->create_batches(requests, sequences, sequences_budget);
      absl::Time t0 = absl::Now();
      engine_->step(batches);
      if (options_.enable_schedule_overlap()) {
        engine_->update_last_step_result(batches);
      }
      const double ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
      block_manager_pool_->deallocate_without_cache(
          request->sequences()[0].get());
      return ms;
    };
    const int32_t max_chunk_cfg =
        ::xllm::SchedulerConfig::get_instance()
            .max_tokens_per_chunk_for_prefill();
    if (::xllm::ParallelConfig::get_instance().cp_size() > 1 && max_chunk_cfg > 0 &&
        prefill_tokens == max_chunk_cfg) {
      chunk_warmup_ms +=
          run_nonfinal_chunk_warmup(prefill_tokens, /*prefix_len=*/0);
      chunk_warmup_ms +=
          run_nonfinal_chunk_warmup(prefill_tokens, /*prefix_len=*/prefix_step);
    }
    for (int32_t short_cold : {256, 512}) {
      if (short_cold < prefill_tokens && short_cold % stride == 0) {
        chunk_warmup_ms += run_request(short_cold,
                                       /*prefix_length=*/0,
                                       /*batch_size=*/1,
                                       /*extra_token_length=*/0,
                                       /*is_graph_warmup=*/true);
      }
    }
    for (int32_t b : chunk_buckets) {
      chunk_warmup_ms += run_request(prefix_step + b,
                                     /*prefix_length=*/prefix_step,
                                     /*batch_size=*/1,
                                     /*extra_token_length=*/0,
                                     /*is_graph_warmup=*/true);
    }
    LOG(INFO) << "MUSA chunked-prefill piecewise graph warmup completed: "
              << "buckets=" << chunk_buckets.size()
              << ", stride=" << stride
              << ", total_latency=" << chunk_warmup_ms << " ms";

    if (options_.instance_role() == InstanceRole::PREFILL &&
        ::xllm::ParallelConfig::get_instance().cp_size() == 2 &&
        util::get_bool_env("XLLM_MUSA_GDN_CHECKPOINT_BUCKET_RUNNER", false) &&
        block_manager_pool_->options().enable_linear_state() &&
        block_manager_pool_->options().linear_state_ckpt_slots() > 0) {
      auto run_single_ckpt_warmup = [&](int32_t b, int32_t prefix_len) -> double {
        std::shared_ptr<Request> request = generate_single_request(
            prefix_len + b,
            prefix_len,
            /*is_graph_warmup=*/true,
            /*tolerate_allocation_failure=*/true);
        if (!request) {
          return 0.0;
        }
        Sequence* seq = request->sequences()[0].get();
        const auto linear_blocks = seq->kv_state().blocks(BlockType::LINEAR);
        std::vector<std::vector<BlockTransferInfo>> swap_infos(
            options_.dp_size());
        if (!linear_blocks.empty()) {
          const int32_t live_slot_id = linear_blocks.back().id();
          const int32_t ckpt_block_id =
              block_manager_pool_->options().linear_state_num_slots();
          BlockTransferInfo transfer(-1, -1);
          if (::xllm::musa::encode_deepseek_v4_gdn_singlepass_checkpoint(
                  live_slot_id,
                  ckpt_block_id,
                  /*offset_tokens=*/64,
                  /*is_end_of_step=*/false,
                  &transfer)) {
            swap_infos[seq->dp_rank()].emplace_back(std::move(transfer));
          }
        }
        std::vector<std::shared_ptr<Request>> requests{request};
        std::vector<Sequence*> sequences{seq};
        std::vector<size_t> budgets{static_cast<size_t>(b)};
        auto batches =
            BatchFactory::get_instance(options_.dp_size())
                ->create_batches(requests, sequences, budgets, &swap_infos);
        absl::Time t0 = absl::Now();
        engine_->step(batches);
        if (options_.enable_schedule_overlap()) {
          engine_->update_last_step_result(batches);
        }
        const double ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
        block_manager_pool_->deallocate_without_cache(seq);
        return ms;
      };
      double ckpt_runner_warmup_ms = 0.0;
      int32_t ckpt_runner_buckets = 0;
      for (int32_t b : {1024, 2048, 3072, 4096, 6144}) {
        if (b < prefill_tokens) {
          const double latency_ms =
              run_single_ckpt_warmup(b, /*prefix_len=*/prefix_step);
          if (latency_ms > 0.0) {
            ckpt_runner_warmup_ms += latency_ms;
            ++ckpt_runner_buckets;
          }
        }
      }
      LOG(INFO) << "MUSA B=1 GDN checkpoint-runner bucket warmup completed: "
                << "profiles=" << ckpt_runner_buckets
                << ", total_latency=" << ckpt_runner_warmup_ms << " ms";
    }

    const std::string profiles =
        util::get_string_env("XLLM_MUSA_PCP_PREFILL_WARMUP_BATCHES");
    if (profiles.empty() ||
        options_.instance_role() != InstanceRole::PREFILL ||
        ::xllm::ParallelConfig::get_instance().cp_size() != 2 ||
        !util::get_bool_env("XLLM_MUSA_PCP_MULTIBATCH_GRAPH", false) ||
        !util::get_bool_env("XLLM_MUSA_PCP_LOCAL_TOKEN_BUCKET", false)) {
      return;
    }

    const int32_t batch_budget = max_graph_tokens > 0
                                     ? std::min(options_.max_tokens_per_batch(),
                                                max_graph_tokens)
                                     : options_.max_tokens_per_batch();
    for (absl::string_view profile : absl::StrSplit(profiles, ';')) {
      if (profile.empty()) {
        continue;
      }
      std::vector<std::pair<int32_t, int32_t>> chunks;
      int64_t query_budget = 0;
      for (absl::string_view chunk : absl::StrSplit(profile, ',')) {
        const std::vector<absl::string_view> lengths = absl::StrSplit(chunk, '/');
        int32_t history = 0;
        int32_t query = 0;
        if (lengths.size() != 2 || !absl::SimpleAtoi(lengths[0], &history) ||
            !absl::SimpleAtoi(lengths[1], &query) || history < 0 || query <= 0 ||
            query > prefill_tokens || history > max_context_len - query) {
          chunks.clear();
          break;
        }
        chunks.emplace_back(history, query);
        query_budget += query;
      }
      if (chunks.size() < 2 ||
          chunks.size() > static_cast<size_t>(options_.max_seqs_per_batch()) ||
          query_budget > batch_budget) {
        LOG(WARNING) << "Skipping invalid CP2 Prefill startup profile: "
                     << profile;
        continue;
      }

      std::vector<std::shared_ptr<Request>> requests;
      std::vector<Sequence*> sequences;
      std::vector<size_t> budgets;
      for (const auto& [history, query] : chunks) {
        auto request = generate_single_request(
            history + query, history, /*is_graph_warmup=*/true,
            /*tolerate_allocation_failure=*/true);
        if (!request) {
          break;
        }
        sequences.emplace_back(request->sequences()[0].get());
        budgets.emplace_back(query);
        requests.emplace_back(std::move(request));
      }

      if (requests.size() == chunks.size()) {
        auto batches = BatchFactory::get_instance(options_.dp_size())
                           ->create_batches(requests, sequences, budgets);
        absl::Time started = absl::Now();
        engine_->step(batches);
        if (options_.enable_schedule_overlap()) {
          engine_->update_last_step_result(batches);
        }
        LOG(INFO) << "CP2 Prefill startup profile executed: " << profile
                  << ", ms=" << absl::ToDoubleMilliseconds(absl::Now() - started);
      } else {
        LOG(WARNING) << "Skipping CP2 Prefill startup profile: " << profile
                     << "; unable to allocate synthetic sequence blocks";
      }
      for (Sequence* sequence : sequences) {
        block_manager_pool_->deallocate_without_cache(sequence);
      }
    }
  }
#endif
}

void ProfileManager::warmup_unified_for_graph() {
  warmup_prefill_for_graph();
  warmup_decode_for_graph();
}

void ProfileManager::warmup_decode_for_graph() {
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  int32_t max_decode_batch_size = options_.max_seqs_per_batch();
  const int32_t max_concurrent_requests =
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests();
  if (max_concurrent_requests > 0) {
    max_decode_batch_size =
        std::min(max_decode_batch_size, max_concurrent_requests);
  }
  if (Platform::is_npu()) {
    max_decode_batch_size =
        static_cast<int32_t>(npu::acl_graph_max_global_batch_size(
            static_cast<uint32_t>(std::max<int32_t>(1, max_decode_batch_size)),
            static_cast<uint32_t>(
                std::max<int32_t>(1,
                                  ::xllm::ExecutionConfig::get_instance()
                                      .acl_graph_decode_batch_size_limit())),
            static_cast<uint32_t>(std::max<int32_t>(1, options_.dp_size()))));
  }
  const int32_t decode_seq_len = warmup_decode_seq_len(
      Platform::is_npu() &&
          ::xllm::ExecutionConfig::get_instance().enable_fia_decode(),
      max_context_len);

#if defined(USE_MUSA)
  const int32_t dp_size = std::max<int32_t>(1, options_.dp_size());
  const int32_t musa_probe_global_seqs = max_decode_batch_size * dp_size;
  const int32_t allocatable_sequences =
      measure_graph_decode_capacity(musa_probe_global_seqs, decode_seq_len);
  const int32_t max_local_seqs =
      std::min(max_decode_batch_size, allocatable_sequences / dp_size);
  decode_graph_warmup_plan_ =
      build_decode_graph_warmup_plan(engine_->decode_graph_warmup_config(),
                                     max_local_seqs * dp_size,
                                     dp_size);
  // On MUSA, spec_verify_graphs_ is keyed by exact (n_tokens, q_len) where
  // n_tokens = local_seqs * num_decoding_tokens, and no-padding decode graphs
  // use fine-grained local token buckets (1*local_seqs and 2*local_seqs on
  // MTP bootstrap). Sweep every DP-local sequence count 1..max_local_seqs so
  // all DP ranks pre-capture draft, bootstrap, and spec-verify graphs at
  // startup before serving live traffic.
  decode_graph_warmup_plan_.batch_sizes.clear();
  decode_graph_warmup_plan_.batch_sizes.reserve(
      static_cast<size_t>(std::max<int32_t>(0, max_local_seqs)));
  for (int32_t local_seqs = 1; local_seqs <= max_local_seqs; ++local_seqs) {
    decode_graph_warmup_plan_.batch_sizes.push_back(local_seqs * dp_size);
  }
#else
  const int32_t allocatable_sequences =
      measure_graph_decode_capacity(max_decode_batch_size, decode_seq_len);
  const int32_t warmup_capacity =
      std::min(max_decode_batch_size, allocatable_sequences);
  decode_graph_warmup_plan_ =
      build_decode_graph_warmup_plan(engine_->decode_graph_warmup_config(),
                                     warmup_capacity,
                                     options_.dp_size());
#endif
  const std::vector<int32_t>& decode_batch_sizes =
      decode_graph_warmup_plan_.batch_sizes;
  const int32_t decode_bucket_count =
      static_cast<int32_t>(decode_batch_sizes.size());
  LOG(INFO) << "Graph warmup started: bucket_count=" << decode_bucket_count
            << ", configured_max_batch_size=" << max_decode_batch_size
            << ", allocatable_sequences=" << allocatable_sequences
            << ", decode_seq_len=" << decode_seq_len;

  // Capture from the largest bucket down to the smallest so every smaller
  // bucket reuses the scratch blocks freed by the largest capture within the
  // shared graph mempool. Ascending capture forces each larger bucket to grab
  // fresh scratch (the freed smaller blocks cannot satisfy it), which makes the
  // pool grow linearly with the bucket count.
  double decode_total_latency = 0.0;
  for (int32_t bucket_index = decode_bucket_count - 1; bucket_index >= 0;
       --bucket_index) {
    const int32_t sequence_batch_size =
        decode_batch_sizes[static_cast<size_t>(bucket_index)];
    const int32_t token_bucket = decode_warmup_token_bucket(
        decode_graph_warmup_plan_, sequence_batch_size, options_.dp_size());
    std::vector<int32_t> total_length_vec(sequence_batch_size, decode_seq_len);
    const double decode_latency = run_graph_decode_request(total_length_vec);
    decode_total_latency += decode_latency;
    LOG(INFO) << graph_warmup_progress(
                     /*completed=*/decode_bucket_count - bucket_index,
                     /*total=*/decode_bucket_count,
                     /*token_bucket=*/token_bucket,
                     /*latency_ms=*/decode_latency)
              << ", sequence_batch=" << sequence_batch_size;
  }

  const int32_t max_sequence_batch_size =
      decode_batch_sizes.empty() ? 0 : decode_batch_sizes.back();
  const int32_t max_token_bucket =
      max_sequence_batch_size == 0
          ? 0
          : decode_warmup_token_bucket(decode_graph_warmup_plan_,
                                       max_sequence_batch_size,
                                       options_.dp_size());
  LOG(INFO) << "Decode warmup completed: bucket_count=" << decode_bucket_count
            << ", decode_max_token_bucket=" << max_token_bucket
            << ", decode_max_sequence_batch=" << max_sequence_batch_size
            << ", decode_seq_len=" << decode_seq_len
            << ", decode_total_latency=" << decode_total_latency << " ms";
}

}  // namespace xllm
