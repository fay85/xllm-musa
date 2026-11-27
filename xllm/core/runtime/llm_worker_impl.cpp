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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/sampling/sampler.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "runtime/params_utils.h"
#include "util/env_var.h"
#include "util/scope_guard.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

void wait_input_ready_events(const ForwardInput& input, const Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait ForwardInput metadata ready event";
}

StreamEventPtr record_current_stream_event(const Device& device) {
  std::unique_ptr<Stream> stream = device.current_stream();
  return stream->record_event_or_sync();
}

}  // namespace

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA)
  const auto& model_config = ModelConfig::get_instance();
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    threadpool_.schedule([this]() mutable {
      // initialize flashinfer workspace
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
          device_);
    });
  }
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  const auto& model_config = ModelConfig::get_instance();
#if defined(USE_MUSA)
  static const bool use_pool_compute_stream = util::get_bool_env(
      "XLLM_MUSA_POOL_COMPUTE_STREAM", /*default_value=*/true);
  const bool is_qwen3 = context.get_model_args().model_type() == "qwen3";
  if (use_pool_compute_stream && !is_qwen3) {
    compute_stream_ = device_.get_stream_from_pool();
  } else if (use_pool_compute_stream) {
    LOG(WARNING) << "MUSA pool compute streams are not validated for Qwen3 "
                    "attention; using the default compute stream.";
  }

  const auto& beam_search_config = BeamSearchConfig::get_instance();
  CHECK(!has_linear_attention_layers(context.get_model_args()) ||
        (!beam_search_config.enable_beam_search_kernel() &&
         beam_search_config.beam_width() <= 1))
      << "MUSA beam search is not supported for models with linear-attention "
         "layers.";
#endif

#if defined(USE_CUDA)
  // Ensure FlashinferWorkspace is initialized on the calling thread before
  // constructing model layers. When called synchronously from
  // SpeculativeWorkerImpl (e.g. MTP target/draft setup), init_model runs on
  // the MTP worker's thread (T_MTP) rather than on the LLMWorkerImpl's own
  // threadpool thread (T_worker) where the scheduled initialize() runs.
  // FlashinferWorkspace is thread_local, so T_MTP's instance must be
  // explicitly initialized here; otherwise FlashInferAttentionImpl captures
  // an undefined int_workspace_buffer_ and crashes at prefill time.
  //
  // Skip when model_impl=python: Python executor uses flashinfer's Python API
  // directly; initializing the C++ workspace would conflict with Python-side
  // TVM-FFI type registration.
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    auto& ws = ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance();
    if (!ws.get_int_workspace_buffer().defined()) {
      ws.initialize(device_);
    }
  }
#endif

  // Try to create a causal LM model
  context.set_model_impl(model_config.model_impl());
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  model_executor_->set_dp_process_group(
      context.get_parallel_args().dp_local_process_group_);

  // MTP draft workers execute a separate one-layer model and must not consume
  // the target model's EPLB prepare/update commands. Only the target reports
  // expert load and participates in expert-weight transfer.
  if (::xllm::EPLBConfig::get_instance().enable_eplb() && !is_spec_draft_) {
    eplb_executor_ = std::make_unique<EplbExecutor>(*model_, device_);
  }

  if (::xllm::BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

#if defined(USE_NPU)
bool LLMWorkerImpl::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  if (model_executor_ == nullptr) {
    return false;
  }
  return model_executor_->prepare_static_mtp_graph_tasks(signal, signal_stream);
}
#endif

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream) {
  return execute_no_sync_on_stream(
      input, compute_stream, /*record_ready_event=*/true);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    bool record_ready_event) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_acl_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_acl_stream);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    } else {
      SET_ATB_EXECUTE_STREAM((&compute_stream), device_, context_);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    }
#else
    wait_input_ready_events(input, compute_stream);
    return step_internal(input, sync_policy, record_ready_event);
#endif
  }
  wait_input_ready_events(input, compute_stream);
  return step_internal(input, sync_policy, record_ready_event);
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
#if defined(USE_NPU)
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
      wait_input_ready_events(input, *compute_stream_);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    }
  }
#endif

  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_for_schedule_overlap(
    ForwardInput& input) {
  if (has_request_state_cache()) {
    c10::StreamGuard state_guard = compute_stream_->set_stream_guard();
    prepare_linear_state_cache(input.input_params);
  }
  return execute_no_sync_on_stream(input, *compute_stream_);
}

ForwardInput
LLMWorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(last_step_output_.ready_event))
      << "failed to wait last step output ready event";
  // Order replace_token's in-place token_ids rewrite after the async H2D that
  // fills token_ids on prepare_stream_.
  wait_input_ready_events(input, *compute_stream_);
  return WorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
      input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input,
    ForwardSyncPolicy sync_policy,
    bool record_ready_event) {
  MULTI_MODEL_STEP_LOCK(::xllm::KVCacheConfig::get_instance().enable_xtensor());

  Timer timer;
  auto& sampling_params = input.sampling_params;

  KVTransferCompletion kv_transfers;
  const std::vector<std::string> canonical_request_ids =
      canonical_transfer_request_ids(input.transfer_kv_infos);
  std::vector<std::string> failed_request_ids;
  std::vector<std::string> unsafe_request_ids;
  bool kv_push_waited = false;
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
  std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer;
#endif

  std::unique_ptr<Stream> push_stream;
  bool push_aborted = false;
  // As in VLM, keep the producer stream alive and drain it before abort wakes
  // a sender waiting for layer events. Install cleanup before starting PUSH.
  auto abort_push = [&]() {
    if (push_aborted) {
      return;
    }
    push_aborted = true;
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
    if (layer_synchronizer != nullptr) {
      CHECK_EQ(push_stream->synchronize(), 0);
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_MUSA)
      // DCU's existing synchronizer has no abort hook; retain its wait policy.
      layer_synchronizer->abort();
#endif
    }
#endif
  };
  auto fail_aborted_push = [&]() {
    abort_push();
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !canonical_request_ids.empty()) {
      KVTransferTaskResult result;
      result.request_ids = canonical_request_ids;
      result.error_code = KVTransferErrorCode::FAILED;
      kv_transfers.add(folly::makeSemiFuture(std::vector<KVTransferTaskResult>{
                           std::move(result)}),
                       canonical_request_ids);
    }
  };
  ScopeGuard push_guard([&]() {
    if (!kv_push_waited) {
      abort_push();
      // Drain real writes without entering collectives during unwinding.
      kv_transfers.wait();
    }
  });

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    push_stream = device_.current_stream();
#if defined(USE_NPU)
    layer_synchronizer = std::make_shared<NPULayerSynchronizerImpl>(
        context_.get_model_args().n_layers());
#elif defined(USE_MLU)
    layer_synchronizer = std::make_shared<MLULayerSynchronizerImpl>(
        context_.get_model_args().n_layers());
#elif defined(USE_DCU)
    layer_synchronizer = std::make_shared<DCULayerSynchronizerImpl>(
        context_.get_model_args().n_layers());
#elif defined(USE_MUSA)
    layer_synchronizer = std::make_shared<MUSALayerSynchronizer>(
        context_.get_model_args().n_layers());
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;

#endif
  }
  // call model executor forward to get hidden states
  const ModelInputParams* forward_params = &input.input_params;
  std::optional<ModelInputParams> prefill_params;
  std::unique_ptr<Stream> prefill_stream;
  if (prefill_layerwise_kv_cache_ != nullptr) {
    // One batch-sized copy per prefill step lets the two fields below be
    // overridden without mutating the caller's params for the duration of the
    // forward; begin() itself takes the params by const reference.
    prefill_params.emplace(input.input_params);
    ModelInputParams& model_params = *prefill_params;
    prefill_stream = device_.current_stream();
    // Keep the Stream wrapper alive across model forward; the manager only
    // borrows it while the current batch runs.
    if (!prefill_layerwise_kv_cache_->begin(
            model_params,
            options_.block_size(),
            prefill_stream.get(),
            /*expect_push=*/!input.transfer_kv_infos.empty(),
            context_.get_parallel_args().tp_cp_group_)) {
      LOG(ERROR) << "Failed to prepare prefill layerwise KV, batch_id="
                 << model_params.meta.batch_id;
      return make_failed_output(model_params.embedding.request_ids, {});
    }
    model_params.parallel.layers_per_event = 1;
    model_params.parallel.layer_wise_load_synchronizer =
        prefill_layerwise_kv_cache_;
    forward_params = &model_params;
  }
  auto abort_prefill = folly::makeGuard([&]() {
    if (prefill_layerwise_kv_cache_ == nullptr) {
      return;
    }
#if defined(USE_NPU) || defined(USE_MLU)
    if (layer_synchronizer != nullptr) {
      layer_synchronizer->abort();
    }
#endif
    // Preserve the original exception while cancelling readers waiting for
    // missing compute events and draining the owner before stack unwinding.
    prefill_layerwise_kv_cache_->abort();
  });
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
  // Begin the cache batch before any PUSH completion can arrive. In particular,
  // invalid prefill metadata must fail without leaving a sender waiting for
  // computation events from a forward that will never execute.
  if (layer_synchronizer != nullptr) {
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_MUSA)
    // Preserve the cancellation hook that releases senders waiting for a
    // layer event when the completion object drains after an exception.
    kv_transfers.add(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_),
        canonical_request_ids,
        [layer_synchronizer] { layer_synchronizer->abort(); });
#else
    kv_transfers.add(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_),
        canonical_request_ids);
#endif
  }
#endif
  auto wait_kv_push = [&]() {
    if (kv_push_waited) {
      return;
    }
    KvPushFailureIds failures =
        finalize_kv_push_failures(kv_transfers,
                                  input.transfer_kv_infos,
                                  options_.kv_cache_transfer_mode(),
                                  options_.instance_role(),
                                  context_.get_parallel_args(),
                                  device_);
    failed_request_ids = std::move(failures.failed_request_ids);
    unsafe_request_ids = std::move(failures.unsafe_request_ids);
    kv_push_waited = true;
  };
  if (eplb_executor_ != nullptr) {
    eplb_executor_->start_eplb_step(input.input_params.expert.eplb_info);
  }
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, *forward_params);
  if (prefill_layerwise_kv_cache_ != nullptr) {
#if defined(USE_NPU) || defined(USE_MLU)
    if (!model_output.hidden_states.defined() &&
        !model_output.draft_token_ids.defined() &&
        layer_synchronizer != nullptr) {
      layer_synchronizer->abort();
    }
#endif
    CHECK(prefill_layerwise_kv_cache_->finish())
        << "Prefill layerwise KV execution/save/PUSH failed; shared cache "
           "buffers cannot be reused, batch_id="
        << input.input_params.meta.batch_id;
  }
  abort_prefill.dismiss();
  if (eplb_executor_ != nullptr) {
    eplb_executor_->finish_eplb_step();
  }
  if (!model_output.hidden_states.defined() &&
      !model_output.draft_token_ids.defined()) {
    fail_aborted_push();
    wait_kv_push();
    if (!failed_request_ids.empty()) {
      return make_failed_output(std::move(failed_request_ids),
                                std::move(unsafe_request_ids));
    }
    return std::nullopt;
  }

  ForwardOutput output;
  output.mtp_topk_state = std::move(model_output.mtp_topk_state);
  if (eplb_executor_ != nullptr) {
    output.expert_load_data = expert_load_data_;
    output.prepared_token = eplb_executor_->consume_ready_prepare_token();
  }

  if (model_output.draft_token_ids.defined()) {
    output.draft_token_ids = std::move(model_output.draft_token_ids);
    output.draft_logits = std::move(model_output.draft_logits);
    output.draft_confidence_probs =
        std::move(model_output.draft_confidence_probs);
  } else {
    torch::Tensor logits;
    torch::Tensor lm_head_selected_token_idxes;
    torch::Tensor selected_hidden;
    if (sampling_params.selected_token_idxes.defined()) {
      torch::Tensor selected_token_idxes = choose_lm_head_selected_token_idxes(
          sampling_params.selected_token_idxes,
          input.input_params,
          context_.get_parallel_args(),
          model_output.hidden_states.size(0),
          model_output.hidden_states.device());
      if (model_output.hidden_states.defined() &&
          selected_token_idxes.device() !=
              model_output.hidden_states.device()) {
        selected_token_idxes = selected_token_idxes
                                   .to(model_output.hidden_states.device(),
                                       /*non_blocking=*/false)
                                   .contiguous();
      }
      lm_head_selected_token_idxes = selected_token_idxes;
      const bool need_selected_hidden =
          input.return_selected_hidden ||
          (options_.cp_size() > 1 && options_.enable_speculative_decode() &&
           !model_output.aux_hidden_states.defined());
      if (need_selected_hidden) {
        // Emit both selected hidden and logits from a single lm_head pass so
        // the ConfidenceHead and CP speculative paths can consume hidden
        // without a second projection.
        logits = model_->logits(
            model_output.hidden_states, selected_token_idxes, selected_hidden);
        if (!selected_hidden.defined() &&
            model_output.hidden_states.defined()) {
          // ATB lm_head backend does not expose a second output tensor
          // (LmHeadParam::outputHidden is false), so we surface the hidden
          // here. selected_hidden must align row-for-row with `logits`, which
          // is produced in selected_token_idxes order. index_select reproduces
          // that order for any selection. We deliberately do NOT alias the full
          // hidden_states on a numel match: equal row count does not imply the
          // idxes are the identity permutation, and a full-but-reordered
          // selection would silently misalign hidden against logits. The gather
          // is one [num_selected, hidden] copy per decode step (not per layer),
          // negligible next to the forward.
          selected_hidden = model_output.hidden_states.index_select(
              /*dim=*/0, selected_token_idxes.to(torch::kLong));
        }
      } else {
        logits =
            model_->logits(model_output.hidden_states, selected_token_idxes);
      }
    }

    if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
        !options_.enable_speculative_decode()) {
      MULTI_MODEL_STEP_UNLOCK();
      if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
        wait_kv_push();
        return make_non_driver_forward_output(ForwardOutput{},
                                              std::move(failed_request_ids),
                                              std::move(unsafe_request_ids),
                                              /*eplb_enabled=*/false);
      }
      int ret = device_.synchronize_default_stream();
      CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
      wait_kv_push();
      return make_non_driver_forward_output(std::move(output),
                                            std::move(failed_request_ids),
                                            std::move(unsafe_request_ids),
                                            eplb_executor_ != nullptr);
    }

    // driver prepare model output
    if (sampling_params.selected_token_idxes.defined()) {
      output.selected_hidden = selected_hidden;
      output.do_sample = sampling_params.do_sample;
      output.logprobs = sampling_params.logprobs;
      output.max_top_logprobs = sampling_params.max_top_logprobs;
      if (!input.skip_sampling_for_logits_only) {
        auto sample_output = sample(logits, sampling_params);
        synchronize_sampled_tokens_for_schedule_overlap(
            sample_output.next_tokens, sampling_params);

        // beam search kernel
        BeamSearchOutput beam_search_output;
        if (sampling_params.use_beam_search &&
            sampling_params.acc_logprob.defined() &&
            sampling_params.acc_logprob.numel() > 0) {
          beam_search_output =
              beam_searcher_->forward(sampling_params.acc_logprob,
                                      sample_output.top_tokens,
                                      sample_output.top_logprobs);
        }

        // set sample output to output
        output.sample_output = sample_output;
        // set beam search output to output
        output.beam_search_output = beam_search_output;
      }
      output.logits = logits;
    }

    if (options_.enable_speculative_decode()) {
      torch::Tensor embeddings;
      if (model_output.aux_hidden_states.defined()) {
        embeddings = model_output.aux_hidden_states;
      } else {
        embeddings = model_output.hidden_states;
      }
      if (!input.input_params.meta.batch_forward_type.is_decode() &&
          !is_spec_draft_) {
        // Target prefill passes all context rows to the draft KV writer.
        output.sample_output.embeddings = embeddings;
        output.aux_hidden_restore_index =
            std::move(model_output.aux_hidden_restore_index);
      } else if (sampling_params.selected_token_idxes.defined()) {
        if (options_.cp_size() > 1 &&
            !model_output.aux_hidden_states.defined()) {
          // CP speculative decode without aux capture (e.g. the MTP draft,
          // which feeds these embeddings back as its input_embedding): reuse
          // the lm_head's selected hidden rows.
          CHECK(selected_hidden.defined())
              << "selected_hidden must be defined when "
                 "selected_token_idxes is defined.";
          output.sample_output.embeddings = selected_hidden;
        } else {
          // Aux-hidden-capture algorithms (DFlash/DFlash2/Eagle3/DSpark)
          // consume the captured columns from the aux embeddings, so select
          // their rows from that tensor. selected_hidden is not the fallback
          // here: need_selected_hidden above excludes the aux path (its CP
          // branch requires !aux_hidden_states.defined()), so unless a caller
          // separately requested return_selected_hidden the lm_head pass
          // never produced it -- and even then it would carry the hidden rows
          // only, without the captured aux columns the draft consumes. Under
          // CP the decode batch runs without a cp_context (EagerRunner only
          // builds one for prefill batches), so the rows are already
          // global-ordered and the selection matches the non-CP path.
          output.sample_output.embeddings = embeddings.index_select(
              /*dim=*/0, lm_head_selected_token_idxes);
        }
      }
    }
  }

  MULTI_MODEL_STEP_UNLOCK();
  bool should_sync_default_stream = true;
#if defined(USE_NPU)
  should_sync_default_stream =
      !can_skip_npu_graph_decode_sync(input.input_params);
#endif
  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    wait_kv_push();
    output.failed_request_ids = std::move(failed_request_ids);
    output.unsafe_request_ids = std::move(unsafe_request_ids);
    output.retained_inputs.emplace_back(std::make_shared<ForwardInput>(input));
    if (enable_schedule_overlap() && record_ready_event) {
      output.ready_event = record_current_stream_event(device_);
    }
    return output;
  }
  if (should_sync_default_stream) {
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
  }

  wait_kv_push();
  output.failed_request_ids = std::move(failed_request_ids);
  output.unsafe_request_ids = std::move(unsafe_request_ids);

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  if (should_sync_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }

  return output;
}

}  // namespace xllm
