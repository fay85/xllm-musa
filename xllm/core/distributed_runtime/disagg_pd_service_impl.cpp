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

#include "core/distributed_runtime/disagg_pd_service_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/types.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "distributed_runtime/llm_engine.h"
#include "framework/request/request_output.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "util/utils.h"

namespace xllm {

DisaggPDServiceImpl::DisaggPDServiceImpl(DisaggPDScheduler* scheduler,
                                         Engine* engine)
    : scheduler_(scheduler), engine_(engine) {
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
}

std::shared_ptr<const JsonObjectGrammar>
DisaggPDServiceImpl::get_json_object_grammar(bool reasoning_enabled,
                                             std::string* error) {
  std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
  std::shared_ptr<const JsonObjectGrammar>& grammar =
      reasoning_enabled ? json_reasoning_grammar_ : json_object_grammar_;
  if (grammar == nullptr) {
    grammar = JsonObjectGrammar::create_from_tokenizer(
        *engine_->tokenizer(),
        engine_->model_args().eos_token_id(),
        engine_->model_args().stop_token_ids(),
        engine_->model_args().vocab_size(),
        reasoning_enabled,
        error);
  }
  return grammar;
}

std::shared_ptr<Request> DisaggPDServiceImpl::generate_request(
    const proto::DisaggRequest& req,
    bool* constraint_rejected) {
  // Only a structured-output constraint rebuild failure is unreconcilable
  // for this migration; mark it so the caller can pick the permanent
  // status instead of a retryable one. Nullptr-safe for callers that do
  // not care about the failure class.
  const auto note_constraint_rejection = [constraint_rejected]() {
    if (constraint_rejected != nullptr) {
      *constraint_rejected = true;
    }
  };
  // best_of > 1 in disaggregated PD requires prefix cache on the decode
  // instance: only the first sequence's KV is pulled from prefill, and
  // the remaining best_of-1 sequences are expanded locally and reuse the
  // first sequence's prompt KV via prefix cache.
  if (req.best_of() > 1 &&
      !::xllm::KVCacheConfig::get_instance().enable_prefix_cache()) {
    LOG(ERROR) << "best_of > 1 in disaggregated PD mode requires "
               << "prefix caching enabled on the decode instance, "
               << "request_id=" << req.req_id()
               << ", best_of=" << req.best_of();
    return nullptr;
  }

  std::string prompt = req.prompt();
  std::vector<int> prompt_tokens(req.prompt_tokens().begin(),
                                 req.prompt_tokens().end());

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = req.frequency_penalty();
  sampling_param.presence_penalty = req.presence_penalty();
  sampling_param.repetition_penalty = req.repetition_penalty();
  sampling_param.temperature = req.temperature();
  sampling_param.top_p = req.top_p();
  sampling_param.top_k = req.top_k();
  sampling_param.logprobs = req.logprobs();
  sampling_param.top_logprobs = req.top_logprobs();
  sampling_param.is_embeddings = req.is_embeddings();
  sampling_param.json_object = req.json_object();
  const bool json_object = sampling_param.json_object;

  SchedulerParam scheduler_param;
  scheduler_param.offline = req.offline();
  scheduler_param.priority = static_cast<xllm::RequestPriority>(req.priority());
  if (!req.offline()) {
    scheduler_param.ttft_slo_ms = req.ttft_slo_ms();
    scheduler_param.tpot_slo_ms = req.tpot_slo_ms();
    scheduler_param.ttlt_slo_ms = req.ttlt_slo_ms();
    scheduler_param.tpot_priority_weight = req.tpot_priority_weight();
    scheduler_param.ttft_priority_weight = req.ttft_priority_weight();
    scheduler_param.ttlt_priority_weight = req.ttlt_priority_weight();
    scheduler_param.priority_weight = req.priority_weight();
  }

  std::unordered_set<int32_t> stop_tokens;
  for (auto& stop_token_id : req.stop_token_ids()) {
    stop_tokens.insert(stop_token_id);
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  for (auto& stop_sequence : req.stop_sequences()) {
    auto stop_seq_tokens = std::vector<int32_t>(
        stop_sequence.seq_tokens().begin(), stop_sequence.seq_tokens().end());
    stop_sequences.push_back(std::move(stop_seq_tokens));
  }
  StoppingChecker stopping_checker(req.max_tokens(),
                                   req.max_context_len(),
                                   req.eos_token_id(),
                                   req.ignore_eos(),
                                   std::move(stop_tokens),
                                   std::move(stop_sequences));

  // Requests can outlive the registry-owned RPC service during shutdown.
  // The master keeps the client and its channels alive until responses drain.
  auto output_callback = [xservice_client = xservice_client_](
                             const RequestOutput& output) -> bool {
    // response to xllm service to avoid the redirect cost.
    if (xservice_client == nullptr) {
      return false;
    }
    auto return_status = xservice_client->generations({output});
    CHECK_EQ(return_status.size(), 1)
        << "return size of generations is not equal to 1";
    return return_status[0];
  };

  auto batch_output_callback = [xservice_client = xservice_client_](
                                   const std::vector<RequestOutput>& outputs) {
    // response to xllm service to avoid the redirect cost.
    if (xservice_client == nullptr) {
      return std::vector<bool>(outputs.size(), false);
    }
    return xservice_client->generations(outputs);
  };

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(sampling_param),
                         std::move(scheduler_param),
                         std::move(stopping_checker),
                         req.seq_capacity(),
                         req.n(),
                         req.best_of(),
                         req.logprobs(),
                         req.stream(),
                         req.echo(),
                         req.skip_special_tokens(),
                         scheduler_->enable_schedule_overlap(),
                         output_callback,
                         batch_output_callback);
  if (req.has_mm_data()) {
    auto metadata = proto_to_mmdata_entry(req.mm_data());
    if (!metadata) {
      LOG(ERROR) << "Invalid VLM PD metadata, request_id=" << req.req_id();
      return nullptr;
    }
    req_state.mm_data = std::move(*metadata);
    req_state.multimodal_metadata_only = true;
  }
  req_state.include_stop_str_in_output = req.include_stop_str_in_output();

  if (json_object && req.grammar_source().empty()) {
    if (!ServiceConfig::get_instance().enable_json_object_output()) {
      // Match the single-instance flag semantics: json_object requests are
      // accepted without the constraint when this instance disables it.
      // Without this gate the sequence-level grammar tracker would stay
      // active while forward inputs carry no grammar masks (the batch input
      // builder respects the same flag), surfacing as random mid-generation
      // grammar violations.
      LOG(WARNING)
          << "json_object request " << req.req_id()
          << " arrives at a decode instance with enable_json_object_output="
             "false; accepting without the constraint. Enable the flag on "
             "every PD instance for constrained decoding.";
    } else {
      std::string grammar_error;
      req_state.json_object_grammar =
          get_json_object_grammar(req.json_reasoning_enabled(), &grammar_error);
      if (req_state.json_object_grammar == nullptr) {
        LOG(ERROR) << "Failed to initialize json_object constraint: "
                   << grammar_error << ", request_id=" << req.req_id();
        return nullptr;
      }
      req_state.json_reasoning_enabled = req.json_reasoning_enabled();
      req_state.constraint_machine =
          wrap_builtin_machine(req_state.json_object_grammar);
    }
  }

  // The allocation request precedes prefill's first sampled token. Rebuild
  // the same grammar before that token arrives and is committed on decode.
  if (!req.grammar_source().empty()) {
#if defined(USE_XGRAMMAR)
    if (!ServiceConfig::get_instance().use_xgrammar()) {
      LOG(ERROR) << "Migrated xgrammar constraint requires the xgrammar backend"
                 << ", request_id=" << req.req_id();
      note_constraint_rejection();
      return nullptr;
    }
    XGrammarMachineCache* cache = nullptr;
    {
      // Admission RPCs may run concurrently; never replace a published cache.
      std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
      if (xgrammar_machines_ == nullptr) {
        xgrammar_machines_ = std::make_unique<XGrammarMachineCache>(
            *engine_->tokenizer(), engine_->model_args());
      }
      cache = xgrammar_machines_.get();
    }
    GrammarSpec spec;
    spec.type = static_cast<GrammarSpec::Type>(req.grammar_type());
    spec.source = req.grammar_source();
    spec.strict = req.grammar_strict();
    std::string grammar_error;
    std::shared_ptr<XGrammarConstraintMachine> machine =
        cache->get(spec, grammar_error);
    if (machine == nullptr) {
      LOG(ERROR) << "Failed to rebuild xgrammar constraint: " << grammar_error
                 << ", request_id=" << req.req_id();
      note_constraint_rejection();
      return nullptr;
    }
    req_state.xgrammar_spec = std::move(spec);
    req_state.constraint_machine = machine;
    // The per-sequence matcher construction reads the compiled grammar;
    // expose it from the machine-wrapped grammar.
    req_state.xgrammar_constraint = machine->grammar();
#else
    // A build without xgrammar must not silently discard the incoming spec.
    LOG(ERROR) << "Migrated constraint requires a build with USE_XGRAMMAR"
               << ", request_id=" << req.req_id();
    note_constraint_rejection();
    return nullptr;
#endif
  }

  req_state.pd_reservation_id = req.reservation_id();
  auto new_request = std::make_shared<Request>(req.req_id(),
                                               req.x_request_id(),
                                               req.x_request_time(),
                                               std::move(req_state),
                                               req.service_req_id(),
                                               req.source_xservice_addr());

  // add one sequence, rest will be added by scheduler
  if (req.request_token_accounting()) {
    new_request->enable_forwarded_token_output();
  }
  return new_request;
}

void DisaggPDServiceImpl::cancel_request(
    const proto::CancelRequestRequest* request,
    proto::CancelRequestResponse* response) {
  response->set_result(scheduler_->cancel_request(*request));
}

void DisaggPDServiceImpl::release_reservation(
    const proto::ReleaseReservationRequest* request,
    proto::ReleaseReservationResponse* response) {
  bool unsafe_leased = false;
  const bool released =
      scheduler_->release_reservation(request->req_id(),
                                      request->reservation_id(),
                                      request->unsafe(),
                                      &unsafe_leased);
  response->set_unsafe_leased(unsafe_leased);
  response->set_result(released
                           ? proto::ReleaseReservationResponse::RELEASED
                           : proto::ReleaseReservationResponse::NOT_WAITING);
}

void DisaggPDServiceImpl::decode_recv_new_requests(
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response) {
  for (auto& req : request->reqs()) {
    auto resp = response->add_resps();
    resp->set_req_id(req.req_id());
    bool constraint_rejected = false;
    auto new_request = generate_request(req, &constraint_rejected);
    if (new_request == nullptr) {
      // Only a constraint-rebuild failure is unreconcilable for the
      // migrated request (the prefill side releases its blocks instead of
      // retrying); unrelated nullptr failures are generic decode errors.
      resp->set_status_code(
          constraint_rejected ? kDecodeConstraintRejectedStatusCode : 500);
      continue;
    }

    auto& sequences = new_request->sequences();
    Sequence* sequence = sequences[0].get();
    for (const auto& admitted_sequence : sequences) {
      admitted_sequence->set_cache_allocation_mode(
          CacheAllocationMode::DECODE_RECEIVE);
    }

    const DecodeAdmissionResult admission = scheduler_->try_admit(new_request);
    if (admission != DecodeAdmissionResult::ADMITTED) {
      if (admission == DecodeAdmissionResult::REJECTED &&
          scheduler_->exceeds_decode_capacity(sequence)) {
        LOG(ERROR) << "Decode cannot fit request prompt at any load, "
                   << "request_id=" << req.req_id()
                   << ", prompt_tokens=" << sequence->num_prompt_tokens();
        resp->set_status_code(kDecodeAddNewPromptTooLongStatusCode);
      } else {
        // Current cache pressure can be relieved, so Prefill may retry.
        resp->set_status_code(404);
      }
    } else {
      // push the request to scheduler request buffer
      bool success =
          scheduler_->decode_schedule(new_request, request->prefill_name());
      if (!success) {
        LOG(ERROR) << "Failed to schedule new decode instance request: "
                   << req.req_id();
        // request and blocks are released in scheduler
        resp->set_status_code(500);
        continue;
      }

      auto dp_rank = sequence->dp_rank();
      resp->set_dp_rank(dp_rank);

      std::vector<int32_t> block_ids;
      if (sequence->kv_state().has_multi_block_export()) {
        const auto export_view = sequence->kv_state().multi_block_export_view();
        for (const auto& [block_type, blocks_ptr] : export_view) {
          proto::KVTransferGroup* group = resp->add_groups();
          group->set_group_id(cache_group_id(block_type));
          const size_t shared_num =
              block_type == BlockType::KV
                  ? sequence->kv_state().shared_blocks_num(BlockType::KV)
                  : 0;
          CHECK_LE(shared_num, blocks_ptr->size());
          group->set_remote_shared_num(static_cast<uint32_t>(shared_num));
          group->mutable_ids()->Reserve(blocks_ptr->size());
          for (const Block& block : *blocks_ptr) {
            // Invalid placeholders are legitimate for SWA: the sliding
            // window release loop leaves moved-from Blocks in place so
            // positional indexing stays stable, and both P and D produce
            // the same invalid pattern for the slid-out window. Ship them
            // as max-uint64 sentinels; BatchInputBuilder skips
            // positions where the local block id is negative, so the
            // remote id at those positions is never dereferenced. The
            // CHECK below only refuses invalid blocks under types where
            // slide-out cannot produce them (history KV), preserving the
            // guard for genuine allocation bugs.
            const int64_t block_id = block.is_valid() ? block.id() : -1;
            if (!block.is_valid()) {
              CHECK(block_type == BlockType::SWA)
                  << "Decode allocated an invalid grouped KV block under a "
                  << "non-SWA leaf, group_id=" << cache_group_id(block_type);
            }
            group->mutable_ids()->Add(static_cast<uint64_t>(block_id));
          }
        }
      } else {
        const size_t shared_num =
            sequence->kv_state().shared_blocks_num(BlockType::KV);
        const auto blocks = sequence->kv_state().blocks(BlockType::KV);

        // Collect block IDs
        block_ids.reserve(blocks.size() - shared_num);
        proto::KVTransferGroup* group = resp->add_groups();
        group->set_group_id(cache_group_id(BlockType::KV));
        group->set_remote_shared_num(static_cast<uint32_t>(shared_num));
        group->mutable_ids()->Reserve(blocks.size() - shared_num);
        for (size_t i = shared_num; i < blocks.size(); i++) {
          int32_t block_id = blocks[i].id();
          group->mutable_ids()->Add(block_id);
          block_ids.push_back(block_id);
        }
      }
      if (has_linear_attention_layers(engine_->model_args())) {
        const int32_t linear_state_id = sequence->get_linear_state_slot_id();
        CHECK_GE(linear_state_id, 0)
            << "Decode did not allocate a linear-state slot.";
        proto::KVTransferGroup* group = resp->add_groups();
        group->set_group_id(cache_group_id(BlockType::LINEAR));
        group->add_ids(static_cast<uint64_t>(linear_state_id));
      }
      resp->set_status_code(200);
    }
  }
}

// TODO: support embedding later, now we only support tokens
void DisaggPDServiceImpl::decode_recv_first_generation(
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response) {
  // TODO: we only support one request generation currently
  for (auto& gen : request->multi_gens()) {
    // Process the first token from the tokens array
    if (gen.tokens().empty()) {
      response->set_ok(false);
      return;
    }

    const auto& first_token = gen.tokens(0);
    std::vector<int64_t> top_tokens(first_token.top_tokens().begin(),
                                    first_token.top_tokens().end());
    std::vector<float> top_logprobs(first_token.top_logprobs().begin(),
                                    first_token.top_logprobs().end());
    std::vector<uint64_t> cluster_ids(gen.cluster_ids().begin(),
                                      gen.cluster_ids().end());
    std::vector<std::string> addrs(gen.addrs().begin(), gen.addrs().end());
    std::vector<KVTransferMapping> source_mappings;
    source_mappings.reserve(gen.source_groups_size());
    for (const proto::KVTransferGroup& proto_group : gen.source_groups()) {
      KVTransferMapping mapping;
      mapping.group_id = proto_group.group_id();
      mapping.remote_ids.assign(proto_group.ids().begin(),
                                proto_group.ids().end());
      source_mappings.emplace_back(std::move(mapping));
    }
    torch::Tensor mtp_bootstrap_embedding;
    if (gen.has_mtp_bootstrap_embedding()) {
      mtp_bootstrap_embedding =
          util::proto_to_torch(gen.mtp_bootstrap_embedding());
    }

    bool success = scheduler_->decode_recv_first_generation(
        gen.req_id(),
        first_token.token_id(),
        first_token.has_logprob(),
        first_token.logprob(),
        first_token.time_to_first_token_latency_seconds(),
        gen.upstream_elapsed_seconds(),
        std::move(top_tokens),
        std::move(top_logprobs),
        gen.kv_cache_transfer_mode(),
        std::move(cluster_ids),
        std::move(addrs),
        std::move(source_mappings),
        gen.dp_size(),
        gen.dp_rank(),
        first_token.prefill_text_withheld(),
        mtp_bootstrap_embedding,
        gen.num_cached_tokens(),
        gen.has_phase_timing() ? &gen.phase_timing() : nullptr,
        gen.has_mrope_position_delta()
            ? std::optional<int32_t>(gen.mrope_position_delta())
            : std::nullopt,
        gen.has_confirmed_upstream_output_tokens()
            ? std::optional<uint64_t>(gen.confirmed_upstream_output_tokens())
            : std::nullopt);
    if (!success) {
      response->set_ok(false);
      return;
    }
  }

  response->set_ok(true);
}

void DisaggPDServiceImpl::link_instance(
    const proto::InstanceClusterInfo* request,
    proto::Status* response) {
  // Extract parameters from proto request
  std::vector<uint64_t> cluster_ids(request->cluster_ids().begin(),
                                    request->cluster_ids().end());
  std::vector<std::string> addrs(request->addrs().begin(),
                                 request->addrs().end());
  std::vector<uint16_t> ports(request->ports().begin(), request->ports().end());
  int32_t dp_size = request->dp_size();
  int32_t kv_split_size = request->kv_split_size();

  // Call scheduler's link_instance method
  bool success = scheduler_->link_instance(request->instance_name(),
                                           cluster_ids,
                                           addrs,
                                           ports,
                                           dp_size,
                                           kv_split_size);

  if (!success) {
    LOG(ERROR) << "Failed to link instance: " << request->instance_name();
    response->set_ok(false);
    return;
  }

  response->set_ok(true);
}

void DisaggPDServiceImpl::unlink_instance(
    const proto::InstanceClusterInfo* request,
    proto::Status* response) {
  // Extract parameters from proto request
  std::vector<uint64_t> cluster_ids(request->cluster_ids().begin(),
                                    request->cluster_ids().end());
  std::vector<std::string> addrs(request->addrs().begin(),
                                 request->addrs().end());
  std::vector<uint16_t> ports(request->ports().begin(), request->ports().end());
  int32_t dp_size = request->dp_size();
  int32_t kv_split_size = request->kv_split_size();

  // Call scheduler's unlink_instance method
  bool success = scheduler_->unlink_instance(request->instance_name(),
                                             cluster_ids,
                                             addrs,
                                             ports,
                                             dp_size,
                                             kv_split_size);

  if (!success) {
    LOG(ERROR) << "Failed to unlink instance: " << request->instance_name();
    response->set_ok(false);
    return;
  }

  response->set_ok(true);
}

}  // namespace xllm
