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

#include "core/distributed_runtime/disagg_pd_service_impl.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/macros.h"
#include "core/common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "core/scheduler/disagg_pd_scheduler.h"
#include "core/scheduler/perf_model.h"
#include "core/util/scope_guard.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "tests/core/common/etcd_registry_fixture.h"
#include "xservice.pb.h"

namespace xllm {
namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string text;
    for (int32_t id : ids) {
      text += std::to_string(id);
    }
    return text;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

// Minimal engine for the decode admission path: the failure classes under
// test return before any forward or tokenizer access, so every override is
// either a small stub or unreachable.
class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    model_args_.n_layers(1)
        .vocab_size(128)
        .hidden_size(8)
        .intermediate_size(16)
        .n_heads(2)
        .n_kv_heads(1);
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    static const TokenizerArgs args;
    return args;
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  bool init() override { return true; }

  bool link_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                    const std::vector<std::string>& /*addrs*/,
                    const std::vector<uint16_t>& /*ports*/,
                    int32_t /*src_dp_size*/,
                    int32_t /*src_kv_split_size*/) override {
    return true;
  }

  bool unlink_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                      const std::vector<std::string>& /*addrs*/,
                      const std::vector<uint16_t>& /*ports*/,
                      int32_t /*src_dp_size*/,
                      int32_t /*src_kv_split_size*/) override {
    return true;
  }

  bool pull_kv_blocks(
      int32_t /*src_dp_size*/,
      int32_t /*src_dp_rank*/,
      const std::vector<uint64_t>& /*src_cluster_ids*/,
      const std::vector<std::string>& /*src_addrs*/,
      int32_t /*dst_dp_rank*/,
      const std::vector<KVTransferMapping>& /*mappings*/) override {
    return true;
  }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

// The service impl's constructor requires an initialized XServiceClient;
// expose the protected no-services constructor without starting the
// dispatch thread or the RPC server.
class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine, const Options& options)
      : DisaggPDScheduler(engine, options, /*start_services=*/false) {}
};

class TestDisaggPDServiceImpl final : public DisaggPDServiceImpl {
 public:
  using DisaggPDServiceImpl::DisaggPDServiceImpl;
  using DisaggPDServiceImpl::generate_request;
};

class GenerationReceiver final : public xllm_service::proto::XllmRpcService {
 public:
  void Generations(google::protobuf::RpcController* /*unused*/,
                   const proto::DisaggStreamGenerations* request,
                   proto::StatusSet* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    received = *request;
    for (int32_t index = 0; index < request->gens_size(); ++index) {
      response->add_all_status()->set_ok(true);
    }
  }

  proto::DisaggStreamGenerations received;
};

// Restores a process-global config field on scope exit, so one test's
// toggles cannot leak into the next case.
template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& field, T value) : field_(field), previous_(field) {
    field_ = std::move(value);
  }
  ~ScopedConfigValue() { field_ = std::move(previous_); }
  ScopedConfigValue(const ScopedConfigValue&) = delete;
  ScopedConfigValue& operator=(const ScopedConfigValue&) = delete;

 private:
  T& field_;
  T previous_;
};

}  // namespace

namespace {

// The fixture outlives the process-global client, so it serves the registry
// RPCs that init() makes without an external etcd.
test::FakeRegistry& registry_fixture() {
  static test::FakeRegistry registry;
  return registry;
}

class DisaggPDServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Construct the fixture in its own statement. In C++17 a call's postfix
    // expression is sequenced before its arguments, so passing
    // registry_fixture().address() directly would build the process-global
    // client first and tear the server down under the client's open streams.
    test::FakeRegistry& registry = registry_fixture();
    // Initialize the process-global client once: the constructor of the
    // service under test LOG(FATAL)s without it. The registry fixture serves
    // the real client's RPCs, and the admission failures under test never
    // exercise the client further.
    static const bool xservice_initialized =
        XServiceClient::get_instance()->init(registry.address(),
                                             "disagg-pd-admission-test");
    ASSERT_TRUE(xservice_initialized);
  }

  static void TearDownTestSuite() {
    // Stop the client's watch and background threads while the registry
    // fixture is still serving; leaving that to static destruction tears the
    // server down under the client.
    XServiceClient::get_instance()->shutdown();
  }
};

TEST_F(DisaggPDServiceImplTest, AdmissionFailureStatusMatchesFailureClass) {
  // 422 (kDecodeConstraintRejectedStatusCode) makes the prefill scheduler
  // release the request's KV blocks as unreconcilable; before the fix every
  // nullptr from generate_request with a grammar_source present was reported
  // as 422, so unrelated failures (best_of without prefix cache, invalid VLM
  // metadata) were misdiagnosed as permanent constraint rejections. The
  // status must now follow the actual failure class.
  struct Case final {
    std::string name;
    bool best_of_without_prefix_cache;
    bool with_grammar_source;
    bool xgrammar_backend;
    int32_t expected_status;
  };
  const std::vector<Case> cases = {
      {"best_of>1 without prefix cache and a grammar stays a generic error",
       /*best_of_without_prefix_cache=*/true,
       /*with_grammar_source=*/true,
       /*xgrammar_backend=*/false,
       /*expected_status=*/500},
      {"constraint rebuild failure stays permanent",
       /*best_of_without_prefix_cache=*/false,
       /*with_grammar_source=*/true,
       /*xgrammar_backend=*/false,
       /*expected_status=*/kDecodeConstraintRejectedStatusCode},
      {"best_of>1 without prefix cache and no grammar stays a generic error",
       /*best_of_without_prefix_cache=*/true,
       /*with_grammar_source=*/false,
       /*xgrammar_backend=*/false,
       /*expected_status=*/500},
  };

  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    DisaggPDScheduler::Options options;
    options.enable_disagg_pd(true)
        .enable_schedule_overlap(false)
        .instance_role(InstanceRole::PREFILL)
        .max_tokens_per_batch(32)
        .max_seqs_per_batch(4)
        .max_tokens_per_chunk_for_prefill(32)
        .dp_size(1);
    TestDisaggPDScheduler scheduler(&engine, options);
    DisaggPDServiceImpl impl(&scheduler, &engine);

    // The best_of path requires the decode-side prefix cache to be off; the
    // constraint path requires the xgrammar backend to reject the spec (a
    // disabled backend in USE_XGRAMMAR builds, the #else branch otherwise).
    ScopedConfigValue<bool> prefix_cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(),
        !c.best_of_without_prefix_cache);
    ScopedConfigValue<std::string> backend_guard(
        ServiceConfig::get_instance().structured_output_backend(),
        c.xgrammar_backend ? "xgrammar" : "builtin");

    proto::DisaggRequests requests;
    proto::DisaggRequest* req = requests.add_reqs();
    req->set_req_id("disagg-pd-admission-failure");
    req->set_prompt("prompt");
    req->set_n(1);
    req->set_best_of(c.best_of_without_prefix_cache ? 2 : 1);
    req->set_max_tokens(16);
    req->set_max_context_len(64);
    req->set_seq_capacity(64);
    req->set_offline(true);
    if (c.with_grammar_source) {
      req->set_grammar_type(
          static_cast<int32_t>(GrammarSpec::Type::JSON_SCHEMA));
      req->set_grammar_source(R"({"type":"object"})");
      req->set_grammar_strict(true);
    }

    proto::DisaggResponses responses;
    impl.decode_recv_new_requests(&requests, &responses);
    ASSERT_EQ(responses.resps_size(), 1);
    EXPECT_EQ(responses.resps(0).req_id(), req->req_id());
    EXPECT_EQ(responses.resps(0).status_code(), c.expected_status);
  }
}

TEST_F(DisaggPDServiceImplTest, RequestOutputCallbacksOutliveService) {
  GenerationReceiver receiver;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&receiver, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);

  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  DisaggPDScheduler::Options options;
  options.enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::DECODE)
      .max_tokens_per_batch(32)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(32)
      .dp_size(1);
  TestDisaggPDScheduler scheduler(&engine, options);
  std::optional<TestDisaggPDServiceImpl> service(
      std::in_place, &scheduler, &engine);

  proto::DisaggRequest wire_request;
  wire_request.set_req_id("retained-decode-output");
  wire_request.set_service_req_id("owning-service-request");
  wire_request.set_source_xservice_addr(address);
  wire_request.set_prompt("prompt");
  wire_request.add_prompt_tokens(1);
  wire_request.set_n(1);
  wire_request.set_best_of(1);
  wire_request.set_max_tokens(16);
  wire_request.set_max_context_len(64);
  wire_request.set_seq_capacity(64);
  auto request = service->generate_request(wire_request);
  ASSERT_NE(request, nullptr);

  // Registry teardown retires the RPC service before queued responses drain.
  // Overwrite its retired storage so this regression cannot pass merely
  // because the allocator left the old client pointer intact.
  void* retired_storage = &service.value();
  service.reset();
  std::memset(retired_storage, 0, sizeof(TestDisaggPDServiceImpl));

  RequestOutput output;
  output.request_id = request->request_id();
  output.service_request_id = request->service_request_id();
  output.target_xservice_addr = request->source_xservice_addr();
  EXPECT_TRUE(request->state().output_func(output));
  ASSERT_EQ(receiver.received.gens_size(), 1);
  EXPECT_EQ(receiver.received.gens(0).req_id(), output.request_id);
  EXPECT_EQ(request->state().outputs_func({output, output}),
            std::vector<bool>({true, true}));
  ASSERT_EQ(receiver.received.gens_size(), 2);
  EXPECT_EQ(receiver.received.gens(1).service_req_id(),
            output.service_request_id);
  server.Stop(0);
  server.Join();
}

}  // namespace
}  // namespace xllm
