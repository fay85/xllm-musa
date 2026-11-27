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

#include "core/distributed_runtime/dit_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/dit_engine.h"
#include "core/framework/request/dit_request.h"
#include "core/platform/device_name_utils.h"
#include "core/runtime/xservice_client.h"
#include "core/scheduler/scheduler_factory.h"
#include "core/util/scope_guard.h"
#include "core/util/timer.h"
#include "models/model_registry.h"

namespace xllm {
DiTMaster::DiTMaster(const Options& options)
    : Master(options, EngineType::DIT) {
  if (!is_leader()) {
    return;
  }

  CHECK(engine_->init());

  DiTScheduler::Options scheduler_options;
  scheduler_options.max_request_per_batch(options.max_requests_per_batch())
      .disable_log_stats(options.disable_log_stats())
      .server_idx(options.server_idx());

  scheduler_ = create_dit_scheduler(engine_.get(), scheduler_options);
  LOG(INFO) << "created dit scheduler in DiTMaster.";

  if (options_.enable_service_routing()) {
    XServiceClient* client = XServiceClient::get_instance();
    client->set_engine(engine_.get());

    const auto* dit_engine = static_cast<const DiTEngine*>(engine_.get());
    const size_t local_device_count = dit_engine->options().devices().size();
    const int32_t nnodes = dit_engine->options().nnodes();
    const int32_t tp_size = dit_engine->options().tp_size();
    CHECK_GT(local_device_count, 0);
    CHECK_GT(nnodes, 0);
    CHECK_LE(
        local_device_count,
        std::numeric_limits<uint32_t>::max() / static_cast<size_t>(nnodes));
    const size_t device_count =
        local_device_count * static_cast<size_t>(nnodes);
    CHECK_GT(tp_size, 0);
    client->set_one_shot_engine_profile(static_cast<uint32_t>(device_count),
                                        static_cast<uint32_t>(tp_size));
    if (!client->init(options_.etcd_addr().value_or(""),
                      options_.instance_name().value_or(""),
                      nullptr,
                      options_.etcd_namespace().value_or(""))) {
      LOG(FATAL) << "XServiceClient init failed for DiT Engine";
    }

    InstanceInfo instance_info;
    instance_info.name = options_.instance_name().value_or("");
    instance_info.rpc_address = instance_info.name;
    CHECK(options_.instance_role() == InstanceRole::DEFAULT)
        << "DiT task-only service routing requires DEFAULT instance role";
    instance_info.type = options_.instance_role().to_string();
    instance_info.dp_size = 1;
    client->register_instance(instance_info);
  }

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"DiTMaster.request");
  LOG(INFO) << "ThreadPool with " << options.num_request_handling_threads()
            << " threads created in DiTMaster.";
}

DiTMaster::~DiTMaster() {
  if (options_.enable_service_routing() && is_leader()) {
    XServiceClient::get_instance()->shutdown();
  }
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void DiTMaster::handle_request(DiTRequestParams params,
                               std::optional<Call*> call,
                               DiTOutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback)](const DiTRequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  // add into the queue
  threadpool_->schedule([this,
                         params = std::move(params),
                         callback = std::move(cb),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry. Dismissed
    // right before DiTRequest takes ownership; any early return releases it.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    Status parse_status;
    const bool preserve_image_alpha = ModelRegistry::get_preserve_image_alpha(
        engine_->model_args().model_type());
    if (!params.parse_input_params(parse_status, preserve_image_alpha)) {
      callback(DiTRequestOutput(std::move(parse_status)));
      return;
    }
    if (!params.verify_params(parse_status)) {
      CALLBACK_WITH_ERROR(parse_status.code(), parse_status.message());
      return;
    }

    DiTRequestState dit_state(std::move(params.input_params),
                              std::move(params.generation_params),
                              callback,
                              nullptr,
                              params.request_kind(),
                              call);
    auto request = std::make_shared<DiTRequest>(params.request_id,
                                                params.x_request_id,
                                                params.x_request_time,
                                                std::move(dit_state),
                                                /*service_request_id=*/"",
                                                /*source_xservice_addr=*/"",
                                                get_rate_limiter());
    rate_limit_guard.dismiss();

    if (!scheduler_->add_request(request)) {
      callback(Status(StatusCode::RESOURCE_EXHAUSTED,
                      "No available resources to schedule request",
                      FailureReason::TEMPORARY_OVERLOAD));
    }
  });
}

void DiTMaster::handle_batch_request(std::vector<DiTRequestParams> params_vec,
                                     BatchDiTOutputCallback callback) {
  const size_t num_requests = params_vec.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(params_vec[i]),
                   std::nullopt,
                   [i, callback](const DiTRequestOutput& output) {
                     return callback(i, output);
                   });
  }
}

void DiTMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "DiTMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    LOG(INFO) << "DiTMaster loop thread exiting.";
    running_.store(false, std::memory_order_relaxed);
  });
}

void DiTMaster::generate() {
  LOG(INFO) << "into DiTMaster::generate";

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

}  // namespace xllm
