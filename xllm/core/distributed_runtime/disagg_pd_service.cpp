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

#include "disagg_pd_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

namespace xllm {

DisaggPDService::DisaggPDService(DisaggPDScheduler* scheduler, Engine* engine) {
  disagg_pd_service_impl_ =
      std::make_unique<DisaggPDServiceImpl>(scheduler, engine);
}

void DisaggPDService::ReleaseReservation(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::ReleaseReservationRequest* request,
    proto::ReleaseReservationResponse* response,
    ::google::protobuf::Closure* done) {
  disagg_pd_service_impl_->release_reservation_async(request, response, done);
}

void DisaggPDService::AddNewRequests(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  // try to allocate blocks for new requests
  disagg_pd_service_impl_->decode_recv_new_requests(request, response);
}

// TODO: support embedding later, now we only support tokens
void DisaggPDService::FirstGeneration(
    ::google::protobuf::RpcController* controller,
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  // Receive first token from Prefill asynchronously on multi-lane Decode pull pool
  disagg_pd_service_impl_->decode_recv_first_generation_async(
      request, response, done);
}

void DisaggPDService::MultiGenerations(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  const int gens_count = request != nullptr ? request->multi_gens_size() : 0;
  LOG(WARNING) << "Rejecting unsupported legacy MultiGenerations RPC in "
               << "DisaggPDService, multi_gens_size=" << gens_count;
  if (response != nullptr) {
    response->set_ok(false);
  }
}

void DisaggPDService::SendPullSignal(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::PullSignal* /*request*/,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  LOG(WARNING)
      << "Rejecting unsupported legacy SendPullSignal RPC in DisaggPDService";
  if (response != nullptr) {
    response->set_ok(false);
  }
}

void DisaggPDService::LinkInstance(
    ::google::protobuf::RpcController* controller,
    const proto::InstanceClusterInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->link_instance(request, response);
}

void DisaggPDService::UnlinkInstance(
    ::google::protobuf::RpcController* controller,
    const proto::InstanceClusterInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->unlink_instance(request, response);
}

void DisaggPDService::EnsureLink(
    ::google::protobuf::RpcController* controller,
    const proto::EnsureLinkRequest* request,
    proto::EnsureLinkResult* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->ensure_link(request, response);
}

void DisaggPDService::DrainLink(
    ::google::protobuf::RpcController* controller,
    const proto::DrainLinkRequest* request,
    proto::EnsureLinkResult* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->drain_link(request, response);
}

void DisaggPDService::QueryRequest(
    ::google::protobuf::RpcController* controller,
    const proto::AttemptControlRequest* request,
    proto::AttemptControlResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->query_request(request, response);
}

void DisaggPDService::CancelRequest(
    ::google::protobuf::RpcController* controller,
    const proto::AttemptControlRequest* request,
    proto::AttemptControlResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  disagg_pd_service_impl_->cancel_request(request, response);
}

}  // namespace xllm
