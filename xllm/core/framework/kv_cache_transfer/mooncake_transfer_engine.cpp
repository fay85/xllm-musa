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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_set>

#include "absl/strings/str_join.h"
#include "framework/kv_cache_transfer/staged_read_plan.h"
#include "util/env_var.h"
#include "util/timer.h"
#if defined(USE_MUSA)
#include <chrono>
#include <thread>

#include "core/framework/kv_cache_transfer/mooncake_open_session_retry.h"
#include "core/framework/kv_cache_transfer/mooncake_segment_reopen.h"
#endif

#include "common/metrics.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "util/net.h"

namespace xllm {

namespace {

Status check_active_plan_requirement(
    const std::optional<ReshardPlanTemplate>& plan,
    CachePeerMode mode,
    bool require_outgoing_plan) {
  if (mode == CachePeerMode::ACTIVE && require_outgoing_plan &&
      (!plan.has_value() || plan->regions.empty())) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "active cache peer requires a non-empty plan");
  }
  return Status();
}

// Whether `local` links `peer` as a pure KV receiver: the local worker is
// cp1 while the peer runs context parallelism. link_sessions runs on the
// decode side and its peers are prefill workers, so this shape is a decode
// worker whose prefill runs CP -- KV bytes only flow peer->local and the
// reverse outgoing plan is empty by construction.
bool links_cp_peer_as_receiver(const ParallelCoordinates& local,
                               const ParallelCoordinates& peer) {
  return local.cp_size == 1 && peer.cp_size > 1;
}

bool close_remote_session(MooncakeTransferEngineCore* core,
                          uint64_t cluster_id) {
  proto::MooncakeTransferEngineService_Stub* stub =
      core->get_or_create_stub(cluster_id);
  if (stub == nullptr) {
    LOG(ERROR) << "create_rpc_channel failed for cluster_id=" << cluster_id;
    return false;
  }

  proto::SessionInfo session_info;
  session_info.set_addr(core->addr());
  proto::Status response;
  brpc::Controller cntl;
  stub->CloseSession(&cntl, &session_info, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "CloseSession failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool set_remote_cache_peer(MooncakeTransferEngineCore* core,
                           uint64_t cluster_id,
                           const WorkerCacheLayoutManifest& manifest,
                           proto::CachePeerMode mode,
                           const std::string& endpoint) {
  proto::MooncakeTransferEngineService_Stub* stub =
      core->get_or_create_stub(cluster_id);
  if (stub == nullptr) {
    LOG(ERROR) << "Create cache peer RPC channel failed for " << endpoint;
    return false;
  }

  proto::CachePeerRequest request;
  cache_layout_to_proto(manifest, request.mutable_destination_manifest());
  request.set_mode(mode);
  proto::Status response;
  brpc::Controller controller;
  stub->SetCachePeer(&controller, &request, &response, nullptr);
  if (controller.Failed() || !response.ok()) {
    LOG(ERROR) << "SetCachePeer failed for " << endpoint
               << ", rpc_error=" << controller.ErrorText()
               << ", peer_ok=" << response.ok();
    return false;
  }
  return true;
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

bool wait_batch(TransferEngine* engine, BatchID batch_id) {
  while (true) {
    TransferStatus status;
    mooncake::Status result = engine->getBatchTransferStatus(batch_id, status);
    if (!result.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      return false;
    }
    if (status.s == TransferStatusEnum::COMPLETED) {
      return true;
    }
    if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      return false;
    }
    if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      return false;
    }
  }
}

}  // namespace

// ============================================================================
// MooncakeTransferEngineCore (Singleton)
// ============================================================================

MooncakeTransferEngineCore::~MooncakeTransferEngineCore() {
  for (auto& pair : stub_map_) {
    if (pair.second != nullptr) {
      delete pair.second->channel();
      delete pair.second;
    }
  }
  stub_map_.clear();

  if (initialized_) {
    server_.Stop(0);
    server_.Join();
  }
}

bool MooncakeTransferEngineCore::initialize(uint16_t listen_port,
                                            const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    LOG(INFO) << "MooncakeTransferEngineCore already initialized, reusing";
    return true;
  }

  listen_port_ = listen_port;
  host_ip_ = net::get_local_ip_addr();

  const char* tcp_protocol = std::getenv("MC_TCP_PROTO");
  if (tcp_protocol != nullptr && std::string(tcp_protocol) == "1") {
    LOG(ERROR) << "MC_TCP_PROTO=1 does not provide the remote visibility "
                  "semantics required by KV cache transfer.";
    return false;
  }

  engine_ = std::make_unique<TransferEngine>(true);

  Device dev(device);
  dev.set_device();
  dev.init_device_context();

  std::string hostname = host_ip_ + ":" + std::to_string(listen_port_);

  if (engine_->init("P2PHANDSHAKE", hostname, "", 0)) {
    LOG(ERROR) << "engine init failed, hostname=" << hostname;
    return false;
  }

  LOG(INFO) << "[Mooncake][PDTransferEngine] transfer engine init success, "
            << "handshake_endpoint=" << hostname;

  service_ = std::make_shared<MooncakeTransferEngineService>();
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return false;
  }

  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server on port " << listen_port_;
    return false;
  }

  rpc_port_ = engine_->getRpcPort();
  addr_ = host_ip_ + ":" + std::to_string(rpc_port_);

  initialized_ = true;
  LOG(INFO) << "[Mooncake][PDTransferEngine] ready, handshake_endpoint="
            << hostname << ", data_endpoint=" << addr_;

  return true;
}

bool MooncakeTransferEngineCore::open_session(const uint64_t cluster_id,
                                              const std::string& remote_addr,
                                              bool increment_existing) {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG(INFO) << "open_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (it != handles_.end()) {
    if (!increment_existing) {
      LOG(INFO) << "Session already exists for " << remote_addr;
      return true;
    }
    // Reuse the existing session until the last caller releases it.
    it->second.ref_count++;
    LOG(INFO) << "Reusing existing session for " << remote_addr
              << ", ref_count=" << it->second.ref_count;
    return true;
  }

#if defined(USE_MUSA)
  // Reaching this branch means no application session remains for this peer.
  // Mooncake closeSegment() does not evict its descriptor or connected QPs;
  // reopening a recycled endpoint must refresh both before any peer refcount
  // increment or publication in handles_. Fail closed if refresh fails.
  if (!detail::prepare_mooncake_segment_reopen(
          *engine_, remote_addr, /*invalidate_endpoints=*/true)) {
    return false;
  }
#endif

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService_Stub* stub =
        get_or_create_stub_locked(cluster_id);
    if (stub == nullptr) {
      LOG(ERROR) << "create_rpc_channel failed";
      return false;
    }

    proto::SessionInfo request;
    request.set_addr(addr_);
#if defined(USE_MUSA)
    int32_t attempts = 0;
    std::string error_text;
    const auto result = detail::run_session_open_rpc(
        [&](int64_t timeout_ms) -> detail::SessionOpenRpcResult {
          ++attempts;
          proto::Status response;
          brpc::Controller cntl;
          cntl.set_timeout_ms(timeout_ms);
          // Only retry an unsent EHOSTDOWN explicitly. Automatic retries could
          // mix in an ambiguous timeout/reset after a peer-side open succeeded.
          cntl.set_max_retry(0);
          stub->OpenSession(&cntl, &request, &response, nullptr);
          error_text =
              cntl.Failed() ? cntl.ErrorText() : response.ShortDebugString();
          return {cntl.Failed(), cntl.ErrorCode(), response.ok()};
        },
        [&](int32_t delay_ms) {
          LOG(WARNING) << "[MUSA_OPEN_SESSION_RETRY] cluster_id=" << cluster_id
                       << ", remote_addr=" << remote_addr
                       << ", attempt=" << attempts << ", delay_ms=" << delay_ms
                       << ", error=" << error_text;
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        },
        []() -> int64_t {
          return std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
              .count();
        });
    if (result.transport_failed || !result.peer_ok) {
      LOG(ERROR) << "OpenSession failed, attempts=" << attempts
                 << ", error_code=" << result.error_code
                 << ", peer_ok=" << result.peer_ok << ", error=" << error_text;
      return false;
    }
    if (attempts > 1) {
      LOG(INFO) << "[MUSA_OPEN_SESSION_RECOVERED] cluster_id=" << cluster_id
                << ", remote_addr=" << remote_addr << ", attempts=" << attempts;
    }
#else
    proto::Status response;
    brpc::Controller cntl;
    stub->OpenSession(&cntl, &request, &response, nullptr);
    if (cntl.Failed() || !response.ok()) {
      LOG(ERROR) << "OpenSession failed, rpc_error=" << cntl.ErrorText()
                 << ", peer_ok=" << response.ok();
      return false;
    }
#endif

    LOG(INFO) << "OpenSession RPC to " << remote_addr
              << ", local_addr=" << addr_;
  }

  return acquire_session_locked(remote_addr);
}

bool MooncakeTransferEngineCore::acquire_session_locked(
    const std::string& remote_addr) {
  auto it = handles_.find(remote_addr);
  if (it != handles_.end()) {
    ++it->second.ref_count;
    return true;
  }
  if (engine_ == nullptr) {
    LOG(ERROR) << "Mooncake engine is not initialized.";
    return false;
  }
  const Transport::SegmentHandle handle = engine_->openSegment(remote_addr);
  if (handle == static_cast<Transport::SegmentHandle>(-1)) {
    LOG(ERROR) << "Fail to connect to " << remote_addr;
    return false;
  }
  handles_[remote_addr] = SessionInfo{handle, 1};
  return true;
}

void MooncakeTransferEngineCore::release_session_locked(
    const std::string& remote_addr) {
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return;
  }
  --it->second.ref_count;
  if (it->second.ref_count > 0) {
    return;
  }
  if (it->second.handle != static_cast<SegmentHandle>(-1)) {
    engine_->closeSegment(it->second.handle);
  }
  handles_.erase(it);
}

bool MooncakeTransferEngineCore::close_session(const uint64_t cluster_id,
                                               const std::string& remote_addr) {
  std::unique_lock<std::mutex> lock(mutex_);

  LOG(INFO) << "close_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (cluster_id != 0) {
    if (it != handles_.end()) {
      it->second.ref_count--;
      LOG(INFO) << "Decremented ref_count for " << remote_addr
                << ", ref_count=" << it->second.ref_count;
      if (it->second.ref_count > 0) {
        return true;
      }
      Transport::SegmentHandle handle = it->second.handle;
      if (handle != static_cast<Transport::SegmentHandle>(-1)) {
        engine_->closeSegment(handle);
      }
      handles_.erase(it);
    }
    // close_remote_session() obtains the core mutex through
    // get_or_create_stub(). Release it after updating local state to avoid
    // recursively locking the same non-recursive mutex.
    lock.unlock();
    return close_remote_session(this, cluster_id);
  }

  if (it == handles_.end()) {
    return true;
  }

  it->second.ref_count--;
  LOG(INFO) << "Decremented ref_count for " << remote_addr
            << ", ref_count=" << it->second.ref_count;

  if (it->second.ref_count > 0) {
    return true;
  }

  SegmentHandle handle = it->second.handle;
  if (handle != static_cast<SegmentHandle>(-1)) {
    engine_->closeSegment(handle);
  }
  handles_.erase(it);

  LOG(INFO) << "Closed session for " << remote_addr;

  return true;
}

SegmentHandle MooncakeTransferEngineCore::get_handle(
    const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return static_cast<SegmentHandle>(-1);
  }
  return it->second.handle;
}

Status MooncakeTransferEngineCore::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  const Status status = validate_worker_cache_layout(manifest);
  if (!status.ok()) {
    return status;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!cache_peer_links_.empty()) {
    return Status(StatusCode::UNAVAILABLE,
                  "cache layout cannot change while cache peers are linked");
  }
  if (local_cache_layout_.has_value() &&
      local_cache_layout_->incarnation_id == manifest.incarnation_id &&
      manifest.layout_generation <= local_cache_layout_->layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "cache layout generation must increase monotonically");
  }
  local_cache_layout_ = manifest;
  return Status();
}

std::optional<WorkerCacheLayoutManifest>
MooncakeTransferEngineCore::local_cache_layout() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_cache_layout_;
}

Status MooncakeTransferEngineCore::set_cache_peer(
    const WorkerCacheLayoutManifest& peer_manifest,
    CachePeerMode mode,
    bool require_outgoing_plan) {
  const Status manifest_status = validate_worker_cache_layout(peer_manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }

  std::optional<WorkerCacheLayoutManifest> local_manifest;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = cache_peer_links_.find(peer_manifest.addr);
    const bool identity_matches =
        existing != cache_peer_links_.end() &&
        existing->second.destination_incarnation ==
            peer_manifest.incarnation_id &&
        existing->second.destination_layout_generation ==
            peer_manifest.layout_generation;
    if (mode == CachePeerMode::ABSENT) {
      if (!identity_matches) {
        return Status();
      }
      if (existing->second.holds_session) {
        release_session_locked(peer_manifest.addr);
      }
      cache_peer_links_.erase(existing);
      return Status();
    }
    if (identity_matches) {
      if (existing->second.mode == mode) {
        return check_active_plan_requirement(
            existing->second.plan, mode, require_outgoing_plan);
      }
      return Status(StatusCode::INVALID_ARGUMENT,
                    "cache peer mode change requires unlink");
    }
    if (existing != cache_peer_links_.end() &&
        existing->second.destination_incarnation ==
            peer_manifest.incarnation_id &&
        peer_manifest.layout_generation <
            existing->second.destination_layout_generation) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "peer cache layout generation is stale");
    }
    local_manifest = local_cache_layout_;
  }
  if (!local_manifest.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout is not registered");
  }

  std::optional<ReshardPlanTemplate> plan;
  if (mode == CachePeerMode::ACTIVE) {
    ReshardPlanTemplate active_plan;
    if (links_cp_peer_as_receiver(local_manifest->coordinates,
                                  peer_manifest.coordinates)) {
      // This branch bypasses build_outgoing_plan, so the instance-identity
      // half of validate_compatibility must run here: a cp1 decode worker
      // may only receive from a CP prefill peer of the same instance
      // layout. Only the partition-pair direction rule stays relaxed.
      std::vector<std::string> mismatched_fields;
      if (!same_instance_layout(
              *local_manifest, peer_manifest, &mismatched_fields)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "cp1 receiver cache peer is layout-incompatible: "
                      "mismatched " +
                          absl::StrJoin(mismatched_fields, ", "));
      }
      // The reverse direction never carries bytes, so keep an empty plan
      // instead of extending the planner: supports_partition_layout only
      // collapses a CP source into a cp1 destination, and fabricating
      // reverse regions would aim real bytes at a prefill. Endpoint
      // identity lives in the CachePeerLink fields, not in this plan.
    } else {
      const Status plan_status =
          ReshardPlanner().build_outgoing_plan(*local_manifest,
                                               peer_manifest,
                                               &active_plan,
                                               /*include_replicas=*/true,
                                               declared_cp_index_write_mode());
      if (!plan_status.ok()) {
        return plan_status;
      }
    }
    plan = std::move(active_plan);
  }
  const Status requirement_status =
      check_active_plan_requirement(plan, mode, require_outgoing_plan);
  if (!requirement_status.ok()) {
    return requirement_status;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_cache_layout_.has_value() ||
      local_cache_layout_->incarnation_id != local_manifest->incarnation_id ||
      local_cache_layout_->layout_generation !=
          local_manifest->layout_generation) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout changed during plan construction");
  }

  const auto existing = cache_peer_links_.find(peer_manifest.addr);
  if (existing != cache_peer_links_.end() &&
      existing->second.destination_incarnation ==
          peer_manifest.incarnation_id &&
      existing->second.destination_layout_generation ==
          peer_manifest.layout_generation) {
    if (existing->second.mode == mode) {
      return check_active_plan_requirement(
          existing->second.plan, mode, require_outgoing_plan);
    }
    return Status(StatusCode::INVALID_ARGUMENT,
                  "cache peer mode change requires unlink");
  }
  if (existing != cache_peer_links_.end() &&
      existing->second.destination_incarnation ==
          peer_manifest.incarnation_id &&
      peer_manifest.layout_generation <
          existing->second.destination_layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "peer cache layout generation is stale");
  }

  const bool holds_session = mode == CachePeerMode::ACTIVE;
  if (holds_session && !acquire_session_locked(peer_manifest.addr)) {
    return Status(StatusCode::UNAVAILABLE,
                  "failed to open cache peer data session");
  }

  CachePeerLink link;
  link.destination_incarnation = peer_manifest.incarnation_id;
  link.destination_layout_generation = peer_manifest.layout_generation;
  link.mode = mode;
  link.plan = std::move(plan);
  link.holds_session = holds_session;
  if (existing != cache_peer_links_.end() && existing->second.holds_session) {
    release_session_locked(peer_manifest.addr);
  }
  cache_peer_links_[peer_manifest.addr] = std::move(link);
  return Status();
}

bool MooncakeTransferEngineCore::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto link_it = cache_peer_links_.find(remote_addr);
  if (link_it == cache_peer_links_.end() || !link_it->second.plan.has_value()) {
    return false;
  }
  return std::any_of(link_it->second.plan->regions.begin(),
                     link_it->second.plan->regions.end(),
                     [cache_namespace](const StridedRegionTemplate& region) {
                       return region.cache_namespace == cache_namespace;
                     });
}

bool MooncakeTransferEngineCore::has_reshard_plan(
    const std::string& remote_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_peer_links_.find(remote_addr) != cache_peer_links_.end();
}

bool MooncakeTransferEngineCore::is_cache_peer_current(
    const WorkerCacheLayoutManifest& local_manifest,
    const WorkerCacheLayoutManifest& peer_manifest,
    CachePeerMode mode) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto peer = cache_peer_links_.find(peer_manifest.addr);
  return local_cache_layout_.has_value() &&
         local_cache_layout_->incarnation_id == local_manifest.incarnation_id &&
         local_cache_layout_->layout_generation ==
             local_manifest.layout_generation &&
         peer != cache_peer_links_.end() && peer->second.mode == mode &&
         peer->second.destination_incarnation == peer_manifest.incarnation_id &&
         peer->second.destination_layout_generation ==
             peer_manifest.layout_generation;
}

Status MooncakeTransferEngineCore::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto link_it = cache_peer_links_.find(remote_addr);
    if (link_it == cache_peer_links_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "cache peer not negotiated: " + remote_addr);
    }
    if (link_it->second.mode == CachePeerMode::PLAN_ONLY) {
      regions->clear();
      return Status();
    }
    plan = link_it->second.plan;
  }
  if (!plan.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "active cache peer has no reshard plan");
  }
  RequestRegionBinder binder;
  return binder.bind(*plan, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeTransferEngineCore::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto link_it = cache_peer_links_.find(remote_addr);
    if (link_it == cache_peer_links_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "cache peer not negotiated: " + remote_addr);
    }
    if (link_it->second.mode == CachePeerMode::PLAN_ONLY) {
      regions->clear();
      return Status();
    }
    plan = link_it->second.plan;
  }
  if (!plan.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "active cache peer has no reshard plan");
  }
  RequestRegionBinder binder;
  return binder.bind_explicit(
      *plan, mappings, cache_namespace, layer_id, regions);
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub(uint64_t cluster_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return get_or_create_stub_locked(cluster_id);
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub_locked(uint64_t cluster_id) {
  auto it = stub_map_.find(cluster_id);
  if (it == stub_map_.end()) {
    auto [remote_ip, remote_port] = net::convert_uint64_to_ip_port(cluster_id);
    std::string remote_addr = remote_ip + ":" + std::to_string(remote_port);

    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = -1;
    std::string load_balancer = "";
    if (channel->Init(remote_addr.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << remote_addr;
      delete channel;
      return nullptr;
    }

    proto::MooncakeTransferEngineService_Stub* stub =
        new proto::MooncakeTransferEngineService_Stub(channel);
    stub_map_[cluster_id] = stub;
    return stub;
  }

  return it->second;
}

// ============================================================================
// MooncakeTransferEngine
// ============================================================================

MooncakeTransferEngine::MooncakeTransferEngine(const uint16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port),
      device_(device),
      core_(MooncakeTransferEngineCore::get_instance()) {}

std::string MooncakeTransferEngine::initialize() {
  if (!core_.initialize(listen_port_, device_)) {
    LOG(ERROR) << "Failed to initialize MooncakeTransferEngineCore";
    return "";
  }
  return core_.addr();
}

bool MooncakeTransferEngine::register_memory(std::vector<void*> addrs,
                                             std::vector<size_t> lens,
                                             std::vector<uint64_t> buf_bytes) {
  if (addrs.size() != lens.size() || addrs.size() != buf_bytes.size()) {
    LOG(ERROR) << "register_memory input size mismatch, addrs=" << addrs.size()
               << ", lens=" << lens.size()
               << ", buf_bytes=" << buf_bytes.size();
    return false;
  }

  TransferEngine* engine = core_.engine();
  for (size_t i = 0; i < addrs.size(); ++i) {
    int32_t ret = engine->registerLocalMemory(
        addrs[i], lens[i], kWildcardLocation, true, true);
    if (ret != 0) {
      LOG(ERROR) << "registerLocalMemory failed, buf_id=" << i
                 << ", addr=" << addrs[i] << ", len=" << lens[i]
                 << ", ret=" << ret;
      return false;
    }
  }

  buf_bytes_.insert(buf_bytes_.end(), buf_bytes.begin(), buf_bytes.end());
  LOG(INFO) << "register_memory success, total_buf_num=" << buf_bytes_.size();

  return true;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngine::create_rpc_channel(uint64_t cluster_id) {
  return core_.get_or_create_stub(cluster_id);
}

bool MooncakeTransferEngine::fetch_cache_layout(
    uint64_t cluster_id,
    const std::string& remote_addr,
    WorkerCacheLayoutManifest* manifest) {
  proto::MooncakeTransferEngineService_Stub* stub =
      create_rpc_channel(cluster_id);
  if (stub == nullptr) {
    LOG(ERROR) << "Failed to create cache layout RPC channel, cluster_id="
               << cluster_id;
    return false;
  }
  proto::Empty request;
  proto::WorkerCacheLayoutManifest response;
  brpc::Controller controller;
  stub->GetCacheLayoutManifest(&controller, &request, &response, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "GetCacheLayoutManifest failed: " << controller.ErrorText();
    return false;
  }
  const Status status = cache_layout_from_proto(response, manifest);
  if (!status.ok()) {
    LOG(ERROR) << "Invalid remote cache layout from " << remote_addr << ": "
               << status.message();
    return false;
  }
  if (manifest->addr != remote_addr || manifest->cluster_id != cluster_id) {
    LOG(ERROR) << "Remote cache layout endpoint mismatch, expected="
               << cluster_id << "/" << remote_addr
               << ", manifest=" << manifest->cluster_id << "/"
               << manifest->addr;
    return false;
  }
  return true;
}

bool MooncakeTransferEngine::set_remote_peer(
    uint64_t cluster_id,
    const std::string& remote_addr,
    const WorkerCacheLayoutManifest& manifest,
    CachePeerMode mode) {
  proto::CachePeerMode proto_mode = proto::CACHE_PEER_MODE_ABSENT;
  if (mode == CachePeerMode::ACTIVE) {
    proto_mode = proto::CACHE_PEER_MODE_ACTIVE;
  } else if (mode == CachePeerMode::PLAN_ONLY) {
    proto_mode = proto::CACHE_PEER_MODE_PLAN_ONLY;
  }
  return set_remote_cache_peer(
      &core_, cluster_id, manifest, proto_mode, remote_addr);
}

bool MooncakeTransferEngine::set_local_peer(
    const WorkerCacheLayoutManifest& manifest,
    CachePeerMode mode) {
  // ACTIVE is selected by the incoming plan. A replicated KV receiver may
  // have no outgoing regions, but still needs a data session for reads.
  const Status status =
      core_.set_cache_peer(manifest, mode, /*require_outgoing_plan=*/false);
  if (!status.ok()) {
    // link_sessions only sees this bool and its rollback loop is silent,
    // so the rejection reason must be logged here to be diagnosable.
    LOG(ERROR) << "Set local cache peer failed for " << manifest.addr << ": "
               << status.message();
    return false;
  }
  return true;
}

bool MooncakeTransferEngine::is_cache_peer_current(
    const WorkerCacheLayoutManifest& local_manifest,
    const WorkerCacheLayoutManifest& peer_manifest,
    CachePeerMode mode) const {
  return core_.is_cache_peer_current(local_manifest, peer_manifest, mode);
}

bool MooncakeTransferEngine::open_session(const uint64_t cluster_id,
                                          const std::string& remote_addr) {
  if (!core_.open_session(cluster_id, remote_addr)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  ++session_ref_counts_[remote_addr];
  return true;
}

bool MooncakeTransferEngine::close_session(const uint64_t cluster_id,
                                           const std::string& remote_addr) {
  std::optional<LocalCachePeer> cache_peer;
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const auto peer_it = cache_peers_.find(remote_addr);
    if (peer_it != cache_peers_.end()) {
      cache_peer = peer_it->second;
    }
  }
  if (cache_peer.has_value()) {
    if (!set_remote_peer(cluster_id,
                         remote_addr,
                         cache_peer->local_manifest,
                         CachePeerMode::ABSENT)) {
      return false;
    }
    if (!set_local_peer(cache_peer->remote_manifest, CachePeerMode::ABSENT)) {
      LOG(ERROR) << "Local cache peer cleanup failed for " << remote_addr;
      return false;
    }
    std::lock_guard<std::mutex> lock(session_mutex_);
    cache_peers_.erase(remote_addr);
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const auto it = session_ref_counts_.find(remote_addr);
    if (it == session_ref_counts_.end()) {
      return true;
    }
    if (--it->second == 0) {
      session_ref_counts_.erase(it);
    }
  }
  if (core_.close_session(cluster_id, remote_addr)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  ++session_ref_counts_[remote_addr];
  return false;
}

bool MooncakeTransferEngine::link_sessions(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs) {
  if (cluster_ids.size() != remote_addrs.size() || cluster_ids.empty()) {
    LOG(ERROR) << "MoonCake link session endpoint sizes are invalid.";
    return false;
  }

  const std::optional<WorkerCacheLayoutManifest> local_manifest =
      core_.local_cache_layout();
  if (!local_manifest.has_value()) {
    LOG(ERROR) << "Local cache layout must be registered before linking.";
    return false;
  }

  std::vector<WorkerCacheLayoutManifest> remote_manifests;
  remote_manifests.reserve(cluster_ids.size());
  for (size_t index = 0; index < cluster_ids.size(); ++index) {
    WorkerCacheLayoutManifest remote_manifest;
    if (!fetch_cache_layout(
            cluster_ids[index], remote_addrs[index], &remote_manifest)) {
      return false;
    }
    remote_manifests.emplace_back(std::move(remote_manifest));
  }

  ReshardPlanner planner;
  std::vector<size_t> selected_indices;
#if defined(USE_MUSA)
  std::vector<ReshardPlanTemplate> selected_plans;
  const Status selection = planner.build_incoming_plans(
      remote_manifests, *local_manifest, &selected_indices, &selected_plans);
#else
  const Status selection = planner.select_sources(
      remote_manifests, *local_manifest, &selected_indices);
#endif
  if (!selection.ok()) {
    LOG(ERROR) << "Remote cache layouts cannot cover local destination: "
               << selection.message();
    return false;
  }

  const std::unordered_set<size_t> selected_set(selected_indices.begin(),
                                                selected_indices.end());
#if defined(USE_MUSA)
  std::vector<std::shared_ptr<const ReshardPlanTemplate>> incoming_plans(
      remote_manifests.size());
  for (size_t index : selected_indices) {
    incoming_plans[index] =
        std::make_shared<ReshardPlanTemplate>(std::move(selected_plans[index]));
  }
#endif
  bool linked = true;
  for (size_t index = 0; index < remote_manifests.size(); ++index) {
    const bool active = selected_set.find(index) != selected_set.end();
    const CachePeerMode mode =
        active ? CachePeerMode::ACTIVE : CachePeerMode::PLAN_ONLY;
    if (!set_remote_peer(
            cluster_ids[index], remote_addrs[index], *local_manifest, mode)) {
      linked = false;
      break;
    }
    if (!set_local_peer(remote_manifests[index], mode)) {
      linked = false;
      break;
    }
  }

  if (!linked) {
    // A failed RPC response may still have published state on the source.
    for (size_t index = 0; index < remote_manifests.size(); ++index) {
      if (!set_remote_peer(cluster_ids[index],
                           remote_addrs[index],
                           *local_manifest,
                           CachePeerMode::ABSENT)) {
        LOG(ERROR) << "Cache peer rollback failed for " << remote_addrs[index];
      }
      if (!set_local_peer(remote_manifests[index], CachePeerMode::ABSENT)) {
        LOG(ERROR) << "Local cache peer rollback failed for "
                   << remote_addrs[index];
      }
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(session_mutex_);
  for (size_t index = 0; index < remote_manifests.size(); ++index) {
    LocalCachePeer cache_peer;
    cache_peer.local_manifest = *local_manifest;
    cache_peer.remote_manifest = remote_manifests[index];
    cache_peer.mode = selected_set.find(index) != selected_set.end()
                          ? CachePeerMode::ACTIVE
                          : CachePeerMode::PLAN_ONLY;
#if defined(USE_MUSA)
    cache_peer.incoming_plan = std::move(incoming_plans[index]);
#endif
    cache_peers_[remote_addrs[index]] = std::move(cache_peer);
  }
  return true;
}

Status MooncakeTransferEngine::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  return core_.set_local_cache_layout(manifest);
}

bool MooncakeTransferEngine::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  return core_.has_outgoing_plan(remote_addr, cache_namespace);
}

bool MooncakeTransferEngine::has_reshard_plan(
    const std::string& remote_addr) const {
  return core_.has_reshard_plan(remote_addr);
}

Status MooncakeTransferEngine::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return core_.bind_outgoing_regions(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeTransferEngine::bind_incoming_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) {
  if (regions == nullptr) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "incoming KV region output must not be null");
  }
  std::shared_ptr<const ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() ||
        !is_cache_peer_current(peer->second.local_manifest,
                               peer->second.remote_manifest,
                               peer->second.mode)) {
      return Status(StatusCode::UNAVAILABLE,
                    "incoming KV peer is not currently negotiated");
    }
    if (peer->second.mode == CachePeerMode::PLAN_ONLY) {
      regions->clear();
      return Status();
    }
    if (peer->second.mode != CachePeerMode::ACTIVE ||
        !peer->second.incoming_plan) {
      return Status(StatusCode::UNAVAILABLE,
                    "active incoming KV peer has no transfer plan");
    }
    plan = peer->second.incoming_plan;
  }
  return RequestRegionBinder().bind_pull(*plan,
                                         mappings,
                                         cache_namespace,
                                         layer_id,
                                         regions,
                                         /*compact_strided_repeats=*/true);
}

Status MooncakeTransferEngine::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return core_.bind_outgoing_regions_explicit(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

// Merge the source and destination block ids into a single block when both are
// consecutive.
void merge_block_ids(const std::vector<uint64_t>& src_blocks,
                     const std::vector<uint64_t>& dst_blocks,
                     std::vector<uint64_t>& merged_src_blocks,
                     std::vector<uint64_t>& merged_dst_blocks,
                     std::vector<uint64_t>& block_lengths) {
  size_t block_num = src_blocks.size();
  if (block_num == 0) {
    return;
  }

  std::vector<uint64_t> indices(block_num);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(), indices.end(), [&src_blocks](uint64_t i, uint64_t j) {
        return src_blocks[i] < src_blocks[j];
      });

  std::vector<uint64_t> sorted_src_blocks;
  std::vector<uint64_t> sorted_dst_blocks;
  sorted_src_blocks.reserve(block_num);
  sorted_dst_blocks.reserve(block_num);
  for (uint64_t id : indices) {
    sorted_src_blocks.emplace_back(src_blocks[id]);
    sorted_dst_blocks.emplace_back(dst_blocks[id]);
  }

  uint64_t current_src_id = sorted_src_blocks[0];
  uint64_t current_dst_id = sorted_dst_blocks[0];
  uint64_t current_length = 1;
  merged_src_blocks.reserve(block_num);
  merged_dst_blocks.reserve(block_num);
  block_lengths.reserve(block_num);
  for (size_t i = 1; i < sorted_src_blocks.size(); ++i) {
    if (sorted_src_blocks[i] == sorted_src_blocks[i - 1] + 1 &&
        sorted_dst_blocks[i] == sorted_dst_blocks[i - 1] + 1) {
      current_length++;
    } else {
      merged_src_blocks.emplace_back(current_src_id);
      merged_dst_blocks.emplace_back(current_dst_id);
      block_lengths.emplace_back(current_length);
      current_src_id = sorted_src_blocks[i];
      current_dst_id = sorted_dst_blocks[i];
      current_length = 1;
    }
  }
  merged_src_blocks.emplace_back(current_src_id);
  merged_dst_blocks.emplace_back(current_dst_id);
  block_lengths.emplace_back(current_length);
}

bool MooncakeTransferEngine::move_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids,
    MoveOpcode move_opcode) {
  if (src_blocks.size() != dst_blocks.size()) {
    LOG(ERROR) << "src_blocks size must equal dst_blocks size, src="
               << src_blocks.size() << ", dst=" << dst_blocks.size();
    return false;
  }

  std::vector<int64_t> active_buf_ids = buf_ids;
  if (active_buf_ids.empty()) {
    active_buf_ids.resize(buf_bytes_.size());
    std::iota(active_buf_ids.begin(), active_buf_ids.end(), 0);
  }

  std::vector<BufferTransferMapping> mappings;
  mappings.reserve(active_buf_ids.size());
  for (int64_t buf_id : active_buf_ids) {
    BufferTransferMapping mapping;
    mapping.buf_id = buf_id;
    if (move_opcode == MoveOpcode::WRITE) {
      mapping.local_ids = src_blocks;
      mapping.remote_ids = dst_blocks;
    } else {
      mapping.local_ids = dst_blocks;
      mapping.remote_ids = src_blocks;
    }
    mappings.emplace_back(std::move(mapping));
  }
  return move_memory_groups(remote_addr, mappings, move_opcode);
}

bool MooncakeTransferEngine::move_memory_groups(
    const std::string& remote_addr,
    const std::vector<BufferTransferMapping>& mappings,
    MoveOpcode move_opcode) {
  std::vector<ByteRegion> regions;
  for (const BufferTransferMapping& mapping : mappings) {
    if (mapping.local_ids.size() != mapping.remote_ids.size()) {
      LOG(ERROR) << "local_ids size must equal remote_ids size, buf_id="
                 << mapping.buf_id << ", local=" << mapping.local_ids.size()
                 << ", remote=" << mapping.remote_ids.size();
      return false;
    }
    const int64_t buf_id = mapping.buf_id;
    if (buf_id < 0 || static_cast<size_t>(buf_id) >= buf_bytes_.size()) {
      LOG(ERROR) << "buf_id out of range, buf_id=" << buf_id
                 << ", buf_cnt=" << buf_bytes_.size();
      return false;
    }

    const uint64_t block_bytes = buf_bytes_[static_cast<size_t>(buf_id)];
    std::vector<uint64_t> merged_local_ids;
    std::vector<uint64_t> merged_remote_ids;
    std::vector<uint64_t> block_lengths;
    merge_block_ids(mapping.local_ids,
                    mapping.remote_ids,
                    merged_local_ids,
                    merged_remote_ids,
                    block_lengths);
    for (size_t i = 0; i < merged_local_ids.size(); ++i) {
      uint64_t local_block_id = merged_local_ids[i];
      uint64_t remote_block_id = merged_remote_ids[i];
      uint64_t block_length = block_lengths[i];
      if (multiply_overflows(local_block_id, block_bytes) ||
          multiply_overflows(remote_block_id, block_bytes) ||
          multiply_overflows(block_length, block_bytes)) {
        LOG(ERROR) << "MoonCake block wrapper offset overflow, buf_id="
                   << buf_id;
        return false;
      }
      ByteRegion region;
      region.local_buffer_id = static_cast<uint64_t>(buf_id);
      region.local_offset = local_block_id * block_bytes;
      region.remote_buffer_id = static_cast<uint64_t>(buf_id);
      region.remote_offset = remote_block_id * block_bytes;
      region.length = block_length * block_bytes;
      regions.emplace_back(std::move(region));
    }
  }

  return move_memory_regions(remote_addr, regions, move_opcode);
}

bool MooncakeTransferEngine::move_memory_regions(
    const std::string& remote_addr,
    const std::vector<ByteRegion>& regions,
    MoveOpcode move_opcode) {
  if (regions.empty()) {
    return true;
  }

  SegmentHandle remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == static_cast<SegmentHandle>(-1)) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return false;
  }

  TransferEngine* engine = core_.engine();
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc =
      engine->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return false;
  }
  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc =
      engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return false;
  }

  TransferRequest::OpCode opcode = TransferRequest::READ;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  }

  std::vector<ByteRegion> flat_regions;
  const std::vector<ByteRegion>* transfer_regions = &regions;
  for (const ByteRegion& region : regions) {
    if (byte_region_repeats(region) > 1) {
      transfer_regions = &flat_regions;
      break;
    }
  }
  if (transfer_regions == &flat_regions) {
    for (const ByteRegion& region : regions) {
      const uint64_t repeats = byte_region_repeats(region);
      for (uint64_t piece = 0; piece < repeats; ++piece) {
        ByteRegion flat = region;
        flat.local_offset += piece * region.local_stride;
        flat.remote_offset += piece * region.remote_stride;
        flat.repeat_count = 1;
        flat.local_stride = 0;
        flat.remote_stride = 0;
        flat_regions.emplace_back(flat);
      }
    }
  }

  std::vector<TransferRequest> entries;
  entries.reserve(transfer_regions->size());
  uint64_t total_bytes = 0;
  for (const ByteRegion& region : *transfer_regions) {
    if (region.local_buffer_id >= local_segment_desc->buffers.size() ||
        region.remote_buffer_id >= remote_segment_desc->buffers.size()) {
      LOG(ERROR) << "MoonCake region buffer id out of range, local_id="
                 << region.local_buffer_id
                 << ", remote_id=" << region.remote_buffer_id
                 << ", local_count=" << local_segment_desc->buffers.size()
                 << ", remote_count=" << remote_segment_desc->buffers.size();
      return false;
    }
    if (region.length == 0) {
      LOG(ERROR) << "MoonCake region length must be positive.";
      return false;
    }

    const auto& local_buffer =
        local_segment_desc->buffers[region.local_buffer_id];
    const auto& remote_buffer =
        remote_segment_desc->buffers[region.remote_buffer_id];
    if (region.local_offset > local_buffer.length ||
        region.length > local_buffer.length - region.local_offset ||
        region.remote_offset > remote_buffer.length ||
        region.length > remote_buffer.length - region.remote_offset) {
      LOG(ERROR) << "MoonCake byte region is outside registered memory, "
                 << "local_buf=" << region.local_buffer_id
                 << ", local_offset=" << region.local_offset
                 << ", remote_buf=" << region.remote_buffer_id
                 << ", remote_offset=" << region.remote_offset
                 << ", length=" << region.length;
      return false;
    }
    if (region.length > std::numeric_limits<uint64_t>::max() - total_bytes) {
      LOG(ERROR) << "MoonCake transfer byte count overflow";
      return false;
    }
    total_bytes += region.length;

    TransferRequest entry;
    entry.opcode = opcode;
    entry.length = region.length;
    entry.source = reinterpret_cast<void*>(
        reinterpret_cast<char*>(local_buffer.addr) + region.local_offset);
    entry.target_id = remote_handle;
    entry.target_offset = remote_buffer.addr + region.remote_offset;
    entry.advise_retry_cnt = 0;
    entries.emplace_back(std::move(entry));
  }

  constexpr size_t kMaxRegionsPerBatch = 4096;
  Timer transfer_timer;
  for (size_t begin = 0; begin < entries.size(); begin += kMaxRegionsPerBatch) {
    const size_t end = std::min(begin + kMaxRegionsPerBatch, entries.size());
    std::vector<TransferRequest> batch_entries(entries.begin() + begin,
                                               entries.begin() + end);
    const BatchID batch_id = engine->allocateBatchID(batch_entries.size());
    mooncake::Status submit_status =
        engine->submitTransfer(batch_id, batch_entries);
    if (!submit_status.ok()) {
      LOG(ERROR) << "submit byte regions failed";
      COUNTER_INC(mooncake_transfer_failed_total);
      engine->freeBatchID(batch_id);
      return false;
    }
    const bool transfer_success = wait_batch(engine, batch_id);
    const mooncake::Status free_status = engine->freeBatchID(batch_id);
    if (!free_status.ok() || !transfer_success) {
      LOG(ERROR) << "MoonCake byte-region batch failed";
      COUNTER_INC(mooncake_transfer_failed_total);
      return false;
    }
  }

  const int64_t latency_microseconds = static_cast<int64_t>(
      transfer_timer.elapsed_seconds() * static_cast<double>(1000000));
  if (move_opcode == MoveOpcode::READ) {
    COUNTER_INC(mooncake_transfer_completed_total_read);
    COUNTER_ADD(mooncake_transfer_bytes_total_read, total_bytes);
    HISTOGRAM_OBSERVE(mooncake_transfer_latency_microseconds_read,
                      latency_microseconds);
  } else {
    COUNTER_INC(mooncake_transfer_completed_total_write);
    COUNTER_ADD(mooncake_transfer_bytes_total_write, total_bytes);
    HISTOGRAM_OBSERVE(mooncake_transfer_latency_microseconds_write,
                      latency_microseconds);
  }
  return true;
}

namespace {
void append_fitting_pull_regions(std::vector<ByteRegion>* output,
                                 const ByteRegion& region,
                                 uint64_t staging_bytes) {
  const uint64_t repeats = byte_region_repeats(region);
  if (repeats <= 1 || region.length == 0) {
    output->emplace_back(region);
    return;
  }
  const uint64_t span = byte_region_remote_span(region);
  const uint64_t payload = byte_region_payload_bytes(region);
  const bool fits = span <= staging_bytes && span / 2 + span % 2 <= payload &&
                    region.local_stride >= region.length &&
                    region.remote_stride >= region.length;
  if (fits) {
    output->emplace_back(region);
    return;
  }
  ByteRegion left = region;
  ByteRegion right = region;
  left.repeat_count = repeats / 2;
  right.repeat_count = repeats - left.repeat_count;
  right.local_offset += left.repeat_count * region.local_stride;
  right.remote_offset += left.repeat_count * region.remote_stride;
  append_fitting_pull_regions(output, left, staging_bytes);
  append_fitting_pull_regions(output, right, staging_bytes);
}

bool try_extend_compact_pull_region(StagedReadRun& run,
                                    const ByteRegion& region,
                                    size_t end,
                                    uint64_t staging_bytes) {
  const uint64_t piece = region.length;
  const uint64_t span = byte_region_remote_span(region);
  const uint64_t payload = byte_region_payload_bytes(region);
  if (piece == 0 || span == 0 || payload == 0 ||
      run.remote_buffer_id != region.remote_buffer_id) {
    return false;
  }
  if (span / 2 + span % 2 > payload) {
    return false;
  }
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  if (run.remote_offset > kMax - run.length ||
      region.remote_offset > kMax - span ||
      run.payload_bytes > kMax - payload) {
    return false;
  }
  const uint64_t run_end = run.remote_offset + run.length;
  if (region.remote_offset < run_end ||
      region.remote_offset - run_end > piece) {
    return false;
  }
  const uint64_t combined = region.remote_offset + span - run.remote_offset;
  const uint64_t combined_payload = run.payload_bytes + payload;
  if (combined > staging_bytes ||
      combined / 2 + combined % 2 > combined_payload) {
    return false;
  }
  run.length = combined;
  run.payload_bytes = combined_payload;
  run.end = end;
  return true;
}

std::vector<size_t> remote_order(const std::vector<ByteRegion>& regions) {
  std::vector<size_t> order(regions.size());
  std::iota(order.begin(), order.end(), static_cast<size_t>(0));
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const ByteRegion& x = regions[a];
    const ByteRegion& y = regions[b];
    return x.remote_buffer_id != y.remote_buffer_id
               ? x.remote_buffer_id < y.remote_buffer_id
               : x.remote_offset < y.remote_offset;
  });
  return order;
}
}  // namespace

size_t MooncakeTransferEngine::count_remote_runs(
    const std::vector<ByteRegion>& regions) {
  const std::vector<size_t> order = remote_order(regions);
  size_t runs = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    const ByteRegion& cur = regions[order[i]];
    if (i == 0) {
      ++runs;
      continue;
    }
    const ByteRegion& prev = regions[order[i - 1]];
    if (prev.remote_buffer_id != cur.remote_buffer_id ||
        prev.remote_offset + prev.length != cur.remote_offset) {
      ++runs;
    }
  }
  return runs;
}

bool MooncakeTransferEngine::move_memory_regions_staged(
    const std::string& remote_addr,
    const std::vector<ByteRegion>& regions,
    double* read_ms,
    double* scatter_ms) {
  if (regions.empty()) {
    return true;
  }
  const int64_t staging_mib =
      util::get_int_env("XLLM_MUSA_PULL_STAGING_MB", 128);
  CHECK_GE(staging_mib, 16);
  CHECK_LE(staging_mib, 1024);
  const int64_t kStagingBytes = staging_mib << 20;
  std::lock_guard<std::mutex> lock(staging_mutex_);
  TransferEngine* engine = core_.engine();
  device_.set_device();
  const torch::Device torch_device = device_.unwrap();
  if (!staging_.defined()) {
    torch::Tensor staging = torch::empty(
        {kStagingBytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch_device));
    const int32_t ret = engine->registerLocalMemory(
        staging.data_ptr(), kStagingBytes, kWildcardLocation, true, true);
    if (ret != 0) {
      LOG(ERROR) << "registerLocalMemory for PULL staging failed, ret=" << ret;
      return false;
    }
    auto desc = engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
    bool found = false;
    for (size_t i = 0; desc && i < desc->buffers.size(); ++i) {
      if (desc->buffers[i].addr ==
          reinterpret_cast<uint64_t>(staging.data_ptr())) {
        staging_buf_id_ = i;
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(ERROR) << "PULL staging buffer not found in local segment";
      return false;
    }
    staging_ = staging;
    LOG(INFO) << "PULL staging buffer registered, bytes=" << kStagingBytes
              << ", buf_id=" << staging_buf_id_;
  }
  auto local_desc = engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return false;
  }

  using Run = StagedReadRun;
  std::vector<ByteRegion> fitting;
  fitting.reserve(regions.size());
  for (const ByteRegion& region : regions) {
    append_fitting_pull_regions(
        &fitting, region, static_cast<uint64_t>(kStagingBytes));
  }
  const std::vector<size_t> order = remote_order(fitting);
  std::vector<Run> runs;
  runs.reserve(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    const ByteRegion& cur = fitting[order[i]];
    const bool extended =
        !runs.empty() &&
        try_extend_compact_pull_region(
            runs.back(), cur, i + 1, static_cast<uint64_t>(kStagingBytes));
    if (!extended) {
      const uint64_t span = byte_region_remote_span(cur);
      runs.emplace_back(Run{cur.remote_buffer_id,
                            cur.remote_offset,
                            span,
                            byte_region_payload_bytes(cur),
                            i,
                            i + 1});
    }
  }
  uint64_t payload_bytes = 0;
  uint64_t read_bytes = 0;
  for (const Run& run : runs) {
    payload_bytes += run.payload_bytes;
    read_bytes += run.length;
  }
  LOG_FIRST_N(INFO, 5) << "PULL staging plan, regions=" << fitting.size()
                       << ", reads=" << runs.size()
                       << ", payload_bytes=" << payload_bytes
                       << ", read_bytes=" << read_bytes
                       << ", staging_bytes=" << kStagingBytes;

  const auto uint8_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch_device);
  const auto index_options = torch::TensorOptions().dtype(torch::kLong);
  double total_read_ms = 0.0;
  double total_scatter_ms = 0.0;
  size_t next = 0;
  while (next < runs.size()) {
    std::vector<ByteRegion> reads;
    std::vector<uint64_t> run_stage_offset;
    const size_t chunk_begin = next;
    uint64_t used = 0;
    while (next < runs.size() &&
           used + runs[next].length <= static_cast<uint64_t>(kStagingBytes)) {
      ByteRegion read;
      read.local_buffer_id = staging_buf_id_;
      read.local_offset = used;
      read.remote_buffer_id = runs[next].remote_buffer_id;
      read.remote_offset = runs[next].remote_offset;
      read.length = runs[next].length;
      reads.emplace_back(read);
      run_stage_offset.emplace_back(used);
      used += runs[next].length;
      ++next;
    }
    if (next == chunk_begin) {
      LOG(ERROR) << "PULL run of " << runs[next].length
                 << " bytes exceeds staging capacity " << kStagingBytes;
      return false;
    }
    Timer read_timer;
    if (!move_memory_regions(remote_addr, reads, MoveOpcode::READ)) {
      return false;
    }
    total_read_ms += read_timer.elapsed_seconds() * 1000.0;

    Timer scatter_timer;
    // (local buffer, length) -> (staging offsets, local offsets)
    struct StridedScatter {
      uint64_t local_buffer_id = 0;
      int64_t length = 0;
      int64_t stage_row = 0;
      int64_t local_row = 0;
      int64_t count = 0;
      int64_t remote_stride_rows = 0;
      int64_t local_stride_rows = 0;
    };
    std::map<std::pair<uint64_t, uint64_t>,
             std::pair<std::vector<int64_t>, std::vector<int64_t>>>
        groups;
    std::vector<StridedScatter> strided_scatters;
    for (size_t r = chunk_begin; r < next; ++r) {
      const Run& run = runs[r];
      const uint64_t stage_base = run_stage_offset[r - chunk_begin];
      for (size_t k = run.begin; k < run.end; ++k) {
        const ByteRegion& region = fitting[order[k]];
        const uint64_t repeats = byte_region_repeats(region);
        const int64_t length = static_cast<int64_t>(region.length);
        if (repeats > 1 && length > 0 &&
            region.local_stride % region.length == 0 &&
            region.remote_stride % region.length == 0 &&
            region.local_offset % region.length == 0 &&
            (stage_base + region.remote_offset - run.remote_offset) %
                    region.length ==
                0) {
          strided_scatters.push_back(StridedScatter{
              region.local_buffer_id,
              length,
              static_cast<int64_t>(
                  (stage_base + region.remote_offset - run.remote_offset) /
                  region.length),
              static_cast<int64_t>(region.local_offset / region.length),
              static_cast<int64_t>(repeats),
              static_cast<int64_t>(region.remote_stride / region.length),
              static_cast<int64_t>(region.local_stride / region.length)});
          continue;
        }
        auto& group = groups[{region.local_buffer_id, region.length}];
        const uint64_t piece_count = repeats;
        for (uint64_t piece = 0; piece < piece_count; ++piece) {
          group.first.emplace_back(static_cast<int64_t>(
              stage_base + region.remote_offset + piece * region.remote_stride -
              run.remote_offset));
          group.second.emplace_back(static_cast<int64_t>(
              region.local_offset + piece * region.local_stride));
        }
      }
    }
    // Upload every aligned group's row indices with one H2D copy.
    std::vector<int64_t> all_rows;
    std::vector<std::tuple<uint64_t, int64_t, size_t, size_t>> aligned_groups;
    std::vector<const std::pair<std::vector<int64_t>, std::vector<int64_t>>*>
        unaligned_offsets;
    std::vector<std::pair<uint64_t, int64_t>> unaligned_keys;
    for (auto& [key, offsets] : groups) {
      const uint64_t local_buffer_id = key.first;
      const int64_t length = static_cast<int64_t>(key.second);
      if (local_buffer_id >= local_desc->buffers.size()) {
        LOG(ERROR) << "PULL staging scatter local buffer id out of range";
        return false;
      }
      const auto& buffer = local_desc->buffers[local_buffer_id];
      bool aligned = static_cast<int64_t>(buffer.length) % length == 0 &&
                     kStagingBytes % length == 0;
      for (size_t i = 0; aligned && i < offsets.first.size(); ++i) {
        aligned =
            offsets.first[i] % length == 0 && offsets.second[i] % length == 0;
      }
      if (!aligned) {
        unaligned_offsets.emplace_back(&offsets);
        unaligned_keys.emplace_back(local_buffer_id, length);
        continue;
      }
      const size_t base = all_rows.size();
      const size_t n = offsets.first.size();
      all_rows.resize(base + 2 * n);
      for (size_t i = 0; i < n; ++i) {
        all_rows[base + i] = offsets.first[i] / length;
        all_rows[base + n + i] = offsets.second[i] / length;
      }
      aligned_groups.emplace_back(local_buffer_id, length, base, n);
    }
    // NPU/MLU Qwen3.5 decode launches one packed kernel for the whole batch.
    // Fold every strided repeat into that same row-index upload so scatter is
    // one index_copy per buffer, not one launch per head or block.
    std::map<std::pair<uint64_t, int64_t>, std::vector<const StridedScatter*>>
        strided_by_key;
    for (const StridedScatter& scatter : strided_scatters) {
      if (scatter.local_buffer_id >= local_desc->buffers.size() ||
          scatter.count <= 0 || scatter.length <= 0 ||
          scatter.remote_stride_rows < 1 || scatter.local_stride_rows < 1) {
        LOG(ERROR) << "PULL strided scatter is out of range";
        return false;
      }
      strided_by_key[{scatter.local_buffer_id, scatter.length}].push_back(
          &scatter);
    }
    for (auto& [key, scatters] : strided_by_key) {
      size_t n = 0;
      for (const StridedScatter* scatter : scatters) {
        n += static_cast<size_t>(scatter->count);
      }
      const size_t base = all_rows.size();
      all_rows.resize(base + 2 * n);
      size_t cursor = 0;
      for (const StridedScatter* scatter : scatters) {
        for (int64_t piece = 0; piece < scatter->count; ++piece) {
          all_rows[base + cursor] =
              scatter->stage_row + piece * scatter->remote_stride_rows;
          all_rows[base + n + cursor] =
              scatter->local_row + piece * scatter->local_stride_rows;
          ++cursor;
        }
      }
      aligned_groups.emplace_back(key.first, key.second, base, n);
    }
    torch::Tensor device_rows;
    if (!all_rows.empty()) {
      device_rows = torch::from_blob(all_rows.data(),
                                     {static_cast<int64_t>(all_rows.size())},
                                     index_options)
                        .to(torch_device);
    }
    for (const auto& [local_buffer_id, length, base, n] : aligned_groups) {
      const auto& buffer = local_desc->buffers[local_buffer_id];
      torch::Tensor dst = torch::from_blob(
          reinterpret_cast<void*>(buffer.addr),
          {static_cast<int64_t>(buffer.length) / length, length},
          uint8_options);
      torch::Tensor src_index = device_rows.narrow(
          0, static_cast<int64_t>(base), static_cast<int64_t>(n));
      torch::Tensor dst_index = device_rows.narrow(
          0, static_cast<int64_t>(base + n), static_cast<int64_t>(n));
      dst.index_copy_(
          0, dst_index, staging_.view({-1, length}).index_select(0, src_index));
    }
    for (size_t g = 0; g < unaligned_offsets.size(); ++g) {
      const auto& buffer = local_desc->buffers[unaligned_keys[g].first];
      const int64_t length = unaligned_keys[g].second;
      torch::Tensor dst_flat =
          torch::from_blob(reinterpret_cast<void*>(buffer.addr),
                           {static_cast<int64_t>(buffer.length)},
                           uint8_options);
      const auto& offsets = *unaligned_offsets[g];
      for (size_t i = 0; i < offsets.first.size(); ++i) {
        dst_flat.narrow(0, offsets.second[i], length)
            .copy_(staging_.narrow(0, offsets.first[i], length));
      }
    }
    device_.current_stream()->synchronize();
    total_scatter_ms += scatter_timer.elapsed_seconds() * 1000.0;
  }
  if (read_ms != nullptr) {
    *read_ms = total_read_ms;
  }
  if (scatter_ms != nullptr) {
    *scatter_ms = total_scatter_ms;
  }
  return true;
}

bool MooncakeTransferEngine::move_memory_by_global_offsets(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_offsets,
    const std::vector<uint64_t>& dst_offsets,
    size_t transfer_size,
    MoveOpcode move_opcode) {
  if (src_offsets.size() != dst_offsets.size() || transfer_size == 0) {
    LOG(ERROR) << "Invalid XTensor byte-region mapping, source="
               << src_offsets.size() << ", destination=" << dst_offsets.size()
               << ", length=" << transfer_size;
    return false;
  }
  std::vector<ByteRegion> regions;
  regions.reserve(src_offsets.size());
  for (size_t index = 0; index < src_offsets.size(); ++index) {
    ByteRegion region;
    region.local_buffer_id = 0;
    region.remote_buffer_id = 0;
    region.length = static_cast<uint64_t>(transfer_size);
    if (move_opcode == MoveOpcode::WRITE) {
      region.local_offset = src_offsets[index];
      region.remote_offset = dst_offsets[index];
    } else {
      region.local_offset = dst_offsets[index];
      region.remote_offset = src_offsets[index];
    }
    regions.emplace_back(std::move(region));
  }
  return move_memory_regions(remote_addr, regions, move_opcode);
}

bool MooncakeTransferEngine::pull_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::READ);
  if (!ret) {
    LOG(ERROR) << "Pull memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::push_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::WRITE);
  if (!ret) {
    LOG(ERROR) << "Push memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

// ============================================================================
// MooncakeTransferEngineService
// ============================================================================

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "OpenSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  MooncakeTransferEngineCore& core = MooncakeTransferEngineCore::get_instance();
  const bool result = core.open_session(
      /*cluster_id=*/0, remote_addr, /*increment_existing=*/true);

  response->set_ok(result);
}

void MooncakeTransferEngineService::SetCachePeer(
    ::google::protobuf::RpcController* controller,
    const proto::CachePeerRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }
  if (!request->has_destination_manifest() ||
      request->mode() == proto::CACHE_PEER_MODE_UNSPECIFIED) {
    LOG(ERROR) << "SetCachePeer request is incomplete.";
    response->set_ok(false);
    return;
  }

  WorkerCacheLayoutManifest peer_manifest;
  const Status decode_status =
      cache_layout_from_proto(request->destination_manifest(), &peer_manifest);
  if (!decode_status.ok()) {
    LOG(ERROR) << "SetCachePeer received an invalid cache layout: "
               << decode_status.message();
    response->set_ok(false);
    return;
  }

  CachePeerMode mode = CachePeerMode::ABSENT;
  if (request->mode() == proto::CACHE_PEER_MODE_ACTIVE) {
    mode = CachePeerMode::ACTIVE;
  } else if (request->mode() == proto::CACHE_PEER_MODE_PLAN_ONLY) {
    mode = CachePeerMode::PLAN_ONLY;
  } else if (request->mode() != proto::CACHE_PEER_MODE_ABSENT) {
    LOG(ERROR) << "SetCachePeer received an invalid mode.";
    response->set_ok(false);
    return;
  }

  const Status status =
      MooncakeTransferEngineCore::get_instance().set_cache_peer(peer_manifest,
                                                                mode);
  if (!status.ok()) {
    LOG(ERROR) << "SetCachePeer failed: " << status.message();
  }
  response->set_ok(status.ok());
}

void MooncakeTransferEngineService::GetCacheLayoutManifest(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::WorkerCacheLayoutManifest* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }
  const std::optional<WorkerCacheLayoutManifest> manifest =
      MooncakeTransferEngineCore::get_instance().local_cache_layout();
  if (!manifest.has_value()) {
    LOG(ERROR) << "GetCacheLayoutManifest called before cache registration.";
    response->Clear();
    return;
  }
  cache_layout_to_proto(*manifest, response);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "CloseSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().close_session(0, remote_addr);

  response->set_ok(result);
}

}  // namespace xllm
