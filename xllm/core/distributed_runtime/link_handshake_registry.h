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

#pragma once

#include <bthread/condition_variable.h>
#include <bthread/mutex.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "disagg_pd.pb.h"

namespace xllm {

// Owns the service-facing handshake ledger, not the transport's data sessions.
// Metadata is protected by mutex_; no callback or physical RPC runs under it.
// Waiters use a bthread condition variable so a slow handshake cannot park all
// brpc workers on a pthread mutex. Drain invalidates publication, not GPU work
// or transfers already owned by a request.
class LinkHandshakeRegistry final {
 public:
  struct Stats {
    uint64_t physical_attempts = 0;
    uint64_t joined_attempts = 0;
    uint64_t ready_replays = 0;
    uint64_t fenced_requests = 0;
    uint64_t superseded_attempts = 0;
    size_t peer_records = 0;
  };

  // Retain bounded drain tombstones for the process lifetime. Forgetting one
  // would allow a delayed EnsureLink from that authority to resurrect READY.
  explicit LinkHandshakeRegistry(size_t max_peer_records = 4096)
      : max_peer_records_(max_peer_records) {}

  proto::EnsureLinkResult ensure(const proto::EnsureLinkSpec& spec,
                                 const proto::InstanceClusterInfo& cluster,
                                 const std::function<bool()>& physical_link) {
    std::unique_lock<bthread::Mutex> lock(mutex_);
    const std::shared_ptr<Peer> peer = find_or_create_peer(spec);
    if (peer == nullptr) {
      return result(proto::ENSURE_LINK_DISPOSITION_DEGREE_LIMITED,
                    "handshake-peer-record-limit");
    }
    const proto::EnsureLinkResult admission = authorize(*peer, spec, cluster);
    if (admission.disposition() != proto::ENSURE_LINK_DISPOSITION_UNSPECIFIED) {
      return admission;
    }
    // A changed physical contract waits for the previous side effect to end.
    // The old owner is fenced from publishing READY while this caller waits.
    while (peer->attempt != nullptr &&
           (peer->attempt->invalidated ||
            !same_contract(
                peer->attempt->spec, peer->attempt->cluster, spec, cluster))) {
      const std::shared_ptr<Attempt> prior = peer->attempt;
      prior->invalidated = true;
      while (!prior->done) {
        prior->changed.wait(lock);
      }
      const proto::EnsureLinkResult current = check_authority(*peer, spec);
      if (current.disposition() != proto::ENSURE_LINK_DISPOSITION_UNSPECIFIED) {
        return current;
      }
    }
    if (peer->ready) {
      ++stats_.ready_replays;
      return ready_result(peer->link_generation, /*replayed=*/true);
    }
    if (peer->attempt != nullptr) {
      const std::shared_ptr<Attempt> shared = peer->attempt;
      ++stats_.joined_attempts;
      while (!shared->done) {
        shared->changed.wait(lock);
      }
      const proto::EnsureLinkResult current = check_authority(*peer, spec);
      if (current.disposition() != proto::ENSURE_LINK_DISPOSITION_UNSPECIFIED) {
        return current;
      }
      proto::EnsureLinkResult replay = shared->outcome;
      replay.set_replayed(true);
      return replay;
    }
    const std::shared_ptr<Attempt> attempt = std::make_shared<Attempt>();
    // RPC messages can be destroyed after a caller times out. The shared
    // operation retains its exact contract independently of every waiter.
    attempt->spec = spec;
    attempt->cluster = cluster;
    peer->attempt = attempt;
    ++stats_.physical_attempts;
    lock.unlock();
    const bool linked = physical_link();
    lock.lock();
    complete(*peer, *attempt, linked);
    peer->attempt.reset();
    attempt->done = true;
    attempt->changed.notify_all();
    const proto::EnsureLinkResult current = check_authority(*peer, spec);
    return current.disposition() == proto::ENSURE_LINK_DISPOSITION_UNSPECIFIED
               ? attempt->outcome
               : current;
  }

  proto::EnsureLinkResult drain(const proto::EnsureLinkSpec& spec) {
    std::lock_guard<bthread::Mutex> lock(mutex_);
    const std::shared_ptr<Peer> peer = find_or_create_peer(spec);
    if (peer == nullptr) {
      return result(proto::ENSURE_LINK_DISPOSITION_DEGREE_LIMITED,
                    "handshake-peer-record-limit");
    }
    if (authority(spec) < peer->authority) {
      ++stats_.fenced_requests;
      return result(proto::ENSURE_LINK_DISPOSITION_FENCED,
                    "stale-handshake-authority");
    }
    if (!peer->authorized_spec.prefill().engine_uid().empty() &&
        (!same_identity(peer->authorized_spec.prefill(), spec.prefill()) ||
         !same_identity(peer->authorized_spec.decode(), spec.decode()))) {
      return result(proto::ENSURE_LINK_DISPOSITION_FENCED,
                    "drain-peer-identity-mismatch");
    }
    peer->authority = authority(spec);
    peer->drained = true;
    peer->ready = false;
    if (peer->attempt != nullptr) {
      peer->attempt->invalidated = true;
    }
    proto::EnsureLinkResult response = ready_result(
        std::max(uint64_t{1}, peer->link_generation), /*replayed=*/false);
    response.set_reason("drained");
    return response;
  }

  Stats stats() const {
    std::lock_guard<bthread::Mutex> lock(mutex_);
    Stats snapshot = stats_;
    snapshot.peer_records = peers_.size();
    return snapshot;
  }

 private:
  using Authority = std::tuple<uint64_t, uint64_t>;

  struct Attempt {
    proto::EnsureLinkSpec spec;
    proto::InstanceClusterInfo cluster;
    bthread::ConditionVariable changed;
    proto::EnsureLinkResult outcome;
    bool invalidated = false;
    bool done = false;
  };

  struct Peer {
    Authority authority{0, 0};
    proto::EnsureLinkSpec authorized_spec;
    proto::InstanceClusterInfo authorized_cluster;
    std::shared_ptr<Attempt> attempt;
    uint64_t link_generation = 0;
    bool ready = false;
    bool drained = false;
  };

  static Authority authority(const proto::EnsureLinkSpec& spec) {
    return {spec.coordinator_epoch(), spec.adjacency_generation()};
  }

  static proto::EnsureLinkResult result(
      proto::EnsureLinkDisposition disposition,
      const std::string& reason) {
    proto::EnsureLinkResult response;
    response.set_disposition(disposition);
    response.set_reason(reason);
    return response;
  }

  static proto::EnsureLinkResult ready_result(uint64_t generation,
                                              bool replayed) {
    proto::EnsureLinkResult response =
        result(proto::ENSURE_LINK_DISPOSITION_READY, "ready");
    response.set_link_generation(generation);
    response.set_replayed(replayed);
    return response;
  }

  static bool same_identity(const proto::ProviderEngineKey& left,
                            const proto::ProviderEngineKey& right) {
    return left.provider_id() == right.provider_id() &&
           left.profile_digest() == right.profile_digest() &&
           left.engine_uid() == right.engine_uid() &&
           left.incarnation_id() == right.incarnation_id();
  }

  static bool same_contract(const proto::EnsureLinkSpec& left,
                            const proto::InstanceClusterInfo& left_cluster,
                            const proto::EnsureLinkSpec& right,
                            const proto::InstanceClusterInfo& right_cluster) {
    return same_identity(left.prefill(), right.prefill()) &&
           same_identity(left.decode(), right.decode()) &&
           left.model_revision() == right.model_revision() &&
           left.compatibility_proof() == right.compatibility_proof() &&
           left.transfer_mode() == right.transfer_mode() &&
           left_cluster.dp_size() == right_cluster.dp_size() &&
           left_cluster.kv_split_size() == right_cluster.kv_split_size() &&
           std::equal(left_cluster.cluster_ids().begin(),
                      left_cluster.cluster_ids().end(),
                      right_cluster.cluster_ids().begin(),
                      right_cluster.cluster_ids().end()) &&
           std::equal(left_cluster.addrs().begin(),
                      left_cluster.addrs().end(),
                      right_cluster.addrs().begin(),
                      right_cluster.addrs().end()) &&
           std::equal(left_cluster.ports().begin(),
                      left_cluster.ports().end(),
                      right_cluster.ports().begin(),
                      right_cluster.ports().end());
  }

  std::shared_ptr<Peer> find_or_create_peer(const proto::EnsureLinkSpec& spec) {
    const auto found = peers_.find(spec.prefill().engine_uid());
    if (found != peers_.end()) {
      return found->second;
    }
    if (peers_.size() >= max_peer_records_) {
      return nullptr;
    }
    const std::shared_ptr<Peer> peer = std::make_shared<Peer>();
    peers_.emplace(spec.prefill().engine_uid(), peer);
    return peer;
  }

  proto::EnsureLinkResult check_authority(const Peer& peer,
                                          const proto::EnsureLinkSpec& spec) {
    if (authority(spec) < peer.authority) {
      ++stats_.fenced_requests;
      return result(proto::ENSURE_LINK_DISPOSITION_FENCED,
                    "stale-handshake-authority");
    }
    return peer.drained ? result(proto::ENSURE_LINK_DISPOSITION_DRAINING,
                                 "handshake-authority-drained")
                        : proto::EnsureLinkResult();
  }

  proto::EnsureLinkResult authorize(Peer& peer,
                                    const proto::EnsureLinkSpec& spec,
                                    const proto::InstanceClusterInfo& cluster) {
    const proto::EnsureLinkResult current = check_authority(peer, spec);
    if (authority(spec) <= peer.authority) {
      if (current.disposition() != proto::ENSURE_LINK_DISPOSITION_UNSPECIFIED) {
        return current;
      }
      return same_contract(
                 peer.authorized_spec, peer.authorized_cluster, spec, cluster)
                 ? proto::EnsureLinkResult()
                 : result(proto::ENSURE_LINK_DISPOSITION_INCOMPATIBLE,
                          "handshake-authority-reused-with-different-contract");
    }
    const bool reusable = same_contract(
        peer.authorized_spec, peer.authorized_cluster, spec, cluster);
    peer.authority = authority(spec);
    peer.authorized_spec = spec;
    peer.authorized_cluster = cluster;
    peer.drained = false;
    peer.ready = peer.ready && reusable;
    if (!reusable && peer.attempt != nullptr && !peer.attempt->invalidated) {
      peer.attempt->invalidated = true;
      ++stats_.superseded_attempts;
    }
    return proto::EnsureLinkResult();
  }

  void complete(Peer& peer, Attempt& attempt, bool linked) {
    if (attempt.invalidated || peer.drained) {
      attempt.outcome = result(proto::ENSURE_LINK_DISPOSITION_FENCED,
                               "handshake-completion-invalidated");
      return;
    }
    if (!linked) {
      attempt.outcome =
          result(proto::ENSURE_LINK_DISPOSITION_FAILED, "physical-link-failed");
      return;
    }
    ++next_generation_;
    if (next_generation_ == 0) {
      ++next_generation_;
    }
    peer.link_generation = next_generation_;
    peer.ready = true;
    attempt.outcome = ready_result(next_generation_, /*replayed=*/false);
  }

  const size_t max_peer_records_;
  mutable bthread::Mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Peer>> peers_;
  Stats stats_;
  uint64_t next_generation_ = 0;
};

}  // namespace xllm
