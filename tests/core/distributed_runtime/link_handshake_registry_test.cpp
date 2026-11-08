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

#include "core/distributed_runtime/link_handshake_registry.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace xllm {
namespace {

proto::EnsureLinkRequest make_request() {
  proto::EnsureLinkRequest request;
  auto* spec = request.mutable_spec();
  spec->set_contract_version(2);
  spec->set_model_revision("model");
  spec->set_compatibility_proof("layout-proof");
  spec->set_transfer_mode(proto::TRANSFER_MODE_PULL);
  spec->set_coordinator_epoch(1);
  spec->set_adjacency_generation(1);
  for (auto* identity : {spec->mutable_prefill(), spec->mutable_decode()}) {
    identity->set_provider_id(proto::PROVIDER_ID_XLLM_NATIVE);
    identity->set_profile_digest("profile");
  }
  spec->mutable_prefill()->set_engine_uid("p0");
  spec->mutable_prefill()->set_incarnation_id("p-incarnation");
  spec->mutable_decode()->set_engine_uid("d0");
  spec->mutable_decode()->set_incarnation_id("d-incarnation");
  auto* cluster = request.mutable_prefill_cluster();
  cluster->set_instance_name("p0");
  cluster->set_incarnation_id("p-incarnation");
  cluster->add_cluster_ids(1);
  cluster->add_addrs("127.0.0.1:12345");
  cluster->add_ports(12346);
  cluster->set_dp_size(1);
  cluster->set_kv_split_size(1);
  return request;
}

// Deterministically holds the physical operation until every intended follower
// has joined the real registry. The timeout protects a broken test from
// hanging.
class BlockingLink final {
 public:
  bool run(bool success = true) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++calls_;
    changed_.notify_all();
    changed_.wait(lock, [this] { return released_; });
    return success;
  }

  bool await_calls(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(3), [this, count] {
      return calls_ >= count;
    });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  uint32_t calls_ = 0;
  bool released_ = false;
};

bool await_joins(const LinkHandshakeRegistry& registry, uint64_t count) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (registry.stats().joined_attempts >= count) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void check_concurrent_result(bool success) {
  LinkHandshakeRegistry registry;
  const proto::EnsureLinkRequest request = make_request();
  BlockingLink link;
  constexpr uint32_t kCallers = 8;
  std::vector<proto::EnsureLinkResult> results(kCallers);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (uint32_t index = 0; index < kCallers; ++index) {
    callers.emplace_back([&, index] {
      results[index] = registry.ensure(request.spec(),
                                       request.prefill_cluster(),
                                       [&] { return link.run(success); });
    });
  }
  EXPECT_TRUE(link.await_calls(1));
  EXPECT_TRUE(await_joins(registry, kCallers - 1));
  link.release();
  for (auto& caller : callers) {
    caller.join();
  }
  const proto::EnsureLinkDisposition expected =
      success ? proto::ENSURE_LINK_DISPOSITION_READY
              : proto::ENSURE_LINK_DISPOSITION_FAILED;
  uint32_t replayed = 0;
  for (const auto& response : results) {
    EXPECT_EQ(response.disposition(), expected);
    EXPECT_EQ(response.link_generation(), success ? 1 : 0);
    replayed += response.replayed();
  }
  EXPECT_EQ(replayed, kCallers - 1);
  EXPECT_EQ(registry.stats().physical_attempts, 1);
  if (!success) {
    const auto retry = registry.ensure(
        request.spec(), request.prefill_cluster(), [] { return true; });
    EXPECT_EQ(retry.disposition(), proto::ENSURE_LINK_DISPOSITION_READY);
    EXPECT_EQ(registry.stats().physical_attempts, 2);
  }
}

TEST(LinkHandshakeRegistryTest, ConcurrentSuccessExecutesPhysicalLinkOnce) {
  check_concurrent_result(/*success=*/true);
}

TEST(LinkHandshakeRegistryTest, ConcurrentFailureIsSharedAndLaterRetryWorks) {
  check_concurrent_result(/*success=*/false);
}

TEST(LinkHandshakeRegistryTest,
     LostResponseAndNewAuthorityReuseExactReadyLink) {
  LinkHandshakeRegistry registry;
  proto::EnsureLinkRequest request = make_request();
  const auto first = registry.ensure(
      request.spec(), request.prefill_cluster(), [] { return true; });
  const auto retry =
      registry.ensure(request.spec(), request.prefill_cluster(), [] {
        ADD_FAILURE();
        return false;
      });
  EXPECT_TRUE(retry.replayed());
  EXPECT_EQ(first.link_generation(), retry.link_generation());
  request.mutable_spec()->set_adjacency_generation(2);
  const auto rebased =
      registry.ensure(request.spec(), request.prefill_cluster(), [] {
        ADD_FAILURE();
        return false;
      });
  EXPECT_EQ(rebased.disposition(), proto::ENSURE_LINK_DISPOSITION_READY);
  EXPECT_TRUE(rebased.replayed());
  EXPECT_EQ(registry.stats().physical_attempts, 1);
}

TEST(LinkHandshakeRegistryTest, SameAuthorityCannotChangeEndpointOrProfile) {
  LinkHandshakeRegistry registry;
  proto::EnsureLinkRequest request = make_request();
  registry.ensure(
      request.spec(), request.prefill_cluster(), [] { return true; });
  request.mutable_prefill_cluster()->set_ports(0, 23456);
  EXPECT_EQ(registry
                .ensure(request.spec(),
                        request.prefill_cluster(),
                        [] {
                          ADD_FAILURE();
                          return false;
                        })
                .disposition(),
            proto::ENSURE_LINK_DISPOSITION_INCOMPATIBLE);
  request.mutable_prefill_cluster()->set_ports(0, 12346);
  request.mutable_spec()->mutable_prefill()->set_profile_digest("new-profile");
  EXPECT_EQ(registry
                .ensure(request.spec(),
                        request.prefill_cluster(),
                        [] {
                          ADD_FAILURE();
                          return false;
                        })
                .disposition(),
            proto::ENSURE_LINK_DISPOSITION_INCOMPATIBLE);
  EXPECT_EQ(registry.stats().physical_attempts, 1);
}

TEST(LinkHandshakeRegistryTest,
     NewIncarnationMustRelinkAndOldAuthorityIsFenced) {
  LinkHandshakeRegistry registry;
  proto::EnsureLinkRequest old = make_request();
  registry.ensure(old.spec(), old.prefill_cluster(), [] { return true; });
  proto::EnsureLinkRequest fresh = old;
  fresh.mutable_spec()->set_adjacency_generation(2);
  fresh.mutable_spec()->mutable_prefill()->set_incarnation_id("restart");
  fresh.mutable_prefill_cluster()->set_incarnation_id("restart");
  fresh.mutable_prefill_cluster()->set_cluster_ids(0, 2);
  const auto relinked = registry.ensure(
      fresh.spec(), fresh.prefill_cluster(), [] { return true; });
  EXPECT_EQ(relinked.link_generation(), 2);
  EXPECT_EQ(registry
                .ensure(old.spec(),
                        old.prefill_cluster(),
                        [] {
                          ADD_FAILURE();
                          return false;
                        })
                .disposition(),
            proto::ENSURE_LINK_DISPOSITION_FENCED);
  EXPECT_EQ(registry.drain(old.spec()).disposition(),
            proto::ENSURE_LINK_DISPOSITION_FENCED);
}

TEST(LinkHandshakeRegistryTest, MismatchedDrainCannotRetireReadyPeer) {
  LinkHandshakeRegistry registry;
  const proto::EnsureLinkRequest request = make_request();
  registry.ensure(
      request.spec(), request.prefill_cluster(), [] { return true; });
  proto::EnsureLinkSpec wrong = request.spec();
  wrong.mutable_prefill()->set_incarnation_id("wrong-process");
  EXPECT_EQ(registry.drain(wrong).disposition(),
            proto::ENSURE_LINK_DISPOSITION_FENCED);
  const auto ready =
      registry.ensure(request.spec(), request.prefill_cluster(), [] {
        ADD_FAILURE();
        return false;
      });
  EXPECT_EQ(ready.disposition(), proto::ENSURE_LINK_DISPOSITION_READY);
  EXPECT_TRUE(ready.replayed());
}

TEST(LinkHandshakeRegistryTest, DrainFencesInflightCompletionAndSameAuthority) {
  LinkHandshakeRegistry registry;
  proto::EnsureLinkRequest request = make_request();
  BlockingLink link;
  proto::EnsureLinkResult response;
  std::thread owner([&] {
    response = registry.ensure(
        request.spec(), request.prefill_cluster(), [&] { return link.run(); });
  });
  EXPECT_TRUE(link.await_calls(1));
  EXPECT_EQ(registry.drain(request.spec()).disposition(),
            proto::ENSURE_LINK_DISPOSITION_READY);
  link.release();
  owner.join();
  EXPECT_EQ(response.disposition(), proto::ENSURE_LINK_DISPOSITION_DRAINING);
  EXPECT_EQ(registry
                .ensure(request.spec(),
                        request.prefill_cluster(),
                        [] {
                          ADD_FAILURE();
                          return false;
                        })
                .disposition(),
            proto::ENSURE_LINK_DISPOSITION_DRAINING);
  request.mutable_spec()->set_adjacency_generation(2);
  EXPECT_EQ(
      registry
          .ensure(
              request.spec(), request.prefill_cluster(), [] { return true; })
          .disposition(),
      proto::ENSURE_LINK_DISPOSITION_READY);
  EXPECT_EQ(registry.stats().physical_attempts, 2);
}

TEST(LinkHandshakeRegistryTest,
     NewEpochFencesOldWaiterButSharesSamePhysicalWork) {
  LinkHandshakeRegistry registry;
  const proto::EnsureLinkRequest old = make_request();
  proto::EnsureLinkRequest fresh = old;
  fresh.mutable_spec()->set_coordinator_epoch(2);
  BlockingLink link;
  proto::EnsureLinkResult first;
  proto::EnsureLinkResult second;
  std::thread owner([&] {
    first = registry.ensure(
        old.spec(), old.prefill_cluster(), [&] { return link.run(); });
  });
  EXPECT_TRUE(link.await_calls(1));
  std::thread waiter([&] {
    second = registry.ensure(fresh.spec(), fresh.prefill_cluster(), [] {
      ADD_FAILURE();
      return false;
    });
  });
  EXPECT_TRUE(await_joins(registry, 1));
  link.release();
  owner.join();
  waiter.join();
  EXPECT_EQ(first.disposition(), proto::ENSURE_LINK_DISPOSITION_FENCED);
  EXPECT_EQ(second.disposition(), proto::ENSURE_LINK_DISPOSITION_READY);
  EXPECT_EQ(registry.stats().physical_attempts, 1);
}

TEST(LinkHandshakeRegistryTest, UnrelatedPeerCanCompleteWhileAnotherIsBlocked) {
  LinkHandshakeRegistry registry;
  const proto::EnsureLinkRequest first = make_request();
  proto::EnsureLinkRequest second = first;
  second.mutable_spec()->mutable_prefill()->set_engine_uid("p1");
  second.mutable_prefill_cluster()->set_instance_name("p1");
  BlockingLink link;
  std::thread owner([&] {
    registry.ensure(
        first.spec(), first.prefill_cluster(), [&] { return link.run(); });
  });
  EXPECT_TRUE(link.await_calls(1));
  EXPECT_EQ(
      registry
          .ensure(second.spec(), second.prefill_cluster(), [] { return true; })
          .disposition(),
      proto::ENSURE_LINK_DISPOSITION_READY);
  link.release();
  owner.join();
  EXPECT_EQ(registry.stats().physical_attempts, 2);
}

TEST(LinkHandshakeRegistryTest, ChangedInflightContractWaitsThenRelinks) {
  LinkHandshakeRegistry registry;
  const proto::EnsureLinkRequest old = make_request();
  proto::EnsureLinkRequest fresh = old;
  fresh.mutable_spec()->set_adjacency_generation(2);
  fresh.mutable_spec()->mutable_prefill()->set_incarnation_id("restart");
  fresh.mutable_prefill_cluster()->set_incarnation_id("restart");
  fresh.mutable_prefill_cluster()->set_cluster_ids(0, 2);
  BlockingLink link;
  proto::EnsureLinkResult first;
  proto::EnsureLinkResult second;
  std::thread owner([&] {
    first = registry.ensure(
        old.spec(), old.prefill_cluster(), [&] { return link.run(); });
  });
  EXPECT_TRUE(link.await_calls(1));
  std::thread replacement([&] {
    second = registry.ensure(
        fresh.spec(), fresh.prefill_cluster(), [] { return true; });
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (registry.stats().superseded_attempts == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(registry.stats().superseded_attempts, 1);
  EXPECT_EQ(registry.stats().physical_attempts, 1);
  link.release();
  owner.join();
  replacement.join();
  EXPECT_EQ(first.disposition(), proto::ENSURE_LINK_DISPOSITION_FENCED);
  EXPECT_EQ(second.disposition(), proto::ENSURE_LINK_DISPOSITION_READY);
  EXPECT_EQ(registry.stats().physical_attempts, 2);
}

TEST(LinkHandshakeRegistryTest, BoundedTombstonesRejectExcessPeers) {
  LinkHandshakeRegistry registry(/*max_peer_records=*/1);
  proto::EnsureLinkRequest request = make_request();
  registry.drain(request.spec());
  request.mutable_spec()->mutable_prefill()->set_engine_uid("p1");
  request.mutable_prefill_cluster()->set_instance_name("p1");
  EXPECT_EQ(registry
                .ensure(request.spec(),
                        request.prefill_cluster(),
                        [] {
                          ADD_FAILURE();
                          return false;
                        })
                .disposition(),
            proto::ENSURE_LINK_DISPOSITION_DEGREE_LIMITED);
  EXPECT_EQ(registry.stats().peer_records, 1);
}

}  // namespace
}  // namespace xllm
