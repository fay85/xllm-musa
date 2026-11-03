/* Copyright 2026 The xLLM Authors.

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

#include "lora_registry.h"

#include <glog/logging.h>

#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <utility>

namespace xllm {

std::optional<uint64_t> LoRARegistry::register_adapter(const LoRARequest& req) {
  if (req.lora_name.empty()) {
    LOG(ERROR) << "[LoRARegistry] refuse to register empty name";
    return std::nullopt;
  }
  std::unique_lock lock(mu_);

  auto it = name_to_id_.find(req.lora_name);
  if (it != name_to_id_.end()) {
    // Name already known. Idempotent path.
    const uint64_t existing_id = it->second;
    auto& entry = id_to_entry_.at(existing_id);
    // Path consistency is checked BEFORE the drain-revive branch. Checking
    // only on the non-drain path let a drain-window re-register silently
    // rebind the name to a completely different path.
    if (entry.req.lora_path != req.lora_path) {
      LOG(ERROR) << "[LoRARegistry] name '" << req.lora_name
                 << "' already bound to '" << entry.req.lora_path
                 << "', refuse to rebind to '" << req.lora_path
                 << "' (unload the adapter before loading a different path)";
      return std::nullopt;
    }
    if (req.lora_int_id > 0 && req.lora_int_id != existing_id) {
      LOG(ERROR) << "[LoRARegistry] id conflict for '" << req.lora_name
                 << "': forced id=" << req.lora_int_id
                 << " but already registered as id=" << existing_id;
      return std::nullopt;
    }
    if (entry.unloading) {
      // Drain in progress. Re-registering during drain is legal: revive
      // the existing entry (skip renumbering) so ongoing requests keep
      // working. This matches vLLM behaviour when the same name is
      // reloaded before drain completes. The path + id checks above
      // already guarantee the revive binds identical weights and id.
      entry.unloading = false;
      LOG(INFO) << "[LoRARegistry] revived '" << req.lora_name
                << "' (was draining) id=" << existing_id;
      return existing_id;
    }
    // Same name, same path -> idempotent, no state change.
    return existing_id;
  }

  // Forced id path: the API process allocated the id on the API side and
  // broadcast it; adopt it verbatim so every rank binds the name to the
  // same id. Refuse when the id is already taken by a different adapter;
  // keep the monotonic counter past any forced id so later local
  // allocations never collide.
  uint64_t id = 0;
  if (req.lora_int_id > 0) {
    if (id_to_entry_.find(req.lora_int_id) != id_to_entry_.end()) {
      LOG(ERROR) << "[LoRARegistry] forced id=" << req.lora_int_id
                 << " already bound to '"
                 << id_to_entry_.at(req.lora_int_id).req.lora_name
                 << "', refuse to register '" << req.lora_name << "'";
      return std::nullopt;
    }
    id = req.lora_int_id;
    uint64_t expected = next_id_.load(std::memory_order_relaxed);
    while (expected <= id && !next_id_.compare_exchange_weak(
                                 expected, id + 1, std::memory_order_relaxed)) {
    }
  } else {
    id = next_id_.fetch_add(1);
  }
  name_to_id_[req.lora_name] = id;
  LoRARequest stored = req;
  stored.lora_int_id = id;
  // Entry holds std::atomic so its copy/move assignments are deleted; a
  // map assignment (operator[] + move-assign) cannot compile. Construct
  // the node in place via piecewise emplace instead (#506 CI).
  id_to_entry_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(id),
      std::forward_as_tuple(stored, /*pin_count=*/0, /*unloading=*/false));
  LOG(INFO) << "[LoRARegistry] registered '" << req.lora_name << "' id=" << id
            << " path=" << req.lora_path;
  // Notify the observability layer of the new adapter binding.
  if (on_register_) on_register_(id, req.lora_name);
  return id;
}

std::optional<LoRARegistry::PinnedAdapter> LoRARegistry::lookup_and_pin(
    const std::string& lora_name) {
  // shared_lock + atomic pin bump. This is the per-request hot path; a
  // unique_lock here would serialise every concurrent resolver.
  std::shared_lock lock(mu_);
  auto it = name_to_id_.find(lora_name);
  if (it == name_to_id_.end()) return std::nullopt;
  const uint64_t id = it->second;
  auto& entry = id_to_entry_.at(id);
  if (entry.unloading) return std::nullopt;
  entry.pin_count.fetch_add(1, std::memory_order_acq_rel);
  return PinnedAdapter{id, entry.req};
}

void LoRARegistry::unpin(uint64_t int_id) {
  FinalRemovalCallback cb;
  {
    std::unique_lock lock(mu_);
    auto it = id_to_entry_.find(int_id);
    if (it == id_to_entry_.end()) {
      // Should never happen: caller pinned so the entry existed.
      LOG(ERROR) << "[LoRARegistry] unpin of unknown id=" << int_id;
      return;
    }
    auto& entry = it->second;
    if (entry.pin_count.load(std::memory_order_acquire) <= 0) {
      LOG(ERROR) << "[LoRARegistry] pin_count already <= 0 for id=" << int_id;
      return;
    }
    const int32_t prior =
        entry.pin_count.fetch_sub(1, std::memory_order_acq_rel);
    if (prior == 1 && entry.unloading) {
      // Deferred removal completes now.
      LOG(INFO) << "[LoRARegistry] drained '" << entry.req.lora_name
                << "' id=" << int_id << ", removing";
      name_to_id_.erase(entry.req.lora_name);
      id_to_entry_.erase(it);
      cb = on_final_removal_;
    }
  }
  // The callback takes LoRARuntime's materialise_mu_. Run it after releasing
  // mu_ to keep the registry/runtime lock order acyclic.
  if (cb) cb(int_id);
}

std::optional<LoRARequest> LoRARegistry::lookup(
    const std::string& lora_name) const {
  std::shared_lock lock(mu_);
  auto it = name_to_id_.find(lora_name);
  if (it == name_to_id_.end()) return std::nullopt;
  const auto& entry = id_to_entry_.at(it->second);
  if (entry.unloading) return std::nullopt;
  return entry.req;
}

std::optional<LoRARequest> LoRARegistry::lookup(uint64_t int_id) const {
  std::shared_lock lock(mu_);
  auto it = id_to_entry_.find(int_id);
  if (it == id_to_entry_.end()) return std::nullopt;
  if (it->second.unloading) return std::nullopt;
  return it->second.req;
}

bool LoRARegistry::unregister(const std::string& lora_name) {
  uint64_t id = 0;
  FinalRemovalCallback cb;
  {
    std::unique_lock lock(mu_);
    auto it = name_to_id_.find(lora_name);
    if (it == name_to_id_.end()) return false;
    id = it->second;
    auto& entry = id_to_entry_.at(id);
    if (entry.pin_count == 0) {
      // No in-flight work; free immediately.
      LOG(INFO) << "[LoRARegistry] unregistered '" << lora_name << "' id=" << id
                << " (no pins)";
      name_to_id_.erase(it);
      id_to_entry_.erase(id);
      cb = on_final_removal_;
    } else {
      // Requests are still using it; caller (unload_lora_adapter handler) is
      // expected to poll until the entry disappears.
      entry.unloading = true;
      LOG(INFO) << "[LoRARegistry] draining '" << lora_name << "' id=" << id
                << " pins=" << entry.pin_count;
    }
  }
  // See unpin(): final-removal callbacks may acquire runtime locks.
  if (cb) cb(id);
  return true;
}

bool LoRARegistry::contains(uint64_t int_id) const {
  std::shared_lock lock(mu_);
  return id_to_entry_.find(int_id) != id_to_entry_.end();
}

std::vector<LoRARequest> LoRARegistry::list() const {
  std::shared_lock lock(mu_);
  std::vector<LoRARequest> out;
  out.reserve(id_to_entry_.size());
  for (const auto& [_, entry] : id_to_entry_) {
    if (!entry.unloading) out.push_back(entry.req);
  }
  return out;
}

size_t LoRARegistry::size() const {
  std::shared_lock lock(mu_);
  return id_to_entry_.size();
}

}  // namespace xllm
