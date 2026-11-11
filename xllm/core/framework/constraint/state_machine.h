/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "core/framework/constraint/snapshot.h"

namespace xllm {

// How advance() reacts to a halt condition: MTP drafting must stop and pad
// the draft with -1; DFlash ignores it because the block is emitted whole.
enum class HaltPolicy : uint8_t {
  RESPECT = 0,
  IGNORE = 1,
};

class ConstraintState;
class ConstraintStateMachine;

// Backend-neutral structured-output constraint state machine: one machine per
// compiled spec, one ConstraintState per sequence. Contract: only tokens the
// engine has committed may advance the live matcher; trial tokens advance
// forks. Implementations: BuiltinJsonConstraintMachine; xgrammar arrives with
// the structured-output roadmap.
class ConstraintStateMachine {
 public:
  virtual ~ConstraintStateMachine() = default;

  virtual ConstraintState initial_state(bool reasoning_phase) const = 0;
  virtual ConstraintState restore(const ConstraintSnapshot& snapshot) const = 0;

  virtual bool can_accept(const ConstraintState& state,
                          int32_t token_id) const = 0;
  virtual bool is_valid(const ConstraintState& state) const = 0;
  virtual bool is_complete(const ConstraintState& state) const = 0;

  // Returns false when the token is rejected; under HaltPolicy::RESPECT the
  // state is also invalidated (frozen), so halt-aware callers see a sticky
  // stop instead of a state that retries later tokens at the same position.
  // Piece-level violations may invalidate under either policy (builtin).
  //
  // Aliasing contract: copies of a ConstraintState share the payload, so an
  // advance through this interface is observed by every copy. Calling
  // through the ConstraintState handle is copy-on-write (a shared payload
  // is forked first), which is what channel-side callers should use; the
  // only sanctioned shared-advance site is the sequence's live commit,
  // which advances the sequence-owned payload through this interface
  // directly.
  virtual bool advance(ConstraintState& state,
                       int32_t token_id,
                       HaltPolicy halt) const = 0;
  virtual ConstraintState fork(const ConstraintState& state) const = 0;
  // Rebuild a matcher from committed history; the result is invalid when the
  // history itself is not acceptable.
  virtual ConstraintState replay(const std::vector<int32_t>& committed_tokens,
                                 bool reasoning_phase) const = 0;

  // Packed allowed-token bitmask, little-endian words, bit i => token i.
  virtual std::vector<uint32_t> allowed_bitmask(
      const ConstraintState& state) const = 0;
  // Optional zero-copy view of the allowed-token bitmask, for backends that
  // cache it (builtin). Returns nullptr when the backend has no shared mask;
  // callers then fall back to allowed_bitmask(). Lets the batch builder avoid
  // both a per-row deep copy and a dynamic_cast to a concrete machine.
  virtual std::shared_ptr<const std::vector<uint32_t>> shared_allowed_bitmask(
      const ConstraintState& state) const {
    return nullptr;
  }
  // Packed-mask row width in 32-bit words for this machine's vocabulary. This
  // is matcher metadata and must not depend on the state's validity, so the
  // batch builder can size an all-deny mask even when every row is invalid.
  virtual size_t bitmask_num_words() const = 0;
  // Whether this backend supports spec (draft) snapshots. A backend that does
  // not produce spec snapshots must keep its live constraint state on an empty
  // snapshot instead of being reset to unconstrained.
  virtual bool supports_spec_snapshot() const { return false; }
  // Fingerprint of the matcher FSM only (no token history); identical FSMs
  // share mask cache entries.
  virtual uint64_t fingerprint(const ConstraintState& state) const = 0;

  virtual ConstraintSnapshot snapshot(const ConstraintState& state) const = 0;
  virtual std::string error_detail(const ConstraintState& state) const = 0;
};

// Type-erased per-sequence matcher handle bundling its state machine, so
// batch channels can carry states that know their own operations. Copying
// shares the payload; advance() through the handle is copy-on-write (a
// shared payload is forked first), and fork() when a branch must diverge
// mutably.
class ConstraintState final {
 public:
  ConstraintState() = default;

  template <typename T>
  ConstraintState(std::shared_ptr<T> payload,
                  std::shared_ptr<const ConstraintStateMachine> machine)
      : payload_(std::move(payload)),
        machine_(std::move(machine)),
        payload_type_(payload_ ? std::type_index(typeid(T))
                               : std::type_index(typeid(void))) {}

  explicit operator bool() const { return payload_ != nullptr; }

  const ConstraintStateMachine* machine() const { return machine_.get(); }

  // Interface shortcuts; require a non-empty state.
  bool initialized() const { return payload_ != nullptr; }
  bool is_valid() const {
    CHECK(initialized());
    return machine_->is_valid(*this);
  }
  bool is_complete() const {
    CHECK(initialized());
    return machine_->is_complete(*this);
  }
  bool can_accept(int32_t token_id) const {
    CHECK(initialized());
    return machine_->can_accept(*this, token_id);
  }
  bool advance(int32_t token_id, HaltPolicy halt) {
    CHECK(initialized());
    if (payload_.use_count() > 1) {
      // Copy-on-write guard for the aliasing contract: copies share the
      // payload, so advancing through a shared handle would be observed by
      // every copy (e.g. a channel state that still aliases a live
      // Sequence under schedule overlap would double-advance). Fork first
      // and keep the mutation local to this handle. The sequence's live
      // commit path intentionally advances the shared payload through the
      // machine interface directly; that is the sanctioned shared-advance
      // site and bypasses this guard.
      *this = machine_->fork(*this);
    }
    return machine_->advance(*this, token_id, halt);
  }
  ConstraintState fork() const {
    CHECK(initialized());
    return machine_->fork(*this);
  }
  std::vector<uint32_t> allowed_bitmask() const {
    CHECK(initialized());
    return machine_->allowed_bitmask(*this);
  }
  uint64_t fingerprint() const {
    CHECK(initialized());
    return machine_->fingerprint(*this);
  }
  ConstraintSnapshot snapshot() const {
    CHECK(initialized());
    return machine_->snapshot(*this);
  }

  template <typename T>
  const T& as() const {
    CHECK(payload_type_ == std::type_index(typeid(T)))
        << "ConstraintState payload type mismatch on as<>";
    return *static_cast<const T*>(payload_.get());
  }

  template <typename T>
  T& as() {
    CHECK(payload_type_ == std::type_index(typeid(T)))
        << "ConstraintState payload type mismatch on as<>";
    return *static_cast<T*>(payload_.get());
  }

  // Typed payload handle; shares ownership with this state.
  template <typename T>
  std::shared_ptr<T> shared() const {
    CHECK(payload_type_ == std::type_index(typeid(T)))
        << "ConstraintState payload type mismatch on shared<>";
    return std::static_pointer_cast<T>(payload_);
  }

 private:
  std::shared_ptr<void> payload_;
  std::shared_ptr<const ConstraintStateMachine> machine_;
  std::type_index payload_type_ = std::type_index(typeid(void));
};

}  // namespace xllm
