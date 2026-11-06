/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <string>
#include <string_view>

namespace xllm::etcd_schema {

// Engine mirror of the xllm-service etcd keys this process reads or writes.
// Service-only keys (load metrics, coordinator members, master incarnation,
// PD-link journal, placement) stay in xllm-service/common/etcd_schema.h.
//
// kDefaultInstancePrefix, kPrefillInstancePrefix, kDecodeInstancePrefix,
// kMixInstancePrefix: registration key prefix chosen by InstanceType in
// XServiceClient::register_instance().
inline constexpr char kDefaultInstancePrefix[] = "XLLM:DEFAULT:";
inline constexpr char kPrefillInstancePrefix[] = "XLLM:PREFILL:";
inline constexpr char kDecodeInstancePrefix[] = "XLLM:DECODE:";
inline constexpr char kMixInstancePrefix[] = "XLLM:MIX:";

// Default master endpoint. XServiceClient::master_service_key_ starts here.
// init() replaces it through master_service_key_for_shard().
// get_master_service() and the master watch use the member after that
// replacement.
inline constexpr char kDefaultMasterServiceKey[] = "XLLM:SERVICE:MASTER";
// Bare master name filtered out of the router-member watch in
// XServiceClient::handle_xservices_watch().
inline constexpr char kDefaultMasterServiceName[] = "MASTER";
// Default router-member prefix. XServiceClient::xservices_key_prefix_ starts
// here. init() replaces it through router_member_prefix_for_shard().
// get_all_xservices() and the router watch use the member after that
// replacement. handle_xservices_watch() compares the member with this default
// before skipping the master key.
inline constexpr char kDefaultRouterMemberPrefix[] = "XLLM:SERVICE:";

// Shard id that keeps the pre-sharding keys above. Checked in
// XServiceClient::init() and used by the two *_for_shard() helpers below.
// Non-default shards are derived from kControlPlaneShardPrefix inside those
// helpers (suffixes ":MASTER" and ":ROUTER:").
inline constexpr char kDefaultControlPlaneShardId[] = "default";
inline constexpr char kControlPlaneShardPrefix[] = "XLLM:CONTROL:SHARD:";

inline bool valid_control_plane_shard_id(std::string_view shard_id) {
  if (shard_id.empty() || shard_id.size() > 64) {
    return false;
  }
  for (const unsigned char value : shard_id) {
    const bool ascii_alphanumeric = (value >= 'a' && value <= 'z') ||
                                    (value >= 'A' && value <= 'Z') ||
                                    (value >= '0' && value <= '9');
    if (!ascii_alphanumeric && value != '-' && value != '_' && value != '.') {
      return false;
    }
  }
  return true;
}

inline std::string master_service_key_for_shard(std::string_view shard_id) {
  if (shard_id == kDefaultControlPlaneShardId) {
    return kDefaultMasterServiceKey;
  }
  return std::string(kControlPlaneShardPrefix) + std::string(shard_id) +
         ":MASTER";
}

inline std::string router_member_prefix_for_shard(std::string_view shard_id) {
  if (shard_id == kDefaultControlPlaneShardId) {
    return kDefaultRouterMemberPrefix;
  }
  return std::string(kControlPlaneShardPrefix) + std::string(shard_id) +
         ":ROUTER:";
}

}  // namespace xllm::etcd_schema
