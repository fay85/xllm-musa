/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
https://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/
#pragma once

#include <glog/logging.h>

#include <string>
#include <unordered_set>

#include "transfer_engine.h"
#include "transport/rdma_transport/rdma_context.h"
#include "transport/rdma_transport/rdma_transport.h"

namespace xllm::detail {

// The caller must hold its session lock and have no live session reference for
// this peer. This is a session-boundary operation, never a per-transfer retry.
inline bool prepare_mooncake_segment_reopen(mooncake::TransferEngine& engine,
                                           const std::string& remote_address,
                                           bool invalidate_endpoints) {
  auto metadata = engine.getMetadata();
  if (!metadata) {
    return false;
  }
  auto previous = metadata->getSegmentDescByName(remote_address);
  auto fresh = metadata->getSegmentDescByName(remote_address,
                                             /*force_update=*/true);
  if (!fresh) {
    LOG(ERROR) << "Failed to refresh Mooncake peer descriptor: "
               << remote_address;
    return false;
  }
  if (!invalidate_endpoints) {
    return true;
  }
  auto* transport = dynamic_cast<mooncake::RdmaTransport*>(
      engine.getTransport("rdma"));
  if (transport == nullptr) {
    return true;
  }
  std::unordered_set<std::string> peer_paths;
  peer_paths.reserve(fresh->devices.size() +
                     (previous ? previous->devices.size() : 0));
  const auto collect_paths = [&peer_paths](const auto& descriptor) {
    if (!descriptor) {
      return;
    }
    const std::string& server_name = descriptor->rdma_server_name.empty()
                                         ? descriptor->name
                                         : descriptor->rdma_server_name;
    for (const auto& device : descriptor->devices) {
      peer_paths.emplace(server_name + "@" + device.name);
    }
  };
  collect_paths(previous);
  collect_paths(fresh);
  for (const auto& context : transport->getContextList()) {
    for (const auto& peer_path : peer_paths) {
      // Mooncake's public per-peer API retires QPs in two phases. It does not
      // reset the device, disconnect unrelated peers, or replay old transfers.
      if (context->deleteEndpoint(peer_path) != 0) {
        LOG(ERROR) << "Failed to retire Mooncake endpoint: " << peer_path;
        return false;
      }
    }
  }
  LOG(INFO) << "[MUSA_SEGMENT_REOPEN] refreshed=" << remote_address
            << " retired_peer_paths=" << peer_paths.size();
  return true;
}

}  // namespace xllm::detail
