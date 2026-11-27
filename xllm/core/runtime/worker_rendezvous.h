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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xllm {

class KVCacheTransfer;

#if defined(USE_NPU)
#endif

class WorkerRendezvous final {
 public:
  explicit WorkerRendezvous(
      const std::shared_ptr<KVCacheTransfer>& kv_cache_transfer);

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports);

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports);

 private:
  bool validate_cluster_endpoints(const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<uint16_t>& ports) const;

  const std::shared_ptr<KVCacheTransfer>& kv_cache_transfer_;
};

}  // namespace xllm
