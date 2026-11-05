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

#include "layers/mlu/deepseek_v41/engram_shared_memory.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

namespace xllm::layer {
namespace {

class SharedEngramMapping final {
 public:
  SharedEngramMapping(int32_t fd, int64_t bytes) : fd_(fd), bytes_(bytes) {
    data_ = ::mmap(
        nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, /*offset=*/0);
    PCHECK(data_ != MAP_FAILED) << "Cannot map shared Engram storage";
  }
  ~SharedEngramMapping() {
    ::munmap(data_, bytes_);
    ::close(fd_);
  }
  SharedEngramMapping(const SharedEngramMapping&) = delete;
  SharedEngramMapping& operator=(const SharedEngramMapping&) = delete;

  void* data() const { return data_; }

 private:
  int32_t fd_;
  int64_t bytes_;
  void* data_;
};

struct SharedEngramDescriptor {
  int64_t pid = 0;
  int64_t fd = 0;
  int64_t device = 0;
  int64_t inode = 0;
  int64_t bytes = 0;
  int64_t host = 0;
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t name = 0;
};
static_assert(sizeof(SharedEngramDescriptor) == 9 * sizeof(int64_t));

int64_t host_identity() {
  std::ifstream file("/proc/sys/kernel/random/boot_id");
  std::string identity;
  CHECK(static_cast<bool>(std::getline(file, identity)) && !identity.empty())
      << "Cannot identify the host for Engram shared memory";
  return static_cast<int64_t>(std::hash<std::string>{}(identity));
}

torch::Tensor gather_metadata(const torch::Tensor& host, ProcessGroup& group) {
  if (group.world_size() == 1) {
    return host.unsqueeze(0);
  }
  return group.allgather_base_sync(host.to(group.device()))
      .to(torch::kCPU)
      .contiguous();
}

}  // namespace

torch::Tensor load_shared_engram_tensor(const torch::Tensor& source,
                                        const std::string& name,
                                        ProcessGroup& group) {
  CHECK(source.device().is_cpu());
  CHECK_EQ(source.dim(), 2);
  CHECK_EQ(source.element_size(), 1) << "Engram shared storage keeps FP8 bytes";
  CHECK_GT(source.numel(), 0);
  CHECK_GT(group.world_size(), 0);
  CHECK_GE(group.rank(), 0);
  CHECK_LT(group.rank(), group.world_size());

  const int64_t bytes = source.numel();
  const int64_t host = host_identity();
  const auto hosts =
      gather_metadata(torch::tensor({host}, torch::kInt64), group);
  const int64_t* host_ids = hosts.data_ptr<int64_t>();
  std::vector<int32_t> node_ranks;
  node_ranks.reserve(group.world_size());
  for (int32_t rank = 0; rank < group.world_size(); ++rank) {
    if (host_ids[rank] != host) {
      continue;
    }
    node_ranks.emplace_back(rank);
  }
  const auto local =
      std::lower_bound(node_ranks.begin(), node_ranks.end(), group.rank());
  CHECK(local != node_ranks.end() && *local == group.rank());
  const int64_t local_rank = local - node_ranks.begin();
  const int64_t local_size = static_cast<int64_t>(node_ranks.size());
  const int32_t owner = node_ranks.front();
  const int64_t table_name =
      static_cast<int64_t>(std::hash<std::string>{}(name));
  std::shared_ptr<SharedEngramMapping> mapping;
  SharedEngramDescriptor descriptor;
  if (group.rank() == owner) {
    const int32_t fd = ::memfd_create("xllm-engram", MFD_CLOEXEC);
    PCHECK(fd >= 0) << "Cannot create shared Engram storage";
    PCHECK(::ftruncate(fd, bytes) == 0);
    mapping = std::make_shared<SharedEngramMapping>(fd, bytes);
    struct stat info{};
    PCHECK(::fstat(fd, &info) == 0);
    descriptor = {static_cast<int64_t>(::getpid()),
                  fd,
                  static_cast<int64_t>(info.st_dev),
                  static_cast<int64_t>(info.st_ino),
                  bytes,
                  host,
                  source.size(0),
                  source.size(1),
                  table_name};
  }
  const auto metadata =
      gather_metadata(torch::from_blob(&descriptor, {9}, torch::kInt64), group);
  std::memmove(
      &descriptor, metadata[owner].data_ptr<int64_t>(), sizeof(descriptor));
  CHECK_EQ(descriptor.host, host)
      << "Engram shared memory owner must be on the same host";
  CHECK(descriptor.bytes == bytes && descriptor.rows == source.size(0) &&
        descriptor.columns == source.size(1) && descriptor.name == table_name)
      << "Ranks on each host must load matching Engram tensors in the same "
         "order";
  if (group.rank() != owner) {
    const std::string path = "/proc/" + std::to_string(descriptor.pid) +
                             "/fd/" + std::to_string(descriptor.fd);
    const int32_t fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    PCHECK(fd >= 0) << "Cannot open shared Engram storage; ranks on each host "
                       "must share a PID namespace";
    struct stat info{};
    PCHECK(::fstat(fd, &info) == 0);
    CHECK(static_cast<int64_t>(info.st_dev) == descriptor.device &&
          static_cast<int64_t>(info.st_ino) == descriptor.inode &&
          info.st_size == bytes)
        << "Shared Engram descriptor identity mismatch";
    mapping = std::make_shared<SharedEngramMapping>(fd, bytes);
  }
  auto table = torch::from_blob(
      mapping->data(),
      source.sizes(),
      [mapping](void*) {},
      torch::TensorOptions().dtype(torch::kUInt8));
  const int64_t begin = source.size(0) * local_rank / local_size;
  const int64_t end = source.size(0) * (local_rank + 1) / local_size;
  table.slice(/*dim=*/0, begin, end)
      .copy_(
          source.slice(/*dim=*/0, begin, end).contiguous().view(torch::kUInt8));
  // Publish all CPU writes before any rank can query the complete table.
  if (group.world_size() > 1) {
    auto ready = torch::ones(
        {1},
        torch::TensorOptions().dtype(torch::kInt64).device(group.device()));
    group.allreduce(ready);
    CHECK_EQ(ready.to(torch::kCPU).item<int64_t>(), group.world_size());
  }
  return table;
}

}  // namespace xllm::layer
