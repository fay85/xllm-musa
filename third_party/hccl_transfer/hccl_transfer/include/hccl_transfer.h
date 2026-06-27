#pragma once

#include <brpc/server.h>
#include <hccl/hccl.h>

#include <thread>

#include "hccl_transfer_service.h"
#include "threadpool.h"

namespace hccl_transfer {

class HcclTransfer final {
 public:
  HcclTransfer(const std::string& addr, int32_t device_id);

  virtual ~HcclTransfer();

  // Start the RPC service.
  bool initialize(int32_t port);

  // Register the memory that needs to be transmitted.
  uint32_t register_memory(const std::vector<void*>& mem_addrs,
                           const std::vector<int64_t>& shape,
                           HcclDataType data_type);

  // Create communication domain.
  bool create_comm_domain(const std::string& remote_addr,
                          HcclRootInfo* root_info = nullptr);

  bool push_memory_blocks(const std::string& remote_addr,
                          uint32_t src_mem_id,
                          const std::vector<uint64_t>& src_blocks,
                          uint32_t dst_mem_id,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& layer_ids);

  bool pull_memory_blocks(const std::string& remote_addr,
                          uint32_t src_mem_id,
                          const std::vector<uint64_t>& src_blocks,
                          uint32_t dst_mem_id,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& layer_ids);

  bool recv_blocks(const std::string& remote_addr,
                   uint32_t mem_id,
                   const std::vector<uint64_t>& blocks,
                   const std::vector<uint64_t>& block_lengths,
                   const std::vector<int64_t>& layer_ids);

  bool send_blocks(const std::string& remote_addr,
                   uint32_t mem_id,
                   const std::vector<uint64_t>& blocks,
                   const std::vector<uint64_t>& block_lengths,
                   const std::vector<int64_t>& layer_ids);

 private:
  // A data structure used to describe transmitted memory. It includes the
  // address of each layer of memory, the shape of the memory and the data type
  // stored within it. The shape of each layer of memory must be consistent, and
  // the first dimension of the shape represents the number of blocks.
  struct MemoryDesc {
    MemoryDesc(const std::vector<void*>& mem_addrs,
               const std::vector<int64_t>& shape,
               HcclDataType data_type)
        : mem_addrs(mem_addrs), shape(shape), data_type(data_type) {}

    std::vector<void*> mem_addrs;
    std::vector<int64_t> shape;
    HcclDataType data_type;
  };

  proto::HcclTransferService_Stub* create_rpc_channel(
      const std::string& remote_addr);

  // The server address of the current instance in the format of ip:port
  std::string addr_;
  // The device ID of the current instance
  int32_t device_id_;
  // The communication stream
  aclrtStream stream_;
  HcclRootInfo root_info_;

  // The flag indicating whether the service has started
  bool has_initialized_ = false;
  brpc::Server server_;
  std::shared_ptr<HcclTransferService> service_;
  // RPC communication thread pool
  std::shared_ptr<ThreadPool> rpc_thread_pool_;

  std::vector<MemoryDesc> memory_vec_;

  // HCCL communication thread pool
  std::shared_ptr<ThreadPool> hccl_thread_pool_;

  // The map storing instance addresses to RPC stubs
  std::unordered_map<std::string, proto::HcclTransferService_Stub*> stub_map_;
  // The map storing instance addresses to hccl communication domian
  std::unordered_map<std::string, HcclComm> comm_map_;
  // The map storing the rank of the current instance in the communication
  // domain created with each instance address
  std::unordered_map<std::string, uint32_t> rank_map_;
};

}  // namespace hccl_transfer