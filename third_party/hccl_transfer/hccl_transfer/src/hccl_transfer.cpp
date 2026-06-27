#include "hccl_transfer.h"

#include <brpc/channel.h>

#include <future>
#include <numeric>

namespace hccl_transfer {
namespace {

// Define a macro to simplify adding elements from a vector to a repeated field
#define ADD_VECTOR_TO_PROTO(proto_field, vec) \
  do {                                        \
    proto_field->Reserve(vec.size());         \
    for (const auto& value : vec) {           \
      *proto_field->Add() = value;            \
    }                                         \
  } while (0)

// Check if the block id is out of range.
bool check_block_ids(const std::vector<uint64_t>& block_ids,
                     int64_t block_num) {
  return std::all_of(
      block_ids.begin(), block_ids.end(), [block_num](uint64_t block_id) {
        return block_id < block_num;
      });
}

// Merge the source and destination block ids into a single block when both are
// consecutive.
void merge_block_ids(const std::vector<uint64_t>& src_blocks,
                     const std::vector<uint64_t>& dst_blocks,
                     std::vector<uint64_t>& merged_src_blocks,
                     std::vector<uint64_t>& merged_dst_blocks,
                     std::vector<uint64_t>& block_lengths) {
  // Create an index array and sort it based on the values of src blocks.
  size_t block_num = src_blocks.size();
  std::vector<uint64_t> indices(block_num);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(), indices.end(), [&src_blocks](uint64_t i, uint64_t j) {
        return src_blocks[i] < src_blocks[j];
      });

  // Generate sorted src blocks and dst blocks.
  std::vector<uint64_t> sorted_src_blocks;
  std::vector<uint64_t> sorted_dst_blocks;
  sorted_src_blocks.reserve(block_num);
  sorted_dst_blocks.reserve(block_num);
  for (auto id : indices) {
    sorted_src_blocks.emplace_back(src_blocks[id]);
    sorted_dst_blocks.emplace_back(dst_blocks[id]);
  }

  // Obtain continuous blocks.
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

static const std::unordered_map<HcclDataType, int64_t> DATA_TYPE_SIZE = {
    {HCCL_DATA_TYPE_INT8, 1},
    {HCCL_DATA_TYPE_INT16, 2},
    {HCCL_DATA_TYPE_INT32, 4},
    {HCCL_DATA_TYPE_FP16, 2},
    {HCCL_DATA_TYPE_FP32, 4},
    {HCCL_DATA_TYPE_INT64, 8},
    {HCCL_DATA_TYPE_UINT64, 8},
    {HCCL_DATA_TYPE_UINT8, 1},
    {HCCL_DATA_TYPE_UINT16, 2},
    {HCCL_DATA_TYPE_UINT32, 4},
    {HCCL_DATA_TYPE_FP64, 8},
    {HCCL_DATA_TYPE_BFP16, 2},
    {HCCL_DATA_TYPE_INT128, 16},
};

}  // namespace

HcclTransfer::HcclTransfer(const std::string& addr, int32_t device_id)
    : addr_(addr), device_id_(device_id) {
  aclrtSetDevice(device_id);
  aclrtCreateStream(&stream_);
  rpc_thread_pool_ = std::make_shared<ThreadPool>();
  hccl_thread_pool_ = std::make_shared<ThreadPool>();
}

HcclTransfer::~HcclTransfer() {
  if (has_initialized_) {
    server_.Stop(0);
    server_.Join();
  }
  aclrtDestroyStream(stream_);
  aclrtResetDevice(device_id_);
}

bool HcclTransfer::initialize(int32_t port) {
  service_ = std::make_shared<HcclTransferService>(this);
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return false;
  }

  // Start the server.
  brpc::ServerOptions options;
  if (server_.Start(port, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server";
    return false;
  }

  has_initialized_ = true;
  return true;
}

uint32_t HcclTransfer::register_memory(const std::vector<void*>& mem_addrs,
                                       const std::vector<int64_t>& shape,
                                       HcclDataType data_type) {
  if (shape.size() <= 1) {
    LOG(ERROR) << "The dimensions of the shape (" << shape.size()
               << ") must be greater than 1.";
    return -1;
  }
  memory_vec_.emplace_back(mem_addrs, shape, data_type);
  return memory_vec_.size() - 1;
}

bool HcclTransfer::create_comm_domain(const std::string& remote_addr,
                                      HcclRootInfo* root_info) {
  auto it = comm_map_.find(remote_addr);
  if (it == comm_map_.end()) {
    // The transmission of the memory between NPUs requires the creation of a
    // communication group with a world size of 2. We assume that the rank of
    // the instance that initiates the communication domain creation is 0, and
    // the rank of the peer instance is 1. In subsequent computations, the peer
    // rank will be calculated as 1 - rank.
    uint32_t rank = 1;
    // A root info of nullptr means that this instance is the initiator of the
    // communication domain creation. This instance needs to get the root info
    // and send it to the peer via RPC.
    if (root_info == nullptr) {
      aclrtSetDevice(device_id_);
      auto hccl_status = HcclGetRootInfo(&root_info_);
      CHECK_EQ(hccl_status, HCCL_SUCCESS)
          << "HCCL get root info failed. hccl_status : " << hccl_status;
      proto::HcclTransferService_Stub* stub = create_rpc_channel(remote_addr);
      rpc_thread_pool_->schedule([this, stub]() {
        proto::CommInfo proto_comm_info;
        proto_comm_info.set_addr(addr_);
        proto_comm_info.set_root_info(root_info_.internal,
                                      sizeof(root_info_.internal));
        proto::Status status;
        brpc::Controller cntl;
        stub->InitComm(&cntl, &proto_comm_info, &status, nullptr);
        if (cntl.Failed() || !status.ok()) {
          LOG(ERROR) << "InitComm failed, " << cntl.ErrorText();
        }
      });
      root_info = &root_info_;
      // Communication domain creation initiator, set the rank to 0.
      rank = 0;
    }

    aclrtSetDevice(device_id_);
    HcclComm comm;
    // Create a communication domain with a world size of 2.
    auto hccl_result =
        HcclCommInitRootInfo(/*nRanks*/ 2, root_info, rank, &comm);
    if (hccl_result != HCCL_SUCCESS) {
      LOG(ERROR) << "HcclCommInitRootInfo failed, hccl_result : " << hccl_result
                 << ", remote_addr : " << remote_addr;
      return false;
    }

    LOG(INFO) << "HcclCommInitRootInfo success, remote_addr : " << remote_addr;
    // Save the communication domain and its corresponding rank to the map for
    // subsequent use.
    comm_map_[remote_addr] = comm;
    rank_map_[remote_addr] = rank;
  }
  return true;
}

bool HcclTransfer::push_memory_blocks(const std::string& remote_addr,
                                      uint32_t src_mem_id,
                                      const std::vector<uint64_t>& src_blocks,
                                      uint32_t dst_mem_id,
                                      const std::vector<uint64_t>& dst_blocks,
                                      const std::vector<int64_t>& layer_ids) {
  auto it = comm_map_.find(remote_addr);
  if (it == comm_map_.end()) {
    LOG(ERROR) << "The communication domain does not exist! Remote addr : "
               << remote_addr;
    return false;
  }

  if (src_mem_id >= memory_vec_.size()) {
    LOG(ERROR) << "Source memory id out of range.";
    return false;
  }
  auto& mem_desc = memory_vec_[src_mem_id];
  int64_t block_num = mem_desc.shape[0];
  if (!check_block_ids(src_blocks, block_num)) {
    LOG(ERROR) << "Source block ids out of range.";
    return false;
  }

  // Merge consecutive block ids to improve transmission efficiency.
  std::vector<uint64_t> merged_src_blocks;
  std::vector<uint64_t> merged_dst_blocks;
  std::vector<uint64_t> block_lengths;
  merge_block_ids(src_blocks,
                  dst_blocks,
                  merged_src_blocks,
                  merged_dst_blocks,
                  block_lengths);

  auto stub = create_rpc_channel(remote_addr);
  // Send block information to the peer instance via RPC.
  rpc_thread_pool_->schedule(
      [this, stub, dst_mem_id, merged_dst_blocks, block_lengths, layer_ids]() {
        proto::BlockInfo proto_block_info;
        proto_block_info.set_addr(addr_);
        proto_block_info.set_mem_id(dst_mem_id);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_blocks(),
                            merged_dst_blocks);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_block_lengths(),
                            block_lengths);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_layer_ids(), layer_ids);
        proto::Status status;
        brpc::Controller cntl;
        stub->RecvBlocks(&cntl, &proto_block_info, &status, nullptr);
        if (cntl.Failed() || !status.ok()) {
          LOG(ERROR) << "RecvBlocks failed, " << cntl.ErrorText();
        }
      });

  auto& comm = it->second;
  uint32_t rank = rank_map_.find(remote_addr)->second;
  auto data_type = mem_desc.data_type;
  auto data_size_it = DATA_TYPE_SIZE.find(data_type);
  if (data_size_it == DATA_TYPE_SIZE.end()) {
    LOG(ERROR) << "Unsupport data type : " << data_type;
    return false;
  }
  int64_t data_size = data_size_it->second;
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < mem_desc.shape.size(); ++i) {
    count_per_block *= mem_desc.shape[i];
  }
  int64_t size_per_block = count_per_block * data_size;

  std::vector<int64_t> addr_ids;
  if (layer_ids.size() == 0) {
    addr_ids.resize(mem_desc.mem_addrs.size());
    std::iota(addr_ids.begin(), addr_ids.end(), 0);
  } else {
    addr_ids = layer_ids;
  }

  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();
  hccl_thread_pool_->schedule([&]() mutable {
    aclrtSetDevice(device_id_);
    for (auto addr_id : addr_ids) {
      void* mem_addr = mem_desc.mem_addrs[addr_id];
      for (size_t i = 0; i < merged_src_blocks.size(); ++i) {
        uint64_t block_id = merged_src_blocks[i];
        uint64_t block_length = block_lengths[i];
        int64_t bias = block_id * size_per_block;
        auto hccl_result = HcclSend((char*)mem_addr + bias,
                                    count_per_block * block_length,
                                    data_type,
                                    1 - rank,
                                    comm,
                                    stream_);
        if (hccl_result != HCCL_SUCCESS) {
          LOG(ERROR) << "HcclSend failed, result : " << hccl_result;
          promise.set_value(false);
          break;
        }
      }
    }
    aclrtSynchronizeStream(stream_);
    promise.set_value(true);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Push blocks failed!";
  }

  return result;
}

bool HcclTransfer::pull_memory_blocks(const std::string& remote_addr,
                                      uint32_t src_mem_id,
                                      const std::vector<uint64_t>& src_blocks,
                                      uint32_t dst_mem_id,
                                      const std::vector<uint64_t>& dst_blocks,
                                      const std::vector<int64_t>& layer_ids) {
  auto it = comm_map_.find(remote_addr);
  if (it == comm_map_.end()) {
    LOG(ERROR) << "The communication domain does not exist! Remote addr : "
               << remote_addr;
    return false;
  }

  if (dst_mem_id >= memory_vec_.size()) {
    LOG(ERROR) << "Destination memory id out of range.";
    return false;
  }
  auto& mem_desc = memory_vec_[dst_mem_id];
  int64_t block_num = mem_desc.shape[0];
  if (!check_block_ids(dst_blocks, block_num)) {
    LOG(ERROR) << "Destination block ids out of range.";
    return false;
  }

  // Merge consecutive block ids to improve transmission efficiency.
  std::vector<uint64_t> merged_src_blocks;
  std::vector<uint64_t> merged_dst_blocks;
  std::vector<uint64_t> block_lengths;
  merge_block_ids(src_blocks,
                  dst_blocks,
                  merged_src_blocks,
                  merged_dst_blocks,
                  block_lengths);

  auto stub = create_rpc_channel(remote_addr);
  // Send block information to the peer instance via RPC.
  rpc_thread_pool_->schedule(
      [this, stub, src_mem_id, merged_src_blocks, block_lengths, layer_ids]() {
        proto::BlockInfo proto_block_info;
        proto_block_info.set_addr(addr_);
        proto_block_info.set_mem_id(src_mem_id);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_blocks(),
                            merged_src_blocks);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_block_lengths(),
                            block_lengths);
        ADD_VECTOR_TO_PROTO(proto_block_info.mutable_layer_ids(), layer_ids);
        proto::Status status;
        brpc::Controller cntl;
        stub->SendBlocks(&cntl, &proto_block_info, &status, nullptr);
        if (cntl.Failed() || !status.ok()) {
          LOG(ERROR) << "SendBlocks failed, " << cntl.ErrorText();
        }
      });

  auto& comm = it->second;
  uint32_t rank = rank_map_.find(remote_addr)->second;
  auto data_type = mem_desc.data_type;
  auto data_size_it = DATA_TYPE_SIZE.find(data_type);
  if (data_size_it == DATA_TYPE_SIZE.end()) {
    LOG(ERROR) << "Unsupport data type : " << data_type;
    return false;
  }
  int64_t data_size = data_size_it->second;
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < mem_desc.shape.size(); ++i) {
    count_per_block *= mem_desc.shape[i];
  }
  int64_t size_per_block = count_per_block * data_size;

  std::vector<int64_t> addr_ids;
  if (layer_ids.size() == 0) {
    addr_ids.resize(mem_desc.mem_addrs.size());
    std::iota(addr_ids.begin(), addr_ids.end(), 0);
  } else {
    addr_ids = layer_ids;
  }

  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();
  hccl_thread_pool_->schedule([&]() mutable {
    aclrtSetDevice(device_id_);
    for (auto addr_id : addr_ids) {
      void* mem_addr = mem_desc.mem_addrs[addr_id];
      for (size_t i = 0; i < merged_src_blocks.size(); ++i) {
        uint64_t block_id = merged_src_blocks[i];
        uint64_t block_length = block_lengths[i];
        int64_t bias = block_id * size_per_block;
        auto hccl_result = HcclRecv((char*)mem_addr + bias,
                                    count_per_block * block_length,
                                    data_type,
                                    1 - rank,
                                    comm,
                                    stream_);
        if (hccl_result != HCCL_SUCCESS) {
          LOG(ERROR) << "HcclRecv failed, result : " << hccl_result;
          promise.set_value(false);
          break;
        }
      }
    }
    aclrtSynchronizeStream(stream_);
    promise.set_value(true);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Pull blocks failed!";
  }

  return result;
}

bool HcclTransfer::recv_blocks(const std::string& remote_addr,
                               uint32_t mem_id,
                               const std::vector<uint64_t>& blocks,
                               const std::vector<uint64_t>& block_lengths,
                               const std::vector<int64_t>& layer_ids) {
  auto it = comm_map_.find(remote_addr);
  if (it == comm_map_.end()) {
    LOG(ERROR) << "The communication domain does not exist! Remote addr : "
               << remote_addr;
    return false;
  }

  if (mem_id >= memory_vec_.size()) {
    LOG(ERROR) << "Destination memory id out of range.";
    return false;
  }
  auto& mem_desc = memory_vec_[mem_id];
  int64_t block_num = mem_desc.shape[0];
  if (!check_block_ids(blocks, block_num)) {
    LOG(ERROR) << "Destination block ids out of range.";
    return false;
  }

  auto data_type = mem_desc.data_type;
  auto data_size_it = DATA_TYPE_SIZE.find(data_type);
  if (data_size_it == DATA_TYPE_SIZE.end()) {
    LOG(ERROR) << "data_type : " << data_type;
    return false;
  }
  int64_t data_size = data_size_it->second;
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < mem_desc.shape.size(); ++i) {
    count_per_block *= mem_desc.shape[i];
  }
  int64_t size_per_block = count_per_block * data_size;

  auto& comm = it->second;
  uint32_t rank = rank_map_.find(remote_addr)->second;

  std::vector<int64_t> addr_ids;
  if (layer_ids.size() == 0) {
    addr_ids.resize(mem_desc.mem_addrs.size());
    std::iota(addr_ids.begin(), addr_ids.end(), 0);
  } else {
    addr_ids = layer_ids;
  }

  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();
  hccl_thread_pool_->schedule([&]() mutable {
    aclrtSetDevice(device_id_);
    for (auto addr_id : addr_ids) {
      void* mem_addr = mem_desc.mem_addrs[addr_id];
      for (size_t i = 0; i < blocks.size(); ++i) {
        uint64_t block_id = blocks[i];
        uint64_t block_length = block_lengths[i];
        int64_t bias = block_id * size_per_block;
        auto hccl_result = HcclRecv((char*)mem_addr + bias,
                                    count_per_block * block_length,
                                    data_type,
                                    1 - rank,
                                    comm,
                                    stream_);
        if (hccl_result != HCCL_SUCCESS) {
          LOG(ERROR) << "HcclRecv failed, result : " << hccl_result;
          promise.set_value(false);
          break;
        }
      }
    }
    aclrtSynchronizeStream(stream_);
    promise.set_value(true);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Recv blocks failed!";
  }

  return result;
}

bool HcclTransfer::send_blocks(const std::string& remote_addr,
                               uint32_t mem_id,
                               const std::vector<uint64_t>& blocks,
                               const std::vector<uint64_t>& block_lengths,
                               const std::vector<int64_t>& layer_ids) {
  auto it = comm_map_.find(remote_addr);
  if (it == comm_map_.end()) {
    LOG(ERROR) << "The communication domain does not exist! Remote addr : "
               << remote_addr;
    return false;
  }

  if (mem_id >= memory_vec_.size()) {
    LOG(ERROR) << "Source memory id out of range.";
    return false;
  }
  auto& mem_desc = memory_vec_[mem_id];
  int64_t block_num = mem_desc.shape[0];
  if (!check_block_ids(blocks, block_num)) {
    LOG(ERROR) << "Source block ids out of range.";
    return false;
  }

  auto data_type = mem_desc.data_type;
  auto data_size_it = DATA_TYPE_SIZE.find(data_type);
  if (data_size_it == DATA_TYPE_SIZE.end()) {
    LOG(ERROR) << "data_type : " << data_type;
    return false;
  }
  int64_t data_size = data_size_it->second;
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < mem_desc.shape.size(); ++i) {
    count_per_block *= mem_desc.shape[i];
  }
  int64_t size_per_block = count_per_block * data_size;

  auto& comm = it->second;
  uint32_t rank = rank_map_.find(remote_addr)->second;

  std::vector<int64_t> addr_ids;
  if (layer_ids.size() == 0) {
    addr_ids.resize(mem_desc.mem_addrs.size());
    std::iota(addr_ids.begin(), addr_ids.end(), 0);
  } else {
    addr_ids = layer_ids;
  }

  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();
  hccl_thread_pool_->schedule([&]() mutable {
    aclrtSetDevice(device_id_);
    for (auto addr_id : addr_ids) {
      void* mem_addr = mem_desc.mem_addrs[addr_id];
      for (size_t i = 0; i < blocks.size(); ++i) {
        uint64_t block_id = blocks[i];
        uint64_t block_length = block_lengths[i];
        int64_t bias = block_id * size_per_block;
        auto hccl_result = HcclSend((char*)mem_addr + bias,
                                    count_per_block * block_length,
                                    data_type,
                                    1 - rank,
                                    comm,
                                    stream_);
        if (hccl_result != HCCL_SUCCESS) {
          LOG(ERROR) << "HcclSend failed, result : " << hccl_result;
          promise.set_value(false);
          break;
        }
      }
    }
    aclrtSynchronizeStream(stream_);
    promise.set_value(true);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Send blocks failed!";
  }

  return result;
}

proto::HcclTransferService_Stub* HcclTransfer::create_rpc_channel(
    const std::string& remote_addr) {
  auto it = stub_map_.find(remote_addr);
  if (it == stub_map_.end()) {
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

    proto::HcclTransferService_Stub* stub =
        new proto::HcclTransferService_Stub(channel);
    return stub;
  }

  return it->second;
}

}  // namespace hccl_transfer