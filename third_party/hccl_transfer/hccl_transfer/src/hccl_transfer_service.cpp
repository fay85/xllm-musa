#include "hccl_transfer_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include "hccl_transfer.h"

namespace hccl_transfer {

HcclTransferService::HcclTransferService(HcclTransfer* hccl_transfer)
    : hccl_transfer_(hccl_transfer) {};

void HcclTransferService::InitComm(
    ::google::protobuf::RpcController* controller,
    const proto::CommInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  HcclRootInfo root_info;
  std::memcpy(root_info.internal,
              request->root_info().data(),
              sizeof(root_info.internal));
  std::string remote_addr(request->addr());
  bool result = hccl_transfer_->create_comm_domain(remote_addr, &root_info);

  response->set_ok(result);
}

void HcclTransferService::RecvBlocks(
    ::google::protobuf::RpcController* controller,
    const proto::BlockInfo* request,
    proto::Status* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  std::vector<uint64_t> blocks(request->blocks().begin(),
                               request->blocks().end());
  std::vector<uint64_t> block_lengths(request->block_lengths().begin(),
                                      request->block_lengths().end());
  std::vector<int64_t> layer_ids(request->layer_ids().begin(),
                                 request->layer_ids().end());
  bool result = hccl_transfer_->recv_blocks(
      remote_addr, request->mem_id(), blocks, block_lengths, layer_ids);

  response->set_ok(true);
}

void HcclTransferService::SendBlocks(
    ::google::protobuf::RpcController* controller,
    const proto::BlockInfo* request,
    proto::Status* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  std::string remote_addr(request->addr());
  std::vector<uint64_t> blocks(request->blocks().begin(),
                               request->blocks().end());
  std::vector<uint64_t> block_lengths(request->block_lengths().begin(),
                                      request->block_lengths().end());
  std::vector<int64_t> layer_ids(request->layer_ids().begin(),
                                 request->layer_ids().end());
  bool result = hccl_transfer_->send_blocks(
      remote_addr, request->mem_id(), blocks, block_lengths, layer_ids);

  response->set_ok(true);
}

}  // namespace hccl_transfer