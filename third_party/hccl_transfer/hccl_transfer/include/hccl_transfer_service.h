#pragma once

#include "hccl_transfer_service.pb.h"

namespace hccl_transfer {

class HcclTransfer;

class HcclTransferService : public proto::HcclTransferService {
 public:
  HcclTransferService(HcclTransfer* hccl_transfer);

  virtual ~HcclTransferService() = default;

  virtual void InitComm(google::protobuf::RpcController* controller,
                        const proto::CommInfo* request,
                        proto::Status* response,
                        google::protobuf::Closure* done) override;

  virtual void RecvBlocks(google::protobuf::RpcController* controller,
                          const proto::BlockInfo* request,
                          proto::Status* response,
                          google::protobuf::Closure* done) override;

  virtual void SendBlocks(google::protobuf::RpcController* controller,
                          const proto::BlockInfo* request,
                          proto::Status* response,
                          google::protobuf::Closure* done) override;

 private:
  HcclTransfer* hccl_transfer_;
};

}  // namespace hccl_transfer