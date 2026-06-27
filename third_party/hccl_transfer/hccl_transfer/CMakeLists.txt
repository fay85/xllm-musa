include(cc_library)

cc_library(
  NAME
    hccl_transfer
  HDRS
    include/concurrent_queue.h
    include/hccl_transfer.h
    include/hccl_transfer_service.h
    include/threadpool.h
  SRCS
    src/hccl_transfer.cpp
    src/hccl_transfer_service.cpp
    src/threadpool.cpp
  DEPS
    absl::synchronization
    ascendcl
    glog::glog
    hccl
    proto::proto_transfer_service
)
target_link_libraries(hccl_transfer PRIVATE brpc-static)

add_subdirectory(proto)