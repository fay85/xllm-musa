include(cc_library)

file(GLOB_RECURSE MUSA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/cuda/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cuh"
)

file(GLOB MUSA_LOCAL_CPP "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB MUSA_CUDA_CPP "${CMAKE_CURRENT_LIST_DIR}/../cuda/*.cpp")
set(MUSA_SOURCE_FILES
  ${MUSA_LOCAL_CPP}
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/activation.cu
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/norm.cu
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/reshape_paged_cache.cu
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/rope.cu
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_fused_topk.cu
  ${MUSA_CUDA_CPP}
)

cc_library(
  NAME
    musa_kernels
  HDRS
    ${MUSA_HEADER_FILES}
  SRCS
    ${MUSA_SOURCE_FILES}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
    MTTOplib
)
