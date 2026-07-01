include(cc_library)

# Native USE_MUSA path (MTTOplib) — unchanged from upstream.
if(USE_MUSA)
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
endif()

# XLLM_TORCH_MUSA: CUDA sources compiled via mcc_wrapper + musamapping.
# All implementation lives under kernels/musa/; kernels/cuda/ stays identical to HEAD.
if(XLLM_TORCH_MUSA)
  include_directories(${CMAKE_CURRENT_LIST_DIR})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda/moe)

  set(_TORCH_MUSA_KERNEL_HDRS
    arch_condition.h
    attention_runner.h
    block_copy_api.h
    cuda_graph_compat.h
    cuda_ops_api.h
    device_utils.cuh
    fused_qknorm_rope_api.h
    gated_delta_fallback.h
    gdn_ops.h
    global_capture_instance.h
    llm_decode_metadata_update.h
    mate_gdn_ops.h
    musa_tvmffi_stream.h
    piecewise_graphs.h
    preserve_cuda_namespace.h
    topk_last_dim.cuh
    type_convert.cuh
    utils.h
    xattention_ops_api.h
    xattention/xattention_ops_api.h
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk.cuh
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_sigmoid_kernels.cuh
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_softmax_kernels.cuh
  )

  set(_TORCH_MUSA_KERNEL_SRCS
    activation.cu
    air_log_softmax_last_dim.cu
    attention_runner.cpp
    batch_decode.cpp
    batch_prefill.cpp
    batch_chunked_prefill.cpp
    block_copy.cu
    beam_search.cpp
    cache_select.cu
    fa3_fwd.cpp
    fa3_metadata.cpp
    fp8_quant.cu
    fp8_scaled_matmul.cpp
    fp8_scaled_quantize.cpp
    fused_qknorm_rope.cu
    fused_gdn_decode.cu
    gated_rms_norm.cu
    gdn_conv1d_update.cu
    gdn_gating.cu
    gdn_split_fused.cu
    gdn_ops.cpp
    gemma_norm.cu
    global_capture_instance.cpp
    kernel_musa_compat.cpp
    llm_decode_metadata_update.cu
    mate_gdn_prefill.cpp
    mate_gdn_decode.cpp
    matmul.cpp
    musa_tvmffi_stream.cpp
    norm.cu
    piecewise_graphs.cpp
    rec_beam_search.cu
    reshape_paged_cache.cu
    rope.cu
    random_sample.cpp
    utils.cpp
    xattention/decoder_reshape_and_cache.cu
    xattention/lse_combine.cu
    xattention/prefill_reshape_and_cache.cu
    moe/moe_fused_topk.cu
    moe/fused_moe.cpp
  )

  cc_library(
    NAME
      cuda_kernels
    HDRS
      ${_TORCH_MUSA_KERNEL_HDRS}
    SRCS
      ${_TORCH_MUSA_KERNEL_SRCS}
    COPTS
      ${_XLLM_MUSIFY_COPTS}
    DEPS
      tvm_ffi
      torch
      :util
      :platform
      musa
      musart
  )
endif()
