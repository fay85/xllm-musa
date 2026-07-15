include(cc_test)

add_subdirectory(moe)
add_subdirectory(xattention)

cc_test(
  NAME
    activation_test
  SRCS
    activation_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)

option(XLLM_ENABLE_BLOCK_COPY_TEST
       "Build and register the expensive CUDA block_copy_test"
       OFF)

if(XLLM_ENABLE_BLOCK_COPY_TEST)
  cc_test(
    NAME
      block_copy_test
    SRCS
      block_copy_test.cpp
    DEPS
      :cuda_kernels
      torch
      GTest::gtest_main
      glog::glog
  )
endif()

cc_test(
  NAME
    fused_qknorm_rope_test
  SRCS
    fused_qknorm_rope_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    cutlass_scaled_mm_test
  SRCS
    cutlass_scaled_mm_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cutlass_scaled_mm_test PRIVATE brpc)

cc_test(
  NAME
    fp8_quant_test
  SRCS
    fp8_quant_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(fp8_quant_test PRIVATE brpc)

if(USE_MUSA)
  cc_test(
    NAME gemma_norm_test
    SRCS gemma_norm_test.cpp
    DEPS :cuda_kernels torch GTest::gtest_main glog::glog
  )
  cc_test(
    NAME fused_gdn_decode_test
    SRCS fused_gdn_decode_test.cpp
    DEPS :cuda_kernels torch GTest::gtest_main glog::glog
  )
  if(USE_MUSA)
    cc_test(
      NAME fused_gdn_decode_bench_test
      SRCS fused_gdn_decode_bench_test.cpp
      DEPS :cuda_kernels torch GTest::gtest_main glog::glog
    )
  endif()
  cc_test(
    NAME piecewise_graphs_test
    SRCS piecewise_graphs_test.cpp
    DEPS :cuda_kernels torch GTest::gtest_main glog::glog
  )
endif()
