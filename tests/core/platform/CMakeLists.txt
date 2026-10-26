include(cc_test)

if(USE_NPU OR USE_CUDA OR USE_ILU OR USE_DCU OR USE_MUSA)
  cc_test(
    NAME
      multi_platform_vmm_test
    SRCS
      shared_vmm_allocator_test.cpp
    DEPS
      :platform
      :config
      GTest::gtest_main
      glog::glog
  )
  if(USE_DCU)
    set_target_properties(multi_platform_vmm_test PROPERTIES LINKER_LANGUAGE HIP)
  endif()
endif()

if(USE_NPU)
  add_subdirectory(npu)
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()

cc_test(
  NAME
    graph_pool_ref_test
  SRCS
    graph_pool_ref_test.cpp
  DEPS
    GTest::gtest_main
    glog::glog
)

if(USE_MUSA)
  cc_test(
    NAME
      musa_graph_vmm_pool_test
    SRCS
      musa_graph_vmm_pool_test.cpp
    DEPS
      :platform
      :config
      :cuda_kernels
      torch
      torch_musa
      Python::Python
      GTest::gtest_main
      glog::glog
  )
endif()
