# Copyright 2026 The xLLM Authors. All Rights Reserved.
include(cc_test)

# CPU tensors exercise ownership without running a MUSA kernel or model.
cc_test(
  NAME musa_tvmffi_lifetime_test
  SRCS musa_tvmffi_lifetime_test.cpp
  DEPS :cuda_kernels tvm_ffi musa_python torch_python Python::Python torch
       GTest::gtest_main glog::glog
)

cc_test(
  NAME
    deepseek_v4_hc_stream_test
  SRCS
    deepseek_v4_hc_stream_test.cpp
  DEPS
    :cuda_kernels
    tvm_ffi
    musa_python
    torch_python
    Python::Python
    torch
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    deepseek_v4_state_gather_test
  SRCS
    deepseek_v4_state_gather_test.cpp
  DEPS
    :cuda_kernels
    tvm_ffi
    musa_python
    torch_python
    Python::Python
    torch
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    deepseek_v4_mate_sparse_attention_test
  SRCS
    deepseek_v4_mate_sparse_attention_test.cpp
  DEPS
    :cuda_kernels
    tvm_ffi
    musa_python
    torch_python
    Python::Python
    torch
    GTest::gtest_main
    glog::glog
)
