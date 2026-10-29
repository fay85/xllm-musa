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

# Indexer scoring is isolated from model loading and uses one MUSA device.
cc_test(
  NAME deepseek_v4_indexer_metadata_test
  SRCS deepseek_v4_indexer_metadata_test.cpp
  DEPS :cuda_kernels tvm_ffi musa_python torch_python Python::Python torch
       GTest::gtest_main glog::glog
)

# Cached Prefill Gate projection and routing numerical checks on MUSA.
cc_test(
  NAME deepseek_v4_prefill_gate_test
  SRCS deepseek_v4_prefill_gate_test.cpp
  DEPS :cuda_kernels tvm_ffi musa_python torch_python Python::Python torch
       GTest::gtest_main glog::glog
)

# Packed compressor layouts and request-local Graph projection equivalence.
cc_test(
  NAME deepseek_v4_joint_projection_test
  SRCS deepseek_v4_joint_projection_test.cpp
  DEPS :cuda_kernels tvm_ffi musa_python torch_python Python::Python torch
       GTest::gtest_main glog::glog
)
