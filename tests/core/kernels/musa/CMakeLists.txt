# Copyright 2026 The xLLM Authors. All Rights Reserved.
include(cc_test)
include(pybind_extension)

pybind_extension(
  NAME musa_python_kernel_test_module
  SRCS musa_python_kernel_test_module.cpp
  DEPS :cuda_kernels :parallel_state :state_dict :platform musa_python torch_python Python::Module torch glog::glog
)
target_include_directories(musa_python_kernel_test_module PRIVATE
  "${PROJECT_SOURCE_DIR}/xllm"
)

# Small production-kernel checks for graph-replayed position/cache inputs.
pybind_extension(
  NAME v41_decode_graph_inputs_test_module
  SRCS v41_decode_graph_inputs_test_module.cpp
  DEPS :cuda_kernels musa_python torch_python Python::Module torch glog::glog
)
target_include_directories(v41_decode_graph_inputs_test_module PRIVATE
  "${PROJECT_SOURCE_DIR}/xllm"
)

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

# Isolated Attention head-layout and TP projection checks.
pybind_extension(
  NAME v41_attention_layout_test_module
  SRCS v41_attention_layout_test_module.cpp
  DEPS :cuda_kernels :util :platform tvm_ffi musa_python torch_python Python::Module torch glog::glog
)
target_include_directories(v41_attention_layout_test_module PRIVATE
  "${PROJECT_SOURCE_DIR}/xllm"
)

# Actual Attention cache writes and accepted-prefix transaction checks.
pybind_extension(
  NAME v41_prefix_checkpoint_test_module
  SRCS v41_prefix_checkpoint_test_module.cpp
  DEPS :cuda_kernels :util :platform tvm_ffi musa_python torch_python Python::Module torch glog::glog
)
target_include_directories(v41_prefix_checkpoint_test_module PRIVATE
  "${PROJECT_SOURCE_DIR}/xllm"
)
