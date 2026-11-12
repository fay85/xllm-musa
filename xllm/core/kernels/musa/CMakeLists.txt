include(cc_library)

include(GNUInstallDirs)

include(ProcessorCount)
ProcessorCount(_MUSA_AOT_DEFAULT_JOBS)
if(DEFINED ENV{MAX_JOBS})
  set(_MUSA_AOT_DEFAULT_JOBS "$ENV{MAX_JOBS}")
elseif(NOT _MUSA_AOT_DEFAULT_JOBS)
  set(_MUSA_AOT_DEFAULT_JOBS 1)
endif()
set(XLLM_MUSA_AOT_JOBS "${_MUSA_AOT_DEFAULT_JOBS}" CACHE STRING "Parallel jobs for MATE operator AOT")
if(NOT XLLM_MUSA_AOT_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "XLLM_MUSA_AOT_JOBS must be a positive integer")
endif()
if(NOT XLLM_MUSA_OPS_OUTPUT_DIRECTORY)
  message(FATAL_ERROR "XLLM_MUSA_OPS_OUTPUT_DIRECTORY must be set by the runtime target")
endif()
set(_MUSA_OPS_INCLUDE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_MUSA_OPS_HEADER "${_MUSA_OPS_INCLUDE_ROOT}/core/kernels/musa/musa_aot_config.h")
set(_MUSA_OPS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/aot/bundle.stamp")
set(_MUSA_OPS_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/aot/bundle.d")
set(_MUSA_OPS_ENV "${CMAKE_CURRENT_BINARY_DIR}/aot/build_environment.txt")
set(_MUSA_OPS_ENV_CONTENT "")
foreach(_env_name MATE_MUSA_ARCH_LIST MUSA_HOME MUSA_LIBDEVICE_PATH
    MUSA_REPLACE_IR TRITON_KERNEL_OVERRIDE TILELANG_REPLACE_MUSAC
    MATE_EXTRA_CFLAGS MATE_EXTRA_MUSAFLAGS MATE_EXTRA_LDFLAGS
    TILELANG_MCC_EXTRA_ARGS TILELANG_OPT_LEVEL ENABLE_MUSA_MMA DISABLE_SQMMA
    TRITON_DISABLE_LINE_INFO MUSA_ENABLE_LLC_OPT)
  string(APPEND _MUSA_OPS_ENV_CONTENT "${_env_name}=$ENV{${_env_name}}\n")
endforeach()
file(GENERATE OUTPUT "${_MUSA_OPS_ENV}" CONTENT "${_MUSA_OPS_ENV_CONTENT}")
# The depfile records toolchain/package inputs and bundle assets. Build flags
# are configure-time inputs, like the host/device compiler configuration.
add_custom_command(OUTPUT "${_MUSA_OPS_STAMP}"
  COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}"
    "${Python_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build.py"
    --output-dir "${XLLM_MUSA_OPS_OUTPUT_DIRECTORY}"
    --build-dir "${CMAKE_CURRENT_BINARY_DIR}/aot"
    --header "${_MUSA_OPS_HEADER}"
    --jobs "${XLLM_MUSA_AOT_JOBS}"
    --stamp "${_MUSA_OPS_STAMP}"
    --depfile "${_MUSA_OPS_DEPFILE}"
  BYPRODUCTS "${_MUSA_OPS_HEADER}"
  DEPFILE "${_MUSA_OPS_DEPFILE}"
  DEPENDS build.py aot_config.py aot_bundle.py triton_kernel/fused_moe.py
    "${_MUSA_OPS_ENV}"
  COMMENT "Build and verify the versioned MUSA attention, GDN and MoE bundle"
  VERBATIM)
add_custom_target(musa_aot_ops DEPENDS "${_MUSA_OPS_STAMP}")

add_custom_target(musa_aot_ops_check
  COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}"
    "${Python_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build.py"
    --output-dir "${XLLM_MUSA_OPS_OUTPUT_DIRECTORY}"
    --build-dir "${CMAKE_CURRENT_BINARY_DIR}/aot"
    --header "${_MUSA_OPS_HEADER}"
    --check-only
  COMMENT "Verify the existing MUSA operator bundle without building"
  VERBATIM)
if(BUILD_TESTING)
  add_test(NAME musa_aot_ops_check COMMAND "${CMAKE_COMMAND}" -E env
    "CXX=${CMAKE_CXX_COMPILER}" "${Python_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/build.py"
    --output-dir "${XLLM_MUSA_OPS_OUTPUT_DIRECTORY}"
    --build-dir "${CMAKE_CURRENT_BINARY_DIR}/aot"
    --header "${_MUSA_OPS_HEADER}" --check-only)
  set_tests_properties(musa_aot_ops_check PROPERTIES LABELS cpu RUN_SERIAL TRUE)
endif()

add_library(xllm_musa_ops INTERFACE)
add_library(xllm::musa_ops ALIAS xllm_musa_ops)
add_dependencies(xllm_musa_ops musa_aot_ops)
target_include_directories(xllm_musa_ops INTERFACE "${_MUSA_OPS_INCLUDE_ROOT}")
# Native installs keep libraries in libdir; wheels keep the bundle adjacent.
set(_MUSA_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_FULL_BINDIR}")
if(GENERATE_SO)
  set(_MUSA_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()
file(RELATIVE_PATH _MUSA_OPS_INSTALL_RELATIVE_DIR
  "${_MUSA_RUNTIME_INSTALL_DIR}" "${CMAKE_INSTALL_FULL_LIBDIR}/musa_ops")
target_compile_definitions(xllm_musa_ops INTERFACE
  "XLLM_MUSA_INSTALL_OPS_RELATIVE_DIR=\"${_MUSA_OPS_INSTALL_RELATIVE_DIR}\"")

install(DIRECTORY "${XLLM_MUSA_OPS_OUTPUT_DIRECTORY}/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/musa_ops"
  COMPONENT musa_ops
  PATTERN ".build.lock" EXCLUDE
  PATTERN "*.tmp" EXCLUDE)

# Host FFI bridges and native MUSA kernels.
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda/moe)

set(_TORCH_MUSA_KERNEL_HDRS
  musa_ops_api.h
  musa_tvmffi_stream.h
  gdn_ops.h
  mrope.h
  llm_decode_metadata_update.h
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_sigmoid_kernels.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_softmax_kernels.cuh
)

set(_TORCH_MUSA_KERNEL_SRCS
  attention_decode.cpp
  gdn_decode.cu
  gdn_prefill.cpp
  causal_conv1d.cu
  gemma_norm.cu
  activation.cu
  rope.cu
  mrope.cu
  gdn_state_commit.cu
  ../cuda/block_copy.cu
  ../cuda/norm.cu
  ../cuda/reshape_paged_cache.cu
  ../cuda/cuda_ops_library.cpp
  ../cuda/moe/moe_fused_topk.cu
  random_sample.cpp
  rejection_sample.cu
  replace_token.cu
  fused_qknorm_rope.cu
  # Block-wise FP8 GEMM through the source-built MATE BMM operator.
  fp8_block_gemm.cpp
  musa_moe_gemm.cpp
  musa_fused_moe_aot.cpp
  moe_align.cu
  moe_combine_indexed.cu
  ../cuda/moe/moe_compute_index.cu
  ../cuda/moe/moe_combine.cu
  # Fused per-token-group FP8 activation quant (MUSA-native intrinsics)
  fp8_act_quant.cu
  # Mate FFI helpers
  musa_tvmffi_stream.cpp
  llm_decode_metadata_update.cu
)

# Enable fast-math for fp8_act_quant.cu only, without changing
# floating-point flags for the other kernels in this library.
set_source_files_properties(
  fp8_act_quant.cu
  PROPERTIES
    COMPILE_OPTIONS "-ffast-math"
)

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${_TORCH_MUSA_KERNEL_HDRS}
  SRCS
    ${_TORCH_MUSA_KERNEL_SRCS}
  DEPS
    tvm_ffi
    OpenSSL::Crypto
    torch
    :util
    :platform
    musa
    musart
    xllm_musa_ops
)

# Host-side FFI bridges stay on C++; only .cu kernels use the MUSA compiler.
target_include_directories(cuda_kernels BEFORE PRIVATE
  "$<TARGET_PROPERTY:xllm_musa_ops,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_options(cuda_kernels PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:${_XLLM_MUSIFY_COPTS}>")

