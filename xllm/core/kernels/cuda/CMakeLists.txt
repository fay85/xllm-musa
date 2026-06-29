include(cc_library)

# CUTLASS headers are provided by root CMakeLists.txt via submodule (third_party/cutlass)
include_directories(${CMAKE_CURRENT_LIST_DIR})

#
# CUDA feature detection
#
if(XLLM_TORCH_MUSA)
  message(STATUS "Skipping CUTLASS SM90+ libs for XLLM_TORCH_MUSA")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  # SM90+ (Hopper/Blackwell) scaled matmul support
  add_compile_definitions(
    ENABLE_SCALED_MM_SM90=1
    ENABLE_SCALED_MM_SM100=1
    ENABLE_SCALED_MM_SM120=1
  )
endif()

if(NOT XLLM_TORCH_MUSA AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  add_compile_definitions(ENABLE_FP8)
endif()

set(CUDA_HEADER_FILES
  arch_condition.h
  attention_runner.h
  cuda_ops_api.h
  gdn_ops.h
  mate_gdn_ops.h
  cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
  cutlass_w8a8/c3x/scaled_mm.cuh
  cutlass_w8a8/c3x/sm120_fp8_kernel_configs.cuh
  cutlass_w8a8/c3x/sm120_fp8_dispatch_policy.hpp
  cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh
  fp8_quant_utils.cuh
  global_capture_instance.h
  llm_decode_metadata_update.h
  piecewise_graphs.h
  topk_last_dim.cuh
  type_convert.cuh
  musa_tvmffi_stream.h
  utils.h
  device_utils.cuh
  xattention/xattention_ops_api.h
  moe/moe_topk.cuh
  moe/moe_topk_sigmoid_kernels.cuh
  moe/moe_topk_softmax_kernels.cuh
)

#
# Per-SM-architecture CUTLASS libraries.
#
# CUTLASS template instantiations are extremely expensive to compile. Splitting each SM generation
# into its own static library improves incremental build times — changing one SM's CUTLASS code
# only recompiles that library, not all CUTLASS files.

set(_CUTLASS_SM_LIBS "")

if(NOT XLLM_TORCH_MUSA AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  # --- SM90 (Hopper) ---
  cc_library(
    NAME cutlass_sm90
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm90.cu
    DEPS
      torch
      tvm_ffi
  )
  set_target_properties(cutlass_sm90 PROPERTIES CUDA_ARCHITECTURES 90a)

  # --- SM100 (Blackwell) ---
  cc_library(
    NAME cutlass_sm100
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm100.cu
    DEPS
      torch
      tvm_ffi
  )
  set_target_properties(cutlass_sm100 PROPERTIES CUDA_ARCHITECTURES 100a)

  # --- SM120 ---
  cc_library(
    NAME cutlass_sm120
    SRCS
      cutlass_w8a8/c3x/scaled_mm_sm120_fp8.cu
      cutlass_w8a8/scaled_mm_c3x_sm120.cu
    DEPS
      torch
      tvm_ffi
  )
  set_target_properties(cutlass_sm120 PROPERTIES CUDA_ARCHITECTURES 120a)

  set(_CUTLASS_SM_LIBS :cutlass_sm90 :cutlass_sm100 :cutlass_sm120)
endif()

# Keep source list explicit to avoid accidentally compiling test files into cuda_kernels.
# NOTE: SM-specific CUTLASS sources are compiled via the per-arch libraries above.
set(CUDA_SOURCE_FILES
  activation.cu
  gdn_ops.cpp
  mate_gdn_prefill.cpp
  mate_gdn_decode.cpp
  air_log_softmax_last_dim.cu
  attention_runner.cpp
  batch_decode.cpp
  fa3_fwd.cpp
  fa3_metadata.cpp
  batch_prefill.cpp
  batch_chunked_prefill.cpp
  block_copy.cu
  cutlass_extensions/common.cpp
  cutlass_w8a8/scaled_mm_entry.cu
  fp8_quant.cu
  fp8_scaled_matmul.cpp
  fp8_scaled_quantize.cpp
  fused_qknorm_rope.cu
  fused_gdn_decode.cu
  gated_rms_norm.cu
  gdn_conv1d_update.cu
  gdn_gating.cu
  gdn_split_fused.cu
  gemma_norm.cu
  global_capture_instance.cpp
  llm_decode_metadata_update.cu
  matmul.cpp
  norm.cu
  piecewise_graphs.cpp
  rec_beam_search.cu
  reshape_paged_cache.cu
  rope.cu
  random_sample.cpp
  musa_tvmffi_stream.cpp
  utils.cpp
  xattention/beam_search.cpp
  xattention/cache_select.cu
  xattention/decoder_reshape_and_cache.cu
  xattention/lse_combine.cu
  xattention/prefill_reshape_and_cache.cu
  moe/moe_fused_topk.cu
  moe/fused_moe.cpp
  kernel_musa_compat.cpp
)


set(_XLLM_CUTLASS_HDRS
  cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
  cutlass_w8a8/c3x/scaled_mm.cuh
  cutlass_w8a8/c3x/sm120_fp8_kernel_configs.cuh
  cutlass_w8a8/c3x/sm120_fp8_dispatch_policy.hpp
  cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh
  cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh
)
set(_XLLM_CUTLASS_SRCS
  cutlass_extensions/common.cpp
  cutlass_w8a8/scaled_mm_entry.cu
)
if(XLLM_TORCH_MUSA)
  foreach(_item IN LISTS _XLLM_CUTLASS_HDRS)
    list(REMOVE_ITEM CUDA_HEADER_FILES ${_item})
  endforeach()
  foreach(_item IN LISTS _XLLM_CUTLASS_SRCS)
    list(REMOVE_ITEM CUDA_SOURCE_FILES ${_item})
  endforeach()
  set(_CUDA_KERNELS_MUSA_DEPS musa musart)
else()
  set(_CUDA_KERNELS_MUSA_DEPS "")
endif()

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${CUDA_HEADER_FILES}
  SRCS
    ${CUDA_SOURCE_FILES}
  COPTS
    ${_XLLM_MUSIFY_COPTS}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
    ${_CUTLASS_SM_LIBS}
    ${_CUDA_KERNELS_MUSA_DEPS}
)

# Keep CUDA benchmarks behind a plain CMake cache option instead of adding
# another dedicated argument in `setup.py`.
#
# Why:
# - `setup.py` already forwards arbitrary CMake definitions from the
#   `CMAKE_ARGS` environment variable, so adding a new Python-side flag would
#   duplicate the existing mechanism.
# - Most normal builds do not need benchmark binaries, so the default stays
#   `OFF` and users enable it only when they really want to build them.
#
# Simplest way to build benchmarks:
# 1. From the repo root, run:
#      CMAKE_ARGS="-DBUILD_CUDA_BENCHMARKS=ON" python setup.py build test
# 2. This is enough: `build test` builds `all_tests`, and when
#    `BUILD_CUDA_BENCHMARKS=ON` the benchmark binaries are added to
#    `all_tests` as build dependencies.
# 3. If the build directory was configured before with this option `OFF`, the
#    old value is cached. Re-run the same command, or delete `CMakeCache.txt`
#    / the build directory before rebuilding.
option(BUILD_CUDA_BENCHMARKS "Build CUDA kernel benchmarks" OFF)
if(BUILD_CUDA_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
