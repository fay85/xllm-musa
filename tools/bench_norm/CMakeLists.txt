include(cc_library)

# Bring the musamapping plugin flag into local scope (defined in xllm/CMakeLists
# but limited to that subdirectory). Without this the cc_binary below compiles
# bench_norm_cuda_graph.cpp as plain CXX, which fails because PyTorch torch/
# include headers (c10/cuda/*) are read raw and reference cudaError_t /
# cudaStream_t -- the musamapping plugin is what rewrites those to musa-prefixed
# variants and path-redirects c10/cuda/ to torch_musa/csrc/core/.
if(USE_MUSA)
  if(DEFINED ENV{MUSAMAPPING_PATH} AND NOT "$ENV{MUSAMAPPING_PATH}" STREQUAL "")
    set(_XLLM_MUSAMAPPING_PATH "$ENV{MUSAMAPPING_PATH}")
  elseif(DEFINED _XLLM_MUSA_HOME)
    set(_XLLM_MUSAMAPPING_PATH "${_XLLM_MUSA_HOME}/tools/musamapping")
  else()
    set(_XLLM_MUSAMAPPING_PATH "/usr/local/musa/tools/musamapping")
  endif()
  set(_XLLM_MUSIFY_COPTS
    "-fplugin=${_XLLM_MUSAMAPPING_PATH}/libMusaMapping.so"
    "-x" "musa"
    "-I."
  )
else()
  set(_XLLM_MUSIFY_COPTS "")
endif()

# Microbench: standalone executable that times xllm CUDA-MUSA rms_norm kernels
# at the shapes the Qwen3.5-27B production path hits. Built alongside xllm,
# linked against the same cuda_kernels static archive + torch/torch_musa.
cc_binary(
  NAME
    xllm_bench_norm
  SRCS
    bench_norm_cuda_graph.cpp
  COPTS
    ${_XLLM_MUSIFY_COPTS}
  DEPS
    cuda_kernels
    torch
)