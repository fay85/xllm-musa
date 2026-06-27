#!/usr/bin/env python3
from pathlib import Path

ROOT = Path("/workspace/xllm-git-master")

def patch_root_cmake():
    path = ROOT / "CMakeLists.txt"
    text = path.read_text()
    if "option(XLLM_TORCH_MUSA" not in text:
        needle = 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")\n\nlist(APPEND CMAKE_MODULE_PATH'
        repl = 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")\n\noption(XLLM_TORCH_MUSA "Build USE_CUDA path with torch_musa (mcc_wrapper as CUDA compiler)" OFF)\n\nlist(APPEND CMAKE_MODULE_PATH'
        if needle not in text:
            raise RuntimeError("option insertion point not found")
        text = text.replace(needle, repl)

    start = text.index("if(USE_CUDA)\n  message(STATUS \"TORCH_CUDA_ARCH_LIST:")
    end = text.index("endif()\n\nif(USE_ILU)", start)
    new_block = r'''if(USE_CUDA)
  message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
  add_definitions(-DUSE_CUDA)
  add_compile_definitions(TORCH_CUDA=1)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  include_directories($ENV{PYTHON_INCLUDE_PATH})

  if(XLLM_TORCH_MUSA)
    add_compile_definitions(XLLM_TORCH_MUSA=1)
    message(STATUS "XLLM_TORCH_MUSA enabled: CUDA graph path via mcc_wrapper + musamapping")

    if(DEFINED ENV{MUSA_HOME} AND NOT "$ENV{MUSA_HOME}" STREQUAL "")
      set(_XLLM_MUSA_HOME "$ENV{MUSA_HOME}")
    else()
      set(_XLLM_MUSA_HOME "/usr/local/musa")
    endif()
    if(DEFINED ENV{PYTORCH_MUSA_INSTALL_PATH} AND NOT "$ENV{PYTORCH_MUSA_INSTALL_PATH}" STREQUAL "")
      set(_XLLM_TORCH_MUSA_ROOT "$ENV{PYTORCH_MUSA_INSTALL_PATH}")
    else()
      set(_XLLM_TORCH_MUSA_ROOT "/usr/local/lib/python3.10/dist-packages/torch_musa")
    endif()
    if(DEFINED ENV{MUSAMAPPING_PATH} AND NOT "$ENV{MUSAMAPPING_PATH}" STREQUAL "")
      set(_XLLM_MUSAMAPPING_PATH "$ENV{MUSAMAPPING_PATH}")
    else()
      set(_XLLM_MUSAMAPPING_PATH "${_XLLM_MUSA_HOME}/tools/musamapping")
    endif()

    list(APPEND CMAKE_MODULE_PATH "${_XLLM_MUSAMAPPING_PATH}/cmake/Modules")

    execute_process(
      COMMAND tvm-ffi-config --includedir
      OUTPUT_VARIABLE _XLLM_TVM_FFI_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND tvm-ffi-config --libdir
      OUTPUT_VARIABLE _XLLM_TVM_FFI_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    include_directories(
      ${_XLLM_MUSA_HOME}/include
      ${_XLLM_TVM_FFI_INCLUDE_DIR}
      ${_XLLM_TORCH_MUSA_ROOT}/share/torch_musa_codegen
    )
    include_directories(SYSTEM
      ${_XLLM_TORCH_MUSA_ROOT}/share/generated_cuda_compatible/include
      ${_XLLM_TORCH_MUSA_ROOT}/share/generated_cuda_compatible/include/torch/csrc/api/include
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    )

    link_directories(
      ${_XLLM_MUSA_HOME}/lib
      ${_XLLM_TVM_FFI_LIB_DIR}
      ${_XLLM_TORCH_MUSA_ROOT}/lib
      /opt/intel/oneapi/mkl/lib/intel64
      $ENV{PYTHON_LIB_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/lib
    )

    option(CUDA_DEV_MODE "Use -O1 instead of -O3 for faster CUDA compilation during development" OFF)
    if(CUDA_DEV_MODE)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1")
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS
      " -DCCCL_DISABLE_NVFP8_SUPPORT"
      " -fplugin=${_XLLM_MUSAMAPPING_PATH}/libMusaMapping.so"
      " -x musa --offload-arch=mp_31")
  else()
    include_directories(SYSTEM
        $ENV{PYTORCH_INSTALL_PATH}/include
        $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    )

    link_directories(
      $ENV{PYTHON_LIB_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/lib
      $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib64
    )

    option(CUDA_DEV_MODE "Use -O1 instead of -O3 for faster CUDA compilation during development" OFF)
    if(CUDA_DEV_MODE)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1")
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()

    string(APPEND CMAKE_CUDA_FLAGS
      " -U__CUDA_NO_HALF_OPERATORS__"
      " -U__CUDA_NO_HALF_CONVERSIONS__"
      " -U__CUDA_NO_HALF2_OPERATORS__"
      " -U__CUDA_NO_BFLOAT16_CONVERSIONS__"
      " --use_fast_math"
      " -Xfatbin -compress-all")

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
      string(APPEND CMAKE_CUDA_FLAGS " --threads 0")
    endif()

    execute_process(COMMAND python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)" OUTPUT_VARIABLE CUDNN_PYTHON_PATH)
    get_filename_component(CUDNN_ROOT_DIR "${CUDNN_PYTHON_PATH}" DIRECTORY)
    link_directories(
        ${CUDNN_ROOT_DIR}/lib64
        ${CUDNN_ROOT_DIR}/lib
    )
  endif()

  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
'''
    text = text[:start] + new_block + text[end:]
    path.write_text(text)
    print("patched", path)


def patch_xllm_cmake():
    path = ROOT / "xllm/CMakeLists.txt"
    text = path.read_text()
    if "XLLM_TORCH_MUSA" not in text:
        insert = '''if(USE_CUDA AND XLLM_TORCH_MUSA)
  if(DEFINED ENV{MUSA_HOME} AND NOT "$ENV{MUSA_HOME}" STREQUAL "")
    set(_XLLM_MUSA_HOME "$ENV{MUSA_HOME}")
  else()
    set(_XLLM_MUSA_HOME "/usr/local/musa")
  endif()
  set(CMAKE_CXX_COMPILER "${_XLLM_MUSA_HOME}/tools/musamapping/mcc_wrapper")
endif()

'''
        text = text.replace("include(cc_binary)\n", insert + "include(cc_binary)\n")
    text = text.replace(
        "if (USE_MUSA)\n  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)\nendif()",
        "if (USE_MUSA)\n  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)\nelseif(XLLM_TORCH_MUSA)\n  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)\nendif()",
    )
    path.write_text(text)
    print("patched", path)


def patch_cuda_kernels_cmake():
    path = ROOT / "xllm/core/kernels/cuda/CMakeLists.txt"
    text = path.read_text()
    if "XLLM_TORCH_MUSA" in text:
        print("already patched", path)
        return
    text = text.replace(
        "if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)",
        "if(XLLM_TORCH_MUSA)\n  message(STATUS \"Skipping CUTLASS SM90+ libs for XLLM_TORCH_MUSA\")\nelseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)",
        1,
    )
    text = text.replace(
        "if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)\n  add_compile_definitions(ENABLE_FP8)\nendif()",
        "if(NOT XLLM_TORCH_MUSA AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)\n  add_compile_definitions(ENABLE_FP8)\nendif()",
    )
    path.write_text(text)
    print("patched", path)


def patch_parallel_state_cmake():
    path = ROOT / "xllm/core/framework/parallel_state/CMakeLists.txt"
    text = path.read_text()
    if "XLLM_TORCH_MUSA" in text:
        print("already patched", path)
        return
    text = text.replace(
        "    $<$<BOOL:${USE_MUSA}>:torch_musa>\n",
        "    $<$<BOOL:${USE_MUSA}>:torch_musa>\n    $<$<BOOL:${XLLM_TORCH_MUSA}>:torch_musa>\n",
    )
    path.write_text(text)
    print("patched", path)


def patch_platform_cmake():
    path = ROOT / "xllm/core/platform/CMakeLists.txt"
    text = path.read_text()
    if "XLLM_TORCH_MUSA" in text:
        print("already patched", path)
        return
    text = text.replace(
        "    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_ILU}>>:cuda>\n    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_ILU}>>:cudart>\n",
        "    $<$<AND:$<BOOL:${USE_CUDA}>,$<NOT:$<BOOL:${XLLM_TORCH_MUSA}>>>:cuda>\n    $<$<AND:$<BOOL:${USE_CUDA}>,$<NOT:$<BOOL:${XLLM_TORCH_MUSA}>>>:cudart>\n    $<$<BOOL:${XLLM_TORCH_MUSA}>:musa>\n    $<$<BOOL:${XLLM_TORCH_MUSA}>:musart>\n",
    )
    path.write_text(text)
    print("patched", path)


def patch_cuda_process_group():
    path = ROOT / "xllm/core/framework/parallel_state/cuda_process_group.h"
    content = '''/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#if defined(XLLM_TORCH_MUSA)
#include <torch_musa/csrc/distributed/ProcessGroupMCCL.h>
#else
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#include "process_group.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  ProcessGroupImpl(int32_t global_rank,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device)
      : ProcessGroup(global_rank, world_size, device) {
#if defined(XLLM_TORCH_MUSA)
    c10::intrusive_ptr<c10d::ProcessGroupMCCL::Options> pg_options =
        c10d::ProcessGroupMCCL::Options::create();
#else
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
        c10d::ProcessGroupNCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    pg_options->group_name = group_name;
#endif
#endif
    int32_t rank = global_rank;
    if (world_size != rank_size) {
      auto [local_rank, group_ranks] =
          get_group_rank(world_size, global_rank, rank_size, trans);
#if !defined(XLLM_TORCH_MUSA)
      pg_options->global_ranks_in_group = group_ranks;
#endif
      rank = local_rank;
    }

    auto store = create_tcp_store(host, port, rank);
#if defined(XLLM_TORCH_MUSA)
    pg_ = std::make_unique<c10d::ProcessGroupMCCL>(
        store, rank, rank_size, pg_options);
#else
    pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
        store, rank, rank_size, pg_options);
#endif
  }
};

}  // namespace xllm
'''
    path.write_text(content)
    print("patched", path)


def patch_env_py():
    path = ROOT / "scripts/build_support/env.py"
    text = path.read_text()
    if "def set_torch_musa_cuda_envs" in text:
        print("already patched", path)
        return
    insert = '''

def set_torch_musa_cuda_envs() -> None:
    """Configure USE_CUDA + mcc_wrapper build on MUSA hardware."""
    set_common_envs()
    musa_home = os.getenv("MUSA_HOME", "/usr/local/musa")
    os.environ["MUSA_HOME"] = musa_home
    os.environ["CUDA_HOME"] = musa_home
    os.environ["CUDAToolkit_ROOT"] = musa_home
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = musa_home
    os.environ["MUSA_TOOLKIT_ROOT_DIR"] = musa_home
    os.environ["MUSAMAPPING_PATH"] = os.path.join(musa_home, "tools/musamapping")
    os.environ["PYTORCH_MUSA_INSTALL_PATH"] = get_torch_musa_root_path() or ""
    import torch_musa
    from torch_musa.utils.musa_extension import MUSA_HOME as _MUSA_HOME

    os.environ["TORCH_MUSA_PYTHONPATH"] = torch_musa.core.cmake_prefix_path
    os.environ["TorchMusa_DIR"] = (
        torch_musa.core.cmake_prefix_path + "/TorchMusa"
    )
    if not os.getenv("MUSA_HOME"):
        os.environ["MUSA_HOME"] = _MUSA_HOME

    for path in (
        os.path.join(musa_home, "lib"),
        "/opt/intel/oneapi/mkl/lib/intel64",
        "/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib",
        os.path.join(get_torch_musa_root_path() or "", "lib"),
        os.path.join(get_torch_root_path() or "", "lib"),
    ):
        if path and os.path.isdir(path):
            prepend_path_env("LD_LIBRARY_PATH", path)
'''
    text = text.replace("def set_musa_envs() -> None:", insert + "\ndef set_musa_envs() -> None:")
    path.write_text(text)
    print("patched", path)


def patch_setup_py():
    path = ROOT / "setup.py"
    text = path.read_text()
    if "set_torch_musa_cuda_envs" in text:
        print("already patched", path)
        return
    text = text.replace(
        "    set_musa_envs,\n    set_npu_envs,\n",
        "    set_musa_envs,\n    set_torch_musa_cuda_envs,\n    set_npu_envs,\n",
    )
    old = '''        elif self.device == "cuda":
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if not torch_cuda_architectures:
                raise ValueError("Please set TORCH_CUDA_ARCH_LIST environment variable, e.g. export TORCH_CUDA_ARCH_LIST=\\"8.0 8.9 9.0 10.0 12.0\\"")
            cmake_args += ["-DUSE_CUDA=ON",
                           f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}"]
            set_cuda_envs()
'''
    new = '''        elif self.device == "cuda":
            use_torch_musa = os.getenv("XLLM_TORCH_MUSA", "").lower() in (
                "1",
                "on",
                "true",
                "yes",
            )
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if use_torch_musa:
                if not torch_cuda_architectures:
                    torch_cuda_architectures = os.getenv(
                        "TORCH_MUSA_ARCH_LIST", "9.0"
                    )
                cmake_args += [
                    "-DUSE_CUDA=ON",
                    "-DXLLM_TORCH_MUSA=ON",
                    f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}",
                ]
                set_torch_musa_cuda_envs()
            else:
                if not torch_cuda_architectures:
                    raise ValueError(
                        'Please set TORCH_CUDA_ARCH_LIST environment variable, '
                        'e.g. export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"'
                    )
                cmake_args += [
                    "-DUSE_CUDA=ON",
                    f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}",
                ]
                set_cuda_envs()
'''
    if old not in text:
        raise RuntimeError("setup.py cuda block not found")
    text = text.replace(old, new)
    path.write_text(text)
    print("patched", path)


if __name__ == "__main__":
    patch_root_cmake()
    patch_xllm_cmake()
    patch_cuda_kernels_cmake()
    patch_parallel_state_cmake()
    patch_platform_cmake()
    patch_cuda_process_group()
    patch_env_py()
    patch_setup_py()
