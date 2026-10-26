#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROLE=${1:?usage: build_site.sh prefill|decode}
case "$ROLE" in prefill|decode) ;; *) exit 2 ;; esac
# Site profile: the existing 90/94 wjm_xllm1008 environment.
export PATH=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin:/usr/local/openmpi/bin:/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/cmake/data/bin:/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/ninja/data/bin:/usr/local/musa/bin:/usr/local/musa/tools/musamapping:/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin
export MUSA_HOME=/usr/local/musa
export CUDA_HOME=/usr/local/musa
export CUDAToolkit_ROOT=/usr/local/musa
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/musa
export MUSAMAPPING_PATH="/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/tooling/musamapping_${ROLE}"
export PYTORCH_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages/torch
export LIBTORCH_ROOT=/usr/local/lib/python3.10/dist-packages/torch
export TORCH_MUSA_PYTHONPATH=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake
export TorchMusa_DIR=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake/TorchMusa
export MKLROOT=/opt/intel/oneapi/mkl
export MUDNN_HOME=/workspace/mudnn_3.4.0
export MCCL_HOME=/workspace/MCCL_2.3.0/mccl
export PYTHON_INCLUDE_PATH=/usr/include/python3.10
export PYTHON_LIB_PATH=/usr/lib/x86_64-linux-gnu
export VCPKG_ROOT=/data/nfs_shared/cmz/workspace/xllm_dv4730/vcpkg-valid-local
export VCPKG_MANIFEST_MODE=OFF
export VCPKG_MANIFEST_INSTALL=OFF
export VCPKG_DOWNLOADS=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/vcpkg-downloads
export VCPKG_DEFAULT_BINARY_CACHE=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/vcpkg-binary-cache
export DEPENDENCES_ROOT="/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/build_${ROLE}/_deps"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MUSA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=''
export LD_PRELOAD=''
export LD_LIBRARY_PATH=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/tvm_ffi/lib:/usr/local/musa/lib:/workspace/mudnn_3.4.0/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/opt/intel/oneapi/mkl/lib/intel64:/usr/local/lib:/usr/lib/x86_64-linux-gnu
export CARGO_HOME=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/cargo-home
export RUSTUP_HOME=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/rustup
export RUSTC=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/rustc
export TorchMusa_INSTALL_PREFIX=/usr/local/lib/python3.10/dist-packages/torch_musa
export CARGO_NET_OFFLINE=true
export MCCL_INCLUDE_DIR=/workspace/MCCL_2.3.0/mccl/include
export MCCL_LIB_DIR=/workspace/MCCL_2.3.0/mccl/lib
export CMAKE_CUDA_COMPILER_LAUNCHER=''
test -f "$MUDNN_HOME/lib/libmudnn.so"
test -f "$MCCL_INCLUDE_DIR/mccl.h"
test -x "$MUSAMAPPING_PATH/mcc_wrapper"

/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/cmake/data/bin/cmake \
  -S \
  /data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/source \
  -B \
  /data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/build_${ROLE} \
  -U \
  XLLM_MUSA_DEEPSEEK_ONLY \
  -G \
  Ninja \
  -DCMAKE_MAKE_PROGRAM=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/ninja/data/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DUSE_MUSA=ON \
  -DUSE_CUDA=OFF \
  -DUSE_NPU=OFF \
  -DUSE_MLU=OFF \
  -DUSE_CXX11_ABI=ON \
  -DUSE_CCACHE=OFF \
  -DPython_EXECUTABLE=/usr/bin/python3.10 \
  -DPython3_EXECUTABLE=/usr/bin/python3.10 \
  -DPython_INCLUDE_DIR=/usr/include/python3.10 \
  -DCMAKE_TOOLCHAIN_FILE=/data/nfs_shared/cmz/workspace/xllm_dv4730/vcpkg-valid-local/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_MANIFEST_MODE=OFF \
  -DVCPKG_MANIFEST_INSTALL=OFF \
  -DVCPKG_INSTALLED_DIR=/data/nfs_shared/cmz/workspace/xllm_dv4730/xllm-musa/build/cmake.linux-x86_64-cpython-310/vcpkg_installed \
  -DVCPKG_TARGET_TRIPLET=x64-linux \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch \
  -DMKL_DIR=/opt/intel/oneapi/mkl/lib/cmake/mkl \
  -D_CMAKE_CUDA_EXTRA_FLAGS=--offload-arch=mp_31 \
  -DCARGO_EXECUTABLE=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/cargo \
  -DRUSTC_EXECUTABLE=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/rustc \
  -DRUSTDOC_EXECUTABLE=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/rustdoc \
  -DCARGO_HOME=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/cargo-home \
  -DXLLM_TVM_FFI_ROOT=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/tvm_ffi \
  -DFFMPEG_DEPENDENCY_pthread_RELEASE=/usr/lib/x86_64-linux-gnu/libpthread.a \
  -DFFMPEG_DEPENDENCY_pthread_DEBUG=/usr/lib/x86_64-linux-gnu/libpthread.a \
  -DFFMPEG_DEPENDENCY_dl_RELEASE=/usr/lib/x86_64-linux-gnu/libdl.a \
  -DFFMPEG_DEPENDENCY_dl_DEBUG=/usr/lib/x86_64-linux-gnu/libdl.a \
  -DFFMPEG_DEPENDENCY_m_RELEASE=/usr/lib/x86_64-linux-gnu/libm.so \
  -DFFMPEG_DEPENDENCY_m_DEBUG=/usr/lib/x86_64-linux-gnu/libm.so \
  '-DFFMPEG_LIBRARY_DIRS=/data/nfs_shared/cmz/workspace/xllm_dv4730/xllm-musa/build/cmake.linux-x86_64-cpython-310/vcpkg_installed/x64-linux/lib;/data/nfs_shared/cmz/workspace/xllm_dv4730/xllm-musa/build/cmake.linux-x86_64-cpython-310/vcpkg_installed/x64-linux/debug/lib' \
  '-DCMAKE_CXX_FLAGS=-Wno-register -Wno-error=unused-value -Wno-unused-value -I/usr/local/include' \
  -DTorchMusa_DIR=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/torch_musa_cmake/TorchMusa \
  -DOMP_LIBRARY=/usr/local/musa/lib/libiomp5.so \
  -DUSE_ETCD=OFF \
  -DSTORE_USE_ETCD=OFF \
  -DXLLM_DSV4_PD_ROLE=${ROLE} \
  -DCMAKE_CXX_COMPILER=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/tooling/musamapping_${ROLE}/mcc_wrapper \
  -DCMAKE_CUDA_COMPILER=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/tooling/musamapping_${ROLE}/mcc_wrapper \
  -DCMAKE_MODULE_PATH=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/tooling/musamapping_${ROLE}/cmake/Modules \
  -DFETCHCONTENT_BASE_DIR=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/build_${ROLE}/_deps \
  -DASIO_INCLUDE_DIR=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/source/third_party/Mooncake/extern/yalantinglibs/include/ylt/thirdparty \
  '-DCMAKE_CUDA_FLAGS=-std=c++20 -DCCCL_DISABLE_NVFP8_SUPPORT' \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON

/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/ninja/data/bin/ninja -C /data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026/build_${ROLE} -j 24 -k 1 xllm spawn_worker
