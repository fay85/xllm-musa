#!/bin/bash
# NOTE: This script is deprecated. Use _build_cuda_graph_musa.sh instead.
# MTTOplib is no longer supported - all MUSA builds now use FlashInfer/Mate path.
set -euo pipefail

export MUSA_HOME=/usr/local/musa
export CUDA_HOME=/usr/local/musa
export CUDAToolkit_ROOT=/usr/local/musa
export MUSA_TOOLKIT_ROOT_DIR=/usr/local/musa
export MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping
export MUSA_INCLUDE_PATH=/usr/local/musa/include

export PYTORCH_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages/torch
export LIBTORCH_ROOT=/usr/local/lib/python3.10/dist-packages/torch
export PYTHON_LIB_PATH=/usr/local/lib/python3.10/dist-packages/torch
export PYTORCH_MUSA_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages/torch_musa
export TorchMusa_DIR=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake/TorchMusa
export TORCH_MUSA_PYTHONPATH=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake
export TORCH_MUSA_ARCH_LIST=31

export MPI_DIR=/usr/local/openmpi
export MPICC=/usr/local/openmpi/bin/mpicc
export CPATH=/usr/local/openmpi/include:${CPATH:-}

export VCPKG_ROOT=/workspace/vcpkg-xllm
export VCPKG_FORCE_SYSTEM_BINARIES=1
export VCPKG_MAX_CONCURRENCY=16

export CMAKE_PREFIX_PATH=/usr/local/yalantinglibs
export PYTHON_INCLUDE_PATH=/usr/include/python3.10
export CPU_UNIFIED_FLAG=False
export SKIP_TEST=1
export SKIP_EXPORT=1
export MAX_JOBS=16

export MKL_DIR=/opt/intel/oneapi/mkl/lib/cmake/mkl
export MKLROOT=/opt/intel/oneapi/mkl
export LIB=/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64

export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}

export PATH=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/root/.cargo/bin:/usr/local/lib/python3.10/dist-packages/cmake/data/bin:/usr/local/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/openmpi/bin:$PATH

export CMAKE_ARGS="-DUSE_CXX11_ABI=ON -D_GLIBCXX_USE_CXX11_ABI=1 -DGENERATE_SO=OFF -DVCPKG_MANIFEST_INSTALL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

cd /workspace/xllm-git-master
exec python3 setup.py build --device musa
