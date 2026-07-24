#!/bin/bash
cd /workspace/xllm-git-master
export PATH="/workspace/xllm-git-master/scripts/ninja_guard:${PATH}"
export XLLM_NINJA_GUARD="/workspace/xllm-git-master/scripts/ninja_guard/ninja"
source <(grep "^export " _build_cuda_graph_musa.sh | grep -v "CMAKE_ARGS=")
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/musa/tools/musamapping/mcc_wrapper -DCMAKE_MODULE_PATH=/usr/local/musa/tools/musamapping/cmake/Modules -DCUDAToolkit_ROOT=/usr/local/musa -DCUDA_HOME=/usr/local/musa -DUSE_CXX11_ABI=ON -D_GLIBCXX_USE_CXX11_ABI=1 -DGENERATE_SO=OFF -DVCPKG_MANIFEST_INSTALL=OFF -DUSE_MUSA=ON -DUSE_CUDA=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
LOG=build_logs/build_full_$(date +%Y%m%d_%H%M%S).log
echo "$LOG" > /tmp/cur_log.txt
python3 setup.py build --device cuda > "$LOG" 2>&1
