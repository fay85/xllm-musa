#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# One source tree; one command; separate verified P/D role artifacts.
# Reuses installed dependencies, not mutable experiment object archives.
set -euo pipefail
if [[ $# -gt 0 ]]; then
  [[ $1 == --help || $1 == -h ]] || { echo 'Unknown build argument' >&2; exit 2; }
  echo 'Usage: MAX_JOBS=12 NINJA_TARGET=xllm bash tools/musa_unified/build.sh'
  echo 'Overrides: XLLM_SOURCE_DIR, XLLM_BUILD_ROOT, XLLM_BUNDLE_DIR, XLLM_DEEPSEEK_OPS, RUSTC, CARGO, CARGO_HOME'
  exit 0
fi
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$kit/../.." && pwd)
source_dir=${XLLM_SOURCE_DIR:-$repo}
build_root=${XLLM_BUILD_ROOT:-$repo/build/musa_unified}
bundle=${XLLM_BUNDLE_DIR:-$build_root/bundle}
logs=${XLLM_BUILD_LOGS:-$build_root/logs}
source_dir=$(realpath "$source_dir")
mkdir -p "$build_root" "$bundle" "$logs"
build_root=$(realpath "$build_root")
bundle=$(realpath "$bundle")
logs=$(realpath "$logs")
python3 "$kit/artifacts.py" idle --bundle "$bundle"
test -f "$source_dir/CMakeLists.txt"
jobs=${MAX_JOBS:-12}
target=${NINJA_TARGET:-xllm}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]]
[[ "$target" == xllm ]] || { echo 'This bundle entry requires NINJA_TARGET=xllm' >&2; exit 2; }
export MUSA_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" MTHREADS_VISIBLE_DEVICES=""
unset LD_PRELOAD XLLM_USE_FA3 XLLM_USE_FA3_DECODE
export MUSA_HOME=/usr/local/musa CUDA_HOME=/usr/local/musa CUDAToolkit_ROOT=/usr/local/musa
export MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping
export PATH="/root/.cargo/bin:/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=${CARGO_HOME:-/root/.cargo}
export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
export RUSTC=${RUSTC:-$(command -v rustc || true)}
export CARGO=${CARGO:-$(command -v cargo || true)}
[[ -x $RUSTC && -x $CARGO ]] || { echo 'Rust/Cargo missing; set RUSTC, CARGO and CARGO_HOME (see README).' >&2; exit 2; }
if [[ -z ${XLLM_DEEPSEEK_OPS:-} && ! -d $bundle/cached_ops ]]; then
  echo 'Set XLLM_DEEPSEEK_OPS to the verified MATE 0.2.7 DeepSeek AOT directory.' >&2
  exit 2
fi
export MATE_HOME=/opt/mate_0.2.7 MATE_WORKSPACE_BASE=/opt/mate027_cache
export MKLROOT=/opt/intel/oneapi/mkl MKL_DIR=/opt/intel/oneapi/mkl/lib/cmake/mkl
export VCPKG_ROOT=${VCPKG_ROOT:-/workspace/vcpkg-xllm} VCPKG_FORCE_SYSTEM_BINARIES=1
vcpkg_installed=${XLLM_VCPKG_INSTALLED:-/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310/vcpkg_installed}
[[ -d $vcpkg_installed && -f $VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake ]]
export PATH="$(dirname "$RUSTC"):/usr/local/musa/bin:/usr/local/openmpi/bin:$PATH"
cd "$source_dir"
eval "$(python3 - <<'PY'
import os, shlex
from scripts.build_support import env
before = dict(os.environ)
env.set_common_envs()
env.set_musa_envs()
for key, value in os.environ.items():
    if before.get(key) != value:
        print(f'export {key}={shlex.quote(value)}')
PY
)"
python3 - <<'PY'
from importlib.metadata import version
for name, expected in {'mate':'0.2.7','apache-tvm-ffi':'0.1.11.post1+musa.1','tilelang-musa':'0.1.12+musa.2'}.items():
    if version(name) != expected:
        raise RuntimeError(f'{name} must be {expected}')
PY
for role in prefill decode; do
  build_dir="$build_root/$role"
  output="$bundle/$role"
  log="$logs/$role"
  mkdir -p "$build_dir" "$output" "$log"
  # mcc depfiles may report this mapped header by its unqualified CUDA name.
  # Resolve that dependency locally; do not modify the SDK or compiler search path.
  mapped_runtime_header="$MUSA_HOME/include/cuda_runtime_api.h"
  test -f "$mapped_runtime_header"
  if [[ -e "$build_dir/cuda_runtime_api.h" || -L "$build_dir/cuda_runtime_api.h" ]]; then
    [[ "$(realpath "$build_dir/cuda_runtime_api.h")" == "$(realpath "$mapped_runtime_header")" ]]
  else
    ln -s "$mapped_runtime_header" "$build_dir/cuda_runtime_api.h"
  fi
  if [[ -e "$build_dir/CMakeCache.txt" ]]; then
    grep -Fxq "XLLM_DSV4_PD_ROLE:STRING=$role" "$build_dir/CMakeCache.txt"
    grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$source_dir" "$build_dir/CMakeCache.txt"
  fi
  export LIBRARY_PATH="$build_dir/xllm/core/layers/musa:$MUSA_HOME/lib:$MKLROOT/lib/intel64"
  stamp=$(date +%Y%m%d_%H%M%S)
  if [[ ! -f "$build_dir/build.ninja" || "${FORCE_CMAKE:-0}" == 1 ]]; then
    cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DUSE_MUSA=ON -DUSE_CUDA=OFF -DUSE_CXX11_ABI=ON -DGENERATE_SO=OFF \
    -DBUILD_TESTING=OFF "-DXLLM_DSV4_PD_ROLE=$role" \
    -DUSE_ETCD=OFF -DSTORE_USE_ETCD=OFF -DUSE_ETCD_LEGACY=OFF \
    -DVCPKG_MANIFEST_INSTALL=OFF \
    "-DVCPKG_INSTALLED_DIR=$vcpkg_installed" \
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    "-DCMAKE_CUDA_COMPILER=$MUSAMAPPING_PATH/mcc_wrapper" \
    -DCMAKE_MAKE_PROGRAM=/usr/local/bin/ninja \
    -DCMAKE_CUDA_ARCHITECTURES=90 -DTORCH_CUDA_ARCH_LIST=9.0 \
    "-DCMAKE_MODULE_PATH=$kit/cmake;$MUSAMAPPING_PATH/cmake/Modules" \
    -DCMAKE_CUDA_FLAGS:STRING=-I/usr/include/python3.10 \
    -DCMAKE_CXX_FLAGS:STRING=-I/usr/include/python3.10 \
    '-DCMAKE_CXX_COMPILER_LAUNCHER:STRING=/bin/bash;-e' \
    '-DCMAKE_CUDA_COMPILER_LAUNCHER:STRING=/bin/bash;-e' \
    '-DCMAKE_CXX_LINKER_LAUNCHER:STRING=/bin/bash;-e' \
    '-DCMAKE_CUDA_LINKER_LAUNCHER:STRING=/bin/bash;-e' \
    -DCMAKE_EXPERIMENTAL_RUST=3cc9b32c-47d3-4056-8953-d74e69fc0d6c \
    -DCMAKE_PREFIX_PATH=/usr/local/yalantinglibs \
    -DPython_EXECUTABLE=/usr/bin/python3.10 -DPython_INCLUDE_DIR=/usr/include/python3.10 \
    -DPython_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
    -DOMP_LIBRARY=/usr/local/musa/lib/libiomp5.so \
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$output" "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$output" \
    2>&1 | tee "$log/configure_$stamp.log"
  fi
  if [[ ${XLLM_CONFIGURE_ONLY:-0} == 1 ]]; then
    continue
  fi
  "$source_dir/scripts/ninja_safe.sh" "$build_dir" -j"$jobs" xllm 2>&1 | tee "$log/compile_$stamp.log"
  if grep -E 'FAILED:|fatal error:|(^|[[:space:]])error:|undefined reference' "$log/compile_$stamp.log"; then
    exit 20
  fi
  cp -a "$build_dir/mooncake-common/libasio.so" "$output/"
  cp "$build_dir/CMakeCache.txt" "$log/"
  cp "$build_dir/compile_commands.json" "$log/"
  cp "$build_dir/xllm/core/kernels/musa/dsv4_pd_kernel_selection.txt" "$log/"
  cp "$build_dir/xllm/core/layers/musa/dsv4_pd_layer_selection.txt" "$log/"
  LD_LIBRARY_PATH="$output:$LD_LIBRARY_PATH" ldd -r "$output/xllm" > "$log/ldd_$stamp.txt" 2>&1
  if grep -E 'not found|undefined symbol' "$log/ldd_$stamp.txt"; then exit 21; fi
  LD_LIBRARY_PATH="$output:$LD_LIBRARY_PATH" LD_BIND_NOW=1 \
    python3 "$kit/artifacts.py" help-xml --binary "$output/xllm" --log "$log/help_$stamp.txt"
  sha256sum "$output/xllm" "$output/libasio.so" > "$log/binary_$stamp.sha256"
done
if [[ ${XLLM_CONFIGURE_ONLY:-0} == 1 ]]; then
  echo 'CONFIGURE_ONLY_COMPLETE (no binary or inference acceptance)'
  exit 0
fi
python3 "$kit/artifacts.py" assemble --bundle "$bundle" --source "$source_dir" \
  --deepseek-ops "${XLLM_DEEPSEEK_OPS:-}" \
  --qwen-ops "${XLLM_QWEN_OPS:-/opt/mate027_cache/mp31/cached_ops_jd_k8d}" \
  --mubin "${MATE_MUBIN_DIR:-/usr/local/lib/python3.10/dist-packages/mate_mubin/mubin}" \
  --manifest "${XLLM_ENGINE_MANIFEST:-$bundle/engine_manifest.json}"
printf 'UNIFIED_ENGINE_BUILD_COMPLETE (model accuracy/performance acceptance is separate)\n'
