#!/bin/bash
# Build xllm-git-master on MUSA via the CUDA-compatibility graph path:
#   USE_MUSA + mcc_wrapper (FlashInfer / MusaGraphExecutorImpl)
#
# This path intentionally uses mcc_wrapper; it does not require nvcc.
set -euo pipefail
#
# Notes:
#   - Graph logic unchanged; validate vs sglang later.
#   - USE_MUSA: no CUTLASS; use MUSA libs (see docs/cuda_graph_musa_port.md).
#   - All ninja invocations go through scripts/ninja_guard (flock + .ninja_log backup).
#     Never run bare "ninja -C ..." or "pkill -9 ninja" on this build dir.

# Canonical toolkit + torch_musa cmake prefix. CUDA_* / MUSAMAPPING_PATH /
# TorchMusa_DIR are derived in scripts/build_support/env.py::set_musa_envs().
export MUSA_HOME=/usr/local/musa
export CUDA_HOME="${MUSA_HOME}"
export CUDAToolkit_ROOT="${MUSA_HOME}"
export MUSAMAPPING_PATH="${MUSA_HOME}/tools/musamapping"
# USE_MUSA builds use FlashInfer/Mate kernels, not native MTTOplib.

export PYTORCH_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages/torch
export LIBTORCH_ROOT="${PYTORCH_INSTALL_PATH}"
export PYTHON_LIB_PATH="${PYTORCH_INSTALL_PATH}"
export TORCH_MUSA_PYTHONPATH=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake
export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl}"
export MKL_DIR="${MKL_DIR:-${MKLROOT}/lib/cmake/mkl}"
export TVM_FFI_LIB_DIR="${TVM_FFI_LIB_DIR:-$(tvm-ffi-config --libdir)}"
export TorchMusa_DIR="${TORCH_MUSA_PYTHONPATH}/TorchMusa"
# Prefer CUDA-language arch list for the mcc graph path; drop stale MUSA ISA export.
unset TORCH_MUSA_ARCH_LIST || true
export TORCH_CUDA_ARCH_LIST="9.0"

resolve_mate_home() {
  local candidate
  if [[ -n "${MATE_HOME:-}" ]]; then
    candidate="${MATE_HOME}"
    [[ -f "${candidate}/version.txt" ]] &&
      [[ "$(tr -d '[:space:]' < "${candidate}/version.txt")" == "0.2.6" ]] && {
      printf '%s\n' "${candidate}"
      return
    }
  fi
  for candidate in /workspace/mate_0.2.6 /data/feihu/mate_0.2.6; do
    if [[ -f "${candidate}/version.txt" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "Mate 0.2.6 source tree not found; set MATE_HOME." >&2
  return 1
}

MATE_HOME="$(resolve_mate_home)" || exit 1
export MATE_HOME
MATE_VERSION="$(tr -d '[:space:]' < "${MATE_HOME}/version.txt")"
[[ "${MATE_VERSION}" == "0.2.6" ]] || {
  echo "Expected Mate 0.2.6, found ${MATE_VERSION} at ${MATE_HOME}" >&2
  exit 1
}
export MATE_MUSA_ARCH_LIST="${MATE_MUSA_ARCH_LIST:-3.1}"
# MATE 0.2.6 is installed from its official wheel so the packaged Mutlass
# headers match the release. Drop stale source-tree overrides from the image.
filtered_pythonpath=""
IFS=: read -ra pythonpath_entries <<< "${PYTHONPATH:-}"
for pythonpath_entry in "${pythonpath_entries[@]}"; do
  case "${pythonpath_entry}" in
    /workspace/mate_0.2.* | /data/feihu/mate_0.2.*) continue ;;
  esac
  filtered_pythonpath="${filtered_pythonpath:+${filtered_pythonpath}:}${pythonpath_entry}"
done
export PYTHONPATH="${filtered_pythonpath}"
python3 - <<'PY'
from importlib.metadata import version

expected = {
    "mate": "0.2.6",
    "apache-tvm-ffi": "0.1.11.post1+musa.1",
    "tilelang-musa": "0.1.12+musa.2",
}
for package, expected_version in expected.items():
    installed_version = version(package)
    if installed_version != expected_version:
        raise RuntimeError(
            f"{package}: expected {expected_version}, got {installed_version}"
        )
PY
if [[ -z "${MATE_WORKSPACE_BASE:-}" ||
      "${MATE_WORKSPACE_BASE}" == */mate025_cache ]]; then
  if [[ -d /data/feihu/mate026_cache ]]; then
    MATE_WORKSPACE_BASE=/data/feihu/mate026_cache
  else
    MATE_WORKSPACE_BASE="${MATE_HOME}"
  fi
fi
export MATE_WORKSPACE_BASE

if [[ -z "${MATE_MUBIN_DIR:-}" || "${MATE_MUBIN_DIR}" == *mate025* ]]; then
  MATE_MUBIN_DIR="${MATE_WORKSPACE_BASE}/mubin"
fi
export MATE_MUBIN_DIR
export FLASHINFER_OPS_PATH="${MATE_WORKSPACE_BASE}/.cache/mate/0.2.6/mp31/cached_ops"
export LD_LIBRARY_PATH="${TVM_FFI_LIB_DIR}:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

CMAKE_MODULE_PATH_VALUE="${MUSAMAPPING_PATH}/cmake/Modules"
if [[ -n "${XLLM_EXTRA_CMAKE_MODULE_PATH:-}" ]]; then
  CMAKE_MODULE_PATH_VALUE+=";${XLLM_EXTRA_CMAKE_MODULE_PATH}"
fi

export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${MUSAMAPPING_PATH}/mcc_wrapper -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH_VALUE} -DCUDAToolkit_ROOT=${MUSA_HOME} -DCUDA_HOME=${MUSA_HOME} -DUSE_CXX11_ABI=ON -D_GLIBCXX_USE_CXX11_ABI=1 -DGENERATE_SO=OFF -DVCPKG_MANIFEST_INSTALL=OFF -DUSE_MUSA:BOOL=ON -DUSE_CUDA:BOOL=OFF -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_EXPERIMENTAL_RUST=3cc9b32c-47d3-4056-8953-d74e69fc0d6c"

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*

export MPI_DIR=/usr/local/openmpi
export MPICC=/usr/local/openmpi/bin/mpicc
export CPATH=/usr/local/openmpi/include:${CPATH:-}
export VCPKG_ROOT=/workspace/vcpkg-xllm
export VCPKG_FORCE_SYSTEM_BINARIES=1
export VCPKG_MAX_CONCURRENCY=16
export VCPKG_CMAKE_CONFIGURE_OPTIONS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH="/usr/local/yalantinglibs:${CMAKE_PREFIX_PATH:-}"
export PYTHON_INCLUDE_PATH=/usr/include/python3.10
export CPU_UNIFIED_FLAG=False
export SKIP_TEST=1
export SKIP_EXPORT=1
export MAX_JOBS="${MAX_JOBS:-16}"
export PATH=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/root/.cargo/bin:/usr/local/lib/python3.10/dist-packages/cmake/data/bin:/usr/local/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/openmpi/bin:$PATH
export CARGO_HOME=/root/.cargo
export RUSTC=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc
export CARGO=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo
export LIB=/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64:${LIB:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BD=build/cmake.linux-x86_64-cpython-310
# mcc g++ link driver resolves -lmusa_layers via LIBRARY_PATH (not cmake LINK_PATH).
export LIBRARY_PATH="${SCRIPT_DIR}/${BD}/xllm/core/layers/musa:${LIBRARY_PATH:-}"

# Guard all ninja/cmake --build calls: one writer + .ninja_log backup/restore.
NINJA_GUARD_DIR="${SCRIPT_DIR}/scripts/ninja_guard"
# mcc g++ link driver: wrapper adds MKL + --allow-shlib-undefined (see scripts/musa_link_wrapper/g++).
mkdir -p "${SCRIPT_DIR}/scripts/musa_link_wrapper"
if [ ! -x "${SCRIPT_DIR}/scripts/musa_link_wrapper/g++" ]; then
  cat > "${SCRIPT_DIR}/scripts/musa_link_wrapper/g++" <<'GXXWRAP'
#!/bin/bash
# mcc_wrapper final link: libtorch_cpu pulls MKL DSOs; resolve at runtime via LD_PRELOAD.
set -euo pipefail
exec /usr/bin/g++ "$@" -Wl,--allow-shlib-undefined
GXXWRAP
  chmod +x "${SCRIPT_DIR}/scripts/musa_link_wrapper/g++"
fi
export PATH="${SCRIPT_DIR}/scripts/musa_link_wrapper:${PATH}"
export PATH="${NINJA_GUARD_DIR}:${PATH}"

ln -sf libmudnncxx.so /usr/local/musa/lib/libmudnn.so 2>/dev/null || true
# cmake 4.2 CUDA toolkit detection requires ${ROOT}/nvvm/libdevice to exist
mkdir -p /usr/local/musa/nvvm/libdevice 2>/dev/null || true
# torch ATen/cuda headers need c10/cuda/impl/cuda_cmake_macros.h (provided as musa shim)
TORCH_C10_CUDA_IMPL=/usr/local/lib/python3.10/dist-packages/torch/include/c10/cuda/impl
MUSA_CMAKE_MACROS=/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/include/c10/musa/impl/musa_cmake_macros.h
mkdir -p "${TORCH_C10_CUDA_IMPL}" 2>/dev/null || true
[ -f "${MUSA_CMAKE_MACROS}" ] && [ ! -e "${TORCH_C10_CUDA_IMPL}/cuda_cmake_macros.h" ] && ln -sf "${MUSA_CMAKE_MACROS}" "${TORCH_C10_CUDA_IMPL}/cuda_cmake_macros.h"

# CUDAContextLight.h expects cusparse/cublasLt/cusolver; musamapping maps these for .cu only.
MUSA_INC=/usr/local/musa/include
# Sync CUDA-compat headers for host .cpp (musamapping include.json; .cu uses plugin)
MUSA_INC=/usr/local/musa/include
MAP=/usr/local/musa/tools/musamapping/mapping/include.json
if [ -f "$MAP" ]; then
  python3 -c "import json,os; d=json.load(open('$MAP')); inc='$MUSA_INC'
for c,m in d.items():
  if c.endswith('.h') and not os.path.lexists(os.path.join(inc,c)) and os.path.exists(os.path.join(inc,m)):
    os.symlink(m, os.path.join(inc, c))" 2>/dev/null || true
fi
for pair in cusparse.h:musparse.h cublasLt.h:mublasLt.h cusolverDn.h:musolverDn.h; do
  cuda=${pair%%:*}; musa=${pair##*:}
  [ ! -e "${MUSA_INC}/${cuda}" ] && [ -f "${MUSA_INC}/${musa}" ] && ln -sf "${musa}" "${MUSA_INC}/${cuda}"
done
# Build reference only (do not copy source from here): jiacun xllm-musa on dev7 host.
# Override with XLLM_MUSA_REF if mounted elsewhere in the container.
XLLM_MUSA_REF="${XLLM_MUSA_REF:-/workspace/xllm-musa}"
if [ ! -d "${XLLM_MUSA_REF}" ] && [ -d /data/jiacun/xllm/xllm-musa ]; then
  XLLM_MUSA_REF=/data/jiacun/xllm/xllm-musa
fi

# Restore libmusa runtime (container stub may be empty)
MUSA_RUNTIME="${XLLM_MUSA_REF}/runtime/libmusa.so.5.1.0"
if [ ! -f "${MUSA_RUNTIME}" ]; then
  MUSA_RUNTIME=/usr/local/musa/lib/libmusa.so.5.1.0
fi
if [ -f "${MUSA_RUNTIME}" ] && [ ! -s /usr/local/musa/lib/libmusa.so.5.1.0 ]; then
  cp -f "${MUSA_RUNTIME}" /usr/local/musa/lib/libmusa.so.5.1.0
  ln -sf libmusa.so.5.1.0 /usr/local/musa/lib/libmusa.so
  ln -sf libmusa.so.5.1.0 /usr/local/musa/lib/libmusa.so.1
fi
if [ -f /usr/lib/x86_64-linux-gnu/libmusa.so.5.1.0 ] && [ ! -s /usr/lib/x86_64-linux-gnu/libmusa.so.5.1.0 ]; then
  cp -f "${MUSA_RUNTIME}" /usr/lib/x86_64-linux-gnu/libmusa.so.5.1.0
fi

# Container: /workspace/*  ==  Host: /data/feihu/*
BUILD_DIR=build/cmake.linux-x86_64-cpython-310
REF="${XLLM_MUSA_REF}"
# Only seed vcpkg when missing. Never rm+cp on every run (that + cache wipe => vcpkg reinstall loop).
if [ ! -d "${BUILD_DIR}/vcpkg_installed" ] && [ -d "${REF}/build/${BUILD_DIR}/vcpkg_installed" ]; then
  echo "==> First-time init: copy vcpkg_installed from xllm-musa reference (${REF})"
  mkdir -p "${BUILD_DIR}"
  cp -a "${REF}/build/${BUILD_DIR}/vcpkg_installed" "${BUILD_DIR}/"
fi

# FULL_RESET=1 only: wipes cmake cache and forces vcpkg reinstall (avoid in normal use).
if [ "${FULL_RESET:-0}" = "1" ]; then
  echo "==> FULL_RESET: wipe cmake cache (will trigger vcpkg reinstall)"
  rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles" "${BUILD_DIR}/build.ninja" "${BUILD_DIR}/.ninja_deps" "${BUILD_DIR}/.ninja_log" "${BUILD_DIR}/.ninja_log.good" 2>/dev/null || true
  find "${BUILD_DIR}/xllm" "${BUILD_DIR}/third_party" -type d -name CMakeFiles -prune \
    ! -path "*safetensors*" ! -path "*tokenizers*" -exec rm -rf {} + 2>/dev/null || true
fi

BD="${BUILD_DIR}"
REFBD="${REF}/build/${BUILD_DIR}"
# Recover a good cache if missing or poisoned (MANIFEST_INSTALL=ON causes reinstall loop).
if [ ! -f "${BD}/CMakeCache.txt" ] || grep -q "^VCPKG_MANIFEST_INSTALL:BOOL=ON" "${BD}/CMakeCache.txt" 2>/dev/null; then
  if [ -f "${REFBD}/CMakeCache.txt" ]; then
    echo "==> Restore CMakeCache from xllm-musa reference (path rewrite, MANIFEST_INSTALL=OFF)"
    sed "s|${REF}|${SCRIPT_DIR}|g; s|/workspace/xllm_qwen3.5/xllm_0526/xllm|${SCRIPT_DIR}|g" "${REFBD}/CMakeCache.txt" > "${BD}/CMakeCache.txt"
    sed -i "s|^VCPKG_MANIFEST_INSTALL:BOOL=.*|VCPKG_MANIFEST_INSTALL:BOOL=OFF|" "${BD}/CMakeCache.txt"
  fi
fi

mkdir -p .git/hooks && touch .git/hooks/pre-commit
mkdir -p build_logs
LOG=build_logs/build_cuda_graph_musa_$(date +%Y%m%d_%H%M%S).log
echo "==> Logging to ${LOG}"
# NINJA_TARGET: default xllm; set to specific .o targets for incremental rebuilds.
NINJA_TARGET="${NINJA_TARGET:-xllm}"
NINJA_SAFE="${SCRIPT_DIR}/scripts/ninja_safe.sh"
if [ -f "${BD}/build.ninja" ] && [ "${FORCE_CMAKE:-0}" != "1" ]; then
  # MUSA builds may emit -lmusa_layers without the musa/ subdir in the cmake
  # graph; point the linker at the static archive directly.
  MUSA_LAYERS_A="${SCRIPT_DIR}/${BD}/xllm/core/layers/musa/libmusa_layers.a"
  if [ -f "${MUSA_LAYERS_A}" ] && grep -q -- '-lmusa_layers' "${BD}/build.ninja" 2>/dev/null; then
    sed -i "s|-lmusa_layers|${MUSA_LAYERS_A}|g" "${BD}/build.ninja"
  fi
  echo "==> Incremental: ninja only (skip cmake reconfigure), target=${NINJA_TARGET}"
  # A failed/aborted CMake probe can leave the generated cache newer than the
  # manifest.  In that case callers may provide the already validated graph
  # (for example, NINJA_FILE=build.ninja.noregen) while still using this
  # guarded build procedure.  No CUDA toolkit compiler is introduced here.
  NINJA_ARGS=(-j"${MAX_JOBS}")
  if [ -n "${NINJA_FILE:-}" ]; then
    NINJA_ARGS=(-f "${NINJA_FILE}" "${NINJA_ARGS[@]}")
  fi
  NINJA_ARGS+=("${NINJA_TARGET}")
  "${NINJA_SAFE}" "${BD}" "${NINJA_ARGS[@]}" 2>&1 | tee "${LOG}"
else
  echo "==> Configure + build via setup.py"
  exec python3 setup.py build --device musa 2>&1 | tee "${LOG}"
fi
