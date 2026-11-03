#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
# Build one DeepSeek-V4 MUSA role. Use separate source directories for P and D:
# setup.py has one CMake/output tree per source directory and Python version.
# Every invocation configures through setup.py and reuses existing build objects.
set -euo pipefail

if [[ $# -ne 1 || ( $1 != prefill && $1 != decode ) ]]; then
  echo "Usage: $0 {prefill|decode}" >&2
  exit 2
fi
role=$1
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"

export MAX_JOBS=${MAX_JOBS:-32}
if [[ ! $MAX_JOBS =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer." >&2
  exit 2
fi

# Match scripts/build_support/utils.py:get_cmake_dir without importing build
# dependencies or creating a build tree before checking its role.
build_tree=$(python3 - <<'PY'
import sys
import sysconfig

version = sysconfig.get_python_version().replace(".", "")
print(f"build/cmake.{sysconfig.get_platform()}-{sys.implementation.name}-{version}")
PY
)
cache=$build_tree/CMakeCache.txt
if [[ -f $cache ]]; then
  configured_role=$(sed -n 's/^XLLM_DSV4_PD_ROLE:[^=]*=//p' "$cache")
  if [[ $configured_role != "$role" ]]; then
    echo "Refusing to change this build tree from '${configured_role:-unknown}' to '$role'." >&2
    echo "Use an independent source directory for each prefill/decode role." >&2
    exit 2
  fi
fi

# The positional role is authoritative; preserve other caller CMake options.
# setup.py also honors XLLM_USE_ETCD and selects the installed Torch ABI.
export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DXLLM_DSV4_PD_ROLE=$role"
python3 setup.py build --device musa --generate-so false
