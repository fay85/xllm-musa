#!/bin/bash
# Builds the xllm binary start_stack.sh runs.
# - With an existing CMake build tree (as in the benchmark image): the
#   offline incremental build the published binary came from, i.e. the
#   environment of setup.py's MUSA build plus `cmake --build` of the xllm
#   target. MAX_JOBS sets the parallelism (default 32).
# - Without one (a fresh clone): the full `python3 setup.py build --device
#   musa`, which needs `git submodule update --init --recursive` first.
set -euo pipefail
source "$(dirname "$0")/env.sh"

cd "$REPO_DIR"
build_tree=build/cmake.linux-x86_64-cpython-310
if [[ -f $build_tree/build.ninja ]]; then
  eval "$(python3 - <<'PY'
import os
import shlex
import sys

sys.path.insert(0, os.getcwd())
before = dict(os.environ)
from scripts.build_support import env as build_env

build_env.set_common_envs()
build_env.set_musa_envs()
for key, value in os.environ.items():
    if before.get(key) != value:
        print(f"export {key}={shlex.quote(value)}")
PY
)"
  cmake --build "$build_tree" --target xllm -j "${MAX_JOBS:-32}"
else
  python3 setup.py build --device musa
fi
echo "xllm: $XLLM_BIN (md5 $(md5sum < "$XLLM_BIN" | cut -c1-32))"
