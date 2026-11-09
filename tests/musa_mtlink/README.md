<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# MTLink host-side regression tests

These standalone tests compile the current production policy, execution binding,
retirement trace, and GPU ticket ledger headers. Queue tests use actual xLLM
`ThreadPool`/`BlockingCounter` and Folly futures. The ledger tests provide controlled
ticket observations through the production interface; their synthetic READY
results are **not evidence of GPU completion**.

The suite runs 30 processes: 17 queue policy/progress cases, 12 binding/state
regression cases, and one ledger executable covering six groups of edge cases.
It does not start MUSA, load a model, exercise RPCs, or measure transport speed.

## Build and run

Use Linux, a C++20 compiler, CMake 3.20+, Bash, and the same built Folly/glog
dependencies and C++ ABI as xLLM. To avoid rebuilding unrelated utility code,
supply the existing production `threadpool.cpp.o` and `cpu_affinity.cpp.o` from a
build of this checkout. Those two source files are unchanged by this branch.
Do not supply objects from a different source revision or ABI.

From the repository root, with `XLLM_BUILD` set to the existing CMake build
directory (the directory containing `vcpkg_installed`):

```bash
cmake -S tests/musa_mtlink -B /tmp/xllm-mtlink-cpu-tests \
  -DXLLM_VCPKG_RELEASE_PREFIX="$XLLM_BUILD/vcpkg_installed/x64-linux" \
  -DXLLM_UTIL_OBJECT_DIR="$XLLM_BUILD/xllm/core/util/CMakeFiles/util.dir" \
  -DXLLM_CXX11_ABI=1
cmake --build /tmp/xllm-mtlink-cpu-tests --parallel 2
ctest --test-dir /tmp/xllm-mtlink-cpu-tests --output-on-failure
```

All generated files stay in the supplied output directory. Intentional invalid
policy cases disable core dumps and must fail with the policy validation message.
`USE_MUSA` on these CPU targets selects the actual policy/trace code, without
linking MUSA or calling GPU APIs. The queue test is also compiled without that
define to check that other backends ignore the MUSA-only environment variable.

The example explicitly selects release archives because the development image
omits vcpkg debug archives, which its generated package configs still reference.
It links the actual release archives in a linker group without changing them.
With a complete dependency installation, omit `XLLM_VCPKG_RELEASE_PREFIX` and
set `CMAKE_PREFIX_PATH` to use the normal Folly/glog CMake package targets instead.
