<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# Separate host C++ and MUSA compilers

Set `XLLM_MUSA_HOST_CXX_COMPILER` to an installed host compiler when configuring
a MUSA build. For example, use Clang 20 for ordinary C++ while retaining the
installed MUSA SDK compiler for device code and CUDA compatibility conversion:

```bash
cmake -S . -B build-musa-hybrid -G Ninja \
  -DUSE_MUSA=ON -DUSE_CUDA=OFF \
  -DXLLM_MUSA_HOST_CXX_COMPILER=/path/to/clang++-20 \
  <the normal dependency and MUSA configuration options>
cmake --build build-musa-hybrid --target xllm --parallel 32
```

`setup.py` also accepts CMake options through its existing `CMAKE_ARGS` input:

```bash
CMAKE_ARGS="-DXLLM_MUSA_HOST_CXX_COMPILER=/path/to/clang++-20" \
  python setup.py build
```

When this option is empty, CMake searches `PATH` for `clang++-20`. If none is
available, configuration fails early with an instruction to install it or set
the host compiler explicitly. It does not silently send ordinary C++ back to the
older SDK frontend. Use a fresh build directory when switching compiler
configurations. Installing a separate host compiler does not replace MCC or the
SDK's Clang-based device frontend.

The generated build-directory drivers pin the resolved host compiler and SDK
wrapper paths at configure time. A later plain Ninja invocation does not need
temporary compiler-selection environment variables. The source template is
`cmake/musa_compiler.sh.in`.

The C++ driver selects MCC for `.mu`/`.cu`, explicit MUSA language or
`libMusaMapping.so` compilation, and the legacy graph-conversion translation
units. Other C++ compilation uses the configured host compiler. The device
driver always selects MCC. Final object-file links retain the SDK wrapper's
MUSA/Torch library mapping. The SDK wrapper is invoked with Bash errexit so a
failed compiler or linker is not hidden by an unconditional successful exit.

Both compilers must use compatible host ABI settings and standard-library
inputs. With the existing Linux Torch/MUSA distribution, retain `libstdc++`
and the matching `_GLIBCXX_USE_CXX11_ABI` value. For Clang host compilers,
`XLLM_MUSA_HOST_CLANG_ABI_COMPAT` defaults to `14`, matching the existing MCC 5.1
and PyTorch artifacts; this adds `-fclang-abi-compat=14` for mangling compatibility.
It does not downgrade the Clang 20 frontend or its C++20 language support. Change
this setting only with an appropriately matched dependency toolchain. Do not load the SDK's Clang AST
conversion plugin into an unrelated upstream Clang version.

Shared headers and C++ files requiring MUSA extensions still need to compile
with MCC. Consequently a newer host compiler does not make every source
compatibility change removable. Validate the actual affected translation
units, the full executable, and model requests before removing such changes.
