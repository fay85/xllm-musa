# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

# The pinned upstream config enables Python bindings. Define only its CPU
# target here, preserving xLLM's compiler and ABI options.
set(XGRAMMAR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/xgrammar")

file(GLOB_RECURSE XGRAMMAR_CPU_SOURCES CONFIGURE_DEPENDS
  "${XGRAMMAR_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_CPU_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
add_library(xgrammar STATIC ${XGRAMMAR_CPU_SOURCES})
target_include_directories(xgrammar PUBLIC "${XGRAMMAR_SOURCE_DIR}/include")
target_include_directories(xgrammar SYSTEM PUBLIC
  "${XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  "${XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include")
target_compile_features(xgrammar PUBLIC cxx_std_17)
target_compile_definitions(xgrammar PRIVATE
  XGRAMMAR_ENABLE_CPPTRACE=0
  XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
find_package(Threads REQUIRED)
target_link_libraries(xgrammar PUBLIC Threads::Threads)
