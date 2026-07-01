if(USE_CUDA)
  if(DEFINED ENV{MUSA_HOME})
    set(_XLLM_MUSA_HOME "$ENV{MUSA_HOME}")
  else()
    set(_XLLM_MUSA_HOME "/usr/local/musa")
  endif()
  if(XLLM_TORCH_MUSA)
    set(CMAKE_CXX_COMPILER "${_XLLM_MUSA_HOME}/tools/musamapping/mcc_wrapper")
  else()
    set(CMAKE_CXX_COMPILER "${_XLLM_MUSA_HOME}/bin/mcc")
  endif()
endif()

include(cc_binary)
include(cc_shared_library)

include_directories(.)

# xllm-musa: musify only on kernel + cuda-graph targets (not server/Mooncake TUs).
if(XLLM_TORCH_MUSA)
  if(DEFINED ENV{MUSAMAPPING_PATH} AND NOT "$ENV{MUSAMAPPING_PATH}" STREQUAL "")
    set(_XLLM_MUSAMAPPING_PATH "$ENV{MUSAMAPPING_PATH}")
  elseif(DEFINED _XLLM_MUSA_HOME)
    set(_XLLM_MUSAMAPPING_PATH "${_XLLM_MUSA_HOME}/tools/musamapping")
  else()
    set(_XLLM_MUSAMAPPING_PATH "/usr/local/musa/tools/musamapping")
  endif()
  set(_XLLM_MUSIFY_COPTS
    "-fplugin=${_XLLM_MUSAMAPPING_PATH}/libMusaMapping.so"
    "-x" "musa"
    "-I."
  )
else()
  set(_XLLM_MUSIFY_COPTS "")
endif()


# Generate build-time version header from version.txt.
file(READ "${CMAKE_SOURCE_DIR}/version.txt" XLLM_BUILD_VERSION_RAW)
string(STRIP "${XLLM_BUILD_VERSION_RAW}" XLLM_BUILD_VERSION)
if(XLLM_BUILD_VERSION MATCHES "^v(.+)$")
  set(XLLM_BUILD_VERSION "${CMAKE_MATCH_1}")
endif()
if(XLLM_BUILD_VERSION STREQUAL "")
  set(XLLM_BUILD_VERSION "unknown")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/core/common")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/core/common/xllm_build_info.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/core/common/xllm_build_info.h"
  @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# 0526 reference: mcc_wrapper build skips -Werror; suppress known mcc noise only.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(NOT XLLM_TORCH_MUSA)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:C>:-Werror>
    )
  else()
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-defaulted-function-deleted>
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion>
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-inconsistent-missing-override>
    )
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
      $<$<COMPILE_LANGUAGE:C>:-Wno-macro-redefined>
    )
  endif()

  if(XLLM_TORCH_MUSA OR DEFINED ENV{MUSAMAPPING_PATH} OR CMAKE_CXX_COMPILER MATCHES "mcc_wrapper")
    add_compile_options(
      -Wno-sign-conversion
      -Wno-conversion
      -Wno-error=sign-conversion
      -Wno-error=conversion
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-macro-redefined>"
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-error=macro-redefined>"
    )
  endif()
endif()

option(GENERATE_SO "Enable compile to generate .so" OFF)

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(parser)
add_subdirectory(processors)
add_subdirectory(proto)
add_subdirectory(pybind)
add_subdirectory(server)

if(GENERATE_SO)
  message(STATUS "build libxllm.so and install")
  cc_shared_library(
    NAME
      xllm
    HDRS
      c_api/llm.h
      c_api/rec.h
      c_api/default.h
      c_api/types.h
      c_api/internal/helper.h
    SRCS
      c_api/internal/llm.cpp
      c_api/internal/rec.cpp
      c_api/internal/helper.cpp
    DEPS
      :config
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

else()
  message(STATUS "build xllm binary and install")
  cc_binary(
    NAME
      xllm
    SRCS
      xllm.cpp
    DEPS
      :config
      :xllm_server
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

endif()

# link brpc
target_link_libraries(xllm PRIVATE glog::glog brpc leveldb::leveldb protobuf::libprotobuf ${FFMPEG_LIBRARIES})
if(NOT XLLM_TORCH_MUSA)
  target_link_libraries(xllm PRIVATE ${OpenCV_LIBS})
endif()
add_dependencies(xllm brpc-static)

if(USE_NPU)
  set(COMMON_LIBS torch_npu torch_python Python::Python ascendcl atb_customize unified_dlog hccl c_sec nnopbase ms_tools_ext)
elseif(USE_MLU)
  set(COMMON_LIBS Python::Python)
endif()

if(USE_MSPTI)
  list(APPEND COMMON_LIBS mspti)
endif()
target_link_libraries(xllm PUBLIC ${COMMON_LIBS})
if (USE_MUSA)
  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)
elseif(XLLM_TORCH_MUSA)
  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)
endif()

if(USE_CUDA AND XLLM_TORCH_MUSA)
  # cuda_graph_executor_impl.cpp's only externally-observable effect is the
  # REGISTER_EXECUTOR static initializer in cuda_graph_executor_impl.h. With
  # normal lazy static-archive linking, no symbol references it so the linker
  # drops the whole .o and ENABLE_GRAPH=1 silently falls back to
  # BaseExecutorImpl (factory throws "No valid backend found: musa" at
  # runtime). Wrap libcuda_graph_executor in --whole-archive at the executable
  # link to force the .o (and thus the registration) into the binary.
  # Same problem applies to base_executor_impl.cpp in libruntime.a: it carries
  # REGISTER_EXECUTOR("llm", BaseExecutorImpl) / REGISTER_EXECUTOR("rec", ...)
  # static initializers and has no external symbol referenced, so the linker
  # drops the TU and backend="llm" lookups fall through to "vlm" with a null
  # model encode (segfault on first forward). Force libruntime in as well.
  target_link_options(xllm PRIVATE
    "LINKER:--whole-archive,$<TARGET_FILE:cuda_graph_executor>,--no-whole-archive"
    "LINKER:--whole-archive,$<TARGET_FILE:runtime>,--no-whole-archive"
  )
endif()

# install xllm
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install all dependencies for xllm
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
        RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
        UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
        EXECUTABLES $<TARGET_FILE:xllm>)

  file(INSTALL
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        FILES ${DEPENDENCIES}
        FOLLOW_SYMLINK_CHAIN)

  # This should not be possible, but error out when a dependency cannot
  # be resolved.
  list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
  if(${UNRESOLVED_LENGTH} GREATER 0)
      message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
  endif()
]])
