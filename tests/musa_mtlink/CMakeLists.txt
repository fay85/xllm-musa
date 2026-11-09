# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(xllm_musa_mtlink_cpu_tests LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The production xLLM ThreadPool objects require Linux.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(XLLM_CXX11_ABI "1" CACHE STRING "ABI used to build the supplied dependencies")
set(XLLM_UTIL_OBJECT_DIR "" CACHE PATH
    "Existing xllm/core/util/CMakeFiles/util.dir with production CPU objects")
set(XLLM_VCPKG_RELEASE_PREFIX "" CACHE PATH
    "Explicit release-only vcpkg prefix for images without debug archives")
get_filename_component(XLLM_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_package(Threads REQUIRED)
find_program(BASH_EXECUTABLE bash REQUIRED)
add_library(cpu_dependencies INTERFACE)
if(XLLM_VCPKG_RELEASE_PREFIX)
  foreach(header folly/futures/Future.h glog/logging.h)
    if(NOT EXISTS "${XLLM_VCPKG_RELEASE_PREFIX}/include/${header}")
      message(FATAL_ERROR "Missing dependency header: ${header}")
    endif()
  endforeach()
  # Release images may deliberately omit debug archives while upstream config
  # files still require them. This explicit mode only uses existing release
  # artifacts, and never changes or rebuilds the dependency installation.
  file(GLOB release_archives CONFIGURE_DEPENDS
       "${XLLM_VCPKG_RELEASE_PREFIX}/lib/lib*.a")
  foreach(library folly glog gflags)
    if(NOT EXISTS "${XLLM_VCPKG_RELEASE_PREFIX}/lib/lib${library}.a")
      message(FATAL_ERROR "Missing release dependency: lib${library}.a")
    endif()
  endforeach()
  target_include_directories(cpu_dependencies INTERFACE
      "${XLLM_VCPKG_RELEASE_PREFIX}/include")
  target_link_libraries(cpu_dependencies INTERFACE
      "-Wl,--start-group" ${release_archives} "-Wl,--end-group"
      Threads::Threads dl rt resolv ssl crypto z m numa)
else()
  find_package(folly CONFIG REQUIRED)
  find_package(glog CONFIG REQUIRED)
  target_link_libraries(cpu_dependencies INTERFACE
      Folly::folly glog::glog Threads::Threads)
endif()

set(UTIL_CPU_OBJECTS
    "${XLLM_UTIL_OBJECT_DIR}/threadpool.cpp.o"
    "${XLLM_UTIL_OBJECT_DIR}/cpu_affinity.cpp.o")
foreach(object IN LISTS UTIL_CPU_OBJECTS)
  if(NOT EXISTS "${object}")
    message(FATAL_ERROR "Missing production CPU object: ${object}")
  endif()
endforeach()

function(add_cpu_binary name source)
  add_executable(${name} "${source}")
  target_include_directories(${name} PRIVATE
      "${XLLM_SOURCE_ROOT}/xllm" "${XLLM_SOURCE_ROOT}/xllm/core")
  target_compile_definitions(${name} PRIVATE
      _GLIBCXX_USE_CXX11_ABI=${XLLM_CXX11_ABI})
  target_link_libraries(${name} PRIVATE cpu_dependencies)
endfunction()

add_cpu_binary(queue_musa queue_policy_cpu_test.cpp)
target_compile_definitions(queue_musa PRIVATE USE_MUSA)
target_sources(queue_musa PRIVATE ${UTIL_CPU_OBJECTS})
add_cpu_binary(queue_other queue_policy_cpu_test.cpp)
target_sources(queue_other PRIVATE ${UTIL_CPU_OBJECTS})

add_cpu_binary(execution_binding execution_binding_cpu_test.cpp)
add_cpu_binary(retirement_state retirement_state_cpu_test.cpp)
add_cpu_binary(gpu_ledger gpu_ledger_cpu_test.cpp)
foreach(binary execution_binding retirement_state gpu_ledger)
  target_compile_definitions(${binary} PRIVATE USE_MUSA)
endforeach()

enable_testing()

function(add_cpu_case name binary argument variable value expected)
  add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}"
      "-DBINARY=$<TARGET_FILE:${binary}>"
      "-DBINARY_ARGUMENT=${argument}"
      "-DVARIABLE=${variable}"
      "-DVARIABLE_VALUE=${value}"
      "-DEXPECTED=${expected}"
      "-DBASH_EXECUTABLE=${BASH_EXECUTABLE}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_case.cmake")
  set_tests_properties(${name} PROPERTIES TIMEOUT 15)
endfunction()

set(QUEUE_VARIABLE XLLM_MUSA_PD_COPY_QUEUE)
add_cpu_case(queue_musa_default_policy queue_musa --policy ${QUEUE_VARIABLE} __UNSET__ request)
add_cpu_case(queue_musa_zero_policy queue_musa --policy ${QUEUE_VARIABLE} 0 request)
add_cpu_case(queue_musa_one_policy queue_musa --policy ${QUEUE_VARIABLE} 1 copy)
add_cpu_case(queue_musa_default_progress queue_musa --request ${QUEUE_VARIABLE} __UNSET__ passed)
add_cpu_case(queue_musa_zero_progress queue_musa --request ${QUEUE_VARIABLE} 0 passed)
add_cpu_case(queue_musa_copy_progress queue_musa --copy ${QUEUE_VARIABLE} 1 passed)
add_cpu_case(queue_other_default_policy queue_other --policy ${QUEUE_VARIABLE} __UNSET__ request)
add_cpu_case(queue_other_zero_policy queue_other --policy ${QUEUE_VARIABLE} 0 request)
add_cpu_case(queue_other_one_policy queue_other --policy ${QUEUE_VARIABLE} 1 request)
add_cpu_case(queue_other_bogus_policy queue_other --policy ${QUEUE_VARIABLE} bogus request)
add_cpu_case(queue_other_request_progress queue_other --request ${QUEUE_VARIABLE} 1 passed)
foreach(value "" 01 true 2 -1 bogus)
  string(MAKE_C_IDENTIFIER "${value}" identifier)
  add_cpu_case(queue_invalid_${identifier} queue_musa --policy ${QUEUE_VARIABLE} "${value}" __FATAL__)
endforeach()

set(TRACE_VARIABLE XLLM_MUSA_PD_RETIRE_TRACE)
add_cpu_case(binding_on execution_binding on ${TRACE_VARIABLE} __UNSET__ passed)
add_cpu_case(binding_default_off execution_binding off ${TRACE_VARIABLE} __UNSET__ default_off)
add_cpu_case(state_on retirement_state on ${TRACE_VARIABLE} __UNSET__ passed)
add_cpu_case(state_default_off retirement_state off ${TRACE_VARIABLE} __UNSET__ default_off)
add_cpu_case(state_zero retirement_state 0 ${TRACE_VARIABLE} __UNSET__ "\"enabled\":0")
add_cpu_case(state_one retirement_state 1 ${TRACE_VARIABLE} __UNSET__ "\"enabled\":1")
foreach(value "" 01 true 2 -1 bogus)
  string(MAKE_C_IDENTIFIER "${value}" identifier)
  add_cpu_case(state_invalid_${identifier} retirement_state "${value}" ${TRACE_VARIABLE} __UNSET__ __FATAL__)
endforeach()
add_cpu_case(ledger_cpu gpu_ledger __NONE__ ${TRACE_VARIABLE} __UNSET__ passed)
