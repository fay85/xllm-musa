cmake_minimum_required(VERSION 3.26)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." OFF)

# Convert the bool variable to integer.
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
endif()

project("hccl_transfer" LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/hccl_transfer/include)

include_directories(
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
)

# find all dependencies from vcpkg
find_package(absl CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# brpc
if (DEFINED ENV{VCPKG_ROOT})
    include_directories("$ENV{VCPKG_ROOT}/packages/zlib_x64-linux/include")
    link_directories("$ENV{VCPKG_ROOT}/packages/zlib_x64-linux/lib")
endif()
set(BRPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/third_party/brpc/output)
include_directories(${BRPC_OUTPUT_DIR}/include)
link_directories(${BRPC_OUTPUT_DIR}/lib)

# add subdirectories
add_subdirectory(hccl_transfer)
add_subdirectory(third_party)