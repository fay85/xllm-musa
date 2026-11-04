include(cc_test)

cc_test(
  NAME
    util_test
  CPU_ONLY
  SRCS
    binary_payload_test.cpp
    blocking_counter_test.cpp
    device_name_utils_test.cpp
    http_downloader_test.cpp
    model_path_utils_test.cpp
    net_test.cpp
    shared_memory_manager_test.cpp
    suffix_decoding_cache_test.cpp
    tensor_proto_test.cpp
    threadpool_test.cpp
    verbose_trace_logger_test.cpp
    vocab_validation_test.cpp
  DEPS
    util
    :platform
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
)
target_link_libraries(util_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(util_test brpc-static)

# Suppress actual madvise calls: the negative control uses an alternate stack.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  cc_test(
    NAME folly_bthread_compat_test
    CPU_ONLY
    SRCS folly_bthread_compat_test.cpp
    DEPS Folly::folly GTest::gtest_main
    LINKOPTS "-Wl,--wrap=madvise"
  )
endif()
