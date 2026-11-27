include(cc_test)

cc_test(
  NAME
    python_subprocess_lifecycle_test
  CPU_ONLY
  SRCS
    python_subprocess_lifecycle_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/distributed_runtime/spawn_worker_server/spawn_worker_server_signal.cpp
  DEPS
    :python_runtime
    GTest::gtest_main
  LINKOPTS
    -Wl,--wrap=alarm
)

cc_test(
  NAME
    util_test
  CPU_ONLY
  SRCS
    binary_payload_test.cpp
    blocking_counter_test.cpp
    device_name_utils_test.cpp
    dspark_model_type_test.cpp
    http_downloader_test.cpp
    model_path_utils_test.cpp
    net_test.cpp
    shared_memory_manager_test.cpp
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
