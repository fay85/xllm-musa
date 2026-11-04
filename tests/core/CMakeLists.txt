include(cc_test)

cc_test(
  NAME forked_rank_session_test
  CPU_ONLY
  SRCS forked_rank_session_test.cpp
  DEPS GTest::gtest_main
)

add_subdirectory(common)
add_subdirectory(distributed_runtime)
add_subdirectory(framework)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(platform)
add_subdirectory(runtime)
add_subdirectory(scheduler)
add_subdirectory(util)
