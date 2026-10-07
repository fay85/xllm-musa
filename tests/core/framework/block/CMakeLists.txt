include(cc_test)

if(USE_MUSA)
  cc_test(
    NAME deepseek_v4_pd_prefix_policy_test
    SRCS deepseek_v4_pd_prefix_policy_test.cpp
    DEPS :xllm_server :block :config :kv_cache :prefix_cache
         absl::random_random Boost::serialization GTest::gtest_main
  )
  target_link_libraries(deepseek_v4_pd_prefix_policy_test PRIVATE
                        Folly::folly OpenSSL::SSL OpenSSL::Crypto
                        protobuf::libprotobuf)
  add_dependencies(deepseek_v4_pd_prefix_policy_test brpc-static)
endif()

cc_test(
  NAME
    concurrent_block_manager_test
  SRCS
    concurrent_block_manager_test.cpp
  DEPS
    :block
    :config
    :prefix_cache
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME
      block_test
    SRCS
      block_manager_test.cpp
      embedding_block_manager_test.cpp
      composite_block_manager_test.cpp
      hierarchy_block_manager_pool_test.cpp
    DEPS
      :xllm_server
      :block
      :config
      :kv_cache
      :prefix_cache
      absl::random_random
      Boost::serialization
      GTest::gtest_main
  )

  target_link_libraries(block_test PRIVATE Folly::folly OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf c_sec)
  target_link_libraries(block_test PRIVATE
    "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
  add_dependencies(block_test brpc-static)
endif()
