include(cc_test)

cc_test(
  NAME
    host_kv_transfer_test
  CPU_ONLY
  SRCS
    host_transfer/transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    push_route_test
  CPU_ONLY
  SRCS
    push_route_test.cpp
  DEPS
    :push_route
    GTest::gtest_main
)

cc_test(
  NAME
    pd_topology_guard_test
  CPU_ONLY
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

cc_test(
  NAME
    kv_transfer_completion_test
  CPU_ONLY
  SRCS
    kv_transfer_completion_test.cpp
  DEPS
    :config
    :kv_transfer_failure
    :kv_transfer_completion
    :parallel_state
    :platform
    torch
    GTest::gtest_main
)

# Rescan the config/platform/util cycle for util's ModelConfig dependency.
target_link_libraries(kv_transfer_completion_test PRIVATE
  "$<LINK_GROUP:RESCAN,config,platform,util>")

cc_test(
  NAME
    prefetch_result_test
  CPU_ONLY
  SRCS
    prefetch_result_test.cpp
  DEPS
    :util
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    prefetch_tail_probe_test
  CPU_ONLY
  SRCS
    prefetch_tail_probe_test.cpp
  DEPS
    :util
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    reshard_planner_test
  CPU_ONLY
  SRCS
    reshard_planner_test.cpp
  DEPS
    :reshard_planner
    :kv_cache
    GTest::gtest_main
)

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    mooncake_transfer_engine_test
  SRCS
    mooncake_transfer_engine_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

target_link_libraries(mooncake_transfer_engine_test PRIVATE
  "$<LINK_GROUP:RESCAN,xllm_server>")

# These cases initialize local Mooncake RPC endpoints from a shared port
# range. Serialize only endpoint users; pure layout/service tests stay parallel.
set_tests_properties(
  MooncakeKVCacheTransferLayoutTest.PublishedTpWidthAndRankAlwaysAgree
  MooncakeKVCacheTransferLayoutTest.LinearHeadPolicyKeepsWidthAndRankTogether
  MooncakeKVCacheTransferDefaultTest.RegistersAliasedLayerBuffersOnceWithLogicalLayerMappings
  MooncakeKVCacheTransferDefaultTest.RegistersCheckpointedSsmAsLogicalSequenceSlots
  MooncakeKVCacheTransferDefaultTest.Dsv4FingerprintIncludesCompressedCacheGeometry
  MooncakeKVCacheTransferDefaultTest.Dsv41FingerprintIncludesGroupedCacheGeometry
  MooncakeKVCacheTransferDefaultTest.Dsv41GroupedPoolsCarryStateRingWithSwa
  MooncakeKVCacheTransferDefaultTest.RegistersCompressedIndexAndKpoolTail
  MooncakeKVCacheTransferDefaultTest.RegistersDsv4TypedPoolsWithPerPoolSpans
  MooncakeKVCacheTransferDefaultTest.RegistersDraftTp8BesideMainCp4Tp2
  MooncakeKVCacheTransferDefaultTest.Glm5NextTransfersMainAndDraftPagesAndRequestState
  MooncakeKVCacheTransferDefaultTest.RegistersTp1SpecDraftBesideTp2MainCache
  MooncakeKVCacheTransferDefaultTest.NpuLinearIndexerScalePushAndPullRoundTrip
  MooncakeKVCacheTransferDefaultTest.PushStopsAfterLayerFailure
  MooncakeKVCacheTransferDefaultTest.Glm5NextRequestStateRoundTripsLogicalSlots
  MooncakeKVCacheTransferDefaultTest.IndexScaleRegistersAndRoundTripsWithKvBlocks
  PROPERTIES RESOURCE_LOCK mooncake_transfer_engine)
endif()

# Host-only width admission, transfer filtering, and byte-region binding.
# Keep CPU_ONLY so this target does not inherit accelerator bootstrap.
# The block manager and scheduler predicate need xllm_server.
cc_test(
  NAME
    kv_split_width_mismatch_test
  CPU_ONLY
  SRCS
    kv_split_width_mismatch_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

target_link_libraries(kv_split_width_mismatch_test PRIVATE
  "$<LINK_GROUP:RESCAN,xllm_server>")

if(USE_MLU)
cc_test(
  NAME
    compact_load_test
  SRCS
    host_transfer/compact_load_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    compact_offload_test
  SRCS
    host_transfer/compact_offload_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

# The unit executable supplies an in-memory Mooncake backend. Compile Store
# directly so the real Mooncake backend implementation is not linked here.
cc_test(
  NAME
    kv_cache_store_test
  CPU_ONLY
  SRCS
    kv_cache_store_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp
  DEPS
    :kv_cache
    :config
    :util
    mooncake_store
    GTest::gtest_main
)

# Resolve the existing config/platform/util static-library cycle.
target_link_libraries(kv_cache_store_test PRIVATE
  "$<LINK_GROUP:RESCAN,config,platform,util>")

# Pure Mooncake backend policy/validation tests need the real backend code but
# do not require a running master or storage node.
cc_test(
  NAME
    mooncake_store_backend_test
  CPU_ONLY
  SRCS
    mooncake_store_backend_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

# Live Mooncake checks need a configured master and TCP storage node. Keep
# this heavier executable out of the default all_tests and CTest sets; build it
# explicitly with XLLM_ENABLE_MOONCAKE_INTEGRATION_TESTS for manual validation.
# Requires XLLM_TEST_STORE_MASTER to name an active TCP storage service.
option(XLLM_ENABLE_MOONCAKE_INTEGRATION_TESTS
  "Build live Mooncake KV cache store integration tests" OFF)
if(XLLM_ENABLE_MOONCAKE_INTEGRATION_TESTS)
  cc_test(
    NAME
      kv_cache_store_integration_test
    CPU_ONLY
    RUN_SERIAL
    SRCS
      kv_cache_store_integration_test.cpp
    DEPS
      :kv_cache_transfer
      GTest::gtest_main
  )
endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    basic_host_kv_transfer_test
  SRCS
    host_transfer/basic_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    hierarchy_kv_cache_transfer_test
  SRCS
    hierarchy_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    kv_split_host_transfer_test
  SRCS
    kv_split_host_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)
endif()
