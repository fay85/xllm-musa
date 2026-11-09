include(cc_test)

add_subdirectory(profile)

# Exercise response-pool ordering without the scheduler's global device setup.
cc_test(
  NAME
    request_phase_output_test
  CPU_ONLY
  SRCS
    request_phase_output_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/scheduler/async_response_processor.cpp
  DEPS
    :config
    :request
    :util
    GTest::gtest_main
)
target_link_libraries(request_phase_output_test PUBLIC
    Python::Python
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:hccl>
    $<$<BOOL:${USE_NPU}>:c_sec>
    $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    scheduler_test
  SRCS
    disagg_pd_scheduler_test.cpp
    continuous_scheduler_test.cpp
    fixed_steps_scheduler_test.cpp
    scheduler_policy_test.cpp
    linear_prefix_admission_policy_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
# BatchFactory fixes DP size on first use; keep DP=2 restore scenarios isolated.
cc_test(
  NAME
    pd_decode_restore_test
  SRCS
    pd_decode_restore_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
cc_test(
  NAME
    pd_inflight_lifetime_test
  CPU_ONLY
  SRCS
    pd_inflight_lifetime_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
# Isolate process-global response metrics without starting model workers.
cc_test(
  NAME
    dit_scheduler_test
  CPU_ONLY
  SRCS
    dit_scheduler_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:torch_python>
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
set_tests_properties(
  DiTResponseMetricsTest.OutputBuildEndsBeforeCallback
  PROPERTIES TIMEOUT 60)

foreach(test_target scheduler_test pd_decode_restore_test pd_inflight_lifetime_test dit_scheduler_test)
  target_link_libraries(${test_target}
                        PUBLIC
                        Python::Python
                        $<$<BOOL:${USE_NPU}>:ascendcl>
                        $<$<BOOL:${USE_NPU}>:hccl>
                        $<$<BOOL:${USE_NPU}>:c_sec>)
  if(test_target STREQUAL "dit_scheduler_test")
    # Rescan execution/request archives without merging shared target cycles.
    # batch.cpp consumes request_timing, so that archive belongs in the rescan
    # group too; otherwise libbatch.a's later repetition reintroduces its
    # symbols after the last request_timing scan.
    target_link_libraries(${test_target} PRIVATE
      "$<LINK_GROUP:RESCAN,xtensor,xllm_server,xgrammar_constraint,$<TARGET_FILE:runtime>,$<TARGET_FILE:batch>,$<TARGET_FILE:request>,$<TARGET_FILE:request_timing>,$<TARGET_FILE:kv_cache>,$<TARGET_FILE:block>,$<TARGET_FILE:block_hasher>>")
  else()
    target_link_libraries(${test_target} PRIVATE
      "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
  endif()
endforeach()
