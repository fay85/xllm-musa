include(cc_library)

cc_library(
  NAME
    parallel_state
  HDRS
    mapping_npu.h
    npu_cp_plan.h
    npu_dp_ep_padding.h
    dit_mapping.h
    rank_generator.h
    parallel_args.h
    parallel_state.h
    process_group.h
    $<$<BOOL:${USE_NPU}>:npu_rank_table_env.h>
    $<$<BOOL:${USE_NPU}>:npu_process_group.h>
    $<$<BOOL:${USE_MLU}>:mlu_process_group.h>
    $<$<BOOL:${USE_MUSA}>:musa_process_group.h>
    $<$<BOOL:${USE_MUSA}>:musa/musa_custom_all_reduce.h>
    $<$<BOOL:${USE_MUSA}>:musa/musa_native_collectives.h>
    $<$<BOOL:${USE_MUSA}>:musa/musa_custom_all_reduce_policy.h>
    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_DCU}>>:cuda_process_group.h>
    $<$<BOOL:${USE_ILU}>:ilu_process_group.h>
    collective_communicator_base.h
    collective_communicator.h
    context_parallel_topology.h
    dit_collective_communicator.h
    $<$<BOOL:${USE_NPU}>:mega_moe_comm_resource.h>
  SRCS
    mapping_npu.cpp
    npu_cp_plan.cpp
    npu_dp_ep_padding.cpp
    dit_mapping.cpp
    parallel_state.cpp
    parallel_state_async.cpp
    process_group.cpp
    $<$<BOOL:${USE_MUSA}>:musa/musa_custom_all_reduce.cpp>
    $<$<BOOL:${USE_NPU}>:npu_rank_table_env.cpp>
    $<$<BOOL:${USE_NPU}>:npu_process_group.cpp>
    $<$<BOOL:${USE_NPU}>:mega_moe_comm_resource.cpp>
    collective_communicator.cpp
    context_parallel_topology.cpp
    dit_collective_communicator.cpp
  DEPS
    :common
    torch
    $<$<BOOL:${USE_MLU}>:torch_mlu>
    $<$<BOOL:${USE_MUSA}>:torch_musa>
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:hccl>
    $<$<BOOL:${USE_NPU}>:hcomm>
    $<$<BOOL:${USE_NPU}>:xllm_atb_layers>
    gflags::gflags
    glog::glog
)

if(USE_MUSA)
  # mccl.h exposes BF16/FP8 enum values only with a host architecture macro.
  # This matches the repository's mp_31 device compilation default; deployments
  # targeting another MUSA architecture must set the corresponding MCCL value.
  set(XLLM_MUSA_MARCH_TYPE 310 CACHE STRING
      "MCCL host API architecture (310 for mp_31)")
  target_compile_definitions(parallel_state PRIVATE
                            MARCH_TYPE=${XLLM_MUSA_MARCH_TYPE})
  find_library(XLLM_MCCL_LIBRARY mccl
               HINTS "${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" REQUIRED)
  target_link_libraries(parallel_state PUBLIC ${XLLM_MCCL_LIBRARY})
endif()
