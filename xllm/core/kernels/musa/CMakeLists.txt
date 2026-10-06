include(cc_library)

# MUSA kernel build - uses FlashInfer/Mate path with mcc_wrapper
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda/moe)

set(_TORCH_MUSA_KERNEL_HDRS
  musa_ops_api.h
  musa_kernel_utils.h
  musa_fp8_quant_utils.cuh
  deepseek_v4_dsa_rope.h
  deepseek_v4_fp8_roundtrip.h
  deepseek_v4_indexer_metadata.h
  musa_tvmffi_stream.h
  gdn_ops.h
  llm_decode_metadata_update.h
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_sigmoid_kernels.cuh
  ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_softmax_kernels.cuh
)

set(_TORCH_MUSA_KERNEL_SRCS
  attention_decode.cpp
  attention_prefill.cpp
  gdn_decode.cu
  gdn_prefill.cpp
  causal_conv1d.cu
  gemma_norm.cu
  activation.cu
  rope.cu
  gdn_state_commit.cu
  ../cuda/block_copy.cu
  ../cuda/norm.cu
  ../cuda/reshape_paged_cache.cu
  ../cuda/moe/moe_fused_topk.cu
  random_sample.cpp
  rejection_sample.cu
  replace_token.cu
  fused_qknorm_rope.cu
  # Native block-wise FP8 GEMM (mate/muDNN groupwise)
  fp8_block_gemm.cpp
  # Mate masked grouped GEMM used by the Qwen3.5 routed experts.
  musa_moe_gemm.cpp
  # MP31 AOT fused-MoE decode dataflow.
  musa_fused_moe_aot.cpp
  # Graph-safe expert histogram/prefix/placement used by Qwen3.5 routing.
  ../cuda/moe/moe_compute_index.cu
  ../cuda/moe/moe_combine.cu
  # MUSA-only indexed combine for expert-major GEMM output.
  moe_combine_indexed.cu
  # Fused per-token-group FP8 activation quant (MUSA-native intrinsics)
  fp8_act_quant.cu
  # Mate FFI helpers
  musa_tvmffi_stream.cpp
  llm_decode_metadata_update.cu
  deepseek_v4_sp_pad.cu
  deepseek_v4_hc_pre.cu
  deepseek_v4_hc_head.cu
  deepseek_v4_wo_a_m1.cu
  deepseek_v4_compress_c4_prepare.cu
  deepseek_v4_rope_hadamard.cu
  deepseek_v4_q_rmsnorm_rope.cu
  deepseek_v4_mega_moe_pre_dispatch.cu
  deepseek_v4_masked_swiglu_quant.cu
  deepseek_v4_fp8_moe_metadata.cu
  deepseek_v4_gate_bf16_fp32.cpp
  deepseek_v4_masked_fp8_moe_gemm.cpp
  deepseek_v4_fp8_moe_gemv.cu
  deepseek_v4_dsa_rope.cu
  deepseek_v4_gate_gemv.cu
  deepseek_v4_gate_topk.cu
  deepseek_v4_shared_expert_gate.cu
  deepseek_v4_fp8_roundtrip.cu
  deepseek_v4_dsa_cache_ops.cu
  deepseek_v4_indexer_metadata.cu
  deepseek_v4_compress_batch.cu
  deepseek_v4_mate_partition_metadata.cu
  deepseek_v4_dsa_attention.cu
)

# Enable fast-math for fp8_act_quant.cu only, without changing
# floating-point flags for the other kernels in this library.
set_source_files_properties(
  fp8_act_quant.cu
  PROPERTIES
    COMPILE_OPTIONS "-ffast-math"
)

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${_TORCH_MUSA_KERNEL_HDRS}
  SRCS
    ${_TORCH_MUSA_KERNEL_SRCS}
  COPTS
    ${_XLLM_MUSIFY_COPTS}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
    musa
    musart
)
