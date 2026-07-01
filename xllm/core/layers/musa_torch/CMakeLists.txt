include(cc_library)

# Qwen3 / Qwen3.5 hybrid torch layers for XLLM_TORCH_MUSA (USE_CUDA + mcc_wrapper).
# Platform-neutral torch C++ sources with cuda_layers deps; no NPU stack.
if(XLLM_TORCH_MUSA)
  cc_library(
    NAME
      musa_torch_layers
    HDRS
      qwen3_gated_delta_net_base.h
      qwen3_next_attention.h
      qwen3_next_gated_delta_net.h
      qwen3_5_gated_delta_net.h
      qwen3_next_hybrid_decoder_layer_base.h
      qwen3_next_decoder_layer_impl.h
      qwen3_5_decoder_layer_impl.h
    SRCS
      qwen3_gated_delta_net_base.cpp
      qwen3_next_attention.cpp
      qwen3_next_gated_delta_net.cpp
      qwen3_5_gated_delta_net.cpp
      qwen3_next_hybrid_decoder_layer_base.cpp
      qwen3_next_decoder_layer_impl.cpp
      qwen3_5_decoder_layer_impl.cpp
    DEPS
      :common_layers
      :cuda_layers
      :parallel_state
      :state_dict
      :kv_cache
      :prefix_cache
      :block
      glog::glog
      gflags::gflags
      torch
  )
endif()
