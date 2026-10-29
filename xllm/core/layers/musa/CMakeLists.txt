include(cc_library)

# MUSA layers - FlashInfer/Mate path
set(_MUSA_LAYERS_HDRS
  attention.h
  flashinfer_attention.h
  flashinfer_planinfo.h
  ../cuda/base_attention_impl.h
  ../cuda/flashinfer_workspace.h
  normalized_hidden_states.h
  pcp_context.h
  qwen3_gated_delta_net_base.h
  qwen3_next_attention.h
  qwen3_next_gated_delta_net.h
  qwen3_next_hybrid_decoder_layer_base.h
  qwen3_next_decoder_layer_impl.h
  qwen3_5_decoder_layer_impl.h
  qwen3_5_fused_moe.h
  qwen3_5/qwen3_5_vision_attention.h
  rms_norm_gated.h
  musa_fp8_linear.h
  deepseek_v4/deepseek_v4_attention.h
  deepseek_v4/deepseek_v4_decoder_layer.h
  deepseek_v4/dsa_metadata_builder_musa.h
  deepseek_v4/deepseek_v4_gate.h
  deepseek_v4/deepseek_v4_moe.h
)
set(_MUSA_LAYERS_SRCS
  attention.cpp
  flashinfer_attention.cpp
  flashinfer_planinfo.cpp
  ../cuda/flashinfer_workspace.cpp
  pcp_context.cpp
  qwen3_gated_delta_net_base.cpp
  qwen3_next_attention.cpp
  qwen3_next_gated_delta_net.cpp
  qwen3_next_hybrid_decoder_layer_base.cpp
  qwen3_next_decoder_layer_impl.cpp
  qwen3_5_decoder_layer_impl.cpp
  qwen3_5_fused_moe.cpp
  qwen3_5/qwen3_5_vision_attention.cpp
  rms_norm_gated.cpp
  musa_fp8_linear.cpp
  deepseek_v4/deepseek_v4_attention.cpp
  deepseek_v4/deepseek_v4_decoder_layer.cpp
  deepseek_v4/deepseek_v4_pcp_context.cpp
  deepseek_v4/dsa_metadata_builder_musa.cpp
  deepseek_v4/deepseek_v4_gate.cpp
  deepseek_v4/deepseek_v4_moe.cpp
)
set(_DSV4_ROLE_LAYER_SOURCES
  deepseek_v4/deepseek_v4_attention.cpp
  deepseek_v4/deepseek_v4_decoder_layer.cpp
  deepseek_v4/deepseek_v4_gate.cpp
  deepseek_v4/deepseek_v4_moe.cpp
)
set(_DSV4_ROLE_LAYER_HEADERS
  deepseek_v4_moe.h
  deepseek_v4_moe_policy.h
  deepseek_v4_indexer_reference.h
)
set(_DSV4_ROLE_LAYER_MANIFEST "role=${XLLM_DSV4_PD_ROLE}\n")
if(XLLM_DSV4_PD_ROLE STREQUAL "decode")
  set(_DSV4_DECODE_SOURCE_ROOT
      "${CMAKE_SOURCE_DIR}/tools/pd_guard_20261006/decode_overlay/xllm")
  set(_DSV4_ROLE_INCLUDE_ROOT
      "${CMAKE_BINARY_DIR}/generated/dsv4_decode_overlay")
  file(MAKE_DIRECTORY
       "${_DSV4_ROLE_INCLUDE_ROOT}/core/layers/musa/deepseek_v4")
  foreach(_DSV4_HEADER IN LISTS _DSV4_ROLE_LAYER_HEADERS)
    set(_DSV4_HEADER_SOURCE
        "${_DSV4_DECODE_SOURCE_ROOT}/core/layers/musa/deepseek_v4/${_DSV4_HEADER}")
    set(_DSV4_HEADER_OUTPUT
        "${_DSV4_ROLE_INCLUDE_ROOT}/core/layers/musa/deepseek_v4/${_DSV4_HEADER}")
    configure_file("${_DSV4_HEADER_SOURCE}" "${_DSV4_HEADER_OUTPUT}" COPYONLY)
    file(SHA256 "${_DSV4_HEADER_SOURCE}" _DSV4_HEADER_SHA256)
    string(APPEND _DSV4_ROLE_LAYER_MANIFEST
           "header=${_DSV4_HEADER_SOURCE} -> ${_DSV4_HEADER_OUTPUT} sha256=${_DSV4_HEADER_SHA256}\n")
  endforeach()
  foreach(_DSV4_SOURCE IN LISTS _DSV4_ROLE_LAYER_SOURCES)
    list(REMOVE_ITEM _MUSA_LAYERS_SRCS "${_DSV4_SOURCE}")
    set(_DSV4_ROLE_SOURCE
        "${_DSV4_DECODE_SOURCE_ROOT}/core/layers/musa/${_DSV4_SOURCE}")
    list(APPEND _MUSA_LAYERS_SRCS "${_DSV4_ROLE_SOURCE}")
    file(SHA256 "${_DSV4_ROLE_SOURCE}" _DSV4_SOURCE_SHA256)
    string(APPEND _DSV4_ROLE_LAYER_MANIFEST
           "source=${_DSV4_ROLE_SOURCE} sha256=${_DSV4_SOURCE_SHA256}\n")
  endforeach()
else()
  foreach(_DSV4_HEADER IN LISTS _DSV4_ROLE_LAYER_HEADERS)
    set(_DSV4_HEADER_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/deepseek_v4/${_DSV4_HEADER}")
    file(SHA256 "${_DSV4_HEADER_SOURCE}" _DSV4_HEADER_SHA256)
    string(APPEND _DSV4_ROLE_LAYER_MANIFEST
           "header=${_DSV4_HEADER_SOURCE} sha256=${_DSV4_HEADER_SHA256}\n")
  endforeach()
  foreach(_DSV4_SOURCE IN LISTS _DSV4_ROLE_LAYER_SOURCES)
    set(_DSV4_ROLE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${_DSV4_SOURCE}")
    file(SHA256 "${_DSV4_ROLE_SOURCE}" _DSV4_SOURCE_SHA256)
    string(APPEND _DSV4_ROLE_LAYER_MANIFEST
           "source=${_DSV4_ROLE_SOURCE} sha256=${_DSV4_SOURCE_SHA256}\n")
  endforeach()
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dsv4_pd_layer_selection.txt"
     "${_DSV4_ROLE_LAYER_MANIFEST}")


set(_MUSA_LAYERS_DEPS
  :cuda_kernels
  :common_layers
  :model
  :kernels
  :parallel_state
  :state_dict
  :kv_cache
  :prefix_cache
  :block
  glog::glog
  gflags::gflags
  torch
)

cc_library(
  NAME
    musa_layers
  HDRS
    ${_MUSA_LAYERS_HDRS}
  SRCS
    ${_MUSA_LAYERS_SRCS}
  COPTS
    ${_XLLM_MUSIFY_COPTS}
  DEPS
    "-Wl,--whole-archive"
    "-Wl,--no-whole-archive"
    ${_MUSA_LAYERS_DEPS}
)

if(XLLM_DSV4_PD_ROLE STREQUAL "decode")
  target_include_directories(musa_layers BEFORE PUBLIC
      "$<BUILD_INTERFACE:${_DSV4_ROLE_INCLUDE_ROOT}>")
endif()
