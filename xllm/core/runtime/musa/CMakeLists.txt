# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_library)

cc_library(
  NAME
    musa_graph_executor
  HDRS
    musa_graph_executor_impl.h
  SRCS
    musa_graph_executor_impl.cpp
  DEPS
    torch
    torch_musa
    glog::glog
    absl::flat_hash_map
    :common
    :platform
    :util
    :cuda_kernels
    :musa_graph_output_buffers
    :kv_cache
    :model
    :layers
    :request
    :parallel_state
    :model_context
    :sampler
)
