# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_test)

cc_test(
  NAME musa_attention_test
  SRCS attention_test.cpp
  DEPS :musa_layers GTest::gtest_main glog::glog
  CPU_ONLY
)

cc_test(
  NAME musa_prefill_test
  SRCS prefill_test.cpp
  DEPS :musa_layers :platform :config torch_musa GTest::gtest_main glog::glog
  ENVIRONMENT "FLASHINFER_OPS_PATH=$<TARGET_FILE_DIR:xllm>/musa_ops"
  RUN_SERIAL
)
target_link_libraries(musa_prefill_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")
