/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include "runtime/forward_params.h"
#include "runtime/spec_input_builder.h"

namespace xllm {
namespace {

TEST(SpecInputLayoutTest, RebuiltTokensInvalidateSerializedLayout) {
  ForwardInput input;
  input.input_host_buffer = torch::zeros({32}, torch::kUInt8);
  input.input_host_buffer_has_layout = true;
  const torch::Tensor original_backing = input.input_host_buffer;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  specBuilder::set_token_position_tensors(
      input, {10, 11}, {64, 65}, options, options);

  EXPECT_FALSE(input.input_host_buffer_has_layout);
  EXPECT_TRUE(input.input_host_buffer.is_same(original_backing));
  EXPECT_TRUE(
      torch::equal(input.token_ids_host, torch::tensor({10, 11}, options)));
  EXPECT_TRUE(
      torch::equal(input.positions_host, torch::tensor({64, 65}, options)));
}

}  // namespace
}  // namespace xllm
