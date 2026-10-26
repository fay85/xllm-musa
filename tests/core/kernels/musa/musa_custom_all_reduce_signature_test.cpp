/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>
#include <torch/types.h>

#include <cstdint>
#include <vector>

namespace xllm {

// Pull the registration object from the static kernel archive, as the
// production MusaCustomAllReduce initialization does.
void register_musa_custom_all_reduce_kernel();

namespace {

TEST(MusaCustomAllReduceSignatureTest, DispatchesMetadataWithoutDevice) {
  register_musa_custom_all_reduce_kernel();
  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("_C_custom_ar::meta_size", "")
                .typed<int64_t()>();
  EXPECT_GT(op.call(), 0);
}

TEST(MusaCustomAllReduceSignatureTest, AcceptsIpcHandleReferenceContract) {
  register_musa_custom_all_reduce_kernel();
  EXPECT_NO_THROW((c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::open_mem_handle", "")
                       .typed<int64_t(torch::Tensor&)>()));
}

TEST(MusaCustomAllReduceSignatureTest, AcceptsRankDataReferenceContract) {
  register_musa_custom_all_reduce_kernel();
  using InitSignature =
      int64_t(const std::vector<int64_t>&, torch::Tensor&, int64_t, bool);
  EXPECT_NO_THROW((c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::init_custom_ar", "")
                       .typed<InitSignature>()));
}

TEST(MusaCustomAllReduceSignatureTest, AcceptsReductionReferenceContract) {
  register_musa_custom_all_reduce_kernel();
  using ReduceSignature =
      void(int64_t, torch::Tensor&, torch::Tensor&, int64_t, int64_t);
  EXPECT_NO_THROW((c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::all_reduce", "")
                       .typed<ReduceSignature>()));
}

}  // namespace
}  // namespace xllm
