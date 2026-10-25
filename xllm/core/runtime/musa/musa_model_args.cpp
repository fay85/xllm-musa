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

#include "runtime/musa/musa_model_args.h"

#include <string>
#include <vector>

#include "framework/model/model_args.h"

namespace xllm::musa {

void configure_deepseek_v4_mtp_model_args(ModelArgs& args) {
  const int32_t mtp_layers = args.num_nextn_predict_layers();
  args.n_layers(mtp_layers);
  args.layer_types(std::vector<std::string>(mtp_layers, "full_attention"));
  args.full_attention_interval(1);
  args.n_hash_layers(0);
}

}  // namespace xllm::musa
