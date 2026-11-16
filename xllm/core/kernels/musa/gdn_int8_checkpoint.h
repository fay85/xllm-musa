/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <vector>

namespace xllm {
class KVCache;
}

namespace xllm::kernel::musa {

void init_gdn_int8_checkpoint_pool(const std::vector<::xllm::KVCache>& kv_caches,
                                   int64_t num_ckpt_slots);

int64_t get_gdn_int8_checkpoint_slots(
    const std::vector<::xllm::KVCache>& kv_caches);

void store_gdn_int8_checkpoint(const std::vector<::xllm::KVCache>& kv_caches,
                               int32_t active_slot_id,
                               int32_t ckpt_logical_id);

void restore_gdn_int8_checkpoint(std::vector<::xllm::KVCache>& kv_caches,
                                 int32_t active_write_id,
                                 int32_t ckpt_logical_id);

}  // namespace xllm::kernel::musa
