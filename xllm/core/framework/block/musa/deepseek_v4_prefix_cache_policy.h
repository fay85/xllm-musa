/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "framework/prefix_cache/block_hasher.h"

namespace xllm {

class BlockManagerPool;
class CompositeBlockManager;
class Sequence;
struct BlockTransferInfo;

namespace musa {

void initialize_deepseek_v4_prefix_cache(const BlockManagerPool* owner,
                                         int32_t dp_size);

void destroy_deepseek_v4_prefix_cache(const BlockManagerPool* owner);

bool try_restore_deepseek_v4_prefix(const BlockManagerPool* owner,
                                    Sequence* sequence,
                                    int32_t dp_rank,
                                    CompositeBlockManager* composite,
                                    uint32_t base_block_size,
                                    BlockHasherType hasher_type,
                                    std::vector<BlockTransferInfo>* transfers);

void prepare_deepseek_v4_prefix_snapshot(
    const BlockManagerPool* owner,
    Sequence* sequence,
    int32_t dp_rank,
    CompositeBlockManager* composite,
    uint32_t base_block_size,
    BlockHasherType hasher_type,
    size_t max_computed_tokens,
    std::vector<BlockTransferInfo>* transfers);

void activate_deepseek_v4_prefix_snapshot(const BlockManagerPool* owner,
                                          Sequence* sequence,
                                          int32_t dp_rank);

void discard_deepseek_v4_prefix_sequence(const BlockManagerPool* owner,
                                         Sequence* sequence,
                                         int32_t dp_rank);

void append_deepseek_v4_prefix_state_restore(
    const BlockManagerPool* owner,
    Sequence* sequence,
    std::vector<BlockTransferInfo>* transfers);

void reset_deepseek_v4_prefix_cache(const BlockManagerPool* owner);

}  // namespace musa
}  // namespace xllm
