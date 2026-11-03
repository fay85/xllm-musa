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

#include <cstdint>

#include "framework/block/block.h"

namespace xllm {

struct BlockTransferInfo;

namespace musa {

inline constexpr char kDeepseekV4PrefixSnapshotMetadataKey[] =
    "__musa_dsv4_prefix_state_snapshot";

enum class DeepseekV4PrefixTransferKind : int8_t {
  NONE = 0,
  BLOCK_COPY = 1,
  STATE_SNAPSHOT = 2,
};

struct DeepseekV4PrefixTransfer {
  DeepseekV4PrefixTransferKind kind = DeepseekV4PrefixTransferKind::NONE;
  BlockType block_type = BlockType::KV;
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  int32_t prefix_tokens = 0;
};

bool encode_deepseek_v4_prefix_block_copy(BlockType block_type,
                                          int32_t src_block_id,
                                          int32_t dst_block_id,
                                          BlockTransferInfo* transfer);

bool encode_deepseek_v4_prefix_state_snapshot(int32_t state_block_id,
                                              int32_t prefix_tokens,
                                              BlockTransferInfo* transfer);

DeepseekV4PrefixTransfer decode_deepseek_v4_prefix_transfer(
    const BlockTransferInfo& transfer);

}  // namespace musa
}  // namespace xllm
