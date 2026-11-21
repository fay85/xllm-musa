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
  GDN_SINGLEPASS_CHECKPOINT = 3,
  GDN_HOST_DEMOTE = 4,
};

struct DeepseekV4PrefixTransfer {
  DeepseekV4PrefixTransferKind kind = DeepseekV4PrefixTransferKind::NONE;
  BlockType block_type = BlockType::KV;
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  int32_t prefix_tokens = 0;
  int32_t gdn_ckpt_offset_tokens = 0;
  bool gdn_ckpt_is_end_of_step = false;
};

bool encode_deepseek_v4_prefix_block_copy(BlockType block_type,
                                          int32_t src_block_id,
                                          int32_t dst_block_id,
                                          BlockTransferInfo* transfer);

bool encode_deepseek_v4_prefix_state_snapshot(int32_t state_block_id,
                                              int32_t prefix_tokens,
                                              BlockTransferInfo* transfer);

bool encode_deepseek_v4_gdn_singlepass_checkpoint(int32_t live_slot_id,
                                                  int32_t ckpt_block_id,
                                                  int32_t offset_tokens,
                                                  bool is_end_of_step,
                                                  BlockTransferInfo* transfer);

bool encode_deepseek_v4_gdn_host_demote(int32_t src_hbm_ckpt_id,
                                        int32_t dst_host_ckpt_id,
                                        BlockTransferInfo* transfer);

uint32_t resolve_gdn_host_ckpt_slots(uint32_t num_hbm_ckpt_slots);

void flush_pending_gdn_singlepass_checkpoints();

DeepseekV4PrefixTransfer decode_deepseek_v4_prefix_transfer(
    const BlockTransferInfo& transfer);

}  // namespace musa
}  // namespace xllm
