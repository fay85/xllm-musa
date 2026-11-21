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

#include "framework/block/musa/deepseek_v4_prefix_transfer.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include "framework/model/model_input_params.h"

namespace xllm::musa {

namespace {

constexpr uint32_t kPrivateTransferMarker = 0x40000000u;
constexpr uint32_t kPrivateTransferTagShift = 27u;
constexpr uint32_t kPrivateTransferIdMask = 0x07ffffffu;
constexpr uint32_t kGdnHostDemoteTag = 4u;
constexpr uint32_t kGdnSinglepassCkptTag = 6u;
constexpr uint32_t kStateSnapshotTag = 7u;
constexpr uint32_t kGdnSinglepassSlotMask = 0xffu;
constexpr uint32_t kGdnSinglepassEndFlagBit = (1u << 8);
constexpr uint32_t kGdnSinglepassOffsetShift = 9u;
constexpr uint32_t kGdnSinglepassMaxOffsetBlocks = 0x3ffffu;

bool encode_source(uint32_t tag, int32_t block_id, int32_t* encoded_source) {
  if (encoded_source == nullptr || tag > kStateSnapshotTag || block_id < 0 ||
      static_cast<uint32_t>(block_id) > kPrivateTransferIdMask) {
    return false;
  }
  const uint32_t payload = kPrivateTransferMarker |
                           (tag << kPrivateTransferTagShift) |
                           static_cast<uint32_t>(block_id);
  *encoded_source = -static_cast<int32_t>(payload);
  return true;
}

bool decode_source(int32_t encoded_source, uint32_t* tag, int32_t* block_id) {
  if (tag == nullptr || block_id == nullptr || encoded_source >= 0) {
    return false;
  }
  const int64_t positive_payload = -static_cast<int64_t>(encoded_source);
  if (positive_payload > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  const uint32_t payload = static_cast<uint32_t>(positive_payload);
  if ((payload & kPrivateTransferMarker) == 0u) {
    return false;
  }
  *tag = (payload >> kPrivateTransferTagShift) & kStateSnapshotTag;
  *block_id = static_cast<int32_t>(payload & kPrivateTransferIdMask);
  return true;
}

bool is_prefix_block_type(BlockType block_type) {
  return block_type == BlockType::SWA || block_type == BlockType::C4 ||
         block_type == BlockType::C128 || block_type == BlockType::LINEAR;
}

}  // namespace

bool encode_deepseek_v4_prefix_block_copy(BlockType block_type,
                                          int32_t src_block_id,
                                          int32_t dst_block_id,
                                          BlockTransferInfo* transfer) {
  if (transfer == nullptr || !is_prefix_block_type(block_type) ||
      dst_block_id < 0) {
    return false;
  }
  int32_t encoded_source = -1;
  if (!encode_source(
          static_cast<uint32_t>(block_type), src_block_id, &encoded_source)) {
    return false;
  }
  *transfer = BlockTransferInfo(encoded_source, dst_block_id);
  return true;
}

bool encode_deepseek_v4_prefix_state_snapshot(int32_t state_block_id,
                                              int32_t prefix_tokens,
                                              BlockTransferInfo* transfer) {
  if (transfer == nullptr || prefix_tokens <= 0) {
    return false;
  }
  int32_t encoded_source = -1;
  if (!encode_source(kStateSnapshotTag, state_block_id, &encoded_source)) {
    return false;
  }
  *transfer = BlockTransferInfo(encoded_source, -prefix_tokens);
  return true;
}

bool encode_deepseek_v4_gdn_singlepass_checkpoint(int32_t live_slot_id,
                                                  int32_t ckpt_block_id,
                                                  int32_t offset_tokens,
                                                  bool is_end_of_step,
                                                  BlockTransferInfo* transfer) {
  if (transfer == nullptr || live_slot_id <= 0 ||
      static_cast<uint32_t>(live_slot_id) > kGdnSinglepassSlotMask ||
      ckpt_block_id < 0 || offset_tokens <= 0 || (offset_tokens % 64) != 0) {
    return false;
  }
  const uint32_t offset_blocks = static_cast<uint32_t>(offset_tokens / 64);
  if (offset_blocks == 0u || offset_blocks > kGdnSinglepassMaxOffsetBlocks) {
    return false;
  }
  const uint32_t packed =
      (offset_blocks << kGdnSinglepassOffsetShift) |
      (is_end_of_step ? kGdnSinglepassEndFlagBit : 0u) |
      (static_cast<uint32_t>(live_slot_id) & kGdnSinglepassSlotMask);
  int32_t encoded_source = -1;
  if (!encode_source(
          kGdnSinglepassCkptTag, static_cast<int32_t>(packed), &encoded_source)) {
    return false;
  }
  *transfer = BlockTransferInfo(encoded_source, ckpt_block_id);
  return true;
}

bool encode_deepseek_v4_gdn_host_demote(int32_t src_hbm_ckpt_id,
                                        int32_t dst_host_ckpt_id,
                                        BlockTransferInfo* transfer) {
  if (transfer == nullptr || src_hbm_ckpt_id < 0 || dst_host_ckpt_id < 0) {
    return false;
  }
  int32_t encoded_source = -1;
  if (!encode_source(kGdnHostDemoteTag, src_hbm_ckpt_id, &encoded_source)) {
    return false;
  }
  *transfer = BlockTransferInfo(encoded_source, dst_host_ckpt_id);
  return true;
}

uint32_t resolve_gdn_host_ckpt_slots(uint32_t num_hbm_ckpt_slots) {
  if (num_hbm_ckpt_slots == 0u) {
    return 0u;
  }
  const char* env = std::getenv("XLLM_GDN_HOST_CKPT_SLOTS");
  if (env != nullptr && env[0] != '\0') {
    char* endptr = nullptr;
    const unsigned long long parsed = std::strtoull(env, &endptr, 10);
    if (endptr != env) {
      return static_cast<uint32_t>(parsed);
    }
  }
  // Auto-size from /proc/meminfo so 650 GiB Cloud VMs stay safely within RAM
  // (~95 GiB max across 6 GPUs at 384 slots, allocated lazily in 64-slot slabs)
  // while >=900 GiB bare-metal hosts scale to 1024 Host L2 slots per Prefill.
  uint64_t mem_total_kib = 0;
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  uint64_t val = 0;
  std::string unit;
  while (meminfo >> key >> val) {
    std::getline(meminfo, unit);
    if (key == "MemTotal:") {
      mem_total_kib = val;
      break;
    }
  }
  const uint64_t mem_total_gib = mem_total_kib / (1024ull * 1024ull);
  return mem_total_gib >= 900ull ? 1024u : 384u;
}

DeepseekV4PrefixTransfer decode_deepseek_v4_prefix_transfer(
    const BlockTransferInfo& transfer) {
  DeepseekV4PrefixTransfer decoded;
  uint32_t tag = 0;
  int32_t block_id = -1;
  if (!decode_source(transfer.src_block_id, &tag, &block_id)) {
    return decoded;
  }
  if (tag == kGdnHostDemoteTag && transfer.dst_block_id >= 0) {
    decoded.kind = DeepseekV4PrefixTransferKind::GDN_HOST_DEMOTE;
    decoded.block_type = BlockType::LINEAR;
    decoded.src_block_id = block_id;
    decoded.dst_block_id = transfer.dst_block_id;
    return decoded;
  }
  if (tag == kGdnSinglepassCkptTag && transfer.dst_block_id >= 0) {
    const uint32_t packed = static_cast<uint32_t>(block_id);
    const uint32_t offset_blocks = packed >> kGdnSinglepassOffsetShift;
    const int32_t live_slot_id =
        static_cast<int32_t>(packed & kGdnSinglepassSlotMask);
    if (offset_blocks > 0u && live_slot_id > 0) {
      decoded.kind = DeepseekV4PrefixTransferKind::GDN_SINGLEPASS_CHECKPOINT;
      decoded.block_type = BlockType::LINEAR;
      decoded.src_block_id = live_slot_id;
      decoded.dst_block_id = transfer.dst_block_id;
      decoded.gdn_ckpt_offset_tokens = static_cast<int32_t>(offset_blocks * 64u);
      decoded.gdn_ckpt_is_end_of_step =
          (packed & kGdnSinglepassEndFlagBit) != 0u;
      return decoded;
    }
    return decoded;
  }
  if (tag == kStateSnapshotTag && transfer.dst_block_id < 0 &&
      transfer.dst_block_id != std::numeric_limits<int32_t>::min()) {
    decoded.kind = DeepseekV4PrefixTransferKind::STATE_SNAPSHOT;
    decoded.block_type = BlockType::SWA;
    decoded.src_block_id = block_id;
    decoded.prefix_tokens = -transfer.dst_block_id;
    return decoded;
  }
  const BlockType block_type = static_cast<BlockType>(tag);
  if (!is_prefix_block_type(block_type) || transfer.dst_block_id < 0) {
    return decoded;
  }
  decoded.kind = DeepseekV4PrefixTransferKind::BLOCK_COPY;
  decoded.block_type = block_type;
  decoded.src_block_id = block_id;
  decoded.dst_block_id = transfer.dst_block_id;
  return decoded;
}

}  // namespace xllm::musa
