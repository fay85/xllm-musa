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

#include "runtime/musa/deepseek_v4_prefix_block_copy.h"

#include <cstdint>
#include <vector>

#include "framework/block/musa/deepseek_v4_prefix_transfer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm::musa {

namespace {

void copy_tensor_block(const torch::Tensor& tensor,
                       int32_t src_block_id,
                       int32_t dst_block_id,
                       const torch::Device& device) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  CHECK_GE(src_block_id, 0);
  CHECK_GE(dst_block_id, 0);
  CHECK_LT(src_block_id, tensor.size(0));
  CHECK_LT(dst_block_id, tensor.size(0));
  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kLong).device(device);
  const torch::Tensor src = torch::tensor({src_block_id}, index_options);
  const torch::Tensor dst = torch::tensor({dst_block_id}, index_options);
  const torch::Tensor selected = torch::index_select(tensor, /*dim=*/0, src);
  tensor.index_copy_(/*dim=*/0, dst, selected);
}

void copy_recurrent_state(KVCache* kv_cache,
                          int32_t src_block_id,
                          int32_t dst_block_id,
                          const torch::Device& device) {
  CHECK(kv_cache != nullptr);
  const BlockTypeTensorMap swa_tensors =
      kv_cache->get_block_type_tensors(BlockType::SWA);
  for (const auto& [role, tensor] : swa_tensors) {
    (void)role;
    copy_tensor_block(tensor, src_block_id, dst_block_id, device);
  }
  copy_tensor_block(
      kv_cache->get_compress_kv_state(), src_block_id, dst_block_id, device);
  copy_tensor_block(
      kv_cache->get_compress_score_state(), src_block_id, dst_block_id, device);
  copy_tensor_block(kv_cache->get_compress_index_kv_state(),
                    src_block_id,
                    dst_block_id,
                    device);
  copy_tensor_block(kv_cache->get_compress_index_score_state(),
                    src_block_id,
                    dst_block_id,
                    device);
}

void copy_compressed_block(KVCache* kv_cache,
                           BlockType block_type,
                           int32_t src_block_id,
                           int32_t dst_block_id,
                           const torch::Device& device) {
  CHECK(kv_cache != nullptr);
  const BlockTypeTensorMap tensors =
      kv_cache->get_block_type_tensors(block_type);
  for (const auto& [role, tensor] : tensors) {
    (void)role;
    copy_tensor_block(tensor, src_block_id, dst_block_id, device);
  }
}

}  // namespace

bool apply_deepseek_v4_prefix_block_copies(
    const std::vector<BlockTransferInfo>& transfers,
    std::vector<KVCache>* kv_caches,
    const torch::Device& device) {
  if (kv_caches == nullptr || transfers.empty()) {
    return false;
  }
  bool has_private_transfer = false;
  for (const BlockTransferInfo& transfer : transfers) {
    if (decode_deepseek_v4_prefix_transfer(transfer).kind !=
        DeepseekV4PrefixTransferKind::NONE) {
      has_private_transfer = true;
      break;
    }
  }
  if (!has_private_transfer) {
    return false;
  }

  for (const BlockTransferInfo& transfer : transfers) {
    const DeepseekV4PrefixTransfer decoded =
        decode_deepseek_v4_prefix_transfer(transfer);
    if (decoded.kind == DeepseekV4PrefixTransferKind::STATE_SNAPSHOT) {
      continue;
    }
    if (decoded.kind == DeepseekV4PrefixTransferKind::BLOCK_COPY) {
      for (KVCache& kv_cache : *kv_caches) {
        if (decoded.block_type == BlockType::SWA) {
          copy_recurrent_state(
              &kv_cache, decoded.src_block_id, decoded.dst_block_id, device);
        } else {
          copy_compressed_block(&kv_cache,
                                decoded.block_type,
                                decoded.src_block_id,
                                decoded.dst_block_id,
                                device);
        }
      }
      continue;
    }

    const torch::TensorOptions index_options =
        torch::TensorOptions().dtype(torch::kLong).device(device);
    torch::Tensor src = torch::tensor({transfer.src_block_id}, index_options);
    torch::Tensor dst = torch::tensor({transfer.dst_block_id}, index_options);
    for (KVCache& kv_cache : *kv_caches) {
      kv_cache.swap_blocks(src, dst);
    }
  }
  return true;
}

}  // namespace xllm::musa
