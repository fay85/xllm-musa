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

#include <torch/torch.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace xllm {
class KVCache;
}

namespace xllm::kernel::musa {

struct GdnSinglepassInteriorTarget {
  int32_t live_slot_id = -1;
  int32_t ckpt_logical_id = -1;
  int32_t offset_tokens = 0;
};

class ScopedGdnSinglepassCheckpointPlan final {
 public:
  explicit ScopedGdnSinglepassCheckpointPlan(
      std::vector<GdnSinglepassInteriorTarget> targets);
  ~ScopedGdnSinglepassCheckpointPlan();

  ScopedGdnSinglepassCheckpointPlan(const ScopedGdnSinglepassCheckpointPlan&) =
      delete;
  ScopedGdnSinglepassCheckpointPlan& operator=(
      const ScopedGdnSinglepassCheckpointPlan&) = delete;

  const std::vector<GdnSinglepassInteriorTarget>& targets() const {
    return targets_;
  }
  void mark_written(int32_t ckpt_logical_id) {
    written_ckpt_ids_.insert(ckpt_logical_id);
  }
  bool was_written_locally(int32_t ckpt_logical_id) const {
    return written_ckpt_ids_.find(ckpt_logical_id) != written_ckpt_ids_.end();
  }

 private:
  std::vector<GdnSinglepassInteriorTarget> targets_;
  std::unordered_set<int32_t> written_ckpt_ids_;
  ScopedGdnSinglepassCheckpointPlan* prev_ = nullptr;
};

bool has_active_gdn_singlepass_checkpoints();

std::vector<GdnSinglepassInteriorTarget>
get_active_gdn_singlepass_checkpoints_for_slot(int32_t live_slot_id);

void init_gdn_int8_checkpoint_pool(const std::vector<::xllm::KVCache>& kv_caches,
                                   int64_t num_ckpt_slots);

int64_t get_gdn_int8_checkpoint_slots(
    const std::vector<::xllm::KVCache>& kv_caches);

void store_gdn_int8_checkpoint(const std::vector<::xllm::KVCache>& kv_caches,
                               int32_t active_slot_id,
                               int32_t ckpt_logical_id);

void store_gdn_int8_checkpoint_from_tensors(
    const torch::Tensor& layer_ssm_cache,
    const torch::Tensor& layer_ssm_state,
    const torch::Tensor& layer_conv_state,
    int32_t ckpt_logical_id);

torch::Tensor allocate_gdn_int8_checkpoint_packet(
    const std::vector<::xllm::KVCache>& kv_caches);

torch::Tensor pack_gdn_int8_checkpoint_slot(
    const std::vector<::xllm::KVCache>& kv_caches,
    int32_t ckpt_logical_id);

void unpack_gdn_int8_checkpoint_slot(
    const std::vector<::xllm::KVCache>& kv_caches,
    int32_t ckpt_logical_id,
    const torch::Tensor& packet);

void demote_gdn_int8_checkpoint_to_host(
    const std::vector<::xllm::KVCache>& kv_caches,
    int32_t src_hbm_logical_id,
    int32_t dst_host_logical_id);

void restore_gdn_int8_checkpoint(std::vector<::xllm::KVCache>& kv_caches,
                                 int32_t active_write_id,
                                 int32_t ckpt_logical_id);

}  // namespace xllm::kernel::musa
