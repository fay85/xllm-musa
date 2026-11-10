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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/kernels/npu/paged_attention_graph_operation.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xllm::npu {

constexpr int64_t kCausalConv1dGraphPadSlotId = -1;
constexpr int64_t kCausalConv1dActivationSilu = 1;
constexpr int64_t kCausalConv1dRunModeForward = 0;
constexpr int64_t kCausalConv1dRunModeUpdate = 1;

enum class CausalConv1dGraphBranch {
  kDecode,
  kSpecVerify,
};

enum class FusedInferAttentionGraphBranch {
  kDecode,
  kSpecVerify,
};

struct CausalConv1dGraphTask {
  torch::Tensor output;
  torch::Tensor x;
  torch::Tensor weight;
  torch::Tensor conv_state;
  std::optional<torch::Tensor> bias;
  int64_t activation_mode = kCausalConv1dActivationSilu;
  int64_t pad_slot_id = kCausalConv1dGraphPadSlotId;
  int64_t run_mode = kCausalConv1dRunModeUpdate;
  CausalConv1dGraphBranch branch = CausalConv1dGraphBranch::kDecode;
  uint64_t capture_order = 0;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

class FusedInferAttentionWorkspaceSignature {
 public:
  torch::ScalarType query_dtype;
  torch::ScalarType key_dtype;
  torch::ScalarType value_dtype;
  torch::ScalarType block_table_dtype;
  c10::DeviceIndex device_index;
  std::vector<int64_t> query_shape;
  std::vector<int64_t> key_shape;
  std::vector<int64_t> value_shape;
  std::vector<int64_t> block_table_shape;
  std::vector<int64_t> actual_seq_lengths;
  std::vector<int64_t> actual_seq_lengths_kv;
  int64_t num_heads;
  int64_t num_key_value_heads;
  int64_t block_size;
  double scale;
  bool softmax_lse_flag = false;

  bool operator==(const FusedInferAttentionWorkspaceSignature&) const = default;
};

struct FusedInferAttentionGraphTask {
  torch::Tensor output;
  torch::Tensor softmax_lse;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor block_table;
  torch::Tensor workspace;
  std::vector<int64_t> actual_seq_lengths;
  int64_t num_heads = 0;
  int64_t num_key_value_heads = 0;
  double scale = 0.0;
  int64_t block_size = 0;
  int32_t dcp_size = 1;
  int32_t dcp_rank = 0;
  FusedInferAttentionGraphBranch branch =
      FusedInferAttentionGraphBranch::kDecode;
  uint64_t capture_order = 0;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

enum class PagedAttentionGraphBranch {
  DECODE,
  SPEC_VERIFY,
};

struct PagedAttentionGraphTask {
  std::shared_ptr<kernel::npu::PagedAttentionGraphOperation> operation;
  int64_t batch_size = 0;
  PagedAttentionGraphBranch branch = PagedAttentionGraphBranch::DECODE;
  uint64_t capture_order = 0;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

enum class AclGraphTaskType {
  CAUSAL_CONV1D,
  FUSED_INFER_ATTENTION,
  PAGED_ATTENTION,
};

struct AclGraphTaskUpdateEntry {
  AclGraphTaskType type;
  size_t index;
  uint64_t capture_order;
};

class AclGraphTaskUpdateContext final {
 public:
  void begin_capture() {
    capturing = true;
    next_capture_order = 0;
    ordered_tasks.clear();
    causal_conv1d_tasks.clear();
    fused_infer_attention_tasks.clear();
    paged_attention_tasks.clear();
    paged_attention_resources.reset();
    fused_infer_attention_workspace_signature.reset();
    fused_infer_attention_workspace = torch::Tensor();
  }

  void end_capture() {
    // Build the update schedule once. Replay must preserve capture order
    // across operator types because their ready events gate dependent tasks.
    ordered_tasks.clear();
    ordered_tasks.reserve(causal_conv1d_tasks.size() +
                          fused_infer_attention_tasks.size() +
                          paged_attention_tasks.size());
    auto append_tasks = [this](const auto& tasks, AclGraphTaskType type) {
      for (size_t index = 0; index < tasks.size(); ++index) {
        ordered_tasks.emplace_back(
            AclGraphTaskUpdateEntry{type, index, tasks[index].capture_order});
      }
    };
    append_tasks(causal_conv1d_tasks, AclGraphTaskType::CAUSAL_CONV1D);
    append_tasks(fused_infer_attention_tasks,
                 AclGraphTaskType::FUSED_INFER_ATTENTION);
    append_tasks(paged_attention_tasks, AclGraphTaskType::PAGED_ATTENTION);
    std::sort(ordered_tasks.begin(),
              ordered_tasks.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.capture_order < rhs.capture_order;
              });
    for (size_t index = 1; index < ordered_tasks.size(); ++index) {
      CHECK_LT(ordered_tasks[index - 1].capture_order,
               ordered_tasks[index].capture_order)
          << "ACL graph task capture order must be unique";
    }
    capturing = false;
  }

  bool capturing = false;
  uint64_t next_capture_order = 0;
  std::vector<CausalConv1dGraphTask> causal_conv1d_tasks;
  std::vector<FusedInferAttentionGraphTask> fused_infer_attention_tasks;
  std::vector<PagedAttentionGraphTask> paged_attention_tasks;
  std::vector<AclGraphTaskUpdateEntry> ordered_tasks;
  std::shared_ptr<kernel::npu::PagedAttentionGraphResources>
      paged_attention_resources;
  std::optional<FusedInferAttentionWorkspaceSignature>
      fused_infer_attention_workspace_signature;
  torch::Tensor fused_infer_attention_workspace;
};

}  // namespace xllm::npu
