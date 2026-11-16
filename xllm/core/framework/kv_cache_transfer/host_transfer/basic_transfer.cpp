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

#include "framework/kv_cache_transfer/host_transfer/basic_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/transfer_utils.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "util/blockingconcurrentqueue.h"

namespace xllm {
namespace {

constexpr int32_t kStreamTimeoutMs = 60000;
constexpr size_t kCopyStreamCount = 6;

struct CopyPlan {
  std::vector<torch::Tensor> src_tensors;
  std::vector<torch::Tensor> dst_tensors;
};

struct RawRoleCopyDesc {
  uint8_t* host_layer_base = nullptr;
  int64_t host_block_stride_bytes = 0;
  uint8_t* device_base = nullptr;
  int64_t device_block_stride_bytes = 0;
  size_t copy_bytes = 0;
};

struct RawLayerCopyDesc {
  int64_t absolute_layer_id = 0;
  BlockType block_type = BlockType::KV;
  std::vector<RawRoleCopyDesc> roles;
};

struct RawCopyPlan {
  std::vector<void*> src_addrs;
  std::vector<void*> dst_addrs;
  std::vector<size_t> byte_counts;
};

using CopyStreamQueue =
    moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>>;

class CopyStreamLease final {
 public:
  explicit CopyStreamLease(CopyStreamQueue* streams) : streams_(streams) {
    streams_->wait_dequeue(stream_);
  }

  ~CopyStreamLease() { streams_->enqueue(std::move(stream_)); }

  CopyStreamLease(const CopyStreamLease&) = delete;
  CopyStreamLease& operator=(const CopyStreamLease&) = delete;

  Stream* get() const { return stream_.get(); }

  void drain_or_die(const char* reason) const {
    try {
      const int32_t result = stream_->synchronize();
      if (result != 0) {
        LOG(FATAL) << "Failed to drain Host KV copy stream: reason=" << reason
                   << ", result=" << result;
      }
    } catch (const std::exception& error) {
      LOG(FATAL) << "Failed to drain Host KV copy stream: reason=" << reason
                 << ", error=" << error.what();
    }
  }

 private:
  CopyStreamQueue* streams_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

CopyPlan build_plan(const HostKVLayout& layout,
                    const std::vector<HostKVMapping>& mappings,
                    const LayerRange& range,
                    bool is_load) {
  CopyPlan plan;
  const size_t estimated_copies =
      mappings.size() * static_cast<size_t>(range.end - range.begin);
  plan.src_tensors.reserve(estimated_copies);
  plan.dst_tensors.reserve(estimated_copies);
  std::unordered_map<const HostKVLayerLayout*,
                     std::unordered_map<size_t, BlockTypeTensorMap>>
      device_tensor_cache;
  for (const HostKVMapping& mapping : mappings) {
    const HostKVGroupLayout& group = layout.group(mapping.group_id);
    for (const HostKVLayerLayout& layer : group.layers) {
      if (layer.absolute_layer_id < range.begin ||
          layer.absolute_layer_id >= range.end) {
        continue;
      }
      auto& cached_offsets = device_tensor_cache[&layer];
      auto tensor_it = cached_offsets.find(mapping.checkpoint_row);
      if (tensor_it == cached_offsets.end()) {
        tensor_it = cached_offsets
                        .emplace(mapping.checkpoint_row,
                                 get_device_transfer_tensors(
                                     layer, mapping.checkpoint_row))
                        .first;
      }
      const BlockTypeTensorMap& device_tensors = tensor_it->second;
      for (const auto& [role, device_tensor] : device_tensors) {
        auto host_it = group.host_roles.find(role);
        if (host_it == group.host_roles.end()) {
          continue;
        }
        torch::Tensor host_block =
            host_it->second[mapping.host_block_id][layer.group_layer_slot];
        torch::Tensor device_block = device_tensor[mapping.device_block_id];
        if (is_load) {
          plan.src_tensors.emplace_back(host_block);
          plan.dst_tensors.emplace_back(device_block);
        } else {
          plan.src_tensors.emplace_back(device_block);
          plan.dst_tensors.emplace_back(host_block);
        }
      }
    }
  }
  return plan;
}

}  // namespace

class BasicHostKVTransfer::Impl final {
 public:
  Impl(const HostKVLayout& layout,
       const Device& device,
       const Stream& compute_stream,
       uint32_t layer_copy_batches,
       std::unique_ptr<BatchMemcpy> batch_memcpy)
      : layout_(layout),
        device_(device),
        compute_stream_(compute_stream),
        layers_per_event_(
            get_layers_per_event(layout.num_layers(), layer_copy_batches)),
        ranges_(build_layer_ranges(layout.num_layers(), layers_per_event_)),
        batch_memcpy_(std::move(batch_memcpy)) {
    if (batch_memcpy_ == nullptr) {
      batch_memcpy_ = create_batch_memcpy(device_);
    } else {
      batch_memcpy_->init(device_.index());
    }
    CHECK(batch_memcpy_ != nullptr) << "Host KV batch memcpy is unavailable.";
    init_raw_copy_descs();
    for (size_t index = 0; index < kCopyStreamCount; ++index) {
      streams_.enqueue(device_.get_stream_from_pool(kStreamTimeoutMs));
    }
  }

  HostKVLoadHandle prepare_load(bool draft) const {
    const uint32_t count = event_count() + static_cast<uint32_t>(draft);
    HostKVLoadHandle handle{create_layer_synchronizer(count),
                            layers_per_event()};
    if (draft) {
      handle.draft_event_index = event_count();
    }
    return handle;
  }

  uint32_t event_count() const { return static_cast<uint32_t>(ranges_.size()); }

  uint32_t layers_per_event() const { return layers_per_event_; }

  bool load(const HostKVRequest& request, const HostKVLoadHandle& handle) {
    CopyStreamLease stream(&streams_);
    bool stream_has_work = false;
    const bool raw_ok = can_use_raw_copy(request);
    for (size_t index = 0; index < ranges_.size(); ++index) {
      if (raw_ok) {
        RawCopyPlan plan = build_raw_plan(
            request.target_mappings,
            ranges_[index],
            /*is_load=*/true,
            /*linear_mode=*/index == 0 ? LinearFilter::ALL_LAYERS
                                       : LinearFilter::SKIP);
        if (!plan.byte_counts.empty()) {
          if (!batch_memcpy_->submit_h2d_raw(
                  plan.src_addrs, plan.dst_addrs, plan.byte_counts, stream.get())) {
            return false;
          }
          stream_has_work = true;
        }
      } else {
        CopyPlan plan = build_plan(
            layout_, request.target_mappings, ranges_[index], /*is_load=*/true);
        if (!plan.src_tensors.empty()) {
          if (!batch_memcpy_->submit_h2d(
                  plan.src_tensors, plan.dst_tensors, stream.get())) {
            return false;
          }
          stream_has_work = true;
        }
      }
      if (!handle.synchronizer->record_stream(static_cast<int64_t>(index),
                                              stream.get())) {
        if (stream_has_work) {
          stream.drain_or_die("layer-ready event recording failed");
        }
        return false;
      }
    }
    if (raw_ok && !request.draft_mappings.empty()) {
      const LayerRange full_range{0, layout_.num_layers()};
      RawCopyPlan plan = build_raw_plan(
          request.draft_mappings, full_range, /*is_load=*/true, LinearFilter::RANGE);
      if (!plan.byte_counts.empty()) {
        if (!batch_memcpy_->submit_h2d_raw(
                plan.src_addrs, plan.dst_addrs, plan.byte_counts, stream.get())) {
          if (stream_has_work) {
            stream.drain_or_die("draft cache copy submission failed");
          }
          return false;
        }
        stream_has_work = true;
      }
    } else {
      for (const LayerRange& range : ranges_) {
        CopyPlan plan =
            build_plan(layout_, request.draft_mappings, range, /*is_load=*/true);
        if (!plan.src_tensors.empty() &&
            !batch_memcpy_->submit_h2d(
                plan.src_tensors, plan.dst_tensors, stream.get())) {
          if (stream_has_work) {
            stream.drain_or_die("draft cache copy submission failed");
          }
          return false;
        }
        stream_has_work = stream_has_work || !plan.src_tensors.empty();
      }
    }
    if (!request.draft_mappings.empty() &&
        !handle.synchronizer->record_stream(
            static_cast<int64_t>(ranges_.size()), stream.get())) {
      if (stream_has_work) {
        stream.drain_or_die("draft cache completion event recording failed");
      }
      return false;
    }
    return true;
  }

  bool offload(const HostKVRequest& request) {
    CopyStreamLease stream(&streams_);
    stream.get()->wait_stream(compute_stream_);
    if (can_use_raw_copy(request)) {
      const LayerRange full_range{0, layout_.num_layers()};
      RawCopyPlan plan;
      if (!request.target_mappings.empty()) {
        RawCopyPlan target_plan = build_raw_plan(
            request.target_mappings, full_range, /*is_load=*/false, LinearFilter::RANGE);
        plan.src_addrs.insert(
            plan.src_addrs.end(), target_plan.src_addrs.begin(), target_plan.src_addrs.end());
        plan.dst_addrs.insert(
            plan.dst_addrs.end(), target_plan.dst_addrs.begin(), target_plan.dst_addrs.end());
        plan.byte_counts.insert(
            plan.byte_counts.end(), target_plan.byte_counts.begin(), target_plan.byte_counts.end());
      }
      if (!request.draft_mappings.empty()) {
        RawCopyPlan draft_plan = build_raw_plan(
            request.draft_mappings, full_range, /*is_load=*/false, LinearFilter::RANGE);
        plan.src_addrs.insert(
            plan.src_addrs.end(), draft_plan.src_addrs.begin(), draft_plan.src_addrs.end());
        plan.dst_addrs.insert(
            plan.dst_addrs.end(), draft_plan.dst_addrs.begin(), draft_plan.dst_addrs.end());
        plan.byte_counts.insert(
            plan.byte_counts.end(), draft_plan.byte_counts.begin(), draft_plan.byte_counts.end());
      }
      if (plan.byte_counts.empty()) {
        return true;
      }
      return batch_memcpy_->copy_d2h_raw(
          plan.src_addrs, plan.dst_addrs, plan.byte_counts, stream.get());
    }
    const auto offload_mappings =
        [this, &stream](const std::vector<HostKVMapping>& mappings) {
          for (const LayerRange& range : ranges_) {
            CopyPlan plan =
                build_plan(layout_, mappings, range, /*is_load=*/false);
            if (!plan.src_tensors.empty() &&
                !batch_memcpy_->copy_d2h(
                    plan.src_tensors, plan.dst_tensors, stream.get())) {
              return false;
            }
          }
          return true;
        };
    return offload_mappings(request.target_mappings) &&
           offload_mappings(request.draft_mappings);
  }

  void drain() {
    if (drained_) {
      return;
    }
    drained_ = true;
    std::unique_ptr<Stream> stream;
    while (streams_.try_dequeue(stream)) {
      if (stream != nullptr && stream->synchronize() != 0) {
        LOG(FATAL) << "Failed to drain Basic Host KV copy stream.";
      }
    }
  }

 private:
  enum class LinearFilter : int8_t { RANGE = 0, ALL_LAYERS = 1, SKIP = 2 };

  void init_raw_copy_descs() {
    if (batch_memcpy_ == nullptr || !batch_memcpy_->supports_raw_copy()) {
      return;
    }
    for (const HostKVGroupLayout& group : layout_.groups()) {
      std::vector<RawLayerCopyDesc> layer_descs;
      layer_descs.reserve(group.layers.size());
      for (const HostKVLayerLayout& layer : group.layers) {
        const BlockTypeTensorMap device_tensors =
            get_device_transfer_tensors(layer, /*checkpoint_row=*/0);
        RawLayerCopyDesc layer_desc;
        layer_desc.absolute_layer_id = layer.absolute_layer_id;
        layer_desc.block_type = layer.block_type;
        for (const auto& [role, device_tensor] : device_tensors) {
          auto host_it = group.host_roles.find(role);
          if (host_it == group.host_roles.end()) {
            continue;
          }
          const torch::Tensor& host_tensor = host_it->second;
          if (!device_tensor.defined() || !host_tensor.defined() ||
              device_tensor.dim() < 2 || host_tensor.dim() < 3) {
            return;
          }
          const torch::Tensor host_sample =
              host_tensor[0][layer.group_layer_slot];
          const torch::Tensor dev_sample = device_tensor[0];
          if (!host_sample.is_contiguous() || !dev_sample.is_contiguous()) {
            return;
          }
          const size_t bytes = static_cast<size_t>(dev_sample.nbytes());
          if (static_cast<size_t>(host_sample.nbytes()) != bytes) {
            return;
          }
          RawRoleCopyDesc role_desc;
          role_desc.host_layer_base =
              static_cast<uint8_t*>(host_tensor.data_ptr()) +
              layer.group_layer_slot * host_tensor.stride(1) *
                  host_tensor.element_size();
          role_desc.host_block_stride_bytes =
              host_tensor.stride(0) * host_tensor.element_size();
          role_desc.device_base =
              static_cast<uint8_t*>(device_tensor.data_ptr());
          role_desc.device_block_stride_bytes =
              device_tensor.stride(0) * device_tensor.element_size();
          role_desc.copy_bytes = bytes;
          layer_desc.roles.emplace_back(role_desc);
        }
        layer_descs.emplace_back(std::move(layer_desc));
      }
      raw_descs_by_group_.emplace(group.group_id, std::move(layer_descs));
    }
    use_raw_copy_ = !raw_descs_by_group_.empty();
  }

  bool can_use_raw_copy(const HostKVRequest& request) const {
    if (!use_raw_copy_) {
      return false;
    }
    for (const HostKVMapping& m : request.target_mappings) {
      if (m.checkpoint_row != 0) return false;
    }
    for (const HostKVMapping& m : request.draft_mappings) {
      if (m.checkpoint_row != 0) return false;
    }
    return true;
  }

  RawCopyPlan build_raw_plan(const std::vector<HostKVMapping>& mappings,
                             const LayerRange& range,
                             bool is_load,
                             LinearFilter linear_mode) const {
    RawCopyPlan plan;
    if (mappings.empty()) {
      return plan;
    }
    const size_t est =
        mappings.size() * static_cast<size_t>(std::max<int64_t>(1, range.end - range.begin)) * 2;
    plan.src_addrs.reserve(est);
    plan.dst_addrs.reserve(est);
    plan.byte_counts.reserve(est);

    auto append_layer = [&](const HostKVMapping& mapping,
                            const RawLayerCopyDesc& layer) {
      for (const RawRoleCopyDesc& role : layer.roles) {
        if (role.copy_bytes == 0) continue;
        uint8_t* host_ptr =
            role.host_layer_base +
            mapping.host_block_id * role.host_block_stride_bytes;
        uint8_t* dev_ptr =
            role.device_base +
            mapping.device_block_id * role.device_block_stride_bytes;
        if (is_load) {
          plan.src_addrs.emplace_back(host_ptr);
          plan.dst_addrs.emplace_back(dev_ptr);
        } else {
          plan.src_addrs.emplace_back(dev_ptr);
          plan.dst_addrs.emplace_back(host_ptr);
        }
        plan.byte_counts.emplace_back(role.copy_bytes);
      }
    };

    if (linear_mode == LinearFilter::ALL_LAYERS) {
      for (const HostKVMapping& mapping : mappings) {
        auto it = raw_descs_by_group_.find(mapping.group_id);
        if (it == raw_descs_by_group_.end()) continue;
        for (const RawLayerCopyDesc& layer : it->second) {
          if (layer.block_type != BlockType::LINEAR) break;
          append_layer(mapping, layer);
        }
      }
    }

    for (const HostKVMapping& mapping : mappings) {
      auto it = raw_descs_by_group_.find(mapping.group_id);
      if (it == raw_descs_by_group_.end()) continue;
      for (const RawLayerCopyDesc& layer : it->second) {
        if (layer.block_type == BlockType::LINEAR &&
            linear_mode != LinearFilter::RANGE) {
          continue;
        }
        if (layer.absolute_layer_id < range.begin ||
            layer.absolute_layer_id >= range.end) {
          continue;
        }
        append_layer(mapping, layer);
      }
    }
    return plan;
  }

  const HostKVLayout& layout_;
  bool use_raw_copy_ = false;
  std::unordered_map<int32_t, std::vector<RawLayerCopyDesc>> raw_descs_by_group_;
  Device device_;
  const Stream& compute_stream_;
  uint32_t layers_per_event_ = 1;
  std::vector<LayerRange> ranges_;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
  CopyStreamQueue streams_;
  bool drained_ = false;
};

BasicHostKVTransfer::BasicHostKVTransfer(
    HostKVLayout layout,
    const Device& device,
    const Stream& compute_stream,
    uint32_t layer_copy_batches,
    std::unique_ptr<BatchMemcpy> batch_memcpy)
    : HostKVTransfer(std::move(layout)),
      impl_(std::make_unique<Impl>(this->layout(),
                                   device,
                                   compute_stream,
                                   layer_copy_batches,
                                   std::move(batch_memcpy))) {}

BasicHostKVTransfer::~BasicHostKVTransfer() { drain(); }

HostKVLoadHandle BasicHostKVTransfer::prepare_load(bool draft) {
  return impl_->prepare_load(draft);
}

void BasicHostKVTransfer::drain() { impl_->drain(); }

uint32_t BasicHostKVTransfer::load_event_count() const {
  return impl_->event_count();
}

uint32_t BasicHostKVTransfer::layers_per_event() const {
  return impl_->layers_per_event();
}

bool BasicHostKVTransfer::load_impl(const HostKVRequest& request,
                                    const HostKVLoadHandle& handle) {
  return impl_->load(request, handle);
}

bool BasicHostKVTransfer::offload_impl(const HostKVRequest& request) {
  return impl_->offload(request);
}

}  // namespace xllm
