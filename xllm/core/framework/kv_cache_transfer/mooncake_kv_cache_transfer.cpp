/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "absl/strings/str_join.h"
#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "framework/kv_cache/cache_layout_builder.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/push_route.h"
#include "framework/parallel_state/linear_attention_parallel.h"
#include "platform/platform.h"
#include "util/net.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {

namespace {

std::string get_merge_key(const uint64_t dst_cluster_id,
                          const std::string& dst_addr) {
  return std::to_string(dst_cluster_id) + "_" + dst_addr;
}

std::vector<KVCacheTensor> get_mooncake_tensors(const KVCache& cache) {
  return cache.get_cache_tensors();
}

void append_mappings(std::vector<KVTransferMapping>& dst,
                     const std::vector<KVTransferMapping>& src) {
  for (const KVTransferMapping& src_mapping : src) {
    auto it = std::find_if(dst.begin(),
                           dst.end(),
                           [&src_mapping](const KVTransferMapping& mapping) {
                             return mapping.group_id == src_mapping.group_id;
                           });
    if (it == dst.end()) {
      dst.emplace_back(src_mapping);
      continue;
    }
    it->local_ids.insert(it->local_ids.end(),
                         src_mapping.local_ids.begin(),
                         src_mapping.local_ids.end());
    it->remote_ids.insert(it->remote_ids.end(),
                          src_mapping.remote_ids.begin(),
                          src_mapping.remote_ids.end());
  }
}

void merge_kv_info(
    std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>&
        merged_kv_infos,
    std::unordered_map<std::string, std::unordered_set<std::string>>&
        seen_request_ids,
    const TransferKVInfo& info,
    const int32_t dst_rank) {
  uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[dst_rank];
  const std::string& dst_addr = info.remote_instance_info.addrs[dst_rank];
  std::string key = get_merge_key(dst_cluster_id, dst_addr);

  auto it = merged_kv_infos.find(key);
  if (it == merged_kv_infos.end()) {
    KVCacheTransfer::KVCacheInfo kv_info;
    kv_info.dst_cluster_id = dst_cluster_id;
    kv_info.dst_addr = dst_addr;
    append_mappings(kv_info.mappings, info.mappings);
    append_unique_request_id(
        seen_request_ids, key, kv_info.request_ids, info.request_id);
    merged_kv_infos.emplace(key, std::move(kv_info));
    return;
  }

  append_mappings(it->second.mappings, info.mappings);
  append_unique_request_id(
      seen_request_ids, key, it->second.request_ids, info.request_id);
}

}  // namespace

// ============================================================================
// MooncakeKVCacheTransferBase
// ============================================================================

MooncakeKVCacheTransferBase::MooncakeKVCacheTransferBase(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : device_id_(device_id),
      device_(device),
      listen_port_(listen_port),
      mooncake_te_(std::move(engine)) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
  ShortUUID uuid;
  incarnation_id_ = uuid.random();
}

void MooncakeKVCacheTransferBase::initialize(int32_t device_id) {
  (void)device_id;
  addr_ = mooncake_te_->initialize();
  // A failed transfer-engine init returns "" here -- most often because the
  // BRPC listen port is already bound by another rank process. Fail at the
  // source, where the cause is still known: letting the empty address through
  // reports it much later as "Failed to publish cache layout: cache layout
  // identity and compatibility fields are required", which blames the layout
  // fields and cost real debugging time on the 26000-port collision.
  CHECK(!addr_.empty()) << "Mooncake transfer engine failed to initialize "
                           "(listen port "
                        << listen_port_ << " may already be in use)";
}

void MooncakeKVCacheTransferBase::configure_cache_layout(
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args,
    int32_t block_token_capacity,
    bool is_spec_draft,
    const LinearAttentionPolicy& linear_policy) {
  CHECK_GT(parallel_args.dp_size(), 0);
  CHECK_GT(parallel_args.cp_size(), 0);
  CHECK_GT(parallel_args.world_size(), 0);
  CHECK_EQ(parallel_args.world_size() % parallel_args.dp_size(), 0);
  CHECK_EQ(parallel_args.world_size() %
               (parallel_args.dp_size() * parallel_args.cp_size()),
           0);

  CacheRegistrationContext context;
  context.cache_namespace =
      is_spec_draft ? CacheNamespace::SPEC_DRAFT : CacheNamespace::MAIN;
  // The attention/model TP width. llm_engine.cpp derives the same value
  // independently as `dp_local_tp_size_ = dp_local_size_ / cp_size_`; the two
  // agree only by this arithmetic, so
  // MooncakeKVCacheTransferLayoutTest.PublishedTpWidthAndRankAlwaysAgree pins
  // the published (width, rank) pair for the settled shapes.
  const int32_t tp_size = parallel_args.world_size() / parallel_args.dp_size() /
                          parallel_args.cp_size();
  const int32_t rank_in_dp =
      parallel_args.rank() % (parallel_args.cp_size() * tp_size);
  context.coordinates.dp_rank =
      parallel_args.rank() / (parallel_args.cp_size() * tp_size);
  context.coordinates.dp_size = parallel_args.dp_size();
  context.coordinates.tp_rank = rank_in_dp % tp_size;
  context.coordinates.tp_size = tp_size;
  context.coordinates.cp_rank = rank_in_dp / tp_size;
  context.coordinates.cp_size = parallel_args.cp_size();
  context.coordinates.kv_split_rank = parallel_args.kv_split_rank();
  context.coordinates.kv_split_size = parallel_args.kv_split_size_effective();

  context.tensor_layout = make_cache_tensor_layout_context(
      parallel_args, model_args, block_token_capacity, linear_policy);
  const CacheTensorLayoutContext& tensor_layout = context.tensor_layout;
  context.layout_family =
      tensor_layout.head_major_layout ? "head_token_dim" : "token_head_dim";

  context.backend = Platform::type_str();
  std::ostringstream fingerprint;
  fingerprint << model_args.model_type() << ":" << model_args.n_layers() << ":"
              << tensor_layout.kv_head_count << ":" << model_args.head_dim()
              << ":" << tensor_layout.index_head_count;
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    const bool npu_dsv4 =
        !util::is_deepseek_v41_family_model_type(model_args.model_type()) &&
        Platform::is_npu();
    const int64_t kv_block_size = dsv4_kv_block_size(npu_dsv4);
    fingerprint << "|dsv4_compressed_geometry:" << kv_block_size << ":"
                << dsv4_physical_dim(kv_block_size, kDsv4C4CompressRatio) << ":"
                << dsv4_physical_dim(kv_block_size, kDsv4C128CompressRatio);
  }
  if (util::is_deepseek_v41_model_type(model_args.model_type())) {
    // V4.1 grouped pools: the kv-source layer list and the per-kv-source
    // compress ratios decide which layers own history and the fp32 compressor
    // state ring (KV_STATE/SCORE_STATE under SWA). Like the V4 physical dims
    // above they are model-config derived, so peers built from the same
    // weights agree; a peer running a different V4.1 layer plan now fails
    // pairing here with an explicit fingerprint mismatch instead of a
    // resource-semantics error at the first transfer.
    auto append_list = [&fingerprint](const char* name, const auto& values) {
      fingerprint << "|" << name << ":" << absl::StrJoin(values, ",");
      if (!values.empty()) {
        fingerprint << ",";
      }
    };
    append_list("dsv41_grouped_geometry", model_args.kv_source_layer_ids());
    append_list("dsv41_compress_ratios", model_args.compress_ratios());
    append_list("dsv41_engram_layers", model_args.engram_layer_ids());
    fingerprint << "|dsv41_engram_max_ngram_size:"
                << model_args.engram_max_ngram_size();
  }
  if (!is_spec_draft && uses_npu_compressed_kpool_tail(model_args)) {
    fingerprint << "|compressed_kpool_tail_fp32:" << model_args.index_head_dim()
                << ":" << model_args.index_kpool() << ":"
                << block_token_capacity;
  }
  context.fingerprint = fingerprint.str();
  context.cluster_id = cluster_id_;
  context.addr = addr_;
  context.listen_port = listen_port_;
  pending_registration_context_ = std::move(context);
}

void MooncakeKVCacheTransferBase::get_cache_info(uint64_t& cluster_id,
                                                 std::string& addr) {
  cluster_id = cluster_id_;
  addr = addr_;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransferBase::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& ports) {
  if (cluster_ids.size() != ports.size()) {
    LOG(ERROR) << "MoonCake link endpoint and port counts differ.";
    return false;
  }
  return mooncake_te_->link_sessions(cluster_ids, remote_addrs);
}

bool MooncakeKVCacheTransferBase::unlink_cluster(const uint64_t& cluster_id,
                                                 const std::string& remote_addr,
                                                 const uint16_t port,
                                                 bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

// ============================================================================
// MooncakeKVCacheTransferDefault
// ============================================================================

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    const std::string& model_type)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)) {
  (void)model_type;
}

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    const std::string& model_type,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : MooncakeKVCacheTransferBase(device_id,
                                  listen_port,
                                  device,
                                  std::move(engine)) {
  (void)model_type;
}

void MooncakeKVCacheTransferDefault::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  const bool is_spec_draft = main_layout_.registered;
  CHECK(!is_spec_draft || !spec_layout_.registered)
      << "Spec draft kv cache is already registered.";

  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  bool has_v_cache = true;
  if (!kv_caches.empty()) {
    torch::Tensor value_cache = kv_caches[0].get_v_cache();
    has_v_cache = value_cache.defined() && value_cache.numel() > 0;
  }

  (void)dtype;

  const int64_t ssm_checkpoint_stride =
      kv_cache_shape.linear_ssm_checkpoint_stride();
  if (pending_registration_context_.has_value()) {
    CacheTensorLayoutContext& tensor_layout =
        pending_registration_context_->tensor_layout;
    apply_cache_shape_layout(kv_cache_shape, &tensor_layout);
  }

  BufLayout layout;
  layout.num_layers = num_layers;
  layout.layers.resize(static_cast<size_t>(num_layers));
  if (is_spec_draft) {
    layout.offset = main_layout_.offset + main_layout_.total_buf_cnt;
  }
  std::vector<CacheTensorManifest> tensor_manifests;
  // Logical layers may alias a physical prefill slot. Assign one SDK buffer
  // id per allocation while preserving every logical layer's manifest entry.
  std::unordered_map<void*, std::pair<RegisteredBufferDesc, uint64_t>>
      physical_buffers;
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    const std::vector<KVCacheTensor> transfer_tensors =
        get_mooncake_tensors(kv_caches[static_cast<size_t>(layer_id)]);
    std::vector<RegisteredBufferDesc>& layer_buffers =
        layout.layers[static_cast<size_t>(layer_id)];
    layer_buffers.reserve(transfer_tensors.size());
    for (const KVCacheTensor& cache_tensor : transfer_tensors) {
      const torch::Tensor& tensor = cache_tensor.tensor;
      CHECK(tensor.defined() && tensor.numel() > 0)
          << "Mooncake cache tensor must be allocated, layer=" << layer_id
          << ", role=" << cache_tensor.role.to_string();
      CHECK_GT(tensor.dim(), 0);
      const int64_t physical_row_count = tensor.size(0);
      CHECK_GT(physical_row_count, 0);
      const int64_t rows_per_resource =
          physical_rows_per_resource(cache_tensor.role,
                                     kv_cache_shape.replicated_block_pages(),
                                     ssm_checkpoint_stride);
      CHECK_EQ(physical_row_count % rows_per_resource, 0)
          << "Cache tensor physical rows must be divisible by its logical "
             "resource geometry, role="
          << cache_tensor.role.to_string();
      const int64_t resource_count = physical_row_count / rows_per_resource;
      const uint64_t logical_bytes = static_cast<uint64_t>(tensor.nbytes());
      CHECK_EQ(logical_bytes % static_cast<uint64_t>(resource_count), 0);

      RegisteredBufferDesc desc{
          layout.offset + layout.total_buf_cnt,
          cache_tensor.role,
          cache_tensor.group_id,
          logical_bytes / static_cast<uint64_t>(resource_count)};
      const auto [physical, inserted] = physical_buffers.emplace(
          tensor.data_ptr(), std::make_pair(desc, logical_bytes));
      CHECK_EQ(physical->second.second, logical_bytes)
          << "Aliased KV buffers must have equal allocation sizes.";
      CHECK_EQ(physical->second.first.block_bytes, desc.block_bytes)
          << "Aliased KV buffers must have equal resource geometry.";
      desc.buf_id = physical->second.first.buf_id;
      if (inserted) {
        ++layout.total_buf_cnt;
      }
      const int64_t buffer_id = desc.buf_id;
      layer_buffers.emplace_back(std::move(desc));
      layout.group_ids.emplace(cache_tensor.group_id);
      if (pending_registration_context_.has_value()) {
        KVCacheTensor described_tensor = cache_tensor;
        std::string descriptor_error;
        CHECK(
            describe_cache_tensor(pending_registration_context_->tensor_layout,
                                  &described_tensor,
                                  &descriptor_error))
            << "Failed to describe cache tensor, layer=" << layer_id
            << ", role=" << cache_tensor.role.to_string() << ": "
            << descriptor_error;
        CHECK(described_tensor.shard_descriptor.has_value());
        CacheTensorManifest tensor_manifest;
        tensor_manifest.cache_namespace =
            pending_registration_context_->cache_namespace;
        tensor_manifest.layer_id = layer_id;
        tensor_manifest.role = static_cast<int32_t>(
            static_cast<KVCacheTensorRole::Value>(cache_tensor.role));
        tensor_manifest.group_id = cache_tensor.group_id;
        tensor_manifest.mooncake_buffer_id = buffer_id;
        tensor_manifest.scalar_type =
            static_cast<int32_t>(tensor.scalar_type());
        tensor_manifest.element_bytes = tensor.element_size();
        tensor_manifest.shape = tensor.sizes().vec();
        tensor_manifest.stride = tensor.strides().vec();
        tensor_manifest.storage_offset_bytes = 0;
        tensor_manifest.contiguous = tensor.is_contiguous();
        tensor_manifest.resource_count = static_cast<uint64_t>(resource_count);
        tensor_manifest.physical_rows_per_resource =
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.resource_stride_bytes =
            static_cast<uint64_t>(tensor.stride(0)) * tensor.element_size() *
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.buffer_bytes = static_cast<uint64_t>(tensor.nbytes());
        tensor_manifest.block_token_capacity =
            static_cast<uint64_t>(std::max<int64_t>(
                effective_block_token_capacity(
                    pending_registration_context_->tensor_layout, cache_tensor),
                0));
        tensor_manifest.explicit_resource_offsets = false;
        tensor_manifest.shard =
            std::move(described_tensor.shard_descriptor.value());
        tensor_manifests.emplace_back(std::move(tensor_manifest));
      }
    }
    CHECK(!layer_buffers.empty())
        << "No Mooncake cache tensor registered at layer " << layer_id;
  }
  layout.registered = true;

  if (!is_spec_draft) {
    num_layers_ = num_layers;
    has_v_cache_ = has_v_cache;
    main_layout_ = layout;
  } else {
    spec_layout_ = layout;
  }

  register_kv_cache_impl(kv_caches,
                         ssm_checkpoint_stride,
                         kv_cache_shape.replicated_block_pages());
  if (pending_registration_context_.has_value()) {
    publish_cache_layout(tensor_manifests, *pending_registration_context_);
    pending_registration_context_.reset();
  }
}

void MooncakeKVCacheTransferBase::publish_cache_layout(
    const std::vector<CacheTensorManifest>& tensor_manifests,
    const CacheRegistrationContext& registration_context) {
  CHECK(!tensor_manifests.empty());
  if (registration_context.cache_namespace == CacheNamespace::MAIN) {
    local_cache_layout_.tensors.clear();
    local_cache_layout_.draft_coordinates.reset();
    local_cache_layout_.schema_version = kCacheLayoutSchemaVersion;
    local_cache_layout_.incarnation_id = incarnation_id_;
    local_cache_layout_.fingerprint = registration_context.fingerprint;
    local_cache_layout_.backend = registration_context.backend;
    local_cache_layout_.layout_family = registration_context.layout_family;
    local_cache_layout_.cluster_id = registration_context.cluster_id;
    local_cache_layout_.addr = registration_context.addr;
    local_cache_layout_.listen_port = registration_context.listen_port;
    local_cache_layout_.coordinates = registration_context.coordinates;
  } else {
    CHECK(!local_cache_layout_.tensors.empty())
        << "Main cache layout must be published before SPEC_DRAFT.";
    CHECK_EQ(local_cache_layout_.backend, registration_context.backend);
    CHECK_EQ(local_cache_layout_.layout_family,
             registration_context.layout_family);
    // Worker coordinates describe the MAIN cache topology. SPEC_DRAFT tensors
    // carry their independent logical TP placement in each shard descriptor,
    // which permits a replicated TP1 MTP draft body beside a sharded target.
    if (registration_context.coordinates.tp_size >
        local_cache_layout_.coordinates.tp_size) {
      local_cache_layout_.draft_coordinates = registration_context.coordinates;
    }
    local_cache_layout_.fingerprint.append("|spec:");
    local_cache_layout_.fingerprint.append(registration_context.fingerprint);
  }
  local_cache_layout_.tensors.reserve(local_cache_layout_.tensors.size() +
                                      tensor_manifests.size());
  local_cache_layout_.tensors.insert(local_cache_layout_.tensors.end(),
                                     tensor_manifests.begin(),
                                     tensor_manifests.end());
  local_cache_layout_.layout_generation = ++layout_generation_;
  const Status status =
      mooncake_te_->set_local_cache_layout(local_cache_layout_);
  CHECK(status.ok()) << "Failed to publish cache layout: " << status.message();
}

void MooncakeKVCacheTransferDefault::register_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  CHECK(main_layout_.registered)
      << "Main KV cache must be registered before spec draft KV cache.";
  register_kv_cache(kv_caches, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferDefault::add_buf(const torch::Tensor& tensor,
                                             std::vector<void*>& addrs,
                                             std::vector<size_t>& lens,
                                             std::vector<uint64_t>& buf_bytes,
                                             int64_t rows_per_resource) const {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  CHECK_GT(tensor.dim(), 0) << "cache tensor dim must be positive";
  CHECK(tensor.is_contiguous())
      << "Mooncake registration requires a contiguous cache tensor";
  const int64_t block_count = tensor.size(0);
  CHECK_GT(block_count, 0) << "cache tensor block dim must be positive";
  CHECK_GT(rows_per_resource, 0);
  CHECK_EQ(block_count % rows_per_resource, 0)
      << "cache tensor rows must be divisible by physical rows per resource";
  const int64_t resource_count = block_count / rows_per_resource;

  const int64_t storage_offset = tensor.storage_offset();
  CHECK_GE(storage_offset, 0) << "tensor storage offset must be non-negative";
  const size_t element_size = tensor.element_size();
  CHECK_GT(element_size, static_cast<size_t>(0))
      << "tensor element byte size must be positive";
  CHECK_LE(static_cast<size_t>(storage_offset),
           std::numeric_limits<size_t>::max() / element_size)
      << "tensor storage offset byte size overflow";
  const size_t storage_offset_bytes =
      static_cast<size_t>(storage_offset) * element_size;
  const size_t storage_bytes = tensor.storage().nbytes();
  CHECK_LE(storage_offset_bytes, storage_bytes)
      << "tensor storage offset exceeds storage capacity";
  const size_t available_bytes = storage_bytes - storage_offset_bytes;

  const size_t logical_bytes = static_cast<size_t>(tensor.nbytes());
  CHECK_EQ(logical_bytes % static_cast<size_t>(resource_count),
           static_cast<size_t>(0))
      << "cache tensor bytes must be divisible by resource count";
  const size_t block_bytes =
      logical_bytes / static_cast<size_t>(resource_count);
  CHECK_GT(block_bytes, static_cast<size_t>(0))
      << "cache tensor block byte size must be positive";

  CHECK_GE(available_bytes, logical_bytes)
      << "Mooncake registration exceeds tensor storage capacity: "
      << "logical_bytes=" << logical_bytes
      << ", available_bytes=" << available_bytes
      << ", block_bytes=" << block_bytes;

  addrs.emplace_back(tensor.data_ptr());
  lens.emplace_back(logical_bytes);
  buf_bytes.emplace_back(static_cast<uint64_t>(block_bytes));
}

bool MooncakeKVCacheTransferDefault::append_buffer_mappings(
    const BufLayout& layout,
    const std::vector<int64_t>& layer_ids,
    const std::vector<KVTransferMapping>& mappings,
    std::vector<MooncakeTransferEngine::BufferTransferMapping>* buffer_mappings)
    const {
  CHECK(buffer_mappings != nullptr);
  CHECK(layout.registered) << "KV cache is not registered.";
  CHECK_EQ(layout.layers.size(), static_cast<size_t>(layout.num_layers));

  std::unordered_map<int32_t, const KVTransferMapping*> mappings_by_group;
  mappings_by_group.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.local_ids.size() != mapping.remote_ids.size()) {
      LOG(ERROR) << "KV cache mapping size mismatch, group_id="
                 << mapping.group_id << ", local=" << mapping.local_ids.size()
                 << ", remote=" << mapping.remote_ids.size();
      return false;
    }
    if (!mappings_by_group.emplace(mapping.group_id, &mapping).second) {
      LOG(ERROR) << "Duplicate KV cache transfer mapping, group_id="
                 << mapping.group_id;
      return false;
    }
  }

  std::vector<int64_t> active_layer_ids;
  if (layer_ids.empty()) {
    active_layer_ids.resize(static_cast<size_t>(layout.num_layers));
    std::iota(active_layer_ids.begin(), active_layer_ids.end(), 0);
  } else {
    active_layer_ids = layer_ids;
  }

  for (int64_t layer_id : active_layer_ids) {
    CHECK_GE(layer_id, 0) << "layer_id must be non-negative";
    CHECK_LT(layer_id, layout.num_layers) << "layer_id out of range";
  }

  for (int64_t layer_id : active_layer_ids) {
    const std::vector<RegisteredBufferDesc>& layer_buffers =
        layout.layers[static_cast<size_t>(layer_id)];
    for (const RegisteredBufferDesc& buffer : layer_buffers) {
      const auto mapping_it = mappings_by_group.find(buffer.group_id);
      if (mapping_it == mappings_by_group.end()) {
        LOG(ERROR) << "Missing KV cache transfer mapping, layer=" << layer_id
                   << ", buf_id=" << buffer.buf_id
                   << ", role=" << buffer.role.to_string()
                   << ", group_id=" << buffer.group_id;
        return false;
      }
      const KVTransferMapping& mapping = *mapping_it->second;
      if (mapping.local_ids.empty()) {
        continue;
      }
      MooncakeTransferEngine::BufferTransferMapping buffer_mapping;
      buffer_mapping.buf_id = buffer.buf_id;
      buffer_mapping.local_ids = mapping.local_ids;
      buffer_mapping.remote_ids = mapping.remote_ids;
      buffer_mappings->emplace_back(std::move(buffer_mapping));
    }
  }
  return true;
}

void MooncakeKVCacheTransferDefault::register_kv_cache_impl(
    const std::vector<xllm::KVCache>& kv_caches,
    int64_t ssm_checkpoint_stride,
    int64_t replicated_block_pages) {
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> buf_bytes;
  addrs.reserve(kv_caches.size() * 4);
  lens.reserve(kv_caches.size() * 4);
  buf_bytes.reserve(kv_caches.size() * 4);
  std::unordered_set<void*> physical_buffers;

  for (const KVCache& cache : kv_caches) {
    const std::vector<KVCacheTensor> transfer_tensors =
        get_mooncake_tensors(cache);
    for (const KVCacheTensor& cache_tensor : transfer_tensors) {
      if (!physical_buffers.emplace(cache_tensor.tensor.data_ptr()).second) {
        continue;
      }
      add_buf(cache_tensor.tensor,
              addrs,
              lens,
              buf_bytes,
              physical_rows_per_resource(cache_tensor.role,
                                         replicated_block_pages,
                                         ssm_checkpoint_stride));
    }
  }

  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(FATAL) << "register_kv_cache_impl failed";
  }

  LOG(INFO) << "register_kv_cache_impl success, registered_layers="
            << kv_caches.size() << ", buffers=" << buf_bytes.size();
}

bool MooncakeKVCacheTransferDefault::prepare_draft_transfer_infos(
    std::vector<TransferKVInfo>& transfer_kv_infos) const {
  if (!spec_layout_.registered || spec_layout_.group_ids.empty()) {
    LOG(ERROR) << "Draft KV cache groups are not registered.";
    return false;
  }
  for (TransferKVInfo& info : transfer_kv_infos) {
    for (const int32_t group_id : spec_layout_.group_ids) {
      const bool has_mapping =
          std::any_of(info.mappings.begin(),
                      info.mappings.end(),
                      [group_id](const KVTransferMapping& mapping) {
                        return mapping.group_id == group_id;
                      });
      if (!has_mapping) {
        LOG(ERROR) << "Missing draft KV cache transfer mapping, request_id="
                   << info.request_id << ", group_id=" << group_id;
        return false;
      }
    }
    std::vector<KVTransferMapping> retained;
    retained.reserve(info.mappings.size());
    for (KVTransferMapping& mapping : info.mappings) {
      if (!spec_layout_.group_ids.contains(mapping.group_id)) {
        continue;
      }
      retained.emplace_back(std::move(mapping));
    }
    info.mappings = std::move(retained);
  }
  return true;
}

bool MooncakeKVCacheTransferDefault::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  std::vector<int64_t> layer_ids;
  std::vector<MooncakeTransferEngine::BufferTransferMapping> buffer_mappings;
  if (!append_buffer_mappings(
          main_layout_, layer_ids, mappings, &buffer_mappings)) {
    return false;
  }
  if (spec_layout_.registered &&
      !append_buffer_mappings(
          spec_layout_, layer_ids, mappings, &buffer_mappings)) {
    return false;
  }
  const bool success = mooncake_te_->move_memory_groups(
      src_addr, buffer_mappings, MooncakeTransferEngine::MoveOpcode::READ);
  if (!success) {
    LOG(ERROR) << "Pull KV cache mappings failed.";
    return false;
  }
  VLOG(1) << "[Mooncake][PDTransfer] direction=pull, cluster_id="
          << src_cluster_id << ", remote=" << src_addr
          << ", mapping_groups=" << mappings.size()
          << ", buffers=" << buffer_mappings.size() << ", success=true";
  return true;
}

void MooncakeKVCacheTransferBase::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  (void)parallel_args;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      seen_request_ids;
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t dst_dp_size = info.remote_instance_info.dp_size;
    const int32_t dst_world_size =
        static_cast<int32_t>(info.remote_instance_info.cluster_ids.size());
    if (dst_dp_size <= 0 || dst_world_size <= 0 ||
        dst_world_size % dst_dp_size != 0 || info.dp_rank < 0 ||
        info.dp_rank >= dst_dp_size ||
        info.remote_instance_info.addrs.size() !=
            info.remote_instance_info.cluster_ids.size()) {
      LOG(ERROR) << "Invalid destination topology while merging KV regions.";
      continue;
    }
    const int32_t dst_tp_size = dst_world_size / dst_dp_size;
    const int32_t begin = info.dp_rank * dst_tp_size;
    const int32_t end = begin + dst_tp_size;
    for (int32_t dst_rank = begin; dst_rank < end; ++dst_rank) {
      merge_kv_info(merged_kv_infos, seen_request_ids, info, dst_rank);
    }
  }
}

std::vector<KVTransferTaskResult>
MooncakeKVCacheTransferDefault::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  const BufLayout& layout = is_spec_draft ? spec_layout_ : main_layout_;
  CHECK(layout.registered) << "KV cache is not registered.";
  const int64_t num_layers = layout.num_layers;

  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    keys.push_back(pair.first);
  }
  if (kv_split_size > 1) {
    keys = rotate_dst_rank(keys, kv_split_rank);
  }

  std::unordered_set<std::string> failed_keys;
  // Failures whose batch retirement proved the destination buffers were no
  // longer written; only these may be reported as UNSAFE to the caller.
  std::unordered_set<std::string> unsafe_keys;
  const CacheNamespace cache_namespace =
      is_spec_draft ? CacheNamespace::SPEC_DRAFT : CacheNamespace::MAIN;
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize KV cache layer failed, layer=" << layer_index;
      failed_keys.insert(keys.begin(), keys.end());
      if (!is_spec_draft) {
        notify_push_completion(layer_index, false);
      }
      continue;
    }
    bool layer_safe = true;
    for (const std::string& key : keys) {
      if (failed_keys.find(key) != failed_keys.end()) {
        continue;
      }
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      std::vector<ByteRegion> regions;
      const Status bind_status =
          mooncake_te_->bind_outgoing_regions(kv_info.dst_addr,
                                              kv_info.mappings,
                                              cache_namespace,
                                              layer_index,
                                              &regions);
      if (!bind_status.ok()) {
        LOG(ERROR) << "Bind KV byte regions failed, layer=" << layer_index
                   << ", destination=" << kv_info.dst_addr << ": "
                   << bind_status.message();
        failed_keys.insert(key);
        continue;
      }
      if (regions.empty()) {
        continue;
      }
      bool unsafe = false;
      const bool success = mooncake_te_->move_memory_regions(
          kv_info.dst_addr,
          regions,
          MooncakeTransferEngine::MoveOpcode::WRITE,
          &unsafe);
      if (!success) {
        LOG(ERROR) << "Push KV byte regions failed, layer=" << layer_index
                   << ", destination=" << kv_info.dst_addr
                   << ", unsafe=" << unsafe;
        failed_keys.insert(key);
        if (unsafe) {
          unsafe_keys.insert(key);
        }
        layer_safe = false;
      }
    }
    if (!is_spec_draft) {
      notify_push_completion(layer_index, layer_safe);
    }
  }
  std::vector<KVTransferTaskResult> results;
  results.reserve(failed_keys.size());
  for (const std::string& key : keys) {
    if (failed_keys.find(key) == failed_keys.end()) {
      continue;
    }
    results.push_back({merged_kv_infos.at(key).request_ids,
                       unsafe_keys.find(key) == unsafe_keys.end()
                           ? KVTransferErrorCode::FAILED
                           : KVTransferErrorCode::UNSAFE});
  }
  VLOG(10) << "[Mooncake][PDTransfer] direction=push, destinations="
           << keys.size() << ", layers=" << num_layers
           << ", failed_tasks=" << results.size();
  return results;
}

// ============================================================================
}  // namespace xllm
