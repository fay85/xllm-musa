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

#include "framework/kv_cache/prefill_layerwise_kv_cache.h"

#include <fcntl.h>
#include <folly/ScopeGuard.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <sstream>
#include <utility>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/process_group.h"
#include "platform/device.h"
#if defined(USE_NPU)
#include <c10/core/DeviceGuard.h>

#include "platform/npu/npu_batch_memcpy.h"
#include "platform/npu/npu_host_registration.h"
#endif

namespace xllm {
namespace {

constexpr int32_t kCopyStreamTimeoutMs = 60000;

int64_t page_aligned_bytes(int64_t bytes) {
  const int64_t page = host_page_size();
  CHECK_LE(bytes, std::numeric_limits<int64_t>::max() - (page - 1));
  return ((bytes + page - 1) / page) * page;
}

torch::Tensor gather_host_metadata(ProcessGroup& group,
                                   const torch::Tensor& metadata) {
  const torch::Tensor result =
      group.allgather_base_sync(metadata.to(group.device()))
          .to(torch::kCPU)
          .contiguous();
  CHECK_EQ(result.dim(), 2);
  CHECK_EQ(result.size(0), group.world_size());
  CHECK_EQ(result.size(1), metadata.numel());
  return result;
}

class HostBatchLock final {
 public:
  explicit HostBatchLock(pthread_mutex_t& mutex) : mutex_(mutex) {
    CHECK_EQ(pthread_mutex_lock(&mutex_), 0)
        << "Prefill Host KV lock lost its owner";
  }

  ~HostBatchLock() { CHECK_EQ(pthread_mutex_unlock(&mutex_), 0); }

  template <typename Ready>
  bool wait(pthread_cond_t& condition, Ready ready) {
    timespec deadline{};
    PCHECK(clock_gettime(CLOCK_MONOTONIC, &deadline) == 0);
    deadline.tv_sec += kCopyStreamTimeoutMs / 1000;
    while (!ready()) {
      const int32_t error =
          pthread_cond_timedwait(&condition, &mutex_, &deadline);
      if (error == ETIMEDOUT) {
        return ready();
      }
      CHECK_EQ(error, 0) << "Prefill Host KV wait lost the shared lock";
    }
    return true;
  }

 private:
  pthread_mutex_t& mutex_;
};

torch::Tensor allocate_device_tensor(const std::vector<int64_t>& shape,
                                     torch::ScalarType dtype,
                                     const torch::Device& device,
                                     const KVCacheCreateOptions& options,
                                     bool indexer_scale = false) {
  if (device.is_cpu()) {
    return torch::zeros(shape, torch::dtype(dtype).device(device));
  }
#if defined(USE_NPU)
  const aclFormat format = indexer_scale
                               ? ACL_FORMAT_ND
                               : get_npu_kv_cache_format(options.model_type());
  // Keep the same allocator selection as the regular KV cache path; the
  // layerwise device slots do not require huge pages for the batched D2H/H2D
  // copies (the Host side is page-aligned separately).
  if (options.enable_kv_cache_huge_page_allocator()) {
    return alloc_npu_huge_page_tensor(shape, dtype, format);
  }
  return at_npu::native::npu_format_cast(
      torch::empty(shape, torch::dtype(dtype).device(device)), format);
#else
  LOG(FATAL) << "Prefill layerwise KV offload requires NPU.";
  return {};
#endif
}

torch::Tensor allocate_host_tensor(const torch::Tensor& device_tensor,
                                   int64_t* host_bytes) {
  if (!device_tensor.defined()) {
    return {};
  }
  if (device_tensor.device().is_cpu()) {
    *host_bytes += static_cast<int64_t>(device_tensor.nbytes());
    return torch::zeros_like(device_tensor);
  }
  auto region = std::make_shared<HostPageAlignedRegion>();
  torch::Tensor tensor;
  create_host_page_aligned_tensor(device_tensor.sizes().vec(),
                                  device_tensor.scalar_type(),
                                  &tensor,
                                  region.get());
  *host_bytes += static_cast<int64_t>(region->total_bytes);
  // Transfer registration is destroyed before the cache owner. Tensor views
  // retain this allocation too, including tensors held by the model executor.
  return torch::from_blob(
      region->base_ptr,
      tensor.sizes(),
      [region](void* /*unused*/) {},
      tensor.options());
}

KVCache make_cache(const KVCacheTensors& kv,
                   const torch::Tensor& index,
                   const std::optional<torch::Tensor>& scale) {
  if (!index.defined()) {
    return KVCache(kv);
  }
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors = kv;
  tensors.index_cache = index;
  tensors.index_cache_scale = scale;
  return KVCache(tensors);
}

void normalize_blocks(std::vector<int64_t>* blocks) {
  std::sort(blocks->begin(), blocks->end());
  blocks->erase(std::unique(blocks->begin(), blocks->end()), blocks->end());
}

}  // namespace

struct PrefillLayerwiseHostPool::BatchState {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  uint64_t batch_id = 0;
  int64_t read_size = 0;
  int64_t write_size = 0;
  int32_t arrivals = 0;
  int32_t remaining = 0;
  bool rejected = false;
  bool failed = false;
};

class PrefillLayerwiseHostPool::Mapping final {
 public:
  Mapping(int32_t descriptor, int64_t bytes, const torch::Device& device)
      : descriptor_(descriptor), bytes_(bytes), device_(device) {
    data_ = mmap(
        nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor_, 0);
    PCHECK(data_ != MAP_FAILED) << "Cannot map Prefill Host KV region";
  }

  ~Mapping() {
#if defined(USE_NPU)
    if (!pinned_.empty()) {
      const c10::DeviceGuard guard(device_);
      for (void* address : pinned_) {
        CHECK_EQ(aclrtHostUnregister(address), ACL_SUCCESS)
            << "Cannot unregister Prefill Host KV region";
      }
    }
#endif
    PCHECK(munmap(data_, bytes_) == 0);
    close(descriptor_);
  }

  void* at(int64_t offset) const { return static_cast<char*>(data_) + offset; }

  void pin(int64_t offset, int64_t bytes) {
#if defined(USE_NPU)
    if (!device_.is_cpu()) {
      const c10::DeviceGuard guard(device_);
      void* address = at(offset);
      PCHECK(mlock(address, bytes) == 0) << "Cannot lock Prefill Host KV pages";
      void* mapped = nullptr;
      CHECK_EQ(
          aclrtHostRegister(address, bytes, ACL_HOST_REGISTER_MAPPED, &mapped),
          ACL_SUCCESS)
          << "Cannot register Prefill Host KV pages, bytes=" << bytes;
      pinned_.push_back(address);
    }
#else
    static_cast<void>(offset);
    static_cast<void>(bytes);
#endif
  }

 private:
  const int32_t descriptor_;
  const int64_t bytes_;
  const torch::Device device_;
  void* data_ = nullptr;
  std::vector<void*> pinned_;
};

PrefillLayerwiseHostPool::PrefillLayerwiseHostPool(ProcessGroup* process_group)
    : process_group_(process_group) {}

bool PrefillLayerwiseHostPool::is_writer(int64_t layer) const {
  return layer % rank_count_ == local_rank_;
}

void PrefillLayerwiseHostPool::initialize(
    const std::vector<KVCache>& device_caches,
    int64_t host_memory_bytes,
    int32_t kv_split_rank) {
  {
    layers_ = static_cast<int64_t>(device_caches.size());
    const torch::Tensor& first_key = device_caches.front().get_k_cache();
    blocks_ = first_key.size(0);
    const auto& device = first_key.device();
    int32_t descriptor = -1;
    int64_t control_bytes = 0;
    bool owner = true;
    int64_t host_data_bytes = 0;
    if (process_group_ != nullptr) {
      ProcessGroup& group = *process_group_;
      std::ifstream boot_file("/proc/sys/kernel/random/boot_id");
      std::string boot_id;
      CHECK(std::getline(boot_file, boot_id) && !boot_id.empty());
      struct stat pid_namespace{};
      PCHECK(stat("/proc/self/ns/pid", &pid_namespace) == 0);
      std::ostringstream layout;
      for (const KVCache& cache : device_caches) {
        for (const KVCacheTensor& entry : cache.get_cache_tensors()) {
          const torch::Tensor& tensor = entry.tensor;
          CHECK(tensor.is_contiguous());
          host_data_bytes += page_aligned_bytes(tensor.nbytes());
          layout << static_cast<int32_t>(entry.role) << ':'
                 << static_cast<int32_t>(tensor.scalar_type()) << ':'
                 << tensor.sizes() << ';';
        }
        layout << '|';
      }
      const int64_t host_tag =
          static_cast<int64_t>(std::hash<std::string>{}(boot_id));
      const int64_t namespace_tag = static_cast<int64_t>(pid_namespace.st_ino);
      const int64_t layout_tag =
          static_cast<int64_t>(std::hash<std::string>{}(layout.str()));
      const auto integer = torch::TensorOptions().dtype(torch::kInt64);
      const auto info = gather_host_metadata(
          group,
          torch::tensor({host_tag,
                         namespace_tag,
                         layout_tag,
                         host_data_bytes,
                         static_cast<int64_t>(kv_split_rank)},
                        integer));
      const auto ranks = info.accessor<int64_t, 2>();
      int32_t leader = -1;
      rank_count_ = 0;
      for (int32_t index = 0; index < group.world_size(); ++index) {
        if (ranks[index][0] != host_tag || ranks[index][1] != namespace_tag ||
            ranks[index][4] != kv_split_rank) {
          continue;
        }
        CHECK(ranks[index][2] == layout_tag &&
              ranks[index][3] == host_data_bytes)
            << "Inconsistent Prefill Host KV layout on local rank " << index;
        if (leader < 0) {
          leader = index;
        }
        if (index == group.rank()) {
          local_rank_ = rank_count_;
        }
        ++rank_count_;
      }
      CHECK_GE(leader, 0);
      CHECK_GT(rank_count_, 0);
      if (shared()) {
        control_bytes = page_aligned_bytes(
            sizeof(BatchState) +
            (rank_count_ + layers_ + 2 * blocks_) * sizeof(int64_t));
        CHECK_LE(host_data_bytes, host_memory_bytes);
        owner = group.rank() == leader;
      }
      const int64_t mapping_bytes = control_bytes + host_data_bytes;
      struct stat identity{};
      if (shared() && owner) {
        descriptor = memfd_create("xllm-prefill-host-kv", MFD_CLOEXEC);
        PCHECK(descriptor >= 0) << "Cannot create shared Prefill Host KV";
        PCHECK(ftruncate(descriptor, mapping_bytes) == 0);
        PCHECK(fstat(descriptor, &identity) == 0);
      }
      const auto locations = gather_host_metadata(
          group,
          torch::tensor({static_cast<int64_t>(getpid()),
                         static_cast<int64_t>(descriptor),
                         static_cast<int64_t>(identity.st_dev),
                         static_cast<int64_t>(identity.st_ino)},
                        integer));
      if (shared() && !owner) {
        const auto leader_info = locations.accessor<int64_t, 2>()[leader];
        const std::string path = "/proc/" + std::to_string(leader_info[0]) +
                                 "/fd/" + std::to_string(leader_info[1]);
        descriptor = open(path.c_str(), O_RDWR | O_CLOEXEC);
        PCHECK(descriptor >= 0) << "Cannot open peer Prefill Host KV: " << path;
        PCHECK(fstat(descriptor, &identity) == 0);
        CHECK(static_cast<int64_t>(identity.st_dev) == leader_info[2] &&
              static_cast<int64_t>(identity.st_ino) == leader_info[3] &&
              identity.st_size == mapping_bytes)
            << "Prefill Host KV descriptor does not match the local leader";
      }
    }
    if (shared()) {
      mapping_ = std::make_shared<Mapping>(
          descriptor, control_bytes + host_data_bytes, device);
      batch_ = static_cast<BatchState*>(mapping_->at(0));
      if (owner) {
        new (batch_) BatchState{};
        pthread_mutexattr_t mutex_attributes;
        pthread_condattr_t condition_attributes;
        CHECK_EQ(pthread_mutexattr_init(&mutex_attributes), 0);
        CHECK_EQ(pthread_condattr_init(&condition_attributes), 0);
        CHECK_EQ(pthread_mutexattr_setpshared(&mutex_attributes,
                                              PTHREAD_PROCESS_SHARED),
                 0);
        CHECK_EQ(pthread_mutexattr_setrobust(&mutex_attributes,
                                             PTHREAD_MUTEX_ROBUST),
                 0);
        CHECK_EQ(pthread_condattr_setpshared(&condition_attributes,
                                             PTHREAD_PROCESS_SHARED),
                 0);
        CHECK_EQ(
            pthread_condattr_setclock(&condition_attributes, CLOCK_MONOTONIC),
            0);
        CHECK_EQ(pthread_mutex_init(&batch_->mutex, &mutex_attributes), 0);
        CHECK_EQ(pthread_cond_init(&batch_->condition, &condition_attributes),
                 0);
        CHECK_EQ(pthread_mutexattr_destroy(&mutex_attributes), 0);
        CHECK_EQ(pthread_condattr_destroy(&condition_attributes), 0);
      }
    }
    const auto host_tensor = [&](const torch::Tensor& tensor) -> torch::Tensor {
      if (!tensor.defined()) {
        return {};
      }
      if (descriptor < 0) {
        return allocate_host_tensor(tensor, &bytes_);
      }
      const int64_t size = page_aligned_bytes(tensor.nbytes());
      const int64_t offset = control_bytes + bytes_;
      mapping_->pin(offset, size);
      bytes_ += size;
      return torch::from_blob(
          mapping_->at(offset),
          tensor.sizes(),
          [mapping = mapping_](void*) {},
          tensor.options().device(torch::kCPU));
    };
    caches_.reserve(device_caches.size());
    for (const KVCache& cache : device_caches) {
      KVCacheTensors kv{host_tensor(cache.get_k_cache()),
                        host_tensor(cache.get_v_cache())};
      std::optional<torch::Tensor> scale = cache.get_indexer_cache_scale();
      if (scale.has_value()) {
        scale = host_tensor(*scale);
      }
      caches_.emplace_back(
          make_cache(kv, host_tensor(cache.get_index_cache()), scale));
      CHECK_LE(bytes_, host_memory_bytes)
          << "Prefill layerwise Host cache exceeds its configured budget.";
    }
    if (process_group_ != nullptr) {
      if (shared()) {
        CHECK_EQ(bytes_, host_data_bytes);
      }
      gather_host_metadata(*process_group_, torch::ones({1}, torch::kInt64));
    }
  }
  if (!shared()) {
    return;
  }
  CHECK_LE(bytes_, host_memory_bytes);
  CHECK_EQ(caches_.size(), device_caches.size());
  for (size_t layer = 0; layer < caches_.size(); ++layer) {
    const std::vector<KVCacheTensor> host = caches_[layer].get_cache_tensors();
    const std::vector<KVCacheTensor> device =
        device_caches[layer].get_cache_tensors();
    CHECK_EQ(host.size(), device.size());
    for (size_t role = 0; role < host.size(); ++role) {
      CHECK(host[role].role == device[role].role &&
            host[role].tensor.sizes() == device[role].tensor.sizes() &&
            host[role].tensor.strides() == device[role].tensor.strides() &&
            host[role].tensor.scalar_type() ==
                device[role].tensor.scalar_type())
          << "Inconsistent shared Prefill Host KV layout, layer=" << layer;
    }
  }
}

bool PrefillLayerwiseHostPool::enter(uint64_t batch_id,
                                     const std::vector<int64_t>& read_blocks,
                                     const std::vector<int64_t>& write_blocks,
                                     bool valid) {
  BatchState& state = *batch_;
  auto* entered = static_cast<int64_t*>(mapping_->at(sizeof(BatchState)));
  int64_t* readers = entered + rank_count_;
  int64_t* batch_reads = readers + layers_;
  int64_t* batch_writes = batch_reads + blocks_;
  HostBatchLock lock(state.mutex);
  if (!lock.wait(state.condition,
                 [&] {
                   return state.failed || state.remaining == 0 ||
                          !entered[local_rank_];
                 }) ||
      state.failed) {
    LOG(FATAL) << "Cannot enter shared Prefill Host KV batch, batch_id="
               << batch_id << ", active_batch=" << state.batch_id
               << ", rank=" << local_rank_ << ", remaining=" << state.remaining;
  }
  if (state.remaining == 0) {
    state.batch_id = batch_id;
    state.arrivals = 0;
    state.remaining = rank_count_;
    state.rejected = !valid;
    if (valid) {
      state.read_size = static_cast<int64_t>(read_blocks.size());
      state.write_size = static_cast<int64_t>(write_blocks.size());
      std::copy(read_blocks.begin(), read_blocks.end(), batch_reads);
      std::copy(write_blocks.begin(), write_blocks.end(), batch_writes);
    }
    std::fill_n(entered, rank_count_, 0);
    std::fill_n(readers, layers_, 0);
  }
  if (!state.rejected && (!valid || batch_id != state.batch_id ||
                          !std::equal(read_blocks.begin(),
                                      read_blocks.end(),
                                      batch_reads,
                                      batch_reads + state.read_size) ||
                          !std::equal(write_blocks.begin(),
                                      write_blocks.end(),
                                      batch_writes,
                                      batch_writes + state.write_size))) {
    state.rejected = true;
    LOG(ERROR) << "Rejected inconsistent shared Prefill Host KV plan, batch_id="
               << batch_id << ", expected_batch=" << state.batch_id
               << ", rank=" << local_rank_ << ", valid=" << valid;
  }
  entered[local_rank_] = 1;
  if (++state.arrivals == rank_count_) {
    CHECK_EQ(pthread_cond_broadcast(&state.condition), 0);
  }
  if (!lock.wait(
          state.condition,
          [&] { return state.failed || state.arrivals == rank_count_; }) ||
      state.failed) {
    LOG(FATAL) << "Shared Prefill Host KV batch lost its cohort, batch_id="
               << state.batch_id << ", arrivals=" << state.arrivals;
  }
  return !state.rejected;
}

bool PrefillLayerwiseHostPool::read_complete(int64_t layer) {
  BatchState& state = *batch_;
  auto* readers =
      static_cast<int64_t*>(mapping_->at(sizeof(BatchState))) + rank_count_;
  HostBatchLock lock(state.mutex);
  if (++readers[layer] == rank_count_) {
    CHECK_EQ(pthread_cond_broadcast(&state.condition), 0);
  }
  if (is_writer(layer) && !lock.wait(state.condition, [&] {
        return state.failed || readers[layer] == rank_count_;
      })) {
    LOG(ERROR) << "Timed out waiting for Prefill Host KV readers, batch_id="
               << state.batch_id << ", layer=" << layer;
    return false;
  }
  return !state.failed;
}

void PrefillLayerwiseHostPool::leave() {
  HostBatchLock lock(batch_->mutex);
  if (--batch_->remaining == 0) {
    CHECK_EQ(pthread_cond_broadcast(&batch_->condition), 0);
  }
}

void PrefillLayerwiseHostPool::fail() {
  if (batch_ != nullptr) {
    HostBatchLock lock(batch_->mutex);
    batch_->failed = true;
    CHECK_EQ(pthread_cond_broadcast(&batch_->condition), 0);
  }
}

PrefillLayerwiseKVCache::PrefillLayerwiseKVCache(
    const KVCacheShape& shape,
    const KVCacheCreateOptions& options,
    int32_t num_buffers,
    int64_t host_memory_bytes,
    std::vector<KVCache>& device_caches,
    ProcessGroup* host_group,
    int32_t kv_split_size,
    int32_t kv_split_rank)
    : device_caches_(&device_caches),
      host_pool_(host_group),
      kv_split_size_(kv_split_size),
      num_buffers_(num_buffers),
      block_count_(shape.key_cache_shape().at(0)) {
  // These are startup allocation invariants, not request validation.
  CHECK(device_caches.empty());
  CHECK_GT(options.num_layers(), 0);
  CHECK_GT(num_buffers_, 0);
  CHECK_LE(num_buffers_, options.num_layers());
  CHECK_GT(host_memory_bytes, 0);
  CHECK_GT(block_count_, 0);
  CHECK_GT(kv_split_size_, 0);
  CHECK_GE(kv_split_rank, 0);
  CHECK_LT(kv_split_rank, kv_split_size_);
  CHECK(!shape.has_grouped_cache_layout());
  CHECK(!shape.has_kpool_tail_shape());
  CHECK(!options.enable_linear_attention());
  CHECK(!options.enable_kv_cache_quant());
  CHECK(options.layer_cache_owned().empty());

  const int64_t layers = options.num_layers();
  const std::vector<bool>& index_mask = options.indexer_cache_enabled_layers();
  CHECK(index_mask.empty() || index_mask.size() == static_cast<size_t>(layers));
  const std::vector<char> slot_has_index = prefill_layerwise_index_slots(
      layers, num_buffers_, shape.has_index_cache_shape(), index_mask);
  num_buffers_ =
      static_cast<int32_t>(std::min<int64_t>(num_buffers_, layers - 1));
  const auto has_index = [&shape, &index_mask](int64_t layer) {
    return shape.has_index_cache_shape() &&
           (index_mask.empty() || index_mask[static_cast<size_t>(layer)]);
  };

  std::vector<KVCache> slots;
  slots.reserve(slot_has_index.size());
  const torch::Device& device = options.device();
  for (size_t slot = 0; slot < slot_has_index.size(); ++slot) {
    KVCacheTensors kv;
    kv.key_cache = allocate_device_tensor(
        shape.key_cache_shape(), options.dtype(), device, options);
    if (shape.has_value_cache_shape()) {
      kv.value_cache = allocate_device_tensor(
          shape.value_cache_shape(), options.dtype(), device, options);
    }
    torch::Tensor index;
    std::optional<torch::Tensor> scale;
    if (slot_has_index[static_cast<size_t>(slot)]) {
      index = allocate_device_tensor(
          shape.index_cache_shape(),
          options.enable_indexer_cache_quant() ? torch::kChar : options.dtype(),
          device,
          options);
      if (options.enable_indexer_cache_quant()) {
        CHECK(shape.has_index_cache_scale_shape());
        scale = allocate_device_tensor(shape.index_cache_scale_shape(),
                                       torch::kFloat16,
                                       device,
                                       options,
                                       /*indexer_scale=*/true);
      }
    }
    slots.emplace_back(make_cache(kv, index, scale));
    for (const KVCacheTensor& tensor : slots.back().get_cache_tensors()) {
      if (kv_split_size_ > 1) {
        CHECK_EQ(fold_tensor_rows_for_transfer(
                     tensor.tensor, tensor.role, /*replicated_block_pages=*/1)
                     .size(0),
                 block_count_)
            << "Invalid folded Prefill KV row count: slot=" << slot
            << ", role=" << tensor.role.to_string()
            << ", shape=" << tensor.tensor.sizes()
            << ", kv_split_size=" << kv_split_size_;
      }
      device_bytes_ += static_cast<int64_t>(tensor.tensor.nbytes());
    }
  }

  device_caches.reserve(static_cast<size_t>(layers));
  for (int64_t layer = 0; layer < layers; ++layer) {
    const KVCache& slot = slots[static_cast<size_t>(
        prefill_layerwise_buffer_index(layer, num_buffers_))];
    const KVCacheTensors device_kv{slot.get_k_cache(), slot.get_v_cache()};
    const torch::Tensor index =
        has_index(layer) ? slot.get_index_cache() : torch::Tensor{};
    const std::optional<torch::Tensor> scale =
        has_index(layer) ? slot.get_indexer_cache_scale() : std::nullopt;
    device_caches.emplace_back(make_cache(device_kv, index, scale));
  }
  host_pool_.initialize(device_caches, host_memory_bytes, kv_split_rank);
  load_ready_.resize(static_cast<size_t>(layers));
  save_ready_.resize(static_cast<size_t>(layers));
  push_completed_layers_.resize(static_cast<size_t>(layers), 0);
  if (!device.is_cpu()) {
    Device platform_device(device);
#if defined(USE_NPU)
    batch_memcpy_ =
        std::make_unique<npu::NPUBatchMemcpy>(/*submit_asynchronously=*/true);
    batch_memcpy_->init(device.index());
#endif
    CHECK(batch_memcpy_ != nullptr);
    load_stream_ = platform_device.get_stream_from_pool(kCopyStreamTimeoutMs);
    save_stream_ = platform_device.get_stream_from_pool(kCopyStreamTimeoutMs);
  }
  LOG(INFO) << "Prefill layerwise KV cache: layers=" << layers
            << ", resident_layer=0, shared_buffers=" << num_buffers_
            << ", device_bytes=" << device_bytes_
            << ", visible_host_bytes=" << host_bytes() << ", unique_host_bytes="
            << (host_pool_.local_rank_ == 0 ? host_bytes() : 0)
            << ", local_host_ranks=" << host_pool_.rank_count_
            << ", kv_split_size=" << kv_split_size_
            << ", kv_split_rank=" << kv_split_rank << ", PD PUSH source=Device";
}

PrefillLayerwiseKVCache::~PrefillLayerwiseKVCache() {
  if (in_host_batch_) {
    host_pool_.fail();
  }
  drain();
  leave_host_batch();
}

uint32_t PrefillLayerwiseKVCache::size() const {
  return static_cast<uint32_t>(host_caches().size());
}

void PrefillLayerwiseKVCache::require_usable() const {
  if (failed_) {
    LOG(FATAL) << "Prefill layerwise KV cache failed and cannot be reused; "
                  "restart the worker before serving more prefills.";
  }
}

bool PrefillLayerwiseKVCache::begin(const ModelInputParams& params,
                                    int64_t block_size,
                                    const Stream* compute_stream,
                                    bool expect_push) {
  require_usable();
  if (!params.block_copy.swap_blocks.empty()) {
    LOG(ERROR) << "Prefill layerwise KV offload does not support beam-search "
                  "block copy, batch_id="
               << params.meta.batch_id;
    return prepare(
        {}, {}, compute_stream, expect_push, params.meta.batch_id, false);
  }
  std::vector<int64_t> read_blocks;
  std::vector<int64_t> write_blocks;
  const torch::Tensor& tables = params.attention.host.block_tables;
  const std::vector<int32_t>& kv_lengths = params.attention.host.kv_seq_lens;
  const std::vector<int32_t>& q_lengths = params.attention.host.q_seq_lens;
  // Untrusted batch metadata is rejected before the first KV DMA submission.
  bool valid = block_size > 0 && params.meta.batch_forward_type.no_decode() &&
               tables.defined() && tables.device().is_cpu() &&
               tables.dim() == 2 && tables.scalar_type() == torch::kInt32 &&
               tables.is_contiguous() &&
               tables.size(0) == static_cast<int64_t>(kv_lengths.size()) &&
               q_lengths.size() == kv_lengths.size();
  if (valid) {
    const int64_t logical_block_size = block_size * kv_split_size_;
    const auto block_ids = tables.accessor<int32_t, 2>();
    read_blocks.reserve(static_cast<size_t>(tables.numel()));
    write_blocks.reserve(static_cast<size_t>(tables.numel()));
    for (size_t row = 0; row < kv_lengths.size(); ++row) {
      const int64_t kv_length = kv_lengths[row];
      const int64_t q_length = q_lengths[row];
      const int64_t count =
          (kv_length + logical_block_size - 1) / logical_block_size;
      if (q_length < 0 || kv_length < q_length || count > tables.size(1)) {
        valid = false;
        break;
      }
      const int64_t prefix_length = kv_length - q_length;
      const int64_t first_write_block = prefix_length / logical_block_size;
      const bool first_write_block_has_prefix =
          prefix_length % logical_block_size != 0;
      for (int64_t col = 0; col < count; ++col) {
        // Blocks after the first write block contain only this batch's tokens,
        // so attention writes them before reading and an H2D load would move
        // data that is never observed.
        if (col < first_write_block ||
            (col == first_write_block && first_write_block_has_prefix)) {
          read_blocks.emplace_back(block_ids[row][col]);
        }
        if (q_length > 0 && col >= first_write_block) {
          write_blocks.emplace_back(block_ids[row][col]);
        }
      }
    }
  }
  if (!valid) {
    LOG(ERROR) << "Invalid prefill layerwise KV metadata, batch_id="
               << params.meta.batch_id;
  }
  return prepare(std::move(read_blocks),
                 std::move(write_blocks),
                 compute_stream,
                 expect_push,
                 params.meta.batch_id,
                 valid);
}

bool PrefillLayerwiseKVCache::begin(const ModelInputParams& params,
                                    int64_t block_size,
                                    const Stream* compute_stream,
                                    bool expect_push,
                                    ProcessGroup* preparation_group) {
  const bool prepared = begin(params, block_size, compute_stream, expect_push);
  if (preparation_group == nullptr || preparation_group->world_size() <= 1) {
    return prepared;
  }
  auto cancel = folly::makeGuard([this, prepared]() {
    if (prepared) {
      cancel_preparation();
    }
  });
  const torch::TensorOptions options =
      torch::dtype(torch::kInt64)
          .device((*device_caches_)[0].get_k_cache().device());
  const torch::Tensor local_status =
      torch::full({1}, static_cast<int64_t>(prepared), options);
  const torch::Tensor rank_status =
      preparation_group->allgather_base_sync(local_status).to(torch::kCPU);
  // A malformed collective result cannot give a rank-uniform execution
  // decision; continuing would let some ranks enter model collectives alone.
  CHECK_EQ(rank_status.numel(), preparation_group->world_size())
      << "Invalid prefill KV preparation collective result.";
  const bool all_prepared = rank_status.eq(1).all().item<bool>();
  if (all_prepared) {
    cancel.dismiss();
  } else {
    LOG(ERROR) << "Prefill KV preparation rejected across TP ranks, batch_id="
               << params.meta.batch_id << ", rank=" << preparation_group->rank()
               << ", local_prepared=" << prepared;
  }
  return all_prepared;
}

bool PrefillLayerwiseKVCache::begin(std::vector<int64_t> read_blocks,
                                    std::vector<int64_t> write_blocks,
                                    const Stream* compute_stream,
                                    bool expect_push) {
  return prepare(std::move(read_blocks),
                 std::move(write_blocks),
                 compute_stream,
                 expect_push,
                 /*batch_id=*/0,
                 /*metadata_valid=*/true);
}

bool PrefillLayerwiseKVCache::prepare(std::vector<int64_t> read_blocks,
                                      std::vector<int64_t> write_blocks,
                                      const Stream* compute_stream,
                                      bool expect_push,
                                      uint64_t batch_id,
                                      bool metadata_valid) {
  require_usable();
  normalize_blocks(&read_blocks);
  normalize_blocks(&write_blocks);
  const auto valid_blocks = [this](const std::vector<int64_t>& blocks) {
    return blocks.empty() ||
           (blocks.front() >= 0 && blocks.back() < block_count_);
  };
  const bool valid = metadata_valid && !active_ && valid_blocks(read_blocks) &&
                     valid_blocks(write_blocks) &&
                     (batch_memcpy_ == nullptr || compute_stream != nullptr);
  if (!valid) {
    LOG(ERROR) << "Invalid prefill layerwise KV batch: active=" << active_
               << ", failed=" << failed_ << ", block_count=" << block_count_;
  }
  if (active_) {
    if (host_pool_.shared()) {
      host_pool_.fail();
      LOG(FATAL) << "Concurrent Prefill KV batches on local rank "
                 << host_pool_.local_rank_;
    }
    return false;
  }
  if (host_pool_.shared()) {
    const bool accepted =
        host_pool_.enter(batch_id, read_blocks, write_blocks, valid);
    in_host_batch_ = true;
    if (!accepted) {
      leave_host_batch();
      return false;
    }
  } else if (!valid) {
    return false;
  }
  compute_stream_ = compute_stream;
  read_blocks_ = std::move(read_blocks);
  write_blocks_ = std::move(write_blocks);
  host_read_write_overlap_ =
      host_pool_.shared() && size() > 1 &&
      std::any_of(
          read_blocks_.begin(), read_blocks_.end(), [this](int64_t block) {
            return std::binary_search(
                write_blocks_.begin(), write_blocks_.end(), block);
          });
  current_layer_ = -1;
  next_load_layer_ = 1 + num_buffers_;
  std::fill(load_ready_.begin(), load_ready_.end(), nullptr);
  std::fill(save_ready_.begin(), save_ready_.end(), nullptr);
  // Layer 0 can compute immediately from its resident prefix while these
  // initial shared-buffer loads run on the independent H2D stream.
  for (int64_t layer = 1; layer < next_load_layer_; ++layer) {
    if (!load_layer(layer)) {
      // Only private Device scratch has been touched so far. Drain it before
      // returning a request error; no persistent Host data needs repair.
      cancel_preparation();
      return false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    expect_push_ = expect_push;
    push_finished_ = !expect_push;
    std::fill(push_completed_layers_.begin(), push_completed_layers_.end(), 0);
  }
  active_ = true;
  return true;
}

bool PrefillLayerwiseKVCache::synchronize_layer(int64_t layer_index) {
  if (!active_ || failed_ || layer_index != current_layer_ + 1 ||
      layer_index >= static_cast<int64_t>(size())) {
    LOG(ERROR) << "Out-of-order prefill KV layer: requested=" << layer_index
               << ", current=" << current_layer_;
    return false;
  }
  if (current_layer_ >= 0 && !save_layer(current_layer_)) {
    abort();
    return false;
  }
  // Prefetch available slots, but do not stall an already-loaded layer behind
  // a future layer's network write. Wait only when the current layer needs
  // that slot. Local save events fence its D2H consumer independently.
  const int64_t last_prefetch = std::min<int64_t>(
      layer_index + num_buffers_ - 1, static_cast<int64_t>(size()) - 1);
  while (next_load_layer_ <= last_prefetch) {
    if (next_load_layer_ > layer_index &&
        !push_layer_complete(next_load_layer_ - num_buffers_)) {
      break;
    }
    if (!load_layer(next_load_layer_)) {
      abort();
      return false;
    }
    ++next_load_layer_;
  }
  if (batch_memcpy_ != nullptr && layer_index > 0 &&
      !compute_stream_->wait_event(
          load_ready_[static_cast<size_t>(layer_index)])) {
    abort();
    return false;
  }
  current_layer_ = layer_index;
  return true;
}

bool PrefillLayerwiseKVCache::copy_layer(int64_t layer_index, bool load) {
  const std::vector<KVCacheTensor> device =
      (*device_caches_)[static_cast<size_t>(layer_index)].get_cache_tensors();
  const std::vector<KVCacheTensor> host =
      host_caches()[static_cast<size_t>(layer_index)].get_cache_tensors();
  const std::vector<int64_t>& blocks = load ? read_blocks_ : write_blocks_;
  if (blocks.empty()) {
    return true;
  }
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  sources.reserve(device.size() * blocks.size());
  destinations.reserve(device.size() * blocks.size());
  for (size_t role = 0; role < device.size(); ++role) {
    const torch::Tensor device_rows =
        kv_split_size_ == 1
            ? device[role].tensor
            : fold_tensor_rows_for_transfer(device[role].tensor,
                                            device[role].role,
                                            /*replicated_block_pages=*/1);
    const torch::Tensor host_rows =
        kv_split_size_ == 1
            ? host[role].tensor
            : fold_tensor_rows_for_transfer(host[role].tensor,
                                            host[role].role,
                                            /*replicated_block_pages=*/1);
    for (int64_t block : blocks) {
      sources.emplace_back((load ? host_rows : device_rows)[block]);
      destinations.emplace_back((load ? device_rows : host_rows)[block]);
    }
  }
  if (batch_memcpy_ == nullptr) {
    for (size_t index = 0; index < sources.size(); ++index) {
      destinations[index].copy_(sources[index]);
    }
    return true;
  }
  const bool success = load ? batch_memcpy_->submit_h2d(
                                  sources, destinations, load_stream_.get())
                            : batch_memcpy_->submit_d2h(
                                  sources, destinations, save_stream_.get());
  if (!success) {
    LOG(ERROR) << "Prefill KV copy failed: layer=" << layer_index
               << ", direction=" << (load ? "H2D" : "D2H")
               << ", blocks=" << blocks.size();
  }
  return success;
}

bool PrefillLayerwiseKVCache::load_layer(int64_t layer_index) {
  if (layer_index > num_buffers_ &&
      !wait_push_completion(layer_index - num_buffers_)) {
    return false;
  }
  if (batch_memcpy_ != nullptr && layer_index > num_buffers_) {
    const StreamEventPtr& saved =
        save_ready_[static_cast<size_t>(layer_index - num_buffers_)];
    if (saved == nullptr || !load_stream_->wait_event(saved)) {
      return false;
    }
  }
  if (!copy_layer(layer_index, /*load=*/true)) {
    return false;
  }
  if (batch_memcpy_ != nullptr) {
    load_ready_[static_cast<size_t>(layer_index)] =
        load_stream_->record_event_or_sync();
    return load_ready_[static_cast<size_t>(layer_index)] != nullptr;
  }
  return true;
}

bool PrefillLayerwiseKVCache::save_layer(int64_t layer_index) {
  if (layer_index > 0 && host_read_write_overlap_) {
#if defined(USE_NPU)
    if (batch_memcpy_ != nullptr) {
      const StreamEventPtr& loaded =
          load_ready_[static_cast<size_t>(layer_index)];
      if (!loaded->synchronize(kCopyStreamTimeoutMs)) {
        LOG(ERROR) << "Cannot confirm Prefill Host KV read, layer="
                   << layer_index << ", local_rank=" << host_pool_.local_rank_;
        return false;
      }
    }
#endif
    if (!host_pool_.read_complete(layer_index)) {
      return false;
    }
  }
  if (batch_memcpy_ != nullptr) {
    save_stream_->wait_stream(*compute_stream_);
  }
  if (host_pool_.is_writer(layer_index) &&
      !copy_layer(layer_index, /*load=*/false)) {
    return false;
  }
  if (batch_memcpy_ != nullptr) {
    save_ready_[static_cast<size_t>(layer_index)] =
        save_stream_->record_event_or_sync();
    return save_ready_[static_cast<size_t>(layer_index)] != nullptr;
  }
  return true;
}

void PrefillLayerwiseKVCache::cancel_preparation() {
  // No model or PUSH has started, so only scratch loads need draining. This
  // is distinct from execution abort: authoritative Host data is intact.
  drain();
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    expect_push_ = false;
    push_finished_ = true;
    std::fill(push_completed_layers_.begin(), push_completed_layers_.end(), 0);
  }
  compute_stream_ = nullptr;
  active_ = false;
  leave_host_batch();
}

void PrefillLayerwiseKVCache::leave_host_batch() {
  if (in_host_batch_) {
    host_pool_.leave();
    in_host_batch_ = false;
  }
}

void PrefillLayerwiseKVCache::drain() {
  // Failure to establish completion can corrupt shared cache or release live
  // DMA memory; it cannot safely be converted to a single-request error.
  if (load_stream_ != nullptr) {
    CHECK_EQ(load_stream_->synchronize(), 0) << "Cannot drain prefill KV H2D.";
  }
  if (save_stream_ != nullptr) {
    CHECK_EQ(save_stream_->synchronize(), 0) << "Cannot drain prefill KV D2H.";
  }
}

bool PrefillLayerwiseKVCache::finish() {
  if (!active_ || failed_ ||
      current_layer_ + 1 != static_cast<int64_t>(size())) {
    abort();
    return false;
  }
  if (!save_layer(current_layer_)) {
    abort();
    return false;
  }
  drain();
  const bool pushed = wait_push_completion(std::nullopt);
  active_ = false;
  failed_ = !pushed;
  if (!pushed && in_host_batch_) {
    host_pool_.fail();
  }
  leave_host_batch();
  return pushed;
}

void PrefillLayerwiseKVCache::abort() {
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    failed_ = true;
  }
  push_cv_.notify_all();
  if (in_host_batch_) {
    host_pool_.fail();
  }
  drain();
  active_ = false;
  leave_host_batch();
}

void PrefillLayerwiseKVCache::notify_push_completion(
    std::optional<int64_t> layer,
    bool safe_to_reuse) {
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    if (!safe_to_reuse ||
        (layer.has_value() &&
         (*layer < 0 || *layer >= static_cast<int64_t>(size())))) {
      failed_ = true;
    } else if (layer.has_value()) {
      push_completed_layers_[static_cast<size_t>(*layer)] = true;
    } else {
      push_finished_ = true;
    }
  }
  push_cv_.notify_all();
}

bool PrefillLayerwiseKVCache::push_layer_complete(int64_t layer_index) {
  std::lock_guard<std::mutex> lock(push_mutex_);
  return !failed_ && (!expect_push_ || push_finished_ ||
                      push_completed_layers_[static_cast<size_t>(layer_index)]);
}

bool PrefillLayerwiseKVCache::wait_push_completion(
    std::optional<int64_t> layer) {
  std::unique_lock<std::mutex> lock(push_mutex_);
  if (!expect_push_) {
    return !failed_;
  }
  const bool completed = push_cv_.wait_for(
      lock, std::chrono::milliseconds(kCopyStreamTimeoutMs), [this, layer]() {
        return failed_ || push_finished_ ||
               (layer.has_value() &&
                push_completed_layers_[static_cast<size_t>(*layer)]);
      });
  if (!completed) {
    LOG(ERROR) << "Timed out waiting for prefill KV PUSH completion, layer="
               << layer.value_or(-1);
    failed_ = true;
  }
  return !failed_;
}

bool PrefillLayerwiseKVCache::record_stream(int64_t /*layer_index*/,
                                            Stream* /*stream*/) {
  LOG(ERROR) << "Prefill layerwise KV cache is a load synchronizer, not an "
                "event recorder.";
  return false;
}

}  // namespace xllm
