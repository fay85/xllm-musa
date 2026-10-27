/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

#include "core/framework/parallel_state/musa_process_group.h"
#include "core/framework/parallel_state/musa/musa_custom_all_reduce_policy.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm {

namespace {

std::mutex g_custom_allreduce_instances_mutex;
std::mutex g_custom_allreduce_algorithm_mutex;
std::unordered_set<MusaCustomAllReduce*> g_custom_allreduce_instances;
std::unordered_map<MusaCustomAllReduce*, uint64_t> g_long_prefill_generations;

int64_t op_meta_size() {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::meta_size", "")
                       .typed<int64_t()>();
  return op.call();
}

std::tuple<int64_t, torch::Tensor> op_allocate_shared_buffer_and_handle(
    int64_t size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::allocate_shared_buffer_and_handle",
                             "")
          .typed<std::tuple<int64_t, torch::Tensor>(int64_t)>();
  return op.call(size);
}

int64_t op_open_mem_handle(torch::Tensor& handle) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::open_mem_handle", "")
                       .typed<int64_t(torch::Tensor&)>();
  return op.call(handle);
}

int64_t op_init_custom_ar(const std::vector<int64_t>& ipc_ptrs,
                          torch::Tensor& rank_data,
                          int64_t rank,
                          bool fully_connected) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::init_custom_ar", "")
          .typed<int64_t(
              const std::vector<int64_t>&, torch::Tensor&, int64_t, bool)>();
  return op.call(ipc_ptrs, rank_data, rank, fully_connected);
}

void op_set_algorithm(int64_t custom_ar_handle, int64_t algorithm) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::set_algorithm", "")
                       .typed<void(int64_t, int64_t)>();
  op.call(custom_ar_handle, algorithm);
}

void op_register_buffer(int64_t custom_ar_handle,
                        const std::vector<int64_t>& ipc_ptrs) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::register_buffer", "")
                       .typed<void(int64_t, const std::vector<int64_t>&)>();
  op.call(custom_ar_handle, ipc_ptrs);
}

void op_all_reduce(int64_t custom_ar_handle,
                   torch::Tensor& input,
                   torch::Tensor& output,
                   int64_t registered_buffer,
                   int64_t registered_buffer_size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::all_reduce", "")
          .typed<void(
              int64_t, torch::Tensor&, torch::Tensor&, int64_t, int64_t)>();
  op.call(custom_ar_handle,
          input,
          output,
          registered_buffer,
          registered_buffer_size);
}

void op_dispose(int64_t custom_ar_handle) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::dispose", "")
                       .typed<void(int64_t)>();
  op.call(custom_ar_handle);
}

void op_free_shared_buffer(int64_t ptr) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::free_shared_buffer", "")
          .typed<void(int64_t)>();
  op.call(ptr);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
op_get_graph_buffer_ipc_meta(int64_t custom_ar_handle) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::get_graph_buffer_ipc_meta", "")
          .typed<std::tuple<std::vector<int64_t>, std::vector<int64_t>>(
              int64_t)>();
  return op.call(custom_ar_handle);
}

void op_register_graph_buffers(
    int64_t custom_ar_handle,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::register_graph_buffers", "")
          .typed<void(int64_t,
                      const std::vector<std::vector<int64_t>>&,
                      const std::vector<std::vector<int64_t>>&)>();
  op.call(custom_ar_handle, handles, offsets);
}

bool supported_world_size(int32_t world_size) {
  return world_size == 2 || world_size == 4 || world_size == 6 ||
         world_size == 8;
}

bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

bool is_weak_contiguous(const torch::Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return true;
  }
  const int64_t storage_nbytes =
      static_cast<int64_t>(tensor.storage().nbytes());
  return storage_nbytes - tensor.storage_offset() * tensor.element_size() ==
         tensor.numel() * tensor.element_size();
}

bool is_16_byte_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

}  // namespace

std::optional<torch::Tensor> musa_allreduce_custom(
    ProcessGroup* process_group,
    torch::Tensor& input,
    const std::optional<torch::Tensor>& output) {
  const char* disable_dsv4_custom_ar =
      std::getenv("XLLM_DSV4_DISABLE_CUSTOM_AR");
  if (disable_dsv4_custom_ar != nullptr &&
      std::string(disable_dsv4_custom_ar) == "1") {
    return std::nullopt;
  }
  ProcessGroupImpl* musa_process_group =
      dynamic_cast<ProcessGroupImpl*>(process_group);
  if (musa_process_group == nullptr) {
    return std::nullopt;
  }
  return musa_process_group->allreduce_custom(input, output);
}

MusaCustomAllReduce::MusaCustomAllReduce(int32_t rank,
                                         int32_t world_size,
                                         const std::string& group_name,
                                         const torch::Device& device,
                                         c10::intrusive_ptr<c10d::Store> store)
    : rank_(rank),
      world_size_(world_size),
      group_name_(group_name),
      device_(device),
      store_(std::move(store)) {
  if (group_name_ != "tp_group" && group_name_ != "world_group" &&
      group_name_ != "moe_tp_group") {
    return;
  }
  if (!supported_world_size(world_size_)) {
    LOG(WARNING) << "Custom all-reduce disabled for " << group_name_
                 << ": unsupported world size " << world_size_;
    return;
  }
  try {
    init();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Custom all-reduce disabled for " << group_name_ << ": "
                 << e.what();
    enabled_ = false;
  }
}

MusaCustomAllReduce::~MusaCustomAllReduce() {
  {
    std::lock_guard<std::mutex> lock(g_custom_allreduce_instances_mutex);
    g_custom_allreduce_instances.erase(this);
    g_long_prefill_generations.erase(this);
  }
  const c10::DeviceGuard device_guard(device_);
  if (custom_ar_handle_ != 0) {
    op_dispose(custom_ar_handle_);
    custom_ar_handle_ = 0;
  }
  if (!meta_ptrs_.empty() && rank_ < static_cast<int32_t>(meta_ptrs_.size()) &&
      meta_ptrs_[rank_] != 0) {
    op_free_shared_buffer(meta_ptrs_[rank_]);
  }
  if (!buffer_ptrs_.empty() &&
      rank_ < static_cast<int32_t>(buffer_ptrs_.size()) &&
      buffer_ptrs_[rank_] != 0) {
    op_free_shared_buffer(buffer_ptrs_[rank_]);
  }
}

void MusaCustomAllReduce::long_prefill_host_barrier() {
  CHECK(store_ != nullptr);
  CHECK_GE(rank_, 0);
  CHECK_LT(rank_, world_size_);
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(g_custom_allreduce_instances_mutex);
    generation = ++g_long_prefill_generations[this];
  }
  const std::string base = "xllm_long_prefill/" + group_name_ + "/";
  const std::string prefix = base + std::to_string(generation) + "/";
  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int32_t rank = 0; rank < world_size_; ++rank) {
    keys.emplace_back(prefix + std::to_string(rank));
  }
  store_->set(keys[rank_], std::vector<uint8_t>{1});
  store_->wait(keys, std::chrono::seconds(120));
  // Arrival in generation N proves every peer has finished waiting for N-1.
  // Keep only the current round; do not leak millions of keys in long runs.
  if (rank_ == 0 && generation > 1) {
    const std::string previous = base + std::to_string(generation - 1) + "/";
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      store_->deleteKey(previous + std::to_string(rank));
    }
  }
}

void musa_long_prefill_host_barrier(ProcessGroup* process_group) {
  auto* musa_group = dynamic_cast<ProcessGroupImpl*>(process_group);
  CHECK(musa_group != nullptr) << "Long-prefill rendezvous requires a MUSA group";
  musa_group->long_prefill_host_barrier();
}

void MusaCustomAllReduce::init() {
  const c10::DeviceGuard device_guard(device_);
  CHECK(store_ != nullptr) << "Custom all-reduce requires a c10d store.";

  const int64_t meta_bytes = op_meta_size();
  meta_ptrs_ = create_shared_buffer(meta_bytes + max_size_bytes_, "meta");
  buffer_ptrs_ = create_shared_buffer(max_size_bytes_, "buffer");

  rank_data_ =
      torch::empty({8 * 1024 * 1024},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device_));
  // Small TP reductions use the stable one-stage path. The world group keeps
  // the two-shot path used by the validated MUSA deployment.
  const int64_t algorithm = group_name_ == "tp_group" ? 1 : 2;
  custom_ar_handle_ =
      op_init_custom_ar(meta_ptrs_, rank_data_, rank_, fully_connected_);
  op_set_algorithm(custom_ar_handle_, algorithm);
  op_register_buffer(custom_ar_handle_, buffer_ptrs_);
  enabled_ = true;
  {
    std::lock_guard<std::mutex> lock(g_custom_allreduce_instances_mutex);
    g_custom_allreduce_instances.insert(this);
  }

  LOG(INFO) << "Custom all-reduce enabled for group " << group_name_
            << ", rank=" << rank_ << ", world_size=" << world_size_
            << ", algorithm=" << algorithm << ", max_bytes=" << max_size_bytes_;
}

std::vector<int64_t> MusaCustomAllReduce::create_shared_buffer(
    int64_t size_in_bytes,
    const std::string& name) {
  const c10::DeviceGuard device_guard(device_);
  auto [local_ptr, local_handle] =
      op_allocate_shared_buffer_and_handle(size_in_bytes);

  const std::string key_prefix =
      "xllm_custom_ar/" + group_name_ + "/" + name + "/";
  const std::string local_key = key_prefix + std::to_string(rank_);
  store_->set(local_key, tensor_to_bytes(local_handle));

  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int32_t i = 0; i < world_size_; ++i) {
    keys.push_back(key_prefix + std::to_string(i));
  }
  store_->wait(keys);

  std::vector<int64_t> ptrs(world_size_, 0);
  for (int32_t i = 0; i < world_size_; ++i) {
    if (i == rank_) {
      ptrs[i] = local_ptr;
      continue;
    }
    std::vector<uint8_t> bytes = store_->get(keys[i]);
    torch::Tensor handle_tensor = bytes_to_tensor(bytes);
    ptrs[i] = op_open_mem_handle(handle_tensor);
  }
  return ptrs;
}

std::vector<uint8_t> MusaCustomAllReduce::tensor_to_bytes(
    const torch::Tensor& tensor) const {
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
  std::vector<uint8_t> bytes(cpu_tensor.numel() * cpu_tensor.element_size());
  std::memcpy(bytes.data(), cpu_tensor.data_ptr<uint8_t>(), bytes.size());
  return bytes;
}

torch::Tensor MusaCustomAllReduce::bytes_to_tensor(
    const std::vector<uint8_t>& bytes) const {
  torch::Tensor tensor = torch::empty(
      {static_cast<int64_t>(bytes.size())},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(tensor.data_ptr<uint8_t>(), bytes.data(), bytes.size());
  return tensor;
}

bool MusaCustomAllReduce::should_use(const torch::Tensor& input) const {
  if (!enabled_ || !input.defined() || input.numel() == 0) {
    return false;
  }
  if (input.device() != device_) {
    return false;
  }
  if (!supported_dtype(input.scalar_type())) {
    return false;
  }
  if (!is_weak_contiguous(input) || !is_16_byte_aligned(input)) {
    return false;
  }
  const int64_t input_bytes = input.numel() * input.element_size();
  if (input_bytes % 16 != 0) {
    return false;
  }
  return input_bytes <= max_size_bytes_;
}

std::optional<torch::Tensor> MusaCustomAllReduce::allreduce(
    torch::Tensor& input,
    const std::optional<torch::Tensor>& output_workspace) {
  if (!should_use(input)) {
    return std::nullopt;
  }
  torch::Tensor output = output_workspace.has_value()
                             ? *output_workspace
                             : torch::empty_like(input);
  if (!output.defined() || output.sizes() != input.sizes() ||
      output.scalar_type() != input.scalar_type() ||
      output.device() != device_ || !is_weak_contiguous(output) ||
      !is_16_byte_aligned(output) || output.is_alias_of(input)) {
    return std::nullopt;
  }
  const char* graph_two_shot = std::getenv("XLLM_DSV4_GRAPH_TP_BF16_TWO_SHOT");
  const bool enable_graph_two_shot = group_name_ == "tp_group" &&
      graph_two_shot != nullptr && std::string(graph_two_shot) == "1";
  auto reduce = [&]() {
    op_all_reduce(custom_ar_handle_, input, output, buffer_ptrs_[rank_], max_size_bytes_);
  };
  if (!enable_graph_two_shot) {
    reduce();
    return output;
  }
  // Serialize selection + launch + restore, including small/eager TP calls,
  // so another host thread cannot observe the temporary capture algorithm.
  std::lock_guard<std::mutex> algorithm_lock(g_custom_allreduce_algorithm_mutex);
  bool capture_two_shot = false;
  if (input.dim() == 2 && musa_graph_tp_bf16_two_shot_shape(
          group_name_, world_size_, input.scalar_type() == torch::kBFloat16,
          input.size(0), input.size(1))) {
    musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
    CHECK_EQ(musaStreamIsCapturing(c10::musa::getCurrentMUSAStream(device_.index()),
                                   &capture_status), musaSuccess);
    capture_two_shot = capture_status == musaStreamCaptureStatusActive;
  }
  musa_with_tp_two_shot(capture_two_shot,
      [&](int64_t algorithm) { op_set_algorithm(custom_ar_handle_, algorithm); }, reduce);
  if (capture_two_shot && input.size(0) == 112) {
    LOG_FIRST_N(INFO, 1) << "[DSV4_GRAPH_TP_TWO_SHOT_B16] rows=112 width=4096 bf16=1 "
                           "capture_algorithm=2 restored_host_algorithm=1";
  }
  return output;
}

std::vector<uint8_t> MusaCustomAllReduce::serialize_graph_metadata(
    const std::vector<int64_t>& handles,
    const std::vector<int64_t>& offsets) const {
  const uint64_t handle_count = handles.size();
  const uint64_t offset_count = offsets.size();
  const uint64_t header_bytes = 2 * sizeof(uint64_t);
  CHECK_LE(handle_count, std::numeric_limits<uint64_t>::max() - offset_count);
  const uint64_t payload_count = handle_count + offset_count;
  CHECK_LE(
      payload_count,
      (std::numeric_limits<size_t>::max() - header_bytes) / sizeof(int64_t));
  std::vector<uint8_t> bytes(
      static_cast<size_t>(header_bytes + payload_count * sizeof(int64_t)));
  std::memcpy(bytes.data(), &handle_count, sizeof(handle_count));
  std::memcpy(
      bytes.data() + sizeof(handle_count), &offset_count, sizeof(offset_count));
  size_t cursor = static_cast<size_t>(header_bytes);
  if (!handles.empty()) {
    std::memcpy(bytes.data() + cursor,
                handles.data(),
                handles.size() * sizeof(int64_t));
    cursor += handles.size() * sizeof(int64_t);
  }
  if (!offsets.empty()) {
    std::memcpy(bytes.data() + cursor,
                offsets.data(),
                offsets.size() * sizeof(int64_t));
  }
  return bytes;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
MusaCustomAllReduce::deserialize_graph_metadata(
    const std::vector<uint8_t>& bytes) const {
  CHECK_GE(bytes.size(), 2 * sizeof(uint64_t));
  uint64_t handle_count = 0;
  uint64_t offset_count = 0;
  std::memcpy(&handle_count, bytes.data(), sizeof(handle_count));
  std::memcpy(
      &offset_count, bytes.data() + sizeof(handle_count), sizeof(offset_count));
  CHECK_LE(handle_count, std::numeric_limits<uint64_t>::max() - offset_count);
  const uint64_t payload_count = handle_count + offset_count;
  const uint64_t expected_size =
      2 * sizeof(uint64_t) + payload_count * sizeof(int64_t);
  CHECK_EQ(bytes.size(), expected_size);
  CHECK_LE(handle_count,
           static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
  CHECK_LE(offset_count,
           static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  std::vector<int64_t> handles(static_cast<size_t>(handle_count));
  std::vector<int64_t> offsets(static_cast<size_t>(offset_count));
  size_t cursor = 2 * sizeof(uint64_t);
  if (!handles.empty()) {
    std::memcpy(handles.data(),
                bytes.data() + cursor,
                handles.size() * sizeof(int64_t));
    cursor += handles.size() * sizeof(int64_t);
  }
  if (!offsets.empty()) {
    std::memcpy(offsets.data(),
                bytes.data() + cursor,
                offsets.size() * sizeof(int64_t));
  }
  return std::make_tuple(std::move(handles), std::move(offsets));
}

void MusaCustomAllReduce::finalize_graph_buffers() {
  if (!enabled_ || custom_ar_handle_ == 0 || store_ == nullptr) {
    return;
  }

  const c10::DeviceGuard device_guard(device_);
  auto [local_handles, local_offsets] =
      op_get_graph_buffer_ipc_meta(custom_ar_handle_);
  const std::string key_prefix = "xllm_custom_ar/" + group_name_ + "/graph/" +
                                 std::to_string(graph_capture_generation_) +
                                 "/";
  const std::string local_key = key_prefix + std::to_string(rank_);
  store_->set(local_key,
              serialize_graph_metadata(local_handles, local_offsets));

  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int32_t i = 0; i < world_size_; ++i) {
    keys.push_back(key_prefix + std::to_string(i));
  }
  store_->wait(keys);

  std::vector<std::vector<int64_t>> all_handles(world_size_);
  std::vector<std::vector<int64_t>> all_offsets(world_size_);
  for (int32_t i = 0; i < world_size_; ++i) {
    auto [handles, offsets] = deserialize_graph_metadata(store_->get(keys[i]));
    all_handles[i] = std::move(handles);
    all_offsets[i] = std::move(offsets);
  }

  const size_t expected_handle_count = all_handles[rank_].size();
  const size_t expected_offset_count = all_offsets[rank_].size();
  for (int32_t i = 0; i < world_size_; ++i) {
    CHECK_EQ(all_handles[i].size(), expected_handle_count)
        << "Graph custom-allreduce handle count mismatch for rank " << i;
    CHECK_EQ(all_offsets[i].size(), expected_offset_count)
        << "Graph custom-allreduce offset count mismatch for rank " << i;
  }
  CHECK_EQ(expected_offset_count == 0, expected_handle_count == 0)
      << "Graph custom-allreduce metadata has an invalid handle payload";
  if (expected_offset_count != 0) {
    CHECK_EQ(expected_handle_count % expected_offset_count, 0)
        << "Graph custom-allreduce IPC handle payload is not divisible by the "
           "number of captured buffers";
    op_register_graph_buffers(custom_ar_handle_, all_handles, all_offsets);
    LOG(INFO) << "Registered " << expected_offset_count
              << " custom-allreduce graph buffers for group " << group_name_
              << ", generation=" << graph_capture_generation_;
  }
  ++graph_capture_generation_;
}

void finalize_musa_custom_allreduce_graphs() {
  std::vector<MusaCustomAllReduce*> instances;
  {
    std::lock_guard<std::mutex> lock(g_custom_allreduce_instances_mutex);
    instances.assign(g_custom_allreduce_instances.begin(),
                     g_custom_allreduce_instances.end());
  }
  std::sort(
      instances.begin(), instances.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->group_name_ < rhs->group_name_;
      });
  for (auto* instance : instances) {
    instance->finalize_graph_buffers();
  }
}

}  // namespace xllm
