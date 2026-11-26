/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/layers/musa/musa_mega_moe.h"

#include <c10/musa/MUSAGuard.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <unordered_map>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::layer {
namespace {
namespace km = kernel::musa;
namespace ffi = tvm::ffi;
constexpr int64_t kRanks = 8;
// CP8 keeps roughly one eighth of a long prefill on each rank.  The previous
// 512-row schedule forced the EP path to use host-side all-to-all for larger
// local shards.  The c4096 MATE schedule handles the complete CP-local shard
// in one dispatch/combine round and keeps routing and grouped GEMM on device.
constexpr int64_t kCapacity = 4096;
constexpr int64_t kExperts = 288;
constexpr int64_t kTopk = 8;
constexpr int64_t kHidden = 4096;
constexpr int64_t kIntermediate = 2048;
// Worst-case assignment count plus per-expert M256 rounding. No assumption
// about balanced routing may reduce this allocation.
constexpr int64_t kPoolRows =
    ((kRanks * kCapacity * kTopk + (kExperts / kRanks) * 255 + 255) / 256) *
    256;

ffi::Function runtime_function(const std::string& name) {
  return km::get_function("mate_mega_moe_runtime_utils_layout3", name);
}

torch::Tensor retain_tensor(ffi::Tensor tensor,
                            torch::ScalarType dtype,
                            const torch::Device& device) {
  const std::vector<int64_t> shape(tensor.shape().begin(),
                                   tensor.shape().end());
  void* pointer = tensor.data_ptr();
  return torch::from_blob(
      pointer,
      shape,
      [owner = std::move(tensor)](void* /*unused*/) {},
      torch::TensorOptions().device(device).dtype(dtype));
}
}  // namespace

std::shared_ptr<MusaMegaMoe> MusaMegaMoe::acquire(ProcessGroup* group) {
  // A worker executes layers in order on one stream, including graph replay.
  // Weak ownership releases IPC mappings with the model rather than leaking
  // one workspace per layer or holding a stale ProcessGroup across reloads.
  static thread_local std::unordered_map<ProcessGroup*,
                                         std::weak_ptr<MusaMegaMoe>>
      workspaces;
  auto& slot = workspaces[group];
  auto workspace = slot.lock();
  if (!workspace) {
    workspace = std::make_shared<MusaMegaMoe>(group);
    slot = workspace;
  }
  return workspace;
}

MusaMegaMoe::MusaMegaMoe(ProcessGroup* group) : group_(group) {
  CHECK(group_ != nullptr);
  CHECK_EQ(group_->world_size(), kRanks);
  c10::musa::MUSAGuard guard(group_->device());
  CHECK(!km::is_stream_capturing());
  auto layout =
      runtime_function("mega_moe_unified_get_buffer_layout")(kRanks,
                                                             kExperts,
                                                             kCapacity,
                                                             kTopk,
                                                             kHidden,
                                                             kIntermediate,
                                                             kPoolRows,
                                                             int64_t{0})
          .cast<ffi::Array<ffi::Any>>();
  const int64_t bytes = layout[0].cast<int64_t>();
  auto allocator = runtime_function("mega_moe_alloc_buffer");
  auto storage =
      allocator(bytes, static_cast<int32_t>(group_->device().index()))
          .cast<ffi::Tensor>();
  auto slices =
      layout[1].cast<ffi::Function>()(storage).cast<ffi::Array<ffi::Tensor>>();
  buffer_ = retain_tensor(storage, torch::kInt8, group_->device());
  local_output_ = torch::empty(
      {kCapacity, kHidden},
      torch::TensorOptions().device(group_->device()).dtype(torch::kBFloat16));
  const std::vector<torch::ScalarType> types = {torch::kFloat8_e4m3fn,
                                                torch::kFloat32,
                                                torch::kInt64,
                                                torch::kFloat32,
                                                torch::kFloat8_e4m3fn,
                                                torch::kFloat32,
                                                torch::kFloat32,
                                                torch::kFloat8_e4m3fn,
                                                torch::kFloat32,
                                                torch::kBFloat16,
                                                torch::kFloat8_e4m3fn,
                                                torch::kFloat32,
                                                torch::kFloat8_e4m3fn,
                                                torch::kFloat32,
                                                torch::kBFloat16};
  CHECK_EQ(slices.size(), types.size());
  views_.reserve(types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    views_.emplace_back(retain_tensor(slices[i], types[i], group_->device()));
  }
  const int64_t handle_size =
      runtime_function("mega_moe_get_ipc_handle_size")().cast<int64_t>();
  auto handle = torch::empty({handle_size}, torch::kUInt8);
  runtime_function("mega_moe_get_ipc_handle")(storage,
                                              km::to_ffi_tensor(handle));
  auto handles = group_->allgather_base_sync(handle.to(group_->device())).cpu();
  handles = handles.reshape({kRanks, handle_size});
  pointers_ = torch::empty({kRanks}, torch::kInt64);
  runtime_function("mega_moe_open_ipc_handles")(km::to_ffi_tensor(handles),
                                                storage,
                                                group_->rank(),
                                                km::to_ffi_tensor(pointers_));
  buffer_.zero_();
  auto barrier = torch::zeros({1}, buffer_.options().dtype(torch::kFloat32));
  group_->allreduce(barrier);
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  LOG(INFO) << "GLM MegaMoE EP8 workspace=" << bytes
            << " bytes, capacity=" << kCapacity
            << ", complete experts per rank=36, token ownership=partitioned";
}

MusaMegaMoe::~MusaMegaMoe() {
  c10::musa::MUSAGuard guard(group_->device());
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  auto barrier = torch::zeros({1}, buffer_.options().dtype(torch::kFloat32));
  group_->allreduce(barrier);
  CHECK_EQ(musaDeviceSynchronize(), musaSuccess);
  runtime_function("mega_moe_close_ipc_handles")(km::to_ffi_tensor(pointers_),
                                                 group_->rank());
}

std::pair<torch::Tensor, torch::Tensor> MusaMegaMoe::transform_gate_up(
    const torch::Tensor& weight,
    const torch::Tensor& scale) {
  CHECK_EQ(weight.sizes(), torch::IntArrayRef({36, 4096, 4096}));
  CHECK_EQ(scale.sizes(), torch::IntArrayRef({36, 32, 32}));
  auto interleaved = weight.view({36, 2, 32, 64, 4096})
                         .transpose(1, 2)
                         .contiguous()
                         .view_as(weight);
  auto expanded_scales = scale.view({36, 2, 16, 32})
                             .repeat_interleave(2, 2)
                             .transpose(1, 2)
                             .contiguous()
                             .view({36, 64, 32});
  return {interleaved, expanded_scales};
}

torch::Tensor MusaMegaMoe::forward_local_shard(
    const torch::Tensor& input,
    const torch::Tensor& ids,
    const torch::Tensor& weights,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    double clamp,
    int64_t max_rank_rows,
    bool is_prefill,
    const torch::Tensor& output_buffer) {
  const int64_t rows = input.size(0);
  CHECK_GE(max_rank_rows, rows);
  auto output = output_buffer.defined() ? output_buffer.narrow(0, 0, rows)
                                        : torch::empty_like(input);
  // Every rank must enter the same IPC rounds. Split proportionally so a
  // one-row CP imbalance also stays balanced at a capacity boundary.
  const int64_t rounds =
      std::max<int64_t>(1, (max_rank_rows + kCapacity - 1) / kCapacity);
  for (int64_t round = 0; round < rounds; ++round) {
    const int64_t begin = round * rows / rounds;
    const int64_t count = (round + 1) * rows / rounds - begin;
    forward(input.narrow(0, begin, count),
            ids.narrow(0, begin, count),
            weights.narrow(0, begin, count),
            w13,
            s13,
            w2,
            s2,
            clamp,
            output.narrow(0, begin, count),
            /*partition_replicated_input=*/false,
            /*force_prefill=*/is_prefill);
  }
  return output;
}

torch::Tensor MusaMegaMoe::forward(const torch::Tensor& input,
                                   const torch::Tensor& ids,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& w13,
                                   const torch::Tensor& s13,
                                   const torch::Tensor& w2,
                                   const torch::Tensor& s2,
                                   double clamp,
                                   const torch::Tensor& output_buffer,
                                   bool partition_replicated_input,
                                   bool force_prefill) {
  CHECK_EQ(input.size(1), kHidden);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.scalar_type(), torch::kFloat32);
  const int64_t tokens = input.size(0);
  const int64_t local_rows =
      partition_replicated_input ? (tokens + kRanks - 1) / kRanks : tokens;
  CHECK_LE(local_rows, kCapacity);
  const int64_t start =
      partition_replicated_input
          ? std::min<int64_t>(tokens, group_->rank() * local_rows)
          : 0;
  const int64_t count = partition_replicated_input
                            ? std::min(local_rows, tokens - start)
                            : tokens;
  // Invalid rows carry no routes. In particular C1 is owned by rank zero,
  // not dispatched eight times from replicated attention outputs.
  if (count < local_rows) {
    views_[2].narrow(0, count, local_rows - count).fill_(-1);
    views_[3].narrow(0, count, local_rows - count).zero_();
  }
  if (count > 0) {
    CHECK(km::deepseek_v4_mega_moe_pre_dispatch(input.narrow(0, start, count),
                                                ids.narrow(0, start, count),
                                                weights.narrow(0, start, count),
                                                views_[0],
                                                views_[1],
                                                views_[2],
                                                views_[3]));
  }
  // Prefill exchanges per-rank counts and accepts unequal positive row counts.
  // Do not feed artificial -1 routes to its persistent dispatch schedule.
  // Decode still needs a sentinel row on ranks with no owned tokens.
  const bool use_prefill = force_prefill || local_rows > 128;
  const int64_t execution_rows = use_prefill ? count : local_rows;
  auto output =
      output_buffer.defined() ? output_buffer : torch::empty_like(input);
  CHECK_EQ(output.sizes(), input.sizes());
  CHECK_EQ(output.scalar_type(), input.scalar_type());
  CHECK_EQ(output.device(), input.device());
  CHECK(output.is_contiguous());
  // Stage 2 can write its owned rows directly to the TP reduction input.
  // A partially padded shard still needs scratch for its sentinel rows.
  torch::Tensor local_output = output;
  const bool padded_shard = count < execution_rows;
  if (partition_replicated_input) {
    if (count < tokens) {
      output.zero_();
    }
    local_output = padded_shard ? local_output_.narrow(0, 0, execution_rows)
                                : output.narrow(0, start, count);
  }
  {
    km::TvmffiStreamGuard stream_guard(input.device());
    if (!use_prefill) {
      km::get_function("glm53_mega_c4096_v2_decode1",
                       "glm53_mega_c4096_v2_decode1")(
          km::to_ffi_tensor(views_[4]),
          km::to_ffi_tensor(views_[5]),
          km::to_ffi_tensor(views_[7]),
          km::to_ffi_tensor(views_[8]),
          km::to_ffi_tensor(w13),
          km::to_ffi_tensor(s13),
          km::to_ffi_tensor(pointers_),
          group_->rank(),
          execution_rows,
          clamp,
          1.0,
          -1.0);
      km::get_function("glm53_mega_c4096_v2_decode2",
                       "glm53_mega_c4096_v2_decode2")(
          km::to_ffi_tensor(views_[7]),
          km::to_ffi_tensor(views_[8]),
          km::to_ffi_tensor(w2),
          km::to_ffi_tensor(s2),
          km::to_ffi_tensor(local_output),
          km::to_ffi_tensor(pointers_),
          group_->rank(),
          execution_rows);
    } else {
      km::get_function("glm53_mega_c4096_v2_stage1",
                       "glm53_mega_c4096_v2_stage1")(
          km::to_ffi_tensor(views_[10]),
          km::to_ffi_tensor(views_[11]),
          km::to_ffi_tensor(views_[12]),
          km::to_ffi_tensor(views_[13]),
          km::to_ffi_tensor(w13),
          km::to_ffi_tensor(s13),
          km::to_ffi_tensor(pointers_),
          group_->rank(),
          execution_rows,
          clamp,
          1.0,
          -1.0);
      km::get_function("glm53_mega_c4096_v2_stage2",
                       "glm53_mega_c4096_v2_stage2")(
          km::to_ffi_tensor(views_[12]),
          km::to_ffi_tensor(views_[13]),
          km::to_ffi_tensor(w2),
          km::to_ffi_tensor(s2),
          km::to_ffi_tensor(pointers_),
          group_->rank());
      km::get_function("glm53_mega_c4096_v2_stage3",
                       "glm53_mega_c4096_v2_stage3")(
          km::to_ffi_tensor(local_output),
          km::to_ffi_tensor(pointers_),
          group_->rank(),
          execution_rows);
    }
  }
  // Decode custom all-reduce exports its input through legacy MUSA IPC.
  // Use the caller's persistent allocation, outside the graph VMM pool.
  if (partition_replicated_input && padded_shard && count > 0) {
    output.narrow(0, start, count).copy_(local_output.narrow(0, 0, count));
  }
  // The caller adds its TP-sharded shared expert, then one TP all-reduce
  // assembles the disjoint routed rows and sums the shared expert shards.
  return output;
}
}  // namespace xllm::layer
