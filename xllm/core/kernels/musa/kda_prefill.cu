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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <musa_runtime.h>
#include <mutlass/device_kernel.h>
#include <mutlass/half.h>

#include <cstdint>
#include <limits>
#include <mutlass/kernel_hardware_info.hpp>
#include <type_traits>

#include "core/kernels/musa/kda_ops.h"
#include "core/kernels/musa/kda_validation.h"
#include "mate/flat/kda/chunk_kda_prepare.hpp"
#include "mate/flat/kda/chunk_kda_recurrence.hpp"

namespace xllm::kernel::musa {
namespace {

namespace kda = mate::flat::kda;
using Element = mutlass::bfloat16_t;
using LogicalShape = kda::ChunkKdaLogicalShape<16, 128, 128>;
using TileShape = mute::Shape<mute::Int<16>, mute::Int<16>, mute::Int<128>>;
template <mate::flat::Tag TagValue, bool Enabled>
using BoolOption = mate::flat::Option<TagValue, std::bool_constant<Enabled>>;
using Prepare = typename kda::ChunkKdaPrepareBuilder<
    mutlass::arch::Mp31,
    Element,
    int64_t,
    LogicalShape,
    TileShape,
    kda::ChunkKdaQkStride,
    kda::ChunkKdaQkStride,
    kda::ChunkKdaQkStride,
    BoolOption<mate::flat::Tag::HasGateParams, true>,
    BoolOption<mate::flat::Tag::IsVarlen, false>,
    BoolOption<mate::flat::Tag::NormalizeQK, true>>::Kernel;
using Recurrence = typename kda::ChunkKdaRecurrenceBuilder<
    mutlass::arch::Mp31,
    Element,
    float,
    int64_t,
    LogicalShape,
    TileShape,
    kda::ChunkKdaVoStride,
    kda::ChunkKdaVoStride,
    BoolOption<mate::flat::Tag::HasStateIn, true>,
    BoolOption<mate::flat::Tag::HasStateOut, true>,
    BoolOption<mate::flat::Tag::HasChunkStatesOut, false>,
    BoolOption<mate::flat::Tag::HasCheckpointCadence, false>,
    mate::flat::Option<mate::flat::Tag::CheckpointEveryNTokens,
                       std::integral_constant<int32_t, 0>>,
    BoolOption<mate::flat::Tag::IsVarlen, false>>::Kernel;
using Workspace = typename Prepare::Components::Workspace;
static_assert(Workspace::RecordStride == 6784);

kda::ChunkKdaQkStride qk_stride(const torch::Tensor& tensor) {
  return mute::make_stride(
      tensor.stride(1),
      mute::_1{},
      mute::make_stride(tensor.stride(2), tensor.stride(0)));
}

kda::ChunkKdaVoStride vo_stride(const torch::Tensor& tensor) {
  return mute::make_stride(
      mute::_1{},
      tensor.stride(1),
      mute::make_stride(tensor.stride(2), tensor.stride(0)));
}

}  // namespace

KdaWorkspace::KdaWorkspace(int64_t max_batch,
                           int64_t max_tokens,
                           int64_t heads,
                           const torch::Device& device)
    : max_batch_(max_batch), max_tokens_(max_tokens), heads_(heads) {
  CHECK_GT(max_batch, 0);
  CHECK_GT(max_tokens, 0);
  CHECK_GT(heads, 0);
  CHECK_LE(max_batch, std::numeric_limits<int32_t>::max());
  CHECK_LE(max_tokens, std::numeric_limits<int32_t>::max());
  CHECK_LE(heads, std::numeric_limits<int32_t>::max());
  CHECK(device.is_privateuseone());
  const auto options = torch::TensorOptions().device(device);
  storage_ = torch::empty(
      {max_batch * ((max_tokens + 15) / 16), heads, Workspace::RecordStride},
      options.dtype(torch::kBFloat16));
  state_indices_ = torch::arange(max_batch, options.dtype(torch::kInt32));
}

void kda_prefill(const KdaInputs& inputs,
                 torch::Tensor& output,
                 torch::Tensor& final_state,
                 KdaWorkspace& workspace) {
  detail::validate_kda(inputs, output, final_state);
  CHECK_LE(inputs.query.size(0), workspace.max_batch());
  CHECK_LE(inputs.query.size(1), workspace.max_tokens());
  CHECK_EQ(inputs.query.size(2), workspace.heads());
  CHECK(workspace.storage().device() == inputs.query.device());
  const c10::cuda::CUDAGuard guard(inputs.query.device());
  const int32_t batch = static_cast<int32_t>(inputs.query.size(0));
  const int32_t tokens = static_cast<int32_t>(inputs.query.size(1));
  const int32_t heads = static_cast<int32_t>(inputs.query.size(2));
  const int32_t chunks = tokens / 16 + static_cast<int32_t>(tokens % 16 != 0);
  const int32_t mp_count =
      mutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          inputs.query.device().index());
  const musaStream_t stream = at::cuda::getCurrentCUDAStream();
  Element* scratch = static_cast<Element*>(workspace.storage().data_ptr());
  typename Workspace::MutablePointers fields{
      scratch,
      scratch + Workspace::KRestoredOffset,
      scratch + Workspace::InverseOffset,
      scratch + Workspace::POffset,
      scratch + Workspace::TotalOffsetBytes / sizeof(Element)};
  typename Prepare::ProblemSize problem{
      typename Prepare::Components::ProblemShape{
          batch, tokens, heads, heads, batch, nullptr, chunks}};
  problem.num_partitions =
      Prepare::Components::prepare_partition_count(problem, mp_count);
  problem.metadata = nullptr;
  typename Prepare::Arguments prepare_arguments{
      problem,
      typename Prepare::CollectivePrepare::Arguments{
          static_cast<const Element*>(inputs.query.data_ptr()),
          static_cast<const Element*>(inputs.key.data_ptr()),
          static_cast<const Element*>(inputs.raw_forget.data_ptr()),
          static_cast<const Element*>(inputs.raw_beta.data_ptr()),
          inputs.a_log.data_ptr<float>(),
          inputs.dt_bias.data_ptr<float>(),
          qk_stride(inputs.query),
          qk_stride(inputs.key),
          qk_stride(inputs.raw_forget),
          inputs.options.scale,
          inputs.options.gate_lower_bound,
          fields}};
  typename Prepare::Params prepare_params =
      Prepare::to_underlying_arguments(prepare_arguments);
  C10_CUDA_CHECK(
      musaFuncSetAttribute(mutlass::device_kernel<Prepare>,
                           musaFuncAttributeMaxDynamicSharedMemorySize,
                           Prepare::SharedStorageSize));
  mutlass::device_kernel<Prepare><<<Prepare::TileScheduler::get_grid_shape(
                                        prepare_params.scheduler, mp_count),
                                    Prepare::MaxThreadsPerBlock,
                                    Prepare::SharedStorageSize,
                                    stream>>>(prepare_params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Recurrence::Arguments recurrence_arguments{
      typename Recurrence::ProblemSize{
          batch, tokens, heads, heads, batch, nullptr, chunks},
      typename Recurrence::CollectiveRecurrence::Arguments{
          static_cast<const Element*>(inputs.value.data_ptr()),
          inputs.initial_state.data_ptr<float>(),
          static_cast<Element*>(output.data_ptr()),
          final_state.data_ptr<float>(),
          nullptr,
          workspace.state_indices().data_ptr<int32_t>(),
          vo_stride(inputs.value),
          vo_stride(output),
          typename Workspace::ConstPointers{fields.decayed,
                                            fields.k_restored,
                                            fields.inverse,
                                            fields.p,
                                            fields.total},
          0}};
  typename Recurrence::Params recurrence_params =
      Recurrence::to_underlying_arguments(recurrence_arguments);
  C10_CUDA_CHECK(
      musaFuncSetAttribute(mutlass::device_kernel<Recurrence>,
                           musaFuncAttributeMaxDynamicSharedMemorySize,
                           Recurrence::SharedStorageSize));
  mutlass::device_kernel<Recurrence>
      <<<Recurrence::get_grid_shape(recurrence_params),
         Recurrence::MaxThreadsPerBlock,
         Recurrence::SharedStorageSize,
         stream>>>(recurrence_params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::musa
