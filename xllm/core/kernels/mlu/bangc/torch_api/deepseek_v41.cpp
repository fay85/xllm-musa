/* Copyright 2026 The xLLM Authors.

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

#include "kernels/mlu/bangc/include/deepseek_v41.h"

#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/deepseek_v41/compress_params.h"
#include "kernels/mlu/bangc/src/deepseek_v41/indexer_k_params.h"
#include "kernels/mlu/bangc/src/deepseek_v41/launch.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"
namespace xllm::kernel::mlu::bangc {
namespace {
using OptTensor = std::optional<torch::Tensor>;

constexpr int64_t kHeadDim = 512;
constexpr int64_t kPageSize = 128;
constexpr int64_t kStateRows = 32;
constexpr int64_t kRopeColumns = 128;
// Protect the kernel int32 block-count ceiling division from overflow.
constexpr int64_t kMaxIndexerQRows =
    std::numeric_limits<int32_t>::max() - (indexer_q::kRowsPerBlock - 1);

bool present(const OptTensor& tensor) {
  return tensor.has_value() && tensor->defined();
}

void check_rows(const torch::Tensor& tensor,
                const torch::Device& device,
                int64_t cols,
                torch::ScalarType dtype) {
  CHECK(tensor.device() == device);
  CHECK_EQ(tensor.dim(), 2);
  CHECK_EQ(tensor.size(1), cols);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK_EQ(tensor.stride(1), 1);
  CHECK(tensor.size(0) <= 1 || tensor.stride(0) >= cols);
}

void check_contiguous(const torch::Tensor& tensor,
                      const torch::Device& device,
                      torch::IntArrayRef shape,
                      torch::ScalarType dtype) {
  CHECK(tensor.device() == device);
  CHECK(tensor.sizes() == shape);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.is_contiguous());
}

const int32_t* valid_row_count_ptr(const OptTensor& num_valid_rows,
                                   const torch::Device& device) {
  if (!present(num_valid_rows)) {
    return nullptr;
  }
  check_contiguous(*num_valid_rows, device, {1}, torch::kInt32);
  return num_valid_rows->data_ptr<int32_t>();
}

void check_device(const torch::Tensor& tensor,
                  const torch::Device& device,
                  const char* name) {
  CHECK(tensor.device() == device)
      << name << " must be on the same MLU device as raw";
}

void check_int_vector(const torch::Tensor& tensor,
                      const torch::Device& device,
                      int64_t size,
                      const char* name) {
  check_device(tensor, device, name);
  CHECK(tensor.scalar_type() == torch::kInt) << name << " must be int32";
  CHECK(tensor.dim() == 1 && tensor.size(0) == size)
      << name << " must have shape [" << size << "]";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

void check_table(const torch::Tensor& tensor,
                 const torch::Device& device,
                 int64_t batch,
                 const char* name) {
  check_device(tensor, device, name);
  CHECK(tensor.scalar_type() == torch::kInt) << name << " must be int32";
  CHECK(tensor.dim() == 2 && tensor.size(0) == batch)
      << name << " must have shape [B, C]";
  CHECK(tensor.size(1) <= 1 || tensor.stride(1) == 1)
      << name << " columns must be contiguous";
}

void check_rows(const torch::Tensor& tensor,
                const torch::Device& device,
                torch::ScalarType dtype,
                const char* name) {
  check_device(tensor, device, name);
  CHECK(tensor.scalar_type() == dtype) << name << " has an invalid dtype";
  CHECK(tensor.dim() == 2 && tensor.size(1) == kHeadDim)
      << name << " must have shape [N, 512]";
  CHECK(tensor.stride(1) == 1) << name << " last dimension must be dense";
  CHECK(tensor.size(0) <= 1 || tensor.stride(0) >= kHeadDim)
      << name << " rows must not overlap";
  // Two-row strided DMA source strides are 32-bit byte counts.
  CHECK(tensor.stride(0) * 2 * static_cast<int64_t>(tensor.element_size()) <
        std::numeric_limits<int32_t>::max())
      << name << " row stride is too large";
}

void check_state(const torch::Tensor& tensor,
                 const torch::Device& device,
                 const char* name) {
  check_device(tensor, device, name);
  CHECK(tensor.scalar_type() == torch::kFloat) << name << " must be float32";
  CHECK(tensor.dim() == 3 && tensor.size(1) == kStateRows &&
        tensor.size(2) == kHeadDim)
      << name << " must have shape [Ns, 32, 512]";
  CHECK(tensor.stride(2) == 1) << name << " last dimension must be dense";
}

}  // namespace

torch::Tensor fused_indexer_q(const torch::Tensor& qr,
                              const torch::Tensor& wq,
                              const torch::Tensor& positions,
                              const torch::Tensor& rope) {
  const torch::Device device = qr.device();
  CHECK(device.is_privateuseone());
  CHECK_EQ(qr.dim(), 2);
  CHECK_EQ(qr.size(1), 1280);
  CHECK_GT(qr.size(0), 0);
  CHECK_LE(qr.size(0), kMaxIndexerQRows);
  check_contiguous(qr, device, {qr.size(0), 1280}, torch::kBFloat16);
  check_contiguous(wq, device, {4096, 1280}, torch::kBFloat16);
  check_contiguous(positions, device, {qr.size(0)}, torch::kInt32);
  CHECK_EQ(rope.dim(), 2);
  CHECK_GT(rope.size(0), 0);
  CHECK_LE(rope.size(0), 1048576);
  check_contiguous(rope, device, {rope.size(0), 128}, torch::kFloat32);
  const torch::DeviceGuard guard(device);
  torch::Tensor output = torch::empty({qr.size(0), 32, 128}, qr.options());
  CHECK_EQ(indexer_q::launch(torch_mlu::getCurMLUStream(device.index()),
                             qr.data_ptr(),
                             wq.data_ptr(),
                             positions.data_ptr<int32_t>(),
                             rope.data_ptr<float>(),
                             output.data_ptr(),
                             static_cast<int32_t>(qr.size(0))),
           0);
  return output;
}

torch::Tensor fused_indexer_k(const OptTensor& x,
                              const OptTensor& latent,
                              const OptTensor& weights_proj,
                              const OptTensor& wk,
                              const OptTensor& gamma,
                              const OptTensor& rope,
                              const OptTensor& positions,
                              const OptTensor& slots,
                              torch::Tensor cache,
                              torch::Tensor head,
                              const OptTensor& num_valid_input_rows,
                              const OptTensor& num_valid_compressed_rows) {
  const torch::Device device = cache.device();
  CHECK(device.is_privateuseone());
  CHECK_EQ(cache.dim(), 4);
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK(cache.is_contiguous());
  CHECK_EQ(cache.size(3), indexer_k::kHeadDim);
  CHECK((cache.size(1) == 1 && cache.size(2) == indexer_k::kBlockSize) ||
        (cache.size(2) == 1 && cache.size(1) == indexer_k::kBlockSize));
  if (present(x)) {
    check_rows(*x, device, indexer_k::kHidden, torch::kBFloat16);
  }
  if (present(latent)) {
    check_rows(*latent, device, indexer_k::kLatent, torch::kBFloat16);
  }
  const int64_t cap_x = present(x) ? x->size(0) : 0;
  const int64_t cap_k = present(latent) ? latent->size(0) : 0;
  CHECK_LT(cap_x, int64_t{1} << 31);
  CHECK_LT(cap_k, int64_t{1} << 31);
  check_contiguous(head, device, {cap_x, indexer_k::kHeads}, torch::kFloat32);
  indexer_k::Params params{};
  const int32_t* num_valid_input_rows_ptr =
      valid_row_count_ptr(num_valid_input_rows, device);
  const int32_t* num_valid_compressed_rows_ptr =
      valid_row_count_ptr(num_valid_compressed_rows, device);
  if (cap_x > 0) {
    CHECK(present(weights_proj));
    check_contiguous(*weights_proj,
                     device,
                     {indexer_k::kHeads, indexer_k::kHidden},
                     torch::kBFloat16);
    params.x = x->data_ptr();
    params.x_stride = x->stride(0);
    params.wp = weights_proj->data_ptr();
    params.head = head.data_ptr<float>();
    params.num_valid_input_rows = num_valid_input_rows_ptr;
    params.cap_x = static_cast<int32_t>(cap_x);
  }
  if (cap_k > 0) {
    CHECK(present(wk) && present(gamma) && present(rope) &&
          present(positions) && present(slots));
    check_contiguous(*wk,
                     device,
                     {indexer_k::kHeadDim, indexer_k::kLatent},
                     torch::kBFloat16);
    CHECK(gamma->device() == device);
    CHECK(gamma->sizes() == torch::IntArrayRef({indexer_k::kHeadDim}));
    CHECK(gamma->is_contiguous());
    CHECK(gamma->scalar_type() == torch::kBFloat16 ||
          gamma->scalar_type() == torch::kFloat32);
    check_rows(*rope, device, 2 * indexer_k::kRopeDim, torch::kFloat32);
    check_contiguous(*positions, device, {cap_k}, torch::kInt32);
    check_contiguous(*slots, device, {cap_k}, torch::kInt32);
    params.latent = latent->data_ptr();
    params.latent_stride = latent->stride(0);
    params.wk = wk->data_ptr();
    params.gamma = gamma->data_ptr();
    params.gamma_fp32 = gamma->scalar_type() == torch::kFloat32;
    params.rope = rope->data_ptr<float>();
    params.rope_stride = rope->stride(0);
    params.rope_rows = rope->size(0);
    params.positions = positions->data_ptr<int32_t>();
    params.slots = slots->data_ptr<int32_t>();
    params.cache = cache.data_ptr();
    params.cache_rows = cache.numel() / indexer_k::kHeadDim;
    params.num_valid_compressed_rows = num_valid_compressed_rows_ptr;
    params.cap_k = static_cast<int32_t>(cap_k);
    const int64_t rope_bytes = params.rope_rows * params.rope_stride *
                               static_cast<int64_t>(sizeof(float));
    const int64_t cache_bytes = cache.numel() * cache.element_size();
    params.meta_u32 = rope_bytes < (int64_t{1} << 32) &&
                      cache_bytes < (int64_t{1} << 32) &&
                      params.rope_rows < (int64_t{1} << 31) &&
                      params.cache_rows < (int64_t{1} << 31);
  }
  if (cap_x != 0 || cap_k != 0) {
    const torch::DeviceGuard guard(device);
    CHECK_EQ(indexer_k::launch(torch_mlu::getCurMLUStream(), params),
             cnrtSuccess);
  }
  return head;
}

void deepseek_v41_decode_gather(const torch::Tensor& swa,
                                const torch::Tensor& compressed,
                                const torch::Tensor& current,
                                const torch::Tensor& swa_table,
                                const torch::Tensor& compressed_table,
                                const torch::Tensor& topk,
                                int64_t position,
                                int32_t window,
                                torch::Tensor output,
                                torch::Tensor length) {
  CHECK(detail::check_tensor(swa, torch::kBFloat16))
      << "swa must be contiguous BF16 on MLU";
  const torch::Device device = swa.device();
  CHECK(detail::check_tensor(compressed, torch::kBFloat16, device))
      << "compressed must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(current, torch::kBFloat16, device))
      << "current must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(swa_table, torch::kInt32, device))
      << "swa_table must be contiguous int32 on the same device as swa";
  CHECK(detail::check_tensor(compressed_table, torch::kInt32, device))
      << "compressed_table must be contiguous int32 on the same device as swa";
  CHECK(detail::check_tensor(topk, torch::kInt64, device))
      << "topk must be contiguous int64 on the same device as swa";
  CHECK(detail::check_tensor(output, torch::kBFloat16, device))
      << "output must be contiguous BF16 on the same device as swa";
  CHECK(detail::check_tensor(length, torch::kInt32, device))
      << "length must be contiguous int32 on the same device as swa";
  CHECK_EQ(swa.size(-1), 512);
  CHECK_EQ(current.numel(), 512);
  CHECK_EQ(length.numel(), 1);
  CHECK_GE(output.numel(),
           (std::min<int64_t>(position + 1, window) + topk.numel()) * 512);
  deepseek_v41_gather_launch(torch_mlu::getCurMLUStream(),
                             swa.data_ptr(),
                             compressed.data_ptr(),
                             current.data_ptr(),
                             swa_table.data_ptr<int32_t>(),
                             compressed_table.data_ptr<int32_t>(),
                             topk.data_ptr<int64_t>(),
                             topk.numel(),
                             position,
                             window,
                             swa_table.numel(),
                             output.data_ptr(),
                             length.data_ptr<int32_t>());
  // Check submission errors only: launches remain asynchronous. Device
  // execution failures surface at the caller's stream/graph synchronization
  // boundary.
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
}

void fused_compress_kv_cache(
    const torch::Tensor& raw,
    const torch::Tensor& norm_weight,
    const torch::Tensor& cos_sin_cache,
    const CompressKvMetadata& metadata,
    const std::optional<CompressKvPoolingInputs>& pooling,
    torch::Tensor main_kv_cache,
    const CompressKvOutputs& outputs,
    const CompressKvOptions& options) {
  CHECK(options.ratio == 1 || options.ratio == 2)
      << "options.ratio must be 1 or 2";
  CHECK(std::isfinite(options.rms_norm_eps) && options.rms_norm_eps > 0)
      << "options.rms_norm_eps must be finite and positive";
  const torch::Device device = raw.device();
  CHECK(device.type() == torch::kPrivateUse1) << "raw must be an MLU tensor";
  const bool pooled = options.ratio == 2;
  CHECK_EQ(pooling.has_value(), pooled)
      << "pooling inputs must be present exactly for options.ratio 2";
  check_rows(raw, device, pooled ? torch::kFloat : torch::kBFloat16, "raw");
  const int64_t batch = metadata.query_start_loc.numel() - 1;
  CHECK(batch >= 0) << "metadata.query_start_loc must have B+1 entries";
  check_int_vector(
      metadata.query_start_loc, device, batch + 1, "metadata.query_start_loc");
  check_int_vector(
      metadata.chunk_begins, device, batch, "metadata.chunk_begins");
  check_int_vector(metadata.compressed_start_loc,
                   device,
                   batch + 1,
                   "metadata.compressed_start_loc");
  check_table(metadata.compressed_block_table,
              device,
              batch,
              "metadata.compressed_block_table");

  check_device(norm_weight, device, "norm_weight");
  CHECK(norm_weight.scalar_type() == torch::kFloat ||
        norm_weight.scalar_type() == torch::kBFloat16)
      << "norm_weight must be float32 or bfloat16";
  CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == kHeadDim &&
        norm_weight.is_contiguous())
      << "norm_weight must be a contiguous [512] tensor";
  check_device(cos_sin_cache, device, "cos_sin_cache");
  CHECK(cos_sin_cache.scalar_type() == torch::kFloat &&
        cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeColumns &&
        cos_sin_cache.is_contiguous())
      << "cos_sin_cache must be a contiguous float32 [P, 128] tensor";

  check_device(main_kv_cache, device, "main_kv_cache");
  CHECK(main_kv_cache.scalar_type() == torch::kBFloat16 &&
        main_kv_cache.is_contiguous())
      << "main_kv_cache must be contiguous bfloat16";
  CHECK((main_kv_cache.dim() == 3 && main_kv_cache.size(1) == kPageSize &&
         main_kv_cache.size(2) == kHeadDim) ||
        (main_kv_cache.dim() == 4 && main_kv_cache.size(1) == kPageSize &&
         main_kv_cache.size(2) == 1 && main_kv_cache.size(3) == kHeadDim))
      << "main_kv_cache must be [Nm, 128, 512] or [Nm, 128, 1, 512]";

  check_device(outputs.latent, device, "outputs.latent");
  CHECK(outputs.latent.scalar_type() == torch::kBFloat16 &&
        outputs.latent.dim() == 2 && outputs.latent.size(1) == kHeadDim &&
        outputs.latent.is_contiguous())
      << "outputs.latent must be a contiguous bfloat16 [Kcap, 512] tensor";
  const int64_t k_capacity = outputs.latent.size(0);
  CHECK(k_capacity <= std::numeric_limits<int32_t>::max())
      << "Kcap must fit int32";
  check_int_vector(outputs.positions, device, k_capacity, "outputs.positions");
  check_int_vector(outputs.slots, device, k_capacity, "outputs.slots");
  check_int_vector(outputs.num_valid_rows, device, 1, "outputs.num_valid_rows");

  FusedCompressKvCacheParams params{};
  if (pooled) {
    CHECK(pooling->scores.defined() && pooling->state_block_table.defined() &&
          pooling->state_raw.defined() && pooling->state_score.defined())
        << "options.ratio 2 requires defined scores, state_block_table and "
           "both states";
    check_rows(pooling->scores, device, torch::kFloat, "scores");
    CHECK(pooling->scores.size(0) == raw.size(0))
        << "scores must have the same rows as raw";
    check_table(pooling->state_block_table, device, batch, "state_block_table");
    CHECK(batch == 0 || pooling->state_block_table.size(1) > 0)
        << "state_block_table needs at least one column";
    check_state(pooling->state_raw, device, "state_raw");
    check_state(pooling->state_score, device, "state_score");
    params.scores = pooling->scores.data_ptr<float>();
    params.score_row_stride = pooling->scores.stride(0);
    params.packed_projection =
        raw.stride(0) == 2 * kHeadDim &&
        pooling->scores.stride(0) == 2 * kHeadDim &&
        pooling->scores.data_ptr<float>() == raw.data_ptr<float>() + kHeadDim;
    params.state_block_table =
        pooling->state_block_table.numel() > 0
            ? pooling->state_block_table.data_ptr<int32_t>()
            : metadata.query_start_loc.data_ptr<int32_t>();
    params.state_table_stride = pooling->state_block_table.stride(0);
    params.state_columns =
        static_cast<int32_t>(pooling->state_block_table.size(1));
    params.state_raw = pooling->state_raw.data_ptr<float>();
    params.state_raw_page_stride = pooling->state_raw.stride(0);
    params.state_raw_row_stride = pooling->state_raw.stride(1);
    params.state_score = pooling->state_score.data_ptr<float>();
    params.state_score_page_stride = pooling->state_score.stride(0);
    params.state_score_row_stride = pooling->state_score.stride(1);
    params.packed_state =
        pooling->state_score.data_ptr<float>() ==
            pooling->state_raw.data_ptr<float>() + kHeadDim &&
        pooling->state_raw.stride(0) == pooling->state_score.stride(0) &&
        pooling->state_raw.stride(1) == pooling->state_score.stride(1) &&
        pooling->state_raw.stride(1) >= 2 * kHeadDim;
  }
  if (metadata.input_row_map.has_value()) {
    check_device(*metadata.input_row_map, device, "metadata.input_row_map");
    CHECK(metadata.input_row_map->scalar_type() == torch::kInt &&
          metadata.input_row_map->dim() == 1 &&
          metadata.input_row_map->is_contiguous())
        << "metadata.input_row_map must be a contiguous int32 vector";
    params.input_row_map = metadata.input_row_map->numel() > 0
                               ? metadata.input_row_map->data_ptr<int32_t>()
                               : nullptr;
    params.row_map_size = metadata.input_row_map->numel();
  }
  params.raw = raw.data_ptr();
  params.raw_row_stride = raw.stride(0);
  params.norm_weight = norm_weight.data_ptr();
  params.norm_weight_bf16 = norm_weight.scalar_type() == torch::kBFloat16;
  params.cos_sin_cache = cos_sin_cache.data_ptr<float>();
  params.query_start_loc = metadata.query_start_loc.data_ptr<int32_t>();
  // Empty B=0 tensors have null storage; keep every kernel pointer valid.
  int32_t* fallback = metadata.query_start_loc.data_ptr<int32_t>();
  params.chunk_begins =
      batch > 0 ? metadata.chunk_begins.data_ptr<int32_t>() : fallback;
  params.compressed_start_loc =
      metadata.compressed_start_loc.data_ptr<int32_t>();
  params.compressed_block_table =
      metadata.compressed_block_table.numel() > 0
          ? metadata.compressed_block_table.data_ptr<int32_t>()
          : fallback;
  params.compressed_table_stride = metadata.compressed_block_table.stride(0);
  params.compressed_columns =
      static_cast<int32_t>(metadata.compressed_block_table.size(1));
  params.main_kv_cache = main_kv_cache.data_ptr();
  params.latent_out = outputs.latent.data_ptr();
  params.positions_out = outputs.positions.data_ptr<int32_t>();
  params.slots_out = outputs.slots.data_ptr<int32_t>();
  params.num_valid_compressed_rows_out =
      outputs.num_valid_rows.data_ptr<int32_t>();
  params.batch = static_cast<int32_t>(batch);
  params.k_capacity = static_cast<int32_t>(k_capacity);
  params.ratio = static_cast<int32_t>(options.ratio);
  params.save_full_tail = pooling && pooling->save_full_tail ? 1 : 0;
  params.rms_norm_eps = static_cast<float>(options.rms_norm_eps);

  const torch::DeviceGuard device_guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream(device.index());
  const cnrtRet_t status = launch_fused_compress_kv_cache(queue, params);
  CHECK(status == cnrtSuccess)
      << "fused_compress_kv_cache launch failed: " << cnrtGetErrorStr(status);
}

}  // namespace xllm::kernel::mlu::bangc
