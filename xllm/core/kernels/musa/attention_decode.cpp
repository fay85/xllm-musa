/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kFa3MetadataRowsPerWarp = 31;
constexpr int32_t kFa3MaxMetadataWarps = 32;

constexpr int32_t kFa3TileM = 32;
constexpr int32_t kFa3TileN = 64;
constexpr int64_t kFa3PageSize = 64;
// Metadata and forward must use the same split policy. Zero lets both invoke
// Mate's dynamic heuristic for the current query and KV lengths.
constexpr int64_t kFa3PagedPrefillNumSplits = 0;
constexpr int32_t kFa3Gqa6MultiBatchDynamicSplitMinKvLength = 8192;

// Paged-KV FA3 (decode / with_kvcache). Qwen3 uses head_dim=128,
// GQA=2/4; Qwen3.5 uses head_dim=256, GQA=6/8.
constexpr const char* kFa3FwdUriHashGqa6 =
    "055332826071b0537d08ac0f855780d9dc53099d976fc6cd544d944b7940db24";
constexpr const char* kFa3FwdUriHashGqa2 =
    "dfced5e5e01a6151b94ebe792f5fb6a125ae8c4d39add457c0b52a7e5321fab4";
constexpr const char* kFa3FwdUriHashGqa4 =
    "c3e22a354954752b17a7c4f4ee52d035f6052a043a85c9f1474aba9f0152f682";
constexpr const char* kFa3FwdUriHashGqa7 =
    "83616583ddc213f94251d2993281b3fa1a5ccd5939c55dff11e6d9ad979306cb";
constexpr const char* kFa3FwdUriHashGqa8 =
    "96c89a71db14cd369d03c0e1cbb43a2815da277c2f592061425e2b436354127a";
constexpr const char* kFa3FwdCombine16Uri =
    "fmha_fwd_combine_bf16_16x64x16_ragged_q_metadata";
constexpr const char* kFa3FwdCombine32Uri =
    "fmha_fwd_combine_bf16_16x64x32_ragged_q_metadata";
constexpr const char* kFa3FwdCombine64Uri =
    "fmha_fwd_combine_bf16_16x64x64_ragged_q_metadata";

// Dense ragged FA3 prefill. GQA6 is a Mate 0.2.7 head-256 kernel: TileHeadDim 128, stages 2, tile 32. GQA 2/4/7 are Mate 0.2.7 head-128 stage 2. GQA 8 prefill is head-256 TileHeadDim 128.
constexpr const char* kFa3PrefillFwdUriHashGqa6 =
    "639113a0bd9f58a20ca13cdc5b37f65cfe8d2b6cba2296105219fd8299cad5bd";
constexpr const char* kFa3PrefillFwdUriHashGqa2 =
    "506d4dc691698bd18af62bcc2e2fe2cfe87d10f2b51f34f529d35a2d35fbb7e4";
constexpr const char* kFa3PrefillFwdUriHashGqa4 =
    "b4531c0d048ce50f14b49ef415a61127e46f6cccc7804f97c059de054eae1b54";
// GQA8 dense prefill uses the M=192 tile configuration.
constexpr const char* kFa3PrefillFwdUriHashGqa8 =
    "3c319be5ad6952b77e51c55060c55e2dc3da6090a8fc6a8734dbd8baf0f6ed53";

// Mate 0.2.5 remains supported for reproducible performance comparisons and
// deployments that have not regenerated the 0.2.6 dense-prefill artifacts.
constexpr const char* kFa3PrefillFwdUriHashGqa6V025 =
    "e140040cf5f04fdf94b0c9d28900659052de34f380b6ecea23afd16d3691d4ea";
constexpr const char* kFa3PrefillFwdUriHashGqa2V025 =
    "782ec5133834c8e9cec31f987596832dccb087fca95a59dfa17b7f4125ac1108";
constexpr const char* kFa3PrefillFwdUriHashGqa4V025 =
    "b6883be7b4cfcc0200f994cdd9ec9844d8ec6175c27a5ea5488723c9a43164bf";
constexpr const char* kFa3PrefillFwdUriHashGqa8V025 =
    "236afb906e2625d51106085cc57d954f7e5b1440b41da68c782fc894801386ee";

bool fa3_module_available(const std::string& uri) {
  const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path == nullptr || ops_path[0] == '\0') {
    return false;
  }
  const std::string so_path =
      std::string(ops_path) + "/" + uri + "/" + uri + ".so";
  return access(so_path.c_str(), R_OK) == 0;
}

std::string resolve_fa3_prefill_uri(const char* current_hash,
                                    const char* v025_hash) {
  const std::string current_uri = std::string("fmha_fwd_") + current_hash;
  if (fa3_module_available(current_uri)) {
    return current_uri;
  }

  const std::string v025_uri = std::string("fmha_fwd_") + v025_hash;
  if (fa3_module_available(v025_uri)) {
    return v025_uri;
  }
  return current_uri;
}

int32_t fa3_metadata_num_warps(int32_t batch_size) {
  CHECK_GT(batch_size, 0);
  return std::min<int32_t>(
      (batch_size + kFa3MetadataRowsPerWarp - 1) / kFa3MetadataRowsPerWarp,
      kFa3MaxMetadataWarps);
}

int64_t fa3_metadata_numel(int32_t batch_size) {
  const int64_t rounded_batch =
      ((static_cast<int64_t>(batch_size) + 3) / 4) * 4;
  return rounded_batch * 4;
}

std::string fa3_metadata_uri(int64_t gqa_ratio, int32_t batch_size) {
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "MUSA FA3 metadata supports GQA ratios 2, 4, 6, 7, and 8, got "
      << gqa_ratio;
  if (gqa_ratio == 7) {
    return "fmha_get_metadata_7x1_bf16_ragged_q_padded_k_causal_packgqa";
  }
  return "fmha_get_metadata_" + std::to_string(gqa_ratio) + "x" +
         std::to_string(fa3_metadata_num_warps(batch_size)) +
         "_bf16_ragged_q_padded_k_causal_packgqa";
}

std::string fa3_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "MUSA FA3 decode supports GQA ratios 2, 4, 6, 7, and 8, got "
      << gqa_ratio;
  const char* hash = kFa3FwdUriHashGqa8;
  if (gqa_ratio == 2) {
    hash = kFa3FwdUriHashGqa2;
  } else if (gqa_ratio == 4) {
    hash = kFa3FwdUriHashGqa4;
  } else if (gqa_ratio == 6) {
    hash = kFa3FwdUriHashGqa6;
  } else if (gqa_ratio == 7) {
    hash = kFa3FwdUriHashGqa7;
  }
  return std::string("fmha_fwd_") + hash;
}

std::string fa3_prefill_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "MUSA FA3 prefill supports GQA ratios 2, 4, 6, 7, and 8, got "
      << gqa_ratio;
  if (gqa_ratio == 2) {
    static const std::string uri = resolve_fa3_prefill_uri(
        kFa3PrefillFwdUriHashGqa2, kFa3PrefillFwdUriHashGqa2V025);
    return uri;
  }
  if (gqa_ratio == 4) {
    static const std::string uri = resolve_fa3_prefill_uri(
        kFa3PrefillFwdUriHashGqa4, kFa3PrefillFwdUriHashGqa4V025);
    return uri;
  }
  if (gqa_ratio == 6) {
    static const std::string uri = resolve_fa3_prefill_uri(
        kFa3PrefillFwdUriHashGqa6, kFa3PrefillFwdUriHashGqa6V025);
    return uri;
  }
  if (gqa_ratio == 7) {
    return "fmha_fwd_"
           "5f890447fb5d797d43f7fb6f874b3560f01514ba7e4bf6df31dc67d8f1377267";
  }
  static const std::string uri = resolve_fa3_prefill_uri(
      kFa3PrefillFwdUriHashGqa8, kFa3PrefillFwdUriHashGqa8V025);
  return uri;
}

bool is_fa3_shape_supported(int64_t head_dim_qk,
                            int64_t head_dim_vo,
                            int64_t gqa_ratio) {
  return (head_dim_qk == 128 && head_dim_vo == 128 &&
          (gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 7 ||
           gqa_ratio == 8)) ||
         (head_dim_qk == 256 && head_dim_vo == 256 &&
          (gqa_ratio == 6 || gqa_ratio == 8));
}

bool enable_fa3_gqa6_dynamic_split() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_FA3_GQA6_DYNAMIC_SPLIT");
    return env == nullptr || std::string(env) != "0";
  }();
  return enabled;
}

int64_t fa3_gqa6_num_splits(int32_t batch_size, int32_t max_seqlen_k) {
  if (!enable_fa3_gqa6_dynamic_split()) {
    return 1;
  }

  // Split-K exposes useful parallelism for isolated decode. With a larger
  // batch, the heads and sequences already fill the device; defer the extra
  // reduction until the KV length is large enough to amortize it.
  if (batch_size <= 2 ||
      max_seqlen_k >= kFa3Gqa6MultiBatchDynamicSplitMinKvLength) {
    return 0;
  }
  return 1;
}

ffi::Optional<ffi::Tensor> none_tensor() {
  return ffi::Optional<ffi::Tensor>();
}

ffi::Optional<int64_t> none_int() { return ffi::Optional<int64_t>(); }

void combine_fa3_split_k(const ffi::Any& fwd_result,
                         const torch::Tensor& cu_seqlens_q,
                         int64_t max_seqlen_q,
                         const torch::Tensor& scheduler_metadata,
                         torch::Tensor& output,
                         torch::Tensor& output_lse) {
  auto result_values = fwd_result.cast<ffi::Array<ffi::Any>>();
  CHECK_EQ(result_values.size(), 2)
      << "FA3 forward must return [accumulators, num_splits]";
  auto accumulators = result_values[0].cast<ffi::Array<ffi::Tensor>>();
  CHECK_EQ(accumulators.size(), 2)
      << "FA3 forward must return output and LSE accumulators";
  const int64_t num_splits = result_values[1].cast<int64_t>();
  CHECK_GT(num_splits, 0) << "FA3 forward returned invalid split count";
  CHECK_LE(num_splits, 64)
      << "deployed FA3 combine artifacts support at most 64 splits";
  const char* combine_uri =
      num_splits <= 16
          ? kFa3FwdCombine16Uri
          : (num_splits <= 32 ? kFa3FwdCombine32Uri : kFa3FwdCombine64Uri);

  get_function(combine_uri, combine_uri)(to_ffi_tensor(cu_seqlens_q),
                                         none_tensor(),
                                         ffi::Optional<int64_t>(max_seqlen_q),
                                         to_ffi_tensor(output),
                                         to_ffi_tensor(output_lse),
                                         accumulators[0],
                                         accumulators[1],
                                         to_ffi_tensor(scheduler_metadata),
                                         num_splits);
}

}  // namespace

int32_t fa3_decode_kv_tile_size() { return kFa3TileN; }

torch::Tensor fa3_decode_scheduler_metadata(const torch::Device& device,
                                            int32_t batch_size,
                                            int32_t num_heads_q,
                                            int32_t num_heads_kv,
                                            int32_t head_dim_qk,
                                            int32_t head_dim_vo,
                                            int32_t max_seqlen_q,
                                            int32_t max_seqlen_k,
                                            int32_t window_size_left,
                                            int32_t window_size_right,
                                            const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_k,
                                            int64_t num_splits) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK_EQ(cu_seqlens_q.numel(), static_cast<int64_t>(batch_size) + 1)
      << "FA3 scheduler query indptr must have batch+1 entries";
  CHECK_EQ(seqused_k.numel(), batch_size)
      << "FA3 scheduler KV lengths must match batch size";
  CHECK(cu_seqlens_q.is_contiguous())
      << "FA3 scheduler query indptr must be contiguous";
  CHECK(seqused_k.is_contiguous())
      << "FA3 scheduler KV lengths must be contiguous";

  torch::Tensor metadata =
      torch::empty({fa3_metadata_numel(batch_size)},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));
  fa3_decode_scheduler_metadata(device,
                                batch_size,
                                num_heads_q,
                                num_heads_kv,
                                head_dim_qk,
                                head_dim_vo,
                                max_seqlen_q,
                                max_seqlen_k,
                                window_size_left,
                                window_size_right,
                                cu_seqlens_q,
                                seqused_k,
                                num_splits,
                                metadata);
  return metadata;
}

void fa3_decode_scheduler_metadata(const torch::Device& device,
                                   int32_t batch_size,
                                   int32_t num_heads_q,
                                   int32_t num_heads_kv,
                                   int32_t head_dim_qk,
                                   int32_t head_dim_vo,
                                   int32_t max_seqlen_q,
                                   int32_t max_seqlen_k,
                                   int32_t window_size_left,
                                   int32_t window_size_right,
                                   const torch::Tensor& cu_seqlens_q,
                                   const torch::Tensor& seqused_k,
                                   int64_t num_splits,
                                   torch::Tensor& metadata) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK_EQ(cu_seqlens_q.numel(), static_cast<int64_t>(batch_size) + 1)
      << "FA3 scheduler query indptr must have batch+1 entries";
  CHECK_EQ(seqused_k.numel(), batch_size)
      << "FA3 scheduler KV lengths must match batch size";
  CHECK(cu_seqlens_q.is_contiguous())
      << "FA3 scheduler query indptr must be contiguous";
  CHECK(seqused_k.is_contiguous())
      << "FA3 scheduler KV lengths must be contiguous";
  CHECK(metadata.defined()) << "FA3 scheduler metadata buffer must be defined";
  CHECK_EQ(metadata.numel(), fa3_metadata_numel(batch_size))
      << "FA3 scheduler metadata buffer has an invalid size";
  CHECK_EQ(metadata.scalar_type(), torch::kInt32);
  CHECK(metadata.is_contiguous());
  CHECK_EQ(metadata.device(), device);

  TvmffiStreamGuard stream_guard(device);

  const int64_t gqa_ratio = num_heads_q / num_heads_kv;
  const std::string uri = fa3_metadata_uri(gqa_ratio, batch_size);
  const int64_t metadata_num_splits =
      num_splits > 0
          ? num_splits
          : (gqa_ratio == 6 ? fa3_gqa6_num_splits(batch_size, max_seqlen_k)
                            : static_cast<int64_t>(0));

  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(seqused_k),
                         none_tensor(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         none_tensor(),
                         to_ffi_tensor(metadata),
                         metadata_num_splits,
                         static_cast<int64_t>(kFa3TileM),
                         static_cast<int64_t>(kFa3TileN),
                         /*mp_margin=*/static_cast<int64_t>(0));
}

// Keep the v0.11 caller ABI while routing it through the MATE 0.2.7
// scheduler contract.  New callers can provide an explicit split policy.
torch::Tensor fa3_decode_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seqused_k) {
  return fa3_decode_scheduler_metadata(device,
                                       batch_size,
                                       num_heads_q,
                                       num_heads_kv,
                                       head_dim_qk,
                                       head_dim_vo,
                                       max_seqlen_q,
                                       max_seqlen_k,
                                       window_size_left,
                                       window_size_right,
                                       cu_seqlens_q,
                                       seqused_k,
                                       /*num_splits=*/0);
}

void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                int64_t num_splits,
                torch::Tensor& output,
                torch::Tensor& output_lse) {
  CHECK(scheduler_metadata.defined())
      << "fa3_decode: scheduler_metadata must be precomputed";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK(page_table.defined() && page_table.scalar_type() == torch::kInt32);
  CHECK_EQ(k_cache.dim(), 4) << "fa3_decode: key cache must be 4D";
  CHECK_EQ(v_cache.dim(), 4) << "fa3_decode: value cache must be 4D";
  CHECK_EQ(k_cache.sizes(), v_cache.sizes())
      << "fa3_decode: key/value cache shapes must match";
  CHECK_EQ(k_cache.size(1), kFa3PageSize)
      << "fa3_decode: deployed FA3 artifacts require page size 64";
  CHECK_EQ(v_cache.size(1), kFa3PageSize)
      << "fa3_decode: deployed FA3 artifacts require page size 64";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_decode: only bf16 is supported";
  CHECK_EQ(k_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(v_cache.scalar_type(), torch::kBFloat16);
  const int64_t batch_size = seqused_k.numel();
  CHECK_GT(batch_size, 0);
  CHECK_EQ(cu_seqlens_q.numel(), batch_size + 1);
  CHECK_EQ(page_table.size(0), batch_size);
  CHECK_EQ(scheduler_metadata.numel(), fa3_metadata_numel(batch_size))
      << "fa3_decode: scheduler_metadata has an invalid size";

  CHECK_GT(k_cache.size(-2), 0);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  CHECK(is_fa3_shape_supported(query.size(-1), k_cache.size(-1), gqa_ratio))
      << "fa3_decode: unsupported attention shape (head_dim_qk="
      << query.size(-1) << ", head_dim_vo=" << k_cache.size(-1)
      << ", gqa_ratio=" << gqa_ratio << ")";
  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  auto fwd_result =
      get_function(uri, uri)(to_ffi_tensor(query),
                             to_ffi_tensor(k_cache),
                             to_ffi_tensor(v_cache),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             to_ffi_tensor(cu_seqlens_q),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             to_ffi_tensor(seqused_k),
                             ffi::Optional<int64_t>(max_seqlen_q),
                             none_int(),
                             to_ffi_tensor(page_table),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             none_tensor(),
                             sm_scale,
                             /*is_causal=*/true,
                             window_left,
                             window_right,
                             /*attention_chunk=*/static_cast<int64_t>(0),
                             /*softcap=*/0.0,
                             /*mp_margin=*/static_cast<int64_t>(0),
                             /*num_splits=*/num_splits,
                             to_ffi_tensor(scheduler_metadata),
                             none_tensor(),
                             to_ffi_tensor(output),
                             to_ffi_tensor(output_lse),
                             /*cp_world_size=*/static_cast<int64_t>(1),
                             /*cp_rank=*/static_cast<int64_t>(0),
                             none_tensor(),
                             /*skip_kernel_launch=*/false);

  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
}

void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse) {
  fa3_decode(query,
             k_cache,
             v_cache,
             cu_seqlens_q,
             seqused_k,
             page_table,
             scheduler_metadata,
             max_seqlen_q,
             window_left,
             window_right,
             sm_scale,
             /*num_splits=*/0,
             output,
             output_lse);
}

void fa3_prefill(const torch::Tensor& query,
                 const torch::Tensor& key,
                 const torch::Tensor& value,
                 const torch::Tensor& cu_seqlens_q,
                 const torch::Tensor& cu_seqlens_k,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 int64_t window_left,
                 int64_t window_right,
                 double sm_scale,
                 torch::Tensor& output,
                 torch::Tensor& output_lse) {
  CHECK(query.defined() && key.defined() && value.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3) << "fa3_prefill: query must be [tokens, heads, dim]";
  CHECK_EQ(key.dim(), 3) << "fa3_prefill: key must be [tokens, heads, dim]";
  CHECK_EQ(value.dim(), 3) << "fa3_prefill: value must be [tokens, heads, dim]";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(cu_seqlens_k.defined() && cu_seqlens_k.scalar_type() == torch::kInt32);
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: only bf16 is supported for the cached dense FA3 URI";
  CHECK_EQ(key.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: key must be bf16";
  CHECK_EQ(value.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: value must be bf16";
  CHECK_EQ(output.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: output must be bf16";
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32)
      << "fa3_prefill: output_lse must be fp32";
  CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous())
      << "fa3_prefill: query, key, and value must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous())
      << "fa3_prefill: cumulative sequence lengths must be contiguous";
  CHECK(output.is_contiguous() && output_lse.is_contiguous())
      << "fa3_prefill: output tensors must be contiguous";
  CHECK_EQ(value.sizes(), key.sizes())
      << "fa3_prefill: key and value shapes must match";
  CHECK_GT(key.size(-2), 0);
  CHECK_EQ(query.size(-2) % key.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / key.size(-2);
  CHECK(is_fa3_shape_supported(query.size(-1), value.size(-1), gqa_ratio))
      << "fa3_prefill: unsupported attention shape";
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "fa3_prefill: requires GQA ratio 2, 4, 6, 7, or 8 (nq="
      << query.size(-2) << ", nkv=" << key.size(-2) << ")";
  CHECK_EQ(output.sizes(), query.sizes())
      << "fa3_prefill: output shape must match query";
  CHECK_EQ(output_lse.dim(), 2)
      << "fa3_prefill: output_lse must be [heads, tokens]";
  CHECK_EQ(output_lse.size(0), query.size(1))
      << "fa3_prefill: output_lse head count must match query";
  CHECK_GE(output_lse.numel(), query.size(0) * query.size(1))
      << "fa3_prefill: output_lse is too small";
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  const std::string uri = fa3_prefill_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  get_function(uri, uri)(to_ffi_tensor(query),
                         to_ffi_tensor(key),
                         to_ffi_tensor(value),
                         none_tensor(),  // k_new
                         none_tensor(),  // v_new
                         none_tensor(),  // q_v
                         to_ffi_tensor(cu_seqlens_q),
                         to_ffi_tensor(cu_seqlens_k),
                         none_tensor(),  // cu_seqlens_k_new
                         none_tensor(),  // seqused_q
                         none_tensor(),  // seqused_k
                         max_seqlen_q,
                         max_seqlen_k,
                         none_tensor(),  // page_table
                         none_tensor(),  // kv_batch_idx
                         none_tensor(),  // leftpad_k
                         none_tensor(),  // rotary_cos
                         none_tensor(),  // rotary_sin
                         none_tensor(),  // seqlens_rotary
                         none_tensor(),  // q_descale
                         none_tensor(),  // k_descale
                         none_tensor(),  // v_descale
                         sm_scale,
                         /*is_causal=*/true,
                         window_left,
                         window_right,
                         /*attention_chunk=*/static_cast<int64_t>(0),
                         /*softcap=*/0.0,
                         /*mp_margin=*/static_cast<int64_t>(0),
                         /*num_splits=*/static_cast<int64_t>(0),
                         none_tensor(),  // scheduler_metadata
                         none_tensor(),  // learnable_sink
                         to_ffi_tensor(output),
                         to_ffi_tensor(output_lse),
                         /*cp_world_size=*/static_cast<int64_t>(1),
                         /*cp_rank=*/static_cast<int64_t>(0),
                         none_tensor(),
                         /*skip_kernel_launch=*/false);
}

torch::Tensor fa3_prefill_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k_new,
    const torch::Tensor& seqused_k) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(cu_seqlens_k_new.defined() &&
        cu_seqlens_k_new.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK_EQ(cu_seqlens_q.numel(), static_cast<int64_t>(batch_size) + 1);
  CHECK_EQ(cu_seqlens_k_new.numel(), static_cast<int64_t>(batch_size) + 1);
  CHECK_EQ(seqused_k.numel(), static_cast<int64_t>(batch_size));
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  const int64_t gqa_ratio = num_heads_q / num_heads_kv;
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "MUSA FA3 prefill metadata supports GQA ratios 2, 4, 6, 7, and 8, got "
      << gqa_ratio;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::empty({fa3_metadata_numel(batch_size)}, options);

  TvmffiStreamGuard stream_guard(device);
  const std::string uri = fa3_metadata_uri(gqa_ratio, batch_size);
  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(seqused_k),
                         none_tensor(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         none_tensor(),
                         to_ffi_tensor(metadata),
                         /*num_splits=*/kFa3PagedPrefillNumSplits,
                         static_cast<int64_t>(kFa3TileM),
                         static_cast<int64_t>(kFa3TileN),
                         /*mp_margin=*/static_cast<int64_t>(0));
  return metadata;
}

void fa3_prefill_paged(const torch::Tensor& query,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& cu_seqlens_k_new,
                       const torch::Tensor& seqused_k,
                       const torch::Tensor& page_table,
                       const torch::Tensor& scheduler_metadata,
                       int64_t max_seqlen_q,
                       int64_t window_left,
                       int64_t window_right,
                       double sm_scale,
                       torch::Tensor& output,
                       torch::Tensor& output_lse) {
  CHECK(query.defined() && k_cache.defined() && v_cache.defined());
  CHECK(cu_seqlens_q.defined() && cu_seqlens_k_new.defined());
  CHECK(seqused_k.defined() && page_table.defined());
  CHECK(scheduler_metadata.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3)
      << "fa3_prefill_paged: query must be [tokens, heads, dim]";
  CHECK_EQ(k_cache.dim(), 4)
      << "fa3_prefill_paged: k_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(v_cache.dim(), 4)
      << "fa3_prefill_paged: v_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(k_cache.sizes(), v_cache.sizes())
      << "fa3_prefill_paged: key/value cache shapes must match";
  CHECK_EQ(k_cache.size(1), kFa3PageSize)
      << "fa3_prefill_paged: deployed FA3 artifacts require page size 64";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill_paged: only bf16 is supported";
  CHECK_EQ(k_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(v_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32);
  CHECK(query.is_contiguous() && k_cache.is_contiguous() &&
        v_cache.is_contiguous() && output.is_contiguous() &&
        output_lse.is_contiguous())
      << "fa3_prefill_paged: tensors must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k_new.is_contiguous() &&
        seqused_k.is_contiguous() && page_table.is_contiguous() &&
        scheduler_metadata.is_contiguous())
      << "fa3_prefill_paged: metadata tensors must be contiguous";
  CHECK_EQ(cu_seqlens_q.scalar_type(), torch::kInt32);
  CHECK_EQ(cu_seqlens_k_new.scalar_type(), torch::kInt32);
  CHECK_EQ(seqused_k.scalar_type(), torch::kInt32);
  CHECK_EQ(page_table.scalar_type(), torch::kInt32);
  CHECK_EQ(scheduler_metadata.scalar_type(), torch::kInt32);
  CHECK_GT(k_cache.size(-2), 0);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  CHECK(is_fa3_shape_supported(query.size(-1), k_cache.size(-1), gqa_ratio))
      << "fa3_prefill_paged: unsupported attention shape";
  CHECK(gqa_ratio == 2 || gqa_ratio == 4 || gqa_ratio == 6 || gqa_ratio == 7 ||
        gqa_ratio == 8)
      << "fa3_prefill_paged: requires GQA ratio 2, 4, 6, 7, or 8, got "
      << gqa_ratio;
  CHECK_EQ(page_table.size(0), seqused_k.numel());
  CHECK_EQ(scheduler_metadata.numel(),
           fa3_metadata_numel(static_cast<int32_t>(seqused_k.numel())));
  CHECK_GT(max_seqlen_q, 0);

  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  auto fwd_result =
      get_function(uri, uri)(to_ffi_tensor(query),
                             to_ffi_tensor(k_cache),
                             to_ffi_tensor(v_cache),
                             none_tensor(),  // k_new
                             none_tensor(),  // v_new
                             none_tensor(),  // q_v
                             to_ffi_tensor(cu_seqlens_q),
                             none_tensor(),  // cu_seqlens_k
                             none_tensor(),  // cu_seqlens_k_new
                             none_tensor(),  // seqused_q
                             to_ffi_tensor(seqused_k),
                             static_cast<int64_t>(max_seqlen_q),
                             none_int(),  // max_seqlen_k
                             to_ffi_tensor(page_table),
                             none_tensor(),  // kv_batch_idx
                             none_tensor(),  // leftpad_k
                             none_tensor(),  // rotary_cos
                             none_tensor(),  // rotary_sin
                             none_tensor(),  // seqlens_rotary
                             none_tensor(),  // q_descale
                             none_tensor(),  // k_descale
                             none_tensor(),  // v_descale
                             sm_scale,
                             /*is_causal=*/true,
                             window_left,
                             window_right,
                             /*attention_chunk=*/static_cast<int64_t>(0),
                             /*softcap=*/0.0,
                             /*mp_margin=*/static_cast<int64_t>(0),
                             /*num_splits=*/kFa3PagedPrefillNumSplits,
                             to_ffi_tensor(scheduler_metadata),
                             none_tensor(),  // learnable_sink
                             to_ffi_tensor(output),
                             to_ffi_tensor(output_lse),
                             /*cp_world_size=*/static_cast<int64_t>(1),
                             /*cp_rank=*/static_cast<int64_t>(0),
                             none_tensor(),
                             /*skip_kernel_launch=*/false);
  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
}

void batch_decode(const std::string& uri,
                  ffi::Array<int64_t> plan_info,
                  torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool use_tensor_core,
                  std::optional<torch::Tensor> qo_indptr,
                  const torch::Tensor& paged_kv_indptr_host,
                  const torch::Tensor& paged_kv_indices_host,
                  const torch::Tensor& paged_kv_last_page_len_host) {
  (void)use_tensor_core;
  {
    VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;

    const torch::Tensor paged_kv_indptr_ffi =
        paged_kv_indptr_host.defined() ? paged_kv_indptr_host
                                       : paged_kv_indptr.to(torch::kCPU);
    const torch::Tensor paged_kv_indices_ffi =
        paged_kv_indices_host.defined() ? paged_kv_indices_host
                                        : paged_kv_indices.to(torch::kCPU);
    const torch::Tensor paged_kv_last_page_len_ffi =
        paged_kv_last_page_len_host.defined()
            ? paged_kv_last_page_len_host
            : paged_kv_last_page_len.to(torch::kCPU);

    TvmffiStreamGuard stream_guard(query.device());
    get_function(uri, "run")(
        to_ffi_tensor(float_workspace_buffer),
        to_ffi_tensor(int_workspace_buffer),
        plan_info,
        to_ffi_tensor(query),
        to_ffi_tensor(k_cache),
        to_ffi_tensor(v_cache),
        to_ffi_tensor(paged_kv_indptr_ffi),
        to_ffi_tensor(paged_kv_indices_ffi),
        to_ffi_tensor(paged_kv_last_page_len_ffi),
        to_ffi_tensor(output),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*kv_layout_code=*/0,
        window_left,
        support_pdl(),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0);
  }
}

}  // namespace xllm::kernel::musa
