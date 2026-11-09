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

#include <bang.h>
#include <stdint.h>

namespace xllm::kernel::mlu::bangc::fused_mhc_detail {
namespace {

// Keep device definitions local to each translation unit so its CNAS options
// also apply to the shared helpers. No VV macros escape this header.
constexpr int32_t kHc = 4;
constexpr int32_t kMix = 24;

// dst[r] = sum(src[r][0..n)) for r < rows, n % 32 == 0. Scratch holds
// 2 * rows * 32 floats; two sumpools avoid the scalar-pipe sum.nram stall.
__mlu_func__ inline void row_sums(float* dst,
                                  const float* src,
                                  int32_t rows,
                                  int32_t n,
                                  float* scratch) {
  float* part = scratch;
  float* part_t = scratch + rows * 32;
  __bang_sumpool(
      part, const_cast<float*>(src), 32, rows, n / 32, 1, n / 32, n / 32, 1);
  __bang_transpose(part_t, part, rows, 32);
  __bang_sumpool(dst, part_t, rows, 1, 32, 1, 32, 32, 1);
}

// dst = bf16(src * scalar * vec), one vector pass.
__mlu_func__ inline void fuse_mul_scalar_mul_vec_cast(bfloat16_t* dst,
                                                      const float* src,
                                                      float scalar,
                                                      const float* vec,
                                                      int32_t num) {
  __asm__ __volatile__(
      "fuse.nram.crn.bf16.f32 [%[dst]], %[size], [%[src]], "
      ".mul(%[scalar]), .mul([%[vec]]) ;\n\t" ::[dst] "r"(dst),
      [src] "r"(src),
      [size] "r"(num),
      [scalar] "r"(scalar),
      [vec] "r"(vec));
}

// bf16 -> f32 on the move engine (NRAM -> NRAM), contiguous rows.
// elems * 2 must be a multiple of 64 bytes. Asynchronous on the move queue.
__mlu_func__ inline void move_cvt_bf16_to_f32(float* dst,
                                              const bfloat16_t* src,
                                              int32_t elems,
                                              int32_t rows) {
  const int32_t sn0 = 64, sn3 = elems * 2 / 64, ss3 = 64, sn4 = rows,
                ss4 = elems * 2;
  const int32_t dn0 = 64, dn3 = sn3, ds3 = 128, dn4 = rows, ds4 = elems * 4;
  __asm__ volatile(
      "move.tiling.async.nram.nram.b16 [%[dst]], [%[src]], %[sn0], 1, 0, 1, 0, "
      "%[sn3], %[ss3], "
      "%[sn4], %[ss4], 1, 0, %[dn0], 1, 0, 1, 0, %[dn3], %[ds3], %[dn4], "
      "%[ds4], 1, 0,"
      ".cvt.rn.f32.bf16();" ::[dst] "r"(dst),
      [src] "r"(src),
      [sn0] "r"(sn0),
      [sn3] "r"(sn3),
      [ss3] "r"(ss3),
      [sn4] "r"(sn4),
      [ss4] "r"(ss4),
      [dn0] "r"(dn0),
      [dn3] "r"(dn3),
      [ds3] "r"(ds3),
      [dn4] "r"(dn4),
      [ds4] "r"(ds4));
}

// In-place sigmoid through pow2 and reciprocal.
__mlu_func__ inline void apply_sigmoid(float* data, uint32_t n) {
  const float neg_log2e = -1.4426950408889634f;  // -log2(e)
  __bang_mul_scalar(data, data, neg_log2e, n);
  __bang_pow2(data, data, n);
  __bang_add_scalar(data, data, 1.0f, n);
  __bang_recip(data, data, n);
}

// In-place comb SoA [16][n]: softmax rows + eps, then column normalization.
// Later row/column sums use pairwise trees; preserve this rounding order.
__mlu_func__ inline void compute_sinkhorn_pairwise(float* comb,
                                                   int32_t n,
                                                   int32_t iters,
                                                   float eps) {
  const float log2e = 1.4426950408889634f;
  vv_float v[kHc * kHc];
  vv_float mx[kHc];
  vv_float s[kHc];
  const int32_t per_vv = __vv_get_length() / sizeof(float);
  asm volatile("vv.begin;\n\t");
  for (int32_t base = 0; base < n; base += per_vv) {
    const int32_t cur = n - base < per_vv ? n - base : per_vv;
    for (int32_t i = 0; i < kHc * kHc; i++) {
      __vv_load(v[i], comb + i * n + base, cur);
    }
#pragma unroll 4
    for (int32_t i = 0; i < kHc; i++) {
      __vv_move(mx[i], v[i * kHc]);
#pragma unroll 4
      for (int32_t j = 1; j < kHc; j++) {
        __vv_max(mx[i], mx[i], v[i * kHc + j]);
      }
      __vv_move(s[i], 0);
      for (int32_t j = 0; j < kHc; j++) {
        __vv_sub(v[i * kHc + j], v[i * kHc + j], mx[i]);
        __vv_mul(v[i * kHc + j], v[i * kHc + j], log2e);
        __vv_pow2(v[i * kHc + j], v[i * kHc + j]);
        __vv_add(s[i], s[i], v[i * kHc + j]);
      }
      __vv_recip(s[i], s[i]);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_mul(v[i * kHc + j], v[i * kHc + j], s[i]);
        __vv_add(v[i * kHc + j], v[i * kHc + j], eps);
      }
    }
#pragma unroll 4
    for (int32_t i = 0; i < kHc; i++) {
      __vv_move(s[i], eps);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_add(s[i], s[i], v[i + j * kHc]);
      }
      __vv_recip(s[i], s[i]);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_mul(v[i + j * kHc], v[i + j * kHc], s[i]);
      }
    }
    // row / column sums as pairwise trees ((a + b) + (c + d)) + eps: dependency
    // depth 3 instead of 5 on the latency-bound vv pipe
    for (int32_t k = 0; k + 1 < iters; k++) {
      vv_float t[kHc];
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], v[i * kHc], v[i * kHc + 1]);
        __vv_add(t[i], v[i * kHc + 2], v[i * kHc + 3]);
      }
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], s[i], t[i]);
      }
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], s[i], eps);
        __vv_recip(s[i], s[i]);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_mul(v[i * kHc + j], v[i * kHc + j], s[i]);
        }
      }
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], v[i], v[i + kHc]);
        __vv_add(t[i], v[i + 2 * kHc], v[i + 3 * kHc]);
      }
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], s[i], t[i]);
      }
#pragma unroll
      for (int32_t i = 0; i < kHc; i++) {
        __vv_add(s[i], s[i], eps);
        __vv_recip(s[i], s[i]);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_mul(v[i + j * kHc], v[i + j * kHc], s[i]);
        }
      }
    }
    for (int32_t i = 0; i < kHc * kHc; i++) {
      __vv_store(comb + i * n + base, v[i], cur);
    }
  }
  asm volatile("vv.end;\n\t");
}

// dst[0] = sum(src[0..n)), n % 64 == 0; scratch holds 64 floats.
// Asynchronous with respect to the scalar core: read after __sync_compute().
__mlu_func__ inline void sum_to(float* dst,
                                const float* src,
                                int32_t n,
                                float* scratch) {
  __bang_sumpool(
      scratch, const_cast<float*>(src), 64, 1, n / 64, 1, n / 64, 1, 1);
  __asm__ volatile("sum.nram.f32 [%[dst]], [%[src]], %[num];" ::[dst] "r"(dst),
                   [src] "r"(scratch),
                   [num] "r"(64));
}

// Prefill variant: later row/column sums accumulate sequentially from eps.
// Keep its arithmetic order and non-volatile VV markers separate from decode.
__mlu_func__ inline void compute_sinkhorn_sequential(float* output,
                                                     float* input,
                                                     uint32_t deal_num,
                                                     uint32_t iters,
                                                     float eps) {
  const float log2e = 1.4426950408889634f;
  vv_float in_vv[kHc * kHc];
  vv_float in_max[kHc];
  vv_float in_sum[kHc];
  uint32_t num_per_vv = __vv_get_length() / sizeof(float);
  uint32_t repeat_vv = deal_num / num_per_vv;
  uint32_t remain_vv = deal_num % num_per_vv;
  if (remain_vv) {
    repeat_vv++;
  }
  asm("vv.begin;\n\t");
  for (uint32_t vv_idx = 0; vv_idx < repeat_vv; vv_idx++) {
    uint32_t cur =
        (vv_idx == repeat_vv - 1 && remain_vv > 0) ? remain_vv : num_per_vv;
    for (int32_t i = 0; i < kHc * kHc; i++) {
      __vv_load(in_vv[i], input + i * deal_num + vv_idx * num_per_vv, cur);
    }
#pragma unroll 4
    for (int32_t i = 0; i < kHc; i++) {
      __vv_move(in_max[i], in_vv[i * kHc]);
#pragma unroll 4
      for (int32_t j = 1; j < kHc; j++) {
        __vv_max(in_max[i], in_max[i], in_vv[i * kHc + j]);
      }
      __vv_move(in_sum[i], 0);
      for (int32_t j = 0; j < kHc; j++) {
        __vv_sub(in_vv[i * kHc + j], in_vv[i * kHc + j], in_max[i]);
        __vv_mul(in_vv[i * kHc + j], in_vv[i * kHc + j], log2e);
        __vv_pow2(in_vv[i * kHc + j], in_vv[i * kHc + j]);
        __vv_add(in_sum[i], in_sum[i], in_vv[i * kHc + j]);
      }
      __vv_recip(in_sum[i], in_sum[i]);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_mul(in_vv[i * kHc + j], in_vv[i * kHc + j], in_sum[i]);
        __vv_add(in_vv[i * kHc + j], in_vv[i * kHc + j], eps);
      }
    }
#pragma unroll 4
    for (int32_t i = 0; i < kHc; i++) {
      __vv_move(in_sum[i], eps);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_add(in_sum[i], in_sum[i], in_vv[i + j * kHc]);
      }
      __vv_recip(in_sum[i], in_sum[i]);
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
        __vv_mul(in_vv[i + j * kHc], in_vv[i + j * kHc], in_sum[i]);
      }
    }
    for (uint32_t k = 0; k + 1 < iters; k++) {
#pragma unroll 4
      for (int32_t i = 0; i < kHc; i++) {
        __vv_move(in_sum[i], eps);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_add(in_sum[i], in_sum[i], in_vv[i * kHc + j]);
        }
      }
#pragma unroll 4
      for (int32_t i = 0; i < kHc; i++) {
        __vv_recip(in_sum[i], in_sum[i]);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_mul(in_vv[i * kHc + j], in_vv[i * kHc + j], in_sum[i]);
        }
      }
#pragma unroll 4
      for (int32_t i = 0; i < kHc; i++) {
        __vv_move(in_sum[i], eps);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_add(in_sum[i], in_sum[i], in_vv[i + j * kHc]);
        }
      }
#pragma unroll 4
      for (int32_t i = 0; i < kHc; i++) {
        __vv_recip(in_sum[i], in_sum[i]);
      }
#pragma unroll 4
      for (int32_t j = 0; j < kHc; j++) {
#pragma unroll 4
        for (int32_t i = 0; i < kHc; i++) {
          __vv_mul(in_vv[i + j * kHc], in_vv[i + j * kHc], in_sum[i]);
        }
      }
    }
    for (int32_t i = 0; i < kHc * kHc; i++) {
      __vv_store(output + i * deal_num + vv_idx * num_per_vv, in_vv[i], cur);
    }
  }
  asm("vv.end;\n\t");
}

// mixes [batch][24] (unscaled sums), rms [batch] -> comb [batch][16],
// pre/post [batch][4]. pre_out and post_out must be adjacent.
__mlu_func__ inline void hc_split_sinkhorn(float* comb_out,
                                           float* pre_out,
                                           float* post_out,
                                           float* mixes,
                                           const float* hc_scale,
                                           const float* hc_base,
                                           float* rms,
                                           float* buf,
                                           int32_t batch,
                                           int32_t iters,
                                           float hc_eps) {
  constexpr int32_t comb_dim = kHc * kHc;
  int32_t pre_n = batch * kHc;
  float* mixes_trans = buf;
  __bang_transpose(mixes_trans, mixes, batch, kMix);
  __bang_cycle_mul(mixes_trans, mixes_trans, rms, kMix * batch, batch);
  __bang_transpose(pre_out, mixes_trans, kHc, batch);
  __bang_transpose(post_out, mixes_trans + pre_n, kHc, batch);
  __bang_transpose(comb_out, mixes_trans + batch * 2 * kHc, comb_dim, batch);
  __bang_fusion(FUSION_FMA,
                pre_out,
                pre_out,
                hc_scale[0],
                const_cast<float*>(hc_base),
                pre_n,
                kHc);
  __bang_fusion(FUSION_FMA,
                post_out,
                post_out,
                hc_scale[1],
                const_cast<float*>(hc_base) + kHc,
                pre_n,
                kHc);
  __bang_fusion(FUSION_FMA,
                comb_out,
                comb_out,
                hc_scale[2],
                const_cast<float*>(hc_base) + 2 * kHc,
                batch * comb_dim,
                comb_dim);
  // pre_out and post_out are adjacent: one sigmoid pass covers both.
  apply_sigmoid(pre_out, batch * 2 * kHc);
  __bang_mul_scalar(post_out, post_out, 2.0f, pre_n);
  __bang_add_scalar(pre_out, pre_out, hc_eps, pre_n);
  float* comb_trans = buf;
  __bang_transpose(comb_trans, comb_out, batch, comb_dim);
  compute_sinkhorn_sequential(comb_trans, comb_trans, batch, iters, hc_eps);
  __bang_transpose(comb_out, comb_trans, comb_dim, batch);
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc::fused_mhc_detail
