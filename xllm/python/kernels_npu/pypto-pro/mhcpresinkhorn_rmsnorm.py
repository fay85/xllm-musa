#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mhcpresinkhorn_rmsnorm -- PyPTO-Pro implementation (v4-K: bf16 hc_fn + Cube/Vector overlap + single K-axis-split scheme).

Changes of v4 relative to v3 (realizing the avoidable items listed in the
"PyPTO-Pro framework-level overhead analysis report"):

  1. The hc_fn input becomes bf16 (the weight dtype conversion has been hoisted
     into the model side, the kernel no longer converts): the Cube runs the
     matmul directly as BF16xBF16->FP32, x no longer needs a cast to fp32, and
     the whole cast -> GM workspace -> L1 staging chain disappears (this avoids
     overhead item 1: the input is already bf16, so no AscendC-style TQue
     double buffering is needed to hide the cast either).
  2. The matmul is split along the K axis and load-balanced across all 28 cube
     cores:
       kstart = core*592 (27 full-block cores x 592 + a last core x 400 =
         16384), load-balanced over all 28 cube cores. K=16384 is sliced with a
         stride of 592 (kstart=core*592; 27 full-block cores x 592 + 1 last
         core x 400, both 16-aligned). Two tile shapes cover the whole load
         ([32,592]/[592,32] and [32,400]/[400,32], L0B totals 63.5KB), so there
         is no tail block and no zero padding; a single [16,K]x[K,32] matmul is
         issued, the 28 partials are stored in GM and Stage2 reduces them with
         pl.sum(dim=1).
         (Note: Mat tiles do not support fillpad (only Vec tiles do); the
          GM->L1 load in the cube section cannot use a bar barrier (both
          bar_mte1/2 generate PIPE_MTE1), so the zero-baseline scheme has a
          load race and was dropped in favour of the two tile shapes.)
  3. The matmul runs in parallel with RMSNorm: the Cube reads bf16 x straight
     from GM (it does not depend on any Vector-produced value) while the Vector
     computes ssq; each token group needs only 1 sync_all(MIX) (v3 needed 2).
     The Cube's mm(g+1) overlaps with the Vector's Stage2(g).
  4. VFs are unrolled per row and move 2xVL per load:
       ssq: a single DINTLV_B16 load_align moves 2*VL=256 bf16 (two rows of 128
            elements) and splits them into even/odd -> astype fp32 ->
            mul_add_dst FMA accumulation;
       y  : the HC=4 rows are explicitly unrolled with mul_add_dst FMA (a single
            multiply-add instruction, so the intermediate product is not
            truncated).
      Note: merging a pair of m rows into one load_align turns out to be
      infeasible after deriving the register lane geometry: after the bf16
      de-interleaving, the even/odd lane layout of the (m0,m1) row pair,
      [m0e|m1e]/[m0o|m1o], conflicts with the pre broadcast pattern, and the
      per-d summation would require adding across lanes (lane i with lane i+32),
      for which the VF has no primitive; so y keeps one load_align (128 bf16) +
      FMA per m row.

Changes of v5 relative to v4-K (perf round 6, driven by the cannsim AIV trace):

  5. Stage2 is re-mapped from "one token group per 16 of the 56 AIVs" to a
     token-flat map: inside a chunk, AIV gidx owns the contiguous window
     [jc + gidx*wc, +wc) with wc = ceil(chunk_t/56).  v4 only let the 16 AIVs
     with gidx < TG do the per-token work (40 of 56 AIVs idled through Stage2),
     which the trace showed as PUSHQ_EXEC busy for 156.7us of a 178.6us kernel
     on the busiest AIV.  Spreading the same instructions over all 56 AIVs is
     worth ~3.5x on the Vector side.
  6. The RMSNorm partial (ssq) is folded into the same token-flat loop: each
     AIV computes the complete int(HC*D) sum of squares of its own tokens
     (Phase 1, still overlapped with the Cube matmul) instead of a 512-wide
     slice of every group.  This removes the ssq staging tile, the
     [64,TG]->[TG,64] transpose, the GM round trip through ssq_ws and the
     32-way pl.sum (0.53us per token group on the old path, second largest
     Vector cost after Sinkhorn), and it removes the 16-row restriction of the
     ssq phase.
  7. The per-token r is handed from Phase 1 to Phase 2 through the r_ws
     workspace ([t,8] fp32). Phase 1 stores the complete per-token ssq;
     Phase 2 reads it before the separate rstd output is written, so the
     intermediate r remains available for every token without a UB-sized
     restriction.

Changes of v6 relative to v5 (Sinkhorn was 46% of the whole Vector time):

  8. The Sinkhorn runs once per batch of NT=32 tokens with lane = token instead
     of once per token with a single active lane.  Each token's 24 gating
     values are written as one row of a [NT,MIXC] batch tile, the batch is
     transposed to [MIXC,NT] (lane = token), and the 20-iteration chain then
     costs the same instruction count for the whole batch: measured 1.13us per
     token, i.e. 640 single-lane divisions each, becomes ~1.2us per batch.  The
     batch depth is nt = wc = ceil(chunk_t/56) <= TPC/NAIV = 10; NT = 32 is the
     tile capacity, not the batch size of a t >= 512 chunk.  The comb result is
     transposed back and stored as one [nt,16] block, which also replaces the
     16 strided single-lane stores per token.
  9. post is taken from lanes 4-7 of the gating vector by _y_pass1_vf (which
     already loads pre from that tile), so the 4 single-lane post stores of the
     old Sinkhorn disappear as well.
"""

import os
import threading
from collections import OrderedDict

import torch
import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

# ======================================================================
# Contract constants (same as SPEC and golden; except the hc_fn dtype --
# the model side already converts it to bf16)
# ======================================================================
HC = 4
D = 4096
MIX_HC = (2 + HC) * HC       # 24
HC_D = HC * D                # 16384
# Scalar reciprocals are hoisted out of the VF bodies: newer pypto parsers
# reject scalar division inside a vector function (check_vf_scalar_division),
# while a module-level float is a legal closure constant in both versions.
INV_D = 1.0 / D
INV_HC_D = 1.0 / HC_D
SINKHORN_ITERS = 20
EPS = 1e-6
RMS_EPS = 1e-6               # eps of the fused RmsNorm (same as production RmsNorm)

# ======================================================================
# Multi-core / tiling constants
# ======================================================================
NB = 28                      # MIX logical blocks (28 AIC + 56 AIV, all cube cores used)
TG = 16                      # pass = token group (the cube M dim)
MIXC = 32                    # gating output 24 -> padded to 32 (tile N width)
PWC = MIX_HC                 # 24: pmix3 workspace row width (unpadded -> the
                             # fixpipe writes contiguous rows and the Vector slab
                             # read becomes one contiguous block)
SGRP = 16                    # max tokens per partial-reduction group (>= wc)
DC = 512                     # d-chunk width of y (tile width of x_sub/y_ub)
NDC = D // DC                # 8: number of d-chunks
NAIV = 2 * NB                # 56: every AIC block carries two AIVs
NT = 32                      # Sinkhorn batch depth: one lane per token
GPCH = 32                    # Cube groups per barrier chunk (token overlap granularity)
TPC = GPCH * TG              # 512 tokens per chunk

# ---- K-axis split (kstart=core*KA; full-block cores 592, last core 400) ----
# See the L1/L0 layout below for details

# ======================================================================
# UB address layout (vector side, ~190KB span < 248KB; slots are spaced >=256B
# apart to prevent load_align from over-reading)
# ======================================================================
# 0x00000..0x05FFF is free since v5: the 16KB ssq staging tile and its two
# 4KB transpose buffers are gone (the ssq now runs on the whole token, in place)
SLAB_UB = 0x24000            # [NB,SGRP*PWC] fp32 43KB (28 partials x 16 tokens)
MIXR_UB = 0x2F000            # [1,SGRP*PWC] fp32 1.5KB (same group, reduced)
SSQR_UB = 0x06F00            # [1,8] fp32 32B (per-token r, read back from r_ws)
SCALE3_UB = 0x07780          # [1,8] fp32 32B
SCALE24_UB = 0x07800         # [1,32] fp32 128B
PMUL24_UB = 0x07900          # [1,32] fp32 128B
EPS24_UB = 0x07A00           # [1,32] fp32 128B
BASE_UB = 0x07B00            # [1,32] fp32 128B
ZBAT_UB = 0x07C00            # [NT,MIXC] fp32 4KB (gating values of one token batch)
ZSD_UB = 0x08C00             # [MIXC,NT] fp32 4KB (after transpose: lane = token)
CBBAT_UB = 0x09C00           # [16,NT] fp32 2KB (sinkhorn comb output, entry-major)
CBTT_UB = 0x0A400            # [NT,16] fp32 2KB (after transpose: one row per token)
PRE_UB = 0x0AC00             # [1,8] fp32 32B (lanes 0-3 pre, lanes 4-7 post)
POST_UB = 0x0AC40            # [1,8] fp32 32B (post of one token, for the GM store)
XSUB_UB = 0x0B000            # 2 slots [HC,D] bf16 32KBx2 (whole-token x, double buffered)
YUB_UB = 0x1B000             # 2 slots [1,D] bf16 8KBx2 (whole y_norm row, double buffered)
YROW_UB = 0x1F000            # [1,D] bf16 8KB (staging of a whole y row: written by pass1 / read by pass2)
GAMMA_UB = 0x21000           # [1,D] bf16 8KB (RmsNorm gamma, preloaded once per AIV)
RSTD_UB = 0x23000            # [1,64] fp32 256B (rstd broadcast, for the pass2 BRC)

# ---- L1/L0 (two tile shapes: full-block cores 592 / last core 400, full
# coverage with no tail block) ----
TGC = 32                      # Cube M per transaction: two 16-token groups
KA = 592                      # K slice of a full-block core (37x16); 27 cores x 592 + last core 400
KLAST = HC_D - (NB - 1) * KA  # K slice of the last core = 400 (25x16)
# The [TGC,*] shape and its last-core variant are only ever used by disjoint
# cores, so the two variants share one L1 base.  The full-block lhs tile is
# double buffered so the GM->L1 load of the next pair overlaps the
# MTE1/MAC/FIXP chain of the current one (aic mte2 49% + fixpipe 35% of
# aicore_time was a fully serialized chain).
LHS_SRC_L1 = 0x00000          # [TGC,KA] bf16 NZ 37.9KB (prefetch buffer A)
LHS_SRC_B_L1 = 0x09400        # [TGC,KA] bf16 NZ 37.9KB (prefetch buffer B)
LHS_SRC2_L1 = 0x00000         # [TGC,KLAST] bf16 NZ 25.6KB (same base, last core only)
RHS_SRC_L1 = 0x13000          # [KA,MIXC] bf16 ZN 37.9KB
RHS_SRC2_L1 = 0x13000         # [KLAST,MIXC] bf16 ZN 25.6KB (same base, last core only)


# ======================================================================
# VF sub-functions
# ======================================================================

@pl.vector_function
def _ssq_tok_vf(x_tok, r_out):
    """Whole-token sum of squares: x [HC,D] bf16 -> r_out = sum(x*x).

    Same DINTLV_B16 primitive as v4 (one load moves 2*VL=256 bf16 and splits
    them by even/odd index; the astype expansion to fp32 feeds mul_add_dst FMA,
    so a 256-element block costs 1 load + 8 vector ops), but the sweep covers
    the whole HC*D token instead of one 512-wide slice, which is what lets the
    ssq run in the same token-flat loop as the rest of Stage2.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for s in pl.range(HC_D // 256):
        ev, od = vf.load_align(x_tok, s * 256,
                               dist=pl.LoadDist.DINTLV_B16)
        e0 = vf.astype(ev, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e0, e0, preg)
        e1 = vf.astype(ev, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(e1, e1, preg)
        o0 = vf.astype(od, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(o0, o0, preg)
        o1 = vf.astype(od, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o1, o1, preg)
    sval = vf.reduce_sum(acc, preg)
    vf.store_align(r_out + 0, sval, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _build_aux_vf(scale3, scale24, pmul24, eps24):
    """Auxiliary vectors: scale24=[s0]*4+[s1]*4+[s2]*16; pmul24=[1]*4+[2]*4+[1]*16; eps24=[eps]*4+[0]*20."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg24 = vf.update_mask(24, dtype=pl.DT_FP32)
    sel4 = vf.update_mask(4, dtype=pl.DT_FP32)
    sel8 = vf.update_mask(8, dtype=pl.DT_FP32)
    s0 = vf.load_align(scale3, 0, dist=pl.LoadDist.BRC_B32)
    s1 = vf.load_align(scale3, 1, dist=pl.LoadDist.BRC_B32)
    s2 = vf.load_align(scale3, 2, dist=pl.LoadDist.BRC_B32)
    v1 = vf.select(s0, s1, sel4)
    sc = vf.select(v1, s2, sel8)
    vf.store_align(scale24, sc, preg24)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    two = vf.full(2.0, preg, dtype=pl.DT_FP32)
    u1 = vf.select(one, two, sel4)
    pm = vf.select(u1, one, sel8)
    vf.store_align(pmul24, pm, preg24)
    ee = vf.full(EPS, preg, dtype=pl.DT_FP32)
    zz = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ev = vf.select(ee, zz, sel4)
    vf.store_align(eps24, ev, preg24)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _gates1_vf(mixr, ssqr, scale24, pmul24, eps24, base,
               pp_ub, zbat, zoff, moff):
    """Single-token gating decomposition (lane = channel 0..23).

    mixr: lane moff..moff+23 = mixes[j,:] (one token of the reduced group);
    ssqr [1,8]: BRC(0) fetches r of this token.  Outputs: pp_ub [1,8]
    (lanes 0-3 = pre for the y BRC, lanes 4-7 = post; both are consumed by
    _y_pass1_vf) and zbat+zoff = the 24 gating values of this token, stored as
    one row of the Sinkhorn batch tile.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg24 = vf.update_mask(24, dtype=pl.DT_FP32)
    preg8 = vf.update_mask(8, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    scale_r = vf.load_align(scale24, 0)
    pmul_r = vf.load_align(pmul24, 0)
    eps_r = vf.load_align(eps24, 0)
    base_r = vf.load_align(base, 0)
    # r_j = 1/sqrt(ssq_j/16384 + eps)
    ssq_b = vf.load_align(ssqr, 0, dist=pl.LoadDist.BRC_B32)
    m1 = vf.muls(ssq_b, INV_HC_D, preg)
    m2 = vf.adds(m1, EPS, preg)
    m3 = vf.sqrt(m2, preg)
    rj = vf.div(one, m3, preg)
    # z = mix * r * scale + base
    mix_r = vf.load_align(mixr, moff)
    z1 = vf.mul(mix_r, scale_r, preg24)
    z2 = vf.mul(rj, z1, preg24)
    z = vf.add(z2, base_r, preg24)
    # sigmoid = 1/(1+exp(-z))
    e = vf.exp_sub(zero, z, preg24)
    d = vf.adds(e, 1.0, preg24)
    sig = vf.div(one, d, preg24)
    # Piecewise: lane0-7 sigmoid, lane8-23 linear
    pp0 = vf.select(sig, z, preg8)
    pp1 = vf.mul(pp0, pmul_r, preg24)
    pp = vf.add(pp1, eps_r, preg24)
    # Row of the Sinkhorn batch tile (24 gating values, contiguous)
    vf.store_align(zbat + zoff, pp, preg24)
    # pre (lanes 0-3) + post (lanes 4-7) for the y pass
    vf.store_align(pp_ub, pp, preg8)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _y_pass1_vf(pre_ub, x_tok, y_row, rstd_t, post_ub):
    """y pass1 (whole token in one VF): y = sum_m pre[m]*x[m,:], yssq += y^2.

    Merged from the former per-chunk VF (8 calls/token).  The masks, the four
    pre BRC loads and the yssq accumulator are set up once per token and reused
    by the NDC chunk loop, so the VF is instruction-dense.  x_tok is [HC, D];
    chunk c covers columns [c*DC, (c+1)*DC); row m starts at offset m*D.

    The tail also copies lanes 4-7 of pre_ub (= post) into post_ub, so the
    per-token post output no longer needs the 4 strided single-lane stores the
    old Sinkhorn VF issued.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p0 = vf.load_align(pre_ub, 0, dist=pl.LoadDist.BRC_B32)
    p1 = vf.load_align(pre_ub, 1, dist=pl.LoadDist.BRC_B32)
    p2 = vf.load_align(pre_ub, 2, dist=pl.LoadDist.BRC_B32)
    p3 = vf.load_align(pre_ub, 3, dist=pl.LoadDist.BRC_B32)
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)

    for c in pl.range(NDC):
        base = c * DC
        b = vf.load_align(x_tok, base + 0)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye0 = vf.mul(e, p0, preg)
        yo0 = vf.mul(o, p0, preg)
        b = vf.load_align(x_tok, base + 128)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye1 = vf.mul(e, p0, preg)
        yo1 = vf.mul(o, p0, preg)
        b = vf.load_align(x_tok, base + 256)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye2 = vf.mul(e, p0, preg)
        yo2 = vf.mul(o, p0, preg)
        b = vf.load_align(x_tok, base + 384)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye3 = vf.mul(e, p0, preg)
        yo3 = vf.mul(o, p0, preg)
        b = vf.load_align(x_tok, 1 * D + base + 0)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye0 = vf.mul_add_dst(e, p1, preg)
        yo0 = vf.mul_add_dst(o, p1, preg)
        b = vf.load_align(x_tok, 1 * D + base + 128)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye1 = vf.mul_add_dst(e, p1, preg)
        yo1 = vf.mul_add_dst(o, p1, preg)
        b = vf.load_align(x_tok, 1 * D + base + 256)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye2 = vf.mul_add_dst(e, p1, preg)
        yo2 = vf.mul_add_dst(o, p1, preg)
        b = vf.load_align(x_tok, 1 * D + base + 384)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye3 = vf.mul_add_dst(e, p1, preg)
        yo3 = vf.mul_add_dst(o, p1, preg)
        b = vf.load_align(x_tok, 2 * D + base + 0)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye0 = vf.mul_add_dst(e, p2, preg)
        yo0 = vf.mul_add_dst(o, p2, preg)
        b = vf.load_align(x_tok, 2 * D + base + 128)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye1 = vf.mul_add_dst(e, p2, preg)
        yo1 = vf.mul_add_dst(o, p2, preg)
        b = vf.load_align(x_tok, 2 * D + base + 256)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye2 = vf.mul_add_dst(e, p2, preg)
        yo2 = vf.mul_add_dst(o, p2, preg)
        b = vf.load_align(x_tok, 2 * D + base + 384)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye3 = vf.mul_add_dst(e, p2, preg)
        yo3 = vf.mul_add_dst(o, p2, preg)
        b = vf.load_align(x_tok, 3 * D + base + 0)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye0 = vf.mul_add_dst(e, p3, preg)
        yo0 = vf.mul_add_dst(o, p3, preg)
        b = vf.load_align(x_tok, 3 * D + base + 128)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye1 = vf.mul_add_dst(e, p3, preg)
        yo1 = vf.mul_add_dst(o, p3, preg)
        b = vf.load_align(x_tok, 3 * D + base + 256)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye2 = vf.mul_add_dst(e, p3, preg)
        yo2 = vf.mul_add_dst(o, p3, preg)
        b = vf.load_align(x_tok, 3 * D + base + 384)
        e = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ZERO)
        o = vf.astype(b, preg_bf, dtype=pl.DT_FP32,
                      layout=pl.CastLayout.ONE)
        ye3 = vf.mul_add_dst(e, p3, preg)
        yo3 = vf.mul_add_dst(o, p3, preg)
        a = vf.astype(ye0, preg, dtype=pl.DT_BF16,
                      layout=pl.CastLayout.ZERO)
        cc = vf.astype(yo0, preg, dtype=pl.DT_BF16,
                       layout=pl.CastLayout.ONE)
        pk0 = vf.add(a, cc, preg_bf)
        vf.store_align(y_row + base + 0, pk0, preg_bf)
        e2 = vf.astype(pk0, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e2, e2, preg)
        o2 = vf.astype(pk0, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o2, o2, preg)
        a = vf.astype(ye1, preg, dtype=pl.DT_BF16,
                      layout=pl.CastLayout.ZERO)
        cc = vf.astype(yo1, preg, dtype=pl.DT_BF16,
                       layout=pl.CastLayout.ONE)
        pk1 = vf.add(a, cc, preg_bf)
        vf.store_align(y_row + base + 128, pk1, preg_bf)
        e2 = vf.astype(pk1, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e2, e2, preg)
        o2 = vf.astype(pk1, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o2, o2, preg)
        a = vf.astype(ye2, preg, dtype=pl.DT_BF16,
                      layout=pl.CastLayout.ZERO)
        cc = vf.astype(yo2, preg, dtype=pl.DT_BF16,
                       layout=pl.CastLayout.ONE)
        pk2 = vf.add(a, cc, preg_bf)
        vf.store_align(y_row + base + 256, pk2, preg_bf)
        e2 = vf.astype(pk2, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e2, e2, preg)
        o2 = vf.astype(pk2, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o2, o2, preg)
        a = vf.astype(ye3, preg, dtype=pl.DT_BF16,
                      layout=pl.CastLayout.ZERO)
        cc = vf.astype(yo3, preg, dtype=pl.DT_BF16,
                       layout=pl.CastLayout.ONE)
        pk3 = vf.add(a, cc, preg_bf)
        vf.store_align(y_row + base + 384, pk3, preg_bf)
        e2 = vf.astype(pk3, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e2, e2, preg)
        o2 = vf.astype(pk3, preg_bf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o2, o2, preg)

    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    sval = vf.reduce_sum(acc, preg)
    m1 = vf.muls(sval, INV_D, preg)
    m2 = vf.adds(m1, RMS_EPS, preg)
    m3 = vf.sqrt(m2, preg)
    rr = vf.div(one, m3, preg)
    vf.store_align(rstd_t + 0, rr, preg)
    # post = lanes HC..2*HC-1 of pre_ub, moved down to lanes 0..HC-1.  A plain
    # load at that offset would be 16B-aligned (RV_VLDS needs 32B), so the four
    # values are broadcast-loaded (BRC tolerates any 4B offset) and re-packed
    # with three selects.
    preg1 = vf.update_mask(1, dtype=pl.DT_FP32)
    preg2 = vf.update_mask(2, dtype=pl.DT_FP32)
    preg3 = vf.update_mask(3, dtype=pl.DT_FP32)
    preg4 = vf.update_mask(4, dtype=pl.DT_FP32)
    q0 = vf.load_align(pre_ub, HC + 0, dist=pl.LoadDist.BRC_B32)
    q1 = vf.load_align(pre_ub, HC + 1, dist=pl.LoadDist.BRC_B32)
    q2 = vf.load_align(pre_ub, HC + 2, dist=pl.LoadDist.BRC_B32)
    q3 = vf.load_align(pre_ub, HC + 3, dist=pl.LoadDist.BRC_B32)
    po = vf.select(q0, q1, preg1)
    po = vf.select(po, q2, preg2)
    po = vf.select(po, q3, preg3)
    vf.store_align(post_ub, po, preg4)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _y_norm_vf(y_row, gamma, rstd_t, y_tok):
    """y pass2 (whole token): y_norm = y * rstd * gamma -> bf16.

    Merged from the former per-chunk VF (8 calls/token): masks and the rstd
    BRC are set up once and reused by all 32 128-wide segments.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    r = vf.load_align(rstd_t, 0, dist=pl.LoadDist.BRC_B32)
    for k in pl.range(D // 128):
        off = k * 128
        pk = vf.load_align(y_row, off)
        e = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        o = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        g = vf.load_align(gamma, off)
        ge = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        go = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        ne = vf.mul(e, r, preg)
        ne = vf.mul(ne, ge, preg)
        no = vf.mul(o, r, preg)
        no = vf.mul(no, go, preg)
        a = vf.astype(ne, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
        c = vf.astype(no, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
        pk = vf.add(a, c, preg_bf)
        vf.store_align(y_tok + off, pk, preg_bf)


@pl.vector_function
def _zero_zbat_vf(zbat):
    """Zero the Sinkhorn batch tile once (lanes of tokens past t stay finite)."""
    preg = vf.update_mask(MIXC, dtype=pl.DT_FP32)
    zz = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for r in pl.range(NT):
        vf.store_align(zbat + r * MIXC, zz, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _sinkhorn_batch_vf(zsd, cbbat, nt):
    """Sinkhorn for one batch of nt tokens: lane = token.

    zsd [MIXC,NT] rows 2*HC..3*HC-1 hold comb_flag of tokens 0..nt-1
    (one token per lane), so every one of the 16 4x4 entries is a
    single register and the whole 20-iteration chain costs the same
    instruction count as one token did before: the 640 divisions of
    v4 become 640 divisions *per batch* (nt <= TPC/NAIV = 10 tokens;
    NT=32 is only the tile capacity).  cbbat [16,NT] receives the
    result transposed (entry-major), so the caller only stores the
    [NT,16] transpose of the batch.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_nt = vf.update_mask(nt, dtype=pl.DT_FP32)

    z00 = vf.load_align(zsd, (2 * HC + 0) * NT)
    z01 = vf.load_align(zsd, (2 * HC + 1) * NT)
    z02 = vf.load_align(zsd, (2 * HC + 2) * NT)
    z03 = vf.load_align(zsd, (2 * HC + 3) * NT)
    z10 = vf.load_align(zsd, (2 * HC + 4) * NT)
    z11 = vf.load_align(zsd, (2 * HC + 5) * NT)
    z12 = vf.load_align(zsd, (2 * HC + 6) * NT)
    z13 = vf.load_align(zsd, (2 * HC + 7) * NT)
    z20 = vf.load_align(zsd, (2 * HC + 8) * NT)
    z21 = vf.load_align(zsd, (2 * HC + 9) * NT)
    z22 = vf.load_align(zsd, (2 * HC + 10) * NT)
    z23 = vf.load_align(zsd, (2 * HC + 11) * NT)
    z30 = vf.load_align(zsd, (2 * HC + 12) * NT)
    z31 = vf.load_align(zsd, (2 * HC + 13) * NT)
    z32 = vf.load_align(zsd, (2 * HC + 14) * NT)
    z33 = vf.load_align(zsd, (2 * HC + 15) * NT)

    qa0 = vf.max(z00, z01, preg_nt)
    qb0 = vf.max(z02, z03, preg_nt)
    rmx0 = vf.max(qa0, qb0, preg_nt)
    qa1 = vf.max(z10, z11, preg_nt)
    qb1 = vf.max(z12, z13, preg_nt)
    rmx1 = vf.max(qa1, qb1, preg_nt)
    qa2 = vf.max(z20, z21, preg_nt)
    qb2 = vf.max(z22, z23, preg_nt)
    rmx2 = vf.max(qa2, qb2, preg_nt)
    qa3 = vf.max(z30, z31, preg_nt)
    qb3 = vf.max(z32, z33, preg_nt)
    rmx3 = vf.max(qa3, qb3, preg_nt)

    e00 = vf.exp_sub(z00, rmx0, preg_nt)
    e01 = vf.exp_sub(z01, rmx0, preg_nt)
    e02 = vf.exp_sub(z02, rmx0, preg_nt)
    e03 = vf.exp_sub(z03, rmx0, preg_nt)
    e10 = vf.exp_sub(z10, rmx1, preg_nt)
    e11 = vf.exp_sub(z11, rmx1, preg_nt)
    e12 = vf.exp_sub(z12, rmx1, preg_nt)
    e13 = vf.exp_sub(z13, rmx1, preg_nt)
    e20 = vf.exp_sub(z20, rmx2, preg_nt)
    e21 = vf.exp_sub(z21, rmx2, preg_nt)
    e22 = vf.exp_sub(z22, rmx2, preg_nt)
    e23 = vf.exp_sub(z23, rmx2, preg_nt)
    e30 = vf.exp_sub(z30, rmx3, preg_nt)
    e31 = vf.exp_sub(z31, rmx3, preg_nt)
    e32 = vf.exp_sub(z32, rmx3, preg_nt)
    e33 = vf.exp_sub(z33, rmx3, preg_nt)

    sa0 = vf.add(e00, e01, preg_nt)
    sb0 = vf.add(e02, e03, preg_nt)
    rs0 = vf.add(sa0, sb0, preg_nt)
    sa1 = vf.add(e10, e11, preg_nt)
    sb1 = vf.add(e12, e13, preg_nt)
    rs1 = vf.add(sa1, sb1, preg_nt)
    sa2 = vf.add(e20, e21, preg_nt)
    sb2 = vf.add(e22, e23, preg_nt)
    rs2 = vf.add(sa2, sb2, preg_nt)
    sa3 = vf.add(e30, e31, preg_nt)
    sb3 = vf.add(e32, e33, preg_nt)
    rs3 = vf.add(sa3, sb3, preg_nt)

    dv00 = vf.div(e00, rs0, preg_nt)
    c00 = vf.adds(dv00, EPS, preg_nt)
    dv01 = vf.div(e01, rs0, preg_nt)
    c01 = vf.adds(dv01, EPS, preg_nt)
    dv02 = vf.div(e02, rs0, preg_nt)
    c02 = vf.adds(dv02, EPS, preg_nt)
    dv03 = vf.div(e03, rs0, preg_nt)
    c03 = vf.adds(dv03, EPS, preg_nt)
    dv10 = vf.div(e10, rs1, preg_nt)
    c10 = vf.adds(dv10, EPS, preg_nt)
    dv11 = vf.div(e11, rs1, preg_nt)
    c11 = vf.adds(dv11, EPS, preg_nt)
    dv12 = vf.div(e12, rs1, preg_nt)
    c12 = vf.adds(dv12, EPS, preg_nt)
    dv13 = vf.div(e13, rs1, preg_nt)
    c13 = vf.adds(dv13, EPS, preg_nt)
    dv20 = vf.div(e20, rs2, preg_nt)
    c20 = vf.adds(dv20, EPS, preg_nt)
    dv21 = vf.div(e21, rs2, preg_nt)
    c21 = vf.adds(dv21, EPS, preg_nt)
    dv22 = vf.div(e22, rs2, preg_nt)
    c22 = vf.adds(dv22, EPS, preg_nt)
    dv23 = vf.div(e23, rs2, preg_nt)
    c23 = vf.adds(dv23, EPS, preg_nt)
    dv30 = vf.div(e30, rs3, preg_nt)
    c30 = vf.adds(dv30, EPS, preg_nt)
    dv31 = vf.div(e31, rs3, preg_nt)
    c31 = vf.adds(dv31, EPS, preg_nt)
    dv32 = vf.div(e32, rs3, preg_nt)
    c32 = vf.adds(dv32, EPS, preg_nt)
    dv33 = vf.div(e33, rs3, preg_nt)
    c33 = vf.adds(dv33, EPS, preg_nt)

    ua0 = vf.add(c00, c10, preg_nt)
    ub0 = vf.add(c20, c30, preg_nt)
    wq0 = vf.add(ua0, ub0, preg_nt)
    cs0 = vf.adds(wq0, EPS, preg_nt)
    ua1 = vf.add(c01, c11, preg_nt)
    ub1 = vf.add(c21, c31, preg_nt)
    wq1 = vf.add(ua1, ub1, preg_nt)
    cs1 = vf.adds(wq1, EPS, preg_nt)
    ua2 = vf.add(c02, c12, preg_nt)
    ub2 = vf.add(c22, c32, preg_nt)
    wq2 = vf.add(ua2, ub2, preg_nt)
    cs2 = vf.adds(wq2, EPS, preg_nt)
    ua3 = vf.add(c03, c13, preg_nt)
    ub3 = vf.add(c23, c33, preg_nt)
    wq3 = vf.add(ua3, ub3, preg_nt)
    cs3 = vf.adds(wq3, EPS, preg_nt)

    c00 = vf.div(c00, cs0, preg_nt)
    c01 = vf.div(c01, cs1, preg_nt)
    c02 = vf.div(c02, cs2, preg_nt)
    c03 = vf.div(c03, cs3, preg_nt)
    c10 = vf.div(c10, cs0, preg_nt)
    c11 = vf.div(c11, cs1, preg_nt)
    c12 = vf.div(c12, cs2, preg_nt)
    c13 = vf.div(c13, cs3, preg_nt)
    c20 = vf.div(c20, cs0, preg_nt)
    c21 = vf.div(c21, cs1, preg_nt)
    c22 = vf.div(c22, cs2, preg_nt)
    c23 = vf.div(c23, cs3, preg_nt)
    c30 = vf.div(c30, cs0, preg_nt)
    c31 = vf.div(c31, cs1, preg_nt)
    c32 = vf.div(c32, cs2, preg_nt)
    c33 = vf.div(c33, cs3, preg_nt)

    for _ in pl.range(0, SINKHORN_ITERS - 1):
        aa0 = vf.add(c00, c01, preg_nt)
        ab0 = vf.add(c02, c03, preg_nt)
        wr0 = vf.add(aa0, ab0, preg_nt)
        rs0 = vf.adds(wr0, EPS, preg_nt)
        aa1 = vf.add(c10, c11, preg_nt)
        ab1 = vf.add(c12, c13, preg_nt)
        wr1 = vf.add(aa1, ab1, preg_nt)
        rs1 = vf.adds(wr1, EPS, preg_nt)
        aa2 = vf.add(c20, c21, preg_nt)
        ab2 = vf.add(c22, c23, preg_nt)
        wr2 = vf.add(aa2, ab2, preg_nt)
        rs2 = vf.adds(wr2, EPS, preg_nt)
        aa3 = vf.add(c30, c31, preg_nt)
        ab3 = vf.add(c32, c33, preg_nt)
        wr3 = vf.add(aa3, ab3, preg_nt)
        rs3 = vf.adds(wr3, EPS, preg_nt)
        c00 = vf.div(c00, rs0, preg_nt)
        c01 = vf.div(c01, rs0, preg_nt)
        c02 = vf.div(c02, rs0, preg_nt)
        c03 = vf.div(c03, rs0, preg_nt)
        c10 = vf.div(c10, rs1, preg_nt)
        c11 = vf.div(c11, rs1, preg_nt)
        c12 = vf.div(c12, rs1, preg_nt)
        c13 = vf.div(c13, rs1, preg_nt)
        c20 = vf.div(c20, rs2, preg_nt)
        c21 = vf.div(c21, rs2, preg_nt)
        c22 = vf.div(c22, rs2, preg_nt)
        c23 = vf.div(c23, rs2, preg_nt)
        c30 = vf.div(c30, rs3, preg_nt)
        c31 = vf.div(c31, rs3, preg_nt)
        c32 = vf.div(c32, rs3, preg_nt)
        c33 = vf.div(c33, rs3, preg_nt)
        ba0 = vf.add(c00, c10, preg_nt)
        bb0 = vf.add(c20, c30, preg_nt)
        wc0 = vf.add(ba0, bb0, preg_nt)
        cs0 = vf.adds(wc0, EPS, preg_nt)
        ba1 = vf.add(c01, c11, preg_nt)
        bb1 = vf.add(c21, c31, preg_nt)
        wc1 = vf.add(ba1, bb1, preg_nt)
        cs1 = vf.adds(wc1, EPS, preg_nt)
        ba2 = vf.add(c02, c12, preg_nt)
        bb2 = vf.add(c22, c32, preg_nt)
        wc2 = vf.add(ba2, bb2, preg_nt)
        cs2 = vf.adds(wc2, EPS, preg_nt)
        ba3 = vf.add(c03, c13, preg_nt)
        bb3 = vf.add(c23, c33, preg_nt)
        wc3 = vf.add(ba3, bb3, preg_nt)
        cs3 = vf.adds(wc3, EPS, preg_nt)
        c00 = vf.div(c00, cs0, preg_nt)
        c01 = vf.div(c01, cs1, preg_nt)
        c02 = vf.div(c02, cs2, preg_nt)
        c03 = vf.div(c03, cs3, preg_nt)
        c10 = vf.div(c10, cs0, preg_nt)
        c11 = vf.div(c11, cs1, preg_nt)
        c12 = vf.div(c12, cs2, preg_nt)
        c13 = vf.div(c13, cs3, preg_nt)
        c20 = vf.div(c20, cs0, preg_nt)
        c21 = vf.div(c21, cs1, preg_nt)
        c22 = vf.div(c22, cs2, preg_nt)
        c23 = vf.div(c23, cs3, preg_nt)
        c30 = vf.div(c30, cs0, preg_nt)
        c31 = vf.div(c31, cs1, preg_nt)
        c32 = vf.div(c32, cs2, preg_nt)
        c33 = vf.div(c33, cs3, preg_nt)

    vf.store_align(cbbat + 0, c00, preg_nt)
    vf.store_align(cbbat + 32, c01, preg_nt)
    vf.store_align(cbbat + 64, c02, preg_nt)
    vf.store_align(cbbat + 96, c03, preg_nt)
    vf.store_align(cbbat + 128, c10, preg_nt)
    vf.store_align(cbbat + 160, c11, preg_nt)
    vf.store_align(cbbat + 192, c12, preg_nt)
    vf.store_align(cbbat + 224, c13, preg_nt)
    vf.store_align(cbbat + 256, c20, preg_nt)
    vf.store_align(cbbat + 288, c21, preg_nt)
    vf.store_align(cbbat + 320, c22, preg_nt)
    vf.store_align(cbbat + 352, c23, preg_nt)
    vf.store_align(cbbat + 384, c30, preg_nt)
    vf.store_align(cbbat + 416, c31, preg_nt)
    vf.store_align(cbbat + 448, c32, preg_nt)
    vf.store_align(cbbat + 480, c33, preg_nt)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

# ======================================================================
# Common tile declarations (vector side, layout shared by both kernels)
# ======================================================================

def _make_vector_tiles():
    slab = pl.make_tile_group(
        type=pl.TileType(shape=[NB, SGRP * PWC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=SLAB_UB, mutex_ids=[12])
    mixr = pl.make_tile_group(
        type=pl.TileType(shape=[1, SGRP * PWC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=MIXR_UB, mutex_ids=[13])
    qr = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SSQR_UB, mutex_ids=[15])
    sc3 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SCALE3_UB, mutex_ids=[16])
    sc24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=SCALE24_UB, mutex_ids=[16])
    pm24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=PMUL24_UB, mutex_ids=[16])
    ep24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=EPS24_UB, mutex_ids=[16])
    base = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=BASE_UB, mutex_ids=[16])
    zbat = pl.make_tile_group(
        type=pl.TileType(shape=[NT, MIXC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=ZBAT_UB, mutex_ids=[21])
    zsd = pl.make_tile_group(
        type=pl.TileType(shape=[MIXC, NT], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=ZSD_UB, mutex_ids=[22])
    cbbat = pl.make_tile_group(
        type=pl.TileType(shape=[16, NT], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=CBBAT_UB, mutex_ids=[23])
    cbtt = pl.make_tile_group(
        type=pl.TileType(shape=[NT, 16], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=CBTT_UB, mutex_ids=[24])
    pre = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=PRE_UB, mutex_ids=[27])
    postb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=POST_UB, mutex_ids=[19])
    xsub = pl.make_tile_group(
        type=pl.TileType(shape=[HC, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=XSUB_UB, mutex_ids=[28, 29])
    yub = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=YUB_UB, mutex_ids=[30, 31])
    yrow = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=YROW_UB, mutex_ids=[17])
    gam = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=GAMMA_UB, mutex_ids=[16])
    rstdg = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=RSTD_UB, mutex_ids=[18])
    return (slab, mixr, qr, sc3, sc24, pm24, ep24, base,
            zbat, zsd, cbbat, cbtt, pre, postb, xsub, yub, yrow, gam, rstdg)


# ======================================================================
# Kernel: K-axis split (27x592 + 400 on the last core), load balanced over
# all 28 cube cores
# ======================================================================

@pl.jit(auto_mutex=True)
def mhcpresinkhorn_rmsnorm_kernel_k(
    x_2d:    pl.Tensor[[pl.DYNAMIC, HC_D], pl.DT_BF16],
    x_3d:    pl.Tensor[[pl.DYNAMIC, HC, D], pl.DT_BF16],
    hc_fn:   pl.Tensor[[MIX_HC, HC_D], pl.DT_BF16],
    hc_scale: pl.Tensor[[1, 3], pl.DT_FP32],
    hc_base:  pl.Tensor[[1, MIX_HC], pl.DT_FP32],
    gamma:    pl.Tensor[[1, D], pl.DT_BF16],
    y:        pl.Tensor[[pl.DYNAMIC, D], pl.DT_BF16],
    rstd:     pl.Tensor[[pl.DYNAMIC, 1], pl.DT_FP32],
    post:     pl.Tensor[[pl.DYNAMIC, HC], pl.DT_FP32],
    comb16:   pl.Tensor[[pl.DYNAMIC, MIX_HC - 8], pl.DT_FP32],
    pmix3:    pl.Tensor[[NB, pl.DYNAMIC, PWC], pl.DT_FP32],
    pmix3f:   pl.Tensor[[NB, pl.DYNAMIC], pl.DT_FP32],
    r_ws:     pl.Tensor[[pl.DYNAMIC, 8], pl.DT_FP32],
):
    t = x_2d.shape[0]
    ng = (t + TG - 1) // TG
    nch = (ng + GPCH - 1) // GPCH      # barrier chunks (see the Cube/Vector loops)

    # ---- Cube tiles (two shapes: full-block cores [TGC,592]/[592,32], last
    #      core [TGC,400]/[400,32]; fully covered load with no tail block.
    #      L0B totals 63.5KB <= 64KB. Note: fillpad is not supported for Mat
    #      tiles (Vec only), and the GM->L1 load cannot use a bar barrier
    #      either (the generated code lowers both bar_mte1/2 in the cube
    #      section to PIPE_MTE1), hence the zero-baseline approach is not
    #      used) ----
    # Two explicit L1 buffers so the Cube loop can prefetch pair g+1 while
    # pair g is still in the MTE1/MAC/FIXP chain (manual software pipeline).
    lhs_a = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, KA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=LHS_SRC_L1, mutex_ids=[0])
    lhs_b = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, KA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=LHS_SRC_B_L1, mutex_ids=[1])
    lhs_src2 = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, KLAST], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=LHS_SRC2_L1, mutex_ids=[0])
    rhs_src = pl.make_tile_group(
        type=pl.TileType(shape=[KA, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                         valid_shape=[-1, -1]),
        addrs=RHS_SRC_L1, mutex_ids=[2])
    rhs_src2 = pl.make_tile_group(
        type=pl.TileType(shape=[KLAST, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                         valid_shape=[-1, -1]),
        addrs=RHS_SRC2_L1, mutex_ids=[2])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, KA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    l0a2 = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, KLAST], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x9800, mutex_ids=[5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[KA, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[6])
    l0b2 = pl.make_tile_group(
        type=pl.TileType(shape=[KLAST, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x9800, mutex_ids=[7])
    l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TGC, MIXC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=0x0, mutex_ids=[8, 9, 10, 11])
    (slab, mixr, qr, sc3, sc24, pm24, ep24, base,
     zbat, zsd, cbbat, cbtt, pre, postb, xsub, yub, yrow, gam, rstdg) =         _make_vector_tiles()

    # ========== Cube: K-split matmul (runs in parallel with the Vector ssq) ==========
    with pl.section_cube():
        core = pl.get_block_idx()
        kstart = core * KA
        la = lhs_a.current()
        lb = lhs_b.current()
        lsrc2 = lhs_src2.current()
        rsrc = rhs_src.current()
        rsrc2 = rhs_src2.current()
        a_t = l0a.current()
        a_t2 = l0a2.current()
        b_t = l0b.current()
        b_t2 = l0b2.current()
        if core < NB - 1:
            # ---- Full-block core (K=592): fully covered, no tail block ----
            pl.set_validshape(rsrc, [KA, MIX_HC])
            pl.load(rsrc, hc_fn, [0, kstart], order=[1, 0])
            pl.move(b_t, rsrc)
        else:
            # ---- Last core (K=400): full coverage with the small tile ----
            pl.set_validshape(rsrc2, [KLAST, MIX_HC])
            pl.load(rsrc2, hc_fn, [0, kstart], order=[1, 0])
            pl.move(b_t2, rsrc2)
        # One barrier per chunk (outside the core-dependent branch, so every
        # core issues exactly nch of them), which lets the Vector run the
        # epilogue of chunk c while the Cube still computes chunk c+1.
        for c in pl.range(0, nch):
            gend = pl.min(ng, (c + 1) * GPCH)
            if core < NB - 1:
                pf = c * GPCH // 2
                pe = (gend + 1) // 2
                # Software pipeline: the GM->L1 load of the next pair is issued
                # before the MTE1/MAC/FIXP chain of the current one, so the
                # 0.74us load and the 1.0us chain overlap.  A/B buffers
                # alternate inside one unrolled step of two pairs.
                pl.set_validshape(la, [pl.min(TGC, t - pf * TGC), KA])
                pl.load(la, x_2d, [pf * TGC, kstart])
                for h in pl.range((pe - pf + 1) // 2):
                    ga = pf + 2 * h
                    gb = ga + 1
                    # ---- pair ga out of A, prefetch gb into B ----
                    if gb < pe:
                        pl.set_validshape(lb, [pl.min(TGC, t - gb * TGC), KA])
                        pl.load(lb, x_2d, [gb * TGC, kstart])
                    ca = l0c.next()
                    pl.move(a_t, la)
                    pl.matmul(ca, a_t, b_t)
                    pl.set_validshape(ca, [pl.min(TGC, t - ga * TGC), MIX_HC])
                    pl.store(pmix3, ca, [core, ga * TGC, 0])
                    # ---- pair gb out of B, prefetch the next A ----
                    if gb < pe:
                        if gb + 1 < pe:
                            pl.set_validshape(
                                la, [pl.min(TGC, t - (gb + 1) * TGC), KA])
                            pl.load(la, x_2d, [(gb + 1) * TGC, kstart])
                        cb = l0c.next()
                        pl.move(a_t, lb)
                        pl.matmul(cb, a_t, b_t)
                        pl.set_validshape(cb, [pl.min(TGC, t - gb * TGC), MIX_HC])
                        pl.store(pmix3, cb, [core, gb * TGC, 0])
            else:
                for g2 in pl.range(c * GPCH // 2, (gend + 1) // 2):
                    tg = pl.min(TGC, t - g2 * TGC)
                    c_t = l0c.next()
                    pl.set_validshape(lsrc2, [tg, KLAST])
                    pl.load(lsrc2, x_2d, [g2 * TGC, kstart])
                    pl.move(a_t2, lsrc2)
                    pl.matmul(c_t, a_t2, b_t2)
                    pl.set_validshape(c_t, [tg, MIX_HC])
                    pl.store(pmix3, c_t, [core, g2 * TGC, 0])
            # NOTE: mode= is omitted on purpose: pypto 0.3.0 (CANN 9.2.0 wheel)
            # defaults mode to SyncAllMode.HARD, while newer pypto builds removed
            # pl.SyncAllMode entirely and take only core_type.  The core_type-only
            # call is the single form both versions accept and keeps the hard
            # (FFTS-signal, workspace-free) barrier on the old build.
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    # ========== Vector: token-flat ssq || Cube, then token-flat Stage2 ==========
    with pl.section_vector():
        gidx = pl.get_block_idx()
        sl_t = slab.current()
        mr_t = mixr.current()
        qr_t = qr.current()
        sc3_t = sc3.current()
        sc24_t = sc24.current()
        pm24_t = pm24.current()
        ep24_t = ep24.current()
        base_tile = base.current()
        zb_t = zbat.current()
        zd_t = zsd.current()
        cbb_t = cbbat.current()
        cbt_t = cbtt.current()
        pr_t = pre.current()
        pb_t = postb.current()
        yr_t = yrow.current()
        gam_t = gam.current()
        rt_t = rstdg.current()

        # Build aux vectors (once per AIV)
        pl.set_validshape(sc3_t, [1, 3])
        pl.load(sc3_t, hc_scale, [0, 0])
        pl.set_validshape(base_tile, [1, MIX_HC])
        pl.load(base_tile, hc_base, [0, 0])
        _build_aux_vf(sc3_t, sc24_t, pm24_t, ep24_t)
        # Preload the RmsNorm gamma (token invariant, once per AIV)
        pl.set_validshape(gam_t, [1, D])
        pl.load(gam_t, gamma, [0, 0])

        # ---- per-chunk pipeline: ssq(chunk c) -> barrier c -> epilogue(c) ----
        # Chunk c owns the tokens [c*TPC, min(t, (c+1)*TPC)); the Cube raises
        # barrier c only after it has written the matching pmix3 groups, so the
        # epilogue of chunk c runs while the Cube computes chunk c+1.  Barrier
        # counts stay 1:1 (nch on both sides) and both sides issue every
        # barrier unconditionally.
        _zero_zbat_vf(zb_t)
        for c in pl.range(0, nch):
            jc = c * TPC
            tc = pl.min(TPC, t - jc)
            wc = (tc + NAIV - 1) // NAIV
            jend = jc + tc
            # ---- Phase 1: whole-token sum of squares of this chunk ----
            # AIV gidx owns [jc + gidx*wc, +wc) and reduces its own tokens (no
            # cross-AIV reduction, no ssq staging tile); r is parked in
            # r_ws[token, 0] and read back by the same AIV below.
            for k in pl.range(0, wc):
                jr = jc + gidx * wc + k
                if jr < jend:
                    xs_q = xsub.next()
                    pl.load(xs_q, x_3d, [jr, 0, 0])
                    _ssq_tok_vf(xs_q, rt_t)
                    pl.set_validshape(rt_t, [1, 1])
                    pl.store(r_ws, rt_t, [jr, 0])
            # NOTE: mode= is omitted on purpose: pypto 0.3.0 (CANN 9.2.0 wheel)
            # defaults mode to SyncAllMode.HARD, while newer pypto builds removed
            # pl.SyncAllMode entirely and take only core_type.  The core_type-only
            # call is the single form both versions accept and keeps the hard
            # (FFTS-signal, workspace-free) barrier on the old build.
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
            # ---- Phase 2: gating + fused RmsNorm per token, Sinkhorn per batch ----
            # The Sinkhorn is evaluated once per batch of up to NT tokens with
            # lane = token, so its 20-iteration / 640-division chain is
            # amortised over the batch instead of being replayed per token.
            nbat = (wc + NT - 1) // NT
            for b in pl.range(0, nbat):
                nt = pl.min(NT, wc - b * NT)
                jb = jc + gidx * wc + b * NT
                # All 28 core-partials of the whole batch in ONE strided read
                # plus ONE reduction (28 x nt*PWC), instead of 28 strided pieces
                # per token.  SGRP bounds the tile: nt = wc <= TPC/NAIV = 10.
                pl.set_validshape(sl_t, [NB, nt * PWC])
                pl.load(sl_t, pmix3f, [0, jb * PWC])
                pl.set_validshape(mr_t, [1, nt * PWC])
                pl.sum(mr_t, sl_t, sl_t, dim=1)
                for kk in pl.range(0, NT):
                    if kk < nt:
                        j = jb + kk
                        if j < jend:
                            xs_cur = xsub.next()
                            pl.load(xs_cur, x_3d, [j, 0, 0])
                            # r of this token (Phase 1, read back as a 32B row)
                            pl.set_validshape(qr_t, [1, 1])
                            pl.load(qr_t, r_ws, [j, 0])
                            # gating: pre/post tile + this token's batch row
                            _gates1_vf(mr_t, qr_t, sc24_t, pm24_t, ep24_t,
                                       base_tile, pr_t, zb_t, kk * MIXC,
                                       kk * PWC)
                            # y pass1 (fused RmsNorm first half) + rstd + post
                            _y_pass1_vf(pr_t, xs_cur, yr_t, rt_t, pb_t)
                            pl.set_validshape(rt_t, [1, 1])
                            pl.store(rstd, rt_t, [j, 0])
                            pl.set_validshape(pb_t, [1, HC])
                            pl.store(post, pb_t, [j, 0])
                            # y pass2: y_norm = y*rstd*gamma -> GM
                            yu_cur = yub.next()
                            _y_norm_vf(yr_t, gam_t, rt_t, yu_cur)
                            pl.store(y, yu_cur, [j, 0])
                # ---- batched Sinkhorn: [nt,24] -> [24,nt] -> comb -> [nt,16] ----
                pl.set_validshape(zb_t, [nt, MIX_HC])
                pl.set_validshape(zd_t, [MIX_HC, nt])
                pl.transpose(zd_t, zb_t)
                _sinkhorn_batch_vf(zd_t, cbb_t, nt)
                pl.set_validshape(cbb_t, [16, nt])
                pl.set_validshape(cbt_t, [nt, 16])
                pl.transpose(cbt_t, cbb_t)
                pl.store(comb16, cbt_t, [jb, 0])


# ======================================================================
# Host wrapper
# ======================================================================

# ACL graphs capture the raw device addresses of these workspaces.  Evicting
# an entry allocated while a graph was captured would let the allocator hand
# the block to a later graph, so those entries are pinned for the process
# lifetime.  Every eager call (prefill or decode) uses a bounded LRU instead,
# so a long-lived process cannot grow the cache without limit.  The key keeps
# the stream id: two streams never share one pmix3/r_ws pair.
_WorkspaceT = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_WS_CACHE: dict[tuple[int, int, int], _WorkspaceT] = {}
_WS_SCRATCH_CACHE: OrderedDict[tuple[int, int, int], _WorkspaceT] = (
    OrderedDict())
_WS_SCRATCH_MAX = 8
_WS_CACHE_LOCK = threading.Lock()
# torch_npu's Stream wrapper costs ~4us per call (object construction), which
# would dominate the host path of this ~220us kernel.  The raw binding returns
# (stream_id, device_index, device_type) in ~0.1us; fall back to the public API
# when it is unavailable (older torch_npu or the CPU test stub).
_GET_CURRENT_STREAM = getattr(getattr(torch_npu, "_C", None),
                              "_npu_getCurrentStream", None)


def _current_stream_id(device: torch.device) -> int:
    index = (device.index if device.index is not None
             else torch.npu.current_device())
    if _GET_CURRENT_STREAM is not None:
        return int(_GET_CURRENT_STREAM(index)[0])
    return int(torch.npu.current_stream(device).stream_id)


def _ws_key(t: int, device: torch.device) -> tuple[int, int, int]:
    """Workspace key: token count, physical device and current stream."""
    index = (device.index if device.index is not None
             else torch.npu.current_device())
    return (t, index, _current_stream_id(device))


def _comb16_rows(t: int) -> int:
    """Rows the kernel can write to comb16, including the last AIV window.

    The batched Sinkhorn stores a whole [nt,16] block per AIV, so the last
    chunk writes up to jc + NAIV*ceil(tc/NAIV) rows.  TPC is not a multiple
    of NAIV, so rounding t up to NAIV is not sufficient (t=1024 would need
    1072 rows, not 1064).
    """
    ng = (t + TG - 1) // TG
    nch = (ng + GPCH - 1) // GPCH
    tc_last = t - (nch - 1) * TPC
    return (nch - 1) * TPC + NAIV * ((tc_last + NAIV - 1) // NAIV)


def _get_workspaces(t: int, device: torch.device) -> _WorkspaceT:
    key = _ws_key(t, device)
    # Only a workspace allocated while a graph is being captured must stay at
    # the same address for replay; eager shapes are reusable and bounded.
    try:
        capturing = bool(torch.npu.is_current_stream_capturing())
    except (AttributeError, RuntimeError):
        capturing = False
    with _WS_CACHE_LOCK:
        cache = _WS_CACHE if capturing else _WS_SCRATCH_CACHE
        ws = cache.get(key)
        if ws is not None:
            if cache is _WS_SCRATCH_CACHE:
                cache.move_to_end(key)
            return ws
    # Allocation may synchronize the device; do not hold the lock.
    ng = (t + TG - 1) // TG
    # per-token r (x-rstd scratch, handed from Phase 1 to Phase 2); the 8
    # columns keep the 32B vector load/store inside the allocation
    r_ws = torch.empty((t, 8), dtype=torch.float32, device=device)
    # +64 token rows of padding: the batched partial reduction of the last
    # AIV window can run up to NAIV-1 = 55 tokens past t
    pmix3 = torch.empty((NB, ng * TG + 64, PWC),
                        dtype=torch.float32, device=device)
    ws = (pmix3, pmix3.view(NB, (ng * TG + 64) * PWC), r_ws)
    with _WS_CACHE_LOCK:
        cache = _WS_CACHE if capturing else _WS_SCRATCH_CACHE
        cached = cache.get(key)
        if cached is not None:
            return cached
        cache[key] = ws
        if cache is _WS_SCRATCH_CACHE:
            # True LRU: a hit moves the entry to the end, a miss evicts the
            # least recently used one.  Bounded so eager shapes cannot grow
            # the process footprint without limit.
            while len(cache) > _WS_SCRATCH_MAX:
                cache.popitem(last=False)
    return ws


def mhcpresinkhorn_rmsnorm(
        x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
        hc_base: torch.Tensor, gamma: torch.Tensor, hc_mult: int = HC,
        sinkhorn_iters: int = SINKHORN_ITERS, eps: float = EPS,
        rms_eps: float = RMS_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public entry of mhcpresinkhorn_rmsnorm PyPTO-Pro v4-K + RmsNorm fusion (bf16 hc_fn).

    The matmul splits the K axis to balance the load across all 28 cube
    cores (27x592 + 400 on the last core); RmsNorm is fused in place right
    after y is computed (y_norm = y*rstd*gamma,
    rstd = 1/sqrt(mean(y^2)+rms_eps)), so y carries the normalized values
    and the original bf16 y is no longer emitted separately (it is only
    consumed by RmsNorm in the production chain).
    """
    if hc_mult != HC:
        raise ValueError(f"hc_mult={hc_mult} != {HC}")
    if sinkhorn_iters != SINKHORN_ITERS:
        raise ValueError(f"sinkhorn_iters={sinkhorn_iters} != "
                         f"{SINKHORN_ITERS}")
    if eps != EPS:
        raise ValueError(f"eps={eps} != {EPS}")
    if rms_eps != RMS_EPS:
        raise ValueError(f"rms_eps={rms_eps} != {RMS_EPS}")
    if x.dtype != torch.bfloat16 or x.dim() != 3 \
            or x.shape[1] != HC or x.shape[2] != D:
        raise ValueError(f"mhcpresinkhorn_rmsnorm: x must be [t,{HC},{D}] bf16, "
                         f"got {tuple(x.shape)}/{x.dtype}")
    if not x.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: x must be contiguous")
    if x.device.type != "npu":
        raise ValueError(
            f"mhcpresinkhorn_rmsnorm: x must be on an npu device, "
            f"got {x.device}")
    if hc_fn.dtype != torch.bfloat16 or tuple(hc_fn.shape) != (MIX_HC, HC_D):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: hc_fn must be [{MIX_HC},{HC_D}] bf16 "
                         f"(model side converts weight dtype), got "
                         f"{tuple(hc_fn.shape)}/{hc_fn.dtype}")
    if not hc_fn.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_fn must be contiguous")
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_scale must be [3] fp32")
    if not hc_scale.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_scale must be contiguous")
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (MIX_HC,):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: hc_base must be [{MIX_HC}] fp32")
    if not hc_base.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_base must be contiguous")
    if gamma.dtype != torch.bfloat16 or tuple(gamma.shape) != (D,):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: gamma must be [{D}] bf16")
    if not gamma.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: gamma must be contiguous")

    device = torch.device(
        f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    # Avoid a redundant ACL device switch on every call: inside an ACL
    # graph capture the context must stay on the capturing device.
    if torch.npu.current_device() != device.index:
        torch.npu.set_device(device)
    x = x.to(device)
    hc_fn = hc_fn.to(device)
    hc_scale = hc_scale.to(device)
    hc_base = hc_base.to(device)
    gamma = gamma.to(device)

    t = x.shape[0]
    if t < 1:
        raise ValueError(
            f"mhcpresinkhorn_rmsnorm: x must have t >= 1, got {t}")
    x_2d = x.reshape(t, HC_D)
    y = torch.empty((t, D), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((t, 1), dtype=torch.float32, device=device)
    post = torch.empty((t, HC), dtype=torch.float32, device=device)
    # comb is allocated on the exact AIV token window: the batched Sinkhorn
    # stores a whole [nt,16] block, so the last AIV window may run past t.
    # The extra rows are never read back (the public result is [:t]).
    tpad = _comb16_rows(t)
    comb16 = torch.empty((tpad, HC * HC), dtype=torch.float32, device=device)

    pmix3, pmix3f, r_ws = _get_workspaces(t, device)
    args = (x_2d, x, hc_fn, hc_scale.view(1, 3), hc_base.view(1, MIX_HC),
            gamma.view(1, D), y, rstd, post, comb16, pmix3, pmix3f, r_ws)
    # kernel_k handles every t, including the decode buckets.
    mhcpresinkhorn_rmsnorm_kernel_k[None, NB](*args)
    return y, rstd, post, comb16[:t].view(t, HC, HC)
