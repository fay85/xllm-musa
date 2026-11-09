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

#include <cnrt.h>
#include <stdint.h>

namespace xllm::kernel::mlu::bangc::fused_mhc_detail {

// P <= 8 keeps at most 32 tokens per round so norm-offload staging fits.
constexpr int32_t round_cap(int32_t p) { return p <= 8 ? 32 : 64; }

struct Topology {
  int32_t clusters = 0;
  int32_t cores_per_cluster = 0;
};

// Topology of the current device, cached per thread for the last device.
inline Topology query_topology() {
  static thread_local int32_t cached_dev = -1;
  static thread_local Topology cached;
  int32_t dev = 0;
  if (cnrtGetDevice(&dev) != cnrtSuccess) {
    return Topology{};
  }
  if (dev != cached_dev) {
    Topology t;
    if (cnrtDeviceGetAttribute(&t.clusters, cnrtAttrClusterCount, dev) !=
            cnrtSuccess ||
        cnrtDeviceGetAttribute(&t.cores_per_cluster,
                               cnrtAttrMcorePerCluster,
                               dev) != cnrtSuccess) {
      return Topology{};
    }
    cached = t;
    cached_dev = dev;
  }
  return cached;
}

// Cores per group for T tokens (measured on mtp_592, 32 IPUs; per-core time ~
// c(P) + (T / G) * t_tok(P)). P = 16 has the smallest fixed cost but each core
// holds T / 2 tokens; P = 8 (T / 4 per core) overtakes it at T ~ 9. P = 4
// (cluster-local groups, 1280-wide slices, deferred split-K conv) has a better
// slope but more fixed cost (240 KB hc_fn slice per core): P = 8 wins below
// T ~ 88 and P = 4 above. P = 8 groups span two clusters and their time moves
// by up to 10% with the allocation history of the tensors; P = 4 groups are
// cluster local and stay within 0.3%.
inline int32_t choose_group_cores(int64_t tokens, int32_t total_cores) {
  int32_t p = 4;
  if (tokens < 89) {
    p = 8;
  }
  if (tokens <= 8) {
    p = 16;
  }
  while (p > total_cores) {
    p /= 2;
  }
  return p;
}

struct Plan {
  int32_t p = 0;
  int32_t groups = 0;
  int32_t group_tokens = 0;
  int32_t round_tokens = 0;
  int32_t rounds = 0;
};

inline Plan make_plan(int64_t tokens, const Topology& topo) {
  Plan pl;
  const int32_t total = topo.clusters * topo.cores_per_cluster;
  pl.p = choose_group_cores(tokens, total);
  pl.groups = total / pl.p;
  pl.group_tokens = static_cast<int32_t>((tokens + pl.groups - 1) / pl.groups);
  const int32_t cap = round_cap(pl.p);
  pl.round_tokens = pl.group_tokens < cap ? pl.group_tokens : cap;
  pl.rounds = (pl.group_tokens + pl.round_tokens - 1) / pl.round_tokens;
  return pl;
}

inline cnrtFunctionType_t union_type(int32_t clusters) {
  switch (clusters) {
    case 1:
      return cnrtFuncTypeUnion1;
    case 2:
      return cnrtFuncTypeUnion2;
    case 4:
      return cnrtFuncTypeUnion4;
    case 8:
      return cnrtFuncTypeUnion8;
    case 16:
      return cnrtFuncTypeUnion16;
    default:
      return cnrtFuncTypeUnion1;
  }
}

}  // namespace xllm::kernel::mlu::bangc::fused_mhc_detail
