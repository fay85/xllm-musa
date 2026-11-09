---
title: "Prefix Cache Optimization"
sidebar:
  order: 50
---
## Feature Introduction
xLLM supports prefix cache matching. The prefix cache is based on `murmur_hash` and uses an LRU eviction policy, delivering superior matching efficiency and increased prefix cache hit rates.
Additionally, the prefix cache has been optimized to support the `continuous_scheduler`, `chunked_scheduler`, and `zero_evict_scheduler`. The cache is updated immediately after prefill operations, enhancing matching timeliness. For the `chunked_scheduler`, multi-stage chunked prefill matching is supported, reducing computational overhead and minimizing KV cache usage as much as possible.

## Usage
The prefix cache is implemented in xLLM and exposed through gflags parameters to control its functionality.

- Enable prefix cache with specific policy and settings:
```
--enable_prefix_cache=true
```

## Cache-Aware DP Routing

When running with data parallelism (`dp_size > 1`), requests can be routed to the DP rank that holds the longest prefix-cache hit, improving KV cache reuse across ranks.

- Enable cache-aware DP routing:
```
--enable_prefix_cache=true --enable_prefix_cache_aware_dp_routing=true
```

- `prefix_cache_aware_dp_match_threshold` (default `0.5`): minimum prefix block hit ratio to prefer the cache-affinity rank. Below this threshold, routing falls back to free-block balancing.
- `prefix_cache_aware_dp_imbalance_threshold` (default `0.1`): maximum cross-rank KV utilization gap (`(max_used-min_used)/total_blocks`). Exceeding this disables affinity routing and selects the least-loaded rank.

## Performance Impact
After enabling prefix cache, on the Qwen3-8B model with a TPOT constraint of 50ms, the E2E latency **decreased by 10%**.

## Upgrading LINEAR State Caches

Regular LINEAR checkpoints and prompt-tail checkpoints use a hash chain that is
separate from the KV chain and indexed at KV-block granularity. When upgrading
from a version that hashes LINEAR states at chunk granularity, existing LINEAR
entries in Mooncake Store may miss and need to be recomputed. Old entries are
reclaimed by the Store eviction policy. KV entry keys are unchanged. The
in-process Host prefix cache is rebuilt on service restart.

With LINEAR prefix cache enabled, chunked prefill stops at regular checkpoints
and reusable tail boundaries to save the state at the corresponding token count.
Unaligned prompts may require more prefill steps. TTFT and throughput benefits
depend on prefix reuse and should be measured separately for cold and warm caches.
The Qwen3-8B result above is not a benchmark of this change.

:::note
For disaggregated PD, prefix cache is supported in specific scheduler roles. See [Disaggregated PD](/en/features/disagg_pd/) for the supported configurations.
:::

## Hot Checkpoint Admission

Each engine's scheduler maintains one block-hash compressed prefix tree shared by its DP
ranks to plan extra LINEAR checkpoints from request prompts and their branching
prefixes. Statistics are not shared across engines or processes, and cached
blocks remain DP-local.
The tree owns no KV blocks or LINEAR slots and does not depend on local cache
residency. Checkpoints can therefore be rebuilt even after KV eviction.

- Enabled automatically with prefix caching and chunked prefill on LINEAR caches
  that support the effective block stride. No additional startup flags are needed.
- By default, a prefix must have at least four distinct direct successor block
  hashes. A sequence matches and then observes its prompt only on its first plan;
  repeating an identical prompt, including parallel samples, does not add
  successors.
- Each sequence plans at most one extra stop and retains it across retries and
  DP changes: the deepest eligible branching prefix on its match path. Candidates
  must be block-aligned and must
  not duplicate regular chunk or prompt-tail checkpoints.
- A plan is not a cache hit. After initial prefix-cache matching, the scheduler
  caps a chunk only if the checkpoint is still ahead. Restoring to or beyond the
  checkpoint allows normal chunked prefill to continue.
- Each statistics tree retains at most 65536 hashes by default, including hashes
  on compressed edges. This is a hash-entry budget, not a request count. Capacity
  pressure first trims long cold leaf suffixes, then evicts leaves without
  orphaning descendants. An existing leaf can gain successors at capacity by
  reclaiming space from other leaves while preserving its matched prefix.
- Observations have a default lazy expiry of 10 minutes, reclaimed on the next
  match or observation. Matching and
  repeated observations still refresh access times. Metadata eviction does not
  directly release physical checkpoints. Explicit prefix-cache resets also
  clear the statistics tree.

Extra stops may increase prefill work. Checkpoint publication remains subject to
LINEAR slot availability.
