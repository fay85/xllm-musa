---
title: "Prefix Cache 优化"
sidebar:
  order: 50
---
## 功能介绍
xLLM支持prefix_cache匹配。prefix_cache基于mermer_hash，使用lru淘汰策略，提供更极致的匹配效率，同时提高prefix_cache命中率。
同时对prefix_cache进行了优化，支持continuous_scheduler、chunked_scheduler和zero_evict_scheduler，在prefill之后即更新
prefix_cache，提高匹配时效性，同时对于chunked_scheduler，支持多阶段chunked_prefill匹配，减少计算量并尽可能减少kv_cache占用。

## 使用方式
prefix_cache已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启zero_evict策略，并设置max_decode_token_per_sequence。
```
--enable_prefix_cache=true
```

## Cache 感知 DP 路由

当使用数据并行（`dp_size > 1`）时，请求可被路由到持有最长 prefix cache 命中的 DP rank，提高跨 rank 的 KV cache 复用率。

- 开启 cache 感知 DP 路由：
```
--enable_prefix_cache=true --enable_prefix_cache_aware_dp_routing=true
```

- `prefix_cache_aware_dp_match_threshold`（默认 `0.5`）：选择 cache 亲和 rank 所需的最低 prefix block 命中比例，低于该阈值时回退到空闲 block 均衡。
- `prefix_cache_aware_dp_imbalance_threshold`（默认 `0.1`）：跨 rank KV 利用率差异上限（`(max_used-min_used)/total_blocks`），超过此值时路由到负载最低的 rank。

## 性能效果
开启prefix_cache之后，在Qwen3-8B模型上，限制TPOT50ms，E2E时延 **下降10%**。

## LINEAR 状态缓存升级说明

LINEAR 常规检查点和 prompt 尾检查点使用独立于 KV 的 hash 链，下标粒度是 KV block。
从使用 chunk 粒度 hash 链的版本升级时，Mooncake Store 中旧的 LINEAR 条目可能无法命中，
需要重新计算并填充缓存；旧条目按 Store 的淘汰策略回收。此变化不改变 KV 条目的 key。
进程内的 Host prefix cache 随服务重启重新建立。

启用 LINEAR prefix cache 后，chunked prefill 会在常规检查点和可复用的尾边界停下，
以保存对应 token 边界的状态。非对齐 prompt 可能增加 prefill 步数；其 TTFT 和吞吐收益
取决于前缀复用率，需要对冷缓存和命中缓存分别测量。上面的 Qwen3-8B 数据不代表此改动的性能结果。

:::note
PD分离下的prefix cache支持范围与调度器角色相关，参见[PD分离](/zh/features/disagg_pd/)中的支持配置说明。
:::

## LINEAR 热点检查点策略

同一 engine 的调度器维护一棵默认跨 DP rank 共享的 block-hash 压缩前缀树，根据请求 prompt 的共享前缀和分叉
规划额外 LINEAR 检查点。统计不跨 engine 或进程共享，缓存块仍按 DP 隔离。
统计树不持有 KV block 或 LINEAR 槽位，也不依赖本地缓存是否驻留；
即使 KV 已被淘汰，仍可在后续 prefill 中重新生成对应检查点。

- 启用 prefix cache 和 chunked prefill，且 LINEAR 缓存支持实际 block 步长时自动生效，无需新增启动参数。
- 默认要求前缀存在至少四个不同的直接后继 block hash。
  sequence 首次规划时先匹配、再观察 prompt；重复相同 prompt（包括并行采样）不会增加不同后继数。
- 每条 sequence 最多规划一个额外停点，重试和 DP 切换沿用原计划，选择当前匹配路径上最深的合格分叉。
  仅考虑按 block 对齐且不重复常规或 prompt 尾检查点的边界。
- 规划不代表实际缓存命中。初始 prefix cache 匹配完成后，调度只在尚未越过检查点时截断 chunk；
  如果已恢复到检查点或更深的位置，则直接继续正常 chunked prefill。
- 每棵统计树默认最多保留 65536 个 hash，压缩边上的 hash 也计入容量，而非按请求数计。
  容量不足时先裁剪冷叶子的长后缀，再淘汰叶子，避免删除祖先后留下不可达节点。
  容量已满时，已有叶子仍可回收其他叶子的空间来记录新后继，保留当前匹配的前缀。
- 统计默认使用 10 分钟的惰性过期时间，在下一次匹配或统计时清理过期叶子。
  匹配和再次观察仍刷新访问时间。
  统计淘汰不直接释放物理检查点；显式重置 prefix cache 时同时清空统计树。

额外停点可能增加 prefill 开销，检查点发布仍受 LINEAR 槽位可用性限制。
