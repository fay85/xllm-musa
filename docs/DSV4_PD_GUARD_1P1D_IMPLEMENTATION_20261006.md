<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# DSpark 1P1D Guard：实现、优化和源码对应关系

## 1. 这次提交了什么

本分支从 `b5cc2b94679db6770b7ed8b6e96d1cb922eaa042` 出发，收集实际 v128 与
PD 实验中的冻结源码，不从后续 `perf/b16-6x-20261005` 的活动工作区直接打包。
默认 `xllm/` 对应 Prefill 源码组合；Decode 保留在
`tools/pd_guard_20261006/decode_overlay/xllm/` 的必要差异，由增量构建脚本选用。

这不是只有文档的提交，也不是只提交最后一个 mask 开关。
同时包含基础 DSpark/Graph/MUSA 改动、PD 交接和长序列修复、Prefill 优化、
固定依赖配方、启动和评测脚本。模型、二进制、编译 cache、AOT 不进入 Git。

原实测组合：P=`caf3db8f...`，D=`5ea86017...`；两者不能随意互换。
源码路径、冻结来源及 SHA 见
[source_provenance.json](../tools/pd_guard_20261006/source_provenance.json)。
构建配方还按原对象名记录实际翻译单元和 archive，避免将增量对象误当成完整源文件。

## 2. 请求和状态如何流转

```text
客户端
  │ HTTP /v1/chat/completions
  ▼
92 / Control                         CPU 路由与协调
  │ 安排输入处理
  ▼
94 / Prefill（8 卡）
  ├─ 分块 Prefill、Attention、Indexer、MoE
  ├─ 发布可传输的 KV 与压缩/滑窗状态
  └─ 准备 DSpark draft context-KV 和 bootstrap embedding
                   │
                   │ Mooncake PULL：D 主动读取 P 的状态
                   ▼
90 / Decode（8 卡）
  ├─ 映射 rank、slot、logical block 与物理页
  ├─ 恢复 SWA、C4/C128、Indexer、target/draft 状态
  └─ DSpark N6 / targetq7 + Graph + overlap
                   │
                   ▼
             Control 返回 token
```

搬运的不是单一普通 KV 数组。DSpark 需要 draft context 和首步 bootstrap；
DeepSeek-V4 还需要滑动窗口、压缩和 Indexer 状态。任一项错位都可能“服务已启动，
但 Decode 输出错误”，所以不能把 HTTP200 当成正确性证明。

## 3. 主要实现落点

| 部分 | 代码位置（相对仓库根） | 解决的问题 |
|---|---|---|
| DSpark PD 接入 | `xllm/core/runtime/dflash_worker_impl.cpp`、`worker_impl.cpp`、`speculative_worker_impl.cpp` 及对应头文件 | 让 DSpark 状态交接走支持 MUSA 的 PD 路径；准备 Decode 接单需要的 bootstrap embedding；保持 state/context-KV 与计算时序一致 |
| worker 连接 | `xllm/core/runtime/worker_rendezvous.cpp` | MUSA worker 参与 KV transfer 的 link/unlink，不因设备宏分支而漏掉真实连接 |
| PULL / session | `xllm/core/framework/kv_cache_transfer/mooncake_transfer_engine.cpp`、`mooncake_segment_reopen.h`、`mooncake_open_session_retry.h` | 远端进程重启后重新获取 segment descriptor，清理对应 peer 的旧连接，避免继续使用旧 session/QP；保留有界重试及失败返回 |
| 调度和失败结束 | `xllm/core/scheduler/disagg_pd_scheduler.cpp` | 传输/入队失败要进入失败请求处理和资源清理，不能让客户端一直等待；Prefill 完成后触发允许的 SWA 回收 |
| 长序列内存 | `xllm/core/framework/block/composite_block_manager.cpp`、`sliding_window_block_manager.{cpp,h}`、`block_manager_pool_pd.cpp` | 逻辑历史长度与实际尾窗物理页分开；发布完成状态后回收无用 SWA 页，保留窗口、speculative margin 和 C4/C128 历史 |
| Graph/Indexer 状态 | `xllm/core/runtime/musa_graph_executor_impl.*`、`dsa_metadata_builder_musa.*`、`deepseek_v4_graph_*` 等 | 继承单机精度修复及按请求/位置更新 metadata 的保护，防止 capture 常量或上一请求状态误复用 |
| 跨卡通信 | `xllm/core/framework/parallel_state/musa/`、`musa_process_group.h` | 继承 v128 的 TP 通信优化和适用条件约束 |

`block_manager_pool_pd.cpp` 是补充翻译单元，**不能替换原 `block_manager_pool.cpp`**；
`worker_rendezvous.cpp` 也不是 `worker.cpp`。发布树保留原文件与补充文件，
并把新的 PD block 单元加入 MUSA 的 CMake 源列表。

传输中已有的 snapshot header/entry/payload、布局/生命周期校验仍由原 PD 基础实现负责。
不是新增一个不兼容协议后绕过校验，也没有仅为通过计数测试而少传模型需要的状态。

## 4. Prefill 性能优化链

| 优化 | 实现位置 | 行为及正确性约束 |
|---|---|---|
| 16K chunk/budget | 启动配置 + block/scheduler | 减少长输入的 chunk 调度开销；与 SWA 回收一起使用，不能只增加 budget |
| compact MoE | `deepseek_v4_moe.cpp`、`deepseek_v4_fp8_moe_metadata.cu` | 压紧实际路由 token 的计算/恢复流程，结合 vector restore、warp SwiGLU 减少中间开销 |
| Gate stream handoff | `deepseek_v4_gate.cpp` | 用 TVM FFI stream guard 保证 Gate 和后续消费者在正确流上衔接，修复前缀复用条件下的时序异常 |
| Prefill native ReduceScatter | `deepseek_v4_moe.cpp`、`deepseek_v4_decoder_layer.cpp`、`deepseek_v4_pd_reduce_scatter_policy.h` | 在已满足 SP shape/拓扑条件时使用适配的通信路径，保持 decoder residual 和输出布局契约 |
| HC tiled Prefill | `deepseek_v4_hc_tiled_prefill.h` 和调用点 | 限制大 Prefill 的 workspace/中间张量规模；不是重新打开之前有精度风险的 MHC+Norm 融合 |
| Indexer BF16 输入 / FP32 score | `deepseek_v4_indexer_reference.h` | 使用已经单独检查过的 `bmm_out` 路径；不是任意改成低精度输出或降低 TopK |
| DSA prepare once | `deepseek_v4_dsa_attention.cu` | 复用同一调用内可共享的数据准备，减少重复处理 |
| RoPE 组装拷贝 | `deepseek_v4_rope_*_assembly.h`、Attention | 减少已具备相同布局的数据重组；不混淆 C1/SWA 与 C4/C128 的 RoPE 语义 |
| Indexer mask once | `deepseek_v4_indexer_reference.h`、`deepseek_v4_attention.cpp` | 本次最终达标的小步改动，见下一节 |

这些开关组合以冻结 launch manifest 为准。最后一步实验没有替换 Decode、Control、
模型、AOT、评分器或测试题目；不要把整个优化链的收益全部归功于 mask once。

## 5. 最后一步：Indexer mask 每次调用只准备一次

此前每个 score tile 都重复构造其因果边界和 mask。现在在本次 Indexer helper 调用内，
一次性算出所有 query 对应的有效历史长度、score mask 和 TopK index mask，再给各 tile
使用 `.narrow()` 视图。

```text
原来：tile0 构造 mask → 算 score/TopK
      tile1 再构造 mask → 算 score/TopK
      tile2 再构造 mask → 算 score/TopK

现在：本次调用构造完整 mask（只一次）
      ├─ tile0 取对应视图 → 原 score/TopK
      ├─ tile1 取对应视图 → 原 score/TopK
      └─ tile2 取对应视图 → 原 score/TopK
```

开关：`XLLM_DSV4_PREFILL_INDEXER_MASK_ONCE=1`，源码默认关闭，由本发布 P 配置显式打开。
只在符合条件的 BF16/FP32 长 Prefill、query ≤2048、绝对起点有效、非 Graph capture 路径启用。
示例 trace：`q=2048 keys=4096 mask_bytes=9437184 per_call=1 unchanged_score_topk=1`。

关键点：

- mask **只在一次函数调用中复用**，不是跨请求 persistent cache，不引入上一请求污染。
- 不改变 GEMM shape、score 的浮点计算、causal 语义、TopK=512 或实际输出校验。
- 最终候选只重新编译 Attention 单元；同一 archive 其他成员做了 hash 一致性检查。
- 此前配对吞吐比 4.47277766×，最后为 4.67788085×；B16 从 405.917509 升到
  428.193054 tok/s（约 +5.49%）。B1 TPOT 基本保持 8.78 ms。

已有定向验证包括 mask/indices/score 比较、replay 检查、真实 K512 shape、开关关闭回归、
长序列严格计数和 Prefix 命中/不命中序列。原始实验记录保存在共享盘，不把局部通过
扩展成所有自然语言长序列均已验证。

## 6. 证据与发布边界

- 原始实验根：`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005`。
- 最终配对：`indexer_mask_once_pair_20261006/paired_audit.json`。
- 快照：`evidence_indexer_mask_once_pair_20261006`、`evidence_indexer_mask_once_round_20261006`。
- 编译来源：`/tmp/pd_indexer_mask_once_model_20261006`、`/tmp/pd_session_reopen_fix_20261006`
  等独立候选；JSON 配方保留每个实际输入的源路径和 hash。
- 首次源码导出 SHA：`a875cb7518e576fdfeb1dbf0e15cced63f15a9543b6accb270a0c370a10b8393`；
  补充导出 SHA：`6dcf606567091437a5ff421897da1a6ebaa71a13f9e9a0e925b67107d144aa1d`。
- 本次重编只作为发布源码的增量编译/链接验证，未将新 binary 偷换进正在进行的 GSM8K。
- 全新环境的无缓存 CMake 全量构建、更多并行拓扑、自然语言 30K 全面语义精度均不在
  本次已经完成的验收范围内。

启动和性能命令见 [完整复现手册](DSV4_PD_GUARD_1P1D_REPRO_20261006.md)，
GSM8K 单独见 [精度报告](DSV4_PD_GUARD_GSM8K_20261006.md)。
