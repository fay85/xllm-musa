# DeepSeek-V4-Flash MUSA DSpark 30K 验证记录

日期：2026-09-14。实验分支：`goal-dspark-30k-20260914`。
所有日志均在本目录 `logs/` 下；模型为
`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。

## 文档基线复现

使用 TP8/EP1/DP1/CP1、DSpark N=5、Graph、Prefix Cache 0.65、chunked
prefill、`max_tokens_per_batch=10240`、显存利用率 0.8。服务使用已存在的
MUSA AOT objects、custom all-reduce 和 Torch/MUSA 2.9.1，不重新编译共享依赖。

实验目录：`logs/repro_dspark_20260914_baseline_run/`。

| 项目 | 结果 |
| --- | --- |
| warmup 64/64 | 2/2 严格全文 PASS |
| 正式请求 | 8 个变体均严格 `1..750`，usage 精确 |
| 热窗 | variant 2--7，6/6 PASS |
| TTFT 平均 | 218.618 ms |
| TPOT 平均 | 8.151 ms |
| 前缀缓存 | 每个变体 cached_tokens=1536 |

汇总证据：`logs/repro_dspark_20260914_baseline_run/window_2_7.summary.json`。
基线首次启动时 manifest 记录的目标工作树仍是迁入修复前的 `046fb526`；二进制
来自同名只读工作树的 `ad17d7f8`，随后已将 `cafa1663`、`b8dc72ba`、`ad17d7f8`
三个提交迁入本分支，因此源代码与该二进制对齐。注意该二进制 SHA256 为
`99bde8b9ece700a5bc452c090663fc74bf7b71e06136a7214cf475f65a16023b`，与本分支
迁入的 stream/state 修复提交一致；后续若重新构建必须重新记录 SHA。

修正后的 runner 为 `tools/dspark_repro_run_suite.sh`。原文档生成的
`run_suite.sh` 因 `service's` 撇号导致 Bash 引号错误，本地 runner 只修复该语法。

## 30K/1024 尝试

本轮新增修复：`CompositeBlockManager::allocate_sequence` 在 SWA ring
增长前先发布已完成前缀并释放已经滑出窗口的物理页，避免 chunked prefill
在第二个 chunk 因“先申请、后释放”而死锁。启动器同时显式传递
`max_tokens_per_chunk_for_prefill`；`max_tokens_per_batch` 不能代替该参数。

### EP1，32768 batch token，显存 0.8

目录：`logs/repro_dspark_20260914_30k_mem80_run/`。

模型加载后在 KV-cache capacity 估算阶段终止：

```text
available memory=26.39 GB
DSV4 c4_count must be > 0 when compress_ratio=4 layers exist
```

原因是一次性 32K batch 的固定 SWA/workspace 预留耗尽 cache 预算，属于启动配置
容量失败，不是 FlashMLA 数值错误。

### EP1，chunk=4096，显存 0.9

目录：`logs/repro_dspark_20260914_30k_chunk4096/`。

服务可启动，但旧二进制 cache shape 为 `c4_count=3744,c128_count=117`；由于
启动器没有传 `max_tokens_per_chunk_for_prefill`，实际仍按整段 prompt 申请，30K
请求被 scheduler 拒绝。

```text
Request prompt is too long, no enough memory to schedule a single sequence
```

### EP8，chunk=1024，显存 0.9

目录：`logs/repro_dspark_20260914_30k_ep8/`。

EP8 将每卡显存占用降至约 72 GiB，但旧启动配置仍没有显式 chunk 参数，30K
同样被 scheduler 拒绝。

### SWA 修复后二进制，chunk=1024，显存 0.9

目录：`logs/repro_dspark_30k_swa_fix_chunk1024/`。服务能够接受并完成
`prompt_tokens=30000, generated_tokens=1024`，证明 SWA 物理页增长死锁已解除；
但 server 端 TTFT 约 `14.5 s`、TPOT 约 `388.7 ms`，客户端没有收到可见
`content`（生成的是不可见 token），严格计数未通过。因此这是“功能可调度但精度
未闭环”，不能作为最终达标结果。

使用与正式基线相同的上下文型 prompt（非纯 `x` 填充）复测，结果一致：server
完成 `generated_tokens=1024`，TTFT 约 `14.24 s`、TPOT 约 `386.8 ms`，原始 SSE
逐 token 返回 `<｜end▁of▁sentence｜>`，不是客户端解析丢失。显式设置
`reasoning_effort=none` 也未改变该结果。

chunk=128 对照也能完成请求，但 TTFT 约 `45.5 s`、TPOT 约 `878.9 ms`，同样
没有可见 content；说明长 chunked prefill 仍存在 compressor/indexer/SWA 状态或
长上下文 logits 问题，尚未达到高性能/正确性目标。

### 显式 full-prefill 尝试

`max_tokens_per_batch=32768`、显存 0.9 可启动（`swa_count=260,
c4_count=1440,c128_count=45`），但 30K 请求首次 prefill workspace 额外申请
6.71 GiB 时 OOM，不能用 full-prefill 绕过 chunk 状态问题。

### 已验证的不可行尝试

- `max_memory_utilization=0.98`、chunk=31104：服务 warmup 即 MUSA OOM，分配
  20 MiB 时每卡仅余约 63 MiB。
- `kv_cache_dtype=fp8_e4m3`：MUSA DSV4 的 model1/state cache 仍占主要空间，
  capacity 未达到 30K，不能据此宣称支持 FP8 KV cache 生产路径。
- `host_blocks_factor=2`：MUSA 平台启动时直接报
  `supports_host_offload ... only supports the LLM engine and the MTP speculative engine`，
  DSV4 DSpark 不支持 host KV offload。

## 结论与后续

当前 80-GiB S5000、DSpark TP8/EP1 已经可以在修复后完成 30K 的 cache 分配和
生成循环，但长 chunked prefill 的严格精度与性能仍未闭环；full-prefill 受 workspace
峰值显存限制。当前不能宣称 30K/1024 达标，需继续修复 chunked compressor/indexer/
SWA 状态演进或降低 prefill workspace 峰值。

若必须支持 30K，需要后续架构级工作（例如降低 DSV4 model1/state 每 token
占用、实现 MUSA 可用的 KV host/offload 或采用更大显存/不同并行切分），并为
这些路径补充 kernel、跨页 state、Graph 和严格全文回归；本轮没有修改共享库、
模型或 GPU 驱动，也没有把失败结果记为性能结果。

## 8192/1024 补测（同日后续）

当前实验二进制（Attention 新请求状态复位提交 `8c64b0a7` 后重编译）
在文档原启动参数及预热顺序下完成 8 个正式请求和 6 个连续重放，
14/14 严格 `1..512` 通过。连续重放 TTFT=537.931 ms、TPOT=8.093630 ms，
整组实际墙钟 Total TGS=1045.107 tok/s（B1，TP8，包含缓存输入）。
长 chunk 实验配置和未预热路径仍有错误，不能由这些 8192 成绩证明 30K 已修好。
所有冷态失败、配置差异、二进制 SHA 及复现命令见
[8192 专项记录](DEEPSEEK_V4_MUSA_DSPARK_8192_VALIDATION.md)。

## 最新状态复位后的 30K/1024 复测

日志目录：logs/repro_dspark_30k_state_reset2_20260914/。使用当前分支
8c64b0a7 状态复位修改后的二进制（SHA256：
396a855499468a9add72321f1b862b5a73231ef0ceb37b098801593363f14269），
TP8/EP1、DSpark N=5、Graph、Prefix、chunk=10240、显存利用率 0.8、
调度显存门槛 1.0。服务重新启动后先执行两次 64/64 预热。

| 请求 | prompt/output | cached | TTFT ms | TPOT ms | E2E ms | 严格计数 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| long_1 | 30000/1024 | 0 | 4345.535 | 8.159 | 12692.216 | 512/512 PASS |
| long_2 | 30000/1024 | 0 | 3768.438 | 8.022 | 11975.305 | 512/512 PASS |

两次请求均 server generated_tokens=1024、finish=length，没有 OOM、调度拒绝
或服务退出；每卡显存峰值仍低于 80 GiB。按客户端 E2E 计算，两次合计输入吞吐
约 1328.3 tok/s、输出吞吐约 85.5 tok/s、总吞吐约 1373.8 tok/s（仅两次串行请求，
不是并发吞吐）。count-to=511/513 的附加请求均实际输出到 512；由于
1024 output-token 上限，不能据此判为状态错误。

该复测证明本轮状态复位和 SWA 页释放后，30K/1024 在当前纯 x 填充计数协议下可
连续完成并正确输出；它仍不等价于 GSM8K 或任意自然语言长上下文全量正确性，也
不意味着 TTFT 达到 250 ms。旧的未预热 EOS 异常及 GSM8K 71% 结果仍需后续
Graph/draft 关闭对照，不能从本次两次通过直接归因于某个算子已完全修好。
