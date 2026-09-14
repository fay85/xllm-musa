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
注意该二进制 SHA256 为
`99bde8b9ece700a5bc452c090663fc74bf7b71e06136a7214cf475f65a16023b`，与本分支
迁入的 stream/state 修复提交一致；后续若重新构建必须重新记录 SHA。

修正后的 runner 为 `tools/dspark_repro_run_suite.sh`。原文档生成的
`run_suite.sh` 因 `service's` 撇号导致 Bash 引号错误，本地 runner 只修复该语法。

## 30K/1024 尝试

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

服务可启动，但 cache shape 为 `c4_count=3744,c128_count=117`，可调度容量约
15K token；30K 请求被 scheduler 正确拒绝：

```text
Request prompt is too long, no enough memory to schedule a single sequence
```

### EP8，chunk=1024，显存 0.9

目录：`logs/repro_dspark_20260914_30k_ep8/`。

EP8 将每卡显存占用降至约 72 GiB，但 cache shape 仍为
`c4_count=3808,c128_count=119`（约 15K token）；30K/1024 同样被 scheduler
拒绝。该路径的拒绝不是请求精度失败，客户端只收到无 content token 的错误响应。

### 已验证的不可行尝试

- `max_memory_utilization=0.98`、chunk=31104：服务 warmup 即 MUSA OOM，分配
  20 MiB 时每卡仅余约 63 MiB。
- `kv_cache_dtype=fp8_e4m3`：MUSA DSV4 的 model1/state cache 仍占主要空间，
  capacity 未达到 30K，不能据此宣称支持 FP8 KV cache 生产路径。
- `host_blocks_factor=2`：MUSA 平台启动时直接报
  `supports_host_offload ... only supports the LLM engine and the MTP speculative engine`，
  DSV4 DSpark 不支持 host KV offload。

## 结论与后续

当前 80-GiB S5000、DSpark TP8/EP1（或 EP8）在此模型和现有 MUSA cache 实现下，
30K 单序列不能仅通过 chunk 或显存比例配置跑通：可用 cache 约 15--21K token，
而完整 30K 需要更多压缩 KV/state 空间。框架已经在调度前给出明确拒绝，不存在
隐性错误输出。

若必须支持 30K，需要后续架构级工作（例如降低 DSV4 model1/state 每 token
占用、实现 MUSA 可用的 KV host/offload 或采用更大显存/不同并行切分），并为
这些路径补充 kernel、跨页 state、Graph 和严格全文回归；本轮没有修改共享库、
模型或 GPU 驱动，也没有把失败结果记为性能结果。
