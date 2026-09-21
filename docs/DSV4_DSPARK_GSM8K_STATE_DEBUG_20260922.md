# DeepSeek-V4 DSpark GSM8K 状态/Graph 回归记录（2026-09-22）

## 目的与固定口径

- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`
- TP8/EP1，MUSA，`max_memory_utilization=0.80`，`max_tokens_per_batch=2048`
- GSM8K parquet：`logs/gsm8k_eval_20260914_1925/gsm8k_test.parquet`
- 固定 20 题序列：`3,23,26,50,64,71,75,122,125,128,130,133,184,195,201,206,216,235,244,255`
- 固定 100 题序列：见 `logs/goal_graph_q2_eager_diag_full100b_20260922/eval100/run_config.json`
- 评测不读取 reference 参与生成，只在最后做数值 exact-match。

## 首次分叉定位

1. 单独启动 Graph+DSpark 请求 `index=128` 为 `880`；同一服务先跑前序请求后再跑 `index=128`，输出变为 `305`。因此不是固定 prompt 本身的不确定性。
2. Graph=true、draft=false 的控制组中，128/206/244 恢复正确，235 仍是语义题错误；说明主要分叉在 target q2 Graph 复用，而不是 q1 draft layer 或普通 prefill。
3. Graph+draft 默认候选 `graph_c4_final9_20260922/xllm` 固定 20 题为 **16/20 (80%)**，失败 `128,206,235,244`。
4. `XLLM_MUSA_GRAPH_SYNC_INPUTS=1` 和 `XLLM_MUSA_GRAPH_SYNC_OUTPUTS=1` 均不能消除 128，排除了单纯 caller/capture stream 同步竞态。
5. metadata 日志显示每个新请求都重新进入 `prefill=1, q_len=kv_len`；因此 sidecar/reset 已生效，剩余问题是 target q2 Graph 的 persistent DSA/accepted-token carrier 与 eager q2 不等价。

## 已做代码修复/控制项

- `dsa_metadata_builder_musa.cpp`：Graph/Eager 均生成 CPU `__musa_dsv4_host_sequence_lengths` sidecar，使 Attention 能识别 chunked prefill 的新请求起点。
- `deepseek_v4_attention.cpp`：新请求起点清理 compressor/indexer history；Graph compact q2 使用无分配 state workspace；保留 `XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER=1` 作为 indexer 对照开关。
- `dflash_worker_impl.cpp`：prefill 发布 target embedding 前清理当前 embedding id，避免无 request-id 时的旧尾状态残留。
- `musa_graph_executor_impl.cpp`：增加 `XLLM_DSV4_FORCE_EAGER_COMPACT_VERIFY=1` 控制项，只将 target q2 走 eager，draft q1 仍可走 Graph，用于隔离 Graph q2 问题；默认值仍未改变。
- `dflash_worker_impl.cpp`：增加 `XLLM_DSV4_FORCE_TARGET_TOKEN_ONLY=1` 对照项，运行 draft/target forward 但只提交 target 第一行，隔离 accepted-token 语义漂移；默认关闭。

## 对照结果

| 配置 | 结果 | 日志 |
|---|---:|---|
| Graph+DSpark 默认（q2 Graph）固定 20 | 16/20 = 80% | `logs/goal_graph_final9_20260922/eval20/summary.json` |
| Graph+target-only，q2 Graph | 3/4（128/206/244 对，235 语义错） | `logs/goal_graph_final8_targetonly_20260922/eval4/summary.json` |
| Graph+DSpark，target q2 eager 对照，固定 20 前 10 | 10/10 | `logs/goal_graph_q2_eager_diag_20260922/eval_prefix128/summary.json` |
| Graph+DSpark，target q2 eager 对照，固定 100 | **94/100 = 94%** | `logs/goal_graph_q2_eager_diag_full100b_20260922/eval100/summary.json` |

q2 eager 对照的剩余失败为 `130,201,333,570,952,1038`，内容检查显示主要是算术/语义解释分叉；其中 130/201/333 是 draft 路径相对 target-only 新增的语义漂移，570/952/1038 在 target-only 也可能出现，不能归因于 Graph q2 kernel。

## 当前结论与下一步

- **已定位**：默认错误集中在 target q2 Graph 的跨请求 persistent DSA/accepted-token 路径；q2 eager 对照可以恢复 128/206/244，证明不是 GSM8K evaluator 或普通 MATE 单算子错误。
- **尚未达成**：默认 Graph+draft 尚未完成固定 100 题 >=95%，不能把 q2 eager 的 94% 宣称为最终通过。
- 下一步应在不改变默认 Graph 性能的前提下，对同一请求的 target q2 Graph/eager 逐 layer 对比 `q/kv/slot/block/state`，重点是 q2 第一行的 accepted-context 与 C4/SWA state；然后再做无 reference 的 arithmetic/semantic audit，处理剩余 6 个语义 case。
