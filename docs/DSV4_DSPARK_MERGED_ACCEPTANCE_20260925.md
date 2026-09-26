# DSpark q6 Graph 精度与性能联合验收

完整全量 GSM8K（对齐 SGLang boxed 协议）的后续结果见
[全量评测报告](DSV4_DSPARK_GSM8K_FULL_SGL_PROTOCOL_20260925.md)：xLLM Graph+DSpark
为1132/1319（85.8226%），与此前固定100题的96/100不是同一评分协议。

完整精度问题分析、分阶段A/B与证据边界见
[GSM8K100整体实验说明](DSV4_DSPARK_GSM8K100_ACCURACY_EXPERIMENT_REPORT_20260925.md)。

## 版本与口径

- 主目录：`/data/nfs_shared/jiaming.wang/xllm0914/xllm-musa`。
- 分支：`fix/dspark-accuracy93-perf-20260922`，基于 `c7d07f5a` 的工作区修改；
  本轮尚未新增 Git 提交，不能仅检出该 HEAD 就认为包含这些修复。
- 已构建候选：`build/gsm8k_candidates/current_graph_shared_address_20260925/xllm`。
- 二进制 SHA256：`5785ab8228debdd5e5e9a9b24be91a53ba650f913d947d225cd0b11052a43f52`。
- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。
- MUSA 8 卡，TP8/EP1，B1，DSpark N=5/target q6，Graph 开启，短上下文 Eager 保护关闭。
- 精度：固定 GSM8K 100 题，seed=20260914，temperature=0，max_tokens=2048。
- 性能：2500/1500，Prefix Cache 0.65，预热 variants 0--1，正式窗口 variants 2--7。
- 精度与性能使用同一个已启动服务、同一二进制、同一组启动参数；没有按题切换路径，
  没有 reference 审计重答或答案修补。

## 验收结果

固定100题已完成 **96/100，零请求错误**，全部输出与前一版 96/100 候选逐字一致。

| 指标 | 实测 | 门槛 |
|---|---:|---:|
| 固定 GSM8K100 | 96/100 | >=95/100 |
| 热态 TTFT_AVG | 215.340 ms | <250 ms |
| 热态 TPOT_AVG | 11.9828 ms | <13 ms |
| 严格计数及 2500/1500 usage | 6/6 | 全部正确 |
| 每次正式请求缓存命中 | 1536 tokens | 与历史热态口径一致 |

六次正式请求 TTFT 范围 211.372--220.319 ms，TPOT 范围 11.9340--12.0442 ms；
不只是平均值，每次正式请求也都满足上述延迟门槛。

`joint_goal_audit.json` 中十项检查全部通过。输出吞吐 **82.518 tok/s**，
输入+输出吞吐 **220.047 tok/s**，均为整个8卡服务的聚合值（不是单卡值）。
正式计时期间没有运行单测或其他本实验 GPU 作业。
本次 TPOT 相比前一版 14.385 ms 降低约 **16.7%**，并保持100份数学输出不变。

正式性能结束后，在同一服务上重复前20道数学题，仍为 **19/20**，20份输出与最初
fixed100 对应记录逐字一致、零请求错误，未观察到这次长计数到短数学请求的状态切换回退。
记录：`post_perf20.jsonl` / `post_perf20`。唯一错误仍为 index 235。

最终 kernel 测试：`build/gsm8k_candidates/c128_shared_address_final_unit_20260925/all_tests.log`，
**12项 PASS、无跳过**，其中11项为正向测试，1项为下文单独说明的拒绝前缀负例。
正向测试包含 int32/int64 table、FP32/BF16、真实 capture/replay、动态长度与映射，以及
新增的 q1 奇数历史长度共享内存对齐检查。`git diff --check` 和 Python/Bash 语法检查通过。

Goal 的固定数据集/热态性能口径已满足。文档末尾的一般性拒绝恢复风险仍未宣称解决。

权威记录目录：`logs/goal_shared_address_graph_20260925/`：

- `launch_manifest.txt`、`ops_artifacts.sha256`：运行参数、二进制及依赖哈希。
- `fixed100.jsonl`、`fixed100.config.json`、`fixed100`：完整原始响应及评分。
- `warmup_0_1/`、`formal_2_7/`：客户端原始计时、usage、文本及汇总。
- `joint_goal_audit.json`：重新核对题目/参考答案、固定样本、精度、六次严格计数、
  1536-token 缓存命中、Graph 路径及 TTFT/TPOT 门槛。

## 本轮主要改动

1. 状态恢复：将 8193 列 block-table 的单线程扫描改为并行最后有效列归约。
2. Graph 区分 full-KV/压缩 Attention 分支，以及 C4/C128 输出形状，绝对位置由设备输入更新。
3. C128 使用动态位置的批量压缩状态更新；target q6 每行按真实 KV 长度过滤压缩历史，
   不再把本轮所有新增压缩行无条件交给前面的 query。
4. C128 paged Attention 每个 token 的分页地址只计算一次，保存在共享内存供所有输出维度复用。
   此项 GPU 微基准从 968.674 us 降至 260.929 us，softmax/累加顺序保持不变。

旧 q2/Eager 与新 q6 Graph 使用当前 Attention 实现，没有链接旧版 legacy Attention。
实验定位细节、前序失败及适用边界见
[状态与 C128 调查记录](DSV4_DSPARK_GRAPH_STATE_C128_20260925.md)。

## 启动

先确认 GPU 上无其他服务，释放本实验占卡；不要 reset GPU。启动脚本会检查占卡进程和端口，
不会自动杀掉其他任务。使用新的日志目录，HTTP/master/master+1 端口都必须空闲。

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
MERGE_RUN_DIR="logs/repro_merged_q6_$(date +%Y%m%d_%H%M%S)"
env \
 DSPARK_BIN="$PWD/build/gsm8k_candidates/current_graph_shared_address_20260925/xllm" \
 MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt \
 XLLM2_EXTERNAL_TP8_HTTP_PORT=19260 \
 XLLM2_EXTERNAL_TP8_MASTER_ADDR=127.0.0.1:21460 \
 XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1 \
 XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1 \
 XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER=0 \
 XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1 \
 XLLM_DSV4_DYNAMIC_GRAPH_STATE=1 \
 XLLM_DSV4_GRAPH_DYNAMIC_INDEXER=1 \
 XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST=1 \
 XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST=1 \
 XLLM_DSV4_GRAPH_KEY_TRACE=1 \
 XLLM_GRAPH_TIMING=1 XLLM_GRAPH_TIMING_ALL=1 \
 bash tools/start_dspark_custom.sh "$MERGE_RUN_DIR" 10240 0.8
```

上述保留了本次测量使用的诊断开关，没有拿关闭诊断后的未测数字替代验收结果。
服务启动后，在另一终端按调查记录中的三个测评命令依次执行 fixed100、warmup_0_1、
formal_2_7，将 URL 改为端口 19260，输出路径改为本次 `MERGE_RUN_DIR`。

联合检查（输出 FAIL 时返回非零状态）：

```bash
python3 tools/audit_dspark_merge_run.py \
 logs/goal_shared_address_graph_20260925 \
 build/gsm8k_candidates/current_graph_shared_address_20260925 \
 logs/goal_repro_q6_legacy_q2_accuracy_20260923/eval100/results.jsonl
```

## 可复现构建

复用现有依赖与经过验证的 kernel donor，不修改原 archive，不编译 Mooncake。
为每次构建使用新目录；builder 保存源码、policy header、编译命令及 archive/object/binary 哈希。

```bash
MERGE_BUILD_TAG="$(date +%Y%m%d_%H%M%S)"
MERGE_KERNEL_BUILD="build/gsm8k_candidates/shared_address_kernel_${MERGE_BUILD_TAG}"
MERGE_MODEL_BUILD="build/gsm8k_candidates/shared_address_model_${MERGE_BUILD_TAG}"
python3 tools/build_gsm8k_isolated_candidate.py "$MERGE_KERNEL_BUILD" \
 xllm/core/kernels/musa/deepseek_v4_dsa_attention.cu \
 --kernel-base-archive build/gsm8k_candidates/dynamic_paged_c128_kernel_20260925/libcuda_kernels.a
python3 tools/build_gsm8k_isolated_candidate.py "$MERGE_MODEL_BUILD" \
 xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp \
 --kernel-archive "$MERGE_KERNEL_BUILD/libcuda_kernels.a"
```

第二步会同时重编当前 Attention、DSA metadata、MUSA Graph executor 和 DFlash，
不会把当前 Attention 与旧 runtime 混用。必须保留 donor archive 及其 manifest，
不能用任意同名旧库替换。

## 边界

- 精度结论限固定100题，不是完整1319题的成绩。
- 性能限 B1、2500/1500、热 Prefix Cache，不代表无缓存 TTFT、8192/1024 或多 batch。
- 四个严格评分未通过题为 235、333、952、1038，均保留原始输出与原 reference。
- 部分接受后的 recurrent-state 负例单测复现的是独立风险，不能把其 PASS 当作已修复 rollback；
  尚未将它归因到某一道实际 GSM8K 错题。一般性拒绝/恢复正确性仍需独立集成验证。
- 达到本目标门槛不等于达到旧版约 7.85 ms 的速度，也不表示所有算子/序列范围已经验证。

## 资源收尾

2026-09-25 01:56 已停止本轮服务及全部 rank。占卡 PID **54152** 已报告 READY，
`mthreads-gmi` 确认 GPU 0--7 均由该进程占用，每卡目标70 GiB。没有执行 GPU reset。
这是本次收尾快照；下一次启动前应重新检查实际 PID/显卡状态，不能直接按历史 PID 杀进程。
