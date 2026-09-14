# DSpark 8192/1024 实测记录（2026-09-14）

本记录属于 `goal-dspark-30k-20260914@8c64b0a7` 的当前实验产物，
不是原高性能分支的历史成绩。模型为
`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。

## 环境与口径（A/B 使用长序列实验启动器，C 恢复文档启动器）

- MTT S5000，8 卡，每卡 80 GiB；TP8/EP1/DP1/CP1。
- **global B1**，请求串行；不是 B8，也不是 total requests=64。
- DSpark N=5，Graph/VMM/decode-no-padding、Prefix Cache 开启，调度 overlap 关闭。
- `max_tokens_per_batch=10240`，显式 `max_tokens_per_chunk_for_prefill=10240`，
  `max_memory_utilization=0.8`，`prefill_scheduling_memory_usage_threshold=1.0`。
- 二进制：`build/dspark-musa-local/xllm/xllm`，SHA256：
  `396a855499468a9add72321f1b862b5a73231ef0ceb37b098801593363f14269`。
  本次仅重新编译 Attention C++ 对象并链接已有归档，不重新构建共享依赖。
- 与原文档启动器有差异：本轮使用 `tools/start_dspark_custom.sh`，
  显式 chunk 大小及调度显存门槛用于当前长输入实验，因此不能宣称是原文档配置逐项不变的复现。
- TTFT/TPOT 为客户端流式计时。吞吐按 `usage tokens / 请求 E2E 秒数` 计算；
  Total TGS 包含输入（含已缓存输入）与输出。它不是实际重新计算的 prefill 吞吐，
  也不是单卡吞吐，不能把 `1000 / TPOT` 当作端到端 Output TGS。

## 实验 A：短请求预热后重复同一 8192 prompt

日志目录：`logs/repro_8192_state_reset_20260914_1726/`。
先执行两次 64/64 请求；随后 4 次 8192/1024，要求只输出 `1..512`。
使用 `tools/benchmark_dsv4_fast_exact.py --show-special-tokens`，未改写模型答案。
同一 prompt 重复并不等于构造了 0.65 前缀变体；实际缓存命中率以 usage 为准。

| 请求 | cached tokens | TTFT ms | TPOT ms | Output TGS | Total TGS | 严格全文 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 1459.242 | 14.095 | 64.487 | 580.384 | FAIL |
| 2 | 5248 | 952.205 | 8.057 | 111.366 | 1002.298 | PASS |
| 3 | 5248 | 535.501 | 8.255 | 114.023 | 1026.205 | PASS |
| 4 | 5248 | 530.497 | 8.487 | 111.144 | 1000.294 | PASS |

所有请求 usage 都是 8192 输入、1024 输出。请求 1 实际只数到 100 后生成额外
德语问答，不是正确数到 158；正则共提取 158 个数字，严格验证失败。
请求 2--4 均严格输出 `1..512`，缓存命中 5248/8192（64.06%）。
这三个缓存命中请求的平均 TTFT=672.734 ms、TPOT=8.266 ms，
按请求耗时之和计算 Output TGS=112.163、Total TGS=1009.463。
四个请求全部计入则均值 TTFT=869.361 ms、TPOT=9.724 ms，
严格正确性为 3/4；不能丢弃首请求后声称冷/热路径均已通过。

## 实验 B：独立服务不预热首请求

日志目录：`logs/repro_8192_fresh_retry_20260914_1750/`。
独立重新加载同一二进制，未先执行 64/64，直接执行相同 8192/1024 请求。
首请求 TTFT=8990.639 ms、TPOT=227.353 ms、E2E=241573.832 ms，
usage=8192/1024、cached=0，但输出为句点和反复 EOS，计数 0/512，FAIL。
执行过程中观察到每卡约 69715 MiB，8 个 rank 仍在运行，没有观察到 OOM。
该记录只能作为未预热异常对照，不能列作合格的稳态性能。

第二个请求 cached=5248，TTFT=952.035 ms、TPOT=225.468 ms，
仅输出 `1 2` 后反复 EOS，FAIL。此后两次 64/64 都仅输出 `1`，严格校验失败。
文档 benchmark 对隐藏 EOS 的流式 TPOT 计算可能显示 0，**这是无效性能值**。
换成文档的背景文本型 8192 prompt 后，调度器报：

```text
Request prompt is too long, no enough memory to schedule a single sequence.
prompt_tokens=8192, pool_blocks=6272, block_size=128
```

所以 B 同时暴露了输出状态异常与新前缀的调度页不足，不能仅称为冷启动慢。

第一次重启尝试 `logs/repro_8192_fresh_20260914_1740/` 因旧服务的 collective
端口冲突未启动成功；旧服务退出并确认显存释放后，改用 18870/20870 独立启动。
重试目录拒绝覆盖的日志也保留。这些启动失败不纳入性能统计。

## 实验 C：文档原启动参数，当前二进制

日志：`logs/repro_8192_document_config_20260914_1806/`。
使用只读参考目录的 `start_dspark_tp8.sh` 和 `benchmark_dspark.py`，显式指定
本项目的二进制及日志目录；不再使用 A/B 的 custom launcher。
恢复默认 chunk/调度门槛，其余 TP8/EP1、DSpark N=5、Graph、Prefix、显存 0.8 不变。
实验 C 的 prompt 是文档的背景文本/变体型；与 A/B 不仅启动参数不同，
因此尚不能把结果差异单因素归因于 chunk 设置。

两次 64/64 预热均严格全文 PASS，再顺序执行 8 个 8192/1024 正式变体，
`prefix-cache-rate=0.65`、`count-to=512`，**8/8 均严格全文 PASS**，
usage 精确。全部原始数据保留；预先指定热窗为 variant 2--7，不剔除窗口内样本。

| variant | cached tokens | TTFT ms | TPOT ms | E2E ms | 严格全文 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 1474.868 | 8.009 | 9668.439 | PASS |
| 1 | 5248 | 958.204 | 8.018 | 9160.700 | PASS |
| 2 | 5248 | 539.539 | 8.164 | 8891.839 | PASS |
| 3 | 5248 | 539.077 | 8.129 | 8854.758 | PASS |
| 4 | 5248 | 541.381 | 8.146 | 8875.179 | PASS |
| 5 | 5248 | 540.191 | 8.105 | 8831.754 | PASS |
| 6 | 5248 | 540.156 | 8.069 | 8794.433 | PASS |
| 7 | 5248 | 537.316 | 8.106 | 8829.643 | PASS |

热窗均值：TTFT=539.610 ms，TPOT=8.119833 ms。
按各请求 E2E 之和计算 Input/Output/Total TGS=926.040/115.755/1041.795 tok/s。
这些 TGS 不含独立 Python 进程之间构造 prompt 的间隔；连续重放另行记录。
全部 8 个请求均值 TTFT=708.8415 ms、TPOT=8.09325 ms，不与热窗混用。

### 连续热态重放：实际整组墙钟吞吐

随后用 `tools/replay_dspark_perf.py` 预先载入 variant 2--7 的完全相同 payload，
在单进程中依次发送，中间不重新构造 prompt；计时包含 6 个 HTTP 请求及请求间开销，
不包含模型加载、预热、读取输入 JSON 和最终落盘。

证据：`logs/repro_8192_document_config_20260914_1806/continuous_replay_2_7/summary.json`。

| 指标 | 数值 |
| --- | ---: |
| global batch / total requests | 1 / 6 |
| 输入/输出 tokens（每请求） | 8192 / 1024 |
| 实际 cached tokens（每请求） | 5248 |
| TTFT_AVG | 537.931 ms |
| TPOT_AVG | 8.093630 ms |
| 6 请求总耗时 | 52.909430 s |
| Input Token Throughput | 928.984 tok/s |
| Output Token Throughput | 116.123 tok/s |
| Total Token Throughput | 1045.107 tok/s |
| 严格计数、精确 usage | 6/6 PASS |

计算：输入总数 49152、输出总数 6144、总数 55296，各除以 52.909430 秒。
以上均为整个 TP8 服务的吞吐，未除以卡数；输入包含缓存命中的 token。
如需只看 decode 阶段的单请求等效速度，`1000/8.093630 ≈ 123.554 tok/s`，
不要把它当作表中的 Output Token Throughput。
正式 8 个加重放 6 个，共 14/14 严格 `1..512` 通过，不代表完整 GSM8K 准确率。

### 实验 C 复现命令

服务终端，保持原文档默认参数，仅指定目标目录、二进制、模型和独立端口：

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
export DSPARK_ROOT=$PWD
export DSPARK_BIN=$PWD/build/dspark-musa-local/xllm/xllm
export DSPARK_RUN=$PWD/logs/repro_8192_document_new
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export DSPARK_SCRIPTS=/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark/logs/dspark-repro-scripts.5vTKvK
export XLLM2_EXTERNAL_TP8_HTTP_PORT=18910
export XLLM2_EXTERNAL_TP8_MASTER_ADDR=127.0.0.1:20910
bash "$DSPARK_SCRIPTS/start_dspark_tp8.sh"
```

客户端终端设定相同 `DSPARK_RUN`、`DSPARK_SCRIPTS`、`MODEL_PATH` 后执行：

```bash
set -euo pipefail
url=http://127.0.0.1:18910/v1/chat/completions
curl -fsS http://127.0.0.1:18910/health
for i in 1 2; do
  python3 -B "$DSPARK_SCRIPTS/benchmark_dspark.py" --url "$url" \
    --model-path "$MODEL_PATH" --prompt-tokens 64 --output-tokens 64 \
    --count-to 32 --output "$DSPARK_RUN/warmup_$i.json"
  python3 -B "$DSPARK_SCRIPTS/check_result.py" "$DSPARK_RUN/warmup_$i.json"
done
for i in 0 1 2 3 4 5 6 7; do
  python3 -B "$DSPARK_SCRIPTS/benchmark_dspark.py" --url "$url" \
    --model-path "$MODEL_PATH" --prompt-tokens 8192 --output-tokens 1024 \
    --count-to 512 --prefix-cache-rate 0.65 --prompt-variant "$i" \
    --output "$DSPARK_RUN/formal_variant_$i.json"
  python3 -B "$DSPARK_SCRIPTS/check_result.py" "$DSPARK_RUN/formal_variant_$i.json"
done
python3 -B tools/replay_dspark_perf.py --url "$url" \
  --input-dir "$DSPARK_RUN" --output-dir "$DSPARK_RUN/continuous_replay_2_7"
```

重放工具另用合成 SSE 验证了正常输出、无可见输出、SSE error 三个分支（3/3）；
这类脚本单测与上面的真实 GPU 实测分开计数。

## A/B 启动与单请求复现（诊断配置，不是推荐高性能命令）

服务（请先确认无其他服务/占卡进程；不执行 GPU reset）：

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
export DSPARK_BIN=$PWD/build/dspark-musa-local/xllm/xllm
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export XLLM2_EXTERNAL_TP8_HTTP_PORT=18870
export XLLM2_EXTERNAL_TP8_MASTER_ADDR=127.0.0.1:20870
export XLLM2_EXTERNAL_TP8_PREFILL_THRESHOLD=1.0
# 每次改为新的目录，启动器拒绝覆盖已有 manifest。
bash tools/start_dspark_custom.sh logs/repro_8192_new_run 10240 0.8
```

另一个终端，服务 ready 后，先将下面命令改成 64/64、count-to=32 做两次预热，
然后执行 8192/1024，每次使用新的输出文件名：

```bash
python3 -B tools/benchmark_dsv4_fast_exact.py \
  --url http://127.0.0.1:18870/v1/chat/completions \
  --model-path /data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt \
  --prompt-tokens 8192 --output-tokens 1024 --count-to 512 \
  --show-special-tokens --output logs/repro_8192_new_run/request_1.json
```

## 验证边界

文档配置的 8192/1024 已在本次独立服务中完成预热及 8/8 严格计数验证；
缓存未命中的首个正式请求也正确。TPOT <14 ms，但此输入长度下 TTFT 不满足 <250 ms。
A/B 的长 chunk 实验配置以及无预热冷启动仍有异常，不能因 C 通过就宣称均已修好。
尚无证据把异常直接归因于 MATE kernel、硬件损坏或某一个 state buffer；
需要独立控制 Graph 首次 capture、前缀恢复和请求间状态后再作归因。
30K/1024 的原 goal 仍在进行，不能由本轮 8192 结果宣称已完成。

## 测试结束资源状态

18:15 后停止测试服务，确认 8 卡显存归零，再恢复 0--7 卡占卡程序。
占卡 PID=19977，每卡申请 70 GiB；18:17 的 `mthreads-gmi` 为每卡 71834 MiB。
占卡日志：`logs/gpu_reservation_after_8192_20260914.log`。没有执行 GPU reset。
部分 rank 在退出清理时记录 `Server CollectiveServer0 doesn't exist`，
发生在全部性能请求结束之后，不是请求执行期间的错误。
