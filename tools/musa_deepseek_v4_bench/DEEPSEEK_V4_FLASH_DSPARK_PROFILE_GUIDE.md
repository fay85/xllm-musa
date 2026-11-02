<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# DeepSeek-V4 Flash / DSpark Profile 采集说明

本文整理已有的两种 profile 采集方式，供定位算子、通信及调度耗时使用。文中的 Linux 脚本路径已在 2026-10-28 通过 90 机器只读确认存在；这不代表当前服务已启动或采集接口已经验证可用。

**Profile 用于诊断。采集期间的 TTFT、TPOT 和吞吐包含采集开销，不能作为正式性能验收结果。**

## 1. 脚本位置与用途

| 脚本 | 用途 | 地址使用方式 |
|---|---|---|
| `/data/nfs_shared/jd/xllm/tools/capture_dspark_profile.py` | 对现有服务采集一条固定请求 | 同一个服务地址负责开始采集、推理、停止采集 |
| `/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/profile_main_sp_prefill.sh` | 历史 1P1D、30K/1024 Prefill 采集 | 在 Prefill 开启采集，通过 Control 发送推理请求 |

两个脚本均不是服务启动脚本，也不会自动完成模型加载。

## 2. 服务端准备

使用支持 MUSA / DSpark 在线采集的已适配二进制，启动目标服务时增加或确认以下参数：

```bash
--enable_online_profile=true \
--profile_backend=torch \
--profile_dir=/实际的trace输出目录
```

- `enable_online_profile`：允许调用在线采集接口，不等于持续开启采集。
- `profile_backend=torch`：选择该版本的 Torch profiler 路径。
- `profile_dir`：服务端写入 trace 的目录，需在容器内可写且有足够空间；建议放在共享盘的独立实验目录。
- 应先确认服务正常推理，再进行短时间采集。需要热态诊断时，先在采集窗口之外完成预热。

**不能只凭参数存在就认定采集成功。** 部分旧 DSpark 版本的 HTTP `/start_profile` 曾返回 `start_profile is not implemented`，当时使用 worker `StartProfile` / `StopProfile` RPC。不同版本需要分别确认支持情况，不能直接照搬旧 worker 的动态端口。

## 3. 通用单请求采集脚本

脚本：

```text
/data/nfs_shared/jd/xllm/tools/capture_dspark_profile.py
```

其流程为：开始采集 → 发送一条固定请求 → 停止采集。即使请求失败，也会尝试执行停止采集，并保存错误信息。

### 参数

| 参数 | 说明 |
|---|---|
| `--url` | 服务根地址，不带 `/v1`；脚本自动拼接 `/start_profile`、`/v1/chat/completions` 和 `/stop_profile` |
| `--record` | 包含顶层 `request_payload` 字段的请求 JSON 文件 |
| `--output` | 采集控制记录 JSON；必须是该脚本所属仓库 `logs/` 下尚不存在的文件 |
| `--max-tokens` | 覆盖请求的输出上限，默认 128；当前实现要求为正偶数 |

脚本沿用递增计数测试的校验逻辑，以 `max_tokens / 2` 作为预期计数长度。例如 128 对应 `1..64`。因此应使用匹配的已验证计数请求记录，不能把任意自然语言请求直接当成严格正确性用例。

### 调用示例

在能访问服务与共享盘的 Linux 容器中执行。以下 `SERVICE_URL` 和 `RECORD` 是占位值，必须替换：

```bash
cd /data/nfs_shared/jd/xllm

SERVICE_URL=http://实际服务IP:端口
RECORD=/实际路径/已验证的请求记录.json
OUTPUT="/data/nfs_shared/jd/xllm/logs/profile_$(date +%Y%m%d_%H%M%S).json"

python3 tools/capture_dspark_profile.py \
  --url "$SERVICE_URL" \
  --record "$RECORD" \
  --output "$OUTPUT" \
  --max-tokens 128
```

保留同仓库的 `tools/replay_dspark_perf.py` 和 `scripts/logger.py` 等依赖，不建议只拷贝这一个脚本运行。

注意：该脚本只有一个 `--url`，**不是独立控制 P、D 和 Control 三个地址的 1P1D 编排脚本**。不能简单将它指向 Control，就假定 P/D 两端都会被采集。

## 4. 历史 1P1D、30K Prefill 采集脚本

Linux 脚本：

```text
/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/profile_main_sp_prefill.sh
```

Windows 本地副本：

```text
C:\Users\jiaming.wang\code\xllm-musa\.codex-pd-n6q7-20260927\recheck\profile_main_sp_prefill.sh
```

### 当时的采集流程

1. 检查前序实验的退出标记和验收记录。
2. 调用 `http://10.121.38.94:18861/start_profile`，开启 Prefill 采集。
3. 使用 `run_dspark_stress.py`，向 `http://10.121.38.92:29888/v1/chat/completions` 发送历史数据集的一条 30K/1024 请求：并发 1、`--start 840 --limit 1`。
4. 退出时通过 shell `trap` 调用 Prefill 的 `/stop_profile`，保存响应及退出状态。

**这个脚本只采集 Prefill，不同时采集 Decode。**

### 历史结果位置

采集控制记录：

```text
/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/profile_main_sp_prefill/
```

该目录用于保存 `start.json`、`stop.json`、错误信息、`client.log`、`exit` 等；推理结果另写入：

```text
/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/logs/profile_main_sp_prefill_one/
```

真正的各 rank trace 写入服务启动参数指定的 `--profile_dir`，不能将上述控制记录目录与 trace 目录混为一谈。

### 重新使用前必须调整

该脚本写死了旧 Prefill / Control 地址、前序验收文件、数据集及输出目录，**不能直接照跑**。应另存一份后：

- 核对当前 P、D、Control 的角色、地址和端口，不依据二进制名称判断角色。
- 按本次实验选择输入记录和前置检查；不要伪造旧验收标记来绕过检查。
- 为采集记录、推理结果和 trace 使用新的目录，保留原实验记录。
- 保留失败时也尝试停止采集的处理逻辑。

## 5. 1P1D 使用原则与结果检查

在 1P1D 中，**采集接口发给要分析的 P/D 服务，推理请求发给 Control**：

- 只分析 Prefill：仅开启 P 的采集。
- 只分析 Decode：仅开启 D 的采集。
- 两端都要分析：分别开启 P/D 采集，经 Control 发请求，再分别停止；需要保证异常时也停止已开启的采集。

采集完成后检查：

1. 开始、停止接口的响应正文是否成功，不能只检查 HTTP 200。
2. 推理输出是否正确，是否存在错误、截断或空输出。
3. `--profile_dir` 是否生成本次时间窗口的新 trace，服务日志是否有写出记录；需要 TP8 全 rank 分析时，核对各 rank 文件是否齐全。
4. 区分客户端 `--output` 记录与服务端 trace。Chrome trace 通常为 `.pt.trace.json`，可用 Perfetto 等工具查看。

本文仅整理现有脚本及使用边界，没有启动服务、修改运行配置或执行新的采集。
