<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# DeepSeek-V4 DSpark 1P1D Guard：完整复现手册

日期：2026-10-06。发布分支：`perf/dspark-pd-30k-b16-20261006`。

本次发布的是 **1P1D、8 卡 Prefill + 8 卡 Decode** 的实验验证快照，
不是单机 8 卡结果，也不是把最新单机二进制直接同时用于 P/D。
包含实现源码、角色差异、增量编译配方、启动脚本、性能/精度评测方法和哈希证据。

## 1. 已验证结果及统计口径

30,000 输入 / 1,024 输出，DSpark N=6、target query width=7，Indexer TopK=512。
同一原始数据集、每组预热 16 次、正式 64 次，按 B1 → B16 顺序测试。
每组重新启动 Prefill，Decode 和 Control 不变；服务端容量始终为 16。

| 正式测试 | 客户端并发 | 严格输出 | TTFT 均值 | TPOT 均值 | 实际输出吞吐 |
|---|---:|---:|---:|---:|---:|
| B1 | 1 | 64/64 | 2,196.170 ms | **8.779710 ms** | **91.535690 tok/s** |
| B16 | 16 | 64/64 | 15,129.774 ms | 19.853986 ms | **428.193054 tok/s** |

吞吐比 **4.67788085 倍**，达到 B1 TPOT <9 ms、B16 吞吐 ≥4.5× 的目标。
B1 正式样本 TPOT 范围 8.504149～8.973826 ms。

- 吞吐 = 实际完成的 output tokens / 整轮客户端 wall time，不用 `并发数 / TPOT`。
- 两组分别完成 65,536 个输出 token，正式耗时 715.961170 s / 153.052460 s。
- B16 指客户端并发上限，不代表每个 Decode step 的实际 batch 都是 16。
- 所有正式请求 `cached_tokens=0`。虽然 Prefix Cache 开启，但这不是热前缀命中成绩。
- 严格校验完整 512 个递增整数、输入 30,000、输出 1,024；无重试、无删题、无请求错误。
- B16 吞吐提高不等于单请求 TPOT <10 ms；当前 B16 单请求 TPOT 约 19.85 ms。
- 纯计数通过不代表通用长上下文或 GSM8K 自动通过。GSM8K 另见独立报告。

机器可读证据：[paired_audit.json](evidence/pd_guard_20261006/paired_audit.json)、
[冻结计划](evidence/pd_guard_20261006/plan.json)。

发布前同一实测 P/D 组合重新完成 GSM8K 全量：**1275/1319 = 96.66%**，
0请求错误、2题达到输出长度上限，使用 SGLang 原版评分器；不是由计数测试推断。
详见 [GSM8K 报告](DSV4_PD_GUARD_GSM8K_20261006.md)。

## 2. 机器、容器和路径

| 角色 | 机器 / 容器 | 用途 |
|---|---|---|
| Prefill | `10.121.38.94 / wjm_xllm` | 8 卡，长输入计算与状态发送 |
| Decode | `10.121.38.90 / wjm_xllm` | 8 卡，DSpark 校验与输出 |
| Control + etcd | `10.121.38.92 / cmz` | CPU 路由/协调，不计 GPU |
| 编译 | `10.20.34.28 / xllm_musa_build_0914` | 仅 CPU 编译、复用依赖 |
| GSM8K 离线评分 | `10.20.34.28 / wjm_sg_dspark` | CPU 官方评分器，不启动 GPU 服务 |

90/94 镜像均为 `registry.mthreads.com/presale/devtech/xllm:wjm0811`，image ID：
`sha256:616f32d488c1c2b7cd5662080a5f7765914f3c7042341c4f0c379709e19f24d9`。
容器内另有已部署依赖，**只拉这个镜像 tag 不等于完整复现环境**。

```bash
# 下文 Linux 控制端公共变量；REPO 是本次新 checkout，不是旧开发目录。
export REPO=/data/nfs_shared/jiaming.wang/xllm_pd_guard_release_20261006
export ARTIFACTS=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005
export COMMON=/data/nfs_shared/jiaming.wang/pd_n6q7_r3_20260927
export MODEL=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export CASES=$ARTIFACTS/logs/dspark_30k_1k_cases_20260926/cases_pd_model_id.jsonl
export RUN=/data/nfs_shared/jiaming.wang/pd_guard_repro_$(date +%Y%m%d_%H%M%S)
```

28 的 `/data/nfs_shared` 和 90～96 的共享盘**不是同一个文件系统**，传输必须显式 copy。
Windows 发布 checkout 是 `C:\Users\jiaming.wang\code\xllm-musa-pd-30k-release-20261006`，
仅用于收集源码/文档和 Git 推送，不是 Linux 运行目录。

### 固定产物

| 用途 | 路径 | SHA256 |
|---|---|---|
| 实测 P | `$ARTIFACTS/candidates/indexer_mask_once/xllm` | `caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43` |
| 实测 D | `$ARTIFACTS/candidates/reopen_decode/xllm` | `5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93` |
| Control | 92/cmz：`/workspace/xllm-service/build-cmake331-native/xllm_service/xllm_master_serving` | `cd18acb99e2f6415a2a83c3791e9bee93a888e132956ca63a54aed0bb7a0f627` |
| 模型配置 | `$MODEL/config.json` | `9a387b757dc12751447ec0f450a30929b3bfab17f2ee4b9a9a3372300c373fa4` |
| 30K 输入 | `$CASES` | `f60ba0784dfd287d4aca59e9e74972f5c0b65bb33d9fa4be6a5551587d153d09` |

AOT：
`/data/nfs_shared/jiaming.wang/gsm8k_goal_20260919_1555/layerdiff_20260921/report_deps_20260925/cached_ops`。
完整 SO 清单在 `tools/pd_guard_20261006/frozen/reference_n6q7.ops_artifacts.sha256`，
清单自身 SHA：`c9f9a0b32cb87c33e3edcce537c42f82cb3ec05d3c6bac82659151f35df29d21`。
启动器逐文件校验并重绑定根目录，不能只换一个 `.so` 或混用另一份 `libtvm_ffi`。

## 3. 拉取和编译

```bash
git clone --single-branch --branch perf/dspark-pd-30k-b16-20261006 \
  https://sh-code.mthreads.com/his-devtech/xllm-musa.git "$REPO"
cd "$REPO"
git rev-parse HEAD
```

若服务器无法访问 GitLab，在 Windows clone 后，排除 `.git` 中的本地凭据配置，
使用 `git archive` 打包已提交版本再中转到共享盘；不要覆盖原开发目录。

### 3.1 不重编：复现已公布成绩的推荐方式

直接使用上表的 P/D 实测二进制，先做 SHA 校验，再按下一节启动。
二进制、模型、AOT、依赖缓存未进入 Git，需保留共享盘产物。

### 3.2 独立增量重编：复用已有依赖

在 **28 / xllm_musa_build_0914** 容器中，将本分支复制到一个新目录，例如：

```bash
export BUILD_REPO=/tmp/xllm_pd_guard_release_20261006
export BUILD_OUT=/tmp/pd_guard_rebuild_$(date +%Y%m%d_%H%M%S)
cd "$BUILD_REPO"
python3 tools/pd_guard_20261006/build_reuse.py \
  --role prefill --output "${BUILD_OUT}_prefill" --jobs 3
python3 tools/pd_guard_20261006/build_reuse.py \
  --role decode --output "${BUILD_OUT}_decode" --jobs 2
```

脚本先校验原始 archive/object/SO 的 SHA，再重编 PD 相关翻译单元、替换私有 archive 副本、链接。
不会修改原缓存、源服务或原二进制；输出目录必须不存在。每个对象保留 `.command`、`.log`、
`.d`，并生成 `link.command`、`link.log`、`result.json`。

本次已验证：P 12 个、D 5 个翻译单元完成增量编译及链接。
新 P SHA 为 `0d40f1c0b681a8f204da487b7afea940dbabde3067bb89ba001fb5e716bfef78`；
新 D SHA 为 `37b6eafca02373003a473eb8808ae84f5dbb2a2382ac85d99314be35605d9fa9`。
这两个是**编译验证产物，尚未独立进行服务精度/性能复测**，不能套用上表成绩。
编译证据位于 `docs/evidence/pd_guard_20261006/pd_release_*_build_result_20261006.json`。

依赖/边界：

- 原 cache：`/data/nfs_shared/jiaming.wang/xllm0914/xllm-musa/build/dspark-musa-local`。
- 实际 CMake cache：`USE_MUSA=ON`、`USE_CUDA=OFF`；C++/CUDA compiler 接口均指向
  `/usr/local/musa/tools/musamapping/mcc_wrapper`。本版不需要 NVIDIA `nvcc`。
- vcpkg include cache：`/data/nfs_shared/jiaming.wang/xllmmerge/xllm/build/musa-standalone-db33797/vcpkg_installed`；
  链接还复用 `musa-current-main/vcpkg_installed` 中的固定库。具体每个输入以 JSON 配方为准。
- Mooncake 使用本次已验证的匹配 headers/archives；不要整体替换成接口不匹配的新 Mooncake。
- `pd_release_build_recipes_20261006.json` 记录原始编译参数及链接依赖哈希，必须保留这些依赖位置。
- 本脚本是**现有开发环境的增量复现**，不是声称在空镜像上无缓存全量 CMake 构建已通过。
- `xllm/` 保存 Prefill 版本，`tools/pd_guard_20261006/decode_overlay/` 保存 Decode 差异。
  编译工具自动选择；不能手工遗漏 overlay 后宣称二者都对应原实测版本。

## 4. 运行前检查

以下检查在 Linux 控制端执行，SSH 认证使用自己的已有配置，文档不包含密码。

```bash
for host in 10.121.38.90 10.121.38.94; do
  ssh "$host" 'mthreads-gmi; df -h / /data/nfs_shared; docker ps --format "{{.Names}} {{.Status}}"'
  ssh "$host" 'docker exec wjm_xllm bash -lc "ps -eo pid,args | grep -E \"[x]llm|[v]llm|[s]glang\"; ss -lntp"'
done
ssh 10.121.38.92 'docker exec cmz bash -lc "curl -fsS http://127.0.0.1:2489/health; ss -lntp"'
```

确认 P/D 各 8 卡空闲，没有其他容器占用；仅容器内检查不足以证明宿主机 GPU 空闲。
端口：Control 29888/29889、etcd 2489；P HTTP18861+、master20861/20862、transfer26000+、PD7777+；
D HTTP18862+、master20862/20863、transfer26100+、PD7787+（分属不同机器）。

端口/显存被占用时先识别具体任务，不自动 kill。**全程禁止 reset GPU、killall、泛化 pkill**。
如果复用正在运行且 hash/config 相符的服务，可跳过启动；不要执行第二套相同端口服务。

## 5. 启动顺序：Control → Decode → Prefill

### 5.1 Control（92 / cmz）

已健康运行的 etcd 不重建、不清空 key/data-dir。当前 etcd binary 位于
`/workspace/etcd-v3.7.1/etcd`，client2489、peer2490，data-dir
`/tmp/dspark-pd-goal-20260915-etcd/data`。如 etcd 未启动，只在确认该实例已停止后，用现存
实例配置恢复；不要对仍在使用的数据目录另起一个 etcd。

Control 是独立 `xllm-service` 项目，不在此 xLLM 分支内；基础提交
`322bcda03793bd28991080d63df559a1b646fb73`，本次按已验证 binary SHA 复用。
生产测试中 `XLLM_SERVICE_ASSUME_PRELINKED_PD` **未启用**，不能跳过真实 LinkInstance。

```bash
# 只有旧 Control 已停止且端口空闲时执行。
ssh 10.121.38.92 \
  "docker exec -d cmz bash '$REPO/tools/pd_guard_20261006/start_control.sh' '$RUN/control'"
```

### 5.2 Decode（90 / wjm_xllm）

```bash
ssh 10.121.38.90 \
  "docker exec -d -u 0 wjm_xllm bash '$REPO/tools/pd_guard_20261006/start_role.sh' DECODE '$RUN/decode'"
ssh 10.121.38.90 \
  "docker exec wjm_xllm bash -lc 'tail -n 40 \"$RUN/decode/rank0.log\"; cat \"$RUN/decode/pids.tsv\"'"
```

确认 8 个 rank 完成加载，18862 HTTP 可访问，所有 rank 没有 FATAL/OOM，再启动 Prefill。

### 5.3 Prefill（94 / wjm_xllm）

```bash
ssh 10.121.38.94 \
  "docker exec -d -u 0 wjm_xllm bash '$REPO/tools/pd_guard_20261006/start_role.sh' PREFILL '$RUN/prefill_b1'"
ssh 10.121.38.94 \
  "docker exec wjm_xllm bash -lc 'tail -n 40 \"$RUN/prefill_b1/rank0.log\"; cat \"$RUN/prefill_b1/pids.tsv\"'"
curl -fsS http://10.121.38.92:29888/v1/models
```

必须等 Control 中 P/D 注册并完成连接。`/v1/models` 成功只是健康检查；
下一节预热请求的完整严格校验通过，才算链路实际可用。
若注册失败，保留 Control 和两侧日志，检查 endpoint/端口；不通过跳过校验来“修复”。

### 固定关键配置

| 参数 | P | D |
|---|---|---|
| 并行拓扑 | TP8/EP1/DP1/CP1 | TP8/EP1/DP1/CP1 |
| DSpark | N6 / targetq7 | N6 / targetq7 |
| Graph | true | true |
| Schedule overlap | false | true |
| Prefix cache | true | true |
| 容量 / 并发上限 | 16 / 16 | 16 / 16 |
| token budget / prefill chunk | 16384 / 16384 | 10032 / 8192 |
| 显存比例 | 0.81 | 0.81 |
| 传输 | Mooncake PULL | Mooncake PULL |

完整环境不是只有这几行：参考 `frozen/reference_v128.launch_manifest.txt`
中的路由变量及脚本中的 Prefill override，启动后保存 `launch_manifest.txt`。
实测配置副本：[P](evidence/pd_guard_20261006/prefill_launch_manifest.txt)、
[D](evidence/pd_guard_20261006/decode_launch_manifest.txt)。

P 保留 `enable_online_profile=true` 以允许诊断接口，但正式性能/GSM8K 测试期间
未开启 profile 采集；不要在测试过程中调用 profiling API。配对审计检查了无活动 profile 产物。

## 6. 30K/1024：B1 和 B16 性能、严格输出复现

只在没有其他请求/评测时运行。性能测试和 GSM8K **不能同时跑**。

```bash
export LABEL=guard_$(date +%Y%m%d_%H%M%S)
# 在 90 容器执行客户端；输出保存在 REPO/logs 下。
ssh 10.121.38.90 \
  "docker exec wjm_xllm bash '$REPO/tools/pd_guard_20261006/bench_30k.sh' b1 '$LABEL'"
```

该命令执行原题索引 304～319 预热、320～383 正式测试，冻结输入校验通过后才能发请求。
每个阶段后读取 raw response，校验完整文本、usage、cached=0 和吞吐。

然后**仅重启本轮 Prefill**，D/Control 不动：

```bash
ssh 10.121.38.94 \
  "docker exec wjm_xllm bash '$REPO/tools/pd_guard_20261006/stop_role.sh' '$RUN/prefill_b1'"
# 手工确认旧 8 rank 已退出、显存释放、端口空闲；未退出不要直接启动。
ssh 10.121.38.94 \
  "docker exec -d -u 0 wjm_xllm bash '$REPO/tools/pd_guard_20261006/start_role.sh' PREFILL '$RUN/prefill_b16'"
# 等 8 rank 加载完成且 Control 重新链接后执行。
ssh 10.121.38.90 \
  "docker exec wjm_xllm bash '$REPO/tools/pd_guard_20261006/bench_30k.sh' b16 '$LABEL'"
```

输出：`logs/${LABEL}_b{1,16}_{warmup,formal}/responses.raw.jsonl`、`summary.json`、
`attempts.jsonl`、`config.json`，及仓库根目录的 `*_audit.json`。
原压力脚本 summary 的历史 `throughput_scope` 文本写成 8GPU；本 1P1D 实验必须按
**16 GPU aggregate** 解读，`audit_counting.py` 与本报告明确修正该说明，未改变计算公式。

最终验收还需比对两组 `launch_manifest.txt`、binary/model/AOT hashes，确认只改客户端并发、
固定相同 64 个题目、无重试，且 B16/B1 实际吞吐 ≥4.5，B1 平均 TPOT <9 ms。
原始结果在 `$ARTIFACTS/indexer_mask_once_pair_20261006/` 及对应 `logs/`；
不能只取单次最好值替代该配对实验。

### 可选：2.5K/1.5K

```bash
cd "$REPO"
python3 tools/deepseek_v4_stream_benchmark.py \
  --url http://10.121.38.92:29888/v1/chat/completions \
  --model DeepSeek-V4-Flash-0731-FP8-mt --model-path "$MODEL" \
  --input-tokens 2500 --output-tokens 1500 --count-to 750 \
  --timeout 600 --ignore-eos --record-stream-chunks \
  --output "$RUN/count_2500_1500.json"
```

记录首次/后续请求和实际 cached tokens，严格检查全文 `1..750`。本发布的主验收为
30K/1024；此前 2.5K/1.5K 的 222.6 ms / 8.95 ms 属于历史配置，**不能直接算作
本次 CAF/5EA 组合已经重新复测的结果**。

## 7. GSM8K 和故障处理

完整命令及当前结果见 [GSM8K 报告](DSV4_PD_GUARD_GSM8K_20261006.md)。
使用固定 1319 题、单轮 boxed prompt、max_tokens=1024、temperature=0、top_p=0.95、
thinking=false、并发1、零重试，以及 SGLang 的原版 MathEvaluator 离线评分。

- 单纯 HTTP200、无 NaN、计数通过都不能代替 GSM8K 分数。
- hang/OOM/错误输出：停止发送新评测请求，保留 rank/client/Control 日志，先记录精确 PID、exe 和 hash。
- 只停止本轮 `pids.tsv` 对应进程，脚本仅发 TERM；不 reset GPU、不自动杀其他容器。
- 更换 AOT、MCCL、TVM FFI、模型 TopK 或任一路由开关后，都属于新实验，需新目录、新哈希和重新验收。
- Prefix 命中/不命中、冷/热 Graph、客户端并发和实际模型 batch 必须分开记录。

## 8. 发布范围

来源：`b5cc2b94679db6770b7ed8b6e96d1cb922eaa042` + v128 冻结源码 + 已验证 PD 增量。
本分支不包含之后另一路 6× 吞吐实验的未验证改动。
本次只新增远端发布分支，不修改已有分支；原 Windows/Linux 脏工作区不做清理或覆盖。
实现细节见 [实现说明](DSV4_PD_GUARD_1P1D_IMPLEMENTATION_20261006.md)。
