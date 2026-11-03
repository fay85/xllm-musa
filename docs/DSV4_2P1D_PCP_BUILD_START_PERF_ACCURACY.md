# DeepSeek-V4 Flash：2P1D + PCP 构建、启动、测试

适用：90/94/96 已安装依赖、挂载 `/data/nfs_shared` 的现有容器；不是空白机器安装教程。源码基于 `02c441f4d` 加本次 executor 修复。下列 Bash 命令按标明的位置执行。

| 主机 | 容器 | 角色 |
|---|---|---|
| 10.121.38.94 | wjm_xllm1008 | Prefill A，Attention CP2×TP4、MoE TP8/EP1 |
| 10.121.38.96 | wjm_xllm | Prefill B，同上 |
| 10.121.38.90 | wjm_xllm1008 | Decode TP8/CP1、DSpark=6；master、etcd |

每个角色为 8 个单卡进程，使用 `--nnodes=8 --node_rank=0..7`。两个 P 是独立副本，每个 P 内执行 PCP。正式配置为 Graph 开、P overlap 关、D overlap 开、KV PULL、prefix cache 开。`0.8` 是复用的 token 比例；P/D 显存预算分别为 0.85/0.81。

## 1. 编译（94 宿主机进入容器）

```bash
docker exec -it wjm_xllm1008 bash
export ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
cd "$ROOT/source"
git status --short
git rev-parse HEAD
git submodule update --init --recursive
bash tools/musa_deepseek_v4_2p1d/build_site.sh prefill >"$ROOT/build_prefill_repro.log" 2>&1
bash tools/musa_deepseek_v4_2p1d/build_site.sh decode >"$ROOT/build_decode_repro.log" 2>&1
sha256sum "$ROOT"/build_{prefill,decode}/xllm/xllm
cat "$ROOT"/build_{prefill,decode}/xllm/core/layers/musa/dsv4_pd_layer_selection.txt
```

先确认 `source` 已同步本次提交。[build_site.sh](../tools/musa_deepseek_v4_2p1d/build_site.sh) 给出完整环境、CMake 和 Ninja 命令，直接使用现有 site 路径，没有需要自行补齐的参数数组。关键设置：

- `USE_MUSA=ON`、`USE_CUDA=OFF`、`XLLM_DSV4_PD_ROLE=prefill|decode`；分开 build 目录。
- `MUSAMAPPING_PATH` 与两项 `CMAKE_*_COMPILER` 都指向角色 wrapper；缺少前者会改回系统 wrapper。
- 固定 TorchMusa、muDNN 3.4、MCCL 头文件/库、TVM FFI、vcpkg。90 的 `wjm_xllm` 不等于 `wjm_xllm1008`，前者缺少本配置要求的 muDNN 目录。
- Mooncake 的嵌套子模块也需初始化。依赖的 `USE_ETCD=OFF/STORE_USE_ETCD=OFF` 不代表 PD 无需外部 etcd。

这是现有共享目录的复现配置；移植路径还需修改 wrapper 内的绝对路径。不要跨角色复制 CMake cache/rules；P/D 的 `libasio.so` 也必须配套。

master/etcd 复用现有二进制，不属于这里的 xLLM P/D 编译目标：`baseline_off/plans/control.json` 指向 `control_relocation_20261008/build_r1/xllm_master_serving`（SHA256 前缀 `244fd53e85a3`），`etcd.json` 指向 `ops/baseline_config/bin/etcd`（`17d19f93c291`）。下节保留其完整启动参数；本次没有重新构建 master/etcd。

## 2. 生成新的启动计划（94 容器）

保留已验证的完整 argv/env，复制本次二进制、更新 SHA，创建新 etcd 数据目录。联合投影默认开；`JOINT=0` 用于关闭对照。旧外置 custom-allreduce preload 不可与当前集成实现重复加载。

```bash
export ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
export TEMPLATE=/data/nfs_shared/jd/pcp_resume_20261025/joint_precision_fix_20261026/runtime_precise_joint_on_20261029
export RUN=$ROOT/repro_$(date +%Y%m%d_%H%M%S)
export JOINT=1
python3 - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
root, template, run = (Path(os.environ[k]) for k in ('ROOT', 'TEMPLATE', 'RUN'))
(run / 'plans').mkdir(parents=True, exist_ok=False)
(run / 'bin').mkdir()
for role in ('prefill', 'decode'):
    shutil.copy2(root / f'build_{role}/xllm/xllm', run / 'bin' / f'xllm_{role}')
for name in ('etcd', 'control', 'p94', 'p96', 'd90'):
    profile = root / 'baseline_off' if name in ('etcd', 'control') else template
    plan = json.loads((profile / 'plans' / f'{name}.json').read_text())
    for record in plan['processes']:
        if name in ('p94', 'p96', 'd90'):
            role = 'decode' if name == 'd90' else 'prefill'
            record['argv'][0] = str(run / 'bin' / f'xllm_{role}')
            env = record['environment']
            env['LD_PRELOAD'] = '/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so'
            env['XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION'] = os.environ['JOINT']
            env['LD_LIBRARY_PATH'] = str(root / f'build_{role}/mooncake-common') + ':' + env['LD_LIBRARY_PATH']
        if name == 'etcd':
            record['argv'] = [f'--data-dir={run}/etcd-data' if s.startswith('--data-dir=') else s for s in record['argv']]
        record['sha256'] = hashlib.sha256(Path(record['argv'][0]).read_bytes()).hexdigest()
    (run / 'plans' / f'{name}.json').write_text(json.dumps(plan, indent=2) + '\n')
print(run)
PY
printf '%s\n' "$RUN" > "$ROOT/repro_current.txt"
```

启动器使用 **plan 内的 env**，仅在 shell 中 export 不会覆盖 plan。修改开关须改全部对应 rank 的 plan 后重启。端口占用时先按第 6 节停止对应实验；不要复用部分启动的目录。

## 3. 启动与健康检查

90 宿主机：

```bash
docker exec -i wjm_xllm1008 bash <<'SH'
set -e
ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
RUN=$(cat "$ROOT/repro_current.txt")
M=$ROOT/evidence_final/runtime_tools/run_manifest.py
python3 "$M" start "$RUN/plans/etcd.json" "$RUN/etcd"
for attempt in $(seq 1 30); do
  curl -fsS --max-time 2 http://10.121.38.90:24379/health && break
  sleep 1
done
curl -fsS --max-time 2 http://10.121.38.90:24379/health
python3 "$M" start "$RUN/plans/control.json" "$RUN/control"
python3 "$M" start "$RUN/plans/d90.json" "$RUN/d90"
SH
```

94 宿主机：

```bash
docker exec -i wjm_xllm1008 bash <<'SH'
set -e
ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
RUN=$(cat "$ROOT/repro_current.txt")
python3 "$ROOT/evidence_final/runtime_tools/run_manifest.py" start "$RUN/plans/p94.json" "$RUN/p94"
SH
```

96 宿主机（注意 Python 路径）：

```bash
docker exec -i wjm_xllm bash <<'SH'
set -e
ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
RUN=$(cat "$ROOT/repro_current.txt")
/usr/bin/python "$ROOT/evidence_final/runtime_tools/run_manifest.py" start "$RUN/plans/p96.json" "$RUN/p96"
SH
```

加载完成后，在可访问三台机器的容器检查：

```bash
for url in http://10.121.38.90:24888 http://10.121.38.90:25882 http://10.121.38.94:25881 http://10.121.38.96:25881; do
  curl --fail --max-time 5 "$url/v1/models"
done
curl --fail --max-time 120 http://10.121.38.90:24888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash-0731-FP8-mt","messages":[{"role":"user","content":"Output only consecutive integers from 1 through 10 separated by one ASCII space."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"thinking":false}}'
```

必须检查实际推理文本和 usage；需要 24 个 rank 存活且两个 P 都收到工作，才称为 2P1D。

## 4. 性能（94 容器）

```bash
export ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
cd "$ROOT/source"
export MODEL_DIR=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export API_URL=http://10.121.38.90:24888

# 20K/100，prefix 0.8，并发逐档 1～16
export PERF=$ROOT/perf20k_$(date +%Y%m%d_%H%M%S)
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" \
  --prepare-only --input-tokens 20000 --max-concurrency 16 \
  --base-url "$API_URL" --model DeepSeek-V4-Flash-0731-FP8-mt \
  --tokenizer "$MODEL_DIR" --output-dir "$PERF"
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" --execute-prepared --output-dir "$PERF"
cat "$PERF/summary.json"

# 40K/100，prefix 0.8，并发逐档 1～5
export PERF=$ROOT/perf40k_$(date +%Y%m%d_%H%M%S)
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" \
  --prepare-only --input-tokens 40000 --max-concurrency 5 \
  --base-url "$API_URL" --model DeepSeek-V4-Flash-0731-FP8-mt \
  --tokenizer "$MODEL_DIR" --output-dir "$PERF"
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" --execute-prepared --output-dir "$PERF"
cat "$PERF/summary.json"
```

每组为两个冷 primer、两个热探针，再升序/降序 **两轮**，每档每轮 32 条。20K/40K 正式请求分别为 1024/320 条，缓存 token 必须精确等于 16000/32000。脚本校验 HTTP/SSE/DONE、100 输出 token、严格计数文本和 usage；失败即停止，不重试、不删慢样本。各档 `valid` 与总 `passed` 应为 true。

TTFT = 首个非空 content 时间 − 请求发出时间；TPOT = 首末 content 间隔 ÷ (实际输出 token − 1)；聚合输出吞吐 = 输出 token 总数 ÷ 整档共同墙钟时间。均值不是 P99/最大延迟保证。比较需固定版本、开关、预热与路由；核查 P 日志 `DSV4_PCP_FINISH`，不能把 TP 回退计为 PCP。

## 5. GSM8K 100 题（94 容器、Graph 正式配置）

```bash
export ROOT=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026
cd "$ROOT/source"
export GSM=$ROOT/gsm100_$(date +%Y%m%d_%H%M%S)
export ASSETS=/data/nfs_shared/jiaming.wang/pd_n6q7_r3_20260927/acceptance_assets_20260927/gsm8k_frozen
python3 tools/musa_deepseek_v4_2p1d/gsm8k.py prepare --output-dir "$GSM" \
  --assets "$ASSETS" --tokenizer /data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt \
  --base-url http://10.121.38.90:24888 --model DeepSeek-V4-Flash-0731-FP8-mt
python3 tools/musa_deepseek_v4_2p1d/gsm8k.py execute --output-dir "$GSM"
python3 tools/musa_deepseek_v4_2p1d/gsm8k.py score --output-dir "$GSM"
cat "$GSM/summary.json"
```

固定 seed=20261026，100 道相同样本；短输入与 40K/32K-cache 各一次，并发 5、greedy、thinking=false、最大输出 1024。无重试，推理阶段不加载 gold。最后一个 balanced `\boxed{...}` 做精确有理数比较，要求正常 stop 和协议通过。完整响应、错误索引均保留；小样本结果不能代表全量 GSM8K。

## 6. 停止本轮服务

在角色所属宿主机执行对应命令；启动器校验 PID/start ticks/exe/argv，不按名称批量杀进程。先保存当前实验路径再生成下一轮，避免指针文件切换后停止错轮次。

```bash
# 94 宿主机
docker exec wjm_xllm1008 bash -c 'R=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026; D=$(cat "$R/repro_current.txt"); python3 "$R/evidence_final/runtime_tools/run_manifest.py" stop "$D/p94/processes.json"'
# 96 宿主机
docker exec wjm_xllm bash -c 'R=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026; D=$(cat "$R/repro_current.txt"); /usr/bin/python "$R/evidence_final/runtime_tools/run_manifest.py" stop "$D/p96/processes.json"'
# 90 宿主机
docker exec wjm_xllm1008 bash -c 'R=/data/nfs_shared/jd/pcp_resume_20261025/refine_merge_20261026; D=$(cat "$R/repro_current.txt"); for role in d90 control etcd; do python3 "$R/evidence_final/runtime_tools/run_manifest.py" stop "$D/$role/processes.json"; done'
```

## 7. Graph-off 问题

真实 MUSA 二进制已重现 `backend=llm` 返回 VLM executor；排除测试自身注册后，工厂类型测试也失败。原宏使用嵌套匿名 lambda；修复改用带类名的 creator 函数指针，避免当前混合编译/链接环境中的错误绑定。同时在 VLM 中保留原始模型指针；文本模型误入时直接构造 `BaseExecutorImpl`，避免空指针和递归查找。

验证：P/D 编译链接成功；工厂回归在旧注册实现上失败、修复后 3/3 通过。96 上 Graph-off、TP8/CP1、target-only 独立服务的 8 个 rank 存活，3 次严格 100-token 计数通过，VLM 降级日志从 8 条降到 0 条。额外小检查修复前后都为 2/3（`17+25` 返回 `34`），因此只确认本次崩溃/映射修复，不宣称 eager 模型精度通过。测试后恢复 P96 原 Graph 配置。

原始证据：`$ROOT/executor_fix_20261027/`，包括 `registration_red.xml`、`registration_green.xml`、`eager_fixed_summary.json`、实际进程参数和构建日志。20K/40K 客户端准备分别校验 1028/324 条请求，GSM8K 准备得到 202 条请求（100 短题+2 primer+100 长题）；这是工具验证，本次未重新完成全部性能矩阵或 GSM100 推理。正式性能/精度按第 4、5 节独立验收。
