# DeepSeek-V4-0731 MUSA DSpark 复现

## 已验证链路

- 工作目录：`/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark`
- 分支：`dsv4/feat/musa-dspark-20260910`
- 算法代码版本：`a9cd485c`。后续文档提交不改变该二进制。
- 二进制 SHA256：`78f4a5acff37b5353510bdf1b4d3a20aa8cb455c6b2e683f584b10d322eceee5`
- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`
- API 的 `model_id` 保留 `DeepSeek-V4-Flash-FP8` 作为接口标签，实际权重版本以以上 0731 路径和 manifest 为准。
- 8 卡 S5000，TP8/EP1/DP1/CP1，batch=1；DSpark 五个草稿 token。
- **target 六 token Graph + draft eager block**，不是整个 DSpark pipeline 都已 Graph 化。
- prefill SP；Prefix Cache 开启，输入共享率 0.65，实测命中 1536 tokens。
- schedule overlap 关闭；静态 greedy，adaptive/confidence 策略未启用。
- 输入/输出 2500/1500，严格 `1..750`；block size 128，max batch tokens 10240，memory utilization 0.8，response threads 4。

两个独立服务的完整热窗均为 variant 2–7：

| 服务 | 平均 TTFT / ms | 平均 TPOT / ms | 严格精度与缓存 |
| --- | ---: | ---: | --- |
| dspark_sp_prefix_07 | 217.370500 | 8.212000 | 6/6 全文 750/750 PASS，cached=1536 |
| dspark_sp_prefix_08_fresh | 215.821167 | 8.188667 | 6/6 全文 750/750 PASS，cached=1536 |

每个正式请求 usage=2500/1500，两窗均满足 TTFT<250 ms、TPOT<14 ms 且不高于参考门限 13.758500 ms。冷态/建缓存请求和全部失败实验另行保留，没有筛选热窗样本。

## 环境与构建

验证环境为宿主机 `10.20.34.28`、driver 5.2.0-server、MCC 5.1.0、Python 3.10、Torch/TorchMUSA 2.9.1。复用机器现有库，未下载 LibTorch/NVIDIA CUDA。

```bash
export DSPARK_ROOT=/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark
export DSPARK_TOOLS=/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_20260910
cd "$DSPARK_ROOT"

# 已有 build cache 时只做增量构建。
MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping MUSA_HOME=/usr/local/musa \
  cmake --build build/dspark-musa-local --target xllm -j16
sha256sum build/dspark-musa-local/xllm/xllm
```

若需要重新配置，先检查并使用 `$DSPARK_TOOLS/configure.sh`。它包含本机 Torch、MKL、ASIO、xxHash 和已有 vcpkg 路径。必须保留 `VCPKG_MANIFEST_MODE=OFF`、`VCPKG_MANIFEST_INSTALL=OFF`，不要对共享依赖树执行安装/替换。MUSA standalone 的 `USE_MUSA=ON`、`USE_CUDA=OFF` 保持不变；树外 toolkit-probe shim 只用于 CMake 探测，实际编译仍为 MCC/MusaMapping。

当前单体 target 的链接图仍包含 Mooncake，不能宣称已经完全跳过。构建失败、共享依赖恢复及检查记录见 [验证记录](DEEPSEEK_V4_MUSA_DSPARK_VALIDATION.md)。

还需要原性能链的本地 AOT 算子和 custom all-reduce：

- AOT 根目录：`/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops`
- custom-AR：`/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so`

本页下面已内嵌完整启动器和测试脚本，不再需要旧的 launch_eager.sh、benchmark_0731.py 或 worker2 提示词脚本。权重、AOT 和动态库这些外置依赖不属于 Git 源码。本文命令针对该共享 NFS 环境；其他机器必须先准备匹配的驱动、TorchMUSA ABI、模型和外置资产，不能仅拉取分支就宣称环境完备。

实际加载库清单见 `$DSPARK_TOOLS/dspark_sp_prefix_08_fresh/loaded_shared_libraries.txt`，SHA256 见 `$DSPARK_TOOLS/dependency_sha256.txt`。其中还包括 DSA Model1、local-prefill、MLA、TF32 HC prenorm 模块；不要把启动器的最小 preflight 列表误当作完整环境清单。


## 1. 一次复现到底运行哪些请求

默认流程是同一个服务、串行 batch=1：

| 阶段 | 次数 | 输入/输出 | 检查 | 是否计入性能热窗 |
| --- | ---: | --- | --- | --- |
| warmup_1 / warmup_2 | 2 | 64/64 | 全文严格 1..32 + 服务端 usage | 否 |
| formal_variant_0 | 1 | 2500/1500 | 全文严格 1..750；冷态通常 cached=0 | 否 |
| formal_variant_1 | 1 | 2500/1500 | 全文严格 1..750；建立/确认 cached=1536 | 否 |
| formal_variant_2–7 | 6 | 2500/1500 | 全文严格 1..750、usage=2500/1500、cached=1536 | 全部六个 |

**一次完整复现共 10 个请求：2 个短预热 + 8 个长请求。** 正式长请求同时验证精度和性能，不需要再额外运行一套精度脚本。独立复验要停止服务、确认八卡释放、换新日志目录、重新启动，再原样执行这 10 个请求。本轮结果来自两个这样的独立服务，不是同一进程中只挑两次快请求。

如要追加热态窗口，先声明连续的 variant 8–13，再完整运行六次并独立统计；不能用它们替换 2–7 中的慢样本。精度、usage 或传输失败时脚本停止并保留错误，窗口不得判为通过。

## 2. 提取本文的五个完整脚本

前提：已经有可执行的 DSpark MUSA 二进制及匹配的本地依赖。下面的脚本只启动/测试，不自动编译、不下载库，不会自动清理原实验。

五个脚本的完整源码在文末；可以手动按文件名保存，也可以从这份 Markdown 一次性提取：

~~~bash
export DSPARK_ROOT=/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark
export DSPARK_BIN="$DSPARK_ROOT/build/dspark-musa-local/xllm/xllm"
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export DSPARK_SCRIPTS=$(mktemp -d /tmp/dspark-repro.XXXXXX)

python3 - "$DSPARK_ROOT/docs/DEEPSEEK_V4_MUSA_DSPARK_REPRO.md" "$DSPARK_SCRIPTS" <<'PY'
import re
import sys
from pathlib import Path

document, destination = Path(sys.argv[1]), Path(sys.argv[2])
pattern = r"<!-- script: ([a-z0-9_]+\.(?:sh|py)) -->\n```(?:bash|python)\n(.*?)\n```"
matches = re.findall(pattern, document.read_text(encoding="utf-8"), re.S)
expected = {"start_dspark_tp8.sh", "benchmark_dspark.py", "check_result.py",
            "run_suite.sh", "summarize_window.py"}
if len(matches) != 5 or {name for name, _ in matches} != expected:
    raise RuntimeError("Expected exactly five named script blocks")
destination.mkdir(parents=True, exist_ok=True)
if any((destination / name).exists() for name, _ in matches):
    raise FileExistsError("Refusing to overwrite existing scripts")
for name, source in matches:
    with (destination / name).open("x", encoding="utf-8") as output:
        output.write(source + "\n")
    (destination / name).chmod(0o700)
print(destination)
PY

bash -n "$DSPARK_SCRIPTS/start_dspark_tp8.sh" "$DSPARK_SCRIPTS/run_suite.sh"
python3 -m py_compile "$DSPARK_SCRIPTS/"*.py
~~~

这些脚本把历史链路的默认值展开成 DSpark N=5、Graph=true、Prefix Cache=true、overlap=false；提示词构造内嵌了原算法，不再 import worker2 的文件。请求内容与计时逻辑保留原协议，文档版额外保存完整 request payload、输出全文，并对失败/缺失 usage 做严格检查。本文指标仍来自前述历史实测；更换机器、模型、二进制或依赖后必须重新验收。

## 3. 启动八卡服务（终端 A）

**先检查宿主机 GPU 占用，手动结束自己已核实的占卡进程。** 容器 PID 和宿主机 PID 不一定相同；不要照搬历史 PID，不用 pkill 清理所有 Python/xllm，也不要 reset GPU。下列启动器只在退出时清理它自己启动的八个 worker，不替操作者判断其他任务的归属。

~~~bash
ssh -p22 mccxadmin@10.20.34.28 mthreads-gmi
# 如已有本任务的占卡程序，核实 PID/PGID 后手动停止，再检查宿主机显存。

mkdir -p "$DSPARK_ROOT/logs"
export DSPARK_RUN=$(mktemp -d "$DSPARK_ROOT/logs/dspark-repro.XXXXXX")
ssh -p22 mccxadmin@10.20.34.28 mthreads-gmi > "$DSPARK_RUN/host_before.txt"
echo "DSPARK_RUN=$DSPARK_RUN"
echo "DSPARK_SCRIPTS=$DSPARK_SCRIPTS"

# 前台运行，保持此终端打开；Ctrl-C 会清理启动器自己的 worker。
bash "$DSPARK_SCRIPTS/start_dspark_tp8.sh"
~~~

默认 HTTP 为 18861，master 为 127.0.0.1:20861。不要让两个服务占用同一端口。需要换端口时在两个终端设置相同的 XLLM2_EXTERNAL_TP8_HTTP_PORT，并在终端 A 设置 XLLM2_EXTERNAL_TP8_MASTER_ADDR。

这份启动器锁定本次已验证的算法、batch、Graph/Prefix、overlap 等配置；旧终端中的 XLLM2_EXTERNAL_TP8_* 若与它冲突，会拒绝启动，避免误用旧 MTP/batch8 参数。路径和端口可以按机器调整。若确实要改变性能配置，请另建实验、修改脚本并按新 manifest 重验，不沿用历史指标。

日志目录包含 launch_manifest.txt 和 rank0.log–rank7.log。manifest 记录二进制/模型配置/脚本 SHA256、Git HEAD/状态、拓扑、Graph/Prefix/overlap、预加载库和运行库路径。构建 SHA 不同并不自动意味着错误，但必须按新的二进制重新验证，不能替换成 donor 二进制后仍称为当前分支结果。

## 4. 完整精度与性能复现（终端 B）

在终端 B 使用终端 A 输出的**同一个实际目录**。不要再次 mktemp，也不要启动第二个服务。模型路径和非默认 HTTP 端口也必须与终端 A 相同。

~~~bash
set -euo pipefail
# 从终端 A 复制实际值，例如 /tmp/dspark-repro.ABC123 和 .../logs/dspark-repro.DEF456
: "${DSPARK_SCRIPTS:?请设置为终端 A 提取脚本的实际目录}"
: "${DSPARK_RUN:?请设置为终端 A 打印的实际实验目录}"
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt

# 模型加载期间连接失败是正常的。健康检查最多等 15 分钟。
dspark_deadline=$((SECONDS + 900))
until curl -fsS --max-time 2 "http://127.0.0.1:${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}/health"; do
  if ((SECONDS >= dspark_deadline)); then
    echo "Service not ready; inspect rank logs." >&2
    exit 1
  fi
  sleep 2
done

# 内部先执行两次短预热，再执行 variant 0–7，最后统计 2–7。
bash "$DSPARK_SCRIPTS/run_suite.sh" "$DSPARK_RUN" 0 7 \
  | tee "$DSPARK_RUN/window_0_7.log"

# 查看整窗结果；也可以随时从已经落盘的六个 JSON 重新统计。
python3 "$DSPARK_SCRIPTS/summarize_window.py" "$DSPARK_RUN" 2 7
~~~

测试期间在第三个终端保留宿主机快照：

~~~bash
ssh -p22 mccxadmin@10.20.34.28 mthreads-gmi > "$DSPARK_RUN/host_window_mid.txt"
# 窗口完成后：
ssh -p22 mccxadmin@10.20.34.28 mthreads-gmi > "$DSPARK_RUN/host_window_after.txt"
~~~

不应出现其他任务的 GPU 进程、缺失 rank 或精度失败。脚本无法仅凭 JSON 判断是否有人同时占卡，必须结合这些宿主机快照人工确认。

## 5. 只验证一条请求的精度

这只是单请求功能检查，**不代表热态性能验收**。建议使用另一轮新服务/日志目录，避免改变上面固定的预热和缓存过程。

~~~bash
set -euo pipefail
python3 "$DSPARK_SCRIPTS/benchmark_dspark.py" \
  --model-path "$MODEL_PATH" \
  --url "http://127.0.0.1:${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}/v1/chat/completions" \
  --prompt-tokens 2500 --output-tokens 1500 --count-to 750 \
  --output "$DSPARK_RUN/accuracy_2500_1500.json"
python3 "$DSPARK_SCRIPTS/check_result.py" "$DSPARK_RUN/accuracy_2500_1500.json"
~~~

检查器直接重算输出全文是否为 1..750，不仅信任 accuracy.passed 字段；同时要求服务端 usage、正常 length 结束以及有效计时。未返回 token、服务崩溃、超时、usage 缺失和 HTTP chunk fallback 都不能算 PASS。

## 6. 统计口径、退出和独立复验

- TTFT：客户端发送请求前，到首个非空 content 到达的时间。
- TPOT：(最后一个非空 content 的时刻 - 首个非空 content 的时刻) / (服务端 completion_tokens - 1)。DSpark 一个 chunk 可含多个 token，不能用 chunk 数作分母。
- E2E：发送请求前到流读取结束。单请求总吞吐如需计算，为 (usage.prompt_tokens + usage.completion_tokens) / E2E秒；不是脚本中 1000/TPOT 的 decode 速率，也不是多并发压测吞吐。
- 正式热窗必须有连续六份文件，全部全文严格正确、usage=2500/1500、cached=1536；平均 TTFT<250 ms，TPOT<14 ms 且 <=13.758500 ms。
- 缺文件/无有效计时不计算“剩下样本”的均值；不删除慢样本，不把冷态/短请求拼进热窗。
- 若事先决定追加复测，完整执行下列另一窗口并单独报告，不能替换首窗：

~~~bash
bash "$DSPARK_SCRIPTS/run_suite.sh" "$DSPARK_RUN" 8 13 \
  | tee "$DSPARK_RUN/window_8_13.log"
~~~

在终端 A Ctrl-C 停止前台服务；等启动器退出后，用宿主机 mthreads-gmi 确认它的八个 worker 已释放并保存 host_after.txt。按使用约定恢复自己管理的占卡程序。不要运行 GPU reset，不要杀掉其他容器任务。

**独立复验**：确认旧服务退出 → 新建 DSPARK_RUN → 重新启动 → 同样的两次短预热和 variant 0–7 → 完整统计 2–7。重复完整流程一次，才对应上面两个独立服务的结果。

## 7. 常见失败如何处理

| 现象 | 处理/判定 |
| --- | --- |
| 找不到二进制、AOT 或预加载库 | 检查脚本开头的路径与本机 ABI；不自动下载，也不拿另一条链路的二进制替代 |
| 端口已占用、health 未就绪 | 查看 rank 日志和端口所属进程；只处理本任务进程 |
| 精度或 usage 检查失败 | 保留 JSON、client.log、exit_code 和 rank 日志；停止本轮，不统计成有效性能 |
| cached_tokens 不是 1536 | 检查服务 Prefix Cache 开关、0.65 提示词和 block size 128；该热窗不通过 |
| 首次请求 TTFT 很高 | 保留冷态/线程初始化数据，只按预先声明的热窗判定 |
| 文件已存在 | 脚本拒绝覆盖；新建实验目录，不删除原记录 |
| 只输出一个 1、TPOT=0 | 通常是中断/无 usage 的无效结果，不是极速或精度通过 |


## 验证边界

- 已验证范围仅为以上 checkpoint、TP8/EP1、batch 1、静态 DSpark N=5、2.5K/1.5K、0.65 前缀。
- 不等于 batch8/16、PD、CP、多机、adaptive 或任意前缀命中比例已验证。
- 其他模型和非 MUSA 设备未做运行回归，不以本次结果代替它们的验收。
- 0731 target-only **eager** 曾出现精度异常；target-only Graph 已通过 `750/750`。本成果不宣称 eager 问题已修复。
- draft block 仍为 eager；不宣称全 draft/context/Markov 路径已捕获为 Graph。
- 精度验收是严格计数协议，不等于 GSM8K 或任意任务全面精度对齐；沿用原性能分支的 compact Graph compressor 行为，其他长上下文任务需另行验证。
- 构建、每次失败原因、旧二进制、完整样本与最终结果均保留在验证记录及 NFS 实验目录。未推送远端、未 reset GPU。

请在独立 DSpark 目录 review/commit。原 `xllm-dsv4-musa-perf` 目录另有与工作文件不一致的暂存改动，本次未清理；在原分支提交前先检查 `git diff --cached`。

## 后续工作与已达标项的区别

上述两轮 TTFT/TPOT 和严格计数门槛已经完成，不需要把历史记录中的“待验证”阶段误当作当前状态。继续推进时应单独定义新验收项：

1. **更广泛的精度验证。** 当前 `freeze_compact_graph_compressor` 路径保留 prefill 建立的压缩前缀，SWA 继续更新。需验证/完善 C4/C128 状态演进、拒绝草稿 token 后的状态恢复，以及短 suffix 的高前缀命中场景；计数 PASS 不能证明这些情形正确。
2. **target-only eager 精度修复。** 这是已记录的独立失败路径，不能写成已修复。
3. **draft 全链路 Graph。** draft block、上下文注入和 Markov 采样尚非全链路 Graph。仅将更多操作捕获成 Graph，不能替代前两项精度工作。

这些是后续工作清单，不是本轮新增实验结果；已有两个完整热窗和对应二进制保持不变。


## 附录：五个可直接保存执行的完整脚本

以下代码块是第 2 节提取命令的唯一来源。除模型、二进制、Python transformers 和已列明的动态库/AOT 外，没有对旧 NFS 测试脚本或 worker2 Python 文件的运行依赖。

文档版已完成离线校验：五个代码块可完整提取并通过 Bash/Python 语法检查；普通 64/2500-token 提示词及 prefix variant 0–13 与历史实现逐字节一致；启动 argv/env 函数和基线 preflight 配置一致；11 项检查覆盖计时公式、请求 payload、历史窗口回放、全文精度、缺失 usage/文件、错误缓存、慢样本及无效计时。两个历史热窗回放均得到与上表完全相同的均值。这里的离线检查没有启动模型 worker、没有重跑 GPU 性能，不把它算作新的实测。

### start_dspark_tp8.sh

<!-- script: start_dspark_tp8.sh -->
```bash
#!/usr/bin/env bash

set -euo pipefail

# DSpark TP8 reproduction. Check GPU ownership on the host before launch.
# Only this script's own worker PIDs are cleaned up; never reset devices.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ds_root=${DSPARK_ROOT:?Set DSPARK_ROOT to the DSpark source checkout}
run_dir=${DSPARK_RUN:?Set DSPARK_RUN to a fresh experiment directory}
master_addr=${XLLM2_EXTERNAL_TP8_MASTER_ADDR:-127.0.0.1:20861}
http_port=${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}
xllm_bin=${DSPARK_BIN:-$ds_root/build/dspark-musa-local/xllm/xllm}
model_path=${MODEL_PATH:-/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt}
prefix_cache=${XLLM2_EXTERNAL_TP8_PREFIX_CACHE:-true}
in_batch_prefix_cache=${XLLM2_EXTERNAL_TP8_IN_BATCH_PREFIX_CACHE:-false}
schedule_overlap=${XLLM2_EXTERNAL_TP8_SCHEDULE_OVERLAP:-false}
enable_graph=${XLLM2_EXTERNAL_TP8_GRAPH:-true}
enable_shm=${XLLM2_EXTERNAL_TP8_SHM:-false}
enable_chunked_prefill=${XLLM2_EXTERNAL_TP8_CHUNKED_PREFILL:-true}
enable_online_profile=${XLLM2_EXTERNAL_TP8_ONLINE_PROFILE:-false}
profile_backend=${XLLM2_EXTERNAL_TP8_PROFILE_BACKEND:-torch}
profile_dir=${XLLM2_EXTERNAL_TP8_PROFILE_DIR:-${run_dir}/profile}
max_tokens_per_batch=${XLLM2_EXTERNAL_TP8_MAX_TOKENS_PER_BATCH:-10240}
max_seqs_per_batch=${XLLM2_EXTERNAL_TP8_MAX_SEQS_PER_BATCH:-1}
max_concurrent_requests=${XLLM2_EXTERNAL_TP8_MAX_CONCURRENT_REQUESTS:-${max_seqs_per_batch}}
max_memory_utilization=${XLLM2_EXTERNAL_TP8_MAX_MEMORY_UTILIZATION:-0.8}
block_size=${XLLM2_EXTERNAL_TP8_BLOCK_SIZE:-128}
num_speculative_tokens=${XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS:-5}
speculative_algorithm=${XLLM2_EXTERNAL_TP8_SPECULATIVE_ALGORITHM:-DSpark}
enable_mtp_draft_body_tp1=${XLLM2_EXTERNAL_TP8_MTP_DRAFT_BODY_TP1:-false}
draft_model=${XLLM2_EXTERNAL_TP8_DRAFT_MODEL:-${model_path}}
draft_model_args=(--draft_model="${draft_model}")
if [[ "${XLLM2_EXTERNAL_TP8_DISABLE_DRAFT:-false}" == true ]]; then
  draft_model_args=()
fi
ops_path=${XLLM2_EXTERNAL_TP8_OPS_PATH:-/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops}
custom_ar_so=${XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO:-/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so}
shutdown_timeout_seconds=${XLLM2_EXTERNAL_TP8_SHUTDOWN_TIMEOUT_SECONDS:-30}
num_response_handling_threads=${XLLM2_EXTERNAL_TP8_NUM_RESPONSE_HANDLING_THREADS:-4}
server_num_threads=${XLLM2_EXTERNAL_TP8_SERVER_NUM_THREADS:-}
bthread_concurrency=${XLLM2_EXTERNAL_TP8_BTHREAD_CONCURRENCY:-}
cpu_affinity=${XLLM2_EXTERNAL_TP8_CPU_AFFINITY:-none}
preflight_only=${XLLM2_EXTERNAL_TP8_PREFLIGHT_ONLY:-false}
benchmark_script="$script_dir/benchmark_dspark.py"
# Reproduce only the declared baseline; do not inherit a previous experiment.
if [[ "$speculative_algorithm" != DSpark || "$num_speculative_tokens" != 5 ||
      "$enable_graph" != true || "$prefix_cache" != true ||
      "$schedule_overlap" != false || "$in_batch_prefix_cache" != false ||
      "$enable_chunked_prefill" != true || "$enable_shm" != false ||
      "$max_tokens_per_batch" != 10240 || "$max_seqs_per_batch" != 1 ||
      "$max_concurrent_requests" != 1 || "$block_size" != 128 ||
      "$max_memory_utilization" != 0.8 || "$num_response_handling_threads" != 4 ||
      "$enable_mtp_draft_body_tp1" != false || "$draft_model" != "$model_path" ||
      "${XLLM2_EXTERNAL_TP8_DISABLE_DRAFT:-false}" != false ||
      "$enable_online_profile" != false || "$cpu_affinity" != none ||
      -n "$server_num_threads" || -n "$bthread_concurrency" ]]; then
  echo "Conflicting settings: use the documented TP8/EP1, DSpark N=5, B1, Graph+Prefix, overlap=false baseline." >&2
  exit 2
fi
if [[ -e "$run_dir/launch_manifest.txt" ]]; then
  echo "Refusing to overwrite an existing run: $run_dir" >&2
  exit 2
fi

server_thread_args=()
if [[ -n "${server_num_threads}" ]]; then
  server_thread_args+=(--num_threads="${server_num_threads}")
fi
if [[ -n "${bthread_concurrency}" ]]; then
  server_thread_args+=(--bthread_concurrency="${bthread_concurrency}")
fi

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
}

require_file "${xllm_bin}" "target xllm executable"
require_file "${model_path}/config.json" "model config"
require_file "${custom_ar_so}" "custom all-reduce shared object"
require_file "${benchmark_script}" "warmup benchmark helper"

required_ops=(
  gemm_ops/gemm_ops.so
  mhc_pre_big_fuse_ops/mhc_pre_big_fuse_ops.so
  mhc_prenorm_ops/mhc_prenorm_ops.so
  mate_jit_mega_moe_fp8_fp8_stage1_native_l1_r8_e256_mt32_topk6_h4096_ih2048_mps60_fm1/mate_jit_mega_moe_fp8_fp8_stage1_native_l1_r8_e256_mt32_topk6_h4096_ih2048_mps60_fm1.so
  mate_jit_mega_moe_fp8_fp8_stage2_r8_e256_mt32_topk6_h4096_ih2048_mps60_fm1/mate_jit_mega_moe_fp8_fp8_stage2_r8_e256_mt32_topk6_h4096_ih2048_mps60_fm1.so
  mate_jit_mega_moe_fp8_fp8_stage1_native_l1_r8_e256_mt320_topk6_h4096_ih2048_mps60_fm1/mate_jit_mega_moe_fp8_fp8_stage1_native_l1_r8_e256_mt320_topk6_h4096_ih2048_mps60_fm1.so
  mate_jit_mega_moe_fp8_fp8_stage2_r8_e256_mt320_topk6_h4096_ih2048_mps60_fm1/mate_jit_mega_moe_fp8_fp8_stage2_r8_e256_mt320_topk6_h4096_ih2048_mps60_fm1.so
)
for rel in "${required_ops[@]}"; do
  require_file "${ops_path}/${rel}" "DeepSeek-V4 hot-path AOT op"
done

preload_candidates=(
  "${XLLM2_EXTERNAL_TP8_PRELOAD_PREFIX:-}"
  "${custom_ar_so}"
  /data/nfs_shared/xllm2_artifacts/libasio_gcc.so
  /usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so
  "${XLLM2_EXTERNAL_TP8_BACKEND_INIT_SO:-}"
)
preload_items=()
for candidate in "${preload_candidates[@]}"; do
  [[ -n "${candidate}" && -f "${candidate}" ]] || continue
  duplicate=false
  for selected in "${preload_items[@]}"; do
    if [[ "${selected}" == "${candidate}" ]]; then
      duplicate=true
      break
    fi
  done
  [[ "${duplicate}" == true ]] || preload_items+=("${candidate}")
done
if ((${#preload_items[@]} == 0)); then
  echo "ERROR: no runtime preload libraries were found." >&2
  exit 1
fi
runtime_preload=$(IFS=:; echo "${preload_items[*]}")
runtime_library_path="/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/openmpi/lib:/usr/local/lib/python3.10/dist-packages/tilelang/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${run_dir}"
{
  date --iso-8601=seconds
  echo "xllm_bin=${xllm_bin}"
  git -C "$ds_root" rev-parse HEAD
  git -C "$ds_root" status --short
  sha256sum "$model_path/config.json"
  sha256sum "$script_dir"/*.py "$script_dir"/*.sh
  sha256sum "${xllm_bin}"
  echo "model_path=${model_path}"
  echo "master_addr=${master_addr}"
  echo "http_port=${http_port}"
  echo "topology=TP8/EP1/DP1/CP1"
  echo "graph=${enable_graph}"
  echo "schedule_overlap=${schedule_overlap}"
  echo "prefix_cache=${prefix_cache}"
  echo "in_batch_prefix_cache=${in_batch_prefix_cache}"
  echo "num_speculative_tokens=${num_speculative_tokens}"
  echo "speculative_algorithm=${speculative_algorithm}"
  echo "max_tokens_per_batch=${max_tokens_per_batch}"
  echo "max_seqs_per_batch=${max_seqs_per_batch}"
  echo "max_memory_utilization=${max_memory_utilization}"
  echo "block_size=${block_size}"
  echo "num_response_handling_threads=${num_response_handling_threads}"
  echo "server_num_threads=${server_num_threads:-default}"
  echo "bthread_concurrency=${bthread_concurrency:-default}"
  echo "cpu_affinity=${cpu_affinity}"
  echo "ops_path=${ops_path}"
  echo "custom_ar_so=${custom_ar_so}"
  sha256sum "${custom_ar_so}"
  echo "ld_preload=${runtime_preload}"
  echo "ld_library_path=${runtime_library_path}"
} >"${run_dir}/launch_manifest.txt"
if [[ "${preflight_only}" == true ]]; then
  echo "Preflight passed; manifest: ${run_dir}/launch_manifest.txt"
  exit 0
fi

worker_pids=()
wait_for_workers() {
  local signal=$1
  local timeout_seconds=$2
  local deadline=$((SECONDS + timeout_seconds))

  kill "-${signal}" "${worker_pids[@]}" 2>/dev/null || true
  while ((SECONDS < deadline)); do
    local alive=false
    local pid
    for pid in "${worker_pids[@]}"; do
      local state=
      if [[ -r "/proc/${pid}/stat" ]]; then
        state=$(awk '{print $3}' "/proc/${pid}/stat")
      fi
      if kill -0 "${pid}" 2>/dev/null && [[ "${state}" != Z ]]; then
        alive=true
        break
      fi
    done
    if [[ "${alive}" == false ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup() {
  trap - INT TERM EXIT
  if ((${#worker_pids[@]} > 0)); then
    if ! wait_for_workers INT "${shutdown_timeout_seconds}"; then
      if ! wait_for_workers TERM "${shutdown_timeout_seconds}"; then
        kill -KILL "${worker_pids[@]}" 2>/dev/null || true
      fi
    fi
    wait "${worker_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

launch_rank() {
  local rank=$1
  local -a affinity_command=()
  if [[ "${cpu_affinity}" == disjoint-smt ]]; then
    local physical_start
    local sibling_start
    if ((rank < 4)); then
      physical_start=$((rank * 8))
      sibling_start=$((64 + rank * 8))
    else
      physical_start=$((32 + (rank - 4) * 8))
      sibling_start=$((96 + (rank - 4) * 8))
    fi
    affinity_command=(taskset -c
      "${physical_start}-$((physical_start + 7)),${sibling_start}-$((sibling_start + 7))")
  elif [[ "${cpu_affinity}" != none ]]; then
    echo "ERROR: unsupported CPU affinity mode: ${cpu_affinity}" >&2
    return 1
  fi
  env \
    LD_PRELOAD="${runtime_preload}" \
    LD_LIBRARY_PATH="${runtime_library_path}" \
    TORCH_MCCL_ASYNC_ERROR_HANDLING=0 \
    TORCH_MCCL_ENABLE_MONITORING=0 \
    MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    FLASHINFER_OPS_PATH="${ops_path}" \
    XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS=120 \
    "${affinity_command[@]}" "${xllm_bin}" \
    "${server_thread_args[@]}" \
    --model="${model_path}" \
    --model_id=DeepSeek-V4-Flash-FP8 \
    --backend=llm \
    --host=127.0.0.1 \
    --port="${http_port}" \
    --master_node_addr="${master_addr}" \
    --nnodes=8 \
    --node_rank="${rank}" \
    --dp_size=1 \
    --ep_size=1 \
    --cp_size=1 \
    --communication_backend=mccl \
    --block_size="${block_size}" \
    --max_memory_utilization="${max_memory_utilization}" \
    --enable_prefix_cache="${prefix_cache}" \
    --enable_in_batch_prefix_cache="${in_batch_prefix_cache}" \
    --max_tokens_per_batch="${max_tokens_per_batch}" \
    --max_seqs_per_batch="${max_seqs_per_batch}" \
    --max_concurrent_requests="${max_concurrent_requests}" \
    --num_response_handling_threads="${num_response_handling_threads}" \
    --enable_chunked_prefill="${enable_chunked_prefill}" \
    --enable_shm="${enable_shm}" \
    --enable_schedule_overlap="${schedule_overlap}" \
    --num_speculative_tokens="${num_speculative_tokens}" \
    --speculative_algorithm="${speculative_algorithm}" \
    --enable_mtp_draft_body_tp1="${enable_mtp_draft_body_tp1}" \
    "${draft_model_args[@]}" \
    --enable_graph="${enable_graph}" \
    --enable_graph_vmm_pool=true \
    --enable_graph_mode_decode_no_padding=true \
    --max_tokens_for_graph_mode=2048 \
    --enable_online_profile="${enable_online_profile}" \
    --profile_backend="${profile_backend}" \
    --profile_dir="${profile_dir}" \
    >"${run_dir}/rank${rank}.log" 2>&1 &
  worker_pids+=("$!")
}

for rank in 1 2 3 4 5 6 7; do
  launch_rank "${rank}"
done
launch_rank 0

wait -n "${worker_pids[@]}"
exit_code=$?
cleanup
exit "${exit_code}"
```

### benchmark_dspark.py

<!-- script: benchmark_dspark.py -->
```python
#!/usr/bin/env python3
"""Measure exact-token DeepSeek-V4 chat latency and counting accuracy.

This self-contained reproduction uses the retained timing protocol. It
builds the current DeepSeek-V4 native chat prompt locally, verifies its exact
token count, then records streaming TTFT/TPOT and the deterministic counting
result returned by the OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


MODEL_PATH = Path("/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt")
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_END = "</think>"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--url", required=True)
  parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
  parser.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
  parser.add_argument("--prompt-tokens", type=int, required=True)
  parser.add_argument("--output-tokens", type=int, required=True)
  parser.add_argument("--count-to", type=int, required=True)
  parser.add_argument("--prefix-cache-rate", type=float, default=0.0)
  parser.add_argument("--prompt-variant", type=int, default=0)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


CONTEXT = (
    "Transformer inference has a prefill phase and a decode phase. "
    "The prefill phase processes all prompt tokens, while decode appends one "
    "token at a time and reuses paged key-value cache entries. Sparse attention "
    "uses an indexer to select compressed history and combines it with a recent "
    "sliding window. Expert-parallel mixture-of-experts layers route each token "
    "to selected experts and use collective communication to combine results. "
)


def _chat_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    bos = tokenizer.bos_token or "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    rendered = (
        bos
        + "<\uff5cUser\uff5c>"
        + prompt
        + "<\uff5cAssistant\uff5c></think>"
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _counting_instruction(count_to: int) -> str:
    return (
        f"\n\nIgnore the background text. Output only the consecutive integers "
        f"from 1 through {count_to}, separated by one ASCII space. Do not add "
        "explanations or punctuation."
    )


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length


def _repeat_token_ids(source_ids: list[int], token_count: int) -> list[int]:
    if not source_ids or token_count <= 0:
        return []
    repeat_count = (token_count + len(source_ids) - 1) // len(source_ids)
    return (source_ids * repeat_count)[:token_count]


def _fit_counting_variant(
    tokenizer: Any,
    input_tokens: int,
    count_to: int,
    common_body_tokens: int,
    variant_index: int,
) -> tuple[str, int]:
    context_ids = tokenizer.encode(CONTEXT, add_special_tokens=False)
    common_ids = _repeat_token_ids(context_ids, common_body_tokens)
    common_text = tokenizer.decode(common_ids, skip_special_tokens=True)
    variant_sentence = (
        f"\nBenchmark background branch {variant_index} begins here. "
        f"Its deterministic marker is branch-{variant_index}-only. "
        "This sentence is background and must not affect the requested answer. "
    )
    variant_ids = tokenizer.encode(
        variant_sentence * (input_tokens + 32), add_special_tokens=False
    )
    instruction = _counting_instruction(count_to)

    def render(suffix_tokens: int) -> tuple[str, int]:
        suffix_text = tokenizer.decode(
            variant_ids[:suffix_tokens], skip_special_tokens=True
        )
        prompt = common_text + suffix_text + instruction
        return prompt, len(_chat_token_ids(tokenizer, prompt))

    low = 0
    high = min(len(variant_ids), input_tokens * 2 + 256)
    while low < high:
        middle = (low + high + 1) // 2
        _, token_count = render(middle)
        if token_count <= input_tokens:
            low = middle
        else:
            high = middle - 1

    candidates: list[tuple[int, str, int]] = []
    for suffix_tokens in range(max(0, low - 12), min(len(variant_ids), low + 12) + 1):
        prompt, token_count = render(suffix_tokens)
        candidates.append((abs(token_count - input_tokens), prompt, token_count))
    _, prompt, token_count = min(candidates, key=lambda item: item[0])
    if abs(token_count - input_tokens) > 2:
        raise RuntimeError(
            "could not build prefix-cache prompt variant within two tokens: "
            f"target={input_tokens}, actual={token_count}, "
            f"variant={variant_index}"
        )
    return prompt, token_count


def _build_prompt_variants(
    tokenizer: Any,
    input_tokens: int,
    count_to: int,
    prefix_cache_rate: float,
    num_variants: int,
    start_index: int = 0,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """Build exact-length prompts with a measured shared chat-token prefix."""
    if not 0.0 < prefix_cache_rate < 1.0:
        raise ValueError("prefix_cache_rate must be between zero and one")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    target_common_tokens = int(input_tokens * prefix_cache_rate)
    common_body_tokens = max(1, target_common_tokens - 16)
    calibration_prefix = 0
    for _ in range(8):
        calibration_prompts = [
            _fit_counting_variant(
                tokenizer,
                input_tokens,
                count_to,
                common_body_tokens,
                variant_index,
            )[0]
            for variant_index in (0, 1)
        ]
        calibration_ids = [
            _chat_token_ids(tokenizer, prompt) for prompt in calibration_prompts
        ]
        calibration_prefix = _common_prefix_length(
            calibration_ids[0], calibration_ids[1]
        )
        prefix_error = target_common_tokens - calibration_prefix
        if abs(prefix_error) <= 2:
            break
        common_body_tokens = max(1, common_body_tokens + prefix_error)

    variants = [
        _fit_counting_variant(
            tokenizer,
            input_tokens,
            count_to,
            common_body_tokens,
            variant_index,
        )
        for variant_index in range(start_index, start_index + num_variants)
    ]
    variant_ids = [_chat_token_ids(tokenizer, prompt) for prompt, _ in variants]
    measured_prefix = calibration_prefix
    if len(variant_ids) > 1:
        measured_prefix = min(
            _common_prefix_length(variant_ids[0], token_ids)
            for token_ids in variant_ids[1:]
        )
    metadata = {
        "configured_prefix_cache_rate": prefix_cache_rate,
        "target_common_chat_tokens": target_common_tokens,
        "calibrated_common_chat_tokens": calibration_prefix,
        "measured_variant_common_chat_tokens": measured_prefix,
        "measured_variant_common_rate": round(measured_prefix / input_tokens, 6),
        "common_body_tokens": common_body_tokens,
        "start_index": start_index,
        "num_variants": num_variants,
    }
    return variants, metadata




def build_exact_content(tokenizer: Any, target: int, count_to: int) -> str:
  instruction = (
      f"Count from 1 to {count_to}, separated by spaces. "
      "Output only the numbers."
  )

  def token_count(filler_count: int) -> int:
    prompt = BOS + USER + ("x " * filler_count) + instruction
    prompt += ASSISTANT + THINK_END
    return len(tokenizer.encode(prompt, add_special_tokens=False))

  base_count = token_count(0)
  if base_count > target:
    raise ValueError(f"base prompt has {base_count} tokens, target is {target}")

  # For this tokenizer, each additional `x ` contributes one token in the
  # filler region.  Search around the direct estimate as a guard against a
  # boundary merge after tokenizer updates.
  estimate = target - base_count
  for filler_count in range(max(0, estimate - 16), estimate + 17):
    if token_count(filler_count) == target:
      return ("x " * filler_count) + instruction
  raise RuntimeError(f"could not construct an exact {target}-token prompt")


def run_stream(args: argparse.Namespace, content: str) -> dict[str, Any]:
  payload = {
      "model": args.model,
      "messages": [{"role": "user", "content": content}],
      "chat_template_kwargs": {"enable_thinking": False},
      "temperature": 0,
      "max_tokens": args.output_tokens,
      "ignore_eos": True,
      "stream": True,
      "stream_options": {"include_usage": True},
  }
  request = urllib.request.Request(
      args.url,
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
  )

  start = time.perf_counter()
  first_content_at: float | None = None
  last_content_at: float | None = None
  content_chunks: list[str] = []
  usage: dict[str, Any] = {}
  finish_reason: str | None = None

  with urllib.request.urlopen(request, timeout=900) as response:
    for raw_line in response:
      line = raw_line.decode("utf-8").strip()
      if not line.startswith("data:"):
        continue
      data = line[5:].strip()
      if data == "[DONE]":
        break
      event = json.loads(data)
      if event.get("usage"):
        usage = event["usage"]
      for choice in event.get("choices", []):
        if choice.get("finish_reason") is not None:
          finish_reason = choice["finish_reason"]
        chunk = choice.get("delta", {}).get("content")
        if not chunk:
          continue
        now = time.perf_counter()
        if first_content_at is None:
          first_content_at = now
        last_content_at = now
        content_chunks.append(chunk)

  end = time.perf_counter()
  if first_content_at is None or last_content_at is None:
    raise RuntimeError("stream completed without a content token")

  text = "".join(content_chunks)
  numbers = [int(value) for value in re.findall(r"\d+", text)]
  expected = list(range(1, args.count_to + 1))
  mismatch_index: int | None = None
  for index, (actual, wanted) in enumerate(zip(numbers, expected)):
    if actual != wanted:
      mismatch_index = index
      break
  if mismatch_index is None and len(numbers) != len(expected):
    mismatch_index = min(len(numbers), len(expected))

  completion_tokens = usage.get("completion_tokens")
  if not isinstance(completion_tokens, int) or completion_tokens <= 0:
    completion_tokens = len(content_chunks)
    completion_tokens_source = "stream_content_chunks"
  else:
    completion_tokens_source = "usage"

  ttft_ms = (first_content_at - start) * 1000
  e2e_ms = (end - start) * 1000
  tpot_ms = (
      (last_content_at - first_content_at) * 1000 / (completion_tokens - 1)
      if completion_tokens > 1
      else 0.0
  )
  return {
      "model": args.model,
      "model_path": str(args.model_path),
      "request_payload": payload,
      "url": args.url,
      "prompt_tokens_built": args.prompt_tokens,
      "output_tokens_requested": args.output_tokens,
      "count_to": args.count_to,
      "metrics": {
          "usage": usage,
          "completion_tokens": completion_tokens,
          "completion_tokens_source": completion_tokens_source,
          "stream_content_chunks": len(content_chunks),
          "finish_reason": finish_reason,
          "ttft_ms": round(ttft_ms, 3),
          "e2e_ms": round(e2e_ms, 3),
          "tpot_ms": round(tpot_ms, 3),
          "tgs_tokens_per_second": round(1000 / tpot_ms, 3)
          if tpot_ms > 0
          else None,
          "last_content_ms": round((last_content_at - start) * 1000, 3),
      },
      "accuracy": {
          "passed": numbers == expected,
          "strict_text_passed": text.split() == [str(value) for value in expected],
          "expected_count": len(expected),
          "actual_count": len(numbers),
          "first_numbers": numbers[:10],
          "last_numbers": numbers[-10:],
          "mismatch_index": mismatch_index,
      },
      "content": text,
      "content_prefix": text[:512],
      "content_suffix": text[-512:],
  }


def main() -> None:
  args = parse_args()
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
  if min(args.prompt_tokens, args.output_tokens, args.count_to) <= 0:
    raise ValueError("Token counts and count-to must be positive")
  tokenizer = AutoTokenizer.from_pretrained(
      args.model_path, trust_remote_code=True, local_files_only=True
  )
  prefix_metadata: dict[str, Any] | None = None
  if args.prefix_cache_rate > 0.0:
    prompts, prefix_metadata = _build_prompt_variants(
        tokenizer,
        args.prompt_tokens,
        args.count_to,
        args.prefix_cache_rate,
        num_variants=1,
        start_index=args.prompt_variant,
    )
    content, actual_prompt_tokens = prompts[0]
    if actual_prompt_tokens != args.prompt_tokens:
      raise RuntimeError(
          f"prefix prompt has {actual_prompt_tokens} tokens, "
          f"expected {args.prompt_tokens}"
      )
  else:
    content = build_exact_content(tokenizer, args.prompt_tokens, args.count_to)
  try:
    result = run_stream(args, content)
  except Exception as exc:
    result = {"error": f"{type(exc).__name__}: {exc}",
              "prompt_tokens_built": args.prompt_tokens,
              "output_tokens_requested": args.output_tokens,
              "count_to": args.count_to, "model_path": str(args.model_path)}
  if prefix_metadata is not None:
    result["prompt_variant"] = args.prompt_variant
    result["prefix_cache_prompt"] = prefix_metadata
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x", encoding="utf-8") as output:
    output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps(result, ensure_ascii=False, indent=2))
  if "error" in result:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
```

### check_result.py

<!-- script: check_result.py -->
```python
#!/usr/bin/env python3
"""Check actual full text, server token accounting, and valid timing."""
import json
import math
import sys
from pathlib import Path


def positive_number(value: object) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and value > 0)


def checks_for(data: dict, prompt: int | None = None,
               output: int | None = None, count: int | None = None) -> dict:
    prompt = data.get("prompt_tokens_built") if prompt is None else prompt
    output = data.get("output_tokens_requested") if output is None else output
    count = data.get("count_to") if count is None else count
    shape_ok = all(type(v) is int and v > 0 for v in (prompt, output, count))
    metrics = data.get("metrics", {})
    usage = metrics.get("usage") or {}
    text = data.get("content")
    strict = (shape_ok and isinstance(text, str)
              and text.split() == [str(i) for i in range(1, count + 1)])
    accuracy = data.get("accuracy", {})
    return {
        "no_stream_error": "error" not in data,
        "declared_shape": shape_ok
            and data.get("prompt_tokens_built") == prompt
            and data.get("output_tokens_requested") == output
            and data.get("count_to") == count,
        "strict_full_text": strict,
        "recorded_accuracy": accuracy.get("passed") is True
            and accuracy.get("strict_text_passed") is True,
        "exact_server_usage": shape_ok
            and usage.get("prompt_tokens") == prompt
            and usage.get("completion_tokens") == output,
        "usage_not_chunks": metrics.get("completion_tokens_source") == "usage",
        "normal_finish": metrics.get("finish_reason") == "length",
        "positive_finite_timing": all(positive_number(metrics.get(k))
                                     for k in ("ttft_ms", "tpot_ms", "e2e_ms")),
    }


def main() -> None:
    path = Path(sys.argv[1])
    try:
        data = json.loads(path.read_text())
        checks = checks_for(data)
        result = {"file": str(path), "checks": checks,
                  "gate": "PASS" if all(checks.values()) else "FAIL"}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result = {"file": str(path), "gate": "FAIL", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
```

### run_suite.sh

<!-- script: run_suite.sh -->
```bash
#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${1:?Supply the service's DSPARK_RUN directory}
first=${2:-0}
last=${3:-7}
[[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || exit 2
if ! ((first == 0 && last == 7)) && ! ((first >= 2 && last - first == 5)); then
  echo "Use 0 7 for a fresh run, or six consecutive variants for a declared repeat." >&2
  exit 2
fi
[[ -f "$run_dir/launch_manifest.txt" ]] || {
  echo "Missing launch_manifest.txt: use the matching service directory." >&2; exit 2;
}
url=${DSPARK_URL:-http://127.0.0.1:${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}/v1/chat/completions}
model_path=${MODEL_PATH:-/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt}
curl -fsS --max-time 5 "${url%/v1/chat/completions}/health"

run_case() {
  local name=$1
  shift
  local json="$run_dir/$name.json"
  local log="$run_dir/$name.client.log"
  if [[ -e "$json" || -e "$log" ]]; then
    echo "Refusing to overwrite $name. Use a fresh run directory." >&2
    exit 2
  fi
  local status=0
  timeout --signal=TERM --kill-after=10s 180 \
    python3 "$script_dir/benchmark_dspark.py" --model-path "$model_path" \
      --url "$url" --output "$json" "$@" >"$log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$run_dir/$name.exit_code"
  if ((status != 0)); then
    echo "$name failed (exit=$status); retain $log and rank logs." >&2
    exit "$status"
  fi
  python3 "$script_dir/check_result.py" "$json" >"$run_dir/$name.accuracy_gate.json"
  python3 - "$json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
x = json.loads(p.read_text())
m = x["metrics"]
print(json.dumps({"request": p.name, "ttft_ms": m["ttft_ms"],
    "tpot_ms": m["tpot_ms"], "cached_tokens": m["usage"]["prompt_tokens_details"]["cached_tokens"],
    "strict": x["accuracy"]["strict_text_passed"]}), flush=True)
PY
}

if ((first == 0)); then
  for warmup in 1 2; do
    run_case "warmup_$warmup" --prompt-tokens 64 --output-tokens 64 --count-to 32
  done
fi
for ((variant=first; variant<=last; ++variant)); do
  run_case "formal_variant_$variant" --prompt-tokens 2500 --output-tokens 1500 \
    --count-to 750 --prefix-cache-rate 0.65 --prompt-variant "$variant"
done
hot_first=$first
if ((first == 0)); then hot_first=2; fi
python3 "$script_dir/summarize_window.py" "$run_dir" "$hot_first" "$last" \
  | tee "$run_dir/window_${hot_first}_${last}.summary.json"
```

### summarize_window.py

<!-- script: summarize_window.py -->
```python
#!/usr/bin/env python3
"""Evaluate all six declared hot requests; never average a partial window."""
import json
import statistics
import sys
from pathlib import Path

from check_result import checks_for, positive_number


def evaluate_window(run: Path, first: int, last: int) -> dict:
    if first < 0 or last - first + 1 != 6:
        raise ValueError("A hot window must contain exactly six consecutive variants")
    rows = []
    for variant in range(first, last + 1):
        path = run / f"formal_variant_{variant}.json"
        try:
            data = json.loads(path.read_text())
            checks = checks_for(data, prompt=2500, output=1500, count=750)
            metrics = data.get("metrics", {})
            cached = (metrics.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens")
            checks["expected_variant"] = data.get("prompt_variant") == variant
            checks["prefix_rate"] = data.get("prefix_cache_prompt", {}).get("configured_prefix_cache_rate") == 0.65
            checks["cached_1536"] = cached == 1536
            rows.append({"variant": variant, "ttft_ms": metrics.get("ttft_ms"),
                         "tpot_ms": metrics.get("tpot_ms"), "cached_tokens": cached,
                         "strict": checks["strict_full_text"], "checks": checks,
                         "valid": all(checks.values())})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            rows.append({"variant": variant, "valid": False, "error": str(exc)})
    # Invalid/missing timing in any row makes the window unevaluable, not shorter.
    complete_timing = all(positive_number(row.get(k))
                          for row in rows for k in ("ttft_ms", "tpot_ms"))
    ttft = statistics.mean(row["ttft_ms"] for row in rows) if complete_timing else None
    tpot = statistics.mean(row["tpot_ms"] for row in rows) if complete_timing else None
    passed = (all(row["valid"] for row in rows) and complete_timing
              and ttft < 250 and tpot < 14 and tpot <= 13.758500)
    return {"first_variant": first, "last_variant": last, "samples": rows,
            "mean_ttft_ms": ttft, "mean_tpot_ms": tpot,
            "gate": "PASS" if passed else "FAIL"}


def main() -> None:
    try:
        result = evaluate_window(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    except (IndexError, ValueError) as exc:
        result = {"gate": "FAIL", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
```
