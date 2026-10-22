# DeepSeek-V4 Flash MUSA 高性能复现

适用分支 dsv4/perf/musa-deepseek-v4-overlap-20260819。本文档复现的是 MUSA
S5000 单机 8 卡、TP8/EP1/DP1/CP1、Full Graph、MTP k=1、prefix-cache 0.65
的 2.5K/1.5K 热态链路。命令默认每个 rank 一个进程，不能在同一服务中并发提交
多个正式请求。

## 1. 已验证版本和结果

~~~text
commit: c663a78d09d062e997f2a5c3aecfe555dc9540f0
binary sha256: 5c24e638b30c3e015b8217c874a9e9bfca591eeac1ddf9f0a12e341ffd3c4576
model: DeepSeek-V4-Flash-FP8
input/output: 2500/1500, batch=1, count-to=750
topology: TP8/EP1/DP1/CP1
graph: Full Graph + graph VMM + decode-no-padding
schedule overlap: true
MTP: speculative_algorithm=MTP, num_speculative_tokens=1
prefix cache: rate=0.65, measured cached tokens=1536
~~~

完整热态窗口均保留 6 个请求、不删除长尾：

| 独立服务 | 平均 TTFT | 平均 TPOT | 严格精度 |
| --- | ---: | ---: | --- |
| Run539 | 249.788833 ms | 13.503667 ms | 6/6 1..750 PASS |
| Run540 fresh | 248.978667 ms | 13.467500 ms | 6/6 1..750 PASS |

验收条件：热态平均 TTFT <250 ms；热态平均 TPOT 同时 <14 ms 且 <=13.758500
ms；所有请求严格输出 1..750。Run540 刚启动后的首窗 251.879000/13.456333 ms
按规则记录为失败，充分热态的第二窗才用于通过判定。

## 2. 环境和外置资产

已验证环境：MTT S5000 8 卡、driver 5.2.0、宿主机 mthreads-gmi 2.4.2、
Python 3.10、PyTorch 2.9.1、TorchMUSA 2.9.1+a18d871、MCC 5.1、MUSA_HOME
=/usr/local/musa。模型权重、AOT ops、custom all-reduce 不在 GitLab 源码中，
必须准备与 TorchMUSA/driver ABI 匹配的本地制品。

最优链路还需要三个外置工具文件：

~~~text
run_external_tp8_target.sh
counting_stream_benchmark.py
deepseek_v4_stream_benchmark.py
~~~

验收机的来源路径：

~~~text
/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/run_external_tp8_target.sh
/data/nfs_shared/xllm2_musa_env/counting_stream_benchmark.py
/data/nfs_shared/jiaming.wang/test/fhxllm/xllm-musa-worker2/tools/deepseek_v4_stream_benchmark.py
~~~

共享 NFS 可执行：

~~~bash
export PERF_TOOLS=/opt/xllm-dsv4-perf-tools
mkdir -p "$PERF_TOOLS"
cp /data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/run_external_tp8_target.sh "$PERF_TOOLS/"
cp /data/nfs_shared/xllm2_musa_env/counting_stream_benchmark.py "$PERF_TOOLS/"
cp /data/nfs_shared/jiaming.wang/test/fhxllm/xllm-musa-worker2/tools/deepseek_v4_stream_benchmark.py "$PERF_TOOLS/"
chmod +x "$PERF_TOOLS/run_external_tp8_target.sh"
# 路径不同则修改 counting_stream_benchmark.py 的 REFERENCE_BENCHMARK 和 MODEL_PATH。
~~~

AOT ops 目录必须包含 gemm_ops/gemm_ops.so、mhc_pre_big_fuse_ops/mhc_pre_big_fuse_ops.so、
mhc_prenorm_ops/mhc_prenorm_ops.so，以及 mt32/mt320 的 Mate MegaMoE stage1/stage2
四个目录。已验证 custom-AR 文件 SHA-256：
c2578d935edc3af6b4a39f124d06c8e13b68d345bc7a07c27f5581ce310082a5。新机器的
AOT/custom-AR hash 不同，必须作为新的环境边界记录。

## 3. 拉取和构建

~~~bash
git clone https://sh-code.mthreads.com/his-devtech/xllm-musa.git xllm-dsv4-musa-perf
cd xllm-dsv4-musa-perf
git fetch origin dsv4/perf/musa-deepseek-v4-overlap-20260819
git checkout dsv4/perf/musa-deepseek-v4-overlap-20260819

export XLLM_ROOT=$PWD
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-FP8
export BUILD_DIR=$XLLM_ROOT/build/musa-perf-validation
export XLLM_BIN=$BUILD_DIR/xllm/xllm
export MUSA_HOME=/usr/local/musa
export TORCH_MUSA_PYTHONPATH=$(python3 -c 'import torch_musa; print(torch_musa.core.cmake_prefix_path)')
export OPS_PATH=/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops
export CUSTOM_AR_SO=/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so
export LAUNCHER=/opt/xllm-dsv4-perf-tools/run_external_tp8_target.sh
export BENCHMARK=/opt/xllm-dsv4-perf-tools/counting_stream_benchmark.py
~~~

新 build 目录：

~~~bash
cmake -S "$XLLM_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$MUSA_HOME/tools/musamapping/mcc_wrapper" \
  -DUSE_MUSA=ON -DUSE_CUDA=OFF -DUSE_NPU=OFF -DUSE_MLU=OFF \
  -DUSE_DCU=OFF -DUSE_ILU=OFF -DDEVICE_TYPE=USE_MUSA \
  -DDEVICE_ARCH=X86 -DBUILD_TESTING=OFF \
  -DTORCH_CUDA_ARCH_LIST=9.0 -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" --target xllm -j16
cmake -N -LA "$BUILD_DIR" | rg '^(USE_(MUSA|CUDA|NPU|MLU|DCU|ILU)|CMAKE_BUILD_TYPE):'
sha256sum "$XLLM_BIN"
ldd "$XLLM_BIN" | rg 'not found' && exit 1 || true
~~~

必须是 USE_MUSA=ON、USE_CUDA=OFF，其他 device 选项 OFF。不能定义 USE_CUDA
冒充 MUSA。已有验证过的 build cache 可跳过 configure，只执行 cmake --build。

## 4. 启动和宿主机占用检查

容器中看不到其他容器 PID 时，必须在宿主机执行 mthreads-gmi；性能窗口前、中、后
分别保存输出：

~~~bash
export RUN_DIR=$XLLM_ROOT/perf_runs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
date -Is | tee "$RUN_DIR/host_before.txt"; mthreads-gmi | tee -a "$RUN_DIR/host_before.txt"

XLLM2_EXTERNAL_TP8_LOG_DIR="$RUN_DIR" \
XLLM2_EXTERNAL_TP8_BIN="$XLLM_BIN" \
XLLM2_EXTERNAL_TP8_MODEL="$MODEL_PATH" \
XLLM2_EXTERNAL_TP8_OPS_PATH="$OPS_PATH" \
XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO="$CUSTOM_AR_SO" \
XLLM2_EXTERNAL_TP8_MASTER_ADDR=127.0.0.1:20640 \
XLLM2_EXTERNAL_TP8_HTTP_PORT=18640 \
XLLM2_EXTERNAL_TP8_GRAPH=true \
XLLM2_EXTERNAL_TP8_SCHEDULE_OVERLAP=true \
XLLM2_EXTERNAL_TP8_PREFIX_CACHE=true \
XLLM2_EXTERNAL_TP8_IN_BATCH_PREFIX_CACHE=false \
XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS=1 \
XLLM2_EXTERNAL_TP8_SPECULATIVE_ALGORITHM=MTP \
XLLM2_EXTERNAL_TP8_MAX_TOKENS_PER_BATCH=10240 \
XLLM2_EXTERNAL_TP8_BLOCK_SIZE=128 \
XLLM2_EXTERNAL_TP8_NUM_RESPONSE_HANDLING_THREADS=4 \
  bash "$LAUNCHER"
~~~

launcher 会启动 8 个 rank，开启 Full Graph/VMM/decode-no-padding/schedule overlap，
并先跑两次 64/64 Graph warmup。驻留时保存：

~~~bash
date -Is | tee "$RUN_DIR/host_during.txt"; mthreads-gmi | tee -a "$RUN_DIR/host_during.txt"
~~~

每张卡应只有对应 xLLM rank、显存约 72 GiB。出现其他进程、rank 缺失、warmup
不通过或端口占用时，停止并记录，不能纳入性能统计。

## 5. 正式请求、复测和统计

variant 0 为冷态，variant 1 建立 prefix cache；验收只取 variant 2..7：

~~~bash
for variant in 0 1 2 3 4 5 6 7; do
  timeout 900s python3 "$BENCHMARK" \
    --url http://127.0.0.1:18640/v1/chat/completions \
    --prompt-tokens 2500 --output-tokens 1500 --count-to 750 \
    --prefix-cache-rate 0.65 --prompt-variant "$variant" \
    --output "$RUN_DIR/formal_variant_$variant.json" || exit $?
done
~~~

每个 JSON 应有 prompt=2500、completion=1500、accuracy.passed=true、
mismatch_index=null；variant 1..7 的 cached_tokens 应为 1536。无删样本统计：

~~~bash
python3 - "$RUN_DIR" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
x = [json.loads((d / f"formal_variant_{i}.json").read_text()) for i in range(2, 8)]
ttft = [v["metrics"]["ttft_ms"] for v in x]
tpot = [v["metrics"]["tpot_ms"] for v in x]
cached = [v["metrics"]["usage"]["prompt_tokens_details"]["cached_tokens"] for v in x]
exact = [v["accuracy"]["passed"] for v in x]
print("TTFT", ttft, "mean", statistics.mean(ttft))
print("TPOT", tpot, "mean", statistics.mean(tpot))
print("cached", cached, "strict_exact", exact)
ok = (statistics.mean(ttft) < 250 and statistics.mean(tpot) <= 13.758500
      and statistics.mean(tpot) < 14 and all(exact) and all(v == 1536 for v in cached))
print("GATE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
strings "$RUN_DIR/rank0.log" | rg 'request.cpp:98' | tail -6 | tee "$RUN_DIR/server_hot_requests.txt"
~~~

建议通过首窗后执行 variant 8..13 的完整复测；独立重启服务后重复 warmup 和
两个窗口，才可称为 fresh-service reproduction。停止服务后保存 host_after.txt；
正常应为 8 卡 0 MiB、无残留 xLLM 进程。禁止 GPU reset。

## 6. 失败处理、源码边界和归档

没有 1536 prefix hit、TTFT/TPOT 超门限、严格 1..750 失败、AOT/custom-AR hash
不同、Graph warmup 失败或宿主机有外部任务时，保留全部 JSON、rank 日志、manifest
和 mthreads-gmi 输出，标记为失败；不能删慢样本、降低输出长度、关闭 Graph/MTP
或改用 USE_CUDA。

最终源码只修改两个 xllm/core/kernels/musa 私有文件，共 3 行；外置 launcher、
benchmark、AOT ops、custom-AR 和模型权重不属于 GitLab 产品代码。每轮至少归档：

~~~text
launch_manifest.txt
launcher_warmup_1.json / launcher_warmup_2.json
formal_variant_0.json ... formal_variant_7.json
repeat_variant_8.json ... repeat_variant_13.json
rank0.log ... rank7.log
host_before.txt / host_during.txt / host_after.txt
~~~

完整历史参考目录：

~~~text
/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/run539_fused_query_expert_metadata_192
/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/run540_fresh_repro_expert_metadata_192
~~~
