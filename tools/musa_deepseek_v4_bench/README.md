<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->

# DeepSeek-V4-Flash MUSA 1P1D：Build 与 Bench

参照 `tools/musa_qwen3_8_vl_bench`，提供 DeepSeek 的角色编译命令和纯文本流式压测客户端。
本文核对的源码为 `feihu-1016-dev`，提交
`361874bcd6a873e674380c45dac82964ecc7f09d`（2026-10-23）。

| 项目 | 配置 |
| --- | --- |
| 模型 / API model ID | `DeepSeek-V4-Flash-0731-FP8-mt` |
| 模型和 tokenizer | `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt` |
| Prefill | `10.121.38.94`，8 卡 TP8，1 个实例 |
| Decode | `10.121.38.90`，8 卡 TP8，1 个实例 |
| 测试入口 | xllm-service 控制面的 `/v1/chat/completions` |
| 20K / 100 | 精确 20,000 输入、100 输出 token；并发 1～16；共享输入比例 0.65 |
| 30K / 1K | 精确 30,000 输入、1,024 输出 token；默认冷缓存，与此前 30K/1024 口径一致 |

下列命令在 Linux 容器内执行。90/94 的实验环境使用 `wjm_xllm1008`；使用其他
容器时，需要具备相同的 MUSA、Torch-MUSA 和模型依赖。

## Build

### 1. 获取同一版本的 P / D 源码

在共享盘新建目录。两个角色必须使用独立源码目录：`setup.py` 在每份源码下使用
固定的 CMake 和输出目录，不能在同一个目录先后切换角色来保留两套二进制。

```bash
export DS_ROOT=/data/nfs_shared/jd/deepseek_v4_build_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DS_ROOT"

git clone --branch feihu-1016-dev --single-branch \
  https://github.com/fay85/xllm-musa.git "$DS_ROOT/prefill"
export DS_REV=$(git -C "$DS_ROOT/prefill" rev-parse HEAD)

git clone --branch feihu-1016-dev --single-branch \
  https://github.com/fay85/xllm-musa.git "$DS_ROOT/decode"
git -C "$DS_ROOT/decode" checkout --detach "$DS_REV"

git -C "$DS_ROOT/prefill" submodule update --init --recursive
git -C "$DS_ROOT/decode" submodule update --init --recursive
printf '%s\n' "$DS_REV" > "$DS_ROOT/source_revision.txt"
```

复现本文检查的版本时，将两个目录都 checkout 到 `361874bcd6a873e674380c45dac82964ecc7f09d`，
再更新子模块。首次配置需要访问 Git、子模块和构建依赖源。

### 2. 分别编译

在已有 MUSA 开发容器中，先确认构建依赖能被当前 Python 找到：

```bash
python3 -c 'import torch, torch_musa, tvm_ffi; print(torch.__version__)'
cmake --version
ninja --version
test -x /usr/local/musa/tools/musamapping/mcc_wrapper
```

还需要容器中的 MKL 开发库；若不在默认搜索路径，设置 `MKLROOT`。以下使用本次
1P1D 环境的 `XLLM_USE_ETCD=OFF`，关闭 Mooncake 的可选 etcd wrapper 构建；
xllm-service 控制面仍使用自己的 etcd。

P 端：

```bash
cd "$DS_ROOT/prefill"
MAX_JOBS=32 XLLM_USE_ETCD=OFF \
  CMAKE_ARGS="-DXLLM_DSV4_PD_ROLE=prefill" \
  python3 setup.py build --device musa --generate-so false
```

D 端：

```bash
cd "$DS_ROOT/decode"
MAX_JOBS=32 XLLM_USE_ETCD=OFF \
  CMAKE_ARGS="-DXLLM_DSV4_PD_ROLE=decode" \
  python3 setup.py build --device musa --generate-so false
```

`MAX_JOBS` 控制编译并行度。重复相同命令会重新配置并复用已有编译对象。
包含本工具目录的源码也可以使用包装脚本；它会检查已有构建目录的角色，防止误覆盖：

```bash
# 在各自源码根目录执行。
MAX_JOBS=32 XLLM_USE_ETCD=OFF bash tools/musa_deepseek_v4_bench/build.sh prefill
# D 的独立源码目录中，将最后一个参数改为 decode。
```

Linux x86_64 / CPython 3.10 的产物位置：

```text
$DS_ROOT/prefill/build/lib.linux-x86_64-cpython-310/xllm/xllm
$DS_ROOT/decode/build/lib.linux-x86_64-cpython-310/xllm/xllm
```

每个角色使用自己构建生成的运行库，包括对应 CMake 构建目录
`build/cmake.linux-x86_64-cpython-310/mooncake-common/libasio.so`。
Python 版本不同，目录后缀也会变化。部署时记录 P / D 二进制的 SHA256 和源码提交，
以便将性能结果对应到实际运行的版本。

### 3. 部署所需的运行依赖

DeepSeek native MoE 还需要 Mate 0.2.7 的 MUBIN 和对应的 launcher `.so`。
90/94 共享盘上已核实存在的实验依赖如下，在启动 P / D 进程的 shell 中设置：

```bash
export DS_DEPS=/data/nfs_shared/jd/feihu_stability_merge_20261022
export DS_RUNTIME=/opt/xllm-pd-20261008-r4/xllm_pd_runtime/lib
export MATE_MUBIN_DIR=$DS_DEPS/deps/mate_mubin/mubin
export FLASHINFER_OPS_PATH=$DS_DEPS/runtime/merged_ops
export LD_PRELOAD=$DS_RUNTIME/libxllm_custom_ar_sglang_current.so:/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so
export XLLM_TILELANG_LIB=$DS_RUNTIME/libtilelang.so
export TVM_LIBRARY_PATH=$DS_RUNTIME
```

`MATE_MUBIN_DIR` 下需包含
`gemm/fbd1a6df72047350033cb4c524dd8145b8f18698/mubin/*.o`，
launcher 位于 `$FLASHINFER_OPS_PATH/run_<kernel>/run_<kernel>.so`。
迁移容器时同步这组匹配的依赖，并让 `LD_LIBRARY_PATH` 能找到当前角色的
`libasio.so`、Torch/MUSA、muDNN、MCCL 和 OpenMPI 库。这些是已有实验依赖路径，
本次没有重新编译或验证它们与新提交的完整推理组合。

## Bench

### 1. 准备服务与客户端

先将对应角色的二进制部署为 94 的 Prefill 和 90 的 Decode，并通过同一个
xllm-service 控制面提供服务。客户端连接控制面，不能把 P / D 的内部端口当作压测入口。

沿用 1P1D 的测试配置：P / D 各 TP8，`block_size=128`，DSpark N6/q7，
graph 和 prefix cache 开启，Mooncake `PULL`；两端设置
`XLLM_DSV4_CHUNK_PREFIX_SNAPSHOT=1`。P 的 chunk/batch 为 16384、关闭 schedule overlap；
D 的 chunk 为 8192、batch 为 10032、开启 overlap；两端最大序列数均为 16。
正式测量时两端显式使用 `--enable_online_profile=false`，并记录完整启动参数。

2026-10-23 已只读确认的现有入口为 `http://10.121.38.90:32888`，其 `/v1/models`
返回上表中的模型。**这个入口当前运行的是恢复后的基线二进制，并非本次拉取的
`361874bcd`。测试最新代码前，需要先部署新编译的 P / D 产物，再设置对应的 `DS_URL`。**
现有基线的 online profile 仍为开启状态，进行正式性能比较前也需统一该配置。

在包含本工具目录的源码根目录执行；如果只同步工具目录，还需同步 `scripts/logger.py`，
保持它相对于仓库根目录的位置。

```bash
python3 -m pip install aiohttp tokenizers
export DS_MODEL=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export DS_MODEL_ID=DeepSeek-V4-Flash-0731-FP8-mt
export DS_API=http://10.121.38.90:32888
export DS_URL=$DS_API/v1/chat/completions
export DS_RESULTS=/data/nfs_shared/jd/deepseek_v4_bench_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DS_RESULTS"
curl --fail --silent --show-error "$DS_API/v1/models"
python3 tools/musa_deepseek_v4_bench/bench.py --help
```

仅检查 tokenizer 和精确输入构造时，可在后面的任意单档命令中增加 `--prepare-only`，
并使用单独的 `--out` 目录。该模式只写提示词和 manifest，不发送 HTTP 请求；
正式执行时使用另一个新目录。

### 2. 20K / 100，并发 1～16，prefix 0.65

```bash
set -euo pipefail
for c in $(seq 1 16); do
  python3 tools/musa_deepseek_v4_bench/bench.py \
    --url "$DS_URL" --model "$DS_MODEL_ID" --tokenizer "$DS_MODEL" \
    --input-tokens 20000 --output-tokens 100 --prefix-ratio 0.65 \
    --concurrency "$c" --requests 64 --warmup 1 \
    --out "$DS_RESULTS/20k100/c$c"
done
```

每档采用闭环并发，完成一个请求后补入下一个，累计 64 个正式请求。`--warmup 1`
在热缓存模式下先发送一条 primer，再发送一条热缓存 probe；两条都不计入正式统计。
每次运行使用独立的共享前缀标记和请求后缀，避免不同档位继承上一档的相同输入。

`--prefix-ratio 0.65` 表示 20,000 个输入 token 中有 13,000 个共享前缀 token，
包含聊天模板开销；它不是显存或 KV cache 容量比例。128-token block 对齐后，
预期实际命中为 12,928 token，即 64.64%。服务需启用 prefix cache。
客户端检查响应中的实际缓存计数，缓存未命中或长度不符时不会把该请求算作有效样本。

需要三轮时，在上述循环外加 `for round in 1 2 3; do ...; done`，并将输出改成
`"$DS_RESULTS/20k100/r$round/c$c"`。保留每轮结果，不覆盖此前目录。

### 3. 30K / 1K

沿用此前历史测试的 30,000 / **1,024** token 口径，先测并发 1 和 16：

```bash
set -euo pipefail
for c in 1 16; do
  python3 tools/musa_deepseek_v4_bench/bench.py \
    --url "$DS_URL" --model "$DS_MODEL_ID" --tokenizer "$DS_MODEL" \
    --input-tokens 30000 --output-tokens 1024 --prefix-ratio 0 \
    --concurrency "$c" --requests 64 --warmup 1 \
    --out "$DS_RESULTS/30k1024/c$c"
done
```

冷缓存模式为每个请求生成不同的开头，预期完整 block 命中为 0。若需要字面意义上的
1,000 输出 token，将 `--output-tokens` 改成 `1000`，并单独命名结果目录。
比较历史结果时，保持输入/输出长度、缓存比例、并发、请求数和服务配置一致；
20K/100 热缓存结果不能直接与 30K/1024 冷缓存结果比较。

### 4. 结果口径

客户端发送 `thinking=false`、`temperature=0`、`ignore_eos=true` 和流式 usage 请求，
按 DeepSeek-V4 的单用户聊天模板构造精确长度输入。正式统计只计入通过 HTTP/SSE、
输入/输出 token 数、缓存校验，以及连续整数输出一致性校验的请求。任一请求失败时
命令返回非零，便于中止 sweep；失败窗口不输出可用于性能比较的汇总成绩。
输出目录必须是新目录。

- TTFT：请求发出到首个非空输出的客户端延迟，包含排队和网络时间。
- TPOT：首个到最后一个输出之间的时间，除以实际输出 token 数减一。
- E2E：客户端请求总耗时。
- QPS / output tokens/s：有效请求数 / 输出 token 数除以正式测量窗口时长。

每档结果目录包含：

```text
manifest.json                # 参数、tokenizer/script/payload 哈希和协议
payloads.json                # 实际请求输入
warmup/、probe/、formal/     # 原始 SSE、事件时间戳、逐请求响应及阶段 summary
summary.json                 # 整体是否通过、正式窗口的延迟/吞吐统计
```

冷缓存模式没有 `probe/`。TTFT、TPOT、E2E 保存 mean、median、p90、p99 等统计；
以 `summary.json` 中 `passed=true` 且正式阶段 `performance_eligible=true` 为有效完成标志。
分位数在排序后的样本上按 `(n-1) × p` 位置线性插值。

保存服务启动参数、源码版本、P / D 二进制 SHA256，以及对应时间段的服务日志，
与客户端结果一起用于对比。

本目录脚本已通过 Bash 语法、Ruff 和 17 项本地验证（含模拟流式接口）。
2026-10-23 在 94 的真实模型 tokenizer 上，20K/100 热前缀、30K/1024 冷缓存、
30K/1000 冷缓存的 `--prepare-only` 也全部通过；196 条准备输入均满足目标长度。
记录保存在 `/data/nfs_shared/jd/deepseek_bench_validation_20261023_175708/validation_summary.json`。
这些检查没有执行模型推理；本次未验证新提交的完整构建和实机性能，
本文不将旧二进制的实测成绩作为最新源码的性能结果。
