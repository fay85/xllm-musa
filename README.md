# Qwen3.8-27B BF16 TTFT 优化（MUSA S5000，分支 `ttft-tp2-decode`）

这个分支记录 2026-10-28 在 dev92 上做的 Qwen3.8-27B BF16 TTFT 优化：改了什么、实测结果、还要做的修复，以及复现步骤。另一个 agent 可以照第 8 节从头复现。上游项目原来的 README 见 `main` 分支或 `README_zh.md`。

## 0. 本分支

- 基础 commit：`92dbd809c`（上游 xLLM-AI/xllm 的 `feat: enable fused addrmsnorm on prefill for the qwen3 moe decoder layer. (#500)`）。
- 在它之上有三个 commit：
  1. `musa: snapshot decode TP2, PCP custom AR and PD pull work on 92dbd809c`：dev92 容器 `decode_TP2_dev` 里 `/workspace/xllm-mt-main` 工作区原有的全部未提交改动，再加上 `xllm`、`tests`、`tools`、`scripts` 下的未跟踪源文件和 `_build_cuda_graph_musa.sh`。不含 `third_party`、`tools/pd_guard_20261006`、`*.bak*`、`*.orig`。内容包括 MUSA TP2 decode 的 custom all-reduce、PCP 在 `cp_group` 上改用 custom all-reduce、PD PULL 的 compact strided KV 绑定和批量 scatter，以及这个工作区里其他进行中的 MUSA 改动。
  2. `musa: reduce TP all-reduce above 8 MiB in custom-AR slices`：第 3.3 节的代码改动，只动了 `musa_custom_all_reduce.{h,cpp}`。
  3. `tools,docs: BF16 TTFT reproduction scripts and README`：`tools/decode_tp2_dev/bf16_ttft/` 下的复现脚本，以及本 README。
- 不在仓库里、复现时需要的东西：模型（`/data/nfs_shared/models/...`）、编译好的 binary、xllm_service 和 etcd 二进制（`/opt/xllm_service/bin/`）、master tokenizer（`/root/xllm_vl_bench/master_tokenizer`）、图片（`/workspace/xllm_vl_bench_ref/images`）、mate 算子缓存（`/opt/mate027_cache/...`）。这些都在第 7 节的镜像里。

## 1. 环境与测法

- 机器：dev92（10.121.38.92），8 张 MTT S5000（80 GB）。容器 `decode_TP2_dev`，源码 `/workspace/xllm-mt-main`。
- 模型：`/data/nfs_shared/models/Qwen3.8-27B`（BF16，约 55.6 GB）。MTP 权重以 `mtp.*` 融合在主 checkpoint 里，`--draft_model` 指向同一目录；draft 的 model type 由 `qwen3_5` 改写为 `qwen3_5_mtp`，embedding 和 lm_head 与 target 共享。
- 拓扑：3 个 prefill 实例（GPU 0,1 / 2,3 / 4,5），decode 为两路 TP1（GPU 6、GPU 7）。
- Workload：`/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv/workload_c6.json`，并发 6，60 条 measured prompt，文本 19511–20467 token，加 5 张 480p 图（每张 407 token），OSL 100，预期 KV 命中 0.7929；MTP K=3，分块 1024，前缀缓存和 chunked prefill 打开。
- 测法：先用 measured prompt 跑一轮捕获，再重复计时波，取第一遍新增 `Lazy captur` 行数为 0 的结果。

背景：decode 用两路 TP1 明显好于一路 TP2。FP8、C6、同一天的数据：两路 TP1 是 QPS 2.622、TTFT 788.6 ms、TPOT 14.45 ms；一路 TP2 是 QPS 2.135、TTFT 851.5 ms、TPOT 19.42 ms。下面 BF16 的结果都用两路 TP1 decode。

## 2. 结果

BF16，C6，稳态（无新捕图），错误均为 0。

| 配置 | TTFT 均值 / 稳态 | TTFT p90 | KV 命中 | QPS | TPOT | MTP 接受率 | run 目录 |
|---|---|---|---|---|---|---|---|
| 起点：prefill DP3 CP2，显存 0.90，prefill 槽 24，decode 槽 8 | 2610.6 / 2624.8 ms | 2952 ms | 0.415 | 1.287 | 19.43 ms | 0.7145 | `c6_bf16_tp1`（第 10 遍） |
| prefill 改 DP3 TP2（槽 96），decode 槽 8→4 | 1172.1 / 1111.2 ms | 1208 ms | 0.7928 | 1.761 | 21.94 ms | 0.7008 | `c6_bf16_pftp2_tp1_s4`（第 5 遍） |
| 再加 TP all-reduce 切片走 custom AR | 1154.9 / 1100.9 ms | 1278 ms | 0.7928 | 1.754 | 22.14 ms | 0.6973 | `c6_bf16_pftp2_tp1_slices`（第 6 遍） |

TTFT 从 2610 ms 降到 1155 ms，降了 56%。run 目录都在 `/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/` 下（镜像里有）。

Binary（在 `/workspace/decode_tp2_dev_artifacts/bin/`）：

- 第二行：`xllm.native_tp2_pcp_ar_switch`，SHA256 `21100d8bbfab70ff564474039926ecad849ca311516855e2516892bb0e4ce636`，对应 commit 1 的源码。
- 第三行：`xllm.native_tp2_pf_tp_slices`，SHA256 `bf61b3af20b20154d30bbd37ac64658a5cf3c9623ed6c616794a3c5511ced6c0`，对应 commit 2 的源码。
- 不能覆盖：`xllm.native_tp2_pull_coalesced.r4`，SHA256 `5817498a4d856329017aaaf713dd97d37db790b80f1e39cac538a9649f14f4ba`。

## 3. 已做的优化和修复

### 3.1 prefill 从 DP3 CP2 换成 DP3 TP2（配置，TTFT 改善的主体）

- launcher 参数 `--prefill-cp-size 1`，即 prefill 用 `--cp_size=1 --nnodes=2`，两卡组成 TP2。
- 每卡权重减半：加载后每卡可用显存从 18.37 GB 涨到 47.53 GB。
- 每个 rank 的 KV 从 10.38 GB / 1585 块涨到 39.54 GB / 15601 块，GDN 线性状态槽从 24 个回到 96 个（每槽从 149.62 MB 降到 74.81 MB）。
- KV 命中从 0.415 回到 0.7928，和 FP8 一致。
- PD 传输是 prefill TP2 → decode TP1，reshard planner 支持这种缩小。实测 decode 从两个 prefill rank 拉取，每条 35–77 ms，`success=true`。

### 3.2 TP1 decode 线性状态槽从 8 降到 4（配置，3.1 的前提）

- launcher 参数 `--decode-state-cache-slots 4`。decode KV 从 1041 块涨到 1589 块（约 10.2 万 token），线性状态预留从 5.68 GB 降到约 3.4 GB。
- 原因：prefill 变快后，三条请求（22172 + 22388 + 21899 token，共 1040 块）同时到了一个 decode，而它只有 1041 块。第一个生成 token 就分不到块，调度器用 `enqueue_decode_restore`（`scheduler_policy.cpp`）把请求抢占回 prefill 队列。这一步是根据块数和调度代码推断的，日志里没有抢占记录；崩溃栈本身是实测。decode 本地重算这条图文 prompt 时，`BatchInputBuilder::get_mrope_positions` 调到 `Qwen3VLMPositionGenerator::generate`（`models/vlm/mposition/mposition.cpp`），但 decode 上的 `MMData` 里没有 `image_grid_thw`，对空 tensor 取下标抛 `c10::Error`，decode 进程退出，服务随后返回 503。
- 这次只是靠腾出 KV 避开了，代码路径还在，见 4.1。

### 3.3 超过 8 MiB 的 TP all-reduce 切片走 custom AR（代码，commit 2）

- 文件：`xllm/core/framework/parallel_state/musa/musa_custom_all_reduce.h` 和 `.cpp`。新增私有成员 `allreduce_in_slices`；`allreduce_tp` 在 `allreduce` 不接受（超过 8 MiB）时改调它，再不行才退回 MCCL。
- 条件：custom AR 已启用、设备一致、dtype 为 fp32/fp16/bf16、contiguous、16 字节对齐、总字节数大于 8 MiB 且能被 16 整除。展平后切成均衡的几段，每段 16 字节对齐、不超过 8 MiB。两个 rank 形状一致，切法也一致。第一段失败就整体退回 MCCL，之后的段失败直接 CHECK。
- 日志：`[MUSA_TP_REDUCE] backend=custom_slices`。prefill TP2 每个 1024 token 分块的 128 次 `[1024, 5120]`（10 MiB）all-reduce 原来全走 MCCL，现在切成两段走 custom AR。
- 编译：`cd /workspace/xllm-mt-main && bash ./_build_cuda_graph_musa.sh`，增量编译约 106 秒。
- 收益：稳态 TTFT 从 1111 ms 到 1101 ms，2–3%，在噪声范围内。

### 3.4 launcher 改动（`tools/decode_tp2_dev/bf16_ttft/launch_pd_comparison.py`）

新增参数，默认值都保持原来的行为：

| 参数 | 作用 |
|---|---|
| `--model` / `--model-id` | 指定模型路径和 HTTP model id；`--draft_model` 跟随 `--model`（用融合 MTP） |
| `--prefill-cp-size {1,2}` | prefill 的 CP 大小，1 即 TP2 |
| `--decode-state-cache-slots` | decode 的 `--max_linear_state_cache_slots`，0 为自动 |
| `--decode-memory` | decode 单独的显存利用率，不设则同 `--memory` |
| `--online-profile-dir` | 给 prefill 加 `--enable_online_profile=true --profile_dir=...` |

### 3.5 原因说明（更正）

起点 CP2 命中只有 0.415，不是因为 KV 块装不下前缀。生成器 `build_comparison_workload_stable.py` 里 measured prompt 只用 2 种前缀（含 2 张图或 3 张图），前缀合计约 2.7 万 token，1585 块（约 10 万 token）装得下。更可能的瓶颈是 GDN 线性状态槽：BF16 CP2 在 0.90 下只能给 24 个，FP8 CP2 和 BF16 TP2 都是 96 个。这一点没有单独做实验验证。TP2 同时放大了状态槽和 KV，所以修法不受影响。

## 4. 要做的 fix 和优化

### 4.1 P0：decode 抢占后本地重算图文请求会崩（正确性）

- 现状：3.2 只是把触发条件推远。一个 BF16 TP1 decode 有 1589 块，同时超过 4 条约 22k token 的请求就会再触发，并发升高或路由不均时都可能碰到。
- 涉及代码：`xllm/core/scheduler/scheduler_policy.cpp`（`enqueue_decode_restore`、decode 阶段块不够时的抢占）、`xllm/core/scheduler/disagg_pd_scheduler.cpp`（decode 端分配和 first-generation 接收）、`xllm/core/framework/batch/batch_input_builder.cpp`（`get_mrope_positions`）、`xllm/models/vlm/mposition/mposition.cpp`（`Qwen3VLMPositionGenerator::generate`）。
- 修法（按推荐顺序）：
  1. decode 接收 PD 请求时，按 prompt + `max_tokens` + MTP（K+1）余量预留块，不够就拒绝或让 prefill 侧排队，不再超卖。
  2. decode 上从 PD 收来、没有本地图片数据的请求，不走"抢占后本地重算"：改为抢占别的请求、等待，或者干净地失败并返回明确错误。
  3. `get_mrope_positions` / `generate` 在缺 `image_grid_thw` 时给出明确的 CHECK 信息，不要让 `c10::Error` 把进程带崩。
  4. 过渡手段：按 decode KV 限制每个 decode 的并发（BF16 TP1 约 4 条）。

### 4.2 P1：拆开 prefill 里非 GEMM 的约 690 ms

当前 prefill 侧 TTFT 约 1105 ms：同实例排队约 90 ms，prefill 本身约 1013 ms（最短 953 ms），对应约 4550 个未命中 token，约 223 µs/token。其中 GEMM 约 320 ms，非 GEMM 约 690 ms，后者是主战场。

- 抓 profile：`tools/decode_tp2_dev/bf16_ttft/profile_prefill.sh`。它用 `--online-profile-dir` 启动、预热到稳态，再对三个 prefill 的 HTTP 端口（28100–28102）调 `/start_profile`，发 measured prompt，调 `/stop_profile`，trace 写成 `*.pt.trace.json`，最后用 `parse_kineto_trace.py` 按 GEMM、注意力、GDN、conv1d、all-reduce、norm/rope、逐元素归类。这一步还没跑完过：上次预热到第 5 遍时为了 FP8 扫描停掉了。
- GDN prefill 走哪条：engine 环境里有 `XLLM_MATE_GDN_PREFILL=1`，`FLASHINFER_OPS_PATH=/opt/mate027_cache/mp31/cached_ops_jd_k8d` 里也有 TP2 头数（每 rank 8 个 q 头、24 个 v 头）的 `mate_gdn_prefill_full_strided_hq8_hv24_bf16` 和 `mate_gdn_prefill_full_varlen_strided_hq8_hv24_bf16`。所以大概率走的是 mate 融合 kernel（`mate_gated_delta_rule_prefill`），而不是 `xllm/core/kernels/musa/gdn_prefill.cpp` 里按 64 token 子块循环的 `chunk_gated_delta_rule` 参考实现。profile 时确认 kernel 名，再看 mate kernel 本身的耗时。
- 其余候选：16 层全注意力（每块都要读约 2 万 token 的前缀）、conv1d、norm 和逐元素、分段图之间的空隙。

### 4.3 P1：加大 prefill 分块（1024 → 2048 或 4096）

- launcher 参数 `--chunk`。TP2 prefill 的 KV 只用了一小部分，显存够。
- 一条请求从约 5 块降到 2–3 块，每块固定要付的成本变少：128 次 all-reduce 的延迟、分段图切换、注意力重复发起。
- 要观察的：捕图桶的数量和显存、尾块大小分布。

### 4.4 P2：路由改成按负载派发

服务现在是 `--load_balance_policy=RR`。改成按未完成 token 数派发，最多省掉约 90 ms 的排队。

### 4.5 需要拍板：BF16 权重用 FP8 GEMM

加载时用 `quantize_weight_to_block_fp8`（`xllm/core/layers/musa/linear_block_fp8.cpp`）把线性层量化成 block-FP8，GEMM 时间大约减半，估计省约 150 ms。代价是计算不再是纯 BF16，必须先确认能接受。

### 4.6 其他

- 可选：在 CP2 上用少一些 KV 块换更多线性状态槽，验证 3.5 的判断。
- decode 一路 TP2 不如两路 TP1（见第 1 节背景和附录 A），TP2 的 `max_seqs_per_batch` / `max_concurrent_requests` 目前是 16，C18 时会限流。

## 5. 试过、没用或排除掉的

| 尝试 | 结果 |
|---|---|
| 默认配置（显存 0.85，prefill 槽 96） | prefill 起不来：需要 98 个线性状态块，预算里只放得下 42 个 |
| prefill 显存 0.93、槽 16（CP2） | prefill rank 1 捕图时 OOM |
| decode 显存 0.93 | decode 捕图时 OOM |
| DP2CP3 | 不可行：`xllm/core/layers/musa/pcp_context.cpp` 的 `build_context` 要求 `cp_group->world_size() == kSupportedCpSize`（2），行切分写死为两段；而且总 KV 不会变多 |
| kv_split / DCP 按 token 切 KV | 只有 MLU 和 NPU 实现，MUSA 的 Qwen3.5 PCP 没有 |
| 怀疑 GEMM 慢 | 不慢：muDNN BF16 实测见下表，接近这张卡的上限；1024 token 分块在一个 TP2 rank 上 GEMM 只占 71.2 ms |
| all-reduce 从 MCCL 换成切片 custom AR | 见 3.3，2–3%，在噪声内 |

muDNN BF16 GEMM 实测（`torch.mm`，单卡，`tools/decode_tp2_dev/bf16_ttft/gemm_bench.py`）：

| 形状（TP2 每 rank） | N | K | M=1024 | M=4096 |
|---|---|---|---|---|
| GDN in_proj_qkvz | 8192 | 5120 | 338.6 TFLOPS | 394.0 TFLOPS |
| 注意力 QKV + gate | 7168 | 5120 | 353.0 TFLOPS | 377.4 TFLOPS |
| MLP gate_up | 17408 | 5120 | 360.6 TFLOPS | 389.8 TFLOPS |
| MLP down | 5120 | 8704 | 362.1 TFLOPS | 366.7 TFLOPS |
| GDN out / o_proj | 5120 | 3072 | 337.3 TFLOPS | 359.6 TFLOPS |
| 方阵 8192³ | | | 391.9 TFLOPS | |

## 6. 文件索引（`tools/decode_tp2_dev/bf16_ttft/`）

| 文件 | 作用 |
|---|---|
| `launch_pd_comparison.py` | PD 栈启停（etcd、xllm_service master/coordinator、3 个 prefill、decode）；`start` / `stop --run <dir>` |
| `run_bf16_c6.sh` | 第 2 节第三行：BF16、prefill DP3 TP2、两路 TP1 decode、C6 跑到稳态，结束自动停栈 |
| `run_bf16_c6_baseline_cp2.sh` | 第 2 节第一行：BF16 起点配置（prefill CP2） |
| `run_fp8_tp2_sweep.sh` | 附录 A：FP8、decode 一路 TP2、并发 6–18 扫描 |
| `bench_c6_until_steady.py` | C6 压测，跑到某一遍无新捕图为止 |
| `bench_until_steady.py` | 同上，可指定 `--dataset` 和 `--concurrency` |
| `profile_prefill.sh`、`profile_one_request.py`、`parse_kineto_trace.py` | 4.2 的 profile 流程 |
| `gemm_bench.py` | 第 5 节的 GEMM 测试 |
| `build_comparison_workload_stable.py` | 生成 c6/c12/c16/c18 workload（会覆盖 `workload_manifest.json`） |
| `gen_extra_concurrency_workloads.py` | 用同一套参数生成 c8/c10/c14 |
| `workload_manifest.json`、`workload_manifest_extra.json` | 各 workload 文件的 SHA256，用来核对重新生成的结果 |

压测依赖 `tools/musa_qwen3_8_vl_bench/vl_bench.py`；launcher 依赖 `tools/decode_tp2_dev/launch.py` 和 `config.json`。这些脚本里的路径都是绝对路径，仓库要放在 `/workspace/xllm-mt-main`。

## 7. 镜像（dev92 本地）

- 镜像：`xllm-musa:decode-tp2-bf16-ttft-20261028`（ID `sha256:904b5d30e130…`，59 GB）。另有本地别名 `registry.mthreads.com/presale/devtech/xllm:musa-decode-tp2-bf16-ttft-20261028`，没有 push 到 registry。
- 做法：从容器 `decode_TP2_dev` 直接 `docker commit`，基础镜像是 `decode-tp2-dev:base-20261027`。容器只挂载了只读的 `/data`，所以 `/workspace` 整体都在镜像里：源码工作区（未 commit 状态；源码部分与本分支 commit 2 一致，另有一些没有纳入分支的备份文件和旧工具）、build 目录、`/workspace/decode_tp2_dev_artifacts` 下的 binary、launcher、workload 和 run 日志。
- release 目录 `/opt/xllm/releases/musa-bf16-ttft-20261028/`：`bin/xllm`（即 `xllm.native_tp2_pf_tp_slices`）、本文档、`manifest.json`、`patches/`（完整差异、未跟踪文件包、3.3 的单独补丁）、`tools/`、`SHA256SUMS` 和 `verify.py`。
- 启动（参数同原容器）：

```bash
docker run -d --name <name> --privileged --runtime=mthreads --network=host \
  --shm-size=64g --ulimit memlock=-1:-1 --security-opt label=disable \
  -v /data:/data:ro xllm-musa:decode-tp2-bf16-ttft-20261028
docker exec -it <name> bash
python3 /opt/xllm/releases/musa-bf16-ttft-20261028/verify.py
```

launcher 会给每个 engine 单独设置可见 GPU；镜像里继承的 `MUSA_VISIBLE_DEVICES=6,7` 只影响直接在 shell 里起的进程。

## 8. 复现步骤

### 8.1 环境

二选一：

- **直接用第 7 节的镜像**：源码、binary、workload、服务二进制都已就位，可以跳过 8.2 和 8.3。
- **从本分支重建**：在 `decode-tp2-dev:base-20261027` 或同等的 MUSA 镜像里，把本分支检出到 `/workspace/xllm-mt-main`。`third_party` 子模块要和镜像里的一致（本分支没有改它们）。

两种方式都要求：

- 8 张卡全部空闲。launcher 启动前会检查，任何一张卡有占用就拒绝启动。
- 端口 57579/57580（etcd）、57888–57891（服务）、28000–30600 段没有被占。同一时间只能跑一套栈。
- 模型在 `/data/nfs_shared/models/Qwen3.8-27B`（BF16）和 `/data/nfs_shared/models/Qwen3.8-27B-FP8`。

### 8.2 编译

```bash
cd /workspace/xllm-mt-main
bash ./_build_cuda_graph_musa.sh
mkdir -p /workspace/decode_tp2_dev_artifacts/bin
cp build/lib.linux-x86_64-cpython-310/xllm/xllm /workspace/decode_tp2_dev_artifacts/bin/xllm.ttft_tp2_decode
```

- 只用 `_build_cuda_graph_musa.sh` 编译，它会经过 `scripts/ninja_guard`。直接调 `ninja -C build` 会重新跑 cmake，然后因为 `TORCH_MUSA_PYTHONPATH` 和 Jemalloc 失败。
- 重新编译出来的 SHA 不一定等于 `bf61b3af…`，不要求逐字节一致，以运行日志为准：prefill 日志里应出现 `[MUSA_TP_REDUCE] backend=custom_slices`。
- 后面的脚本用环境变量 `BIN=<binary 路径>` 指定 binary。不设时用镜像里的 `xllm.native_tp2_pf_tp_slices`。

### 8.3 Workload

镜像里已经有 c6–c18 七个文件，在 `/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv/`。要重新生成：

```bash
cd /workspace/xllm-mt-main/tools/decode_tp2_dev/bf16_ttft
python3 build_comparison_workload_stable.py         # c6/c12/c16/c18
python3 gen_extra_concurrency_workloads.py          # c8/c10/c14
sha256sum /workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv/workload_c*.json
```

把得到的 SHA256 和本目录下 `workload_manifest.json`、`workload_manifest_extra.json` 里记录的值对一下。生成器依赖 `tools/musa_qwen3_8_vl_bench/build_20k_prompts.py`、master tokenizer 和 `/workspace/xllm_vl_bench_ref/images` 下的 5 张图。

### 8.4 BF16 C6

```bash
cd /workspace/xllm-mt-main/tools/decode_tp2_dev/bf16_ttft
BIN=<binary 路径> ./run_bf16_c6.sh                # 第 2 节第三行
BIN=<binary 路径> ./run_bf16_c6_baseline_cp2.sh   # 第 2 节第一行（对照）
```

- 每个脚本大约 10–15 分钟：启动约 2.5 分钟，捕获轮约 4.5 分钟，之后每遍约 50 秒，直到某一遍无新捕图。结束时会自动停栈。
- 结果在 `<run>/bench/steady<N>.json`，标准输出里的 `SUMMARY` 行和最后一行 `{"steady": true, "pass": N}` 指出取哪一遍。
- 判定：`server_errors` 为 0，`prefill_new_captures` 和 `decode_new_captures` 都为 0，`kv_hit_rate` 约 0.79（起点配置约 0.41）。TTFT 应接近第 2 节的值，允许 ±10% 左右的波动。

### 8.5 FP8 decode TP2 扫描（附录 A）

```bash
cd /workspace/xllm-mt-main/tools/decode_tp2_dev/bf16_ttft
BIN=<binary 路径> ./run_fp8_tp2_sweep.sh
```

大约 20 分钟，C6 收敛最慢（上次第 12 遍才没有新捕图），后面几档 1–5 遍。

### 8.6 注意

- 4.1 的崩溃还在。BF16 两路 TP1 下，如果单个 decode 同时有超过 4 条请求（比如提高并发），可能看到 decode 进程退出、请求返回 503，日志里是 `Qwen3VLMPositionGenerator::generate` 上的 `c10::Error`。
- 停栈之后，日志末尾的 `Segfault encountered` 出现在 `Shutting down` 之后，是退出阶段的问题，不影响测量。
- 栈异常时手动清理：`python3 tools/decode_tp2_dev/bf16_ttft/launch_pd_comparison.py stop --run <run>`。

## 附录 A：FP8、decode 一路 TP2、并发 6–18

配置：FP8 模型 `/data/nfs_shared/models/Qwen3.8-27B-FP8`；prefill 为 GPU 0–5 上的 DP3 CP2（PCP 走 custom AR）；decode 为 GPU 6,7 上一个 TP2；显存 0.85，分块 1024，MTP K=3，前缀缓存开。binary `xllm.native_tp2_pf_tp_slices`。每档 60 条请求，workload 为 `workload_c{6,8,10,12,14,16,18}.json`。run 目录 `performance_comparison_20261027/sweep_fp8_tp2_c6_c18`。

| 并发 | QPS | TTFT 均值 | TTFT 稳态 | TTFT p90 | TPOT | MTP 接受率 | KV 命中 | 输出 tok/s | 第几遍稳态 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 2.135 | 851.5 | 818.8 | 1036.7 | 19.42 | 0.726 | 0.7928 | 213.5 | 12 |
| 8 | 2.520 | 889.3 | 814.4 | 1009.3 | 21.60 | 0.731 | 0.7928 | 252.0 | 2 |
| 10 | 2.916 | 948.2 | 820.6 | 1421.3 | 23.93 | 0.739 | 0.7928 | 291.6 | 3 |
| 12 | 3.154 | 970.1 | 805.2 | 1482.6 | 26.93 | 0.718 | 0.7959 | 315.4 | 4 |
| 14 | 3.251 | 1091.7 | 798.1 | 2191.6 | 29.68 | 0.722 | 0.7928 | 325.1 | 5 |
| 16 | 3.574 | 1158.4 | 872.4 | 2286.5 | 30.58 | 0.737 | 0.7990 | 357.4 | 1 |
| 18 | 3.767 | 1324.1 | 1027.0 | 2426.4 | 31.74 | 0.734 | 0.8021 | 376.7 | 2 |

TTFT、TPOT 单位为 ms；错误均为 0。C18 时 decode 受 `max_seqs_per_batch=16`、`max_concurrent_requests=16` 限制；C16、C18 一波只有约 16 秒，样本较少。
