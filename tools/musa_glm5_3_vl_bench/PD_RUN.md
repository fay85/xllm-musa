# GLM-5.3-Flash 双机 P/D 运行

两端进入各自的 `albert_xllm` 容器，工作目录均为 `/data/nfs_shared/albert/Projects/xllm-musa/tools/musa_glm5_3_vl_bench`。服务运行期间两台各占 8 张 GPU。

| 执行位置 | 命令 | 作用 |
|---|---|---|
| sj7 | `bash start_prefill.sh start` | 启动专用 etcd、router、coordinator 和 CP8/EP8 Prefill；P 产生第 1 个 token |
| sj8 | `bash start_decode.sh start` | 启动 Attention TP8、CP1 的 Decode（框架 EP1，routed experts 在 8 卡上分片）；D 从 sj7 用 Mooncake RDMA PULL 读取状态并续算 |
| 任一容器 | `curl -fsS http://10.121.34.85:28888/readyz` | 确认路由已发现可接单的 P/D 对；新注册后允许几秒收敛 |
| sj7 | `bash bench.sh pd 1 --out pd_logs/result.json` | 同一请求在两端的日志和客户端 20,000/16 指标；默认一次、并发 1、纯文本、无预热 |
| 任一容器 | `tail -F pd_logs/{prefill,decode}/rank_0.log` | 同时查看两台服务器的实时日志，NFS 可有短暂延迟 |

两个实例已启动时也可直接执行 `bash bench.sh`。显式 `pd` 可固定路由入口；`bash bench.sh standalone 1` 使用原合部客户端配置。原始 `bash start_service.sh start` 仍为合部启动命令。两个角色均使用共享树下的 `build/lib.linux-x86_64-cpython-310/xllm/xllm`。两端必须指向同一份编译结果。

双端停服在 sj7 或 sj8 的容器中执行一次 `bash stop_service.sh pd`。脚本先等 sj8 清完 D，再等 sj7 清完 P 和本实验的控制面。两边通过 NFS 按启动实例标识传递停止请求，由各容器核实本地进程后终止。成功会输出 `decode stopped` 和 `prefill stopped`。若一端不可达或监听进程已退出，命令会超时并报告角色；可在该端执行 `bash start_decode.sh stop` 或 `bash start_prefill.sh stop`。只停合部用 `bash stop_service.sh standalone`。每个角色也支持在自己的机器上执行 `status`。重启按上表顺序。每次 start 会重启该角色并覆盖该角色的旧 rank 日志，需保留的结果先复制到独立目录。控制面日志在 `pd_logs/control/`。独立实验端口为 etcd 22379，router HTTP/RPC 28888/28889，coordinator 28890/28891，P/D 引擎首端口 19100/19200，传输端口分别为 26100～26107、26200～26207。推理请求统一发往 router。

## 编译与传输

在 sj7 容器的共享仓库根目录执行 `ninja -C build/cmake.linux-x86_64-cpython-310 -j4 xllm`。sj8 直接使用该产物。源码改动在 `xllm/core/distributed_runtime/llm_engine.cpp`、`xllm/core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.cpp`、`xllm/core/runtime/worker_impl.cpp`。当前 GLM 的 FP8 KV 分离使用 PULL。P 发布全部 8 个 KDA head-owner rank，MoE 并行组与各自实例绑定。D 依据 manifest 分配并拉取本地 block 和 sequence slot，包含 MLA KEY、压缩 INDEX、KPOOL_TAIL、CONV 和 SSM。两机并行组独立。NFS 用来共享程序、模型、报告和日志。

## 判读 bench

按同一个 `vl-…-c0` 查找 `Prefill Decode allocation`、P 侧 `generated_tokens: 1`、D 侧 `Decode KV restore` 和 D 侧 `generated_tokens: 16`。16 是累计值，包含 P 产生的首 token。跨机搬运可在 D 的逐 rank 日志中搜索 `[Mooncake][PDTransfer]` 中的非零 regions 和 success。readyz 若暂时返回 503，查看 `control/coordinator.log` 中的邻接关系和两端的首次致命错误。bench 的 TPOT 以首 token 到最后 token 的时间除以 15，包含中间的交接等待。

## 2026-11-20 实测记录

两台机器原有的合部实例已切换为 P/D。用两台容器的 `/proc/<rank0-pid>/exe` 校验，运行中的 P、D 和共享二进制的 SHA256 都是 `1f6cc48a3565b76237eb56efa49bcee777c5dadd0db523db0450a4049425b2a8`。此二进制已移除诊断用的状态复制和打分代码。可用本目录的 `glm53_pd_engine.patch` 检查三个引擎源文件的完整改动，共享工作树已应用。

目标拓扑在最终二进制上连续执行三次独立 `bench.sh`，每次 1 请求、并发 1、20,000 个实际 prompt token、16 个实际生成 token、0 张图。三次输出文本一致，日志分别见 `pd_logs/final-{1,2,3}.{json,log}`。下列时延单位为 ms。

| 请求 | TTFT | D 侧 KV restore | 客户端 TPOT | 客户端端到端 |
|---|---:|---:|---:|---:|
| 服务重启后第 1 条 | 11973.8 | 10069.8 | 3295.41 | 61405.0 |
| 第 2 条 | 826.7 | 25.55 | 61.13 | 1743.7 |
| 第 3 条 | 1218.9 | 25.83 | 57.29 | 2078.2 |

第 3 条的 request ID 是 `vl-1795186225399016075-c0`。P 完成 1 个 token，D 日志记录 PULL 后入队和累计 16 个 token、finish_reason=length。首次请求有明显冷启动开销，表中分别列出冷、热数据。两端 GPU 活动采样在 `pd_logs/gpu-sj7.log` 和 `pd_logs/gpu-sj8.log`。

**交接数值检查。** 用临时诊断构建在 P 的最后一次 prompt forward 后、D 的第一次 decode forward 前，按逻辑顺序取请求有效的数据做 SHA256。每个 rank 覆盖 34 个 CONV、34 个 SSM、11 个 KEY、11 个 INDEX、11 个 KPOOL_TAIL。8 个 rank 共 808 项逐项一致；诊断记录了 P 的首个 token ID 和 D 的后续 15 个 ID。另将同一份 P 状态载入独立的单机 TP8 进程，5 次回放都校验了载入的状态，第 4 次的 16 个 ID 与双机序列逐个相同。其余回放有贪心输出波动。直接重复原合部推理也观察到波动，合部与 P 使用相同的输入 token 摘要；其中一次对照里，层 0～3 的有效缓存逐字节相同，层 4 开始出现差异。这些实验验证了状态交接，未确定导致模型重复执行数值波动的具体算子。逐次 token ID 和摘要计数在 `pd_logs/audit/handoff-check.json`，同拓扑 TP8→TP8 实验在 `pd_logs/audit/tp8-pd.json`。耗时数据取自最终服务，诊断回放的耗时另存。

服务重启会覆盖 `pd_logs/prefill`、`pd_logs/decode` 当前运行日志。上述三次 bench 的独立结果文件已保留。

## 2026-11-21 并行组与热态 TPOT

| 角色 | 实际拓扑 | 单请求的工作 |
|---|---|---|
| sj7 Prefill | DP1，Attention 的 CP8、TP1；KDA 按 head 分到 8 个 rank；路由专家 EP8 | 8 卡处理 20,000 个输入 token，生成第 1 个输出 |
| sj8 Decode | DP1、CP1；Attention 和 shared expert 用 TP8；MegaMoE 在同 8 卡上分片 288 个 routed experts，每卡 36 个完整专家 | RDMA PULL 恢复一次，生成余下 15 个 token |

读日志时，P 的 dist_manager 会打印 tp_size=8，实际 attention 子组见 collective_communicator 的 tp_size=1、pcp_size=8。D 的命令行 ep_size=1，MoE communicator 复用 TP 组，模型层实际打印 GLM MegaMoE EP8。D 的 custom TP all-reduce 实际输入为 BF16 [1,4096]，8192 字节。启动与运行证据在 `pd_logs/profile-validation/before-{prefill,decode}.log`。

热请求 `vl-1795258891059982647-c0` 的 D 侧日志记录 TPOT 55.8 ms，KV restore 24.69 ms，均摊到 15 个 token 间隔约 1.65 ms。D 入队至最后输出约 808 ms。首次建链的秒级数据单列，比较 TPOT 时用相同输入、两次以上预热、关闭采样并连续测多次。

## 双端 trace

两个 albert_xllm 容器都使用本文顶部的共享工作目录。以下配置在服务启动时生效。先从任一端停止原实例，再在两机分别启动。

```bash
# sj7 或 sj8
bash stop_service.sh pd
# sj7
GLM5_TORCH_PROFILE=1 bash start_prefill.sh start
# sj8
GLM5_TORCH_PROFILE=1 bash start_decode.sh start
# 任一端确认 ready 后采样
curl -fsS http://10.121.34.85:28888/readyz
bash profile_bench.sh
```

默认并发 1、20,000 输入、16 输出、纯文本、两次预热。脚本沿用原 profile_bench.sh 的 worker StartProfile/StopProfile RPC，从 P、D 各自的 rank 日志获取本次启动的地址。预热后同时开启两个角色的 rank 0，请求经 router 流式完成后停止 profiler，并核对服务返回的 token 数以及每个文件中的 GPU kernel 事件。用 `bash profile_bench.sh --ranks 0,1` 可检查 rank 差异；`--ranks all` 采两端各 8 个 rank。两端启动时分别使用默认角色 trace 目录；自定义目录时用 `--prefill-trace-dir` 和 `--decode-trace-dir` 指定两条共享路径。

输出在 `pd_logs/profiles/<时间戳>/`，包含请求 ID、预热和采样时延、按角色分开的 Chrome trace 以及 kernel 排名 summary.json。用 Perfetto 分别打开 `prefill/*.pt.trace.json` 和 `decode/*.pt.trace.json`。同一 ID 可到各端 rank_0.log 查 P 首 token、D KV restore 和累计生成数；比较不同主机时间戳时需要考虑时钟偏差。采样完需要切回正常服务时执行双端停服，再各自按本文第一张表启动。

### 本次样本和优化顺序

样本目录 `pd_logs/profiles/20261121-191356-1795259636500378952/`，请求 `pd-prof-1795259644320954555-profile`。P rank 0 有 2,348 个 GPU kernel，D rank 0 有 34,365 个，平均每个后续 token 有 2,291 次 kernel 执行。D 的首末 kernel 跨度 1,272 ms，将同一 rank 上所有 kernel 的执行区间取并集约 496 ms。其余间隔可能包含 CPU 发射、同步、等其他 rank、拷贝和 profiler 开销。该轮热请求 TPOT 为 54.81 ms，采样时升至 86.72 ms。耗时结论限于本样本的 rank 0。

1. 对并发 1 的延迟，优先保留 D 的 TP8 + 路由专家 EP8，对比 D 单独开启 MUSA decode graph 前后的热态 TPOT，重点看 kernel 间隙、graph replay 和输出一致性。该早期实验使用 eager。后续 D 单独启用 graph 的实测见下文。
2. TP 的 cross_device_reduce_1stage 实测 1,290 次，合计 50.71 ms，约每后续 token 86 次、3.38 ms。MegaMoE stage1/2 各执行 630 次，两者合计约 94.08 ms。先看多 rank 是否有拖尾，再决定是否优化小消息归约或融合 kernel。以上均为采样的单 rank GPU 时间。
3. 如果目标是试另一种并行布局，可研究 Attention TP4 + routed-expert EP8，分别核对两套并行组的映射、显存、shared expert 权重和 P→D 的 KV/KDA head ownership，再用相同 20k/16 比较。不宜只改 ep_size 或把卡数减半。DP 副本适合提高多请求吞吐；同一条自回归序列每步只有一个新 token，照搬 P 的 CP8 也缺少可切的新 token。若目标是一次验证多个后续 token，可另行验证 MTP 推测解码的 MUSA 模型接入、状态回滚与接受率。

停服从 sj7、sj8 各发起并完成过一次。验证记录在 `pd_logs/profile-validation/stop.log`、`stop-from-sj8.log` 和 `stopped-{prefill,decode}.log`；原 eager 服务恢复后的 ready 与 bench 在同目录。

## 20:35 和 22:01 的 rank 0 trace 对照

两个样本在 `pd_logs/profiles/20261121-203521-1795264521461605463/` 和 `pd_logs/profiles/20261121-220126-1795269686362687107/`。按每个 GPU kernel 的起止时间扫描同一设备上的重叠区间，并用 correlation 关联其 CPU launch。下表的发射提前量是 CPU launch 返回到对应 GPU kernel 开始的中位时间。

| 角色 | 两次发射提前量 | 两次至少两个 GPU kernel 同时执行的总时间 | 两次 GPU kernel 数 |
|---|---|---|---|
| P | 308 ms、312 ms | 5.71 ms、6.08 ms | 各 2,348 |
| D | 57 µs、56 µs | 均为 0 ms | 各 34,365 |

P 的实际交叠主要发生在 stream 5 的 MCCL AllGather 和 stream 17 的 GEMM 之间。D 每次有 34,065 个 kernel 在同一条计算流 14 上，15 步各生成一个 token。两次 D 分别有 28,487、29,110 次关联到的 CPU launch 在同流前一个 GPU kernel 执行结束前已经返回。两次各有 60 个显式 musaStreamSynchronize，CPU 总时长分别为 16.05 ms、15.39 ms。两组样本的 kernel 执行方式相同。

当前 eager VLM worker 常规 step 使用 ForwardSyncPolicy::LEGACY，step 返回前会同步 stream。GLM 的 RoutedMoe 对 Decode 关闭 shared-expert 侧流分支（use_shared_expert_overlap 要求 is_prefill）。这两处与驱动 MUSA_LAUNCH_BLOCKING 独立。现有 rank 0 trace 足以验证上述重叠情况，无法单凭它给 GPU 的每个空隙归因。将环境变量从 0 改为 unset 不会使同一流上的 kernel 重叠。

## Decode Graph 优化与复现（2026-11-21）

共享分支基线 11022b7c5。用同一份 xllm 启动两端，sj7 保持 CP8/EP8 eager，sj8 在原有 TP8、单并发拓扑上默认启用 Decode Graph。两个实例都保留在线抓 trace 的能力。性能计时期间 profiler 停止采样。

| 同一组 20,000/16、无图、并发 1 请求 | 热态 TPOT 中位数 | 8 条范围 | rank 0 每 15 个续算 token 的 stream synchronize |
|---|---:|---:|---:|
| 原 eager | 57.51 ms | 55.42～59.73 ms | 60 |
| graph，回读 KDA 槽位 | 39.81 ms | 39.63～40.20 ms | 45 |
| graph，复用槽位主机副本 | 39.32 ms | 39.25～39.60 ms | 30 |

每组先预热 2 次，再轮换 3 个相同的纯文本 prompt 做 8 次串行请求。客户端以首末文本时间差除以 15；三个配置的 KV restore 中位数分别为 24.85、24.95、24.74 ms。热态中位数比 eager 下降 31.6%，相应生成速率提升 1.46 倍。优化后的冷请求包含首次建链、缓存与 graph 捕获，单次客户端 TPOT 曾达 1032 ms；最终样本的 capture begin 到 end 约 1.87 s。单独运行标准 bench 也在 `pd_logs/decode-opt/bench-final.json` 记录了 39.59 ms TPOT、0 个服务错误。

**实现。** MUSA graph capture 从本实例已分配的各层卷积状态缓存取得最小可用槽位容量。此前专用 D 首条请求直接 capture，依赖 eager forward 填充的容量为 0。capture 校验槽位及实际 query offsets，标准 Decode replay 的每行一个 token、query offsets 固定。replay 使用与输入槽位张量一同打包的主机槽位编号检查范围和唯一性，并在设备上直接更新捕获的 int64 索引缓冲区，省掉两个逐步同步回读；没有匹配主机副本时保留槽位回读。Prefill 和 standalone eager 的校验路径照常执行。运行时改动在 `xllm/core/runtime/musa_graph_executor_impl.cpp`、`xllm/models/llm/musa/glm5_next.h`。

**Trace。** 最终 trace 在 `pd_logs/profiles/20261121-225336-1795272816436855509/`。D rank 0 记录 15 次 musaGraphLaunch，普通 musaLaunchKernel / musaLaunchKernelExC 合计从 32,595 次变成 360 次。Kineto 对 graph 节点的展示与 eager 不同，390 个可见 GPU kernel 不代表模型只执行了 390 个算子。剩余每步两次同步对应 worker 等 GPU 完成计算和输出回拷后主机消费 token。采样的最后版本前者约 35 ms/步，后者约 0.05～0.21 ms/步。graph 将大量设备工作推迟到一次发射后的等待中；不能拿同步 API 累计时长当作可直接扣除的模型延迟。

**输出和状态验证。** 除性能组外，还用 logprobs=true 直接获取端点返回的 token_id，3 个 prompt、8 个 graph 请求与同二进制 eager 的重复输出集比较，8/8 在同 prompt 下逐个 16 个 ID 完整匹配。eager 自身的重复请求存在不同贪心续写。3 个现有 KDA 单测通过，包括非法槽位和序列边界、cache slot 重置，以及切换输入和状态槽位的 graph replay 与 eager 状态对比。测量范围为纯文本、并发 1；本次未测多请求并发和图像输入。

在两个容器共享仓库根目录编译 `ninja -C build/cmake.linux-x86_64-cpython-310 -j4 xllm`。之后回到本文目录，按下列命令运行，常规延迟测量和采样分开执行。

```bash
# sj7
GLM5_TORCH_PROFILE=1 bash start_prefill.sh start
# sj8
GLM5_TORCH_PROFILE=1 bash start_decode.sh start
# sj7，等待 router 完成 P/D 配对
GLM5_WARM=2 bash bench.sh pd 1
bash profile_bench.sh
```

要复测 eager，在 sj8 启动命令前加 `GLM5_DECODE_GRAPH=false`。连续单独重启 D 时，coordinator 曾报告旧边 retiring、当前可用边为零并返回 503；此次通过本实验双端停服并依次重启两端恢复。根据 D 已关联 sj7 的日志及真实请求确认就绪。

`pd_logs/decode-opt/comparison.json` 收录三组统计。可在本文目录运行 `python3 pd_logs/decode-opt/compare.py eager graph2 optimized` 和 `python3 pd_logs/decode-opt/compare_ids.py` 复核时延和 ID。单测结果在同目录的 `kda-tests.log`。

## 39 ms 到 10–20 ms 的 Decode 优化判断（2026-11-21）

当前 39 ms 的热态 TPOT 不是主机同步阻塞造成的。最终 Decode Graph trace 中，每步 `musaGraphLaunch` 约 0.3–0.7 ms，随后等待 GPU 完成约 35.2–35.9 ms。剩余一次短同步约 0.05–0.21 ms，用于输出消费。继续删除这类短同步不能把 39 ms 降到 10–20 ms。

eager trace 可用来定位 Graph 内的计算组成，rank 0 的 15 步累计 GPU kernel 时间包括：

- MegaMoE stage2 约 58 ms，约 3.9 ms/步
- TP cross-device reduce 约 49–55 ms，约 3.3–3.7 ms/步
- MegaMoE stage1 约 36 ms，约 2.4 ms/步
- mHC pre/post 约 37 ms，约 2.5 ms/步
- KDA prepare/recurrence、MLA、投影、量化和归一化还占用其余时间

这些是单 rank 的 kernel 累计时间，不能直接相加当作端到端时间。它们说明 10–20 ms 需要同时缩短主关键路径上的 MoE、TP 通信和多层小 kernel 序列。

### 优先级

1. **先做 Decode Graph 的 kernel 级 trace。** 用 eager trace 识别每个 decode layer 的关键路径，用 Graph trace 只测 capture/replay 总时延。不能把 Graph trace 中的 390 个可见 kernel 误认为模型只执行了 390 个 kernel。
2. **优化 TP8 的小消息通信。** 当前 D 是 Attention TP8、shared expert TP8，Moe communicator 复用 TP group。需要在相同输出和显存约束下比较 TP4/TP8 的 attention 及 shared expert 路径，并确认 routed expert 仍然保持 EP8。只改 `ep_size` 不会得到这个布局，当前启动脚本也不支持任意 TP/EP 组合。
3. **融合 Decode 的 MoE stage1、stage2 和通信边界。** 现在每个 token 要经过 gate、dispatch、stage1、专家计算、stage2、combine 和 TP reduce。应以 MATE/现有 MegaMoE decode kernel 为基线，检查是否能把 token=1 的 staging、scale 和小 tensor copy 合并到 stage1/stage2，避免重复读写。需要覆盖 8 rank 的 collective 顺序和非拥有专家的 sentinel 行。
4. **减少 mHC 和 KDA 的小 kernel 数量。** Graph 已经消除了 CPU launch 空隙，剩下的是设备侧 kernel 数量和 kernel 间依赖。重点是把 decode-only 的 cast、rmsnorm、index/select、KDA prepare/recurrence 与相邻投影融合，避免用 host scalar 或 D2H metadata 重新建立依赖。
5. **最后评估 MTP/Speculative Decode。** 当前服务没有启用 draft/MTP。MTP 的收益来自一次验证多个候选 token，适合把单 token 39 ms 摊到多个已接受 token。必须先验证 GLM5 MUSA 的 draft 权重、状态回滚、KDA/MLA recurrent state 和 PD 首 token 交接，不能只添加一个开关。

### 目标可行性

- 20–25 ms 可能通过 Decode Graph、TP 通信优化和 MoE 小 token kernel 融合实现，需要真实 A/B。
- 10–20 ms 不能由当前的 stop/synchronize 修改单独实现。更接近 10 ms 的方案通常需要 TP/EP 重新布局、专用 token=1 MegaMoE kernel，或 MTP 提高每次验证的有效输出 token 数。
- 当前 P/D 交接的 KV restore 约 25 ms 只发生一次，不是稳定 TPOT 的主要来源。

### 每次 A/B 的固定验收

使用同一份 20,000 输入、16 输出、并发 1，至少两次预热后测 8 次。必须同时记录：

- 客户端 TPOT、D 侧每步日志和 KV restore
- 8 个 rank 的 Decode trace，不能只看 rank 0
- 16 个 token 的 token ID 或 logprob
- peak 显存、MCCL/TP collective 错误和 P/D ready 状态
- 首次 Graph Capture 延迟与稳态延迟分开

当前已验证的结果和 trace 位于 `pd_logs/decode-opt/comparison.json` 及 `pd_logs/profiles/20261121-225336-1795272816436855509/`。

## SGLang Decode 对照

SGLang 的 decode graph、TBO、DSA/MLA、MegaMoE 和并行策略对照，以及针对当前 xLLM 的阶段化借鉴方案，见本目录的 SGLANG_DECODE_COMPARISON.md。报告同时记录了源码文件、行段、当前 xLLM 39 ms trace 的对应关系和 C=1/C>1 的验证矩阵。


## Decode 逐项优化记录（2026-11-21 23:46–23:59）

### 1. Scheduler overlap，仅 Decode 开启

`pd_role.sh` 为 Decode 增加 `GLM5_DECODE_SCHEDULE_OVERLAP`，默认 `true`；Prefill 明确导出 `GLM5_ENABLE_SCHEDULE_OVERLAP=false`。`start_service.sh` 同时打印并传递 `--enable_schedule_overlap`，`MUSA_LAUNCH_BLOCKING=1` 时自动关闭该选项。sj8 日志确认 `enable_schedule_overlap: 1`，sj7 日志确认 `schedule_overlap=false`。

相同 20k/16、并发 1、预热 2 次的单次实测为 TPOT **37.68 ms**，D 侧累计 16 token，服务错误为 0。之前 Decode Graph、无 overlap 的热态中位数约 39.32 ms。结果保留该 Decode-only 配置。

### 2. MLA absorb-query 固定权重布局

`Glm5NextMLAAttentionImpl::materialize_weights()` 现在直接保存 `w_k` 为 `[head, 256, 512]`。`glm5_next_mla_attention()` 直接使用该布局，移除了每个 token、每个 MLA 层重复执行的 `w_k.transpose(1, 2).contiguous()`。Prefill 和 Decode 共用这个数学等价的权重布局。

新二进制在共享目录编译成功。Decode 重启后相同口径得到 TPOT **37.59 ms**，单次样本的差异在测量噪声内，未宣称有稳定收益。Prefill 用新二进制重启并测试，TTFT **826.5 ms**，D 侧 TPOT **37.42 ms**，没有看到 Prefill 退化。测试输出保留在 `pd_logs/decode-opt/absorb-weight-cache*.{log,json}`。

### 3. EP A/B 结论

尝试 `GLM5_DECODE_EP_SIZE=8`。服务在模型构造阶段由 `routed_moe.cpp:483` 直接终止，错误为 `RoutedMoe EP1 requires XLLM_GLM5_PREFILL_EP=1`。该配置不支持，已恢复 Decode 为 EP1。现有 Decode 的 MegaMoE 内部日志仍显示每卡 36 个完整专家、内部 EP8，因此不能用 `ep_size` 旗标代替该内部布局。

### 数学请求

在恢复后的 P/D 服务上，经 router 发起“计算 37*48，并给出简要计算过程。”，请求 ID 为 `chatcmpl-1663645912712605800-bf6bHeMtB9uVXtbAh56Na8`。返回内容给出 `37 × 48 = 37 × 50 − 37 × 2 = 1850 − 74 = 1776`，并用加法形式复核为 1776。D 日志记录 `generated_tokens: 88`、`finish_reason: stop`，无服务错误。原始响应保存在 `pd_logs/decode-opt/math-request.json`。

本轮仅确认了 overlap 的稳定小幅收益，固定权重布局尚未形成可重复的 TPOT 降幅。后续应优先按 `SGLANG_DECODE_COMPARISON.md` 中的 kernel 级 trace、TP 小消息通信和 token=1 MegaMoE 融合顺序继续做 A/B。



### 最终重启后的数学请求

在 Prefill/Decode 都使用新二进制并重新启动后，又发送了“请计算 12345 + 67890，只给出最终结果和一句简要说明。”。请求 ID 为 `chatcmpl-1663645912712605800-4ok7sHKTSThnpUsZavtpCP`，服务返回 **80235**，并说明是两数直接相加。完整响应保存在 `pd_logs/decode-opt/math-request-final-complete.json`。此前同 prompt 的 128-token 请求被长度上限截断，不能作为最终准确性样本。

