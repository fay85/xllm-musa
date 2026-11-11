# GLM-5.3-Flash TP8 trace 分析（2026-11-11）

分析对象：`/data/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/xllm_profile/baseline_tp8_1111` 的全部 8 个 rank。用户提供负载：并发 1、输入 2048、输出 8、TP8。本报告只做分析，没有修改推理实现、重启服务或执行压测。

## 结论与优先级

**核心算子已经部分接入 MATE，但不能说“所有应该使用 MATE 的地方都用了”。最优先的工作是 mHC finish、IndexPool prefill 和去除逐层 host 同步；MegaMoE 与 PD 应在这些问题之后做有约束的实验。**

1. KDA 的 prepare/recurrence 全部使用 MATE，34 层 × 8 步，各 272 次；核函数合计仅 18.479 ms。不能把 KDA 上层函数约 2.3 秒的 CPU inclusive 时间说成 KDA 算力瓶颈。
2. DSA/MLA 核心使用 MATE TileLang BF16 sparse MLA，11 层 × 8 步 = 88 次。**decode 也用 `dsa_prefill_kernel`**，未走已有的 FP8 scheduled decode 接口。名称虽叫 prefill，当前 BF16 路径确实允许单 query；不是单凭名称就能判定错误。
3. MoE prefill 的 84 次 ragged FP8 GEMM 来自 MATE MUBIN；decode 的 294 组专家前向走 xLLM 自定义 FP8 GEMV。**没有 MegaMoE dispatch/combine 或融合专家管线证据**。普通 MATE MoE GEMM 与 MegaMoE 是两件事。
4. mHC prenorm GEMM 用了 MATE，后半段却是自定义 `mhc_pre_finish_kernel`：720 次合计 **462.213 ms**，其中 decode 每次约 **429 µs**。当前安装的 MATE 已有 `mhc_pre_big_fuse`，值得优先导出并对比。
5. prefill 的 IndexPool `update_pools` 11 次合计 **182.327 ms**，每次约 **16.575 ms**。源码每个 request 只发一个 128-thread CTA，并串行遍历该请求所有 token；并发 1 时这是明确的并行度问题。
6. 每 rank 有约 1400 次 device→host 小拷贝和 1400 次 stream sync；rank 0 在 prefill/decode 内 `aten::item` 分别有 413/2513 次。大量层内数据检查强制 host 等 GPU，阻断异步流水。但等待时间包含必要设备工作，**不能全算可回收时间**。
7. 当前单请求 P→D 是因果串行的。PD 隔离主要帮助持续多请求吞吐和避免 prefill 干扰 decode，不能直接让此 C=1、OSL=8 请求更快；迁移开销可能反而增加延迟。

## 统计口径与可信边界

- Chrome trace 的 `ts/dur` 按 **微秒**解释，表格统一换算 ms；`displayTimeUnit=ms` 是界面显示偏好，不是将原始数值当 ms 的理由。
- `kernel_sum` 是本 rank 所有 kernel duration 相加；`gpu_union` 是 kernel、memcpy、memset 时间区间取并集。多 stream 重叠不重复计算。8 张卡的时长不能相加当作请求延迟。
- phase 用 runtime/driver correlation 对应的 **host launch 时间**归属首步 prefill 或后续七步 decode，避免设备排队使上一阶段工作错归下一阶段。未找到 correlation 时才用事件自身时间；边界外事件另列。
- `python_function` 中的 xLLM 调用栈是后处理合成事件，`cpp_callstack_synthesized=true`。注入脚本从 ATen 事件及采样调用栈拼接区间，不是精确 C++ 入口/出口计时。因此函数 inclusive 归属和层标签仅辅助定位；GPU 实测 kernel 是主证据。
- 观察到每 rank 8 次 worker step、360 个 decoder layer slices、34 次 conv prefill 和 238 次 conv update，以及 24 个 vision attention。与 **一次图文 prefill + 七次 decode** 一致。2048 来自用户负载和采集脚本，精确网络 TTFT/E2E 应以同一次请求日志/客户端结果复核。
- 采集包含 C++ unwind preload 和大量 profiler 事件；host 空洞、rank 到达偏差可能受采集放大。所有建议必须用关闭 profiler 的相同负载验证收益。
- 此 trace 没有权威显存峰值/分配快照、硬件带宽计数器、历史二进制哈希和动态库 maps；不据此宣称 HBM 带宽饱和、内存泄漏或数值正确。

## 阶段与 rank 统计

完整数值见 [ranks.csv](ranks.csv)、[phases.csv](phases.csv)；[events.csv](events.csv) 保存全部分阶段 kernel/runtime/CPU 事件聚合。

![rank0 kernel breakdown](kernel_breakdown.png)

所有卡为 **MTT S5000，MP31，56 SM，约 79.92 GiB**；trace 的 runtime/driver 为 50100、MUPTI 14。

|rank|GPU 活动首尾跨度 ms|GPU 活动并集 ms|kernel 总和 ms|prefill host 合成区间 ms|decode host 平均 ms|
|---:|---:|---:|---:|---:|---:|
|0|3254.615|1950.965|1953.634|1142.530|299.954|
|1|3231.799|2033.862|2036.636|1119.266|299.820|
|2|3205.385|1897.456|1898.688|1089.718|299.812|
|3|3211.521|1924.831|1924.386|1096.187|299.908|
|4|3225.063|1882.074|1883.463|1109.691|299.804|
|5|3233.572|1937.407|1940.388|1117.792|299.807|
|6|3230.495|1989.567|1989.003|1118.478|299.930|
|7|3231.221|1951.842|1950.992|1115.853|299.969|

rank 0：prefill GPU phase 跨度 1143.870 ms、活动并集 942.353 ms（82.4%）；decode 七步跨度 2109.200 ms、并集 1007.947 ms（47.8%）。整个 GPU span 内约 1303.650 ms 没有记录到设备 activity。这里的“有 activity”包括通信等待，并非有效计算利用率。

decode 稳态步长跨 rank 非常接近（299.804–299.969 ms），prefill 起始跨度差更大；不能据此认定某个 rank 的计算卡更慢。8 个文件共享 baseTimeNanoseconds，但仍需同步事件核对跨设备时钟偏差，尤其不能仅凭某 rank allreduce 时长给其定责。

## 耗时占比（rank 0）

以下百分比以**相应阶段 kernel duration 总和**为分母，不是 E2E 占比。CPU、GPU、通信等待三个视角有重叠，不能把表间数值相加。

|GPU 项目|prefill 次数 / ms|prefill kernel 占比|decode 次数 / ms（7 步）|decode kernel 占比|
|---|---:|---:|---:|---:|
|custom cross-device allreduce|145 / 209.381|22.03%|602 / 500.722|49.95%|
|mHC pre finish|90 / 191.999|20.20%|630 / 270.213|26.96%|
|IndexPool update|11 / 182.327|19.18%|77 / 0.955|0.10%|
|mHC post|90 / 90.128|9.48%|630 / 3.963|0.40%|
|ragged FP8 expert GEMM|84 / 58.814|6.19%|0|0|
|MCCL allreduce|42 / 32.161|3.38%|28 / 29.616|2.95%|
|DSA BF16 core|11 / 21.436|2.25%|77 / 8.709|0.87%|
|MATE mHC prenorm|90 / 16.847|1.77%|630 / 10.770|1.07%|
|MATE KDA prepare + recurrence|68 / 8.701|0.92%|476 / 9.778|0.98%|
|全部 kernel|4616 / 950.596|100%|25150 / 1002.372|100%|

全程 kernel 总和 1953.634 ms：custom AR 710.103 ms（36.35%）、mHC finish 462.213 ms（23.66%）、IndexPool update 183.282 ms（9.38%）、mHC post 94.091 ms（4.82%）。仅这四项就占约 74.2%。

decode 专家 GEMV 的 W13+SiLU 为 8.420 ms，W2 为 5.670 ms，route combine 为 1.507 ms，合计 **15.597 ms / 7 步 ≈ 2.23 ms/步**。将它单独无限加速也不可能解释或消除约 300 ms/步的大部分延迟。MoE 改造还可能影响通信和 host launch，收益需端到端衡量。

CPU 视角：rank 0 phase 内 stream sync prefill 215 次 / 786.900 ms，decode 1204 次 / 856.312 ms；`aten::item` prefill 413 次 / 781.126 ms，decode 2513 次 / 864.696 ms。这些嵌套于前向及等待区间，不是独立新增耗时。部分最长 item 落在 DSA 后的下一 KDA 层，所以会把此前 DSA/通信/mHC 排队时间记在 KDA 的 host 函数里。

## MATE 使用审计与接口约束

|模块|本次执行证据|结论与下一步|
|---|---|---|
|KDA|`mate::flat::kda::ChunkKdaPrepareKernel`、`ChunkKdaRecurrenceKernel` 各 272|核心已用 MATE；decode 同样走 chunk=16 prepare/recurrence；优先解决外围同步，再评估专用单 token recurrent 路径|
|DSA sparse core|88 × `dsa_prefill_kernel`，CPU `sparse_mla_bf16` 88|MATE BF16 TileLang，prefill/decode 共用；未使用 FP8 scheduled decode|
|DSA indexer score|77 × `Mp31Fp8PagedMqaLogits`，以及 metadata/clean logits|decode 已用 MATE FP8 paged MQA score；prefill 恰好 2048，不超过 topk=2048，源码跳过 score/topk，属于合理分支|
|IndexPool cache 更新|`update_pools`、`quantize_query`、`expand_pools`|自定义 kernel，非完整 MATE DSA pipeline；pool 更新是 prefill 热点|
|MoE prefill|`ragged_moe_gemm_fp8` 84，MUBIN group-block FP8 kernel|MATE 专家 GEMM 已用；路由/量化/恢复为 xLLM；不是 MegaMoE|
|MoE decode|`fp8_moe_w13_gemv_swiglu_kernel`、`fp8_moe_w2_gemv_kernel` 各 294|自定义设备实现，未用 MegaMoE；不能因为没用库就认定性能差|
|mHC|`Mp31Tf32HcPrenormGemm` 720，自定义 finish/post 720|只接了 MATE 前半段；`mhc_pre_big_fuse` 是高优先级候选|
|vision attention|24 × `musa_asm_bf16tce_flash_atten...`|源码走 torch SDPA，实际为 muDNN attention；没有 MATE FMHA 证据；MATE 优先顺序的适用性评估尚缺|
|vision RoPE、norm/conv|自定义 rotary/rmsnorm、conv + ATen 操作|不应把所有辅助 kernel 都视为已经 MATE 化；逐项比较融合价值|

KDA：TP8 每卡 8 heads、D=128；q/k/v/raw gate 逻辑 `[1,T,8,128]`，BF16；beta BF16，A_log/dt_bias FP32，state `[B,8,128,128]` FP32；scale=1/sqrt(128)，varlen cu_seqlens 转 int64，Q/K norm 在 MATE 内执行。prefill 有 initial-state validity 处理，decode 从 state slot gather，输出 scatter 回去。工作区每次 `empty`，chunk=16、record stride=6784，每卡 prefill T=2048 的 scratch 约 13.35 MiB，decode 约 0.207 MiB（当前公式多预留一个 sequence record）。不是永久 KV cache。

MLA：TP8 每卡 8 heads；q 从 `[T,8,256]` 经 Wk 吸收到 latent 空间 `[T,8,512]`，cache 为 `[P,256,1,512]` BF16，输出再经 Wv 回 `[T,8,256]`。NoPE，scale=1/16，causal 由 candidate mapping/lengths/positions 保证，并非通用 dense causal mask。BF16 wrapper 对连续布局、head 数 8/16/32/64、int32 索引、候选宽度 64 对齐、32-bit 地址范围有明确限制。

选择顺序：KDA/MLA 已停在 MATE，无须退到 muDNN/muBLAS。外围投影使用 GEMM 不代表 attention core fallback。vision 采用 muDNN，但没有发现针对当前 shape 的 MATE 拒绝原因或性能对比记录，所以只能说“尚未证明符合 MATE 优先选择要求”，不能虚构 MATE 不支持。vision core 总计仅约 1.308 ms，当前优先级低。

MATE 与 xLLM 通过 TVM-FFI/AOT 导出的 `.so` 交互，`TvmffiStreamGuard` 对 Torch/FFI stream 做顺序交接；trace 可见 stream 0/14/17 和大量 event record/wait。优化 stream 绑定可减少交接，但必须保留工作区与输出生命周期，不能直接删除 event。

## 明确值得优化的地方

### P0：mHC finish 的单线程标量循环

`mhc_pre.cu:173` 起的 finish 将 split partial reduction、sigmoid、4×4 Sinkhorn 全放入 `thread_id==0`；20 次 Sinkhorn 的多重循环和除法串行执行，然后其他线程才处理 hidden mixing。decode grid=1，仅一个 CTA，平均 429 µs × 90 次 ≈ **38.60 ms/步**；prefill 90 次合计 192 ms。MATE `hyperconnection.mhc_pre_big_fuse` 支持 split-K FP32 partials、BF16 residual、hc_mult=4、Sinkhorn 等参数，输入语义接近。

行动：导出当前安装版本 TileLang big-fuse 为独立 FFI 模块，与当前 finish 对比；核对 eps、先行 row normalization/column normalization、post multiplier、20 次迭代与 BF16 cast 时机。不能仅减少迭代次数冒充等价优化。再看 mHC post 的 prefill 90 ms，尝试 vectorized copy/mix 或跨层 post+pre fusion。最保守的无重叠理想上界：只删除整个 finish 462 ms，相对 rank0 GPU span 最多约 1.17×；这不是预测收益，跨 rank 等待变化也不包含在该上界模型里。

### P0：prefill IndexPool 并行化

`glm5_next_indexer.cu:60` 每个 request 一个 CTA，循环处理 2048 token，四个 token 一次 softmax compress + Hadamard + quant。当前有 512 个可并行的完成 pool，却让它们串行共享同一个 CTA。

行动：prefill 把完整的四 token pool 按 pool 分块并行执行；另处理未满四个 token 的 tail 与有 prefix 的跨 chunk 边界；每个 query 的可见 pool_lengths 仍必须按位置写。decode 保留增量 update。该优化直接针对本次 prefill 182 ms，远比改已经只有 8.7 ms 的 KDA 核心更有价值。短上下文可跳过 score，但不能跳过 cache/tail 维护，否则第 2049 token 以后失去历史池。

### P0：将元数据检查移出逐层热路径

`glm5_next_linear_attention.cpp:442–470` 每个 KDA 层重复 `ge/lt/all/item`、sort、cu 首尾 item 和单调性检查。同一 batch 的 indices/cu 没有必要在 34 个层里重新向 host 读取。还有 conv update 的 ATen gather/scatter、每步 conv_weight cat、重复 dtype 转换与临时分配。

行动：在 metadata builder 的 host 已知数据处验证一次，向层传递已验证的 device metadata；保留入口边界检查和调试模式验证。构造/加载阶段拼接 conv weight；复用 scratch 和 int32/int64 indices；用 fused norm+sigmoid gate 替代多次 cast/pow/mean/rsqrt/mul。不要把去掉 item 的 1.6 秒等待全部列为收益；真实收益来自 launch 提前、减少 host/device 往返和更少的 GPU bubbles。

### P1：graph、通信及 rank 同步

decode 约 3593 个 kernel/步（rank0），launch 密度高，但活动并集不到半个阶段。MUSA graph 有价值；**不能直接把启动参数 enable_graph 改为 true**：当前 MLA forward、sparse MLA wrapper、indexer 都显式拒绝 capture。先改 graph-safe buffer、稳定地址和 metadata 更新，解决 item、FFI stream capture、allocator 生命周期，再做连续两次以上 replay 与请求长度变化测试。

custom AR 在 decode 合计 500.7 ms（约 71.5 ms/步），平均 832 µs/次，747 次全程最慢约 52.5 ms，显示等待影响很强。下一轮应按相同 collective 序号匹配 8 rank 的 launch/start/end，区分到达偏差、busy-wait 与传输，并记录 payload bytes、拓扑。先减少前序串行工作和 profiler 扰动；再 A/B custom AR 与 MCCL、1-stage 与分片策略。不能将 50% kernel 占比直接当作可通过更高带宽省掉 50%。

### P1：MegaMoE、路由融合与 FP8 decode

MATE MegaMoE 需要 intranode group≤8、完整专家按 EP 分配、对称 IPC buffer、FP8 dispatch、权重转换，hidden/intermediate 128 对齐，topk≤32。模型 hidden4096/intermediate2048/experts288/topk8 表面满足整除约束。当前 RoutedMoe 以 TP shard 的 intermediate 权重布局运行；把其换成 EP8 的完整专家权重、处理 replicated token 的 ownership，才有可能使用 MegaMoE，不能仅替换一个 GEMM 调用。

实验先保留 attention TP8，只改变专家并行与通信；检查 shared expert、sigmoid noaux routing、correction bias、归一化、routed_scaling_factor=2.5、SwiGLU clamp=10 是否完全一致。特别避免 8 个 TP rank 把同一 token 重复当作 8 个独立 token dispatch。导出 stage1/2/3、IPC buffer/runtime utils 与 weight-transform，确认 C++ 可调用和实际加载符号。当前 Python API 存在不代表 xLLM 有开关可直接开启。

更低成本可先评估 `mate.moe_fused_gate`，替换现有 FP32 sigmoid/topk/gather/normalize 多 kernel；必须匹配 sigmoid/noaux_tc，而非误用已有仅 softmax 的 topk wrapper。再比 prefill ragged 的 M=128 padding 与 compact/contiguous 方案。

现有 FP8 NoPE cache/scheduled decode wrapper 可作为 DSA 实验分支：cache record 528 bytes，当前 BF16 为 1024 bytes，理论减少约48.4%主 cache 字节。不过本次 decode DSA core 仅约1.24 ms/步；短上下文收益上限小。应在 8k/32k 等长度和更大 batch 上测试，不能只为换 kernel 名字而量化。

## KV cache 是否合理

**表示方法总体合理，容量预留是否合理无法仅凭这份 trace 判定。**

- 11 个 DSA 层存压缩 latent，未存 64 heads 展开的 K/V；每 token 每层 BF16 512×2=1024 B。TP ranks 复制 latent/indexer，与每卡独立 Q heads 的 MLA 计算相容；不能再简单除以 TP8。
- IndexPool 每四 token 一条 128-byte FP8 payload +4-byte scale，平均33 B/token/layer，页为 `[P,64,1,132]`；main page `[P,256,1,512]`，同一页覆盖256个原 token。另保留 `[slots,2,4,128]` BF16 tail。
- KDA 是 recurrent state，不应按2048 token给每层再乘2048。每层每请求本 rank SSM=8×128×128×4=524288 B；conv=3072×3×2=18432 B；34层共 **17.598 MiB**，加11层 KPool tail **22 KiB**。
- L=2048 时每 rank DSA main22 MiB、index0.709 MiB、SSM+conv17.598 MiB、tail0.0215 MiB，逻辑有效状态约 **40.33 MiB**。进入第2049 token时page_size256，main/index页从8页到9页（尾页未用部分也占空间）；并非8个输出就只增加8个token的物理字节。
- 本次8个step是一次prefill后增量decode；write_cache88次、KDA state读写272次，与正常cache使用一致；没看到把所有历史hidden复制到host重算的证据。但这是执行路径证据，不是cache数值正确性证明。

**当前环境旁证（不是采集进程的证据）：** `/tmp/glm5_3_flash_run/rank_0.log` 属于16:09启动的 PID622921，而 trace 的 rank0 PID532116、时间约14:15；日志已被覆盖/更新。该后续进程报告 blocks3446、state_slots498、state reserve8.57GiB、总cache预算18.12GiB、max_seqs16、prefix cache关闭。这不能回填为 baseline 的实际容量。

若 baseline 同样自动预留498个state slot，则对 C=1 以及 max_seqs16 显著过多：仅17个slot（含padding）约299.5MiB，32个约563.8MiB，相比498个约8.57GiB。可先依据 scheduler 活跃槽、padding、prefix snapshot、speculative rollback 和PD双缓冲需要确定 slot 上限，再通过 `max_linear_state_cache_slots` 做对比。释放的容量主要改善余量/并发，不会自动降低单请求TPOT；还应检查预算是否转给大量无用KV页、初始化耗时、workspace争用。

prefix cache关闭对单次独立请求合理。若真实业务共享1024-token前缀，需要同时保存/恢复KDA SSM、conv以及IndexPool tail，不可只开启普通paged KV prefix复用。当前MUSA路径还限定checkpoint_stride=1，不能把MLU的speculative/PCP能力当成本后端已经具备。

## PD 分离怎样做才可能加速

### 对本次单请求

用观测近似：P≈1.14s，7D≈2.10s。孤立单请求总时延 `P + transfer + 7D`，没有另一请求可与P重叠。TP8的8张卡直接拆成P4+D4还会改变模型驻留显存与算子性能，不能保证可放下或更快；当前后续进程每卡非空闲显存约45.8GiB也提示TP4容量风险，但它不是baseline权重精确占用。

### 对持续到达、多并发流量

先用额外设备做 **P TP8 + D TP8** 的对照，保证shape/layout一致，再评估异构TP。理想化每请求阶段服务需求约P1.14s、D2.10s，单P单D、相同每请求成本的流水吞吐上界改善约 `(1.14+2.10)/max(1.14,2.10)≈1.54×`，忽略传输、排队、batch变化，且使用双倍卡；不是单请求加速，也不是单位卡吞吐收益保证。服务池容量按各阶段无profile实测吞吐/TTFT/TPOT SLO配比，不能由这一条C=1曲线直接定生产部署。

P做图片编码、文本prefill、cache建立；D保留权重和收到的状态，从prefill产出的首token之后继续。首token返回、state ready、decode启动需一致，防止重算/跳算一个token。

必须迁移：11层MLA latent页、IndexPool FP8 payload及其page内scale、KPool tail；34层KDA FP32 SSM与BF16 conv；有效长度、slot/block映射、当前位置、first-token/sampling状态。不能只传传统K/V。

相同TP8、L2048，按有效状态估算约40.33MiB/rank，8rank简单逐对复制约322.6MiB（约0.338GB）；若整个任务实测聚合传输带宽分别10/25/50GB/s，则纯payload时间约33.8/13.5/6.8ms，另加握手、排队、页注册、同步。不应把链路标称带宽当实测值。main/index复制可以考虑只传一份再在D组内广播，KDA需按head shard传；是否值得取决于网络/机内拓扑和广播开销。

代码已有 `mooncake_kv_cache_transfer.cpp` 和 `reshard_planner.cpp` 的 MUSA/linear-state/INDEX 分支，可复用manifest、typed tensor role和层完成通知；这并不证明当前GLM混合cache的PD流程已经跑通。先做等TP整请求迁移正确性，再做layer-wise ready/transfer与剩余prefill重叠，后续才做P/D不同TP的SSM head reshard、conv feature分片、replicated latent/index广播。两个实例若使用BF16/FP8不同cache格式，需要显式转换，不是字节copy。

当前GLM MLA构造明确要求cp_size=1、kv_split_size=1，因此不能直接通过PCP配置加速P。MLU后端的KDA PCP有局部prepare、prefix-state merge/replay和conv边界广播，可参考语义，但需独立实现MUSA接口与正确性测试。

## 版本、参考与证据定位

本地当前 xLLM HEAD：`c14384129d647d27ffc4bfa9aff45b98f24aa123`；MATE checkout `/data/albert/Projects/mate` HEAD：`9d3260d633597f08fc9d5d4dc73344872241c34d`。当前安装 MATE `0.2.8.post1`，位置 `/usr/local/lib/python3.10/dist-packages/mate`。checkout带新的`mate/ops`结构，而安装包布局不同，不能互换当成同一ABI；本报告接口能力判断优先读安装包。trace本身不能证明采集时二进制正好对应此HEAD，也不能恢复当时加载的.so哈希。

|证据/参考|位置与用途|
|---|---|
|KDA调用、state及同步|`xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.cpp:430`、`xllm/core/kernels/musa/kda.cpp:74`|
|MLA main/index布局、FP8分支|`xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.h:39`、`:130`；同名cpp`:284`；`xllm/core/kernels/musa/flash_mla_sparse.cpp:41`|
|IndexPool串行更新/评分|`xllm/core/kernels/musa/glm5_next_indexer.cu:60`、`:290`|
|mHC串行finish|`xllm/core/kernels/musa/mhc_pre.cu:173`；MATE安装包`hyperconnection.py:183`可用big-fuse接口|
|MATE MoE GEMM|`xllm/core/kernels/musa/musa_moe_gemm.cpp:639`，直接选择MATE MUBIN launcher|
|MoE调用方|`xllm/core/layers/musa/routed_moe.cpp:784`与`:891`，grouped/自定义GEMV分支|
|MATE MegaMoE|安装包`mate/mega_moe.py:150`、`jit/mega_moe*.py`；checkout`tests/test_mega_moe_prefill.py`，含权重转换及多rank replay；已有测试不等于本模型已验证|
|AOT/导出|本目录`compile_musa_kda.py`、`compile_musa_sparse_mla.py`、`compile_musa_mhc.py`；验证FFI function和部分manifest；只导出prenorm不等于导出mHC big-fuse|
|另一backend约定|`xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:498`、`glm5_next_kda_pcp.cpp:173`，读写槽、checkpoint、conv边界与PCP；`xllm/models/llm/mlu/glm5_next.h`的pending MHC|
|vision|`xllm/models/vlm/musa/glm5_next_vlm.h:263`，逐sequence非causal SDPA|
|采集开销/合成区间|`.../xllm_profile/torch_callstack_preload.cpp`、`inject_cpp_callstack_slices.py:220`|

技能的 MATE API 文档仅给出 [MATE 0.2.8 文档入口](https://mate-docs.mthreads.com/0.2.8/index.html)，此次在线读取失败；未以不可读取网页支持任何接口结论，接口均核查本地安装文件与源码。本任务没有新增模型，因此未以vLLM代替当前实现；模型shape/语义来自实际config及xLLM调用链，其他backend参考仅用于对齐cache/并行约定。

## 可执行的验证顺序

1. 归档同次启动完整命令/env、模型config、二进制/算子SHA256、maps/ldd、设备拓扑和客户端结果。关闭callstack/profiler测C1/ISL2048/OSL8，再增加OSL128/256确认稳态；报告中约300ms仅是本次profile观察。
2. 分别A/B mHC big-fuse、并行pool update、一次性metadata验证/weight拼接。每项先测kernel，再测真实图文请求；逐项保持其余配置不变，不叠加理想收益。
3. 用同一组输入验证prefill输出、至少两个decode步、state复用、slot复用、token位置3/4、2047/2048/2049边界、非满page、多个request和不同长度；检查因果候选和tail、FP32 state误差及最终token。FP8 cache再单独做数值精度验证。
4. 清除graph阻塞后做capture/replay；确保第二次及后续请求复用、shape切换和采样结果正确。记录host sync数、launch数、GPU活动并集和无profile TPOT。
5. 之后比较FP8 cache、fused gate、MegaMoE/EP8；同时记录routing skew、padding容量、显存峰值、通信bytes、TTFT/TPOT/吞吐，不能仅报专家GEMM microbench。
6. PD先等TP8完整状态迁移并逐token对齐单实例结果，再测持续多并发、长输出、layer-wise迁移、异构TP。记录迁移字节/带宽/尾延迟，比较相同总卡数的非PD基线。

本次完成：8份trace解析、阶段归属、kernel/runtime统计、GPU区间并集、模型/源码/MATE接口核查。没有执行算子替换、数值测试、编译或新推理，因此所有优化倍数只能标为分析上界/实验假设。

复现统计（逐rank加载，避免同时展开8份trace）：

```bash
python baseline_tp8_1111_analysis/analyze_trace.py /data/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/xllm_profile/baseline_tp8_1111
```

输出校验：8个rank的聚合kernel数量、总耗时与原始汇总一致，GPU活动并集不超过跨度。
