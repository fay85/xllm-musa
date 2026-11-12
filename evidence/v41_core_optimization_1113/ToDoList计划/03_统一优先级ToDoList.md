# 统一优先级 ToDoList

本表合并各来源，按**同场景可缩短的全局关键路径**组织；不是分人各做一轮、也不是每轮退回 eager。当前起点为 StageI。正式未采集到各模块无 profiler 关键路径占比，故下面是**初始排查/实现顺序**；不是已经有数值证明的收益排行榜。G02 是必需步骤，完成后在本表和 JSON 写入上下界、理由，再按理论端到端收益上限降序排序。未知项先测，不伪装成零收益。

证据强度：A=同条件正式 A/B；B=校准后的无重 profiler 关键路径；C=详细 trace/源码线索；D=外部参考。目前未来任务主要 C/D，已有 SWA 改进是 A，但它已完成。

| 顺序 | ID | 待办 | 进入理由与当前上限状态 |
|---|---|---|---|
| 1 | G01（前置） | 恢复固定基线、环境与真实快路径 | 先核验，不计为性能改进；未测真实局部上限 |
| 2 | G02（前置） | 轻量全局计时、关键路径分解与重新排序 | 消除现有重 profiler 的归因盲区；未测真实局部上限 |
| 3 | T01（核心1） | 拆解慢 rank 等待，适配实际小消息 AllReduce | 当前 MCCL 轨道最显著，真实可缩短部分待测；未测真实局部上限 |
| 4 | T02（核心2） | 融合 decode 页地址、SWA/压缩缓存 gather 与 metadata | 当前 host 大量细碎操作，SWA commit 已完成，重点转读侧；未测真实局部上限 |
| 5 | T05（核心3） | 分段 Graph → 可验证范围的 decode Graph | 覆盖启动空隙但与 T02/T03 重叠，先量化捕获范围；未测真实局部上限 |
| 6 | T03（核心4） | mHC metadata/combine/norm 周边融合 | 当前 prepare 已有 MATE，找剩余 launch 与重复读写；未测真实局部上限 |
| 7 | T04（核心5） | MoE gate/路由/共享专家/小 M 选路及负载均衡 | 不能先假定 W4A8 GEMM 本体占比最大；未测真实局部上限 |
| 8 | T06（核心6） | Dense/投影与 QDQ 的真实 shape 专项 | 已有7类慢候选，必须定位新瓶颈再做；未测真实局部上限 |
| 9 | T07（核心7） | 服务调度、CPU/NUMA、采样及日志的有据优化 | G02 若证实服务差值大则升到 T01 前；未测真实局部上限 |
| 10 | T09（核心8） | Prefill chunk 与批量投影扫点 | 当前 C1 E2E 收益受整个 TTFT 0.78%/5.04% 限制；未测真实局部上限 |
| 11 | T08（核心9） | P/D 传输、GPU/NIC 亲和和层级重叠 | 跨机传输主要作用 TTFT，不混同 D 内 TP/EP 归约；未测真实局部上限 |
| 12 | T10（扩展） | CED / SWA bounded replay | 近似/策略差异需精度协议，不按普通等价融合晋级；未测真实局部上限 |
| 13 | T11（扩展） | 前缀缓存、状态池与避免重算 | prefix 关闭的 C1 无直接收益，需新工作负载；未测真实局部上限 |
| 14 | T12（扩展） | 多请求批处理与服务吞吐曲线 | 当前 C1 实现有硬限制，先完成正确性和资源管理；未测真实局部上限 |
| 15 | T13（扩展） | 同16卡重新分配 TP/EP/DP/CP 与 DeepEP；再考虑多机 | 维度不同不能把所有并行数机械相乘；未测真实局部上限 |
| 16 | T14（扩展） | DSpark / MTP 固定验证后再动态验证 | 新增接受/回滚状态机制，以真实接受 token 计吞吐；未测真实局部上限 |
| 17 | T15（储备） | 长上下文/容量瓶颈下的专家缓存与 offload | 当前全GPU常驻不适用，不主动制造搬运；未测真实局部上限 |

## G01｜环境、基线和快路径恢复

先确认 SJ9/SJ10 空闲与我方容器用途，核对 toolkit 实际版本、驱动、固件、有效 core、GPU 时钟/温度、CPU NUMA、libmtml、镜像 digest、MATE/FFI/MCCL/Mooncake/AOT SHA、模型 revision、实际 exe。禁止用“同一个 tag”代替同一镜像。

按 I 的完整有效参数恢复两组 1暖+5正式；若环境升级或发生明显漂移，建立 I@新环境基线，后续只与它比较。验证 16 rank、每 rank 40层、TP8/EP8、模型权重路径；确认 W4A8、mHC、Attention 的实际 kernel。G01 不借机升级依赖或修驱动。

输出环境差异表、实际启动命令、逐请求结果、历史 I 与新 I 的差异及当前最优 manifest。源码位置：`tools/` 的本任务现有 launcher/bench 包；快路径在 `w4a8_experts.h`、`hyper_connection.h`、`csa2_attention.h`。若正常退出异常，归档并停止扩大实验；不得沿用历史一次性 SIGKILL 授权。

## G02｜先回答时间究竟花在哪里

按 `02_全局占比与收益上限.md` 完成服务端/设备/各 rank 三层计时。捕获一个短输入稳态窗口和一个 2048 对应的早/中/晚 decode 窗口，另捕获多 chunk prefill；不要把 256/8 的最后三步代表整个 2048/256。

先测 profiler 开销；补每个 collective 的 group/bytes/dtype/后端、各 rank 到达偏差、依赖关系。记录 Attention、MoE、mHC、Engram、采样以及服务空隙的可缩短时间区间。对“CPU-bound / memory-bound / compute-bound / communication-wait”分别举证：CPU 发射空隙；有效带宽与搬运字节；同精度硬件计算吞吐；各 rank 等待及通信链路。仅凭 GPU utilization 低或 kernel 名字不能分类。

产出 `critical_path.csv`、所有 rank 的概要、局部可缩短 `p_low/p_high`、重叠图、排序变更。优先整合已有计时，不为了观测改模型方程。若无法低开销分解，就用保守界限与微基准探针继续，保持“上限未知”；不得编数字满足表格。

## T01｜MCCL 等待与小消息 AllReduce

**证据/来源：**S3/S4/S5/S7。详细 trace 三窗 MCCL 占窗口约56%～67%，但含采样干扰与等待；该比例不能直接作为 p。先区分“某 rank 来迟了”与“所有 rank 就绪后仍很慢”。如果是前者，优先优化慢 rank 的 host/算子/亲和，换通信库未必有效。

**实现落点：**`xllm/core/framework/parallel_state/musa_process_group.h`、`musa/musa_custom_all_reduce.{h,cpp}`、`core/kernels/musa/custom_all_reduce_ops.cu`；模型端 `decoder.h` 的 Attention TP 归约和 `moe.h` 的 F32 EP 归约。现有 custom AR 已支持若干 dtype/group，但并非任意 `ep_group` 默认可用，且记录中当前确实有 MCCL。检查实际 group 名称、IPC 设备拓扑、size 门限、contiguity、current stream 与所有 rank 一致选择。

**最小实验：**先真实 BF16/F32 message shape 的8卡归约；所有 rank 统一选择原后端/候选后端，检查返回值是否被消费。先只替换 Attention TP，再单独评估 EP，保持共享专家只加一次，禁止重复归约。随后同一模型执行配对 A/B。数值覆盖大小/正负/边界、跨 step 缓冲复用；浮点归约次序变化记录误差及固定前缀 logits。

**看图：**按相同 collective 序号比较各 rank 的到达、开始、消费者等待；不能仅看 rank0 的 MCCL 变短。**上限：**只计没有被其他工作遮住的归约/等待减少量；等待与造成等待的 CPU 优化不可叠加。

**停止/升级：**全就绪微基准不快、端到端无收益或实际仍走 fallback 则不晋级。最小外援是具体 shape/dtype/group 的可用 custom AR 接口或死锁复现，不是泛问“帮忙优化通信”。

## T02｜读侧分页地址、SWA 与压缩缓存融合

**证据/来源：**S2/S3/S4/S5/S7。`paged_attention.h` 每层含 arange、floor_divide、remainder、where、index_select、cat；`token_cache.h` 还有压缩缓存 gather。40层反复启动，在当前详细 trace 是显著 host 热区。SWA 单 token commit 已经完成，不重复做。

**实现：**优先仅 q=1，将物理地址映射、无效行零填充、gather/picks metadata 合并为少量 native MUSA kernel；再考虑使 Attention 直接读取 paged cache，减少整块临时 rows。参考交付 `deepseek_v4_backend.py` 和家明 `deepseek_v4_indexer_metadata.cu`，保留现有 MATE Attention 方程。新 `.cu` 接口放 `core/kernels/musa/`，C++ 编排放 `paged_attention.h/token_cache.h/indexer.h`。先缓存静态 arange/只读 workspace，再考虑跨层复用；任何复用须验证 source layer 与请求/step 生命周期。

**验证：**1/127/128/129/255/256/257及跨SWA片界；padding/负slot/NaN无效KV；q=1及prefill回退；先读后写、稳定 latest-wins、current stream、TP8和P/D状态。索引/写回要求逐位，Attention输出按既有容差并核对 logits。

**上限/图：**仅计可移除 metadata/搬运/launch 的关键路径；不能计整个 Attention。检查对应 `aten::` 调用、memcpy、kernel 次数、临时内存、GPU空隙、通信到达偏差；真实端到端仍按两组复测。一次做一个融合点，不同时换量化/Attention公式。

## T05｜Graph 捕获与 replay

**证据/来源：**S2/S3/S4/S6/S7。交付 full graph 不是当前 xLLM 自动可用。当前模型有 `.item()`/host 长度、动态索引、临时分配、状态和通信，必须先列 graph break。它与 T02、T03 都可能省同一批 launch；任一项成功后重测另一个剩余上限。

**实现：**从纯设备、固定地址的子区间捕获，逐步扩到 decode step；落点 `runtime/musa_graph_executor_impl.{h,cpp}`、`text_model.h/request_metadata.h` 与缓存类。GPU metadata 保持地址固定、内容更新；MATE/FFI workspaces 预分配；检查 custom AR/MCCL capture 合同及非默认 stream。只声明实际覆盖比例，不能以传了 graph 参数当完成。

**验证：**同一图重放不同长度/不同请求、页表变化、ring回绕、请求重置、P/D切换；固定 token 前缀比较 logits；覆盖有效 batch，而不是只 replay 暖机输入。分段先通过再尝试更大捕获，不修改宿主机驱动来追 full graph。

**验收：**trace 出现对应 graph replay，launch空隙下降且实际 token 正确；记录 graph内存和首次capture成本，正式暖态测试另计。候选收益上限为被覆盖区间内真实 launch/调度间隙，不能算全部 forward。若捕获范围小或状态不安全，保留分段或明确暂停。

## T03｜mHC 剩余融合

当前已有 MATE m32/m64 prenorm；不是从慢参考分支首次接入。参考 S2 的 mix-dot、combine-norm/tail chain 和 S3 的真实快路径检查；当前 `hyper_connection.h` 的 metadata/Sinkhorn、carried_pre combine、post、邻接 norm 可逐项分析。

落点 `hyper_connection.h`、`decoder.h`、`deepseek_v41_mhc_prenorm.*` 和已有 `deepseek_v4_hc_*`。先尝试兼容的小 M metadata/combine/norm 合并，保留 V4.1 shifted pre_mix：本子层算出的 pre 留给下一子层，当前 collapse 用 carried_pre。不能直接套 V4 的方程顺序。

数值覆盖真实Attention/FFN参数、M1/17/128/256/512、Sinkhorn次数、极值、空输入、跨子层传递，检查完整模型固定前缀。图上应看到相关 reduce/cast/pointwise 减少；上限仅是该链条可移除的关键路径，不是把 mHC prepare 的45.8ms直接从正式TPOT减掉。若现有MATE接口不能表达融合，提交最小shape/公式/误差/时序包给MATE，而非笼统称缺mHC。

## T04｜MoE 周边、真实小 M 与负载

参考 S1/S2/S3。当前 W4A8 grouped GEMM 已在工作；先查 `v41_gate_kernel`、路由/dispatch_quant、padding、SwiGLU量化、combine、共享专家以及各rank有效token/expert数。记录每rank实际选中专家和 padded-M，`最慢rank时间/平均rank时间` 与负载分布；EP8的8张卡不等于每个token都分到相同专家数。

落点 `moe.h/w4a8_experts.h/feed_forward.h`、`deepseek_v41_gate_topk.cu`、`deepseek_v41_w4a8.cu/.cpp`。优先优化metadata/小M工作量；仅在MATE现有group128合同内试tile/小输入选路，不重回逐专家Python或每token解量化。共享专家计算可尝试与路由路径重叠，但要确认内存/SM争用。

验证 top6、sqrt-softplus、scale1.5、clamp10、tie及hash路由、专家归属/顺序、共享只加一次。负载重排须同步修改所有rank映射并验证实际权重；先静态小规模，再考虑EPLB。上限用最慢rank关键路径，不用平均rank FLOPs；专家缓存或大规模DeepEP另见T13/T15。

## T06｜Dense/Attention 投影专项

参考 S2/S3/S4。当前 Dense 保留block32 QDQ后BF16 GEMM；此前7个真实shape的AOT候选数值通过却更慢，不能重复无差别重写。wo_a单token已使用现成V4 GEMV。先根据G02明确哪一项投影、dtype转换或布局copy仍占大头。

落点 `attention_projection.h/feed_forward.h`、`deepseek_v41_dense_block32.*`。候选是缓存静态权重布局、去重复transpose/cast、同量化边界的投影合并、特定shape的MATE/MUBLAS选择。联合GEMM不能改变per-group scale及FP32/BF16舍入顺序；参考家明request-local shape修正，不能把两个请求错误拼成一个状态。

对真实权重、M1和prefill shape做微基准，数值通过后才两组整模型测试。加速只出现在大M而当前M1无效时，归入Prefill任务，不报decode提升。原版FP4/group32开发不是本W4AFP8路线当前外部阻塞。

## T07｜主机、调度、NUMA和采样

G02若发现设备forward远小于服务step，提前到核心队列首位；否则不凭CPU利用率猜。参考S3/S4/S5/S6/S8，分析 `continuous_scheduler.cpp/disagg_pd_scheduler.cpp`、`distributed_runtime/llm_engine.cpp`、`framework/sampling/sampler.cpp` 以及本任务启动配置。

优先核对NUMA/CPU亲和、线程竞争、温度降频、同步D2H和日志开销；亲和是进程级设置，不改全机调度策略。采样检查bulk D2H与top-k功能当前是否已存在且被使用；温度0的argmax路径不能拿随机采样优化证明收益。默认无progress超时另建隔离诊断，不一次删光日志再让通信静默卡住。

图上对齐请求/step边界、host gap、设备idle、通信入队点。只修改一个变量；若numactl与日志变化共同影响就拆开。上限是已证明可去掉的host串行关键路径，`服务step-设备时间`只是诊断量，不是可全部消除的CPU预算。

## T09｜Prefill chunk 与投影合批

先256/512/1024/2048，必要再看3K/4K；用相同2048 prompt和输出，记录实际chunk数、峰值显存、TTFT、D稳态TPOT。长于prompt的chunk设置不能伪装成更多计算测试。参考S2/S3/S8，落点scheduler token budget、`request_metadata.h`、`compressor.h`和Attention投影。

C2奇数跨块、Engram history、跨层KV/TopK复用必须回归；连续chunk结果与固定前缀单段forward对比。当前C1的整个TTFT消除上限已仅0.78%/5.04%时间，故按E2E目标排在decode后；若领导改为TTFT目标则上调。prefix仍关闭，另行测试cache命中，不混口径。

## T08｜P/D传输及亲和

区分D机器内部TP/EP通信与P→D跨机Mooncake。检查GPU/NIC rail、PCIe/NUMA、实际RDMA而非意外TCP、传输字节、状态布局、control handshake、拉取完成到首decode的等待。`mlx5_*`逐台发现，不复制固定编号。

落点 `framework/kv_cache_transfer/`、`disagg_pd_service`相关实现、P/D启动配置和`request_metadata.h/token_cache.h`。先亲和与冗余传输；再考虑分层ready后提前搬运、消费前等待，参考Albert的依赖重排。每次保留cache版本及可见性，不能提前读未完成数据。

与总TTFT同上限约束；小输入C1传输不显著时降级。需要网络协助时给实际端口、rail、带宽/延迟、失败信息；不修改交换机、驱动或其他容器。

## T10｜CED / bounded replay（条件任务）

当前40层完整chunk路径仍保留。交付从late层21的尾部512行重建，与官方Q&A说明的近似128尾长不是同一参数；当前chunk256直接裁为512不会省行。先明确跨chunk延后执行、状态恢复和近似误差，再实现 `text_model.h/decoder.h/request_metadata.h` 与P/D缓存协同。

须用固定token前缀检查decoder SWA、compressedKV、首个decode及长输入logits；明确精度协议和真实节省prefill行数。它可能不是逐位等价优化，单列结果，不混入无损核心基线。若无可接受误差口径，记录待确认而不默认启用。

## T11｜前缀与状态池（条件任务）

先启用受控前缀cache工作负载：0%、固定可核验命中、真实复用轨迹；输出真实cached/uncached tokens、重算、P/D字节、峰值内存和尾延迟。S1/S3/S6的state生命周期、SWA/C2/Engram hash/跨层缓存全部要遵循，不能只缓存K/V。

落点block/cache manager、V4.1 prefix policy及状态快照；只有观察到一池用尽而另一池空闲才尝试共享池/淘汰策略。命中率提高但TTFT变差不晋级。前缀无效或stale-state样本单列。

## T12｜多请求与吞吐

当前 `paged_attention.h` 对block_table batch=1有CHECK，扩并发需要先打通request-local metadata/状态/工作区、ragged batch、expert路由以及请求结束回收；不能只把max-seqs改大。参考S3/S4的请求局部shape和组预算。

C1继续作为回归；新增C2→4→8…逐级测，容量够再升。每点独立1暖+5为快速筛查；正式吞吐需固定请求数和到达率、观察稳定区间与p50/p95/p99，少量5样本不能声称可靠p99。报告aggregate output/s、per-request TPOT、QPS、OOM/失败、缓存口径和显存。不同并发不是“同条件代码加速”。

## T13｜并行布局和DeepEP

参考S1/S2/S3/S4/S6；先显存模型与分组语义，后实测。Dense/Attention TP、请求DP、专家EP、序列CP是不同维度，不能机械把8×8算成64卡；也不能把Qwen的DP8照搬为每卡装510GB全模型。

当前同token复制到EP rank +结果allreduce，可评估交付的请求DP+专家EP token分发/回收、DeepEP normal/LL与FP8通信；必须统一token归属、专家权重布局、共享专家归约和P/D重分片。先通信数值/空rank一致性，后完整模型。CP需覆盖C2边界/Engram lookback/跨层TopK/source layer并做all-gather/reduce的顺序验证。

扩到2P1D/1P2D/2P2D需新的服务器可用许可。相同总GPU预算与不同GPU预算分开报告；多P对单请求未必提速。正文详见07，未获资源不自动占sj11/12。

## T14｜DSpark/MTP

参考官方DeepInfer与交付。先核对对应draft/Confidence Head权重、Tokenizer、quant和执行接口，再固定少量draft长度验证，最后考虑动态verify预算；不要假设当前主模型权重已包含全部可用draft资产。

落点speculative/mtp worker、draft模型、GPU verifier与V4.1的状态snapshot/rollback。统计每轮draft数/接受数/有效输出、draft/verify耗时、通信、内存和端到端；吞吐按实际接受并输出token算，不能按候选token算。接受率高但额外计算抵消收益也算失败。须验证拒绝后的SWA/C2/Engram/mHC状态和输出协议，属于独立阶段。

## T15｜容量受限时的offload储备

当前W4A8专家已GPU常驻，不为模仿llama.cpp主动做offload。仅当新并行/长上下文确因容量受阻，再用专家访问分布估计缓存命中和H2D代价；分别核对专家与Engram的offload。实际带宽、pin内存、NUMA、缓存管理与延迟尾部证明可行后再实现。当前无需编码，保持deferred。

## 每项共同的完成/失败标准

以当前最优提交作为before，先固定p与Tideal（测不清写unknown）；仅一项行为改动；必要数值/通信回归；官方工具两组1暖+5无profiler；独立同场景trace、ATen/原生调用与GPU关联、真实火焰图/时间线；记录晋级或拒绝。性能差小于波动、语义回退、未真实选中候选、收益来自输入/输出变短，都不能写有效优化。

正式5次用于本轮快速对比；若差异与噪声相当，再做一轮交错A/B或ABBA，不能反复测到挑出好结果。建议把至少3%的稳定改善作为“有实际意义”的初始阈值，明确这是实验决策阈值而非领导SLA；更小但低风险稳定的收益可保留为待评审项。任何两组取舍都单列，不隐藏另一个case的回退。更新累计最优清单后重新测热点与剩余上限。
