# G01/G02结果与进入核心优化的边界

## 当前做到哪里

完成当前StageI基线复核、OFF/HOST/DEVICE各两组完整正式测试，以及16rank原生CPU/设备范围计时。当前进入T01的通信数值和微基准诊断，还没有新的整模型性能优化被晋级；整个goal仍active，核心T01～T09尚未完成。

G02按计划允许的保守边界退出：短输入观测开销、16rank覆盖已验证；长输入存在既有文本重复性及流间隔非一一对应问题，局部开销和严格可缩短关键路径上限保持unknown。不能把下面的inclusive设备跨度当作纯kernel时间或可直接相加的收益。

## 固定协议与正式结果

共同分支xiaopangV41FlashCppFromSglangtoXllm，数学基线89a2572fc6c58eebdedb353762939ecd0a0bf0cc；v2观测二进制SHA896191bc0a14d8d0e7f5e96c15d090250b3a2dc637416cfec4c374f141d70cdf。模型MooreThreads W4AFP8-mt1017，P/D各8卡、Attention TP8/专家EP8，eager、C1、chunk256、1GiB cache、block128、graph/prefix/overlap/MTP关闭。每模式2基础题＋2048-token功能题，再256/256和2048/256各1暖5正式；前置功能题单列，不隐瞒额外预热。EvalScope1.14.1原版，无修改计时语义。

|模式|输入/输出|TTFT ms|TPOT ms/token|E2E s|输出token/s|
|---|---|---:|---:|---:|---:|
|OFF|256/256|310.801|179.3247|46.0386|5.5606|
|OFF|2048/256|2429.820|180.8120|48.5369|5.2743|
|HOST|256/256|391.281|179.5100|46.1663|5.5452|
|HOST|2048/256|2447.239|180.8194|48.5562|5.2722|
|DEVICE|256/256|398.229|179.7694|46.2394|5.5364|
|DEVICE|2048/256|2454.875|181.1129|48.6387|5.2633|

均为5条正式均值，warmup不计。30条正式请求实际token数全部严格对应；输出速度为总输出token/请求latency之和，QPS及每条原始值在各模式summary.json。该表是观测方法校准，不是一次性能提升。

## 观测开销与正确性

预注册目标为3%。HOST整体TPOT变化+0.1033%/+0.0041%；DEVICE +0.2480%/+0.1664%。短输入抽样步及紧随步157个ITL，HOST平均+0.0848%，DEVICE平均+1.6296%，均未超目标。不能只看每16步抽一次所稀释的总体变化。

长输入SSE有252～255个到达间隔，不能强行映射255个token转移，局部开销unknown。更重要的是OFF对同一2048输入5次出现4种输出，HOST3种、DEVICE4种；不是这次才由计时打开产生的现象。跨模式短输入10组文本全部一致，长输入10组仅1组一致。保留该边界，不将2简单题称完整精度合格，也不对不一致开启漫长全量对分。后续数值/通信改动仍要必要的算子与固定前缀回归，显著新增错误不晋级。

HOST两条短输入慢TTFT约503/506ms，对应P阶段仍295/297ms；客户端相对P多208/209ms，也出现于客户端总耗时减D服务request total。其余差值约16～19ms。根因尚未细分到client/router，不把这些慢点当GPU prefill增加；所有原始点保留。

## 设备与各rank的测量发现

P共256个rank-step采样，D共1536个，967680条scope的event_status全部0。D正式窗口含短输入632、长输入640个rank-step，覆盖8rank和早/中/晚decode。P every2覆盖部分multi-chunk，不宣称每个chunk都有采样。

D原生forward设备span最大rank的中位约176.011/178.221ms，已接近对应179.769/181.113ms服务TPOT；不能用CPUroot148ms与TPOT相差约31ms，便宣称有31ms纯调度可消除。root外还有输入组装、logits、采样、网络等，且存在CPU/GPU重叠。

短输入中段每step inclusive设备范围中位：Attention TP约38.30ms、MoE EP约32.95ms；mHC prepare约26.71ms；W4A8 routed约22.31ms；SWA/sparse周边约19.82ms；压缩cache gather约3.80ms。Attention/MoE等父区间含子区间，不能重复相加。TP/EP相互不嵌套，但其71ms包含依赖/慢rank等待，不是纯链路传输成本。

实际10KiB BF16/20KiB F32、group8、当前MCCL调用路径已核验。同step同层8rank的CPU进入范围偏差中位约0.83/0.93ms（短输入TP/EP），GPU span最短rank约0.425/0.441ms、最长约1.472/1.340ms。CPU到达早晚与GPU跨度相关性中位近0，故不能仅凭CPU入队时间判定GPU就绪；精确GPU跨rank起点与紧关键路径上限尚未知。

## 优先级与下一步

先T01真实小消息数值/全就绪探针，分开判断库效率与慢rank等待；不直接假设71ms全可消除。然后依据结果在T02（SWA/压缩缓存/metadata）与T04（MoE周边/负载）之间选择可验证候选，之后T03、T05、T06、T07、T09、T08。该顺序是已测proxy与可行性的诊断顺序，未知紧上界不伪装成严格收益降序。

T02已核对38次key＋8次index gather可在同forward内按4个源复用至8次；V4原生gather的环绕/页编号语义不同，不能原样套用。T05原生模型当前明确禁止graph，捕获可行范围尚未验证，不能只开flag。以OFF当前基线测算，全部TTFT归零的E2E降幅上限仅约0.675%/5.006%，T09/T08继续位于decode后。每个行为改动仍需两组正式测速、对应trace/火焰图和接受/拒绝结论。

## 失败与资产位置

ToDoList_1的HOST/NFS长输入停滞已完整保留；用户授权7旧PID时6个已自行退出，仅对旧host6899发一次SIGKILL，此授权已消费。ToDoList_2首DEVICE在任何推理前因Mooncake42307绑定失败退出，之后正常TERM，无强杀；P固定端口改为经核验空闲的22000体系后，本目录DEVICE完整通过。不是设备event崩溃，也没有重启/改驱动/停止别人容器。

本地OFF/HOST原始性能与服务日志在../ToDoList_2，DEVICE在本目录；NFS分别对应/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_new_start/ToDoList_2和ToDoList_3。模型只读引用models。各tar.gz收回时均核验大小/SHA，计时大文件保留NFS及本地原始包，仓库提交压缩性能/服务日志与可读摘要，不放模型或二进制。

关键文件：05_trace/critical_path.csv、phase_summary.json、DEVICE_overhead.json、collective_CPU到达与设备跨度.json、模型设备跨度与服务TPOT核对.json；03_正确性/G02相同请求输出核对.json与长输入关闭计时自身重复性核对.json。CPU实测图在../ToDoList_2/06_图，图宽为原生CPUwall，非GPUbusy/ATen火焰图。新的优化仍需独立匹配ATen/GPU trace。
