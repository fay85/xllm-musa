# DSpark GSM8K 固定100题：精度问题定位、修复与性能恢复实验说明

日期：2026-09-25。本文整理 09-23 至 09-25 的 q2 Eager / q6 Graph 合并实验，
依据当前源码、原始评测结果、构建清单和服务日志，不是重新跑了一轮测试。
范围不包含更早的非 DSpark 全量 GSM8K 算术/语义审计实验。

## 1. 结论：这次精度主要解决了什么

核心问题不是“模型不会算数学”，而是 **Graph+DSpark 多 token 验证时，Attention
读到的状态、有效长度和可见历史，可能与当前请求、当前 query 的语义不一致**。

重点修复了以下几类问题：

1. **状态没有随真实 decode 位置更新。** 旧快速 q6 路径存在 compressor/indexer
   状态冻结；部分压缩状态 kernel 把 capture 时的 host 位置固定进了 Graph。
2. **复用了不该复用的 Graph。** query 数量相同，不代表计算图相同：跨越 128-token
   SWA 窗口后，full-KV 与压缩/Indexer 路径会变化；C4/C128 本轮输出行数也可能变化。
3. **把容量或旧长度当成当前有效长度。** Indexer 有效压缩行数、SWA 映射等必须读
   replay 时的设备输入，而不能一直沿用第一次 capture 的长度。
4. **q6 各行的因果可见性不正确。** C128 的新压缩行不能无条件给全部六个 query 使用；
   前面的 query 不应看到后面 proposal 才完成的压缩结果。
5. **缓存发布与请求隔离不完整。** 更新 BF16 SWA 不等于更新 MATE 读取的 packed SWA；
   请求边界的 compressor/indexer 状态、decode workspace 和 persistent metadata 也需要管理。

最终在同一版本、同一启动配置的 **q6 Graph+draft** 上得到：

| 项目 | 结果 |
|---|---:|
| 固定 GSM8K100 | **96/100，零请求错误** |
| 该固定集合的前20题 | **19/20** |
| B1、2500/1500、热 Prefix Cache：TTFT_AVG | **215.340 ms** |
| 同口径 TPOT_AVG | **11.9828 ms** |
| 正式计数请求、精确输入输出 usage | **6/6 通过** |

这不是“q2 测精度、q6 测速度”的组合成绩，也不是读取 reference 后修正答案的成绩。
最终没有启用短上下文 target-verify Eager 保护来代替 q6 Graph。

**因果判断边界：** 上面是源码缺陷、算子单测和端到端回归共同支持的解释。
并非每一项都有完整100题的单因素消融；不能把某一行代码单独说成“贡献了多少个百分点”，
也不能说所有 MATE、MoE、MHC 或其他数值问题均已排除。

## 2. 实验对象与统一口径

| 配置项 | 本轮最终配置 |
|---|---|
| 代码目录 | `/data/nfs_shared/jiaming.wang/xllm0914/xllm-musa` |
| 分支 | `fix/dspark-accuracy93-perf-20260922` |
| 基础提交 | `c7d07f5a676a6c3dac28ad10a1df037b4d945438`，修复包含未提交的工作区改动 |
| 模型 | `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt` |
| 设备与并行 | MUSA 8卡，TP8/EP1，global B1 |
| 推测解码 | DSpark，`num_speculative_tokens=5`，target q6 |
| Graph | 开启 compact verify，短上下文 Eager 保护关闭 |
| 调度 | `max_tokens_per_batch=10240`，chunk=10240，schedule overlap=false |
| Cache | Prefix Cache 开启，block size=128，KV dtype=auto，memory utilization=0.8 |
| GSM8K 采样 | seed=20260914，temperature=0，top_p=1，max_tokens=2048 |
| Prompt | `baseline`，与历史评测脚本的 `reasoning` 文本一致；无 few-shot |
| 模板参数 | thinking_mode=chat，reasoning_effort=high，enable_thinking=false |

### 2.1 “固定100题”的准确含义

从 GSM8K test 的1319题中，使用固定 seed 抽样100题并排序。
**不是原始数据集的前100题。** 本文“前20题”也指这组固定样本的前20题。
`dataset_index` 为零基索引。

- 数据：[gsm8k_test.jsonl](../logs/gsm8k_eval_20260914_1925/gsm8k_test.jsonl)。
- 数据 SHA256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
- 完整样本索引及参数：[fixed100.config.json](../logs/goal_shared_address_graph_20260925/fixed100.config.json)。
- 原始响应：[fixed100.jsonl](../logs/goal_shared_address_graph_20260925/fixed100.jsonl)。

评分从响应中提取最终数字并规范化后 exact match，优先取最后的 `####`，其次 boxed，
再取最后一个数字。最终答对的96题都包含显式 `####`，不是依赖模糊文本截取凑出的分数。
reference 仅用于评分，不进入模型请求，不用于选择重试、修改 prompt 或修补输出。

### 2.2 q2、q6 与 batch 的区别

- q2：一次 target verify 有2行，通常对应 N=1 的 draft 设置。
- q6：一次 target verify 有6行，本轮对应 N=5。
- **q6 不是六个并发请求**；本轮仍是 global B1。
- “Graph 开关打开”不等于 target verify 真走了 Graph；历史默认保护可能把 compact
  verify 导向 Eager，因此必须核对 capture/replay 和 forced-Eager 日志。

## 3. 实验过程：从精度/性能分离到同配置验收

### 3.1 历史路径对照

以下数值已与相应原始 summary 核对。20题、100题不能直接混算；各历史候选也不是
严格的连续单因素实验，表中不把分数变化解释为某个补丁的独立贡献。

| 阶段 | 路径或变化 | 样本 | 结果 | 说明 |
|---|---|---:|---:|---|
| 精度基线 | q2 Eager target verify | 固定100 | 95/100 | 可作为精度基线，不能证明 q6 Graph 精度 |
| 旧快路径 | legacy q6 Graph | 同一固定集合前20 | 13/20 | 约8 ms 性能不能与 q2 的95/100拼成联合结果 |
| 当前实现早期版本 | current Attention q6 Graph | 前20 | 14/20 | 错题含3、23、128、235、244、255 |
| 恢复完整状态更新 | q6 不再冻结 compressor/indexer state | 前20 | 18/20 | 精度改善，但历史记录 TPOT 约50 ms |
| 批量 C4 更新 | 用批量 kernel 替代逐行状态更新 | 前20 | 18/20 | 仅减少部分更新开销，还未达到性能目标 |
| 动态 C4 位置/Graph key | replay 读取真实位置 | 前20 | 17/20 | 仍错122、235、244，说明不是单一问题 |
| 最终版本 | 动态状态、分支区分、逐行因果与性能优化 | 前20 / 固定100 | 19/20 / 96/100 | 最终在同一 q6 配置上联合测性能 |

原始记录：

- [q2 固定100基线](../logs/goal_repro_q6_legacy_q2_accuracy_20260923/eval100/summary.json)。
- [legacy q6 前20](../logs/repro_q6_legacy_gsm8k20_20260923/eval20/summary.json)。
- [current q6 前20](../logs/repro_current_q6_gsm8k20_20260923/eval20/summary.json)。
- [完整状态更新前20](../logs/repro_q6_unfrozen_state_20260923/eval20/summary.json)。
- [批量 C4 前20](../logs/repro_q6_batched_state_v2_20260923/eval20/summary.json)。
- [动态位置前20](../logs/repro_q6_dynamic_state_keyed_v3_20260923/eval20/summary.json)。

历史曾出现97/100的记录，但后续路由核查确认其属于 Eager target verify / 短上下文
Eager 保护结果，不能用来证明当时的纯 q6 Graph 已达97%。本文最终采用的是原始日志
确认无 compact-verify Eager 回退的96/100。

### 3.2 三题复现集：122、235、244

这三题用于快速重现、筛选修改方向，不代替固定100题验收。

| 候选变化 | 三题结果 | index 244 | 结论 |
|---|---:|---|---|
| packed SWA 发布修复候选 | 1/3 | 答17，生成1714 token，121.162 s | 只修缓存发布不够 |
| 上一候选 + 并行状态块查找 | 1/3 | 仍答17，1714 token，69.219 s | 三份输出逐字不变，属于性能修复 |
| 动态 C128 更新 + Graph 分支/形状键 | 2/3 | 答20，177 token，7.535 s | 复现错误得到改善，但仍不足以替代100题 |
| C128 逐行因果过滤候选 | 固定100为96/100 | 答20，168 token | 完成纯 q6 Graph 的固定100精度验证 |
| 再复用 paged row address | 固定100仍96/100 | 仍答20，168 token | 100份输出逐字不变，进一步恢复性能 |

前三阶段也存在一次 prefill budget 从1024改为10240的配置差异；三题 prompt 均为
97--112 token，在两种预算下都无需切块，但仍不能把此对照写成完全单因素实验。
单题耗时包含生成长度等影响，**不是同口径性能 benchmark 的 TPOT**。

index 244 是薯条题。按题意应满足：

`24 - 5 - 2×5 + x - (5-3) = 27`，所以 `x = 20`。

修复前曾输出17并继续长篇生成；修复后得到20。这个结果是完整状态/Graph 路径修复后的
端到端证据，不是向模型加入了上述算式，也不是已有逐层日志证明“某一层单独贡献了这道题”。

对应目录：`logs/goal_swa_publish_graph_20260925/`、
`logs/goal_parallel_state_graph_20260925/`、`logs/goal_dynamic_c128_key_graph_20260925/`、
`logs/goal_causal_c128_graph_20260925/`、`logs/goal_shared_address_graph_20260925/`。

## 4. 精度问题与修复机制

### 4.1 恢复必要的 compressor/indexer 状态更新

DSV4 不只有普通 K/V，还维护压缩器和 Indexer 的 recurrent state。
旧快速 q6 路径会冻结相关状态；如果只沿用此前状态，后续 Attention 看到的压缩历史
就可能与已处理的 token 不一致。

处理方式：允许 q6 执行完整状态恢复、更新和写回；批量更新减少逐 token kernel 调用。
Graph capture 的 warmup/FFI 记录阶段不能提前推进这些状态，避免真实 forward 之前状态
已经改变。相关入口：`freeze_compact_graph_compressor`、`use_persistent_compressor_state`、
`use_batched_decode_state`。

证据：完整更新的历史前20题从14/20改善为18/20；只更新首行仍有明显问题。
边界：这不等于 accepted-token rollback 已经解决，见第8节。

### 4.2 将实际位置、KV长度、Indexer有效行数改为 replay 时读取

Graph 固定的是计算拓扑和 buffer 地址，不应该固定“当前已经处理到第几个 token”。
如果把 capture 时的 `start_position`、`q_abs_start` 等 host 值作为后续 replay 的
不变参数，就可能写错状态行、漏掉新压缩历史，或使用错误的 SWA 映射。

处理方式：

- C4/C128 sequence kernel 从 persistent device position 读取实际位置。
- Indexer 的 Graph 入口使用 device `dynamic_kv_len` 更新有效行数。
- MATE SWA 映射随同一实际长度刷新；workspace 容量仅作容量，不替代有效行数。

相关 kernel：`deepseek_v4_compress_decode_sequence_graph`、
`deepseek_v4_indexer_topk_graph`。
真实 capture/replay 单测覆盖位置变化、127/128、439/440及块映射变化。

### 4.3 Graph key 必须区分实际 host 分支与输出形状

只按 q6 或 C4 行数区分 Graph 不够：

- 完整历史还在 SWA 窗口内时，full-KV 分支与超过窗口后的 compressed/Indexer 分支不同。
- 同样六行 query，可能产生不同数量的 C4、C128 压缩输出。
- Indexer score buffer 的容量分桶也会随上下文增长变化。

处理方式：Graph key 纳入 query bucket、Indexer capacity、C4输出数、C128输出数和
full-KV标志；Attention 与 runtime 使用同一个 full-KV 判定 helper。
绝对位置本身仍由设备输入动态提供，而不是为每一个位置建立一张新图。

注意默认 full-KV threshold=2048 还要与模型 SWA window=128 共同判定，
并不表示“2048以内全部按完整历史 Attention 处理”。旧的短 Eager 保护阈值是另一件事。

### 4.4 C128 q6 必须逐行保持因果可见性

此前 paged fallback 把本轮产生的压缩尾部交给所有 query，并携带 capture 时的
压缩历史长度。这既可能使前面的 query 看到未来压缩结果，也可能使后续 replay 漏掉新增历史。

示意：以下是零基位置上的 C128 边界，不是 index 244 的逐层 trace。

| query 的绝对位置 | 可见原始 token 数 | 此 query 可见的已完成 C128 行数 |
|---:|---:|---:|
| 126 | 127 | 0 |
| 127 | 128 | 1 |
| 128 | 129 | 1 |

处理方式：先发布本轮压缩输出到 paged cache，再从每行 device KV length 计算有效
C128 行数与 SWA 可见范围。共享容量只限定空间，不能让无效行参与 Attention。
新的 `deepseek_v4_paged_attention_verify_c128` 把 q6 六行放在同一次 GPU launch 中执行，
每行独立计算因果范围，并沿用 FP32 reduction。

该路径与独立 dense Attention reference 比较，覆盖真实 Graph replay、不同表类型与
数值类型；不是只检查是否 crash。最终包含此修复的候选在固定100题达到96/100。

### 4.5 packed SWA 发布、请求隔离与 capture stream

BF16 paged SWA 与 MATE MODEL1 packed SWA 是不同 tensor。
原先 Graph decode 跳过 pack 的条件不适用于消费 packed SWA 的 batched C4 verify；
而 draft 的 accepted-context 写回不能代替 target packed cache 的写回。

处理方式：在该 target 路径补齐 packed cache 发布；请求边界按配置清理相关状态/
workspace，Graph persistent metadata 也执行重置。同时修复 capture 期间跨 stream
event 依赖及 native MUSA capture stream 的使用。

这些是正确运行的支撑修复，但 packed SWA 单独修复后三题仍只有1/3；不能宣称它单独
解决了全部精度问题。最终长计数后重复20题、输出逐字不变，是一次有限的状态切换验证，
不是所有请求退出/加入、多 batch 场景的完整证明。

## 5. 哪些修改只提升性能，没有改善数学答案

为了恢复精度而直接开启全部状态更新，会增加状态恢复、压缩、Indexer、Cache 发布的开销。
本轮不是再关掉这些必要操作来追求速度，而是在保留语义的前提下优化其实现。

### 5.1 并行查找状态块

旧 `state_gather` 每次让一个线程扫描8193列 block-table。8193是预分配容量，
不是当前请求已经使用8193个块。该操作在多个层、多个状态 tensor 上重复发生。

改成256线程并行找到最后一个有效的**逻辑列**，再取对应物理块；不能简单取最大物理块ID。
空表、空洞、非单调物理映射以及跨页时读前一页状态的语义保持不变。

| GPU 微基准：8193列、8状态行 | 每次耗时 |
|---|---:|
| 串行查找 | 1449.05 μs |
| 并行查找 | 18.314 μs |

约79倍是这个 kernel 的加速，不是整模型加速。
三题A/B输出逐字一致，分数仍是1/3，明确说明这是性能修复，不是这三题的精度修复。

### 5.2 C128 Attention 复用分页地址

旧实现对每个输出维度重新计算同一 token 的物理页地址，重复执行64位块号/页内偏移计算。
现在在 score 计算阶段每个 token 只解析一次地址，存在共享内存中供 value 累加复用。
每次 replay 仍重新读取真实 block table；没有把地址固化成 capture 时的映射。

| GPU 微基准：q6、8 heads、D=512、BF16、320次 captured 调用 | 每次耗时 |
|---|---:|
| 重复计算地址 | 968.674 μs |
| 共享 row address | 260.929 μs |

softmax 与累加顺序未改变。最终100份模型输出与此前96/100候选逐字一致。

### 5.3 同口径端到端性能恢复

两版均为完整固定100题+正式六次热态窗口；不是把冷启动、无缓存或三题耗时与热态TPOT比较。

| 候选 | GSM8K100 | TTFT_AVG | TPOT_AVG | 计数 |
|---|---:|---:|---:|---:|
| C128 因果修复版 | 96/100 | 213.762 ms | 14.3852 ms | 6/6 |
| 再加共享分页地址 | 96/100 | 215.340 ms | 11.9828 ms | 6/6 |

TPOT 降低约16.7%，精度输出不变。最终六次请求 TPOT 为11.9340--12.0442 ms，
TTFT为211.372--220.319 ms；每次正式请求也都满足13 ms/250 ms门槛。

最终 output throughput=82.518 tok/s，input+output throughput=220.047 tok/s，
分母是六次顺序请求的完整墙钟时间；均为整个8卡服务聚合值，不是单卡值。
输入吞吐包含缓存命中的输入 token。Prefix Cache 配置率为0.65，实际每次命中1536 token，
不能把0.65直接当作1536/2500的实测比值。

没有达到旧版约7.85 ms的速度；达到的是本次约定的“正确率>=95/100且TPOT<13 ms”。

## 6. 如何验证，不把单测或小样本当成端到端验收

### 6.1 Kernel / Graph 单测

最终测试日志：[all_tests.log](../build/gsm8k_candidates/c128_shared_address_final_unit_20260925/all_tests.log)。
共12项报告PASS、无跳过：**11项正向测试 + 1项刻意复现风险的负例**。

| 覆盖项 | 检查内容 |
|---|---|
| Graph 分支策略 | full-KV/压缩分支、C4/C128输出数量不同不能复用同一key |
| Q1/Q2 MATE 对照 | 与独立 dense reference 比较 |
| C4 状态更新 | sequence更新与逐token参考比较，检查状态、score和压缩输出 |
| 真实 capture/replay | 固定buffer地址，更改位置、长度、block table，不只跑Eager |
| C128 因果 Attention | FP32/BF16，int32/int64 table，跨128边界；逐行对照含sink的dense参考 |
| 动态 Indexer | 有效长度变化、TopK范围、无效行padding及物理映射 |
| SWA及packed发布 | rolling边界、当前q6写回、过期cache负对照 |
| Capture event | 避免旧Eager event给capture制造跨stream依赖 |
| 状态块查找 | 空表、孔洞、非单调物理块、容量1/3/257/8193、跨步更新 |
| 标量旧路径 | q1奇数压缩历史长度，验证共享内存对齐与dense结果 |
| 拒绝前缀负例 | 证明不恢复accepted state会产生差异；**不是证明rollback已修复** |

C128 reference 检查的绝对误差上限为FP32 2e-5、BF16 2e-3。
其他测试使用各自的比较容限或精确相等条件，不应统一宣传为“所有算子零误差”。

### 6.2 完整100题与运行路径核实

- 对比历史 q2 基线：100个样本索引、题目、reference 和 prompt token 数全部一致。
- 最终请求全部使用 q6 Graph+draft，N=5；服务日志有实际 q6 replay。
- 没有 `target compact verify forced eager` 记录；不是仅看 `enable_graph=true`。
- 最终二进制、源文件/policy header、kernel archive、MATE/AOT清单均核对哈希。
- 原始响应不丢弃，不对错误题定向重试后覆盖分数；请求错误数为0。

### 6.3 性能测试后再检查数学输出

流程为：固定100题 → 两次计数预热 → 六次正式计数 → 重复固定集合前20题。
最后仍为19/20，20份输出与前面的固定100题记录逐字一致，错误仍只有235。
这验证了本次“长计数到短数学”的请求/Graph状态切换，不能外推至全部并发生命周期。

最终数学、性能和联合检查：

- [固定100题原始结果](../logs/goal_shared_address_graph_20260925/fixed100.jsonl)。
- [六次正式性能汇总](../logs/goal_shared_address_graph_20260925/formal_2_7/summary.json)。
- [性能之后重复20题](../logs/goal_shared_address_graph_20260925/post_perf20.jsonl)。
- [联合验收结果](../logs/goal_shared_address_graph_20260925/joint_goal_audit.json)。

联合验收明确检查>=95/100、TTFT<250 ms、TPOT<13 ms，以及相同二进制、q6路径、
严格计数、精确2500/1500 usage和热态缓存命中。仅计数工具中的 `gate: PASS` 不包含
这些延迟门槛，不能单独拿它宣告目标达标。

## 7. 剩余4道错题是什么

以下全部按原始 reference 严格记错，没有因人工认为“有道理”而修改分数。

| index | 模型答案 | reference | 原始输出中的可观察问题 |
|---:|---:|---:|---|
| 235 | 11 | 10 | 把20个朋友再加John作为21人，按每人4片向上取整 |
| 333 | 0 | 100 | 生成了 `1350+2100=3350`，正确应为3450，导致两报价被算成相等 |
| 952 | 1800 | 360 | 按题面8 AM到11 PM解释成15小时；reference对应3小时口径 |
| 1038 | 6 | 4 | 用新增收入50,000/年计算回收时间，reference使用总收入75,000/年 |

这四题也都出现在历史 q2 Eager 的错误名单中。该基线另错255，最终q6版本的255答对，
因此从95/100变为96/100；不是所有错误消失了。

这些描述可以解释文本的直接失分原因，但**不能仅凭文本宣布底层数值计算全无问题**。
比如333是生成文本的算术错误，不等于证明某个GEMM kernel做错了 `1350+2100`。

## 8. 算子精度、MATE/mutlass与未解决事项的边界

### 8.1 已有证据支持什么

本轮主要支持的是 Graph/runtime 与 Attention/cache kernel 的**语义正确性问题**：
状态冻结、capture时标量未动态更新、错误的图复用、cache发布缺失、因果可见性错误。
这些问题会把错误的输入送给数值上正常的算子，也会产生显著的最终logits/答案差异。

本轮没有更换模型权重、重新量化checkpoint，也没有通过替换MATE/mutlass库来取得成绩。
C4的MATE sparse路径仍启用；C128使用带逐行动态长度的paged Attention，
不是把整个模型统一改成全历史dense fallback。

单测通过说明覆盖形状上的结果满足容限，完整100题通过说明这组端到端请求达标。
两者结合仍不等于MATE/mutlass、所有shape、所有设备的数值正确性已经完全证明。

### 8.2 accepted-prefix状态恢复风险仍在

负例步骤：先处理六个proposal，只接受第一个，再直接沿用完整六行后的recurrent state
处理下一步，与“只处理已接受前缀再继续”的参考结果比较。

- C4压缩输出最大差异：0.187227。
- C128压缩输出最大差异：0.00613724。

该测试刻意期望观察到差异，其PASS只表示成功复现了风险。
它不是实际 `DFlashWorkerImpl` 完整调度循环的修复证明，也未被归因到上述某道GSM8K错题。
后续需单独验证真实接受/拒绝循环和checkpoint恢复；不能将它写成“所有rollback问题已解决”。

### 8.3 不能外推的范围

- 96/100不是全量1319题成绩，更不是此前非DSpark审计后的96.06%。
- B1 q6不是B6，也没有在本次验收中证明B8/B16长序列正确性。
- 2500/1500热态指标不能代替8192/1024、30K/1024、无缓存冷启动或PD指标。
- 前述现象不能仅凭本轮结果归因于GPU硬件、MoE/MHC或某个ABI版本。

## 9. 代码改动入口与作用分类

| 文件/入口 | 作用 | 分类 |
|---|---|---|
| [deepseek_v4_attention.cpp](../xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp)：`freeze_compact_graph_compressor`、`use_batched_decode_state` | 恢复状态更新、动态C4/C128调用 | 精度语义 |
| 同文件：`skip_mate_swa_graph_decode_pack` | 更新target消费的packed SWA | 缓存一致性 |
| 同文件：`dynamic_c128_verify` | 新压缩行发布后逐query进行因果Attention | 精度语义兼顾性能 |
| [deepseek_v4_graph_policy.h](../xllm/core/layers/musa/deepseek_v4/deepseek_v4_graph_policy.h) + [musa_graph_executor_impl.cpp](../xllm/core/runtime/musa_graph_executor_impl.cpp)：`dsv4_graph_shape_key` | 统一full-KV判定、区分Graph形状/分支，维护persistent metadata | 精度语义 |
| [deepseek_v4_dsa_cache_ops.cu](../xllm/core/kernels/musa/deepseek_v4_dsa_cache_ops.cu)：sequence graph / dynamic Indexer | replay时读真实位置和长度、capture-safe缓存操作 | 精度语义/运行稳定性 |
| [deepseek_v4_dsa_attention.cu](../xllm/core/kernels/musa/deepseek_v4_dsa_attention.cu)：`deepseek_v4_paged_attention_verify_c128` | q6独立有效长度、动态压缩历史可见性 | 精度语义兼顾性能 |
| 同文件：`row_addresses` | 复用分页地址，保持累加顺序 | 性能 |
| [deepseek_v4_sp_pad.cu](../xllm/core/kernels/musa/deepseek_v4_sp_pad.cu)：`last_valid_state_block` | 并行查找状态块 | 性能 |
| [单测文件](../tests/core/kernels/musa/deepseek_v4_mate_sparse_attention_test.cpp) | reference、capture/replay、映射与负例验证 | 验证 |
| [联合验收工具](../tools/audit_dspark_merge_run.py) | 同时检查精度、路径和性能门槛 | 验证 |

构建时一并重编当前Attention、DSA metadata、MUSA Graph executor和DFlash，避免
把新Attention与旧runtime对象混用。表中未把没有独立收益证据的所有历史diff都说成精度主因。

## 10. 复现与工件定位

最终已验证二进制：

```text
build/gsm8k_candidates/current_graph_shared_address_20260925/xllm
SHA256 5785ab8228debdd5e5e9a9b24be91a53ba650f913d947d225cd0b11052a43f52
```

完整启动、依赖复用和私有增量构建命令见
[联合验收与运行文档](DSV4_DSPARK_MERGED_ACCEPTANCE_20260925.md)。
源码仍包含未提交工作区改动，不能只根据基础HEAD复现；构建目录保存了源码、policy header、
编译/链接命令和对象/archive/binary哈希。现有Torch/MUSA、MATE/AOT等依赖复用，
不覆盖先前二进制或库，不重编单机不需要的Mooncake。

服务按该文档启动后，在另一终端使用新输出目录，依次运行：

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
# 改为本次服务的日志目录；工具拒绝覆盖已有实验结果。
GSM8K_RUN_DIR=logs/repro_gsm8k100_new
python3 tools/eval_gsm8k_goal.py \
 --dataset logs/gsm8k_eval_20260914_1925/gsm8k_test.jsonl \
 --url http://127.0.0.1:19260/v1/chat/completions \
 --model DeepSeek-V4-Flash-FP8 \
 --output "$GSM8K_RUN_DIR/fixed100" --samples 100 --seed 20260914 \
 --max-tokens 2048 --reasoning-effort high --prompt-style baseline --timeout 180
python3 tools/replay_dspark_perf.py \
 --url http://127.0.0.1:19260/v1/chat/completions \
 --input-dir logs/goal_repro_q6_legacy_perf_20260923 \
 --output-dir "$GSM8K_RUN_DIR/warmup_0_1" --first 0 --last 1
python3 tools/replay_dspark_perf.py \
 --url http://127.0.0.1:19260/v1/chat/completions \
 --input-dir logs/goal_repro_q6_legacy_perf_20260923 \
 --output-dir "$GSM8K_RUN_DIR/formal_2_7" --first 2 --last 7
python3 tools/audit_dspark_merge_run.py "$GSM8K_RUN_DIR" \
 build/gsm8k_candidates/current_graph_shared_address_20260925 \
 logs/goal_repro_q6_legacy_q2_accuracy_20260923/eval100/results.jsonl
```

`GSM8K_RUN_DIR` 必须与服务启动的run directory相同，联合工具需要其中的启动清单和rank0日志；
若换了二进制，也要同步修改audit的candidate目录。不要直接覆盖上述历史验收目录。

### 避免重犯的实验问题

- 漏开 `XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY` 会实际走Eager；不能只看Graph总开关。
- 开了总Graph开关、或仅counting通过，都不能证明GSM8K达标。
- 一些历史启动FPE是master/TCPStore端口被占，一次OOM是占卡程序尚未释放；不能据此判定ABI或GPU故障。
- 缺少有效客户端文本的服务端计时不是合格性能结果。
- `replay_ms` 是一次target验证的GPU时间，不等于客户端每token的TPOT；中间还有draft、接受率等因素。
- 历史记录中存在不同batch预算、首次capture请求和库版本混用，应以各次manifest为准，不做未经控制的单因果推断。

## 11. 最终可交付结论

本轮证明：在指定模型、8卡TP8/EP1、B1和固定测试协议下，当前 **同一q6 Graph+draft版本**
可以达到固定GSM8K100的96%及热态TPOT<13 ms、TTFT<250 ms。

主要精度工作是让Graph下的状态、动态输入和因果范围正确；主要性能工作是去掉状态块查找、
分页地址计算的重复开销。它不是更换权重、关闭全部MATE或参考答案修补，也不是全场景正确性证明。
