# Graph+DSpark 全量GSM8K >=95% / 热态TPOT <10ms

本目标由用户在2026-09-25/26明确提出。目录保持为
`/data/nfs_shared/jiaming.wang/xllm0914/xllm-musa`，实验分支
`perf/dspark-gsm95-tpot10-hcpost-20260926`。继承的dirty worktree不清理、不覆盖旧候选。

## 最终结果（2026-09-26 12:12）

**本次用户目标通过：同一Graph+DSpark q6版本，完整GSM8K>=95%，热态平均TPOT<10ms。**
这不是“所有题目正确/没有截断”的声明；额外的零截断质量检查仍为FAIL，证据保留。

| 指标 | 本次起始HC修复基线 | 最终短C4修复版 |
|---|---:|---:|
| GSM8K完整1319题，原SGLang/NeMo评分 | 1269/1319，96.21% | **1272/1319，96.44%** |
| 热态TTFT平均 | 215.290ms | **215.662ms** |
| 热态TPOT平均 | 12.657ms | **9.920936ms** |
| 输出吞吐，8卡聚合 | 78.173tok/s | **99.419tok/s** |
| 输入+输出吞吐，8卡聚合 | 208.461tok/s | **265.118tok/s** |

性能采用全量GSM8K之后、未重启的同一服务，B1、2500/1500、Prefix Cache配置0.65，
块对齐后的实际cached_tokens=1536；2次预热不计，正式6次，计数及输入/输出长度6/6严格通过。
该候选全量之前另一个完整6请求窗口TPOT9.891047ms、TTFT214.750ms，两轮均<10ms。
相对本次起始12.657ms基线，TPOT降低21.62%，输出吞吐提高27.18%；不是不同batch/卡数换算。
**这些是指定热态计数负载的客户端均值，不是冷启动结果，也不是每一道GSM8K的TPOT。**

验收配置：8卡MUSA、TP8/EP1/DP1、N=5（target q6）、Graph+draft、Prefix Cache、
模型`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。
GSM8K与性能均使用此runtime配置，没有用q2/Eager精度替代q6/Graph；GSM8K一次生成，
temperature=0、top_p=0.95、max_tokens=1024、thinking=false，不读取reference修补生成。

最终binary：`build/gsm8k_candidates/short_c4_causal_model_20260926/xllm`，SHA256
`493424fd4b553463be2fe8f3ab8e2169e911e0532c36864b743c5158c388f133`。
对应kernel archive与18/18单测一致，源码/二进制/评分器/原始数据hash核验通过；
全量后对launch记录的AOT文件执行`sha256sum --check --quiet`也通过。
验收针对这个复用已有依赖的独立候选，不把未经测试的全量CMake重编结果当作同一已验收binary。

主要改动：

1. q6 Indexer按query并行，score与radix TopK分离，128线程打分，保留每行因果有效长度。
2. C128复用已有MODEL1 packed cache；C4/C128分别共享稳定metadata，长度计算融合为设备kernel，
   capture ID变化时强制记录首次更新，避免跳过Graph节点。state gather使用8个CTA。
3. 修复短C4 Graph的真实长度问题：原scalar回退将capture时压缩行数当成有效行数，
   更短请求可能读入陈旧行；新路径保留32行容量，但按每个query的runtime位置做causal mask。
   保留Graph+draft，不额外计算Indexer，不改变2500/1500长上下文路由。

验证与边界：

- 完整1319题仍有47错：45个boxed答案不符、1个未输出boxed、1个1024-token截断；全部保留计分。
- 已修复原435/436/437三连提前EOS：本轮完整连续运行均正常答对，且与隔离顺序测试文本一致。
  434的语义循环/截断仍存在，不声称所有剩余错误都已定位到具体算子或模型能力。
- 独立短C4负例检出旧调用最大偏差12.6316；新正例通过FP32/BF16因果参考和Graph请求切换测试。
- `acceptance.json`的`goal_gate=PASS`，但`strict_zero_truncation_gate=FAIL`；
  显式`--permit-scored-truncation`只区分用户百分比要求与额外零截断要求，没有改答案/评分。
- 相同评分口径下，SGLang参考1276/1319，差4题；SGLang-only10、xLLM-only6。
- 仅验收本次B1场景，不外推到B8/B16或其他序列长度。

结果目录：`logs/goal_gsm95_tpot10_short_c4_20260926/`。
关键文件为`acceptance.json`、`xllm.summary.json`、`xllm.scored.jsonl`、`comparison.json`、
`perf_after_gsm_formal/summary.json`及`short_after_long_diagnostic.jsonl`。

复现：先确认并释放本任务占卡（不要reset GPU），在当前容器/目录中运行：

```bash
bash tools/run_dspark_gsm95_tpot10_candidate.sh logs/NEW_RUN_NAME
# 另一终端；脚本会等待服务ready并按相同口径预热和测试：
python3 tools/run_dspark_perf_diagnostic.py \
  --url http://127.0.0.1:19465 --run-dir logs/NEW_RUN_NAME
```

入口固定最终binary哈希及全部主要runtime参数，拒绝继承target-only等诊断开关，
端口忙时不启动新进程。模型/依赖只读复用，不编译Mooncake、不修改其他checkout。
单独调用`replay_dspark_perf.py`时必须传完整`/v1/chat/completions` URL（不同于上述包装脚本）。

12:12测试服务已正常停止，8张卡均已恢复本任务占卡（各约8346MiB，其中张量8GiB），
未reset。占卡目录`logs/goal_gsm95_tpot10_final_holder_20260926/`，parent PID记录在
`parent.pid`、8个child记录在`children.pid`；停止前核对进程身份，不使用全局kill。
代码保留在上述新分支的工作区，未将继承的dirty修改擅自清理或打包提交。

## 基线与验收

- 模型 `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`，8卡TP8/EP1。
- 已验收binary `build/gsm8k_candidates/hc_post_orientation_model_20260925/xllm`，
  SHA256 `bbe400315d657ab7c436b7baf3ec6c0f2f8a5396f56b74392c7c7c59a7715635`。
- 基线Graph+DSpark N=5（target q6），完整1319题原NeMo评分1269/1319=96.20925%。
- 基线B1、2500/1500、Prefix Cache0.65热态：TTFT215.290ms、TPOT12.657ms，
  严格计数6/6，8卡聚合输出78.173tok/s。
- 新版本必须在同一binary/config下同时满足：完整1319题至少1254题正确（>=95%），
  上述热态客户端TPOT严格<10ms，并尽量保持TTFT<250ms。不能将小样本或旧binary精度拼入验收。
- 固定boxed prompt、temperature0、top_p0.95、max_tokens1024、thinking=false，
  原SGLang NeMo评分器。没有reference注入、答案重试、算术修补或评分放宽。

基线证据：`logs/goal_hc_post_fix_full_live_20260925/acceptance.json`。
旧约8ms legacy路径存在已知精度/状态问题，不直接恢复为默认性能路径。

## 首轮实验设计

1. 检查实际decoder/Graph配置，优先尝试已有实现支持的draft宽度调整，减少每个输出token摊到的
   target forward开销。checkpoint训练block size为5，runtime明确允许用户覆盖；扩大宽度是否
   保持接受率和正确性需要实际验证，不能仅按q宽度换算TPOT。
2. 使用同一已验收binary，仅改变N，先做严格计数和短GSM8K诊断；达标候选必须再跑全量。
3. 若宽度调整不足，再对当前Attention/Indexer/C128/state/MoE/collective做实测profiling，
   只对量化确认的瓶颈做私有候选修改，不冻结真实位置、cache或accepted-prefix状态来提速。
4. GPU闲置保持占卡，实验仅释放本任务占卡组，不reset、不修改其他项目或原编译依赖。

本文件创建时目标仍ACTIVE，尚没有TPOT<10ms的新验收结果。

## N=9（target q10）宽度实验

目录 `logs/goal_gsm95_tpot10_n9_20260926/`，同一HC修复binary，只将draft数从5改为9。
完整6请求严格计数6/6、cached_tokens均1536；TPOT **15.547ms**，客户端TTFT357.652ms，
聚合输出63.389tok/s。前两次正式请求的客户端首包仍受冷态影响，不能删去。
日志中实际q10路由已验证，但许多步仍只前进4--6个token，扩大draft没有带来等比例接受数。
该配置不满足性能目标，不继续以它做全量验收；前20题原始响应另存 `first20.jsonl`，
不拿此小样本替代全量精度。

## Profiling入口修复

现有`/start_profile`在SpeculativeEngine返回HTTP500，日志明确为
`start_profile is not implemented for this engine!`，不是MUSA kernel或GPU故障。
新增SpeculativeEngine的start/stop转发，沿target engine的worker clients在实际计算线程采集，
不重复经draft_engine启动进程级profiler。默认仍关闭；复用已有Kineto/MUSA依赖。
候选构建和运行验证待完成，目标继续ACTIVE。

Profiling转发候选已构建并运行，SHA256
`97063aabcfed26c5f72e136f86aafc7b28e177b11eed6d5b8ae106cf4261519f`，
目录 `build/gsm8k_candidates/hc_post_profile_engine_20260926/`。
N5/q6未采集的2次计数探针2/2正确、TPOT12.548ms，接近基线；此2次不是正式6请求验收。

## 已测到的解码热点

在已捕获Graph之后才启动Kineto时，trace没有target Graph内部kernel，不能据此归因于通信。
改为在首次capture之前开启采集后，target kernel出现在trace中；两次采集均能正常start/stop。
原始8-rank trace分别在：

- `logs/goal_gsm95_tpot10_q6_profile_20260926/profile/`（仅部分GPU路径可见）
- `logs/goal_gsm95_tpot10_capture_profile_20260926/profile/`（含target Graph）

第二份rank0 trace的最后8次q6 replay之后的窗口，按kernel累计时间除8：

| kernel族 | 每step累计时间 | 调用数/step |
|---|---:|---:|
| Indexer paged score + radix TopK | 10.220ms | 126 |
| C128 paged Attention | 5.968ms | 20 |
| recurrent state gather | 4.117ms | 124 |
| MoE w13 GEMV+SwiGLU | 3.465ms | 43 |
| MoE w2 GEMV | 1.976ms | 43 |
| MATE decoding metadata | 0.843ms | 21 |

这些是**采集下的kernel累计耗时，不是客户端TPOT，也不是无重叠的端到端分解**；
窗口还包含邻接draft/host准备，通信核中包含等待其他rank的时间。首次capture及大prefill已排除。
完整trace也保留了MUPTI关于可能未返回所有HW时间戳的警告，应以未采集HTTP测试验收最终收益。

当前最先优化Indexer：q6每个C4层发6次score/TopK，各行使用相同cache但各自因果长度。
新增grid-y查询行与独立completion counter，保持每行原有点积/规约/TopK逻辑，
workspace按查询行数和容量分桶保持Graph地址稳定。先补单测与scalar6/batch6微基准，
再决定是否接入模型；目前未宣称模型TPOT已经降低。

首个batch Indexer单测已PASS（q2/q6、int32/int64表、6次不同长度/行映射Graph重放、FP32分数参考）。
同一单测的captured Graph微基准，q6、KV=2500、capacity=1024：
scalar6=546.878us，batch6=430.919us，约减少21.2%。这不足以单独达到模型TPOT<10ms。
工件：`build/batched_indexer_unit_20260926/batched_indexer.log`。

后续增加两个独立opt-in A/B：warp内共享页地址/scale计算，以及共享反量化的Indexer key；
原点积/规约顺序不变。正在单测与微基准比较，模型接入由
`XLLM_DSV4_GRAPH_BATCHED_INDEXER=1` 控制，默认仍走原逐行路径。

地址/反量化共享A/B四种开关均通过正确性，但新kernel即使两个开关都关闭也变慢。
交叉复测：plain scalar6/batch6=516.137/433.169us，address variant=715.594/638.425us。
因此已撤回这两个实验的源码修改，保留独立build/log作为反例，不将慢版本作为候选默认。
plain cache-ops源码与 `batched_indexer_kernel_20260926` 快照逐字节一致。

plain版完整kernel回归15/15通过（含先前的拒绝前缀负例，不等于该负例的集成问题已修复）。
模型候选仅重链已编译且source/header hash一致的模型objects，复用plain kernel archive：
`build/gsm8k_candidates/batched_indexer_model_plain_20260926/xllm`，SHA256
`b8496b05ef618a7b57ac01002bf75ced3e526a7feb0b5e5c874794eb5e7195f8`。
模型实验目录 `logs/goal_gsm95_tpot10_batched_indexer_20260926/`，N5/q6保持不变。
先发4个max_tokens=1的非评测请求预热HTTP响应线程，再按原2500/1500请求0--1预热、2--7正式测试；
正式输入/输出与Prefix命中仍必须逐条验证，不能用HTTP预热结果充当性能或精度。

## 模型级筛选结果与尚未达标项

所有成功性能行均为B1、2500/1500、热态1536 cached tokens、6请求严格计数6/6：

| 组合 | TTFT ms | TPOT ms | 状态 |
|---|---:|---:|---|
| 原HC修复基线 | 215.290 | 12.657 | 全量96.21% |
| Batched Indexer | 215.342 | 12.189 | 前20题NeMo20/20，未全量 |
| Batched Indexer + C128 context split2 | 215.242 | 11.888 | 前20题NeMo20/20，全量进行中 |
| 同上split4 | 215.399 | 11.930 | 性能不再改善；前20评测被提前结束服务中断，不算完整 |
| Batched Indexer + MATE共享metadata（C128默认1） | 213.788 | 12.084 | 无进一步收益；不作为最终配置 |
| Batched Indexer + TP two-shot collective | — | — | formal第3个请求超时，整体FAIL |

TP two-shot仅前两个formal请求约12.1ms不能报为有效性能：后续请求在Graph中hang，
8rank进程仍在且GPU busy，已停止本任务服务释放，不reset GPU。该opt-in不可用于正式运行。
端口30801/30841两次启动前检查失败，未启动模型或发送评测请求；换空闲端口后正常启动，
不当作GPU、kernel或精度错误。

C128 split2让多个CTA重复相同score/softmax计算、分担互不重叠的输出维度，
不改变每个输出的token累加顺序。该新kernel尚需补齐独立CPU/Graph单测和微基准，
此前15/15只对应plain Batched Indexer，不能误用为split kernel证明。
候选：`build/gsm8k_candidates/batched_indexer_c128split_model_20260926/xllm`，SHA256
`5d5e39e06000d9440e4023bb79b5546f7d34219b5d846d7dc1d02acbb3d7ce22`。

`logs/goal_gsm95_tpot10_final_20260926/` 名称中的final仅表示本轮选定的精度评测候选，
**不是目标已完成**：当前TPOT11.888ms仍大于10ms，不能缩小Goal或仅凭>=95%结束。
完整1319题正在单一服务连续运行，用于发现改动的精度/状态问题；之后仍需继续优化速度。

继续研究state gather的单CTA大状态拷贝：保持原block-table最后有效列查找、previous-page恢复和
slot发布语义，尝试2/4/8CTA分担互不重叠的uint4拷贝。只在独立候选/开关实验，尚未验收。

## 09:44 全量评分与后续验证

已完成候选 `5d5e39e0...` 的完整单次1319题：**1271/1319=96.36087945%**，
原NeMo评分器、0请求错误、0截断。比HC基线多2题正确（不代表输出逐位相同）。
与SGLang参考配对：双方正确1263、xLLM-only8、SGLang-only13、双方错35；
原始请求协议/所有prompt token数一致，q6路由26540次，无强制Eager。
目录 `logs/goal_gsm95_tpot10_final_20260926/` 中已保存正式评分和comparison。
精度门槛已过，但11.888ms仍未达到10ms，目标保持ACTIVE。

补齐C128 split和state-copy Graph单测：split1/2/4、copy1/2/4/8均通过已有FP32/BF16独立
causal Attention、动态block-table/长度、state previous-page/空表/holes用例。
单测目录 `build/state_copy_blocks_unit_20260926/`。C128 q6单次约261.041us→201.043us(split2)，
split4约204.710us；8行state-gather约18.363us→16.898us(copy4)，性能收益需模型验收。
未在全量/性能运行中插入GPU单测；这些单测是在确认服务已停止后执行。

下一轮继续尝试N7以及Indexer512线程配置。N9退化不等于N7已被测过，仍必须实测接受率和正确性；
512线程保留原head规约顺序和>=257条目histogram初始化，不更改TopK语义。

N7完整性能门失败：3个formal请求通过，第四个超时，不能只统计成功的前三个。
失败后8rank仍在busy，TERM未完全退出，已核对本任务rank的exe与PID后清理，未reset。
一次单测误在残留rank尚未退出时开始，已停止且结果作废；清理后的结果使用`*_clean.log`。

新Indexer打分/TopK解耦：所有score CTA不再为只有最终选择阶段才用到的radix预留32KiB共享内存；
第二个kernel按query行执行相同radix/映射，依赖同一stream的launch顺序。
同q2/q6 Graph重放、因果长度/映射及独立FP32 score单测均PASS。
`build/indexer_split_score_unit_20260926/*clean.log`微基准：

| variant | scalar6 us | batch6 us |
|---|---:|---:|
| fused1024 | 516.898 | 430.518 |
| fused512 | 445.555 | 328.646 |
| split score256 + TopK1024 | 390.022 | 204.534 |

解耦后batch6比原batch6快约52.5%；尚待同模型性能与全量精度，不能据微基准报TPOT<10ms。

## Split-score模型级进展（10:04）

`indexer_split_score_model_20260926/xllm`（SHA256
`dcc921c08771b4dbcb132017582341252c88830beaf66261140f931defc3981c`）组合
split score、Batched Indexer、C128 split2、state copy8，在正式6请求热态得到
**TTFT215.172ms、TPOT10.789ms、输出91.532tok/s、总吞吐244.085tok/s**，计数6/6，
每次2500/1500且cached1536。目录 `logs/goal_gsm95_tpot10_split_score_20260926/`。
本组合完整kernel测试15/15 PASS，无skip。前20精度仍在运行；不能引用上一个11.888ms候选的
全量96.36%充当这个binary的全量结果。Goal仍ACTIVE，TPOT还需至少降低0.789ms。

10:06补查，该10.789ms版本前20题NeMo20/20。Indexer split打分线程64/128/256/512均单测通过，
batch6分别218.338/195.279/204.706/249.293us，128略优；没有因此跳过最终全量。
C128 warp内共享页地址实验也通过独立CPU/Graph测试，但200.489→240.218us反而变慢，
已从源码撤回，不合入有效配置；保留`build/c128_warp_address_unit_20260926/`失败性能证据。
继续评估保持累加顺序的context循环展开，目标是隐藏逐token访存延迟而非更换数学公式。

循环展开1/2/4/8均通过C128独立FP32/BF16因果/Graph测试，q6/KV2500均值
199.731/197.269/194.190/198.102us，只有小幅收益。

## C128 MODEL1 attention候选

现有C128 cache本来就有MODEL1打包副本，之前q6只消耗BF16 paged版本。
新opt-in `XLLM_DSV4_C128_MATE_VERIFY=1` 允许消费已有MODEL1副本，补齐每步SWA pack，
并用设备上的逐query长度构建C128全因果压缩行和128-token SWA tail；不复用capture时的长度。
逻辑indices/physical indices与长度buffer保持稳定，真实动态值每次Graph replay更新。
不改MATE库/模型文件，不减少压缩状态恢复。超过MODEL1 512个压缩行容量时保留原paged路径。

新增`C128Model1CausalGraphMatchesDense`测试，覆盖0/1/127/128/439/440/1023/2500/4096/32768
附近长度、物理块映射变化、query行切换、真实Graph。独立FP32参考读取打包cache并自行计算
每行完整因果选择集，max_abs<=0.00556、mean_abs<=0.00178（阈值0.02/0.003），不声称逐位相同。
q6 KV2500微基准包括动态索引与长度准备，118.078us；原paged split2约201us。
`build/c128_mate_unit_20260926/all_tests.log` 全16项PASS（仍含单独标注的既有负例）。

模型候选 `build/gsm8k_candidates/c128_mate_model_20260926/xllm` SHA256
`9d90cbb88001fd3cf8a69c2ffed32cb860c03612e00aef6ac9d007f07fe51315`。
`logs/goal_gsm95_tpot10_c128mate_20260926/` 开始模型级验证，尚未宣称达到TPOT<10ms或全量95%。

C128 MATE模型级正式结果：**TTFT216.824ms、TPOT10.376ms、输出95.114tok/s**，
计数6/6，cached1536/2500/1500口径不变；前20题原NeMo20/20。
仍未达到<10ms，也未将上一binary全量分数移植到该版本。

继续尝试按compression ratio分组的MATE调度metadata复用：C4和C128各自维护稳定workspace，
同一步只生成一次，新的Graph capture ID强制包含首次metadata更新节点，不能沿用eager/warmup
的host命中状态而漏录节点。新增unit将C128→C4→C128交错调用、同key两次capture、动态长度及
块映射切换与独立dense reference对比，验证跨group和跨capture不会互相污染。实现仍为opt-in。

分组metadata单测整套16/16通过，C128→C4→C128三次调用约223.546us。
在此基础上增加统一设备长度kernel，精确实现`min(max(kv/ratio,0),topk)`及
`min(max(kv,0),window)`，替换每层多次ATen copy/div/clamp。测试也改为实际调用新长度kernel。
合并候选为`mate_grouped_lengths_model_20260926`，正在构建/验证；此前仅metadata的中间
model未启动，不用它的混合源码状态作为验收证据。builder增加compile前后source hash检查，
禁止源文件在编译中改变后仍生成有效manifest。

10:42合并候选构建完成，SHA256
`c8f726ddc319ddc2d3b7c09b395d17e33536d69604c0d1b3ef98e3cc42de7dac`，
kernel回归16/16 PASS，交错三次MODEL1调用约201.756us。
`logs/goal_gsm95_tpot10_grouped_lengths_20260926/` 开始正式性能/前20题诊断。

该候选第一轮完整6请求窗口：**TTFT213.870ms、TPOT9.909671ms、输出99.543tok/s、
总吞吐265.447tok/s**，计数6/6、cached1536、2500/1500一致，终于跨过速度门槛。
不因此提前完成Goal：10:47开始冻结同一binary/config的完整1319题，之后按预先安排复测热态
6请求；验收脚本显式使用`--min-accuracy-percent 95 --max-tpot-ms 10 --expected-unit-tests 16`，
原评分器不变、不重试/修补答案。当前全量和重复性能仍待完成。

复现入口（当前候选被冻结，不能在其全量运行中改动对应source/header）：

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
bash tools/run_dspark_gsm95_tpot10_candidate.sh logs/NEW_RUN_NAME
# 在另一个终端，服务ready后：
python3 tools/run_dspark_perf_diagnostic.py \
  --url http://127.0.0.1:19465 --run-dir logs/NEW_RUN_NAME
# 该脚本的前20题仅诊断。完整1319题仍需：
python3 tools/eval_gsm8k_sgl_protocol.py \
  --run-dir logs/NEW_RUN_NAME \
  --dataset /data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/dataset_test.jsonl \
  --prompt-yaml /data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/eval_prompt.yaml \
  --url http://127.0.0.1:19465/v1/chat/completions \
  --model DeepSeek-V4-Flash-FP8 --thinking-mode chat
```

脚本只复用固定candidate，默认TP8/EP1/N5/q6/Graph/Prefix、10240 batch-token预算、0.8显存比例；
不重编依赖、不下载模型、不自动释放别人的进程。运行前需释放本任务占卡，切勿reset GPU。
已验证继承`XLLM_DSV4_FORCE_TARGET_TOKEN_ONLY`时拒绝启动，以及端口占用时拒绝启动，
都没有创建新的测试目录或扰动正在运行的服务。

## 全量中途发现的请求切换异常（11:05，尚非最终验收）

同一冻结服务已完成前500题原NeMo诊断：481/500=96.20%，0请求错误。
其中gsm8k-434误解题意并循环讨论，在1024-token上限截断；之后435/436/437三个请求
只输出`Ray`/`We`/`We`，各generated_tokens=2、finish=stop，没有boxed答案，均按错计。
438恢复正常。三次异常的prompt_tokens分别67/62/65，首个q6 verify的KV长度73/68/71，
均走`full_kv=1`的真实Graph；438的prompt长度108、首次verify长度114，同样先走full-KV。
全部43层请求边界state/workspace reset日志存在，因此不能仅凭现象称“漏了整个reset”。

此前11.888ms候选及12.657ms基线的434均在619token正常结束，435--437均答对；
SGLang参考434用686token答对，后续三题也答对。当前候选是否存在长输出后的Graph/cache
污染仍需顺序复现，不能将这三次EOS直接归因于算术kernel。该诊断不修改正式响应、不重试替换，
全量按原计划继续，之后先完成既定热态复测，再进行额外顺序诊断。

验收脚本历史上还附加了“全量0截断”的强检查；这比本次用户的>=95%更严格。
如最终该检查失败，必须同时保留其FAIL证据、完整原评分及截断/提前停止明细，
不能删除异常题或悄悄改评分让旧检查通过。

## 11:24 冻结候选的完整结果及后续修复边界

`mate_grouped_lengths_model_20260926/xllm` 完整1319题原NeMo为
**1265/1319=95.90598939%**（补充numeric-only为1242/1319，不混用评分口径）。
与参考配对：both-correct1259、xLLM-only6、SGLang-only17、both-wrong37；
所有prompt token数相同，1319次单次请求，无重试/答案修补。
同一服务全量后的6请求热态复测：**TTFT215.988ms、TPOT9.931096ms、
输出99.316951tok/s、总吞吐264.845203tok/s**，严格计数6/6、cached1536；
比本次12.657ms起始基线的输出吞吐提高27.05%。首次窗口9.909671ms也保留。

用户的两个数值门槛已由同一binary/config的完整数据满足，但旧验收脚本的附加
`no_errors_or_truncation`失败（1次截断，0传输错误），其`acceptance.json=FAIL`原样保留。
并且435--437确实出现异常提前EOS；Goal暂不关闭，继续验证/修复短C4 Graph因果长度问题。
下面的额外顺序回放只作诊断，不能替换1319题正式结果或抬高上述准确率。

额外顺序回放435,436,437,434,435,436,437,438没有再次出现三连EOS，434也正常终止。
这说明仅重放这8次不足以复现完整运行的状态/物理页历史，不能据此声称问题已消失。
原记录在`short_after_long_diagnostic.jsonl`，没有替换正式失败样本。

短C4独立负例已复现：`build/c4_static_length_diagnostic_unit_20260926/negative_control.log`。
同一Graph从capture_kv112切到replay_kv67，旧scalar API继续使用28个压缩行，而真实只允许16个；
将后12个陈旧行填成101后，与正确causal输出最大差异12.6316。这个测试PASS表示成功检出
旧接口调用方式的限制，**不是修复已通过**，也还不是三连EOS全部因果链的模型级证明。

新候选将C128的逐query动态causal paged核泛化为ratio4/128（保留旧C128 ABI），
短C4 Graph opt-in `XLLM_DSV4_GRAPH_CAUSAL_SHORT_C4=1`：先写回本步compressed proposal，
按每个query的device KV长度遮挡未来/前请求行，32个compressed行只作容量。
不启用额外Indexer，不关闭Graph/draft，不改2500/1500长上下文路由；待新unit及模型回归。
编译在独立`short_c4_causal_{kernel,model}_20260926`目录，已启动8卡各8GiB占卡，无reset。

新kernel archive SHA256 `fa3e0008f6e833e24dedac6b29e5ed69121b62204291fa99b8760a5403fb87e6`。
`build/short_c4_causal_unit_20260926/all_tests.log` **18/18 PASS，无skip**，包括新增的旧路径负例
和`CapturedShortC4MasksStaleRowsAfterRequestSwitch`正例；后者覆盖FP32/BF16、int32/int64表、
6个query的逐行长度、物理页/请求行变换、未来缓存行大值污染，容差分别2e-5/2e-3。
原C128、HC post、Batched Indexer等测试也通过。测试运行时本任务8GiB内存占卡仍在，
因此日志中的微基准不作为新的性能结论；正式模型性能前会完整释放占卡。
未变的Indexer cache_ops对象在新旧archive中SHA256相同：
`1ea4fe8ed9c88df28aebe326d954de925645fb400f628a75ea379426b802cb16`。

11:36新模型候选SHA256 `493424fd4b553463be2fe8f3ab8e2169e911e0532c36864b743c5158c388f133`，
目录`logs/goal_gsm95_tpot10_short_c4_20260926/`。完整6请求热态为
**TTFT214.750ms、TPOT9.891047ms、输出99.721622tok/s、总吞吐265.924327tok/s**，
计数6/6、2500/1500、cached1536不变。短C4修复未使该窗口的性能回退到10ms以上。
本次已确认`[DSV4_SHORT_C4_CAUSAL]`实际启用，仍是N5/q6、Graph+draft，未强制Eager。
尚待同一新binary/config全量1319题及全量后6请求复测，不能沿用上一binary的95.91%。

新候选前20题原NeMo20/20。顺序诊断中434仍1024token截断，但435/436/437在它前后分别
生成40/75/47token并答对，**三题前后文本逐字相同**，没有上轮2token提前EOS；434按错计，
并未通过改变prompt或增加max_tokens修补。11:38启动新binary/config全量1319题，不做重试。

本次明确区分用户的数值目标和“零截断”这一额外质量要求：审计新增显式
`--permit-scored-truncation`，只允许使用原评分器评分的截断响应留在1319分母内，不改变任何
答案/评分规则。`checks.no_errors_or_truncation`和`strict_zero_truncation_gate`仍真实报告，
请求错误仍必须为0，所有截断ID单列；不把有截断的模型描述为完全无异常。
之前的FAIL证据不重写。默认不传此参数时仍要求零截断。

新全量控制脚本的compare阶段漏传两个required文件参数，预期会在生成/CPU评分完成后
退出；这不影响生成内容。已将正确的显式参数及后续热态复测/审计保存在本run的
`POSTPROCESS.sh`，届时从该阶段继续，**不会重新请求或覆盖1319题**。

11:49新版本完整连续评测已经过原故障段（非替换式重试）：434依然1024token截断且按错计，
435/436/437紧随其后正常生成40/75/47token并全部答对，438也正常；不再出现上一版本的
三连2token EOS。这些响应与短顺序诊断中的对应文本逐字一致。该结果支持修复解决了
短C4捕获长度复用导致的请求切换问题；不据此声称434的语义循环或全部GSM8K错误已解决。

## 12:07 新候选全量完成

同一新binary原NeMo **1272/1319=96.43669447%**，0请求错误，1题截断（434，计错）。
补充numeric-only为1251/1319=94.84457923%，其简化提取不支持部分单位/百分比/LaTeX表达，
不是用来替换SGLang原评分器的另一个验收口径。剩余47错包括45个boxed答案不符、1截断、
1个缺少boxed答案（758）。与SGLang配对：both-correct1266、xLLM-only6、SGLang-only10、
both-wrong37，准确率差0.30326个百分点。请求payload及113369个prompt tokens与参考一致，
评测时间窗内q6 Graph路由26528次，forced-eager为0。

生成/原CPU评分均完成一次，没有重跑或替换数据。后处理控制脚本先因误写参考token计数文件
源目录而退出，已复用上一轮同数据集核验过的文件；比较阶段通过后，性能脚本误把服务根URL
作为streaming API，返回空内容/无usage，而不是模型推理失败。这些空请求保存在
`perf_after_gsm_warmup/`（FAIL），不计入任何有效TTFT/TPOT。正确API
`/v1/chat/completions`的预热单独记录在`perf_after_gsm_warmup_api/`，之后正式6请求仍为
`perf_after_gsm_formal/`；命令及失败日志均保留，模型服务没有重启。
