# Graph+DSpark GSM8K >=96% / 热态TPOT <9ms / TTFT <240ms

## 最终验收结果（2026-09-26 16:14）

**用户目标通过：同一binary、同一N6/target-q7 Graph+DSpark配置，完整GSM8K >=96%，
B1 2500/1500 Prefix0.65热态平均TPOT <9ms、平均TTFT <240ms。**

| 指标 | 旧验收基线 N5/q6 | 本次最终 N6/q7 |
|---|---:|---:|
| GSM8K完整1319，原SGLang/NeMo评分 | 1272/1319，96.44% | **1275/1319，96.66%** |
| 全量后6请求 TTFT_AVG | 215.662 ms | **220.108 ms** |
| 全量后6请求 TPOT_AVG | 9.920936 ms | **8.723131 ms** |
| 聚合输出吞吐（整组8卡，不除卡数） | 99.419 tok/s | **112.568 tok/s** |
| 聚合输入+输出吞吐（整组8卡） | 265.118 tok/s | **300.183 tok/s** |
| 严格计数正确性 | 6/6 | **6/6** |

TPOT降低12.07%，聚合输出吞吐提高13.23%。全量前另一完整窗口为TTFT219.089ms、TPOT8.749611ms，
也计数6/6通过。两窗口均为实际2500输入/1500输出、cached1536，未混用不同宽度的精度和性能。
**上表是客户端热态均值，不是冷启动/任意长度/多batch结果，也不是逐请求延迟上限。**
全量后6请求TPOT8.641--8.797ms；TTFT213.693--242.916ms，其中1次TTFT242.916ms超过240ms，
没有丢弃该请求或挑另一窗口替代，均值仍220.108ms。

- 最终binary：`build/gsm8k_candidates/grouped_state_model_20260926/xllm`，SHA256
  `8228891e3c9a4994c38131a869738a477a6fdef8a581c846ee693a26f77b5c87`。
- 原始证据及审计：`logs/goal_gsm96_tpot9_n6_groupedstate_20260926/acceptance.json`，`goal_gate=PASS`。
  `responses.raw.jsonl`/`requests.jsonl`均1319条、唯一且每题一次；无reference注入、修复、重试替换。
- 精度和性能共用MUSA8卡TP8/EP1、0731-FP8-mt、N6/target-q7；target Graph、draft eager。
  全量区间q7 Graph路由24944条，无强制Eager compact，prompt tokens与参考全部一致。
- 算子单测24/24通过：`build/grouped_state_unit_20260926/all_tests.log`，无skip。
  最终AOT依赖哈希复核全部通过：`ops_artifacts.after_validation.check`；未修改外部MATE/MUTLASS。
- 剩余44题未答对，含截断119及无boxed的758。`strict_zero_truncation_gate=FAIL`：
  本次完成的是用户要求的准确率门槛，不是0截断/100%正确。所有错题均保留并按原规则评分。
- 补充简化numeric matcher为1253/1319=95.00%，与正式原NeMo差22题（单位/LaTeX等格式）；
  它不是本次新换的评分规则，正式评分器与历史SGLang及旧xLLM基线hash完全一致，详见`xllm.summary.json`。
  已逐项检查22题：20题boxed携带百分号，另2题为`95 \\text{ minutes}`和`34 \\text{ hours}`，
  与参考数字分别一致；没有因此替换生成文本或修改评分器。
- 16:15正常停止本任务模型服务（无残留xllm进程），未reset GPU。
  已以容器内独立后台占卡程序恢复8卡，每卡8346MiB；记录在
  `logs/holder_gsm96_tpot9_final_20260926/`，含parent/children PID和holder日志。

复现命令见下方“已验收版本复现”。下文按时间保留试验过程，早期“未通过/进行中”均为当时状态。

新目标开始于2026-09-26，分支`perf/dspark-gsm96-tpot9-20260926`。
只修改本checkout，保留前一目标工件及dirty worktree，复用依赖、独立候选构建；不用GPU时占卡，
不reset GPU。不把历史结果、小样本、Eager/target-only或答案修补结果替代最终验收。

## 已核实基线与新验收标准

- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`，MUSA8卡TP8/EP1，
  Graph+DSpark N5/target-q6。
- 基线binary：`build/gsm8k_candidates/short_c4_causal_model_20260926/xllm`，SHA256
  `493424fd4b553463be2fe8f3ab8e2169e911e0532c36864b743c5158c388f133`。
- 原NeMo完整GSM8K：1272/1319=96.4367%；热态TTFT215.6616ms、TPOT9.920936ms，输出99.419tok/s。
  原始证据`logs/goal_gsm95_tpot10_short_c4_20260926/`，不覆盖。
- 新要求：**同一最终binary/config全量>=1267/1319（96%），平均TPOT严格<9ms、TTFT严格<240ms**。
  B1、2500/1500、Prefix Cache0.65（实际1536 cached tokens），预热0--1，正式2--7计数6/6。
  GSM8K后须在同一服务再次完成正式性能窗口。
- GSM8K保持原boxed prompt、temperature0、top_p0.95、max_tokens1024、thinking=false、原SGLang/NeMo。
  错题/截断保留并按原评分器计分，不要求全部1319题正确，不用reference影响生成。

## 首轮工作

1. 核对保留基线、占卡进程、分支及依赖工件。
2. 对当前9.92ms版本在首次Graph capture前采集profiling，定位优化后的真实瓶颈。
   旧12.66ms版本的profile仅作历史参考，采集时间不作性能验收。
3. 优先消除重复内存搬运、kernel launch和metadata准备，不冻结动态cache/state或削减因果历史。
4. 候选经算子回归、6请求性能和短诊断后，才进入同版本全量1319题与性能复测。

创建时Goal ACTIVE，尚未得到<9ms的新候选结果。

## 当前基线热点及第一轮候选

Profile目录`logs/goal_gsm96_tpot9_profile_20260926/`，采集前没有q6 capture。
必须按最后8个`musaGraphLaunch`的correlation过滤kernel，不能只用CPU时间截窗口：
后者会误计C128边界新Graph的warmup/capture准备，导致MoE等调用数膨胀到约3倍。

| target q6 kernel族 | 采集中累计ms/step | 次数/step |
|---|---:|---:|
| MoE W13 / W2 | 3.420 / 1.980 | 43 / 43 |
| state gather | 3.064 | 124 |
| Indexer score / TopK | 2.987 / 0.838 | 21 / 21 |
| TP one-stage reduce | 2.784 | 43 |
| HC pre GEMM / fuse | 1.628 / 1.727 | 86 / 86 |

目标Graph kernel累计44.641ms/step，非客户端TPOT，也不含全部draft/host开销；通信含等待。
这不支持直接按上述数值声称<9ms，仍需未采集的模型测试。

第一轮候选：state逻辑列索引max改成warp内规约+两次block barrier（不改表语义或扫描范围）；
MoE仅改变W13输出tile N=4/8/16、W2 N=32/64/128，K分块/累加顺序保持不变。
新单测对比原4/32分块的activated/routed/final三个张量，覆盖BF16/FP16、动态路由/无效expert、
query行变换及实际Graph replay；GPU微基准只用于选择候选，不能替代整机性能。

第一轮19/19单测通过，MoE9种分块的activated/routed/final在测试中均逐位相同。
4/128微基准149.236us对比原4/32的156.108us，state gather15.726us；但模型6请求
`logs/goal_gsm96_tpot9_moe_tiles_20260926/`仅得到TTFT215.900ms、TPOT9.955784ms、计数6/6，
**没有显著整机收益，更未达到9ms**。前20诊断完成，不以其替代全量。

进一步检查最后两个稳定Graph之间的GPU区间：约26.077ms（含profiling开销），kernel累计
10.291ms；CPU侧有166次copy、73次clone、69次contiguous等。draft/上下文准备也有明显开销。
不直接把这些采集数值换算为正常TPOT。

下一轮尝试HC pre的小token融合finish：保持原MATE split-K prenorm，复用本地已有q1 finish的
逐token索引能力，为2--8token合并HC finish和RMSNorm，保留两次BF16舍入边界。opt-in
`XLLM_DSV4_HC_PRE_BATCHED_FINISH=1`，需独立比较post/comb/output及动态Graph，再看模型精度。
不改HC post方向，不修改MATE库。该候选当前尚未通过新增测试。

HC单测首次因缺少`libtilelang.so`动态库搜索路径而abort，属于测试启动环境，与算子数值无关；
补充已存在的`XLLM_TILELANG_LIB`和TileLang lib路径后通过。2/5/6/8token、4组动态输入，
output最大差7.63e-6、post差0、comb最大8.94e-8；保留首次失败日志，没有修改或重编TileLang。
该微基准包含Graph launch开销，不能据其声称模型性能提升。

draft SAS目前使用`selected_kv_attention_reference`：gather逻辑KV历史再切尾窗口，随后多次
FP32张量操作。新增opt-in `XLLM_DSV4_FUSED_PAGED_SAS=1`直接从分页BF16 cache计算相同的
非因果context+diffusion-block窗口；仅当前eager draft路径启用，不把capture时的host KV长度
误用于动态Graph replay。新增测试对比独立FP32非因果参考、页映射/query变化、31/127/128/129/
439/2505/32768长度及576字节行宽；包含显式“所有query都能看到块尾”的对照。
SAS候选正在编译，尚未验证模型指标。

13:46 `build/paged_sas_unit_20260926/all_tests.log`整套21/21 PASS（包含HC融合及SAS新测试）。
SAS单位测试按每个固定长度分别capture，重放时改变query/物理页；不声称旧capture能动态改变
host标量KV长度。模型绑定只在eager draft启用该核，每step用真实KV长度重新launch。
模型candidate `paged_sas_model_20260926/xllm` SHA256
`ad95af225c0f8ca109c69719428c36afeeb7970292f6a3b4c7eca2a1ccd0bb8e`，
`logs/goal_gsm96_tpot9_sas_hc_20260926/`开始未采集的6请求性能与前20题诊断。
MoE tile参数回到原4/32（第一轮无显著收益），开启HC batched finish和fused paged SAS。

该组合6请求计数6/6、TTFT214.476ms、TPOT9.873560ms，前20诊断完成；仍未达到<9ms，
不能以微基准或旧完整准确率结束Goal。

下一轮独立配置实验N6/target-q7（HC batched finish关闭，fused paged SAS开启），使用同一
`paged_sas_model`binary。前轮N7/N9的失败/退化发生于更早实现，不代表当前N6已经测试过。
但必须先验证计数与状态，再进行同q7配置的完整1319题和性能复测；不把q6精度移植给q7。
新run：`logs/goal_gsm96_tpot9_n6_20260926/`。

N6/q7计数6/6，TTFT215.828ms、TPOT9.249162ms、输出106.223tok/s；前20题原NeMo20/20。
尚未达到<9ms，不以20题替代96%全量验收。
发现cache中已有32x32x32 HC prenorm模块，可只在2--8token选择此较小tile，避免64行tile的
大量padding；仍需验证ABI和数值。只复用现有`.so`，不重编/修改外部MATE。

32tile新增q7对照及整套21/21通过，post差0、comb<1e-7、output<8e-6。
N6/q7模型`logs/goal_gsm96_tpot9_n6_tile32_20260926/`计数6/6，TTFT215.714ms、
TPOT9.129077ms、输出107.518tok/s，仍差0.13ms，不能按四舍五入说已达到9ms以下。

下一候选优化greedy验证：纯greedy不构建未使用的dense draft probability；在无penalty/无logprobs
的greedy模式复用target sampler已算出的token IDs，将first-mismatch及后缀mask融合成一个整数核。
保留random/mixed/logprobs及诊断target-only路径，不修改概率或接受条件。新增独立顺序参考测试
覆盖N1/5/6/7/9/31、B1/B4、所有首次拒绝位置、全接受及实际Graph输入变化，待验证。

14:21融合greedy候选整套22/22单测PASS（功能检查时有本任务内存占卡，微基准不作性能验收）。
模型`greedy_verify_model_20260926/xllm` SHA256
`1db2b94485b70e99efcb46c50ce7e663b9f8cf5ec30d682c1fd258c23f9b11c4`。
新run `logs/goal_gsm96_tpot9_n6_greedy_20260926/`沿用N6/q7、small-HC tile、fused SAS，
只增加greedy验证融合；尚待模型验证。
评测比较/审计脚本增加记录真实target query width及显式expected-draft-tokens，保留旧q6字段，
不会把q7路由标成q6。评分器、问题和正确率门槛没有更改。

融合greedy的N6/q7计数6/6，TTFT216.346ms、TPOT9.150896ms；日志确认融合分支实际执行，
但与9.129ms前轮相比无明显提升。继续小幅增加proposal宽度至N7/target-q8做独立探针，
保持当前的SAS/HC small tile及全部动态状态修复；历史旧N7有过hang，故先完整计数回归，
若停滞只清理本任务进程，禁止reset。

N7/q8本轮稳定完成计数6/6，并非hang；但TTFT214.181ms、TPOT10.189522ms明显退化，
不采用该配置。继续回到N6/q7，测试只作用于本服务进程的OMP/MKL线程数1，避免小CPU tensor
准备时线程池开销；不修改机器环境或模型数学公式。启动manifest新增记录这两个环境变量。

OMP/MKL=1的N6结果TTFT214.362ms、TPOT9.160496ms，没有明显改善。
转向合并Graph compressor投影：C4复用已有SP打包权重，四次GEMM变一次；C128加载时打包
KV/score两组权重，两次变一次。复用`decode_compressor_kv_projection_`已有容量，向支持row-stride
的compressor核传递列视图，不新增capture分配或split-copy。prefill和不匹配的形状保持原路径。
`joint_compressor_model_20260926/xllm` SHA256
`98751879493047634d9237204787d2495556d55f7549506c1557dea96c2bcd31`。

新投影单测初次失败在测试程序直接调用MUSA `torch::equal`提取bool时：独立C++进程没有注册
pinned scalar allocator。不是GEMM数值断言失败。改为显式拷贝到CPU后做相同的逐位比较，
不放宽容差，旧失败日志保留；其余22项已通过。

CPU逐位比较消除环境错误后，严格equal仍失败，证明该GEMM合并**不是bitwise等价**。
独立FP64诊断：q7/C4 packed-vs-separate最大0.00390625、均值3.14e-7、17920元素中8处不同；
C128最大0.0078125、均值2.17e-6、7168元素中11处不同。对FP64参考，packed最大0.0078125，
separate最大分别0.0078125/0.00390625。属于形状改变后的累加/舍入差异，不能称为原精度完全不变。
原`joint_compressor_unit_cpucompare_20260926/all_tests.log`的FAIL不改写。
新测试改名为明确的BoundedBf16Roundoff，并增加独立FP64全矩阵参考，使用BF16相对误差尺度
rtol0.008及近零取消误差atol2e-5；这不是GSM8K评分放宽。该候选只有在全量原评分>=96%后才可采用。

上述数值检查仍有1处失败：C128、q7、frame1，compressed output最大差0.00012207且近零元素
超过预设误差边界。因此**撤回C128联合投影**，恢复原两次GEMM，亦移除新增C128权重副本。
不继续提高容差。保留两版FAIL日志；新候选仅启用C4联合投影，其main KV/score、state、
compressed output继续要求逐位一致，Indexer允许前述小舍入差并仍需全量96%门槛。
新构建`joint_c4_model_20260926`与`joint_c4_unit_20260926`独立保存，未覆盖旧工件。

15:20 C4-only整套23/23 PASS，main projection/score、state/score-state、compressed output逐位一致；
独立FP64及Indexer有界舍入检查也通过。binary SHA256
`6b15f349b3740b54390a54df1479f2d90a77ed028bbfbff7820784857fcfe182`。
N6/q7新run `logs/goal_gsm96_tpot9_n6_jointc4_20260926/`开始完整6请求性能+前20题诊断。

该run得到TPOT9.143787ms，但**新投影分支并未执行**：实际`decode_workspace_tokens_=16`，
原buffer只有16*1024元素，小于q7所需7*2560，容量guard回退。单测原用32行未覆盖这项绑定。
在构造阶段仅为开启联合投影的C4预留ceil(16*2560/1024)=40行，增加约48KiB/层，
不在capture分配，不改class布局。单测同步使用真实16-token容量公式并检查容量覆盖。
`joint_c4_capacity_model/unit_20260926`独立构建；必须确认新分支日志后才谈收益。

15:33 capacity 修正后整套23/23通过，无skip，日志
`build/joint_c4_capacity_unit_20260926/all_tests.log`。模型 SHA256
`7cd96c575f5312e2f858f0bd44266ceca5437e9b5e2a8098fd20cee09c08f893`，
run `logs/goal_gsm96_tpot9_n6_jointc4_capacity_20260926/`，N6/target-q7，
仍保留独立候选及旧基线。只有该配置实际运行通过原1319题评分后才声明新版本准确率。

容量修正后已观察到`[DSV4_JOINT_DECODE_PROJECTION] ratio=4 columns=2560`，
正式6请求计数6/6，TTFT217.044ms、TPOT9.031935ms、输出108.962tok/s。原NeMo前20题20/20，
仅诊断，尚未全量；仍未达到严格<9ms。

下一候选`XLLM_DSV4_GROUPED_STATE_GATHER=1`合并同层2/4种独立state恢复，
grid.y为cache角色，各角色保留自己的block table、slot和previous-page source；
与旧scalar调用共享完全相同的整数lookup/uint4复制代码，不改变数学累加或有效历史长度。
加入同Graph复用地址下的2/4角色、容量1/257/8193、混合i32/i64表、position i32/i64、
空表、holes、物理页重排及128/129/440边界测试，除了旧路径还比较独立CPU oracle。
独立构建`grouped_state_kernel/model/unit_20260926`，编译期间已切换本任务占卡程序。

15:40 grouped state整套24/24单测PASS（含独立CPU oracle、无skip），日志
`build/grouped_state_unit_20260926/all_tests.log`。功能测试时本任务8GiB/卡占卡仍在，
其微基准时间不作为性能结论；端到端性能前会释放全部本任务占卡进程。

15:42 模型候选编译完成 SHA256
`8228891e3c9a4994c38131a869738a477a6fdef8a581c846ee693a26f77b5c87`。
释放本任务占卡后启动`logs/goal_gsm96_tpot9_n6_groupedstate_20260926/`，
N6/q7 + fused SAS + small-HC tile + fast greedy + C4 joint projection + grouped state。
先验证实际分支与固定6请求性能，再进行同二进制全量；尚未宣称通过新Goal。

15:46 真实分组state与C4 joint分支均有运行日志。固定正式6请求计数6/6、
TTFT219.088995ms、TPOT8.749611ms、输出112.322922tok/s、总吞吐299.527793tok/s，
实际输入/输出2500/1500，cached=1536；首次同时满足性能门槛。
即将冻结此二进制与N6/q7配置做完整1319题及全量后性能，不将旧版96.44%代入。

15:47 原NeMo前20诊断20/20；全量1319正式开始，同服务、不重启、单次请求、不重试替换。
`tools/run_gsm8k_full_goal_validation.py`会自动完成原评分、逐题对比、同服务性能复测及验收审计。
完整运行期间不再修改模型源码、二进制或配置。

## 已验收版本复现

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
bash tools/run_dspark_gsm96_tpot9_candidate.sh logs/repro_gsm96_tpot9_xxx
# 另一终端：等待ready后自动预热0--1，测量2--7，最后做20题诊断
PYTHONDONTWRITEBYTECODE=1 python3 tools/run_dspark_perf_diagnostic.py \
  --url http://127.0.0.1:19466 --run-dir logs/repro_gsm96_tpot9_xxx
```

脚本固定binary、基线manifest及launcher哈希，清除继承的XLLM诊断开关；
默认HTTP19466/master31230，可用第2/3个参数替换端口。启动前须确认本任务旧服务/占卡已正常退出，
脚本不会kill其他任务、更不会reset GPU。模型为0731 FP8-mt；target Graph+eager draft，
DSpark N6（target q7）、TP8/EP1、max concurrent1、Prefix Cache、10240 prefill chunk、mem0.8。
不能将这套B1热态2500/1500指标外推到冷启动、其他长度或多batch。

相对9.92ms验收基线，本次采用：draft SAS分页融合、小token HC 32tile、整数greedy verify融合、
C4联合投影、分组state恢复、N5→N6；保留全部动态compressor/indexer更新及短C4因果修复。
没有启用试验性MoE tile覆盖或HC batched finish；C128联合投影已撤回。MATE/MUTLASS等外部库未修改。

15:49 全量途中原NeMo前100诊断98/100，与旧验收基线同100题完全同分，错题均为37/93。
该prefix100评分仅监测，无重试/答案替换，完整验收仍待1319题。
首轮6个正式请求分别TPOT8.859077/8.800391/8.716461/8.603428/8.739175/8.779133ms，
标准差0.08719ms；TTFT212.101--226.573ms，6个请求各自均满足<9/<240。

15:57 原NeMo前500诊断487/500=97.40%，13错（12 boxed mismatch、1截断错），0请求错误；
继续全量不更改长度/采样/模型，不替换任何错误。旧434本轮818tokens stop且boxed=5正确，
但仍有反复推导，不能称所有生成漂移都已解决。prefix500也不替代全量验收。

分组state优化开启前后，固定前20题生成文本20/20完全相同（对照joint_c4_capacity候选），
不是只比最终答案；结合逐位状态单测，仅支持本次覆盖范围内语义一致，不能替代全量。
16:07 前1000原NeMo969/1000=96.90%，31错（29 mismatch、1截断119、1无boxed），0请求错误。
该检查为partial诊断，完整1319及全量后性能仍在进行。

16:12 全量原NeMo评分完成：1275/1319=96.6641395%，0请求错误、1截断（119，计错）。
44错为42 boxed mismatch、1截断、1无boxed（758）；不改评分规则，不宣称所有剩余错误已定位到算子。
旧基线1272→1275，11题变对、8题变错，净+3，不是逐题严格不回退：
变对298/304/409/434/515/675/754/777/831/934/1078；
变错368/494/510/969/1003/1174/1176/1187。
历史SGLang同协议1276/1319：双方对1267、仅xLLM对8、仅SGLang对9、双方错35；
该比较只涉及准确率，不对比两者不同concurrency的性能。
GSM8K时间窗口内实际q7 Graph路由24944条，无强制Eager compact；所有prompt token数与参考一致。
全量后2次预热已通过（TPOT8.826899ms、TTFT218.134586ms），正式6请求仍在进行。
