# DSV4 HC post 残差混合方向修复

本轮从case55的同输入跨框架Prefill张量比较出发，发现前两层之后差异放大，继续检查HC。
这是新的逻辑错误证据；不再将该类问题笼统归因于 MATE Attention 数值差异。

## 完整1319题结果（2026-09-25 23:30）

**xLLM Graph+DSpark q6：1269/1319 = 96.20924943%。**
同一服务、同一binary从第0题到第1318题单次连续运行，无中断拼接、无重试、无答案修补。
使用同SGLang的boxed prompt、temperature=0、top_p=0.95、max_tokens=1024，
显式thinking=false对齐其默认chat模式，使用原NeMo评分器，未修改grader规则。

| 项目 | 结果 |
|---|---:|
| 本次完整NeMo正确数 | 1269/1319（96.20925%） |
| SGLang历史参考 | 1276/1319（96.73995%） |
| 差距 | 7题，0.53071个百分点 |
| 严格纯数字辅助评分 | 1247/1319（94.54132%） |
| 请求次数 / 唯一题目数 | 1319 / 1319 |
| 请求异常 / 输出长度截断 | 0 / 0 |
| 完整prompt token计数不一致 | 0 |
| 双方均正确 / xLLM-only / SGLang-only / 双方均错 | 1261 / 8 / 15 / 35 |
| GSM8K时间窗口内实际q6 Graph路由事件 | 26500 |
| target compact verify forced Eager | 0 |

22题的纯数字与NeMo评分差异来自百分号/单位等格式，原SGLang grader本身就按相同规则处理；
不是为了提分新改的规则。原始生成不变，`xllm.summary.json`记录了两种分数及grader源码hash。
本次未开启同步式Graph计时，所以不能用缺少`GRAPH_TIMING replay_ms`认定Graph未运行。
`comparison.json`只统计GSM8K请求的起止时间内Graph路由证据，不借用计数性能预热日志。

优先错题集的变化：

- 相对最初1132/1319基线，原149个SGLang-only中 **147个恢复**；总计153题变对、16题变错，
  净增加137题。不能隐藏新出现的错题。
- 相对采样修复后1179/1319诊断基线，原104个SGLang-only中 **103个恢复**；
  总计106题变对、16题变错，净增加90题。该旧基线跨两个binary拼接，不能当成新版本连续验收。

剩余15个SGLang-only（0-based dataset index）：
`120,121,122,123,368,394,425,741,758,831,926,951,1046,1092,1174`。
其中120--123是极短未完成输出，758数值133正确但没有boxed；其余10题为boxed答案不匹配。
这些均保留为错误，达到96%不等于所有精度问题已消失，也不证明所有算子/所有batch都无问题。

原始数据：`logs/goal_hc_post_fix_full_live_20260925/`，包括
`responses.raw.jsonl`、`requests.jsonl`、`xllm.scored.jsonl`、`xllm.summary.json`、
`comparison.json`、`disagreement_cases.jsonl`、`launch_manifest.txt`。
原始responses的SHA256为
`58197c6c9857d62079fc815d774a0fbe8cc5d36ba852595835a403ca03852a38`。
参考SGLang为Graph target-only、client32；本次xLLM为Graph+DSpark、B1串行，
这是完整数据集精度对照，不据此进行两个框架的速度对比。

相同binary、同一服务的全量后热态性能复测已完成：B1、2500/1500、Prefix Cache 0.65、
Graph+DSpark q6，预热0--1后正式variant2--7，**TTFT 215.290 ms、TPOT 12.657 ms**，
严格计数6/6，usage每次2500/1500、cached_tokens每次1536。
8卡整体输入吞吐130.288、输出吞吐78.173、总吞吐208.461tok/s（均未除卡数）。
输出/总吞吐约为历史同口径82.518/220.047的 **94.73%**，达到TTFT<250ms、TPOT<13ms。
记录：`perf_after_gsm_formal/summary.json`。这轮在全量结束前就已排好，不是根据结果挑选快请求；
全部6次都纳入统计。首轮TTFT362.422ms的慢样本仍完整保留，见下文。

下文保留定位过程和逐阶段前缀诊断。

### 剩余极短输出的补查（不改全量成绩）

全量和性能结束后，仍使用同一服务、同一binary、同一请求payload：

- 单独重放120、121、122、123、758：4/5正确，只有758仍为未boxed的`$133`。
- 按119→120→121→122→123→124→758重放：5/7正确，119是SGLang也错的题，758仍未boxed。
- 120--123在两组补查中均正常回答，暂未稳定复现全量过程中的提前结束；
  不能宣称已修复其具体原因，更不能用这4个正确重跑替换原始全量错误。

记录为 `post_full_short_solo.*` 和 `post_full_short_order.*`；原始全量仍是1269/1319。
后续若继续消除残余问题，应保留冷启动、完整请求顺序、Graph buffer/块分配历史进行复现，
而不是仅把119题当作充分触发条件。

### 验收核查与资源

`tools/audit_gsm8k_full_goal.py` 已对单一binary/source/kernel hash、1319次唯一请求、
原评分器hash、原始生成未改变、Graph路由、六次严格性能结果、14项kernel回归逐项核查，
`logs/goal_hc_post_fix_full_live_20260925/acceptance.json` 为 **PASS（20项）**。
原AOT依赖列表的 `sha256sum --check --status ops_artifacts.sha256` 返回0，未覆盖原编译依赖。
本轮达到约96%及保留主要性能的目标，不等于15个SGLang-only都已解决。

模型服务和评测进程已结束，未reset GPU。通过 `10.20.34.25/wjm_xllm` 的独立docker exec
启动 `tools/reserve_idle_musa.sh`，8张卡各保留8GiB张量，实测每卡8346MiB。
占卡日志/PID记录：`logs/goal_hc_post_gpu_reservation_20260925/`。
下一次实验前需先核对该组父PID及命令再发送TERM，由脚本清理其自身子进程，不能按通配符杀其他任务。
启动器已验证会在本占卡组仍运行时拒绝启动，不会自行杀占卡或reset。

## 正确公式与旧实现

HC pre 的 `comb` 布局是 `[token, input_lane, output_lane]`。
SGLang `DeepseekV4DecoderLayer.hc_post()` 参考公式和其实际MUSA TileLang实现一致：

```text
output[t,j,d] = post[t,j] * branch[t,d] + sum_i(comb[t,i,j] * residual[t,i,d])
```

xLLM旧 `dsv4_hc_post_kernel`、`dsv4_hc_post_2d_h4_kernel` 都读取了 `comb[t,j,i]`。
HC pre 中未发现补偿转置。初始embedding复制到各HC lane时，该错误可能被掩盖；
各lane分化后，不能以comb双随机、各行列和约1推导出矩阵可以转置。

修复位于 `xllm/core/kernels/musa/deepseek_v4_dsa_cache_ops.cu`，只纠正两个kernel
的comb索引；不改变shape、workspace地址、launch次数或Graph拓扑。

## 独立单测

新增 `DeepseekV4HCPostTest` 两项：

- 非对称双随机矩阵（循环置换与identity混合），FP32/BF16、HC=3/4、T=6、D=4096；
- HC=4真实Graph capture/replay，包含output与residual共用存储、动态更新混合矩阵。

CPU reference按上述输入轴求和，不复用GPU kernel的索引代码。旧archive两项均失败，
HC=4最大绝对误差约0.31：
`build/hc_post_orientation_old_20260925/result.log`。

修复kernel donor保存在：
`build/gsm8k_candidates/hc_post_orientation_kernel_20260925/`。
依赖/旧archive只读复用，不覆盖原有编译结果。

修复后两项测试均通过，HC=3/4与FP32/BF16的max_abs都是0，Graph原地复用3次重放通过。
完整同一套kernel测试14项均PASS（其中有1项既有的拒绝前缀负例验证，不是集成正确性证明）：
`build/hc_post_orientation_fixed_20260925/all_tests.log`。

source-complete模型候选：
`build/gsm8k_candidates/hc_post_orientation_model_20260925/xllm`，SHA256
`bbe400315d657ab7c436b7baf3ec6c0f2f8a5396f56b74392c7c7c59a7715635`。
不启用之前的gate/HC/FP8/packed activation诊断切换，保持原性能路径。

## 验收边界

case55 的同 token IDs Prefill 探针已完成：首 token 从旧版1449（`28`）变成2581（`We`），
与 SGLang 一致。完整 Graph+draft boxed 回答恢复为正确答案14。

同输入的跨框架 Prefill 张量 relative L2 对比：

| 位置 | 修复前 | 修复后 |
|---|---:|---:|
| Layer0 Attention输入 | 0.002814 | 0.002814 |
| Layer1 Attention输入 | 0.039394 | 0.020111 |
| Layer2 Attention输入 | 0.165561 | 0.037536 |
| Layer42 Attention输入 | 0.616190 | 0.168480 |
| Layer42 Attention输出投影 | 0.368330 | 0.091349 |

原始证据：`logs/goal_hc_post_fix_case55_20260925/first_token.jsonl`、
`sglang_layer_comparison.json`。这是相同真实输入的诊断，不代表所有层已逐位对齐。

固定困难20题（不是随机样本）按原 NeMo 评分器：旧版1/20，修复后 **19/20**；
SGLang 在同20题也是19/20。剩余case37三者均受题目“卖掉所有玩具”措辞影响，
新xLLM与SGLang均回答0，而数据集期望2。18个原先xLLM错/SGLang对的题已全部恢复。
该批次没有请求异常、没有长度截断；`diagnostic20.summary.json` 明确标记为非全量诊断。

本节为当时的验收过程记录；完整1319题结果见文首。
已关闭 Prefill dump，重新启动同一修复binary进行独立全量验证。
`logs/goal_hc_post_fix_full_20260925/` 在加载期间进程退出、未产生评测请求，
保留原日志，不计入准确率或性能。重新核查进程不存在、端口未占用、GPU空闲后，
启动 `logs/goal_hc_post_fix_full_live_20260925/`；当前尚不能宣称全量达到96%。

## 同一修复binary的首轮性能

2026-09-25 22:50--22:53，TP8/EP1、B1、2500/1500、Prefix Cache 0.65、
Graph+DSpark q6，预热variant0--1后测variant2--7，完整保留6个请求：

| 指标 | 实测 |
|---|---:|
| 严格计数正确 | 6/6 |
| 输入/输出usage | 每次2500/1500 |
| cached_tokens | 每次1536 |
| 客户端TTFT平均 | 362.422 ms |
| 客户端TPOT平均 | 12.663 ms |
| 聚合输出吞吐（8卡整体） | 77.543 tok/s |
| 聚合输入+输出吞吐（8卡整体） | 206.782 tok/s |

工件：`logs/goal_hc_post_fix_full_live_20260925/perf_formal/summary.json`。
前两次客户端TTFT为639.239/673.734ms，后四次为218.328/214.510/213.562/215.158ms；
不删除慢样本，也不据后四次单独宣称平均小于250ms。服务端同6次的TTFT约205--210ms，
客户端首包差异原因尚需复测。这轮总体吞吐为历史同口径220.047tok/s的约94.0%，
TPOT仍小于13ms；全量精度独立评测已从第0题开始，不使用困难20题拼接或答案后处理。

## 全量过程中的前100题诊断

上述独立全量运行的索引0--99，原NeMo评分器结果 **99/100**，只有case37错误，
无请求错误、无截断；严格数字辅助评分为98/100，差异为case14输出带百分号，
与SGLang保持相同NeMo数学评分规则。报告：
`logs/goal_hc_post_fix_full_live_20260925/diagnostic_first100.summary.json`。
这是运行过程中的前缀诊断（`complete=false`），未提前停止全量，也不用于替代1319题验收。

前250题按同评分器为243/250（97.2%），工件为同目录
`diagnostic_first250.summary.json`。其中case120--123只生成 `We` / `We know` 后stop，
全部原样记错，不能当作正确答案或通过最短输出长度强制补齐。其前一题case119输出405个token，
随后case124恢复正常；需要在完整评测结束后单独验证是否与请求顺序/Graph状态切换相关。
当前日志只证明它们进入真实q6 Graph路由，不能据此直接认定kernel数值错误。

排查中核对了当前安装的Torch/MUSA 2.9.1：
`torch_musa/csrc/aten/musa/MUSAGraph.cpp:228` 的replay使用
`getCurrentMUSAStream()`；实际 `libmusa_kernels.so` 中
`at::musa::MUSAGraph::replay()`（0x19ec4e0）反汇编也在0x19ec663调用当前流、
0x19ec67f调用 `musaGraphLaunch`。所以xLLM旧诊断注释关于“replay固定在capture_stream”
的前提不成立，不能未经实验将capture-stream barrier当成修复。

前500题结果为 **483/500（96.6%）**，修复前同范围454/500，SGLang同范围488/500。
其中37个旧xLLM错/SGLang对的题已恢复，仍有7个SGLang-only（120--123、368、394、425），
2个xLLM-only（93、306）；这些都属于前缀诊断，不改变全量分母。
case462此前的空SP分片崩溃在此次连续运行中未重现，正确回答71，前后请求正常完成。
原始结果和请求逐条保存在 `responses.raw.jsonl` / `requests.jsonl`，中途没有重启、换binary或重试。

## 复现入口

启动服务（另一个终端运行评测，先确认本人的旧服务/占卡已经释放，不会自动杀进程）：

```bash
cd /data/nfs_shared/jiaming.wang/xllm0914/xllm-musa
bash tools/run_dspark_hc_post_candidate.sh logs/NEW_HC_POST_RUN
```

入口固定8卡TP8/EP1/DP1、DSpark5个draft token（target q6）、Graph、Prefix Cache和
此次状态恢复开关，复用已编译的候选。旧诊断shell继承的force-eager、禁MATE、
target-only或Prefill dump等额外路由开关会被拒绝；全部有效XLLM路由变量会进入 `launch_manifest.txt`。
HTTP/master可用同名环境变量覆盖，默认19447和127.0.0.1:30710。

完整1319题单次评测：

```bash
python3 tools/eval_gsm8k_sgl_protocol.py \
  --run-dir logs/NEW_HC_POST_RUN \
  --dataset /data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/dataset_test.jsonl \
  --prompt-yaml /data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/eval_prompt.yaml \
  --url http://127.0.0.1:19447/v1/chat/completions \
  --model DeepSeek-V4-Flash-FP8 --thinking-mode chat
```

`progress.json`中的numeric评分仅辅助监控；最终使用原SGLang环境的
`tools/score_gsm8k_sgl_reference.py`（不加partial/limit）评分。参考答案只在评测/评分记录里，
不会发送给模型。不要将中断、不同binary或诊断重跑拼接成一次全量结果。
