# xLLM Graph+DSpark：参考 SGLang 的全量 GSM8K 评测

状态：已完成。xLLM Graph+DSpark 主评分为 **1132/1319 = 85.82259287%**；
SGLang参考为 **1276/1319 = 96.73995451%**，xLLM低 **10.91736164 个百分点**。

**运行中发现可靠性问题：** 完成462题后，gsm8k-462的49-token prefill在rank7触发
muDNN dtype转换 `invalid configuration argument`，该rank的Prefill-SP有效行数为0。
已保留现场并用同二进制/同参数恢复服务；该题最终保留为执行失败，没有重试覆盖。
因此即使最终覆盖1319题，也不能宣称是无故障、连续一次跑通。详见正式目录
`WORKER_FAILURE_462.md` 和 `recovery_01/`。

最终确实覆盖1319个唯一ID；其中1题是执行失败，不能视为模型答对或答错。

## 1. 目标与参考证据

用户要求参照 SGLang 的单轮 boxed 协议，评测 xLLM Graph+DSpark 的 GSM8K test 全量1319题。
SGLang 原始记录位于：

`/data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30/`

其中 `SGLANG_DEEPSEEK_V4_GSM8K_EXPERIMENT_20260920.md`、`metrics.json` 和逐题输出
确认历史成绩为 **1276/1319 = 96.73995451%**。
只读连接 `10.20.34.28` 的 `wjm_sg_dspark` 容器核对了评测源码；四个记录过的源码哈希一致。
使用其中现有的 CPU MathEvaluator 对 SGLang 1319份历史输出重新评分，仍为1276/1319，
每一题的 symbolic_correct 均与原记录相同。没有在参考环境重启服务、申请GPU或修改库。

## 2. 不能忽略的参考口径差异

1. 参考 SGLang 开启 Graph，但 **没有开启 DSpark/MTP/speculative decode**；
   容器名字含 dspark 不能替代实际服务配置。
2. 参考评分器是 sgl-eval vendored NeMo-Skills `MathEvaluator(config={})`，从最后的
   boxed 提取答案，经过 LaTeX 规范化和必要的数学等价比较，字段为 `symbolic_correct`。
   它不是单纯字符串相等。
3. 同一份 SGLang 输出用补充的“纯数字/简单分数 boxed exact-match”规则得1257/1319，
   与主评分有19题表示方式/单位差异。主对比必须使用原评分器，不能将两个数字混报。
4. SGLang 当时使用32个客户端线程；本次 xLLM 使用已验收的 B1 服务、客户端串行。
   因此评测耗时/吞吐不作框架性能对比；并发和执行模式差异保留在结论中。
5. 服务默认 thinking 模式不同，必须显式对齐，见下一节。

## 3. 默认 thinking 差异与诊断运行

初始运行忠实复制了不带 chat_template_kwargs 的 sgl-eval JSON 请求。
随后源码核对发现：

- SGLang `serving_chat.py` 使用 `SGLANG_DEFAULT_THINKING`，其源码默认是 `EnvBool(False)`；
  归档环境及当前参考容器均无该变量覆盖，实际为 chat。
- xLLM `deepseek_v4_cpp_template.cpp::get_thinking_mode()` 在没有显式 thinking 标志时
  默认返回 thinking。响应中可看到推理文本和 `</think>`。

因此相同“省略字段”在两个框架中并非相同模式。初始运行保留27份完成响应、28条派发记录
（最后一条中断），只有23份通过补充纯数字评分、2份输出达到1024上限；它不是全量结果，
不作为正式框架对比或重新挑选答案的依据。

诊断目录：`logs/goal_gsm8k_full_sgl_protocol_20260925/`，详见其中 `DIAGNOSTIC_ONLY.md`。

正式运行使用新服务、新目录，从 index 0 开始，只增加：

```json
{"chat_template_kwargs": {"thinking": false}}
```

该设置用于对齐参考的实际 chat 模式，**不关闭 Graph 或 DSpark**，不修改模型代码/权重。
模型、boxed prompt、temperature、top_p、1024上限均保持一致；正式每题只生成一次，
不按答案正确性进行重试或修补。不能称这项适配为“请求体逐字完全相同”。

## 4. 正式配置

| 项目 | xLLM 正式值 |
|---|---|
| 模型 | `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt` |
| 模型config SHA256 | `9a387b757dc12751447ec0f450a30929b3bfab17f2ee4b9a9a3372300c373fa4`，与参考一致 |
| 二进制 | `build/gsm8k_candidates/current_graph_shared_address_20260925/xllm` |
| binary SHA256 | `5785ab8228debdd5e5e9a9b24be91a53ba650f913d947d225cd0b11052a43f52` |
| 拓扑 | 8卡TP8/EP1，global B1 |
| 解码 | DSpark N=5 / target q6，compact Graph开启，短 Eager 保护关闭 |
| Prefill/内存 | batch/chunk token预算10240，memory utilization=0.8，Prefix Cache开启 |
| 数据 | 原参考目录 `dataset_test.jsonl`，全部1319条，按原顺序 |
| 数据SHA256 | `67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456` |
| 采样 | temperature=0，top_p=0.95，max_tokens=1024，n_repeats=1 |
| 其他请求设置 | 无system message/seed/reasoning_effort；thinking=false对齐参考默认值 |
| HTTP | `http://127.0.0.1:19280/v1/chat/completions` |
| 正式目录 | `logs/goal_gsm8k_full_sgl_chat_20260925/` |

用户提示词直接从归档 `eval_prompt.yaml` 渲染，无few-shot：

```text
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

{problem}
```

当前协议不同于此前固定100题的 `####` prompt、top_p=1和2048输出上限。
不能拿两个协议的分数差直接归因于新算子精度变化。

另外用参考容器的原生 `encoding_dsv4.py`（chat、reasoning_effort=None）和同一
`tokenizer.json` 在CPU上编码全部1319题，合计 **113369 tokens**，与历史SGLang指标一致。
每题长度保存在正式目录 `reference_prompt_tokens.json`，最终与xLLM实际usage逐题检查。
输入长度一致是对齐证据之一，不单独宣称两套tokenizer已完成逐token ID证明。

运行环境仍有明确差异：xLLM侧驱动5.1.0、Torch/TorchMUSA 2.9.1；参考SGLang侧记录为
驱动5.2.0、Torch/TorchMUSA 2.9.0。参考page size=256、KV cache=fp8_e4m3，
xLLM使用现有DSV4混合cache实现与block size=128、KV dtype=auto。不能把最终分数差
仅归因于DSpark开关、单个kernel或某一个环境版本。

## 5. 工具与证据保存

- `tools/eval_gsm8k_sgl_protocol.py`：串行HTTP适配器；保存逐题请求、完整响应、结束原因、
  token数及实时覆盖进度。仅补充显示纯数字boxed分数，不冒充NeMo主评分。
- `tools/score_gsm8k_sgl_reference.py`：在原SGLang容器内仅用CPU，调用未修改的
  MathEvaluator；输出到当前xLLM项目目录，不修改参考代码/结果。
- 官方评分环境：sgl-eval=0.0.1、math-verify=0.9.0、latex2sympy2-extended=1.11.0、sympy=1.14.0。
- `responses.raw.jsonl`、`requests.jsonl`：正式每题响应与请求；reference不进入模型payload。
- `eval_config.json`、adapter/data/prompt快照、`launch_manifest.txt`、AOT哈希：版本和协议。
- `progress.json`：仅是进行中的覆盖及补充数值评分，不能当成最终完整准确率。
- 最终验收必须核对1319唯一ID、无遗漏/重复、原始题目/reference一致、Graph+draft真实路径。
- `tools/compare_gsm8k_sgl_full.py`：核对所有正式请求的payload、评分前后原始文本不变，
  生成两框架逐题四象限统计、失败列表、输入token数对照及Graph路径检查。

本轮不编译新模型或依赖，复用已验收的候选及原参考CPU评分环境；不用GPU时恢复占卡，
不执行reset。

## 6. 全量结果与逐题对照

正式目录：`logs/goal_gsm8k_full_sgl_chat_20260925/`。

| 指标 | xLLM Graph+DSpark | SGLang历史参考 |
|---|---:|---:|
| 主评分器正确数 | **1132/1319** | **1276/1319** |
| 主评分器准确率 | **85.82259287%** | **96.73995451%** |
| 差异 | **-144题 / -10.91736164个百分点** | — |
| 唯一ID覆盖 | 1319/1319 | 1319/1319 |
| xLLM请求执行错误 | 1（gsm8k-462） | 0 |
| xLLM输出截断 | 10 | 0 |
| 其他xLLM评分错误 | 177 | 43 |

主评分由参考容器中未修改的 sgl-eval vendored `MathEvaluator(config={})`完成，
不是本地纯数字boxed规则。xLLM补充纯数字boxed结果是1114/1319=84.45792267%，
仅用于诊断；两种评分有18题表示/数学等价差异，不能将1114当作主成绩。

逐题四象限：

| 分类 | 题数 |
|---|---:|
| 两者都正确 | 1127 |
| 仅xLLM正确 | 5 |
| 仅SGLang正确 | 149 |
| 两者都错误 | 38 |

主要工件：

- [xLLM逐题原始响应](../logs/goal_gsm8k_full_sgl_chat_20260925/responses.raw.jsonl)
- [xLLM评分结果](../logs/goal_gsm8k_full_sgl_chat_20260925/xllm.scored.jsonl)
- [xLLM评分汇总](../logs/goal_gsm8k_full_sgl_chat_20260925/xllm.summary.json)
- [逐题对照汇总](../logs/goal_gsm8k_full_sgl_chat_20260925/comparison.json)
- [不一致题逐题记录](../logs/goal_gsm8k_full_sgl_chat_20260925/disagreement_cases.jsonl)

## 7. 全量结果的解释

输入协议已经核对：SGLang原生DSV4 chat encoder对全部题计算113369 prompt tokens，
与历史SGLang metrics一致；成功xLLM请求逐题 prompt token 数全部匹配，xLLM总usage少49
正是执行失败的 gsm8k-462。正式rank日志观察到 q6 Graph replay，forced-Eager compact计数为0。

xLLM主评分失败分类：

- 91题没有可提取的 boxed answer；
- 87题 boxed answer 与标准答案不匹配；
- 8题达到1024上限且答错；
- 1题 worker 执行失败。

其中91道“无boxed”里，有53道用宽松 `####/最后数字` 规则可以得到参考数字；这只是
诊断，不得改写SGLang主评分。它说明当前xLLM chat模式下的格式遵循/停止行为是主要问题之一，
因此不能把85.82%的全部差异归因于MATE、FlashMLA或其他算子数值精度。

恢复前完成0--461题，评分403/462；恢复后463--1318题评分729/856，没有新增请求执行错误。
这是一次发生worker崩溃后恢复的全量覆盖，不是无故障连续单轮测试。执行错误的完整原因、
49-token输入和rank7空Prefill-SP分片见 `WORKER_FAILURE_462.md`。

此前固定100题96/100使用 `####` prompt、top_p=1、max_tokens=2048和固定100抽样，
不是本次SGLang boxed协议，不能用来替代本次全量成绩。

## 8. 针对“xLLM错、SGLang对”的第一轮定位

从149道 `sglang_only_correct` 中抽取了格式/短输出代表题，使用同一原始 payload
在 xLLM Graph+DSpark 上做了两种诊断：

| 诊断 | 20题结果 | 结论 |
|---|---:|---|
| 正式 q6 draft 路径 | 依照全量主结果 | 大量 bare number、残缺 brace 或错误算式 |
| `XLLM_DSV4_TARGET_ONLY_VERIFY_COUNT=4` | 20题替换后全量混合分从1132变1142 | 只改善10题，不能解释149题 |
| `XLLM_DSV4_TARGET_ONLY_VERIFY_COUNT=32` | 只做诊断，未作为成绩 | 仍有错误，且 latency明显增加，不是可交付修复 |

这些 A/B 没有读取 reference，也没有按答案重试。target-only A/B 说明 accepted draft/
target 状态生命周期确实影响输出，但“强制 target-only”本身既不能恢复到 SGLang
96.74%，也不能保留 q6 性能；不能把它直接设成生产默认值。

当前最强证据是：

- 91题无 boxed 中，有53题的宽松最后数字其实与 reference 一致；这是输出格式/停止
  路径问题，不是全都 Attention 数值错误。
- 其余 boxed mismatch 需要再按 `target_logits`、accepted token、draft token 和
  请求状态逐题分组；不能从最终文本单独定位到某个 MATE kernel。
- `XLLM_DSV4_TARGET_ONLY_VERIFY_COUNT=4/32` 只验证因果方向，不能作为最终 accuracy。

诊断原始响应：`logs/goal_gsm8k_target_only_diag_20260925/replay2.jsonl` 和
`logs/goal_gsm8k_target_only32b_20260925/replay3.jsonl`。这些目录不属于正式全量成绩。

下一步应优先新增不改变默认性能的 instrumentation：每次 verify 记录 draft token、
target argmax、accepted length、masked token 与最终 response token；对代表题比较 q2
Eager、q6 Graph+draft 和 q6 target-only 的首个 token 分叉。只有首个分叉稳定落在
accepted-state/Graph 输入后，才继续改 kernel；若 target argmax 本身已分叉，则需要
按 layer 对照 Attention/DSA，而不是继续改输出格式。
