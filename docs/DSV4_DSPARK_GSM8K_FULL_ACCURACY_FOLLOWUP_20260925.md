# Graph+DSpark 全量 GSM8K 精度问题后续定位

日期：2026-09-25。基于全量 SGLang protocol 实验继续分析；所有准确率使用同一
SGLang vendored NeMo `MathEvaluator(config={})`，没有读取 reference 改写 prompt、
按答案重试或修改评分。

## 1. 全量问题基线

正式全量目录：`logs/goal_gsm8k_full_sgl_chat_20260925/`。

| 项目 | 结果 |
|---|---:|
| xLLM Graph+DSpark q6 | 1132/1319 = **85.8226%** |
| SGLang Graph target-only参考 | 1276/1319 = **96.7400%** |
| xLLM 错、SGLang 对 | **149题** |
| 两者都对 | 1127题 |
| xLLM 对、SGLang 错 | 5题 |
| 两者都错 | 38题 |

149题的 xLLM 失败分类：

- 91题严格评分没有提取到 boxed answer；其中53题用宽松的最后 `####`/数字规则能够
  得到参考数字，但这只是格式诊断，不能加入正式成绩。
- 87题 boxed 内容与参考不匹配。
- 8题到达 max_tokens=1024 且答错。
- 1题为 worker 执行失败 `gsm8k-462`，不是模型生成的错误答案。

正式结果：[xllm.summary.json](../logs/goal_gsm8k_full_sgl_chat_20260925/xllm.summary.json)，
逐题对照：[comparison.json](../logs/goal_gsm8k_full_sgl_chat_20260925/comparison.json)。

## 2. 第一轮定位：不是单纯 MATE 数值误差

从149题中选取20道短输出/无boxed代表题，使用完全相同的原始请求 payload 做 A/B。
原始正式答案、target-only输出和 SGLang reference 都逐题保留。

### 2.1 target-only 接受路径 A/B

环境变量：`XLLM_DSV4_TARGET_ONLY_VERIFY_COUNT=4`，仍保留 Graph 和 DSpark draft forward，
但前4个验证 step 只提交 target 首行，隔离 draft 接受状态影响。

将20道诊断响应替换到完整 raw 文件仅用于 CPU 评分（不作为正式全量结果）：

| 诊断 | symbolic correct |
|---|---:|
| 正式 q6 Graph+DSpark 全量基线 | 1132/1319 |
| 20题 target-only=4 混合 A/B | 1142/1319（只替换20题） |
| 20题 target-only=32 混合 A/B | 1147/1319（只替换20题） |

这说明 accepted draft/state lifecycle 会影响部分答案，但并不能解释全部149题，也不能
把 target-only 永久打开：它显著增加 decode 工作且仍有错误。target-only=32 的部分输出
甚至出现不同错误答案，说明“强制 target-only”不是正确的最终方案。

诊断工件：

- `logs/goal_gsm8k_target_only_diag_20260925/replay2.jsonl`
- `logs/goal_gsm8k_target_only32b_20260925/replay3.jsonl`
- `logs/goal_gsm8k_full_sgl_chat_20260925/target_only4_hybrid.raw.jsonl`
- `logs/goal_gsm8k_full_sgl_chat_20260925/target_only20_hybrid.raw.jsonl`
- `logs/goal_gsm8k_full_sgl_chat_20260925/target_only4_hybrid.summary.json`
- `logs/goal_gsm8k_full_sgl_chat_20260925/target_only20b.summary.json`

### 2.2 关键新发现：temperature=0 被 top_p=0.95 错误地变成随机采样

SGLang 全量协议是 `temperature=0, top_p=0.95`。旧 xLLM
`SamplingParameters::init()` 使用：

```text
do_sample = do_sample || temperature != 0 || top_p != 1 || top_k > 0
```

因此 xLLM 把 SGLang 的零温度请求标成随机采样。实际 verify 日志中出现
`target_sampled != target_argmax`，与 GSM8K 应有的确定性贪心行为不一致。

已修复为：

```text
do_sample = explicit_do_sample || temperature != 0
```

也就是 temperature=0 时即使 top_p=0.95 仍走 greedy；只有显式 `do_sample=true`
才在 temperature=0 时随机采样。新增了 sampling regression test，覆盖：

- temperature=0、top_p=0.95、top_k=50、do_sample=false → greedy；
- 相同参数但 do_sample=true → random；
- mixed batch 中两种请求分别保持各自模式。

此修复候选：`build/gsm8k_candidates/greedy_temp0_source_complete_20260925/xllm`。
它在20题诊断中改善了部分响应，但仅凭20题不能宣称全量已接近96%；必须重新跑完整
1319题进行确认。早期20题 hybrid 只替换诊断题，不能当全量 score。

## 3. verify token instrumentation

新增非默认环境变量 `XLLM_DSV4_LOG_VERIFY_TOKENS=1`，在 DFlash target verify 中记录：

- draft token block；
- target logits argmax；
- target worker sampled token；
- rejection sampler 的 accepted/masked token。

示例日志表明某些请求存在 target argmax 与 target sampled 不一致，这正是上述 sampling
判定问题的直接证据。新候选的日志位于
`logs/goal_gsm8k_verify_tokens_20260925/rank0.log`，不打开该变量时无性能日志开销。

## 4. 空 Prefill-SP 执行错误

全量第462题 prompt 长度49。TP8均匀切分后局部有效行是：

```text
[7, 7, 7, 7, 7, 7, 7, 0]
```

rank7在 C++ Attention prefill path 报：

```text
muDNN Unary::Run: invalid configuration argument
mtgpu_impl_datacast MUDNN failed in Copy.cpp:301
```

普通 Python TorchMUSA 对 `(0,512) BF16 -> FP32` 的最小转换通过，因此不能简单说“所有
空 tensor 转换都坏”。目前定位到生产 C++ Prefill-SP 空分片/后续 fused path，尚未完成
最小 C++ tensor-shape 定位；该题在正式全量中保留为 execution failure，没有重试覆盖。
详情：[WORKER_FAILURE_462.md](../logs/goal_gsm8k_full_sgl_chat_20260925/WORKER_FAILURE_462.md)。

## 5. 当前结论与下一步

目前最有价值的修复是 sampling 语义修正，它直接对齐了 SGLang 的实际
`temperature=0, top_p=0.95` 行为；target-only A/B 说明 draft/state 仍有第二个问题。
两者不能混为一个“算子精度问题”。

下一步验收顺序：

1. 用 `greedy_temp0_source_complete_20260925/xllm` 完整重跑1319题同口径协议；
2. 对比新全量主评分、149题四象限和输出截断数；
3. 若仍明显低于96%，用 verify token 日志挑选首个 target argmax 分叉题，做 q2 Eager、
   q6 Graph+DSpark、target-only 的 layer-by-layer 对照；
4. 单独修复49-token空 Prefill-SP路径，避免全量测试被执行错误打断；
5. 全量通过后再测 B1 热态 TTFT/TPOT，不能用诊断 target-only 或无 draft 性能替代。

所有 GPU 实验结束后已恢复0--7占卡，未执行 GPU reset。当前目标仍未完成，85.82%不能作为
Graph+DSpark可交付精度。
