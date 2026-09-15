# DeepSeek-V4-Flash MUSA DSpark GSM8K 精度记录

日期：2026-09-14。实验分支：goal-dspark-30k-20260914，二进制为当前分支
build/dspark-musa-local/xllm/xllm（SHA256：
396a855499468a9add72321f1b862b5a73231ef0ceb37b098801593363f14269）。
模型：/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt。

## 数据集与评分

使用 GSM8K 官方测试集 grade_school_math/data/test.jsonl，下载地址为：
https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl。
下载文件 SHA256=3730d312f6e3440559ace48831e51066acaca737f6eab99bccb9e4b3c39d14，
共 1319 题；Parquet 转换副本和全部结果只保存在本目录下的 logs/。

本次固定随机种子 20260914 抽取 100 题，不是人工挑题；样本题号、题目、参考答案、
模型全文和预测值均在 logs/gsm8k_eval_20260914_1925/sample100/results.jsonl。
参考答案取官方答案中 #### 后的数值；预测答案取模型输出的最后一个 ####、boxed
或数值，并做逗号、整数、小数、分数规范化。该规则适用于 GSM8K 数值准确率，
不等同于严格文本匹配。

## 运行配置

- MTT S5000 8 卡，TP8/EP1/DP1/CP1，global batch=1，串行请求。
- DSpark N=5，Graph/VMM/decode-no-padding 开启，Prefix Cache 开启，schedule overlap 关闭。
- max_memory_utilization=0.8，max_tokens_per_batch=10240，MUSA Torch 2.9.1，
  使用已存在 AOT/custom-AR 依赖；没有修改共享依赖或 GPU 驱动。
- evaluator：prompt-style=reasoning、thinking-mode=chat、reasoning-effort=high、
  temperature=0、top-p=1、max_tokens=2048。

## 结果

| 数据量 | 正确 | 错误 | 请求错误 | 准确率 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 71 | 29 | 0 | **71.0%** |

总请求耗时 666.576 秒。服务端 100 个请求均正常返回，没有 OOM、进程退出或 HTTP
请求错误。逐请求服务日志在 logs/gsm8k_eval_20260914_1925/service/rank*.log；
可见短题 TTFT 约 120 ms，但部分请求生成到 2048 token 才停止，说明答案格式遵循
并不稳定。

## 错误定位初步结论

错误不是单一的“数值精度误差”表现：

1. 部分是正常的算术/读题错误，例如 index=125 预测 0、参考 10，index=128 预测
   130、参考 880，index=705 预测 10、参考 120。
2. 多个输出混入与当前题目无关的历史问答、模板或重复 token，例如 index=244、255、
   570、589、716、724、911、959；index=567 反复输出 2.2，index=1096 出现大量
   [end]。这类现象更像请求间 KV/Graph/DSpark 状态或输出约束问题，而不是单纯
   FP8/MXFP4 算子舍入；本轮没有做关闭 Graph、关闭 draft、Eager 或 MATE 对照，
   因而不能把它归因到某个算子。
3. 0 个请求异常且显存稳定，说明本轮主要是模型输出正确性未闭环，不是服务可用性或
   显存问题。71% 也不能代表官方 GSM8K test 全量准确率；要宣称全量结果必须跑完
   剩余 1219 题，并固定相同 prompt/scoring 口径。

## 可复现命令

服务启动和 GPU 占用方式见 DSpark 8192/1024 记录。服务 ready 后，在当前项目目录
执行（使用新的输出目录）：

    python3 -B /data/nfs_shared/jiaming.wang/test/fhxllm/xllm-musa-worker25/logs/goal_gsm8k100_20260811/deepseek_v4_gsm8k_eval.py \
      --dataset=logs/gsm8k_eval_20260914_1925/gsm8k_test.parquet \
      --url=http://127.0.0.1:18920/v1/chat/completions \
      --model=DeepSeek-V4-Flash-FP8 \
      --output-dir=logs/gsm8k_eval_new/sample100 \
      --samples=100 --seed=20260914 --max-tokens=2048 --timeout=600 \
      --temperature=0 --top-p=1 --thinking-mode=chat --reasoning-effort=high \
      --prompt-style=reasoning

本次测试结束后已停止 xLLM 服务，并恢复 0--7 卡占卡程序（PID=24062，每卡 70 GiB）；
未执行 GPU reset。

## 2026-09-15 诊断对照

关键源码改动已经先提交，工作树保持干净；当前生产二进制重新编译后 SHA256 为 `396a855499468a9add72321f1b862b5a73231ef0ceb37b098801593363f14269`。针对上一轮 29 个失败题做了以下隔离：

- `short-answer`、max_tokens=128：24/100（同一 100 题），去掉推理反而恶化。
- `expression`、max_tokens=256：67/100，不能消除错误。
- `thinking_mode=thinking`、max_tokens=4096：前 8 题均未修复，并出现平方和/弹珠等无关历史内容。
- `Graph=false`、Prefix Cache=false、draft 保留：代表性 3 题 0/3，输出长时间重复或答非所问。
- target-only（Graph=false、draft=false、Prefix Cache=false）：worker collective health check 不连通，未形成可用精度对照。
- 临时 `full_kv_threshold=0`：29 题几乎只输出 `Let’s` 后停止，证明不能直接强制所有短 prompt 走当前压缩 prefill。该改动已回退，未进入生产二进制。

trace 元数据确认普通 GSM8K prefill 为 `is_prefill=1`、`is_chunked_prefill=0`，例如 q/kv=`20/154`、`15/119`、`14/112`，所以“首个 prefill 没被识别”不是当前证据支持的原因。

上一轮 71% 的错误需要拆成两类处理：一部分模型正文已经出现正确答案，后续又生成无关内容（应改 stop/EOS 或请求状态）；另一部分是真实算术/读题错误。部分 GSM8K 官方题目还存在题面与标签不一致（例如 index=952 题面是 8AM 到 11PM，官方答案按 8AM 到 11AM 的 3 小时计算），不能把这类样本归咎于算子精度。当前尚未有足够证据安全修改 Attention/KV 默认路径；本轮没有保留实验性阈值修改。

证据目录：`logs/gsm8k_eval_20260914_1925/sample100/`、`logs/gsm8k_short_answer_20260915/`、`logs/gsm8k_threshold0_20260915/`、`logs/gsm8k_graph_eager_20260915/`。

## 2026-09-15 GSM8K goal 进展

独立分支 `fix/gsm8k-short-context-20260915` 增加了固定索引、dataset SHA、逐题
JSONL 落盘的评测工具，以及可独立控制 Graph、Prefix Cache、draft 和 chunk 的启动器。
可复现工具均已提交，生产二进制 `396a8554` 未被覆盖。

最好的稳定隔离结果是 Graph=true、DSpark draft=false、Prefix Cache=false、
reasoning_effort=none、temperature=0：

| 批次 | 正确率 | 请求错误 |
| --- | ---: | ---: |
| 固定前 20 题 | 17/20 = 85% | 0 |
| 固定前 60 题（3 批） | 47/60 = 78.3% | 0 |

high 思考模式在同一前 20 题也是 17/20；显式 `enable_thinking=false` 没有改变结果。
Prefix Cache 开启时观测到后续请求 prompt token 膨胀到 1200+ 并混入旧模板，关闭后
该现象消失，但仍有真实读题/算术错误。标准 train 8-shot 会显著增加 prompt 长度，
前 20 题结果更差，不能作为修复。

截至本记录，目标 90% 尚未达到，不能标记 goal 完成；仍需修复短上下文多阶段算术/
请求输出污染，或实现经过严格验证的失败题复核策略。GPU 测试服务停止后应恢复
0--7 卡占卡；生产性能二进制未被替换。

## GSM8K goal 复测（2026-09-15）

新分支 `fix/gsm8k-short-context-20260915` 增加了可复现实验工具，并将 Graph、
draft、Prefix Cache 和 chunk 大小全部写入 manifest。当前最好的可用隔离配置是：

```text
TP8/EP1/DP1/CP1, Graph=true, DSpark draft=false,
enable_prefix_cache=false, reasoning_effort=none, temperature=0
```

固定随机 100 题中的前 20 题结果为 `17/20=85%`；前三个 20 题批次合计
`47/60=78.3%`。这组结果没有请求错误，且 server prompt token 保持约 70--160，
没有 Prefix Cache 开启时出现的 1200+ token 旧模板膨胀。错误主要是真实读题/算术错误，
例如把 5 次/周、4 周和 2 次/周、3 周读错，或把多阶段金额题漏算加薪。

对比：默认 Graph+DSpark+Prefix Cache 的固定 100 题为 71/100；只输出数字为
24/100；算式输出为 67/100。关闭 Graph/draft 的 target-only 路径会发生 collective
health-check 断连，不能作为正确性基线。

当前仍未达到用户要求的 90%，因此本 goal 不能标记完成。测试工具会拒绝覆盖已有
输出、记录 dataset SHA、每题原始 response，并在 HTTP/worker 错误时停止而不是把
连接失败计入模型准确率。下一步应在 no-draft/no-prefix 隔离路径上实现受控的失败题
复核或修复短上下文多阶段算术/输出污染，然后重新跑完整 100 题；不要把 few-shot
test-derived 演示（会导致复制演示答案）当成标准 GSM8K 结果。

## 2026-09-15 晚间：短上下文状态隔离候选

候选二进制：`build/gsm8k_candidates/state_reset_20260915/xllm`，
SHA256=`be9c2b465794555a3dd30a678f8e5933a0c7fc3571a18bb63882cf5178c29e4c`。
生产二进制 `build/gsm8k_baseline_396a/xllm` 未覆盖。

新增防御性修复：普通 prefill 在进入 full-KV/compressed 路由选择前清理本层
compressor/indexer 的 live history/state，避免短 prompt 绕过 compressed 分支后
保留上一请求的 live state。此修复只做请求隔离，不声称已经解决全部精度问题。

同一固定 100 题的前 20 题，使用正确的 0731 权重重新验证：

| 配置 | 提示词 | 结果 | 请求错误 |
| --- | --- | ---: | ---: |
| Graph=true, draft=false, Prefix Cache=false | baseline, high, max_tokens=2048 | 17/20=85% | 0 |
| Graph=true, DSpark N=5, Prefix Cache=true | baseline, high, max_tokens=2048 | 14/20=70% | 0 |
| Graph=true, draft=false, Prefix Cache=false | structured, high, max_tokens=1024 | 16/20=80% | 0 |
| 同上，固定第 21--40 题 | structured, high, max_tokens=1024 | 15/20=75% | 0 |

structured 提示词的前 40 题合计 31/40=77.5%，没有提升到 90%。将失败样本输出
上限改为 512 也没有修复：10 个诊断样本为 2/10；这不是完整回归准确率。
无 draft/no-prefix 路径的稳态 TPOT 约 22.2--22.5 ms，仍慢于历史高性能 DSpark
路径，不作为最终性能配置。

重要边界：最初一轮候选实验误用了旧路径 `DeepSeek-V4-Flash-FP8`，其 3 题结果
仅保存在 `failed3.json`，不计入 0731 的准确率。之后所有 `0731_*` 结果均由
manifest 确认实际权重为 `DeepSeek-V4-Flash-0731-FP8-mt`。

当前仍能看到两类错误：

- 真实读题/算术错误：index=201 把 $5000 奖励改写成 $1000，或漏算加薪；
  index=235 把题目的 20 人改成 21 人。
- 答案后续漂移：index=128 先正确计算 880，之后改写为另一道水果题；index=244
  把最终 27 改成 25，随后继续生成无关问题。仅清理 live state 和调整提示词
  不能消除，尚不能归因于某个 FP8/MXFP4/MATE 算子。

证据全部位于 `logs/gsm8k_state_reset_20260915/`。目标 90% 尚未达到，goal 保持
active；后续需要对同一请求做 Graph/Eager logits 对照，并检查短上下文 SWA
slot/block-table 与 graph persistent metadata，而不是继续把提示词收益当作
运行时正确性修复。实验结束恢复 0--7 卡占卡，未执行 GPU reset。

## 2026-09-15 后续状态隔离候选

在前述修复基础上，普通 prefill 现在也清理 prefix compressor/indexer snapshot
向量，防止 full-KV 短请求绕过压缩路径时遗留按 sequence slot 保存的快照污染下一次
带 prefix 的请求。候选二进制为
`build/gsm8k_candidates/state_reset_prefix_20260915/xllm`，SHA256=
`ef1e20f3815b4524e3bb81b9c63687674489481287eb20e15cd193be7faede01`。

该候选已完成编译检查，但本轮启动复测时 MUSA allocator 报告每卡仅约 249 MiB
可用，服务未进入可测试状态；没有把这次 OOM 当作精度结果，也没有覆盖生产二进制。
待 GPU allocator 状态恢复后，必须重新跑固定 100 题并与前一候选按相同 manifest
对比。

## 2026-09-15 dense-KV 阈值对照

为验证 DSA 压缩 decode 是否是短题精度主因，新增环境变量
`XLLM_DSV4_FULL_KV_THRESHOLD`；默认仍为 `2048`，只用于隔离实验，不改变默认性能
路径。候选 `build/gsm8k_candidates/fullkv4096_20260915/xllm` 使用阈值 `4096`，
SHA256=`612432d0c82a283a8b9813d388807d8a2258e3e240c1c29a215e46b8d54ff5f8`。

在正确的 0731 权重、Graph=true、draft=false、Prefix Cache=false、前 20 题下，
阈值 4096 结果仍为 **17/20=85%**，请求错误为 0；201、206、244 仍失败。说明
把短序列更多地固定到 dense-KV 不能消除主要精度问题，因此没有将 4096 设为默认值。
实验结果位于 `logs/gsm8k_state_reset_20260915/fullkv4096/`，生产二进制未覆盖。

## 2026-09-15 MUSA Graph/Eager 对照

补充修复了 MUSA Graph executor 对 `max_tokens_for_graph_mode` 不生效的问题，并在
`tools/start_dspark_custom.sh` 增加 `XLLM2_EXTERNAL_TP8_MAX_GRAPH_TOKENS` 覆盖项。
默认值仍为 2048；设置为 `0` 时显式走 Eager，便于精度对照。候选二进制为
`build/gsm8k_candidates/graphlimit_20260915/xllm`，SHA256=
`9f4cf97c295246d6a02712c3ecf588f9a1d2a28cb7924fe8d5e7881ca7f8fd7a`。

同一 0731 权重、无 draft、无 Prefix Cache、固定前 20 题的 Eager 运行中，前 8
题完成 **8/8**；随后在长生成题上单题耗时显著增加，完整 20 题尚未结束。之前同
口径 Graph 运行前 20 题为 17/20，且 201/206/244 错误；因此 Graph 路径确实是
精度风险来源之一，但 Eager 尚未形成完整 100 题结果，不能据此宣称达到 90%。
本次长 Eager 任务已停止，生产性能路径未改变。

后续 Eager 长回归（`logs/gsm8k_eager_full_20260915/full100_1024.jsonl`）完成前
24 题，结果为 **19/24=79.2%**，因单题生成时间约 60 秒以上而停止；这进一步说明
Eager 不是足以达到 90% 的单独修复。前 7 题曾达到 7/7，但扩大样本后出现
index=125、184、235、244、255 等错误。该结果仅作为 Graph/Eager 对照证据，不是
完整 100 题结论。

## 2026-09-16 动态 MATE FlashMLA 对照

对 Graph 路径加入实验开关 `XLLM_DSV4_DISABLE_DYNAMIC_MATE=1`，仅禁用带动态
`kv_seq_lens` 的 MATE FlashMLA，默认仍启用。候选二进制为
`build/gsm8k_candidates/disable_dynamic_mate_20260915/xllm`，SHA256=
`04d9bcf6e9d99beafb779279598d87f1755d62b0d7e1875d566308c7131ea26`。

正确的 0731 权重、Graph=true、draft=false、Prefix Cache=false、固定前 20 题、
max_tokens=1024 下，动态 MATE 禁用结果为 **16/20=80%**，默认 Graph 为 17/20。
错误集合也发生变化（128、201、235、244 等），因此动态 MATE 是数值差异来源，
但不是唯一精度根因，不能把它设为默认关闭。完整逐题结果在
`logs/gsm8k_token_compare_20260915/dynamic_mate_off2/`。

## 2026-09-16 host mirror 刷新对照

开启 `XLLM_KEEP_HOST_MIRROR_REFRESH=1`，并同时开启 Graph 输入/输出同步后，
固定前 20 题仍为 **17/20=85%**，与默认 Graph 相同。由此排除当前默认跳过的
paged-KV host mirror 刷新是主要精度根因；问题继续收敛到 Graph 内部某层状态或
算子数值差异。结果位于 `logs/gsm8k_token_compare_20260915/hostmirror_refresh/`。

## 2026-09-16 Graph replay 完成屏障对照

新增 `XLLM_MUSA_GRAPH_SYNC_OUTPUTS=1` 诊断开关，在 Graph replay 返回前等待
`capture_stream` 完成，以排除上一请求仍写共享 KV/output buffer 的竞态。候选二进制
`build/gsm8k_candidates/graph_sync_both_20260916/xllm`，SHA256=
`e95a9743152b459f2b585d6f6a2c4584551243b9dc50934f754acef60366e4f8`。

与输入同步同时开启时，固定前 20 题仍为 **17/20=85%**，与默认 Graph 完全相同；
因此简单的 replay 前后 stream race 不是主要精度根因。Graph/Eager 首个 token 对照
仍显示约第 15 个 token 开始分歧，后续应继续比较 persistent embedding、Q/KV
projection 和每层 attention 输出。该屏障保持诊断开关，未设为默认，避免性能回退。

## 2026-09-16 自定义 all-reduce 对照

加入 `XLLM_DSV4_DISABLE_CUSTOM_AR=1`，让 Attention/MoE 跳过自定义 all-reduce，
回退到普通 process-group collective。候选二进制为
`build/gsm8k_candidates/disable_custom_ar_20260916/xllm`，SHA256=
`a5b63c11a6c390128cfd21937e7dfbffc7705380183fa07f1b4b6174aee313ca`。

正确的 0731 权重、Graph=true、无 draft、无 Prefix Cache、固定前 20 题、
max_tokens=1024 下，关闭 custom AR 结果为 **16/20=80%**，低于默认 Graph 的
17/20=85%。因此 custom AR 不是唯一或主要精度根因，默认通信路径保持不变。
结果位于 `logs/gsm8k_token_compare_20260915/custom_ar_off/`。
