# Graph+DSpark GSM8K goal progress

本轮目标是优先修复 xLLM Graph+DSpark 与 SGLang 的逐题差异，并将全量 GSM8K
准确率提升到约 96%。所有实验只修改本 xLLM checkout；SGLang 容器和模型目录只读。

## 最新进展：HC post 矩阵方向修复

发现并修复了HC post两条MUSA kernel将 `[input_lane, output_lane]` 读成
`[output_lane, input_lane]` 的逻辑错误。旧archive新增两项单测均失败，修复后
完整14项kernel测试通过；真实Graph replay、原地residual复用也通过。
困难20题从1/20提升至19/20（SGLang同组19/20），其中原先xLLM错/SGLang对的18题
全部恢复。case55首token与答案也已恢复，逐层误差明显下降。

完整证据与验收边界见 [HC post修复报告](DSV4_HC_POST_ORIENTATION_FIX_20260925.md)。
同一binary的新一轮完整1319题已完成：**1269/1319 = 96.20925%**，无请求异常、无截断，
没有中断重启或逐题重试。SGLang参考1276/1319，差7题；原149个SGLang-only恢复147个，
同时也出现新错题，当前SGLang-only共15个，均已记录。此结果不是由小样本或答案审计提分。
Graph+DSpark q6仍实际运行，全量窗口内q6路由26500次，无强制Eager。
同binary全量后热态性能：B1/2500/1500/Prefix0.65，**TTFT215.290ms、TPOT12.657ms**，
严格计数6/6，8卡聚合输出78.173tok/s、输入+输出208.461tok/s，保留历史同口径吞吐约94.73%。
该轮完整验收 `acceptance.json` 20项PASS。补查没有改写全量成绩，剩余15个对照差异及5个极短输出
均如实记录；停止模型后已恢复8卡占卡，每卡约8346MiB，未reset GPU。
下文89.3859%为修复前诊断基线；旧占卡状态、未闭环项是各阶段的历史记录。

## 已完成的修复

1. `SamplingParameters` 现在把 `temperature=0` 视为 greedy，即使请求同时带
   `top_p=0.95`；只有显式 `do_sample=true` 才开启随机采样。
2. 修复了 GSM8K-462 暴露的空 Prefill-SP shard：`local_valid_tokens=0` 时，Attention
   使用一个不参与语义状态的 dummy query 行，KV/compressor collective 仍以空 local
   shard 参与 gather，最终丢弃 dummy 输出，避免 MUDNN 空 datacast 崩溃或 collective hang。
3. isolated candidate builder 增加了 chat-template archive 映射、显式 kernel archive
   和 source-complete provenance，构建不覆盖原有 build archive。
4. omitted `reasoning_effort` 改为 low 的实验未证明提升，已撤销该次源码/测试修改，
   保留历史 candidate 与输出。SGLang 默认 chat、xLLM 默认 thinking 不同；本目标用
   xLLM 显式 `thinking=false` 对齐 SGLang，不能将 omitted-thinking 实验当作对齐结果。

## 全量证据

输入审计现已升级为全量 token 级：使用 xLLM 已编译的 native template/FastTokenizer
对象，以及 SGLang 参考容器实际 fastokens/encoding_dsv4，两边 chat 模式
1319/1319 rendered strings、完整 token IDs 一致。证据：
`build/prompt_probe_20260925/token_identity_audit.json`。之前仅比较长度的证据不足以
支持此结论，以本次完整审计为准。

此前 `chat_template_kwargs={"thinking":false}` 的 Graph+DSpark 全覆盖结果：
0--461 使用 sampling-fix binary，462--1318 使用加入 empty-SP guard 的 binary。
这是崩溃后拼接的诊断结果，不是最终单一 binary、无中断全量验收。

| 项目 | 结果 |
| --- | ---: |
| xLLM symbolic | 1179/1319 = 89.3859% |
| 采样修复前基线 | 1132/1319 = 85.8226% |
| SGLang reference | 1276/1319 = 96.73995% |
| xLLM request errors | 0 |
| xLLM Graph q6 replay | observed |

该结果目录为 `logs/goal_gsm8k_sampling_fix_resume3_20260925/`，逐题对照为
`comparison.json` 和 `disagreement_cases.jsonl`。其中 SGLang-only 104 题中，43 题
的 xLLM 最后数值已经正确但没有严格 boxed；这属于输出/停止路径问题，不应通过评分
放宽解决。

低 reasoning-prefix 候选 `build/gsm8k_candidates/greedy_temp0_low_reasoning_20260925_v2/xllm`
已启动过 omitted-thinking 全量，但因该口径与 SGLang 默认 thinking 语义尚未完成完整
渲染对照而停止；其部分结果不作为最终准确率。正式 chat=false 结果仍是当前可复现主基线。

## 参考无关算术审计

对正式 1319 条 xLLM 原始输出做了 reference-free 显式等式审计：23 条被标记，其中
9 条产生局部修正，其中6条与标准答案一致、3条错误（包含把原本正确的答案改错）。
故不能称这是一项已验证安全的修复，未将其结果合入主评分。仅凭等式审计也不能断言
其余错误都是 accepted-token/state 问题。
工件：`logs/goal_gsm8k_chat_arithmetic_audit_20260925/` 和
`logs/goal_gsm8k_chat_arithmetic_repair_20260925/`。

## 当前未闭环项

- Graph+DSpark 全量仍未达到 96%，不能宣称目标完成。
- rendered prompt/token IDs 的全量对照已完成；尚需验证输出/停止以及模型逐层语义。
- 对仍错误的 61 个数值不匹配题，需要继续做 q2 Eager、q6 Graph+draft 的 target
  logits/accepted-token/state 首次分叉定位；不能用 reference answer 重试。
- 服务器停止后已用 8 个独立 MUSA holder 进程恢复占卡（每卡约 8 GiB），未执行 GPU reset。

## Decode A/B 排除项

- `XLLM_DSV4_DISABLE_GATE_GEMV=1`：将 decode `sqrtsoftplus` gate 从 MUSA GEMV
  切回 native matmul/top-k；此前所谓11/20是宽松末数字诊断，不是NeMo准确率；gsm8k-55仍为28（参考
  14），没有解释主要误差。候选：`build/gsm8k_candidates/gate_native_decode_ab_20260925/`。
- `XLLM_DSV4_DISABLE_MATE_HC_PRE=1`：禁用 MATE HC prenorm/big-fuse，走普通 HC
  decode kernel；代表题输出仍保留 gsm8k-55=28，未证明 HC fused path 是主因。
  候选：`build/gsm8k_candidates/hc_native_decode_ab_20260925/`。

这些 A/B 仅说明相关开关没有单独解决代表错题，不能据此排除整个 gate/HC/MoE。

- `XLLM_DSV4_DISABLE_MOE_FP8_GEMV=1` 的 native per-expert fallback 在 Graph capture
  中触发 `operation not permitted when stream is capturing`；强制 Eager 后
  gsm8k-55 仍为 28，暂未证明 FP8 GEMV 是主因。候选：
  `build/gsm8k_candidates/moe_native_decode_ab_20260925_v4/`。
- 修正了 MUSA grouped-FP8 MoE 的 `moe_compute_index` namespace 声明：musamapping
  实际导出 `kernel::musa`，共享头文件保留历史 `kernel::cuda`。候选构建已验证可链接，
  这是 ABI/构建一致性修复，不等同于精度已达标。
- `XLLM_DSV4_DISABLE_FP8_BF16_CACHE=1` 的 Eager/native A/B 改变了多个代表题的输出，
  例如 gsm8k-72 从 `$203` 变成 `$221`、gsm8k-158 变成 `34`；说明低 token FP8
  dequant-BF16 cache 确实影响 decode 数值/生成路径。但该 A/B 同时使部分题变差，且
  只在 Eager、禁 draft、禁 MATE 条件下验证，尚未形成 Graph+DSpark 可用修复。
  候选：`build/gsm8k_candidates/fp8_linear_native_decode_ab_20260925/`。
  在 Graph+draft 前 100 题替换 hybrid 的官方 NeMo 评分为 1178/1319，略低于
  基线 1179/1319，因此目前不能直接合入为默认路径；该开关只保留为后续逐层/逐算子
  对照工具。

## 首 token / Prefill 定位（后续新增证据）

- `logs/goal_gsm8k_prefill_token_origin_20260925/origin.jsonl`：绕过聊天模板，直接输入
  已审计 token IDs，`max_tokens=1` 返回 gsm8k-55 token1449（28），gsm8k-3 token93
  （左花括号），gsm8k-72 token6（美元符号）。这些响应尚未执行 target-q6 验证，
  因此至少这类错误已在 Prefill 首步出现，不能全归因于 draft 接受。
- `XLLM_DSV4_FULL_KV_THRESHOLD=0` 的同一四题探针未改变首 token；工件在
  `logs/goal_gsm8k_full_kv_zero_origin_20260925/`。该开关不是已验证修复。
- `logs/goal_prefill_core_case55_20260925/core_cpu_reference.json`：真实 case55 的
  rank7、43层 Prefill Q/KV/压缩KV/indices/output 与独立CPU FP64 causal SWA+
  compressed reference比较，各层relative L2为0.001704--0.001817，所有输出有限。
  这只验证给定输入的 Attention core；上游输入、projection及跨框架模型语义尚待比较。
- `tools/sgl_dsv4_prefill_probe_server.py` 是进程内只读 hooks，不改SGLang package。
  同机参考首次Graph启动失败于 `full_to_swa_index_mapping[kv_indices]` 的capture兼容性，
  完整错误日志 `logs/sgl_prefill_probe_launcher_20260925.log` 已保留。

## 同机 SGLang / xLLM Prefill 实际张量对照

同机 `10.20.34.25/wjm_sg_dspark` 的源码 HEAD 为
`07f061297417019e501fd801925af502a2906b7f`。首次Graph启动确实失败，第二次仅关闭参考服务
Graph，复用该checkout内私有缓存，完成 **Prefill诊断**；它不是SGLang历史96.74%的重测。
SGLang package/model 未修改；临时hooks只作用于本次进程，编译cache从已有结果复制到
`build/sgl_prefill_probe_cache_20260925/` 后使用，不改donor缓存。

同一个case55、76个完整token IDs：

- xLLM Prefill首token1449=`28`（max_tokens=1，尚无draft verify）。
- SGLang Prefill首token=`We`，logprob约-0.07966；候选`28`约-3.45466。
- SGLang运行时保存的input_ids与xLLM输入dump逐元素一致。
- 原始Q和Attention输出逐层保存于
  `logs/goal_prefill_stages_case55_20260925/core/` 和
  `logs/goal_sgl_case55_same_host_eager_20260925/dump/`。

对齐SP rank7的绝对query行70--75、SGLang TP0的heads0--7，relative L2如下：

| Layer | Attention归一化输入 | Q | Core输出 | Attention输出投影 |
|---|---:|---:|---:|---:|
| 0 | 0.002814 | 0.027192 | 0.019677 | 0.028989 |
| 1 | 0.039394 | 0.037835 | 0.034563 | 0.045796 |
| 2 | 0.165561 | 0.097868 | 0.088262 | 0.100502 |
| 42 | 0.616190 | 0.548670 | 0.535083 | 0.368330 |

全部43层的量化结果：
`logs/goal_prefill_stages_case55_20260925/sglang_layer_comparison.json`。
这说明首个可测差异已经在layer0输入/投影，且layer1后的FFN/HC更新放大明显。
不能据此直接断言某个kernel错误；两种FP8执行路径本来就存在舍入差异，需要进一步分段
验证同输入运算。

`tools/audit_dsv4_projection_quant.py` 使用checkpoint权重和CPU独立线性/RMS运算，
比较FP8 activation和BF16 activation路径。在SGLang相同归一化输入上，Q非RoPE维度
的相对误差由BF16路径约3.66%缩小至FP8路径约1.44%，说明激活量化可以解释部分差异，
但尚未完全对齐。

新增默认关闭的 `XLLM_DSV4_FP8_PACKED_ACTIVATIONS=1`，配合
`XLLM_DSV4_DISABLE_FP8_BF16_CACHE=1` 做诊断；候选
`build/gsm8k_candidates/packed_fp8_activation_diag_20260925/xllm`。
该候选保持Graph+draft可运行，layer0 Q relative L2变成2.16%，但layer2输入仍约16.41%，
case55首token仍是28。20道选定难题按原NeMo评分7/20、零请求错误，记录在
`logs/goal_packed_fp8_activations_case55_20260925/diagnostic20.summary.json`。
它不是随机样本、不是新的全量准确率，也没有形成性能验收，不启用为默认配置。

后续优先：在同一case55的layer0/1，继续拆分HC post、FFN input、hash routing、routed
expert与shared expert输出；先解释layer2输入的大幅增长，再重测Graph+draft完整1319题
及同一binary性能。全目标依然未完成，不属于等待外部输入的阻塞。
