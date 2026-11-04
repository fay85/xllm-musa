# GLM-5.3-Flash MUSA 1P1D：阶段结果与停止点

本轮已按用户要求停止扩展精度与性能测试，转为归档已有实现、过程、问题和后续计划。**本地完整100题未完成，两组正式性能未运行；不宣称完整精度验收或性能优化完成。**

## 实现与部署

- 仓库分支：[fay85/xllm-musa / xiaopang1104benchmark1P1D](https://github.com/fay85/xllm-musa/tree/xiaopang1104benchmark1P1D)。
- 被测实现提交：`30b7051ef1c840a9635794e6950f7507ee01b829`，基线`7005a557b577955407f619adf7871bbcff19cb34`（feihu-1016-dev）。本次结果提交与被测实现提交分开。
- 1102-dev工具链重编xLLM worker，运行时MATE0.2.8.post1、TileLang0.1.12+musa.5；控制服务沿用镜像预编译版本。
- sj9 P(TP8) + sj10 D(TP8)，总16卡；MoE EP1、eager、custom-allreduce关闭、MCCL机内TP通信、Mooncake/RDMA跨机状态交接。
- 设备GMI报告X10000，每台8卡、每卡81920MiB；不将未独立核实的销售型号映射写成事实。
- FP8 checkpoint在加载时按TP分片并反量化为BF16，随后GPU计算。**不是FP16，也不是原生FP8 MegaMoE路径。**

## 对分数据与参数

使用MathQA作者原始test集前100道固定五选一数学文字题，涵盖比例、利润、行程、平均数、几何等；不是AIME/AMC。双方相同Prompt、选项顺序、temperature=0、top_p=1、reasoning_effort=low、max_tokens=4096、stream=false；允许思考，最后提取A～E选项。

AIME/AMC等只核对过模型卡成绩与协议，未运行；没有可用对应分数，不能称官方对齐。AutomationBench v1.0.6有开源框架，但当前部署只返回正文工具XML而不是结构化tool_calls，前置门槛未通过，按约定回退MathQA，没有运行5/20题，也没有为此次额外开发Agent框架。

## 一目了然的结果

| 范围 | 官方 | 本地 | 解释 |
|---|---:|---:|---|
| 官方完整100题参考 | 93/100，93.00% | 未完成100题 | 不能与不同题数的本地百分比直接相减 |
| 双方相同的65题 | 62/65，95.38% | 60/65，92.31% | 本地减官方−3.08个百分点；包含本地0065截断这一失败样本 |
| 相同65题答案一致率 | — | 63/65，96.92% | 0062选项不同；0065本地没有最终答案 |
| 双方都有有效选项的64题一致率 | — | 63/64，98.44% | 仅用于补充观察，不能替代上行分母 |
| 同65题输入token | 8419 | 8419 | 实际usage逐题核对，无输入计数差异 |
| 同65题生成token | 7358 | 11938 | 包含思考；生成长度不同不能直接证明effort设置不同 |
| 正式256输入/256输出性能 | — | 未运行 | 用户停止本轮测试；没有TTFT/TPOT/QPS正式数值 |
| 正式2048输入/256输出性能 | — | 未运行 | 预热与测量方案已准备，留待未来新路径 |

官方完整100题输入12779、输出13776token；1个裸字母格式偏差按可提取选项评分，0截断。本地65个HTTP200响应有1个截断/无有效答案，即0065；另0031保留两次HTTP504，0066～0099共34题未运行。未运行题不伪装成答错，也不从已完成子集中推断完整100题准确率。

## 两个关键样本

| 样本 | 初始结果 | 追加诊断 | 能说明什么 |
|---|---|---|---|
| 0062，星期/天数文字题 | 官方D、本地B，双方正常结束 | 官方D/D/D（426/291/84token）；本地B/B/B（均54token且内容相同） | 当前三次观察中差异稳定；不能归为一次偶然抽样，也不能直接定位内核原因 |
| 0065，分段佣金文字题 | 官方A；本地生成4096token后length截断，无最终选项 | 官方A/C/A（340/729/486token）；用户停止时本地一次追加请求已在途，独立收尾、不计入65题 | 官方同参数也出现选项波动；本地存在长推理/截断问题，题面计算137.5台与选项不相容 |

重复结果不替换初始评分，不采用best-of。实际抓取的0033请求全部155个token与low模板逐个一致、与high/max不一致，前缀为Reasoning Effort: Low；官方内部实现仍不透明。0062/0065的题面疑点单独说明，固定标签不改。

## 已完成的底层适配与回归

复用MATE KDA、RMSNorm、mHC pre；补MUSA模型注册/metadata、GPU参考attention与外围操作、FP8离线载入、TP分片的KDA状态交接、Folly/bthread兼容和后台线程退出。代码相对基线34文件、1600新增/37删除；无新增底层.cu/.mu kernel。

KDA5项、RMSNorm/mHC pre5项、Folly3项、缓存布局23项、FP8加载10项、生命周期/ABI5项通过。Python集合首次347通过/115跳过/1失败，修复夹具后仅该项定向复测1通过，不能写成全套重跑全绿。随机fixture单机TP8/跨机1P1D在4/17/129输入结果一致；真实模型3道基础问答通过。

## 过程中的主要问题与处理

- 第三方MathQA镜像5题选项缺损：改用作者数据，保存旧记录，仅完全相同的题目/请求复用，未择优换答案。
- v3增量构建混用类布局ABI：重建相关native对象得到v4，补跨翻译单元回归。
- 0031两次900秒HTTP504：runtime flags可查为3600秒不代表请求预算生效；另起我方容器内CPU Router，抓包确认实际约3600秒。
- 0065推理结束但HTTP响应延迟：观察到VPN连接发送队列未确认与重传，仅对我方客户端socket启用保活后收到原响应，未重跑该题。该题GPU请求约2873秒、客户端约3132秒，不当成正式性能。
- Tool Calling当前部署未通过：按约定回退MathQA，未虚构AutomationBench成绩。

## 为什么暂未接MegaMoE、为什么使用BF16

当前路由专家和共享专家正常参与计算；尚未完成的是原始FP8专家权重布局、EP进程组、共享专家归约及GLM语义回归的接入验证。先用BF16是为了先核对模型结构、KDA状态和1P1D通信，减少同时变化的因素，并非MATE没有FP8能力。

此前“split-prefill不支持activation_clamp”的说法已纠正：安装MATE已接收并传递该参数。真正缺口是我方集成验证。MegaMoE已归约的路由输出不能重复all-reduce，共享专家不能漏算；权重和scale、gate/up顺序及GLM路由/限幅语义需逐层验证。这些错误可能不崩溃却改变答案。

## 除MegaMoE之外的完整适配边界

KDA卷积/gate/递推状态、mHC流组合、DSA/IndexPool缓存与选择语义、权重/量化契约和真实1P1D状态交接仍需逐层参考对齐与边界测试。当前这些多数已有可运行实现，清单是待验收范围，不代表已经证明它们算错。Tool Calling/reasoning解析有服务功能缺口；MTP/视觉等未纳入本轮文本基线。Graph、融合算子、FP8缓存及通信优化另列为性能/额外能力，不与模型数学结构混淆。官方服务内部细节不完全公开，不能声称仅凭当前对分实现完整复现。

## 后续计划（本轮不实施）

先接入MATE FP8 MegaMoE，利用已有算子及Sivan实现；完成算子/通信数值回归，跑基础问答和10题；再重新完成本地100题，逐项研究0062/0065及其余问题，并做2048输入/256输出预算的固定题正确性对比；之后才进行256/256、2048/256两组正式性能与必要trace。不能预设MegaMoE会自动修复本轮两个问题，也不提前承诺性能收益。

## 原始记录入口

- [精度汇总](accuracy/summary.json)与[100题状态/逐题表](accuracy/per_question.csv)：明确完成、HTTP失败和未运行，不补造缺失答案。
- [官方初始响应](accuracy/raw/official/)、[本地初始响应](accuracy/raw/local/)、[复测](accuracy/raw/discordance_repeats/)、[HTTP失败历史](accuracy/raw/local_failed_attempts/)。
- [冻结协议](accuracy/protocol.json)、[固定题目](accuracy/questions.jsonl)、[评分协议](accuracy/evaluation_protocol.md)、[问题分析](accuracy/analysis.md)。
- [回归原始日志](raw_logs/regressions/)、[完整模型P/D运行日志](raw_logs/runtime/)、[三道基础问答](raw_logs/smoke/)。
- [性能未运行说明与计划](performance/README.md)：没有正式测量样本或TTFT/TPOT/QPS结果。
- [详细模型适配说明](model_adaptation.md)：包含MegaMoE未接入、FP8转BF16原因、KDA/mHC/DSA等结构和五个框架层次。
- [源码/二进制/依赖清单](provenance/source_binary_runtime_manifest.json)与[发布文件哈希/脱敏清单](provenance/publication_manifest.json)。

本目录是按用户停止决定冻结的阶段快照。单个停止前已发出的0065追加请求由只接收程序收尾，没有继续发题，其状态/结果独立存放于[停止时在途记录](accuracy/raw/stopped_inflight_0065/)，不计入初始65题。快照不等于承诺GPU已卸载。

本地完整实验原件和私人工作资料没有全部放入仓库。这里包含与本轮结论直接有关的记录；API/SSH凭据、私人聊天文档、第三方SDK源码不发布。必要脱敏在publication_manifest逐文件标明，测量结果不改写。

## Offline verification

Run `python verify_results.py` in this directory to check file hashes and recompute the partial paired results from raw responses. No model/API call is made.
