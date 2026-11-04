# 快速精度验收协议：当前固定版本

> 用户最新决定：停止本轮后续发题与复测，本地冻结65个初始HTTP200响应，完整100题未完成；本文件描述已采用协议，原计划的后续步骤不再自动执行。

## 目的与比较边界

比较智谱官方 `glm-5.3-flash` API 与当前本地 xLLM 的同题结果，建立当前 BF16 eager、1P1D P/D各TP8 的服务级参考基线。不是复现 ModelScope 官方榜单，也不能单凭服务间答案差异隔离出硬件或某个内核的影响。设备实际GMI名称为X10000，每台8卡、每卡81920MiB；与S5000销售型号的对应关系本轮未独立核实。

## 数据与完全相同的请求参数

使用作者原始 MathQA test 2985题的前100题，ID `mathqa-test-0000`～`0099`，题序和选项顺序固定。作者原始test JSON SHA256为 `dfe7bc4691caf26842ccd4cf14e8f978327bab4f2989e0b076a4e6b38a9371d1`；固定题集为 `评测/mathqa_author_first100/test.jsonl`。

每题独立对话。两边Prompt、选项顺序、temperature=0、top_p=1、reasoning_effort=low、max_tokens=4096、stream=false完全相同。允许思考，最后要求 `ANSWER: A/B/C/D/E`。max_tokens是上限，不强制生成4096token。冻结请求位于 `评测/paired_protocol_v2/requests_without_model_or_credentials.json`，SHA256 `e0091d5c2c923243e14681f899b49b70f02a49ba661404bad9186985e6f35c4b`。

第三方镜像原先有5题选项缺损，已改用作者数据；旧v1完整保留，只有题目、标签和请求完全一致的响应才复用，审计见《MathQA数据镜像缺损与修正.md》。不能混用不同数据版本的分数。

## 思考参数的验证

官方和本地原始JSON都保存low参数。实际Router→Prefill请求的protobuf字段35已观察到 `chat_template_kwargs={"reasoning_effort":"low"}`，见 `证据/own_chat_protocol_fields.json` 与新Router抓包。模板对low会加入对应指令，不能用“HTTP200”单独证明参数生效。

补充实证：已将真实转发的0033题全部155个输入token，与相同题目通过当前模型模板生成的token逐个比较，全部与low一致，与high/max不一致；解码的系统前缀确实为 `Reasoning Effort: Low`。这比只检查字段传递更进一步，证据为 `证据/actual_low_effort_tokens_verified.json`。只验证已抓取请求，不夸大为逐题都已抓包；所有题仍由同一固定客户端与服务路径执行。

相同low参数不保证双方内部的思考预算实现、隐藏系统处理、模型数值路径完全相同；实际生成token数也不会因此必然相同。官方内部参数实现不透明，不能把不同生成长度直接判断成请求使用了不同思考力度。

## 评分

官方只提取最终content，不从reasoning_content猜答案；本地在最后一个 `</think>` 后提取最终选项。本地既无该边界又无独立reasoning字段时标记格式无效。严格要求ANSWER格式，但只有一个合法裸字母时可提取，并另外记录格式偏差。

无答案或多义答案为格式错误；finish_reason=length为截断，不能算正确。统计分母保持固定100题；HTTP错误、格式错误、截断、正常回答但标签不匹配分别列出。遇到题面/标签疑点仍按原标签评分，不删题、不改答案，另写解释。

报告包含两边Accuracy、本地减官方百分点、全部100题的一致率、有效配对上的一致率、逐题清单、格式/截断、输入/输出token总量。100题是快速筛查，不是统计意义上的严格等价证明。

## 用户新增：不一致题双方各追加三次

对双方初始答案不一致的题目，逐题读取两边已保存的请求，断言请求体完全相同，再各执行3次。目录为 `评测/paired_protocol_v2/discordance_repeats/<题号>/{official,local}/repeat-01/02/03.json`。

复测保存原始response、request、请求哈希、原始结果哈希、参数、最终选项、finish_reason、token数和用时。三次结果独立用于诊断稳定性，绝不替换初始100题得分，也不采用best-of或多数票来美化主分数。temperature=0不保证不同服务或同一服务多次调用逐字一致，三次复测也不足以证明不存在任何随机波动。

本地请求顺序执行，不与测速或其他本任务推理重叠。配额错误仍立即保存并停止官方调用。

## 截止时间与失败处理

本地当前使用 `http://10.121.32.9:49880/v1/chat/completions`。新Router启动时设置3600000ms业务期限，客户端read timeout3900秒。真实请求字段48与ExecutionPlan预算均已确认约3600秒；并未改变采样、思考力度或模型实现。

0031原先两次900秒504均保留在 `local_failed_attempts`，待其他题结束后原请求补测。HTTP超时不会保证GPU请求立即取消，需等待D端running=0再发新请求；禁止为赶进度反复stop/kill。

旧运行时flags改为3600秒但实际仍900秒的反证、额度充值、v1→v2迁移及各时间点进展保留在 `证据/快速精度验收协议_截至1633历史.md`，当前有效协议以本文为准。

## 当前实现与后续范围

本轮不接入FP8 MegaMoE、不切换graph、不改变被测模型代码。用户最新要求停止当前剩余精度与两组性能，改为归档现有部分结果并生成汇报。未来新实现必须重新完成其本地对分，不能沿用本轮分数作为新路径正确性证明。
