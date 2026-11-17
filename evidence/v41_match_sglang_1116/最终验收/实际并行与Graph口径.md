# 以实际参数解释并行和Graph

本轮目标是每台8卡的双机1P1D，允许调整并行。当前E56/E57与E52源代码、ELF一致，后续最终配置仍以各轮role_config.json和实际/proc/environ核验为准。

|模块|P/SJ9|D/SJ10|
|---|---|---|
|Attention|TP8|TP8|
|路由专家|EP8，本地专家后归约|EP8，本地专家后custom归约|
|共享专家|复制完整计算，EP归约后加一次|TP8切片，与路由输出在同一归约中合并|
|Graph|P前向eager|原生target完整decoder Graph及draft Graph|
|投机|服务配置DSpark proposal3|proposal3，target verify q4|
|短上下文Indexer|关闭|E56起开启，超预算恢复正常score/top-k路径|
|缓存|chunk256、prefix cache关闭|静态图桶、commit/snapshot与缓存基本检查已做|

不要简单写“两边所有模块都是TP8”。共享专家依据 xllm/core/layers/musa/deepseek_v41/moe.h:212及242、models/llm/musa/deepseek_v41_native.h:926与1254；P开关0、D开关1。它没有导致共享专家漏算。

通用CLI --enable_graph=false 不代表D以eager运行。原生V4.1专用 FULL_DECODE_GRAPH/FULL_DRAFT_GRAPH 开关为1，实际trace捕获目标图和draft图回放。正式测速 --enable_online_profile=true 只开放控制端点，没有start_profile；最终identity/profiler状态及无armed文件共同证明计时阶段profiler关闭。

D并非冻结SG B的DP8 Attention/DeepEP LL配置；后者是另一个真实实现，必须在比较表中列出，不能说两框架所有并行方式相同。共同条件是双机16卡、同一模型资产、同实际256/256/C1请求和统计口径。

P共享专家TP/CP等仍可能进一步缩短TTFT，但当前缺单独暴露耗时证据，不能宣称已经实现或估出百分比。本轮短请求E2E以D为主；后续并发/长输入再按全局实测排序。
