perf(musa): 完成V4.1双机推理累计优化及T14验证

家明，我在SJ9/SJ10上继续推进了V4.1 Flash W4AFP8-mt 1017的原生C++路径，
把本轮G01/G02及T01～T14的实现、实测和未完成边界整理到同一份报告中。
P/D各8卡，TP8/EP8、DP1/CP1，保留eager。每组先暖机，再正式请求5次，
核对实际输入/输出token数和成功状态，性能测试不打开profiler。

单请求冷输入：256/256使用chunk256，TTFT 347.04ms、TPOT 66.60ms，
端到端17.3313s、输出14.7710token/s；2048/256使用chunk512，
TTFT 1510.82ms、TPOT 67.31ms，端到端18.6741s、输出13.7088token/s。
它们是分别验证的两个长度配置。相对本轮起点，两组TPOT约降低62.9%/62.8%；
短输入TTFT从314.05升到347.04ms，没有把这一项也说成改善。

我保留了此前有效的TP/EP custom all-reduce、缓存复用、SWA读取融合、
共享专家TP和Decode模型stream；本次mHC尾归约融合的独立对照让TPOT降低
10.51%/13.13%。前后实际trace确认每步少400次GPU kernel launch，
sum/div/rsqrt等分散调用被融合，矩阵计算未被跳过。
长输入chunk256改512后，P从8次q256变为4次q512，TTFT降低43.92%。
重复长Prompt命中1792个prefix token时另有收益，单列为热缓存负载。

我还完成了真实多请求Decode批处理。并发2的聚合输出为16.8045/16.4516token/s，
比同并发q1排队提高16.41%/10.74%；并发4为25.5344/24.3334token/s，
比同并发q1排队提高66.87%/58.56%。这是总吞吐，不能当成C1单请求速度。
对应实际q分布、全部rank的SHA/原生调用链和GPU事件均保留。
C2单独叠加小批mHC只得到约+1.08%/-0.10%的吞吐变化，没有单独晋级；
C4按同一small-MHC配置对照，包含batch触发多行融合的效果，不做虚假拆分归因。

Graph、packed投影、Gate缓存和CED的负/混合结果都保留，未强行打开。
原生DSpark/MTP已接入实际draft、辅助hidden、状态恢复/回放和P/D，
并修复drafter的512/576缓存维度识别；但接受率约20～23%，
额外验证/回放令TPOT升到172～174ms，所以默认spec0。
T08完成搬运日志分析，T13完成容量与接口分析；没有声称做成新网络协议或TP4布局。
删除逐层日志后P端长请求曾RPC超时，仍有时序可靠性问题，需要下一轮优先处理。

仅通过基础功能检查，未完成模型精度验收。本轮没有跑MathQA/API/logits对分。
改动相关的算子、通信、缓存短回归保留；正式请求没有靠变短或失败取得好成绩。
SOL口径仍待确认，不把TPOT下降62.8%说成达到60%SOL；局部关键路径占比未知时
没有编造理论收益兑现率。CPU trace审核与后续轮次有后台重叠，已记录测量边界，
小幅差异不作为稳定收益。下一轮先做P端可靠性、batch Attention/metadata，
再考虑MTP有效工作比和其他并行布局。

代码分支：
https://github.com/fay85/xllm-musa/tree/xiaopangV41FlashCppFromSglangtoXllm
本提交收录30个已编译源码文件、11个短回归脚本、真实ATen trace离线转换和证据。
30文件与已测E51快照逐字节对应，最终ELF SHA前缀f54260ea；C1历史正式ELF931741c3
单独注明，没有伪称在最终ELF上重跑过C1。镜像1102-dev，实际mate元数据0.2.7，
显式加载的MATE/SGLang修正SO另以SHA对应，没有笼统称为最新0.2.8。

可先阅读evidence/v41_core_optimization_1113/README.md。
原始测速SQLite/SSE/配置/服务日志的小归档、简表与真实截图随本提交保存；
大trace、交互Perfetto和ELF保留在NFS并记录大小/SHA，不提交权重或凭据。
结构化NFS：
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_new_start
原始运行根：
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp
短输入原始报告在new_start_t14_native_cache_off_v1/performance/input256_output256；
长输入在new_start_t09_chunk512_cumulative_v1/performance/input2048_output256；
并发4在new_start_t12_batched_c4_final_v1/performance，两组原始报告均在其中。
只提交共同仓库个人分支，不创建MR/PR。
