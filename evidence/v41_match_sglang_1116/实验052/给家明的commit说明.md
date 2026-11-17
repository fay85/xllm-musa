perf(musa): 我将直接 RoPE 表接入 V4.1，首轮达到追平速度门槛

家明，我在累计最优原生 C++ 路径上，将短输入的 RoPE 取表、转换和计算合为 GPU 内核，保留 FP32 表、BF16 输出和原 FP8 QDQ。

SJ9/SJ10 双机1P1D、Attention/共享专家TP8、路由专家EP8、target/draft完整Graph、DSPARK proposal3；同W4AFP8-mt1017、实际256/256、C1、各独立1暖5：
- 关闭：TTFT 297.10ms，TPOT 25.85ms/token，端到端输出37.17token/s。
- 开启：TTFT 325.86ms，TPOT 21.01ms/token，端到端输出45.04token/s。
- 开启组首次同时达到21.57ms、39.45token/s、990.04ms三个门槛；还不是最终三轮验收。

独立trace确认target图节点3967降到3367；同时投机轮数122变为107、输出文本改变，所以我没有把全部提速归因于RoPE。局部数值误差和失败原型均保留，不宣称逐位等价。相关检查、两道计算题、长输入冒烟与回短通过，32个rank均正常退出；仅通过基础功能检查，未完成模型精度验收。

原始工具报告、数据库、SSE、服务日志、检查结果、图和配置在evidence/v41_match_sglang_1116/实验052。完整trace/ELF保存在NFS，目录及SHA见结论。被测ELF为7d2ffa0662fd53f0074d6f29c7a28e0f4db5e057cb1c0ad6febc706bfc74303f。

我继续尝试RoPE与block32 QDQ融合，随后按计划完成后续候选、最终同版本三轮、2048/256正式回归和同窗口SGLang复测，不因首次达标提前结束。
