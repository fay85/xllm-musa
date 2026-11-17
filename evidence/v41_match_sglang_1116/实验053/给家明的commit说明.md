docs(perf): 我完成 RoPE 与 QDQ 融合对照，未发现稳定收益并保留原方案

家明，我在累计最佳的直接 RoPE 表实现上，尝试把 RoPE 与 block32 QDQ 合成一个 GPU kernel。相关生产算子检查通过，正式实验使用同一 ELF，只切换本项开关。

SJ9/SJ10 双机1P1D、实际256输入/256输出、并发1，每组1次暖机后5次正式请求：
- 关闭：TPOT 21.1716ms/token，输出44.9844token/s，TTFT292.10ms。
- 开启：TPOT 21.1900ms/token，输出44.5465token/s，TTFT343.36ms。
- 反向关闭复测：TPOT 21.3581ms/token，输出44.6007token/s，TTFT293.50ms。

三组15次请求全部成功且实际长度符合要求，输出文本与投机接受轨迹相同。独立trace确认target每图节点3367降到3287、draft从232降到226，但端到端没有稳定收益，所以这项不晋级，当前生产源码仍保留E52。重型profiler扰动明显，不能把图内节点减少直接当成正式性能提升。

本提交保存原始小日志、EvalScope数据库、逐题统计、真实Perfetto图与失败判断；大trace和ELF保存在NFS /data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116，原始包SHA与完整路径见evidence/v41_match_sglang_1116/实验053。三组48个rank均正常退出，没有SIGKILL。

累计最佳E52单轮为21.0099ms/token、45.0437token/s、TTFT325.86ms。下一步先完成proposal和draft Gate的候选对照，再做最终同版本三轮、2048/256正式回归及同窗口SGLang B复测。目前只是通过基础功能检查，未完成模型精度验收，也尚未完成整个goal的最终验收。不提MR。

