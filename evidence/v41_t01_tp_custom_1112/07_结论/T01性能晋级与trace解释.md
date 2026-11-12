# T01：只在D启用Attention TP custom归约，作为后续速度基线

**仅通过基础功能检查，未完成模型精度验收。** 本报告采用用户2026-11-12更新的速度优先口径；不做API对分、MathQA或固定前缀logits/logprobs晋级检查。历史数值/文本差异留档，不在本轮追查精度原因。

## 结论与边界

本次接受P关闭custom、仅D开启Attention TP custom的配置：两道简单题通过，两组各1暖机+5正式全部成功，输入/输出token计数正确，独立trace的16rank验证和前后实际图完成，P/D全部正常退出。本轮没有使用SIGKILL。之前P/D都开的第一轮失败单独保留在ToDoList_4，不能从记录中删掉，也不能声称已找到其长Prefill故障根因。

| 输入/输出 | 指标 | before P0/D0 | after P0/D1 | 变化 |
|---|---|---:|---:|---:|
|256/256|TTFT ms|354.1111|433.5875|上升22.44%|
|256/256|TPOT ms/token|179.5082|162.7300|下降9.35%|
|256/256|端到端 s|46.1287|41.9297|下降9.10%|
|256/256|输出 token/s，由256/平均时延计算|5.549690|6.105451|上升10.01%|
|256/256|QPS，由1/平均时延计算|0.0216785|0.0238494|上升10.01%|
|2048/256|TTFT ms|2348.8672|2355.8590|上升0.30%|
|2048/256|TPOT ms/token|180.8850|164.3116|下降9.16%|
|2048/256|端到端 s|48.4745|44.2553|下降8.70%|
|2048/256|输出 token/s，由256/平均时延计算|5.281122|5.784616|上升9.53%|
|2048/256|QPS，由1/平均时延计算|0.0206294|0.0225962|上升9.53%|

每格均保留全部五次正式值，完整原始工具report、SQLite与SSE在04_性能的raw目录及压缩包。官方工具吞吐有运行区间和取整差别，详见ab_summary.json中的official_evalscope_summary，不替换官方原报告。没有把输出少于256或失败请求用于提速。

接受的是TPOT和端到端输出速度的改进，不声称TTFT改善。短输入首包均值恶化保留：before有1个约200ms额外点，after有3个。按真实request ID核对，两侧P-stage TTFT均约295–312ms；高点的client−P差约203–222ms，而普通点约16–20ms，client E2E−D total也出现相近额外量。它缩小了缺失的计时边界，但还不能证明是网络、HTTP客户端或调度的哪一种原因。后续T07要继续排查，不能删除慢点再报平均。

![两组五次正式全部原始点](../06_图/两组正式性能全部原始点.png)

## 固定条件和两道基础题

权重只读引用`/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt`，revision `a41b72782c27dbd505c84c13fcefaabb1fe2f273`。两台各8GPU，1P1D；每台Attention TP8、专家EP8，二者使用同8张GPU，不是64张。eager、C1、chunk256、cache1GiB、block128；prefix、schedule overlap、Graph和MTP关闭，依赖沿用冻结StageI环境。

输入token分别256/2048，max_tokens=256且ignore_eos=true，temperature=0、top_p=1、reasoning_effort=none、thinking_mode=chat。EvalScope在SJ10容器内运行，1次暖机和5次正式分别计数。profiler在两组正式全部结束后单独启动，另发256/8请求。

两道题分别为“只输出计算结果，不要解释：1.3 + 2.9 = ?”和“只输出计算结果，不要解释：7 × 8 = ?”，得到4.2和56。此前归约基本检查覆盖8rank BF16/F32已知/随机输入与重复共享buffer依赖链。已做的2048功能检查、8-token公共logprob探针只作历史资料，本轮不以其模型数值一致性作为晋级条件。

## 从何想到、实际改了哪一层

G02在不把重profiler宽度当正式占比的前提下，发现TP/EP归约及其等待是优先诊断对象；先做了真实10KiB BF16、20KiB FP32的8卡微基准。已有xLLM `musa_custom_all_reduce.cpp` / `custom_all_reduce_ops.cu`支持这类小消息，可以直接接入，不需要重写整个通信库。家明/飞狐已有TP custom、GPU IPC及graph-buffer管理是参考来源，具体源码落点见计划来源及候选patch。

只改两个原生C++文件：

1. `xllm/core/layers/musa/deepseek_v41/decoder.h`：增加配置字段；q=1的Attention局部输出选择现有musa_allreduce_tp_custom，并接收它返回的新Tensor。没有把旧输入误当成归约结果。选中但不支持则明确CHECK失败，不在执行失败后偷偷切后端。
2. `xllm/models/llm/musa/deepseek_v41_native.h`：用已有、跨TP rank一致检查的XLLM_MUSA_ENABLE_TP_CUSTOM_AR构造上述配置。P显式0，D显式1。q>1 Prefill仍MCCL，路由专家仍FP32 MCCL，共享专家在归约后只加一次。

没有新增Python forward，没有改模型权重/量化/MATE算子，没有改变1P1D或减少模型层数。复用的是现有native MUSA kernel。实际binary SHA `20376a77ad348ffd889d2592aa982c7a6b35962c0bb3908bbe866d7eca1d8489`；源码2文件SHA、20.1.6格式化和构建记录均已核对。模型源码与被测编译产物逐文件对应，依赖没有换版。

## 真实trace：改善发生在哪

16rank、各40层、32个原始trace/stack文件已逐一核大小/SHA与RecordFunction ID/name/tid。P8日志均没有decode TP custom启用行；D8均有，且实际GPU kernel一致。以下比较来自同形状独立256/8诊断：8个输出中第一个在P产生，D有7步，40层×7=280次TP归约。

|D0整个抓取窗口|before|after|
|---|---:|---:|
|BF16 MCCL AllReduce kernel|280|0|
|BF16 cross_device_reduce_1stage kernel|0|280|
|FP32 MCCL EP AllReduce kernel|280|280|
|AllGather kernel|28|28|
|总GPU kernel|42147|42147|

因此**不是kernel次数减少**。前后是一个归约kernel换成另一个；新增custom路径还有D2D staging copy。P0整个窗口的40 BF16/40 FP32 MCCL归约不变。不能从D的改变夸大为跨机Mooncake也优化了。

D0最后一步40个TP区间，MCCL侧记录到400次运行时API，其中EventRecord200、StreamWaitEvent80；custom侧160次，主要为StreamIsCapturing80、MemcpyAsync40、LaunchKernel40。TP CPU包装scope总和约5.454→2.392ms。这些是重profiler下的机制观察：减少事件/等待管理并替换小消息归约实现，解释它为什么是可行优化；不能把这些数值直接从正式TPOT减去。

![before实际CPU/GPU时间线](../06_图/BEFORE_D0_实际原生轨迹/真实CPU区间与GPU时间线.png)

before的stream14主要看到交替BF16/FP32 MCCL归约；after的stream0出现单阶段custom kernel，stream14只保留专家FP32及原有AllGather。两图均来自原始真实区间。

![after实际CPU/GPU时间线](../06_图/AFTER_D0_实际原生轨迹/真实CPU区间与GPU时间线.png)

![before真实C++入口计数火焰图](../06_图/BEFORE_D0_实际原生轨迹/完整实际C++调用链_ENTRY_COUNT.png)

![after真实C++入口计数火焰图](../06_图/AFTER_D0_实际原生轨迹/完整实际C++调用链_ENTRY_COUNT.png)

火焰权重为540个真实语义入口的COUNT，不是CPU时间；它证明相同原生模型/层链在执行，宽度不能当占比。before/after最后一步重profiler分别约513.33/500.91ms，不能代替正式约179.51/162.73ms，也不拿两者比例解释全部9%改善。

部分MCCL GPU kernel缺CPU runtime/driver回调，不能伪造单scope逐条launch关联。完整step图保留全部窗口内GPU；custom TP的40次kernel则有真实correlation。严格可缩短全局预算和理论上限兑现率仍为unknown，不能声称已完成精确上限测算。

## 问题、决定与后续

第一轮P/D都开时长正式失败，已单独归档并在新授权下清理；此次只改变P开关做分离，不声称根因已解决。本次P/D的16rank全部exit0，结束后显存释放。trace离线脚本曾假定tar先出现stack定义；现已两遍读取，复用同一批原始文件完成核验，没有重跑模型来掩盖分析错误。addr2line对某些DWARF行信息有警告；实际ELF SHA和44个native函数符号已核对，不凭未知行信息编来源。

接受本配置作为速度基线，下一项首先验证同一communicator上BF16 TP→FP32 EP交替复用，再单独接入FP32 EP custom；之后按新的实测剩余热点推进缓存读侧、MoE、mHC、Graph等，并继续到T14。若EP微基准不成立或整模型无收益则保留失败并回到此基线，不把理论收益相加。

所有模型现象只记录，不扩展精度排查。主goal范围已按用户新增要求扩展到T14；当前并未完成全部队列，没有60% SOL口径，不宣称达标。

## 可复核路径

- before完整原始：`../ToDoList_4/04_性能/t01_off_v1_raw`（原NFS R/new_start_t01_off_v1）。
- after完整原始：本目录`04_性能/t01_donly_custom_v1_raw`及同名tar.gz；NFS R/new_start_t01_donly_custom_v1。
- 完整原始trace：R/new_start_t01_{off_v1,donly_custom_v1}/trace/raw；R=`/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp`。
- after原始trace archive SHA `7625708a24dce62a629faf9e6e885df0331a2da5d0ad44f0785b099acd74c13c`；before `aad107c1fa26a1310b8790dc7f72f46cadca8f16f6019801d0e14f73ab2a7c7d`。
- 本地16rank摘要、完整窗口通信计数和API对照在05_trace；可读图在06_图。累计最优运行manifest.json固定本配置、源文件SHA和数据。
