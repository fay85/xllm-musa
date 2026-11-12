# T01 第二项：Decode 路由专家 EP custom-allreduce

**结论：接受为新的速度基线。仅通过基础功能检查，未完成模型精度验收。**

在已接受的 Attention TP custom 基线上，只把 D 端 q=1、FP32 路由专家输出的归约改成现有 custom-allreduce。P 端不改，D 端 Attention 仍保持 TP custom，shared expert 仍在路由专家归约后只加一次。没有更换权重、量化方式或少算专家。

## 同一编译产物下的正式对照

SJ9(P)/SJ10(D)，各8卡，1P1D、TP8/EP8、eager、并发1、chunk256、Graph/前缀缓存/MTP关闭。每组1暖机+5正式，官方 EvalScope 不修改。两模式用完全相同 ELF `ef2cbf5b6afadd17f104d63a2e393f65a09c59424a148122c51a2def371c55fb`。仅改变 `XLLM_DSV41_DECODE_EP_CUSTOM_AR` 的 D 端0/1。

| 输入/输出 | EP模式 | TTFT ms | TPOT ms/token | 端到端 s | 输出 token/s | QPS |
|---|---|---:|---:|---:|---:|---:|
|256/256|off|398.359|162.478|41.830141|6.119989|0.023906|
|256/256|custom|398.052|146.687|37.803318|6.771892|0.026453|
|2048/256|off|2603.157|164.889|44.649894|5.733496|0.022396|
|2048/256|custom|2417.883|148.592|40.308829|6.350966|0.024808|

两组TPOT分别下降9.72%、9.88%，端到端输出吞吐分别上升10.65%、10.77%。输出吞吐按256/平均请求端到端耗时计算，完整官方工具报告另行保留。每种模式10次正式请求均成功，实际输入数分别256或2048，输出均256；对照请求内容哈希完全一致，没有用失败或缩短输出换速度。

2048组TTFT从2603.16降到2417.88ms是这批样本的观测；P的算法和通信没有改变，不能单凭这一差值把TTFT改善归因于D端EP优化。累计相对首次全MCCL基线，TPOT约从179.51/180.89ms降至146.69/148.59ms；正式归因以前后同一ELF的本项对照为准。

## 为什么想到、改在哪里

G02与上一轮trace显示，D端已把BF16 Attention TP换成custom，但每层还保留一次20,480字节的FP32 EP MCCL归约。在本配置中EP8和TP8采用相同8个rank顺序，可复用已有IPC小消息实现。`moe.h`在q=1且显式开关开启时消费custom helper返回的Tensor；不支持的输入直接报错，不能执行失败后悄悄切回。`deepseek_v41_native.h`在启动时让所有rank投票核对同一开关，并检查EP=TP、TP custom已启用。其他布局不能直接套用。

## trace 能证实什么

前后各16rank的原始trace、native backtrace全部核验大小/SHA和Record function关联；40层齐全，同一256输入/8输出诊断请求，D0观察q=1、kv=263。

- 前：每层仍有FP32 `imcclDevKernel_AllReduce_Sum_f32_RING_LL`；后：对应归约替换为 `cross_device_reduce_1stage<float,8>`。BF16 TP custom保持不变，P端BF16/F32 MCCL保持不变。
- 40个EP CPU调用内的runtime API，从400次降到160次：原来包含200次EventRecord、80次StreamWaitEvent等；现在是80次StreamIsCapturing、40次MemcpyAsync、40次LaunchKernel。
- 不是把40次归约少算成几次，也不是GPU kernel launch数量减少；主要替换了小消息通信实现并减少相应CPU事件/等待管理。custom需要D2D staging copy，代价也在实际trace中。
- 选中的CPU step窗口里有39个完整归约kernel，最后一个异步kernel可跨出CPU区间。不能把窗口内39误写成整步只执行39层；CPU语义scope有40个，完整原始trace和实际关联保留了窗口外异步事件。
- C++火焰图的宽度是入口COUNT，540个真实入口，不能当CPU采样耗时或用图宽直接算收益。重profiler下step耗时也不能替代上表的profiler-off正式TPOT。

严格可消除关键路径占比仍未知，因此不声称已兑现理论上限百分之多少。当前证据支持通信路径发生预期替换，并且正式服务性能变快；不能把inclusive MCCL区间全部当作可移除时间。

## 基本检查、退出与精度边界

两模式都回答4.2和56。额外做过100次依赖串联的BF16 TP→FP32 EP、同一IPC buffer复用检查，8rank均与参考精确相等、自然退出。该微测试只验证算子和顺序，不代表完整模型精度。未运行官方API/MathQA/固定前缀logits或logprobs。本次不追查模型文本/数值差异。

两模式P/D所有rank均正常退出0，未发送SIGKILL，没有重启主机或改驱动。

## 产物与后续

- 原始正式报告、SQLite、服务日志：[04_性能](../04_性能/)。
- 数值基本检查与原始压缩包：[03_正确性](../03_正确性/)。
- 原始trace位置和校验：[processed_trace_bundle_v1.receipt.json](../05_trace/processed_trace_bundle_v1.receipt.json)。大trace保留SJ共享盘，具体路径见其中raw_archives。
- 通信实际scope/API：[通信路径确认.json](../05_trace/通信路径确认.json)。
- 图：[优化前](../06_图/BEFORE_D0/)、[优化后](../06_图/AFTER_D0/)。离线渲染的完成状态以各目录rendered.json为准，不以文件夹存在为准。

下一项T02是当前forward内复用压缩缓存的只读materialization，已通过32个实际GPU/C++ cache检查，正式对照待启动。后续仍继续T03～T14，不以本项成功为全任务完成，不承诺60% SOL或“几十ms”已达成。

## 图文核对

![全部正式样本](../06_图/两组正式性能全部原始点.png)

优化前，D0的stream14周期性出现FP32 MCCL归约；优化后该路径换成stream0上的FP32 custom kernel，stream14只保留相关AllGather。注意原始stream17等其他算子轨道没有因此消失。

![优化前真实时间线](../06_图/BEFORE_D0/真实CPU区间与GPU时间线.png)

![优化后真实时间线](../06_图/AFTER_D0/真实CPU区间与GPU时间线.png)

![优化后真实C++入口火焰图](../06_图/AFTER_D0/实际C++回溯_ENTRY_COUNT火焰图.png)

前后图均来自本轮实际抓取和对应ELF符号化，已打开检查；火焰图下半部分完整调用链另见同目录的完整调用链PNG。时间线约506ms是重profiler诊断窗口，不能当作正式TPOT，也不能仅凭两张图的总宽度求速度提升。
