# DT6：2000/1024 的 target、draft 和写回实测

PSQ 配置独立重启，不重跑 S2Ret 或 A3。仅中间请求用服务已有的 arm 文件启用设备 event，未计入正式性能。按 SSE ID 对到三个唯一的 D 端 `sequence_terminal`，3/3 均为实际 2000 输入、1024 输出、length、零前缀，703/1620；正文哈希均为 `48051d2b…`。暖机/开启采样/关采样复核的 HTTP E2E 分别 **165.151 / 18.944 / 18.561 s**，暖机含建图；这不是 EvalScope 计分，后两条的差也不是可直接减去的固定采样成本。

设备 event 每 8 步取 1，8 个 D worker 的采样步号一致（1,9,…,321；各 41 步），仅属于中间的请求。以下取每 worker 各阶段平均，再取八个均值的中位数，不跨 rank 相加：

|边界（GPU stream span）|ms/采样步|
|---|---:|
|准备 decode|0.007|
|draft|4.500|
|target inputs ready|0.016|
|target forward + head|46.350|
|rejection / accepted host ready|0.104 / 0.108|
|context cache|1.529|

若 324 块的接受轨迹不变，仅补齐 TPOT 需约 4.91 ms/块。写回就算全部省掉也不够；target 区间约需省 10.6%。46.350 **不是** Indexer 或 KV gather 花费，也不是总和都能回收。4096→3072 只有 25% 的容量变化；例如 select 的 radix 宽度会上取整到 2 的幂，并不随逻辑宽度等比缩短。不能据此声称已省 4.91 ms。证据：NFS `01_candidate/DT6/runtime_timing/audit.json`、同目录原始 SSE/event、D rank0 终态；本地 `sj10-dt6-audit.json`、`DT6/{sj9,sj10}-released.json`。

**下一步的单变量与退出条件**

- NSQ：PSQ 上只令 P 模型使用非默认流。代码在有效非空流上可让 Torch/TVM-FFI 同流；保留 FFI 准备策略、collective 当前流完成 fence，不复用历史非默认流停滞作成功证据。考察 TTFT、接受轨迹和完整 E2E；先做无 profiler 的 1+2 及基本题。单独 2000/32 D-only trace 核对 q=6 target 真实节点，不能代替 2000/1024 计分。
- 若 NSQ 未达标，SAQ 从 **PSQ** 开始，只切换 P 路由专家的激活量化执行路径：原 checkpoint 静态 scale + 对应 Apertoken GEMM，P 端的开关和 ABI 必须匹配；D 本来就是该路径、保持不变。普通 Linear、权重文件、cache 与计分均不改。假设来自冻结 SGLang 的 MoE 量化和本工程 E39；E39 的 D/短输入收益不外推到 P/2k，数值轨迹可能改变。两道题仅作基础检查。
**已有快筛与下一步（11/19 19:28）**

|组别|1暖后计分 n|TTFT ms|TPOT ms/token|E2E s|每条 accepted/proposed|
|---|---:|---:|---:|---:|---:|
|NSQ|2|1114.039|17.191|18.701|703/1620|
|SAQ|2|1270.442|15.793|17.427|732/1470|

两组暖机和计分都核对实际 2000/1024、SSE ID、D 终态 length、零缓存，两道实际 2k 基本题均答出 4.2 和 56。NSQ 与 PSQ 正文相同，P 内部 TTFT 从 1024/1039 降为 911/907 ms，但客户观测的均值仍慢于 S2Ret 49 ms，E2E 没有提高。后验在 SJ10 用单调时钟与当地墙钟对齐：两条 NSQ DB 完成时间都紧贴 D 终态，而客户端 E2E 减服务端总时长分别约 204/206 ms；这部分差出现在请求前部，并非可直接删掉的计时偏置，不能以服务端 TTFT 代替 EvalScope。

SAQ 是迄今最快的 **2k 快筛 E2E**，未晋级。相对 PSQ，提议块 324→294，E2E 快 1.221 s；相对 S2Ret 仍分别慢 205.425 ms、0.241 ms/token、0.452 s。不能把变化全算作量化内核加速，SAQ 正文与 PSQ 不同。实际 SGLang 与 SAQ 的提议数现均为 1470，不代表相同执行路径。SNQ 已于双方释放后启动：从 SAQ 只改 P 非默认流，以验证是否能保留 294 块并缩短 Prefill。NSQ 的单轮有限采样未证明 P 流长期稳定。

NSQ 独立 2000/32 D-only trace：三次 off/on/off 的 HTTP E2E 为 1.801/7.452/1.990 s。按同线程 CPU `V41/step/q=6` 内 `musaGraphLaunch` 的 correlation 连接 D rank0 GPU kernel，16 个 target replay，每图 3349 节点；不计 draft/eager。score/select 各 8 次，计时分别平均 0.367/0.615 ms/图；40 次 fused SWA verify read 为 2.581 ms/图，含复制 compressed_rows。两种 custom reduce 合计平均 5.514 ms/图，不等于可消去的净通信。单个 AllGather 平均 3.347 ms、中位仅 0.692 ms，第一图 45.004 ms，故不拿其均值估算收益。NFS `01_candidate/NSQ/trace/rank0_target_detail.json` 与 `rank0_selected_kernel_families.json`；整个模型不能按这次短 trace 替代实测。

在 PSQ 的 4.91 ms/块差距下，不建议仅靠 3072 容量；SAQ 的 TPOT 差现在只相当于约 **0.84 ms/块**（1023×0.24075/294）。这给窄桶提供了实际试验理由，不证明会胜。候选 `XLLM_DSV41_DECODE_3072_BUCKET` 默认关，只在 D full graph 明确打开：保留 512、在 513..3072 选择 3072，3073..4096 回退 4096；原有按实际末行可见性、奇偶、SWA 表宽、行数的 key 以及 snapshot/restore 不改。跨 TP 检查策略相同。主机已穷举 1..4096 的开/关容量选择；不以 CPU 检查冒充 GPU 图验证。

独立无 GPU 的 CPU 容器只做构建准备。归档构建 image 内的 ELF 与冻结运行 ELF 的 `.text` 不同，**不能冒充同一 ELF**。核对 `980d9da12` 的 xllm/CMake 等 2485 个跟踪文件，初始不一致 4 个，已在独立容器 overlay 内还原；复核 2485/2485 一致，E52 六个源码快照字节一致。没有采用旧 dirty 工作区、未在计分窗口编译，未把源文件清单冒充新二进制验证。
