# T1Q：首字均值过线，但 TPOT 和 E2E 未过

与 SNQ 比只改 Router `scalable_control_tick_ms=20`。实际16 rank、ELF、与 prepared_config 的 argv/env 均核对；SJ10 的 /flags 证明 router=20、coordinator=100，credit TTL 仍 3000、event pull 仍 250、flow dispatch 仍 10、request 和 verbose trace 都关。独立 1 暖 2 测与两道基本题成功，两个角色正常释放。全部三条真实 2000/1024、length、零前缀、正文 `16b1d823…`、732/1470。

|样本|TTFT ms|TPOT ms/token|E2E s|
|---|---:|---:|---:|
|1|1114.372|15.605284|17.078578|
|2|946.196|15.619368|16.924810|
|均值|1030.284|15.612326|17.001694|
|相对 S2Ret|−34.733|+0.059769|+0.026411|

第二条 E2E 单独小于基线不改变判断。按 SJ10 UUIDv7 和终态反推，T1Q 两条前部间隔为 205.33 / 18.56 ms，**未改 router 周期的 C0Q 也有 201.51 / 18.78 ms**。故只认为 T1Q 当前整体快筛最接近，不能称缩短 tick 消除了 190 ms，也不能称 tick 优化了 decode。还需在每个 294 块请求的生成阶段平均约 **0.208 ms/块** 才补齐 TPOT。

下一组 G6Q 在同 T1 router、同模型 ELF 上只开 D 的位保持 BF16 汇集。依据是 q6 target 图 AllGather 中位约 0.692 ms/图且图外 head 另有汇集，当前缺口小于 1 ms/块；E51 的 q4 短请求并未获可靠收益，故不直接晋级。核对所有输出与计数相同后以完整 2k metrics 决定，并在全部计分结束之后另做 router 只读 CPU/IO 诊断。

G6Q 已完成并释放。逐请求身份和 732/1470 轨迹均不变，计分均值 **1164.007 ms / 15.568316 ms/token / 17.090394 s**。相对 T1Q，TPOT 又低 0.044011 ms/token，但首字多 133.723 ms，E2E 反慢 88.700 ms；距 S2Ret 仍差 +98.990 ms、+0.015758 ms/token、+0.115110 s。不能取一条的 15.460 TPOT 或把它和 T1Q 最快的首字合起来放行。

G6Q 的两条计分 UUIDv7 至反推引擎起点 231.36 / 219.76 ms。随后三条另发 2000/32 诊断请求均匹配独立 length、零缓存 D 终态，其前部**只有 15.25 / 5.49 / 25.98 ms**。只在这三条前 350 ms 观察任务 router CPU、读量和 FD：进程 CPU 增量 0 / 10 / 0 ms（时钟量化，非函数计时），rchar 6.3–6.9 kB，read_bytes 全零，未采到 tokenizer 文件。说明这三条**快路径请求**没有持续重读大词表，不解释较早的慢请求，也不是统一 32 输出就能避免该阶段。后续须在全新 router 的最初几个 2000/1024 上配对 lifecycle 和 IO，而不是以服务已热的短诊断排除冷初始化。

当前 E2E 最接近的父配置为 T1Q。下个独立 M0Q 保持该配置，只把 D 的 `XLLM_MUSA_SPEC_TIMING_PREFIX` 置空，直接走计时对象 early return。两组都不 arm、不记录事件；区别是控制每个 proposal step 仍创建 DeviceGuard、调用 flush 和从 NFS 查不存在的 arm 文件。单主机只读探测该缺失路径 96 次中位 16.744 μs、p95 30.734 μs，不等于能收回的关键路径时间。