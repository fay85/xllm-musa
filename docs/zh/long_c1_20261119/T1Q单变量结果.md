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