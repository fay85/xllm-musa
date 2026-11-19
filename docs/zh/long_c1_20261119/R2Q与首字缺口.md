# R2Q：快筛 TPOT 和 E2E 首次同时过线，仍缺 TTFT

R2Q 相对 G6Q 只将 Router HTTP workers 从 4 改 2。两端实际 /flags 已核对：Router HTTP/RPC=2/4，Coordinator=4/4，两个 control tick 仍分别为 20/100 ms。verbose 和 request trace 都关，没有额外暖机、设备采样或 profiler。暖机与两条计分逐条对上实际 2000/1024、length、零 prefix、正文 SHA `16b1d823…`、732/1470。两道基本题为 4.2/56，不是精度验收。

|计分|TTFT ms|TPOT ms/token|E2E s|
|---|---:|---:|---:|
|1|1187.724842|15.505804|17.050163|
|2|954.137079|15.427027|16.735986|
|均值|1070.930961|15.466416|16.893074|
|相对 S2Ret|**+5.913956**|**−0.086142**|**−0.082209**|

不是正式通过，不能舍入、去掉第 1 条或再多暖几次。两条的 UUIDv7 到反推引擎起点为 201.35 / 15.72 ms；2 个 HTTP worker 没有消灭冷路径。TPOT 改善不是 HTTP worker 优化 D 内核的因果证据。H1D 五条独立诊断、真实 ELF/词表读量证据见 [入口阶段实测](入口阶段实测.md)。

下一候选 P4Q 从 E2E 更好的 R2Q 出发。**只在 P 缓存同一 gate 权重的 FP32 表示**。实读 safetensors 索引、各分片头：40 个 `layers.N.ffn.gate.weight` 均 BF16 [384,5120]；native loader 复制 BF16，512/464 行走 FP32 linear，而非小行 SIMT。OFF 每次 `weight_.to(float)`；ON 在模型加载时做同一转换，不改 input 转换、矩阵乘、top-k、权重文件、D 或 Router。4 个 P chunk、40 层可免至多 160 次重复权重转换；40 份 FP32 共 300 MiB/rank，实测显存与完整正文仍必查，不承诺转换一定在关键路径。目标是检验能否稳定收回首字的 5.914 ms，不以理论带宽当实际收益。

**P4Q 快筛结果：1041.496342 ms / 15.577327 ms/token / 16.977102 s。** 相对 S2Ret，首字快 23.520663 ms，但 TPOT 慢 0.024769 ms/token，E2E 慢 **0.001818491425364499 s**，严格不通过。相比 R2Q，缓存 Gate 权重这一组的首字低 29.435 ms，TPOT 和 E2E 却更高，不拼组。实测 P 基本题窗口每卡最大读数从 71407 变为 71754 MiB；不是整轮峰值。三条 2000/1024 正文与 732/1470 不变；两道基本题通过。

下一组 L0Q 选 E2E 更好的 R2Q 作父配置，不叠加 P4Q。依据是每个 P rank 在此前 5 次 2k 输入（暖机、快筛与两道基本题）中记录了 1600 条 layer begin/enqueued 进度日志，即每请求八卡共 2560 条。只在 P 取消该可选逐层打印，保留 PD/终态计数、prefix 审计和客户端计时。变量必须**取消设置**，设成字符串 0 仍会触发源码的 getenv presence 判断；进程现场核对。没有层日志就不再声称有该组的 chunk span 分解。

当前路由服务加载 `/opt/xllm_service/lib/librust_tokenizers.so`，SHA256 b26878b681bfd6280e031509fd01a39c64a6adf623fa304a771bdc64bc03d456、build-id a9a02a19bda9524a608cec5b39ea18d821ee35eb；包含 `tokenizers_outstanding_encode_results` 等 ABI 符号。不要因本仓库也有一个 Rust Tokenizer 包就推断可直接替换生产 DSO。
