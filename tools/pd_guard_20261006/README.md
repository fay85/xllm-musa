<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# DSpark 1P1D Guard 发布快照

- [完整启动、构建与性能复现](../../docs/DSV4_PD_GUARD_1P1D_REPRO_20261006.md)
- [实现及优化说明](../../docs/DSV4_PD_GUARD_1P1D_IMPLEMENTATION_20261006.md)
- [GSM8K 协议、命令与结果](../../docs/DSV4_PD_GUARD_GSM8K_20261006.md)

`frozen/` 保留实际启动脚本与路由环境；`decode_overlay/` 保留实测 Decode
与 Prefill 的源码差异。默认 `xllm/` 是 Prefill 候选，不能未经验证把它的新
二进制同时替换两个角色。`build_reuse.py` 会校验已有链接输入后，在新目录
重编本次 PD 的关键翻译单元；它不是无缓存全量构建工具。

模型、AOT、依赖库、二进制、大体量请求/响应和 GPU 日志均保留在实验盘，
不提交 Git。SHA256 和位置见说明文档及 `docs/evidence/pd_guard_20261006/`。
