# T14：DSpark 非因果块与上下文写回

本项以累计最优4411为母本，暂不启用新服务策略。先补三层 drafter 必需的 GPU 计算语义，再接 target capture、权重、验证与回滚，不把96例状态快照检查等同于完成 MTP。

本子步骤明确区分：已接受 target hidden 投影形成的历史 KV 可以写入 draft SWA；anchor/noise proposal 的 KV 只存在本次局部张量中，不写 ring。draft 的所有查询读取同一份前128个已接受上下文加整个 query block，块内非因果。复用 MATE Sparse Attention 的显式索引接口，不直接复用 target 的逐查询因果 SWA 窗口。

参考共同仓库 Python CSA2 `_block_visible_idxs` 和 V41 DSpark `write_context_kv`，以及已有 V4 原生 DSpark 构建接口。当前仍需完成 draft 权重和 target/serving 的集成，尚无性能收益结论。

仅通过基础功能检查，未完成模型精度验收。
