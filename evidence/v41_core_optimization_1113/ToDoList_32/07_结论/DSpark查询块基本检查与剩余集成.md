# T14：DSpark 查询块与上下文写回基本检查通过

仅通过基础功能检查，未完成模型精度验收。

新增原生 `forward_draft_block`，复用 MATE Sparse Attention 的显式索引接口。每个 draft query 读取同一组最近128个已接受上下文及整个 anchor/noise 查询块，块内非因果；proposal KV 不写入 SWA ring。独立 `write_draft_context` / `commit_draft_context` 只接收已接受目标状态形成的 KV，保留 padding 跳过和重复 slot 最新值覆盖语义。普通 target forward 行为没有打开新的运行开关。

实际 MATE 稀疏注意力对照独立 FP32 GPU 参考：7种历史长度（含0、页边界、回绕）、q1/q5、I32/I64页表、默认/非默认流，共56例通过；验证改变最后一个 proposal 可以影响块首 query，同时历史缓存逐位不变。另12例已接受上下文写回覆盖空输入、重复slot、padding、NaN和负零，逐位通过。编译与中转 SHA 已保存。

这是完整 DSpark 的一个必需环节，尚未完成三层 draft 权重加载、target37/38/39 entering residual捕获、FP8 main_proj及scale、Markov/confidence head、P/D bootstrap、spec verify、scheduler长度与缓存回滚/重放。此前96例 checkpoint也只是基础组件，不能据此声称MTP已完成或得到吞吐提升。后续按该顺序接入，吞吐只统计被接受并返回的实际输出token。
