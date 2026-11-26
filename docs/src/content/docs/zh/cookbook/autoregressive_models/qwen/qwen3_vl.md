---
title: "Qwen3-VL"
description: "Qwen3-VL Python 模型的文本侧上下文并行配置"
---

## 文本侧上下文并行（CP）

NPU 上的 Qwen3-VL Python 实现支持完整 prefill 的文本侧 CP。图像或视频特征先与文本 embedding 融合，然后按 CP 计划切分文本网络的 token 行。mRoPE 三轴位置和 DeepStack 特征使用同一份切分计划。文本网络输出在计算 logits 前恢复为全局 token 顺序。

Vision encoder 仍按现有路径执行。`cp_size` 控制文本网络的 prefill 并行度。Decode 在各 CP rank 上读取完整 KV cache。

启动时设置以下参数：

```bash
--model_impl=python \
--cp_size=2 \
--dp_size=1 \
--kv_split_size=1 \
--enable_chunked_prefill=false \
--enable_prefix_cache=false \
--enable_schedule_overlap=false \
--enable_graph=false \
--enable_prefill_piecewise_graph=false
```

设备总数必须能被 `cp_size` 整除；文本侧 TP 大小为设备总数除以 `cp_size`。例如，4 张卡、`cp_size=2` 使用 CP2 × TP2；2 张卡、`cp_size=2` 使用 CP2 × TP1。

当前支持 `generate` 任务、`DEFAULT` 角色和独立部署。暂不支持 speculative decoding、PD、prefix cache 或带历史 KV 的分块 prefill。关闭分块 prefill 后，`max_tokens_per_batch` 应覆盖请求的完整 prompt（含媒体 token）。不满足这些条件时，启动检查会报告对应错误。`cp_size=1` 保持原有模型行为。

`priority_strategy=multi_slo_and_prio` 会强制开启分块 prefill，因此该策略也不支持 CP。

启用 CP（`cp_size > 1`）的 prefill 必须提供 host query 和 KV 序列长度，缺少任一长度时会明确报错。此前非 MLA 的 eager 路径在缺少 host query 长度时会静默跳过 CP。
