---
title: "Graph Mode"
sidebar:
  order: 81
---
## 概述

xLLM 支持 Graph Mode，通过预捕获计算图并在后续执行中重放，减少 CPU 开销并提高推理性能。Graph Mode 在不同硬件平台上均有对应实现。

## 功能介绍

为了优化 Host 侧调度性能，图模式通过在 CPU 一次提交大任务后，设备内部流式执行小 kernel，显著降低启动时间和设备气泡。

在 xLLM 引擎中，Graph Mode 实现了以下特性：

### 动态维度参数化
  - 将除 num_tokens 以外的关键动态维度作为整图输入参数，包括 batch_size、kv_seq_lens、q_seq_lens、block_table_size 等，从而提高灵活性。在进行图的内存分配和内核配置时，利用这些动态参数计算实际所需值。在图启动阶段，将上述实际参数传入，以确保 kernel 能够使用正确的 stride 访问数据。

### Piecewise Graph
  - 当部分算子不支持 graph 导致整图无法捕获（break graph）时，对 break 之后的各段（piece）分别捕获 graph。这样在无法整图捕获的情况下，仍能尽可能获得 graph mode 的收益，常用于 prefill、chunked_prefill 等场景。

### 多 shape 复用的显存池
  - 为了避免不同 shape 的 graph capture 分别占用独立显存，我们让不同 capture 使用不同虚拟地址空间，并共享同一组底层物理内存；同时，输入 tensor 通过持久化 buffer 与 slice 方式复用。
 
## 使用方式

上述功能已在 xLLM 引擎内部实现，通常通过 gflags 参数控制。

最小配置只需要开启 `enable_graph`，用于打开 decode 阶段的 Graph Mode：

```shell
--enable_graph=true
```

常见的配套开关包括：

- `enable_graph`：开启 decode 阶段的 Graph Mode 基础能力
- `enable_prefill_piecewise_graph`：开启 prefill 阶段的 Piecewise Graph
- `enable_graph_mode_decode_no_padding`：decode 阶段按实际 `num_tokens` 建图，而不是按 padding 后的 shape 建图
- `max_tokens_for_graph_mode`：限制 Graph Mode 覆盖的最大 token 数；`0` 表示不限制

如果希望同时开启 decode Graph 和 prefill Piecewise Graph，示例如下：

```shell
--enable_graph=true \
--enable_prefill_piecewise_graph=true \
--max_tokens_for_graph_mode=2048
```

如果需要在 decode 阶段启用无 padding 建图，可额外开启：

```shell
--enable_graph=true \
--enable_graph_mode_decode_no_padding=true
```

更完整的参数说明可参考 [CLI 参数说明](/zh/cli_reference/)。

### 指定 NPU decode capture buckets

Native ACL executor 与 Python ACL runner 支持 `graph_capture_sizes`，用来替换默认 decode token 桶。它不增加模型的图支持，也不改变 CUDA/MLU 或 prefill 的选桶策略。

```shell
--enable_graph=true --graph_capture_sizes=3,7,10
```

JSON 配置使用整数数组：

```json
{"enable_graph": true, "graph_capture_sizes": [3, 7, 10]}
```

- 单位是 **DP-local token rows**，不是全局 sequence 数。普通 decode 请求选择不小于组内最大 token rows 的最小有效桶：例如 4 rows 选择 7，11 rows 在执行前选择 eager，不捕获默认的 16-row 图。
- 列表必须包含正的 int32 整数，自动排序去重；CLI 优先于 JSON。未设置参数或显式 `--graph_capture_sizes=` 保留旧策略；显式 JSON 空数组、非法值、容量过滤后没有可用 shape 均报配置错误。
- Uniform speculative decode 按对应 runner 的 query width 向上对齐，去重后排除超过原始最大 size、`max_tokens_per_batch` 或 sequence/ACL 容量的 shape。例如 `[1,4,8,16]`、width=3 得到 `[3,6,9]`，不捕获 18。Target 与 draft 按各自宽度解析，不直接共用原始数组。
- 首版不支持与 decode no-padding、adaptive speculative decode 或 native packed ATB speculative kernel 组合。其他现有模型能力与通信限制仍然生效。
- 预热按有效 token 桶构造全局 sequence batch，并保持从大桶到小桶捕获；受 KV 容量限制无法预热的桶会输出警告。设置 `disable_graph_warmup=true` 时仍只允许在有效列表内延迟捕获。图实例数还取决于 slot、模型 variant 和 attention plan，不等于桶数。
- 启动日志输出 `effective capture token rows` 与 query width；超过列表时有执行前分流日志，并计入现有 eager execution counter。捕获/回放失败仍正常传播，不以失败后切 eager 掩盖错误。此参数不是 `max_tokens_for_graph_mode` 的别名。

## 性能效果

- 开启 Graph Mode 后，在 Qwen3-0.6B 和 Qwen3-1.7B 等模型上，decode 阶段吞吐 **提升约 8%–10%**。

## 模型支持

下表列出目前各模型在 ACLGraph、CudaGraph、MLUGraph 上的支持情况。

| 模型 | ACLGraph | CudaGraph | MLUGraph |
|------|----------|-----------|----------|
| Qwen3/Qwen3-MoE | ✅ | ✅ | ✅ |
| DeepseekV3.2 | ✅ | | ✅ |
| GLM4.5/4.6/4.7 | ✅ | | |
| Qwen2.5-VL | | | ✅ |
| Qwen3-VL/Qwen3-VL-MoE | ✅ | | |
| GLM4V | ✅ | | |
| GLM4V-MoE | ✅ | | |

## 相关文档
- 更详细的 Graph Mode 设计与实现说明（含 ACL Graph / CUDA Graph 基本原理、动态维度参数化、Piecewise Graph 与多 shape 复用内存方案）见：[Graph Mode 设计文档](/zh/design/graph_mode_design/)

