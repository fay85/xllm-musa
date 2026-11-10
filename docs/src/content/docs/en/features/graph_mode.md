---
title: "Graph Mode"
sidebar:
  order: 81
---
## Overview

xLLM supports Graph Mode: computation graphs are pre-captured and replayed in subsequent runs to reduce CPU overhead and improve inference performance. Graph Mode has corresponding implementations on different hardware platforms.

## Feature Description

To optimize Host-side scheduling, graph mode submits a large task from the CPU once and then executes small kernels in a streaming manner on the device, significantly reducing startup time and device bubbles.

In the xLLM engine, Graph Mode provides the following:

### Dynamic Shape Parameterization
  - Key dynamic dimensions other than num_tokens are treated as whole-graph input parameters, including batch_size, kv_seq_lens, q_seq_lens, block_table_size, and the like, for flexibility. During memory allocation and kernel configuration. At graph launch, the actual values of these parameters are passed so kernels use the correct strides to access data.

### Piecewise Graph
  - When some operators do not support graph capture and thus break the full graph, each segment (piece) after the break is captured as a separate graph. This maximizes graph-mode benefits even when the full graph cannot be captured, and is commonly used for prefill and chunked prefill.

### Multi-Shape Reusable Memory Pool
  - To avoid waste from separate memory buffers (input, output, and intermediate tensors) per shape, we use an expandable memory pool. Multiple shapes share the pool base address, with different shapes using different offsets from that base.

## Usage

These capabilities are implemented inside the xLLM engine and are generally controlled through gflags.

The minimal configuration only needs `enable_graph` to turn on Graph Mode for the decode phase:

```shell
--enable_graph=true
```

Common companion flags include:

- `enable_graph`: enables the base Graph Mode capability for the decode phase
- `enable_prefill_piecewise_graph`: enables Piecewise Graph for the prefill phase
- `enable_graph_mode_decode_no_padding`: builds decode graphs with the actual `num_tokens` instead of the padded shape
- `max_tokens_for_graph_mode`: limits the maximum number of tokens covered by Graph Mode; `0` means no limit

If you want to enable both decode Graph and prefill Piecewise Graph, use:

```shell
--enable_graph=true \
--enable_prefill_piecewise_graph=true \
--max_tokens_for_graph_mode=2048
```

If you need decode graph capture without padding, add:

```shell
--enable_graph=true \
--enable_graph_mode_decode_no_padding=true
```

For a more complete description of the flags, see [CLI Reference](/en/cli_reference/).

### Custom NPU decode capture buckets

The native ACL executor and Python ACL runner accept `graph_capture_sizes` to replace the default decode token buckets. This does not add graph support for new models or change CUDA/MLU or prefill bucket policies.

```shell
--enable_graph=true --graph_capture_sizes=3,7,10
```

JSON configuration uses an integer array:

```json
{"enable_graph": true, "graph_capture_sizes": [3, 7, 10]}
```

- Sizes are **DP-local token rows**, not global sequence counts. Plain decode selects the smallest effective bucket covering the shared maximum row count: 4 rows select 7; 11 rows select eager before execution, without capturing a default 16-row graph.
- Values must be positive int32 integers and are sorted/deduplicated. CLI overrides JSON. Omission or explicit `--graph_capture_sizes=` retains the old policy. An explicit empty JSON array, invalid values, or no usable shape after capacity filtering is a configuration error.
- Uniform speculative decode rounds up to the runner's query width, deduplicates, and excludes shapes exceeding the raw maximum size, `max_tokens_per_batch`, or sequence/ACL capacity. `[1,4,8,16]` at width 3 becomes `[3,6,9]`, not 18. Target and draft resolve their own geometry.
- The first implementation rejects decode no-padding, adaptive speculative decode, and native packed ATB speculative kernels with this option. Existing model and communication capability gates still apply.
- Warmup maps effective token buckets to global sequence batches and captures largest first. KV-capacity-limited omissions produce a warning. `disable_graph_warmup=true` retains lazy capture only within the effective list. Slot, model variant, and attention-plan keys can create multiple graphs per bucket.
- Startup logs show `effective capture token rows` and query width. Over-limit requests are routed before execution, logged, and counted by the existing eager execution counter. Capture/replay failures still propagate; they do not trigger failure-to-eager recovery. This option is not an alias for `max_tokens_for_graph_mode`.

## Performance Impact

- With Graph Mode enabled, decode-phase throughput **improves by about 8%–10%** on models such as Qwen3-0.6B and Qwen3-1.7B.

## Model Support

The following table lists each model’s support on ACLGraph, CudaGraph, and MLUGraph.

| Model | ACLGraph | CudaGraph | MLUGraph |
|------|----------|-----------|----------|
| Qwen3/Qwen3-MoE | ✅ | ✅ | ✅ |
| DeepseekV3.2 | ✅ | | ✅ |
| GLM4.5/4.6/4.7 | ✅ | | |
| Qwen2.5-VL | | | ✅ |
| Qwen3-VL/Qwen3-VL-MoE | ✅ | | |
| GLM4V | ✅ | | |
| GLM4V-MoE | ✅ | | |

:::caution[Adding Graph Mode support for new models]
Ensure that the kernels used in the computation implement dynamic dimension parameterization; otherwise the graph may break and kernels may need to be re-implemented.

:::
## Related Documentation

- For more detailed Graph Mode design and implementation notes, including ACL Graph / CUDA Graph fundamentals, dynamic dimension parameterization, Piecewise Graph, and multi-shape memory reuse, see: [Graph Mode Design Document](/en/design/graph_mode_design/)

