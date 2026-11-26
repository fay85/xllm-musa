---
title: "Qwen3-VL"
description: "Text context parallelism for the Qwen3-VL Python model"
---

## Text context parallelism (CP)

The Qwen3-VL Python implementation on NPU supports text CP for full prefill. Image or video features are fused with text embeddings before token rows are sharded across the text network. Three-axis mRoPE positions and DeepStack features follow the same CP plan. Text outputs are restored to global token order before computing logits.

The vision encoder uses its existing execution path. `cp_size` controls prefill parallelism in the text network. Decode reads a complete KV cache on each CP rank.

Set these launch options:

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

The total device count must be divisible by `cp_size`; text TP size is the device count divided by `cp_size`. Four devices with `cp_size=2` use CP2 × TP2; two devices with `cp_size=2` use CP2 × TP1.

This path currently supports standalone `generate` serving with the `DEFAULT` role. Speculative decoding, PD, prefix caching, and chunked prefill with historical KV are unsupported. With chunked prefill disabled, set `max_tokens_per_batch` to cover the complete prompt, including media tokens. Startup validation rejects unsupported configurations. `cp_size=1` preserves existing model behavior.

`priority_strategy=multi_slo_and_prio` forces chunked prefill and is also unsupported with CP.

CP prefill fails explicitly if host query or KV sequence lengths are missing. The non-MLA eager path previously silently skipped CP when host query lengths were absent. This requirement applies only when CP is enabled (`cp_size > 1`).
