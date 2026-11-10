---
title: "DeepSeek-V4.1"
description: "Ascend checkpoint adaptation and preparation for DeepSeek-V4.1-Flash-1017 W8A8"
---

<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# DeepSeek-V4.1-Flash-1017 W8A8

This page describes the Eco-Tech DeepSeek-V4.1-Flash-1017 W8A8 checkpoint format and preparation steps for xLLM's Ascend Python executor.

DeepSeek-V4 and V4.1 differ in model structure, Engram memory requirements, and draft models. Do not reuse the device count, parallel configuration, or MTP export procedure from the [DeepSeek-V4 deployment example](/en/cookbook/autoregressive_models/deepseek/deepseek_v4/). Section 6 gives a validated single-host generation smoke configuration, not a production topology or performance result; resource budgets in other environments need independent validation.

## 1. Model and environment

Checkpoint:
https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8

Use an xLLM source checkout or package containing the 1017 adaptation, and build, prepare, and launch in the same Ascend environment. The NPU Python model executor uses `--model_impl=python` and `--npu_kernel_backend=TORCH`.

This page covers the W8A8 export, not deployment of the official 1017 FP8 checkpoint.

After building and installing the current xLLM custom operators, load each vendor's `bin/set_env.bash` in the launch environment so `ASCEND_CUSTOM_OPP_PATH` includes the installed operator directories. The CANN 9.1 validation used `custom_xllm_math` and `custom_transformer`; installing their files does not itself load their environment. Without this setup, a real forward may fail with `Cannot find kernel by name SmlaVaMetadata`. Do not bypass this by changing dtypes or skipping Attention.

## 2. Mixed-precision weights

| Module | 1017 export format | xLLM loading path |
|---|---|---|
| Attention `wq_a` / `wkv` | BF16 | Unquantized Linear |
| Attention `wq_b` and quantized FFN | W8A8 | Dynamic W8A8 Linear / MoE |
| Engram embedding tables | INT8 + UE8M0 group-32 scales | Retain INT8 tables and decode scales to FP32 in bounded loading chunks |

xLLM reads `quant_model_description.json` to select each target and DSpark layer's projection precision. Retain this file; do not bypass checks by deleting quantization metadata, fabricating scales, or directly casting BF16 weights to INT8.

UE8M0 scales use the safetensors dtype `F8_E8M0`. Loading decodes only the owning TP rank's rows, retaining FP32 runtime scales. Numerically casting the raw bytes to float is not UE8M0 decoding. Earlier all-W8A8 Attention and FP32 Engram scale exports remain supported.

## 3. Prepare Engram hash state

Engram inference requires tokenizer-derived `engram_hash_state.bin`. If the checkpoint does not include it, generate it using that checkpoint's tokenizer. This step requires `tokenizers` and `numpy`.

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-w8a8
python - "$MODEL_PATH" <<'PY'
import sys
from xllm.python.models.deepseek_v41_engram import build_engram_hash_state

print(build_engram_hash_state(sys.argv[1]))
PY
```

The generator preserves special tokens and strips all Unicode Mark categories, matching the reference normalization. Do not change `engram_compressed_vocab_size` to bypass validation: doing so rehashes the table to incorrect rows. The 1017 compressed vocabulary size is `99092`.

For read-only checkpoints, create a writable model view: link the original safetensors files and `optional/`, retain configuration, tokenizer, quantization-description, and weight-index files, and write the sidecar into the view. Point `--model` at that view; large weight files need not be copied or rewritten.

## 4. Engram memory and offload

NPU Engram CPU offload and DP sharing are both disabled by default. Runtime scales remain FP32, so UE8M0 checkpoint storage does not automatically reduce the runtime table memory budget. Evaluate TP, DP, EP, HBM, and host memory independently rather than using V4's resource configuration.

When host memory is sufficient, explicitly select CPU offload without DP sharing through `--engram_config`:

```json
{"cpu_offload":true,"dp_shared_memory":false}
```

DP sharing additionally needs Gloo communication and sufficient shared memory; validate large-table loading time and timeout conditions. See [Engram host offload](/en/features/engram_host_offload/) for configuration and constraints. The configuration above is not a complete launch command.

## 5. DSpark draft model

Both target and DSpark Attention in 1017 require checkpoint-driven mixed precision. With DSpark, point target and draft at the same prepared checkpoint or model view; do not apply `export_mtp.py` from V4's legacy MTP path.

Validate DSpark startup and generation separately. Target weight loading or individual kernel success does not establish end-to-end inference success. See the [CLI reference](/en/cli_reference/) for the relevant options.

## 6. Validated minimal generation configuration

On 2026-11-09, ordinary generation and DSpark end-to-end generation smoke tests passed on one Ascend A3 host (16 logical NPUs, 64 GiB HBM each; approximately 3 TiB host RAM), with CANN 9.1 and Torch / torch_npu 2.10. Real `/v1/chat/completions` requests returned `Paris`, `2`, and `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` in both modes, with 2, 2, and 29 completion tokens and normal termination. Nonzero DSpark draft and accepted-token metrics confirmed actual drafting and verification.

The tests used a block-wise byte-verified local checkpoint cache and a prepared Engram sidecar, without modifying the original checkpoint. Large-table mmap reads on network storage can greatly increase startup time; evaluate I/O on your storage or use local caching with sufficient capacity.

This single-host smoke configuration resolves to TP8 / Attention DP2 / MoE EP16. `MODEL_PATH` may also point to a prepared writable model view. Load the Ascend and custom-vendor environments before launching:

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-w8a8
XLLM_BIN=/path/to/xllm
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m xllm.launch_server serve \
  --binary-path "$XLLM_BIN" --nnodes=16 --port=29381 \
  --model="$MODEL_PATH" --model_id=deepseek-v41-1017 \
  --host=127.0.0.1 --master_node_addr=127.0.0.1:29481 \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --dp_size=2 --ep_size=16 --max_memory_utilization=0.90 \
  --max_cache_size=1073741824 --block_size=128 \
  --max_tokens_per_batch=256 --max_seqs_per_batch=1 \
  --enable_prefix_cache=false --enable_graph=false \
  --enable_schedule_overlap=false --enable_fused_mc2=0 \
  --engram_config='{"cpu_offload":true,"dp_shared_memory":false}'
```

For DSpark, append these arguments to the launch command above:

```bash
--draft_model="$MODEL_PATH" --speculative_algorithm=DSpark --num_speculative_tokens=5
```

Requests used `temperature=0` and `reasoning_effort="none"`. DSpark used compatibility SAS, not native SAS. Graph mode, scheduling overlap, and DP-shared tables were disabled. Those modes, concurrency, long contexts, accuracy evaluations, multimodal inference, and performance claims are outside this validation.
