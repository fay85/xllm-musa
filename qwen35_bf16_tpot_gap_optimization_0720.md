# Qwen3.5-35B-A3B BF16 TPOT gap optimization (2026-07-20)

## Outcome

The long-context C=1 TPOT gap is closed. With graph decode, FA3, prefix cache
off, ISL=20,000, OSL=300, four warmup requests and ten measured requests,
xLLM now measures **12.00 ms mean TPOT**. A matched rerun of the requested
`sglang_qwen35` source measures **12.05 ms** with the same benchmark client.

The final C=5 regression also passes. With ISL=2,500, OSL=1,500, a two-request
prefill admission cap, four warmup waves (20 requests) and ten measured waves
(50 requests), xLLM measures **18.90 ms mean TPOT**, versus the existing
matched SGLang result of **20.68 ms**.

## Bottleneck localization

The initial graph-mode context sweep separated fixed decode cost from the
full-attention context scan:

| ISL | xLLM baseline mean TPOT | xLLM after fix |
| ---: | ---: | ---: |
| 128 | 12.53 ms | 12.00 ms |
| 2,500 | 12.16 ms | 12.03 ms |
| 10,000 | 13.41 ms | 12.00 ms |
| 20,000 | 15.96 ms | 12.00 ms |

The baseline grew by 3.43 ms from ISL 128 to 20k, while the fixed result is
flat. This isolated the gap to long-K full-attention decode rather than the
GDN or MoE fixed cost.

The exact blocker was FA3 scheduler metadata in the MUSA decode graph:

1. xLLM captures decode graphs using a synthetic KV length of 16.
2. The GQA=8 metadata call was captured inside that graph with
   `num_splits=1`.
3. Later 20k-token replays updated `seqused_k`, but retained the capture-time
   scheduler policy and therefore ran the long-K attention with one split.
4. `sglang_qwen35` instead uses `num_splits=0`, regenerates scheduler metadata
   from the current KV lengths before every replay, and copies it into the
   tensor address retained by the graph.

This also explains the previous graph correctness failure that produced
repeated punctuation: the captured scheduler metadata was stale, not merely
slow.

## Implemented fix

- `attention_decode.cpp`: the GQA=8 Mate metadata call now uses
  `num_splits=0`, allowing Mate to select the split count from the live KV
  length.
- `cuda_graph_executor_impl.cpp/.h`: Qwen3.5 FA3 scheduler metadata is generated
  outside graph capture and replay. The capture-time tensor is retained by
  `CudaGraph`; each replay generates fresh metadata and copies it into that
  stable tensor before graph launch. The change is restricted to
  `qwen3_5_text` and `qwen3_5_moe_text` so unrelated model graph paths do not
  pay the extra update.
- `qwen3_next_hybrid_base.h`: graph execution consumes the externally prepared
  metadata, while eager execution keeps the existing first-attention-layer
  generation behavior.
- `qwen3_next_attention.cpp`: the MUSA path now respects
  `use_sliding_window`. This checkpoint sets it to false, matching SGLang's
  global full attention instead of incorrectly applying the parser's 4096
  fallback value.
- The C=5 helper now permits explicit memory-utilization and maximum-concurrent
  request overrides, so the tested graph buckets are exactly `[1,2,4,5]`.

## Validation

All builds ran inside `xllm-musa2.9.1-sdk5.1-dev` through
`_build_cuda_graph_musa.sh` and `mcc_wrapper`:

- baseline/revert build:
  `build_logs/build_cuda_graph_musa_20260720_055347.log`;
- dynamic scheduler implementation:
  `build_logs/build_cuda_graph_musa_20260720_060257.log`;
- global-attention correctness fix:
  `build_logs/build_cuda_graph_musa_20260720_061010.log`;
- final model-scope guard:
  `build_logs/build_cuda_graph_musa_20260720_062621.log`.

The post-final-build `correctness_check.sh` run passed all checks in graph
mode. The deterministic `17 * 23` response contains `391` and is byte-for-byte
the same content as the previously validated eager response. The earlier
repeated-`!` graph output is no longer reproducible.

### C=1 matched result

Both servers used GPU 2, BF16 weights, FA3, prefix cache off, graph decode,
ISL=20,000, OSL=300, four warmups and ten measurements. SGLang used the host
`sglang_qwen35/python` source through a temporary container `PYTHONPATH`, not
the older source checkout baked into `sglang-wf`. The table uses the same
`benchmark_serving_parallel.py` client for both rows.

| implementation | mean TTFT | mean TPOT | total TGS* | success |
| --- | ---: | ---: | ---: | ---: |
| xLLM final | **861.68 ms** | **12.00 ms** | 4,100.63 tok/s | 10/10 |
| `sglang_qwen35` mixed MoE | 925.21 ms | 12.05 ms | **4,485.14 tok/s** | 10/10 |

xLLM is 6.87% faster in mean TTFT and 0.42% faster in mean TPOT. Relative to
the pre-fix xLLM result of 15.93 ms, mean TPOT is reduced by 24.68%.

`*` C=1 total TGS includes input tokens and offered-load idle gaps. xLLM is
fast enough to expose gaps in the one-request-per-second arrival schedule,
so this number is not a saturation throughput comparison. The barrier-wave
C=5 total TGS below is the meaningful throughput gate.

Artifacts:

- xLLM baseline sweep:
  `/data/feihu/bench_results/qwen35_bf16_decode_context_sweep_xllm_20260720`;
- xLLM final sweep:
  `/data/feihu/bench_results/qwen35_bf16_decode_context_sweep_xllm_fa3_dynamic_global_20260720`;
- xLLM final C=1:
  `/data/feihu/bench_results/qwen35_bf16_fa3_dynamic_global_c1_4p10_20260720`;
- SGLang same-client C=1:
  `/data/feihu/bench_results/qwen35_bf16_sglang_qwen35_same_client_c1_10_20260720`.

### C=5 regression

| implementation | mean TTFT | mean TPOT | total TGS | success |
| --- | ---: | ---: | ---: | ---: |
| xLLM final | **413.25 ms** | **18.90 ms** | **683.86 tok/s** | 50/50 |
| SGLang cap=2 reference | 439.79 ms | 20.68 ms | 640.95 tok/s | 50/50 |

xLLM is 6.03% faster in TTFT, 8.61% faster in TPOT, and 6.69% higher in raw
total TGS. The xLLM server log reports zero fatal/MUSA/OOM errors.

Artifacts:

- xLLM:
  `/data/feihu/bench_results/qwen35_bf16_fa3_dynamic_global_c5_cap2_bs5_20p50_20260720`;
- SGLang:
  `/data/feihu/bench_results/qwen35_bf16_c5_cap2_barrier_clean_4w10m_20260720`.

## Rejected experiments

- A routed/shared-expert dual-stream experiment was not promotable because the
  pre-existing stale graph metadata already made graph output invalid. It was
  fully reverted before the scheduler fix.
- Direct and JIT-imported fused shared-expert gate kernels either produced
  invalid output or hung through the xLLM FFI integration. All source changes
  from those attempts were reverted; no such kernel is on the final path.

The accepted optimization follows the established SGLang graph-metadata
refresh design and does not add a new memory layout or port an unvalidated
kernel.
