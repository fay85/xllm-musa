# Qwen3.5 C1 adaptive piecewise prefill report (2026-08-01)

## Outcome

The fixed-request regression was not caused by a fake piecewise replay.
Piecewise replay is faster when the selected graph shape is close to the live
token count, but slower when a 2049--2073 token request is promoted to the
2112-token graph.

The validated fix is an adaptive C1 routing gate:

- Qwen3.5 single-sequence pure prefill uses piecewise replay when graph padding
  is at most 32 tokens.
- Larger graph-padding gaps use eager prefill.
- Packed B>1, chunked prefill, non-Qwen3.5 models, and decode are unchanged.
- Set
  `XLLM_QWEN35_C1_PIECEWISE_MAX_PADDING_TOKENS=-1` to restore forced
  piecewise routing for rollback or A/B.

With the fixed 2k C1 request sequence, the resulting piecewise configuration is
now faster than pure eager:

| Mode | Mean TTFT | Mean server forward |
|---|---:|---:|
| Adaptive piecewise (2 trials) | 267.437 ms | 234.327 ms |
| Forced piecewise, same binary (2 trials) | 268.792 ms | 235.048 ms |
| Eager baseline (2 trials) | 268.171 ms | 235.068 ms |

Adaptive versus forced piecewise:

- TTFT: -1.355 ms
- server forward: -0.721 ms

Adaptive piecewise versus eager:

- TTFT: -0.734 ms
- server forward: -0.741 ms

## Root-cause isolation

The authoritative paired baseline is:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/final_alias_fixed_warm20`

Every trial used the same serialized prompts, warmup 20, measure 20, C1,
ISL 2000, OSL 16. Stratifying the measured server forwards by live token count
shows the crossover:

| Token group | Piecewise minus eager, trial 1 | Trial 2 |
|---|---:|---:|
| `n_tokens <= 2048` (3 requests) | -2.856 ms | -2.262 ms |
| `n_tokens > 2048` (17 requests) | +1.041 ms | +0.603 ms |

The first group selects the 2048 graph and proves that the 17-segment/16-runner
replay provides a real launch-overhead benefit. The second group selects the
2112 graph. Its static-shape expansion consumes more time than replay saves.
The global forced-piecewise result therefore hid a beneficial replay behind an
unfavorable bucket mix.

The adaptive threshold of 32 is conservative: it falls back only beyond half
of the 64-token GDN/KKT granularity. All observed losing requests had gaps of
39--62 tokens; the winning near-2048 requests had gaps of 0--2 tokens.

## Replay attribution

The steady piecewise Kineto trace is:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/gdn_kineto_piecewise/torch_prefill_trace.json`

Observed structure:

- 17 graph segments and 16 eager FA3 runners.
- The 16 runners match the model's 16 full-attention layers; the 48 GDN layers
  remain captured in graph segments.
- Persistent-parameter update is approximately 0.4 ms in the non-intrusive
  replay profile.
- Runner replay contains 48 Q/K/V contiguous clones, but eager FA3 has the same
  logical copies; removing them from the host runner was not automatically a
  win (see rejected experiments).

This supports a bucket-cost root cause, rather than missing graph replay.

## Rejected experiments

### Capture-safe unpadded GDN with a 2080 graph

An experimental B=1 exception enabled the existing full-varlen KKT/GDN path
during graph capture and added a 2080 bucket.

Evidence:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/capture_unpadded_c1_2080_smoke_r2`

The log confirmed all 48 GDN layers used
`mate_gdn_prefill_full_varlen_strided_hq16_hv48_bf16` with
`q=[1,2080,16,128]` and `pad_size=0`. Capture and five measured requests
completed, but measured forward remained 240.568--242.276 ms. The varlen GDN
kernel cost outweighed the finer dense shape. The source change was reverted.

### Move FA3 Q/K/V contiguous copies into graph segments

An experimental capture path materialized Q/K/V before ending each graph
segment, so replay runners consumed graph-owned contiguous buffers.

Reverse A/B evidence:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/fa3_graph_owned_contiguous_ab`

| Variant | Mean TTFT | Mean forward |
|---|---:|---:|
| Graph-owned contiguous, 2 trials | 270.245 ms | 236.895 ms |
| Rollback, same binary, 2 trials | 268.658 ms | 235.243 ms |

The change regressed TTFT by 1.586 ms and forward by 1.652 ms. It was reverted.

## SGLang reference boundary

The reference source in `sglang-musa-reference` registers both
`unified_attention_with_output` and `unified_linear_attention_with_output`
as piecewise split operations. Its backend replays fixed token shapes selected
with `bisect_left`, so it is subject to the same static bucket economics.

A matched runnable SGLang piecewise number is not available for this Qwen3.5
checkout. Existing attempts under:

`/workspace/sglang_qwen35/benchmark_results/sglang_qwen35_27b_fp8_pcg_eager_c1_2k_16_20260801`

fail during TorchDynamo compilation on a graph break under an active context
manager and unsupported `ModelWeightParameter.__torch_function__` access.
SGLang eager remains usable, but its piecewise path is not a valid performance
oracle in this container.

## Final validation

Performance reverse A/B:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/adaptive_padding32_ab`

- Order: adaptive, forced, forced, adaptive.
- All four trials completed 20/20 measured requests after 20 warmups.
- Each adaptive measured trial routed 17 requests to eager and replayed three
  requests in the 2048 graph.
- Each forced trial replayed all 20 requests.

Deterministic correctness:

`/workspace/bench_results/gdn_core_piecewise_20260801_175500/adaptive_padding32_correctness`

- Adaptive: 5/5 complete, zero errors.
- Forced piecewise: 5/5 complete, zero errors.
- Temperature-0 generated texts and output lengths were exactly equal.

Build:

- Canonical build wrapper: `./_build_cuda_graph_musa.sh`
- Validation build log:
  `/workspace/xllm-git-master/build_logs/build_cuda_graph_musa_20260801_221537.log`
- Validation binary SHA256:
  `1174d797ceffaab0609ae09ba8ce2eaaf2f8f7f71309360e346bce0b07f871ee`
