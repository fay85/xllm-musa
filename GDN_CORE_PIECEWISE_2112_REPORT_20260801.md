# GDN Core and Piecewise Prefill Report (2026-08-01)

## Authoritative scope

This report supersedes the earlier provisional conclusion in this file. It uses fixed serialized prompts for all final sub-millisecond comparisons.

- Host checkout: /data/feihu/xllm-git-master
- Container: xllm-musa2.9.1-sdk5.1-dev
- Model: /workspace/model_weights/Qwen3.5-27B-FP8
- Device: physical MUSA GPU 3
- Workload: C=1, nominal ISL=2000, OSL=16
- Runtime: graph on, FA3 on, packed prefill off, MATE GDN backend
- Evidence root: /workspace/bench_results/gdn_core_piecewise_20260801_175500

## Executive result

Two changes are retained:

1. add a 2112-token piecewise shoulder so a prompt just above 2048 no longer jumps to 2304;
2. remove a redundant per-GDN-layer KKT cu-seqlens clone in eager unpadded C1 varlen prefill.

The GDN change improves fixed-prompt eager TTFT by 0.507 ms and server prefill forward by 0.361 ms. It removes 48 clone/copy/fill/select sequences and reduces the Kineto CPU scope for all 48 MATE wrappers from 8.151 ms to 5.442 ms.

Piecewise replay is real and the original 10.499 ms regression is almost eliminated. However, on the final fixed-prompt protocol it is still 0.858 ms slower in TTFT and 0.315 ms slower in server forward than eager. Therefore the answer is: piecewise should eventually be faster in principle, but the current xLLM implementation is not yet measurably faster for this 2k C1 workload.

## 1. Original piecewise bottleneck

A same-binary reverse-ordered baseline selected bucket 2304 for actual prefill lengths near 2040-2089.

| Mode | Mean TTFT (ms) | Mean server prefill forward (ms) |
|---|---:|---:|
| eager, 2 trials | 268.472 | 235.518 |
| piecewise 2304, 2 trials | 278.971 | 245.620 |
| piecewise - eager | +10.499 | about +10.1 |

Piecewise profiling showed 17 graph segments and 16 attention runners. Graph compute consumed about 236-243 ms, runner compute about 11 ms, and metadata update about 0.4 ms. The path was a real replay path; padded graph work was the dominant loss.

Evidence: eager_r1, piecewise_r1, piecewise_r2, eager_r2, and piecewise_profile_2304.

## 2. Retained 2112 shoulder

generate_piecewise_prefill_graph_tokens() includes a 2112-token shoulder aligned to the 64-token MATE GDN/KKT granularity.

The initial warmup-4 paired result changed the piecewise gap from +10.499 ms to +1.647 ms and recovered 8.852 ms.

| Mode | Mean TTFT (ms) | Mean server prefill forward (ms) |
|---|---:|---:|
| eager | 270.433 | 236.969 |
| piecewise 2112 | 272.080 | 238.409 |
| piecewise - eager | +1.647 | +1.440 |

Evidence: after_2112/{piecewise_r1,eager_r1,eager_r2,piecewise_r2}.

Warmup=4 is not sufficient for final piecewise claims because generated prompts can land on both sides of 2048. One retained diagnostic first captured 2112 during warmup and then captured 2048 on measured request 3, producing a 540 ms TTFT outlier. This is a cold-capture artifact, not steady replay.

## 3. Fixed-prompt benchmark correction

The original benchmark seeded Python random, but tokenizer.get_vocab() iteration differed across client processes. Consequently nominally identical trials had different actual token sequences. PYTHONHASHSEED did not stabilize the Rust tokenizer vocabulary order.

A fixed-request benchmark was therefore created under gdn_kkt_cu_alias_ab_fixed_requests. It serializes the exact prompt strings once and reuses them across every server process. Final eager and piecewise trials have the same 20-token-length sequence and mean actual prefill length 2058.5.

The final protocol uses warmup=20 and measure=20. Both 2048 and 2112 buckets are captured before measurement. Trial order is eager, piecewise, piecewise, eager.

| Mode | Trial 1 TTFT | Trial 2 TTFT | Mean TTFT | Mean forward |
|---|---:|---:|---:|---:|
| eager | 268.277 | 268.065 | 268.171 | 235.068 |
| piecewise | 269.020 | 269.038 | 269.029 | 235.383 |
| piecewise - eager |  |  | **+0.858** | **+0.315** |

All 80 measured requests completed successfully. These fixed-prompt results supersede the earlier non-fixed order-balanced point estimate that suggested piecewise was 0.192 ms faster.

Evidence: final_alias_fixed_warm20.

## 4. Retained GDN KKT cu-seqlens alias

Kineto isolated a concrete host-side GDN cost in eager unpadded C1 varlen prefill:

- 48 xllm/mate_gdn_prefill CPU scopes totaled 8.151 ms.
- Each layer cloned live cu_seqlens, copied it, selected its endpoint, and filled the endpoint.
- For no-pack/no-pad C1, live cu_seqlens already ends at num_tokens and KKT only reads it.

The optimized path aliases kkt_cu_seqlens to cu_seqlens only when:

- no padded rectangular batch is being packed;
- pad_size is zero; and
- no dedicated device KKT cu-seqlens was supplied.

All other layouts retain the old clone-and-fill behavior. Runtime rollback is XLLM_MATE_GDN_DISABLE_KKT_CU_ALIAS=1.

### Fixed-prompt A/B

Four legacy and four alias trials used the same serialized 20-request prompt list in both orders.

| Mode | Mean TTFT (ms) | Mean server forward (ms) |
|---|---:|---:|
| legacy clone | 269.480 | 236.234 |
| KKT cu alias | 268.973 | 235.873 |
| alias - legacy | **-0.507** | **-0.361** |

All 160 measured requests completed successfully.

### Kineto confirmation

| Metric across 48 GDN layers | Legacy | Alias |
|---|---:|---:|
| xllm/mate_gdn_prefill CPU scope | 8.151 ms | 5.442 ms |
| aten::clone | 48 | 0 |
| aten::copy_ inside wrapper | 48 | 0 |
| aten::fill_ for KKT endpoint | 48 | 0 |
| aten::select for KKT endpoint | 48 | 0 |

The wrapper CPU scope fell by 2.709 ms, while end-to-end forward improved by 0.361 ms because much of the host work had overlapped queued GPU work.

Temperature-zero validation used five fixed measured requests per mode. Legacy and alias generated texts were exactly equal for all 5/5 requests, with no errors.

Evidence:

- gdn_kkt_cu_alias_ab_fixed_requests
- gdn_kkt_cu_alias_correctness
- gdn_kineto_eager
- gdn_kineto_eager_alias

## 5. GDN core status versus SGLang

The current kernel-level evidence does not show a large slow GDN primitive:

- xLLM GDN projections are about 31 ms, similar to SGLang's 31-34 ms.
- xLLM token-major causal conv is about 11.4 ms. Disabling it raises conv to about 38.9 ms.
- The eager Kineto trace reports 48 MATE main kernels totaling 7.191 ms and 48 KKT kernels totaling 4.286 ms.
- Nested xLLM GDN core components are close to the SGLang MUSA GDN core total.

The remaining gap is distributed host launch and wrapper work across layers, not one defective MATE math kernel.

## 6. Falsified and reverted candidates

1. 2080 bucket with relaxed alignment: correct output, but regular GDN internally padded/copied and replay slowed to about 242 ms forward.
2. C1 packed-varlen piecewise GDN: 5/5 success, but graph/runner segmentation expanded from 17/16 to 65/64 and TTFT regressed to about 294 ms.
3. MATE module-availability cache: cache-on TTFT 269.252 ms versus cache-off 268.971 ms; no benefit.
4. FFI TensorView conversion: view TTFT 268.993 ms versus legacy 268.768 ms; no benefit.
5. 2064 shoulder: invalid candidate because 2064 is not divisible by 64. Bucket selection aligns Qwen3.5 prefill tokens upward to 64 before lookup, so it was never selected. The source experiment was reverted.

## 7. Piecewise next bottleneck

A steady piecewise Kineto trace confirms:

- 17 graph segments and 16 attention runners;
- non-intrusive metadata update around 0.4 ms;
- repeated slice/contiguous/temporary-buffer preparation between runner and graph launches;
- residual 2048-to-2112 padding for prompts above 2048.

The next piecewise optimization should change the runner ABI or make a partial final 64-token GDN chunk graph-safe. Adding a non-64-aligned bucket does not help because it reintroduces per-layer GDN padding work.

## Build and final state

- Final source changes: xllm/core/kernels/musa/gdn_prefill.cpp plus the already committed 2112 shoulder.
- Final build log: /workspace/xllm-git-master/build_logs/build_gdn_alias_piecewise_2112_final_20260801.log
- Final binary SHA256: a667b36f51c0f2f6273de6db82e5444fead2e1a5609c0c67040fbdac62f16008
- git diff --check: clean
- No benchmark server remains listening on port 8092.
