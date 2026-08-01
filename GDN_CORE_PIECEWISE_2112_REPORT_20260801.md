# GDN Core and Piecewise 2112-Bucket Report (2026-08-01)

## Scope

This stage continued the Qwen3.5-27B-FP8 GDN prefill investigation on MUSA and re-tested eager versus piecewise graph replay on the same xLLM binary.

- Host checkout: /data/feihu/xllm-git-master
- Container: xllm-musa2.9.1-sdk5.1-dev
- Model: /workspace/model_weights/Qwen3.5-27B-FP8
- Device: physical MUSA GPU 3
- Workload: C=1, nominal ISL=2000, OSL=16, temperature=0.9, top-k=20, top-p=0.95
- Common runtime: FA3 on, packed prefill off, MATE GDN backend, graph on
- Evidence root: /workspace/bench_results/gdn_core_piecewise_20260801_175500

## Finding 1: the old piecewise result was dominated by the 2304 bucket

A fresh same-binary reverse-ordered baseline with the stride-aware MATE implementation selected bucket 2304 for actual prefill lengths near 2040-2089.

| Mode | Mean TTFT (ms) | Mean server prefill forward (ms) |
|---|---:|---:|
| eager (2 trials) | 268.472 | 235.518 |
| piecewise 2304 (2 trials) | 278.971 | 245.620 |
| piecewise - eager | +10.499 | about +10.1 |

Piecewise profiling confirmed real graph replay: 17 graph segments and 16 FA3/MATE runners. Graph segments consumed about 236-243 ms and runners about 11 ms; metadata update was only about 0.4 ms. The bottleneck was graph compute on padded bucket work, not a fake replay path and not runner dispatch.

Evidence: eager_r1, piecewise_r1, piecewise_r2, eager_r2, and piecewise_profile_2304 below the evidence root.

## Implemented fix: add a 2112-token shoulder

generate_piecewise_prefill_graph_tokens() now includes a 2112-token shoulder. This prevents a prompt crossing 2048 by one token from jumping directly to 2304, while preserving the 64-token MATE GDN/KKT granularity.

The first warmup-4 paired validation selected bucket 2112 and recovered 8.852 ms of the prior TTFT gap:

| Mode | Mean TTFT (ms) | Mean server prefill forward (ms) |
|---|---:|---:|
| eager | 270.433 | 236.969 |
| piecewise 2112 | 272.080 | 238.409 |
| piecewise - eager | +1.647 | +1.440 |

Evidence: after_2112/{piecewise_r1,eager_r1,eager_r2,piecewise_r2}.

## Steady replay result

Warmup=4 is not sufficient for this prompt generator because measured requests can land on both sides of 2048. One final run first captured 2112 during warmup, then lazily captured 2048 on measured request 3, producing a 540 ms TTFT outlier. That run is retained under final_validation but is not a valid steady-replay mean.

The final protocol used warmup=20 and measure=20 so both 2048 and 2112 were captured before measurement. Two opposite orders were run on the same production binary:

| Order | eager TTFT (ms) | piecewise TTFT (ms) | piecewise - eager (ms) | eager forward (ms) | piecewise forward (ms) |
|---|---:|---:|---:|---:|---:|
| eager -> piecewise | 270.253 | 268.151 | -2.102 | 237.174 | 234.105 |
| piecewise -> eager | 269.649 | 271.367 | +1.718 | 235.924 | 237.257 |
| order-balanced mean | 269.951 | 269.759 | **-0.192** | 236.549 | 235.681 |

All 80 measured requests completed successfully. Piecewise processed about 1.25 more actual tokens on average across the two trials, yet its server-forward point estimate remained 0.868 ms faster.

Conclusion: after the 2112 fix, steady piecewise replay is no longer materially slower than eager. It is at parity with a small favorable point estimate, not yet a statistically strong speedup. The expected graph advantage is currently almost completely consumed by residual bucket padding and 17 graph/16 runner boundaries.

Evidence: final_validation_warm20 and final_validation_warm20_reverse.

## GDN core investigation

The current detailed breakdown and SGLang comparison indicate that the MUSA GDN primitives are already close to reference performance:

- xLLM GDN projections are about 31 ms, similar to SGLang's 31-34 ms.
- xLLM token-major causal conv is about 11.4 ms. Disabling it raises conv time to about 38.9 ms, so the optimized token-major path must remain enabled.
- xLLM MATE is about 16.1 ms total across the GDN stack, and the nested GDN core components sum to about 32.7 ms, close to SGLang's approximately 32.4 ms GDN core.
- The remaining eager GDN outer gap is distributed host/launch spacing across layers rather than one slow MATE or conv kernel.

Evidence: gdn_conv_ab and EAGER_GAP_XLLM_VS_SGLANG_20260801.md.

## Falsified optimizations (reverted)

1. **2080 bucket with relaxed 64-alignment coupling**: correctness smoke passed, but the regular GDN path internally padded/copied and replay was slower. The experiment was reverted.
2. **C=1 packed-varlen GDN in piecewise**: 5/5 requests succeeded, but graph segmentation expanded from 17/16 to 65/64 and TTFT regressed to about 294 ms. The experiment was reverted. Evidence: gdn_packed_c1_smoke*.
3. **Caching MATE module availability probes**: reverse-order A/B showed cache-on TTFT 269.252 ms versus cache-off 268.971 ms, with no forward improvement. The experiment was reverted. Evidence: gdn_module_cache_ab.

## Build and final state

The retained source change is limited to xllm/core/runtime/musa/musa_graph_executor_impl.cpp.

- Final build log: /workspace/xllm-git-master/build_logs/build_piecewise_2112_final_20260801.log
- Final binary SHA256: bcab4a681a2b250fd663238a524cc3f40ab9ed6c5b12ca9c92fe1e1fc0aa01f3
- git diff --check: clean
- No benchmark server remains listening on port 8092.

## Next optimization target

The next useful GDN/piecewise work should target one of these two measured costs:

1. reduce the 17 graph / 16 runner boundaries without routing C=1 through the current packed-varlen path; or
2. add capture prewarming/policy so neighboring 2048 and 2112 buckets cannot first-capture inside a measured or production request.

Further work on MATE primitive math or the token-major conv is lower priority because those kernels are already near SGLang parity and the attempted host module cache produced no gain.
