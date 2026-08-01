# Piecewise Replay Bottleneck Validation — 2026-08-01

## Executive conclusion

The current xLLM C1 prefill path is real piecewise replay, and its steady forward is faster than eager. The July 31 end-to-end result that made piecewise look slower is not a stable forward regression.

The primary piecewise-specific loss at this workload is bucket over-padding at the 2048-token boundary. The benchmark asks for input length 2000, but chat-template expansion produces 2040–2089 model tokens. Requests only one token above 2048 select the 2304 bucket, so all captured graph segments execute approximately 250 unnecessary token rows. This costs about 14.4 ms relative to eager at the boundary and reduces piecewise's forward advantage from about 24.4 ms below the boundary to about 10.0 ms above it.

Inside replay, captured graph compute is dominant. Metadata update, replay-parameter construction, and the 16 FA3 runners are not the bottleneck.

## Scope and environment

- Remote host: dev92
- Host checkout: /data/feihu/xllm-git-master
- xLLM container: xllm-musa2.9.1-sdk5.1-dev
- Branch: b2-mate-prefill-fix
- HEAD: 0ade546bd22083d440dcffb66a8dece52afa3e9c
- Model: /workspace/model_weights/Qwen3.5-27B-FP8
- Hardware: MTT S5000
- Core controlled shape: C1, requested ISL 2000, OSL 16, warmup 4, measure 20
- Client protocol: closed-loop barrier waves, identical seeds and sampling
- Main artifact root in container: /workspace/bench_results/xllm_piecewise_profile_20260801_083000
- Host-visible artifact root: /data/feihu/bench_results/xllm_piecewise_profile_20260801_083000

OSL 16 was used for bottleneck attribution so decode does not dominate wall time. The prefill request construction and input length are the same as the July 31 configuration. A fresh OSL 2000 eager comparison was also run earlier, but it is not used as the main causal comparison because the July 31 piecewise run contains a 577 ms TTFT outlier.

## Controlled xLLM A/B

Three piecewise and three eager measurement runs were interleaved. Client barrier-aligned TTFT means were:

| Trial | Mode | Mean barrier TTFT |
|---|---:|---:|
| trial_p1 | piecewise | 302.32 ms |
| trial_e1 | eager | 303.58 ms |
| trial_p2 | piecewise | 305.63 ms |
| trial_e2 | eager | 298.86 ms |
| trial_postbuild_p3 | piecewise, profile disabled | 291.51 ms |
| trial_postbuild_e3 | eager, same binary/time window | 297.65 ms |

The endpoint metric has several milliseconds of run-level noise: one pair favors piecewise, another favors eager. The same-binary post-build pair favors piecewise by 6.14 ms, or 2.06%. This is why one old run cannot establish a replay regression.

Server-side forward timing is consistent. The following aggregation uses the last 20 measured requests in each trial and separates actual model token counts at the bucket boundary:

| Mode | Actual tokens | Samples | Mean forward |
|---|---:|---:|---:|
| piecewise | at most 2048 | 4 | 228.64 ms |
| piecewise | above 2048 | 56 | 258.06 ms |
| eager | at most 2048 | 6 | 253.08 ms |
| eager | above 2048 | 54 | 268.07 ms |

Therefore:

- Below/equal to 2048, piecewise is 24.44 ms faster than eager.
- Above 2048, piecewise is still 10.01 ms faster than eager.
- The piecewise boundary jump is 29.43 ms.
- The eager boundary jump is 14.99 ms.
- The additional piecewise-specific boundary penalty is approximately 14.43 ms.

A final profile-disabled smoke on the final binary reproduced the discontinuity in one server lifetime:

- 2044 actual tokens, bucket 2048: 228.83 ms
- 2049 actual tokens, bucket 2304: 259.11 ms

Only five actual tokens changed, but forward time changed by 30.28 ms.

## Replay structure: real piecewise, not a flag no-op

The corrected capture log from the final smoke is:

    bucket_num_tokens: 2304, num_graphs: 17, num_runners: 16, num_instructions: 33
    bucket_num_tokens: 2048, num_graphs: 17, num_runners: 16, num_instructions: 33

The old log said num_graphs=33 because it printed instruction count, not graph count. The actual replay alternates 17 MUSA graph segments with 16 eager attention runners. The final smoke completed 5/5 requests, produced finite output, and had mean barrier TTFT 292.53 ms.

## Intrusive replay attribution

XLLM_PIECEWISE_PROFILE=1 synchronizes around each replay instruction. These timings are for attribution only and must not be compared directly with normal serving latency.

For the 2304 bucket, eight samples gave:

| Component | Mean | Share of graph plus runner time |
|---|---:|---:|
| 17 graph segments | 248.56 ms | 95.9% |
| 16 FA3 runners | 10.66 ms | 4.1% |
| Total | 259.22 ms | 100% |
| persistent_param update | 0.415 ms | separate, about 0.16% of total |
| replay parameter construction | 0.001 ms | negligible |

The single observed 2048-bucket profile was:

| Component | 2048 bucket | 2304 bucket mean | Increase |
|---|---:|---:|---:|
| Graph segments | 223.17 ms | 248.56 ms | 25.39 ms |
| FA3 runners | 8.18 ms | 10.66 ms | 2.47 ms |
| Total | 231.35 ms | 259.22 ms | 27.86 ms |

This independently matches the non-intrusive approximately 29–30 ms boundary discontinuity.

The 17 graph segment means at bucket 2304 are one approximately 12.7 ms head segment, fifteen approximately 15.4–15.8 ms middle segments, and one approximately 2.8 ms tail segment. There is no single anomalous graph launch; the extra padded work is distributed across the model.

## What dominates graph compute

An eager measurement-only operator breakdown, five post-warmup requests, averaged:

| Bucket | Mean |
|---|---:|
| MLP total | 135.89 ms |
| MLP gate/up | 84.85 ms |
| MLP down | 50.86 ms |
| GDN attention total | 101.02 ms |
| GDN projection | 36.86 ms |
| GDN output projection | 39.51 ms |
| Mate portion | 6.85 ms |
| Full attention total | 26.84 ms |
| Full QKV | 10.32 ms |
| Full output projection | 14.63 ms |
| Full FA kernel | 1.07 ms |
| Norm | 3.05 ms |
| Other | 6.70 ms |
| Total wall | 273.50 ms |

This shows why graph replay cannot produce a huge gain at 2K prefill: most time is real GEMM/model compute, not launch overhead. After bucket granularity, the next optimization targets are MLP gate/up and down projections, then GDN projection and output projection.

## FA3 runner falsification

A piecewise run with XLLM_USE_FA3=0 completed 20/20 requests but mean barrier TTFT increased from 291.51 ms to 305.61 ms, a 14.10 ms regression. FA3 runners are beneficial and should remain enabled. They are not the source of the slow piecewise result.

## SGLang reference validation

The imported SGLang runtime is /sgl-workspace/sglang at commit 12d78dd0ed535b09428d3cbae9cd9a55127091a8. It is not the mounted source checkout at /workspace/sglang_qwen35, although their core piecewise runner implementations match.

SGLang normally auto-disables piecewise CUDA graph for this Qwen3.5 class because it is classified as multimodal. Forcing piecewise with one 2304-token capture bucket did not produce a valid server:

| Trial | Model/compiler | Failure before readiness |
|---|---|---|
| pcg_r1 | FP8, default | Torch Dynamo graph break at GenericContextWrappingVariable |
| pcg_runtime_ctxfix_r1 | FP8, diagnostic context bypass | ModelWeightParameter tensor-subclass attribute access unsupported |
| pcg_inductor_r1 | FP8, inductor | Same ModelWeightParameter failure |
| pcg_bf16_r1 | BF16, diagnostic context bypass | Generated FX code NameError: device is not defined |

The temporary context-bypass diagnostics were reverted from both SGLang checkouts. No SGLang runtime source changes remain.

A valid SGLang eager baseline completed 20/20 at C1, ISL 2000, OSL 16:

- Mean barrier TTFT: 272.45 ms
- Mean TTFT: 272.44 ms
- Mean TPOT: 30.07 ms

There is no valid SGLang Qwen3.5 piecewise result, so no SGLang piecewise-versus-eager speed claim is justified. The forced failures explain why the reference implementation auto-disables this path.

SGLang artifacts:

- /workspace/sglang_qwen35/benchmark_results/sglang_qwen35_27b_fp8_pcg_eager_c1_2k_16_20260801
- /workspace/sglang_qwen35/benchmark_results/sglang_qwen35_27b_bf16_pcg_eager_c1_2k_16_20260801

## Code changes retained

Two xLLM files contain default-off diagnostic instrumentation:

- xllm/core/kernels/musa/piecewise_graphs.cpp
- xllm/core/runtime/musa/musa_graph_executor_impl.cpp

XLLM_PIECEWISE_PROFILE=1 enables per-segment synchronization and logging. With the variable absent, replay takes the original hot loop without per-instruction profiling checks. The capture log now distinguishes graph count, runner count, and instruction count.

The focused diff passes git diff --check. The full dirty worktree has unrelated pre-existing CRLF/trailing-whitespace findings, especially fp8_act_quant.cu; those files were not modified in this validation.

Final build:

- Binary size: 202595680 bytes
- Binary mtime: 2026-08-01 09:18:00 +0800
- Build log: /workspace/bench_results/xllm_piecewise_profile_20260801_083000/build_final.log
- Final smoke: /workspace/bench_results/xllm_piecewise_profile_20260801_083000/trial_final_smoke

## Recommended next validation

The cheapest direct fix experiment is to add a 2112-token shoulder bucket between 2048 and 2304, rebuild, and rerun the same interleaved C1 A/B. Current actual lengths are mostly 2049–2089, so 2112 should remove most padded graph work while adding only one capture shape.

Acceptance criteria:

1. Requests from 2049 through 2112 select 2112, not 2304.
2. Piecewise forward above 2048 improves by at least 10 ms without output corruption.
3. Repeated interleaved endpoint TTFT is no worse than eager and has stable direction.
4. C4 packed-prefill correctness remains gated separately; the previously documented C4 NaN means no C4 performance claim is allowed.

## Cleanup status

At the final audit all eight GPUs reported 0 MiB and no running accelerator processes. No benchmark ports were listening. The xLLM container has many historical defunct xllm entries owned by its init process, but none holds a port or GPU allocation.
