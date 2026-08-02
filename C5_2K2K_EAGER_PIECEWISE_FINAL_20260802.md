# C=5, ISL=2K, OSL=2K eager/piecewise validation (2026-08-02)

## Outcome

The final fixed-request, matched-cap results establish two valid performance
gates and leave the SGLang-piecewise gate unevaluable:

1. **PASS:** xLLM eager is faster than SGLang eager.
   - 742.68 ms versus 831.63 ms mean TTFT.
   - xLLM is 88.94 ms / 10.70% faster.
2. **PASS:** xLLM piecewise graph is faster than xLLM eager.
   - 731.40 ms versus 742.68 ms mean TTFT.
   - Piecewise is 11.28 ms / 1.52% faster.
3. **NO VALID COMPARISON:** SGLang Qwen3.5 FP8 piecewise still has no runnable
   result in `sglang-musa-reference`. Forced PCG can be made to compile and
   capture with diagnostic source changes, but real replay either raises a
   MUSA error or hangs. It is therefore invalid to claim a measured
   xLLM-piecewise versus SGLang-piecewise speedup.

xLLM piecewise is 100.23 ms / 12.05% faster than the valid SGLang eager
baseline, but this is not a substitute for gate 3.

## Final protocol

- Model: `Qwen3.5-27B-FP8`.
- Device: one MTT S5000 MUSA device, physical GPU 3.
- Client concurrency: barrier waves of 5 requests.
- Measured work: 10 waves / 50 requests.
- Nominal client input/output: ISL 2000 / OSL 2000.
- Sampling: temperature 0.9, top-k 20, top-p 0.95, 2000 output tokens.
- Cache: prefix/radix cache disabled.
- Prefill request cap: 1 in both runtimes.
- Decode concurrency: up to 8 in both runtimes; C=5 decode remains enabled.
- Warmup: one 5-request wave containing both observed 2048 and 2080 xLLM
  graph buckets.
- Fixed-request SHA256:
  `a9a3832e0bfa7af1c17df2410655beaef1fa49b08bc746bc969e68a213fc07af`.

The fixed request file is:

`/workspace/bench_results/c5_2k2k_latestfix_20260802/fixed_requests_warm1_measure10_dual_bucket_verified.json`

## Final OSL=2K results

| Runtime/mode | Completed | Mean TTFT | Barrier TTFT | Mean TPOT | Output throughput | Duration | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| xLLM eager, strict prefill cap 1 | 50/50 | 742.68 ms | 744.01 ms | 33.33 ms | 148.43 tok/s | 673.73 s | 0 |
| xLLM piecewise, strict prefill cap 1 | 50/50 | 731.40 ms | 732.42 ms | 33.92 ms | 145.90 tok/s | 685.40 s | 0 |
| SGLang eager, prefill cap 1 | 50/50 | 831.63 ms | 832.59 ms | 34.93 ms | 141.50 tok/s | 706.72 s | 0 |
| SGLang piecewise | N/A | N/A | N/A | N/A | N/A | N/A | startup/replay NO-GO |

All three valid runs processed exactly 100000 input and 100000 output tokens.
The slower xLLM-piecewise TPOT in this particular long run shows about 1.8%
decode/device drift. Despite that unfavorable drift, its TTFT remained lower.

Artifacts:

- xLLM eager:
  `/workspace/bench_results/c5_2k2k_latestfix_20260802/final_cap1_strict_dualwarm/xllm_eager`
- xLLM piecewise:
  `/workspace/bench_results/c5_2k2k_latestfix_20260802/final_cap1_strict_dualwarm/xllm_piecewise`
- SGLang eager:
  `/workspace/sglang_qwen35/benchmark_results/c5_2k2k_latestfix_20260802/final_cap1_fixed_dualwarm/sglang_eager`

## Why the earlier xLLM eager comparison was invalid

`ENABLE_PACKED_PREFILL=0` disabled the packed layout but did not cap the
number of ordinary dense prefill sequences. At external C=5, xLLM eager formed
B=2/3/4 prefill forwards. Representative forward times were approximately
510, 750, and 1000 ms. SGLang was explicitly running
`prefill_max_requests=1`, so the earlier xLLM 1055.68 ms versus SGLang
831.63 ms comparison did not have matched prefill admission semantics.

The retained scheduler change adds `XLLM_MAX_PREFILL_SEQS`. Setting it to 1
caps pure-prefill requests without lowering `max_seqs_per_batch`, so decode
still runs at C=5. In the final eager run all 56 logged prefill scheduling
steps had `n_seq=1`.

An OSL=16 isolation run confirms the root cause:

| Mode | Mean TTFT | Barrier TTFT | Mean prefill forward |
|---|---:|---:|---:|
| xLLM eager before independent cap | 1047.10 ms | 1047.95 ms | mixed B=1..4 |
| xLLM eager, cap 1 | 745.05 ms | 745.77 ms | 233.74 ms |
| xLLM piecewise, cap 1 | 731.47 ms | 731.94 ms | 228.92 ms |
| SGLang eager, cap 1 | 834.87 ms | 835.27 ms | not instrumented |

The piecewise forward saves 4.82 ms per C1 prefill. The cumulative C=5 TTFT
gain is therefore expected to be around 3 times that value, matching the
observed 13.83 ms OSL=16 barrier-TTFT gain.

OSL=16 artifacts are under:

`/workspace/bench_results/c5_2k2k_latestfix_20260802/diagnostics/c5_osl16`

## Piecewise replay evidence

- Both 2048 and 2080 buckets were captured during warmup.
- Each bucket contains 17 graphs, 16 eager runners, and 33 instructions.
- No new capture occurred during the final measured requests.
- All final piecewise prefill batches were C1.
- OSL=2K piecewise was faster in 9/10 paired waves.
- Paired barrier-TTFT gain: 11.59 ms mean; approximate 95% CI
  `[6.14, 17.04]` ms.
- OSL=16 piecewise was faster in 9/10 paired waves.
- Paired barrier-TTFT gain: 13.83 ms mean; approximate 95% CI
  `[9.25, 18.41]` ms.

## Correctness

A final temperature-0, C=5, prefill-cap-1 run recorded 50 generated texts
from both xLLM eager and piecewise. The request-index/output-length/text tuples
were exactly equal:

`3aa157cfacae5dbed00f53e87b5427d49a9ddaaab2f54d20498425cf9378d9b8`

Artifacts:

- `/workspace/bench_results/c5_2k2k_latestfix_20260802/correctness_c5_cap1/eager`
- `/workspace/bench_results/c5_2k2k_latestfix_20260802/correctness_c5_cap1/piecewise`

Packed B=2 FP8 piecewise remains a separate NO-GO: the diagnostic capture
produced NaN logits and a sampling failure. The validated production path in
this report is C1 prefill with C=5 decode.

## SGLang piecewise diagnostics

The unmodified reference auto-disables Qwen3.5 PCG because the model is
classified as multimodal. Forced PCG failed at a TorchDynamo context-manager
graph break and then at `ModelWeightParameter.__torch_function__`.

Diagnostic patches advanced it through compile and capture by bypassing the
statistics-only layer context during PCG, demoting loader-only parameter
subclasses after weight load, and injecting `device=torch.device` into MUSA FX
GraphModule globals. Real replay still failed or hung for both sparse and
small-shape capture lists. No successful measured request set exists.

The diagnostic patch was saved and the SGLang source changes were reverted:

- Patch:
  `/workspace/sglang_qwen35/benchmark_results/c5_2k2k_latestfix_20260802/diagnostics/sglang_piecewise_runtime_diagnostics.patch`
- SHA256:
  `91ede6356759a15dab92a9af005a2f20f7643d1c1fb6888791453907d82bb497`

Therefore gate 3 remains fail-closed. It requires a runnable, correctness-valid
SGLang Qwen3.5 FP8 PCG implementation before a speed comparison is permitted.
