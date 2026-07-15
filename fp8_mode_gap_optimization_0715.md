# Qwen3.5-27B-FP8 xLLM vs SGLang optimization log (2026-07-15)

## Scope and fixed benchmark protocol

- Model: `Qwen3.5-27B-FP8`, dynamic activation quantization, 128x128 block-wise FP8 weights.
- C=5: ISL=2500, OSL=1500, 5 warmup requests, 50 measured requests.
- C=1: ISL=20000, OSL=300, 4 warmup requests, 10 measured requests.
- xLLM flags are the same best-performing graph, packed-prefill, FA3, fused-GDN, and adaptive-one-shot settings used in `bench_result_fp8_0714.md`.
- Total TGS below is input-token throughput plus output-token throughput.

## FP8 path comparison and initial blockers

Both runtimes ultimately use the same MUSA mate/muDNN group-wise FP8 GEMM primitive. The important differences are around it:

1. SGLang creates `weight_scale_inv` as FP32 and converts the checkpoint scale once while loading. Before Fix 1, xLLM retained the checkpoint-native BF16 scale grid and executed `to(torch::kFloat32).contiguous()` inside every block-FP8 linear forward. Under decode graph capture this conversion becomes part of every graph replay, so it affects TPOT as well as eager prefill.
2. SGLang's specialized BF16-to-FP8, group-128 MUSA quantizer uses 16 lanes with 8 BF16 values per lane. xLLM currently uses 32 lanes with 4 values per lane and a fixed eight groups per block. A direct lane-count port was tested below and rejected; the complete SGLang load/store and launch policy is needed before revisiting it.
3. The group-wise GEMM implementation itself is not yet evidence of a runtime difference: both paths call mate with row-major FP32 activation scales, row-major FP32 weight scales, `(1,128,128)` granularity, and the muDNN backend.

## Fix 1: convert block-weight scales once at model load

### Change

File: `xllm/core/layers/common/linear.cpp`

- Register block-FP8 `weight_scale_inv` parameters as FP32 in column-parallel, QKV-parallel, and row-parallel linears.
- Let the existing checkpoint `copy_` convert BF16 checkpoint values to FP32 once during loading.
- Pass the persistent, contiguous FP32 scale tensor directly to the native block-FP8 GEMM.
- Add dtype and contiguity checks at the native GEMM boundary.
- Remove the per-forward `to(FP32).contiguous()` conversion and allocation.

### Rebuild and correctness

- Rebuild: PASS via `./_build_cuda_graph_musa.sh`.
- Build log: `build_logs/build_cuda_graph_musa_20260714_234546.log`.
- FP8 correctness: PASS with eager prefill plus graph decode, HTTP completion, non-garbage output, generated tokens, and expected arithmetic result `391`.
- Correctness log: `build_logs/correctness_fp8_scale_cache_prodflags_256_20260714_235113.log`.
- A first tiny-prompt smoke using piecewise-prefill graph hit the pre-existing `Piecewise graph must have attention runners` harness assertion. Re-running with the official eager-prefill flags passed. A later first long run ended prematurely, but isolated 20k/OSL=1 tests with and without `MUSA_LAUNCH_BLOCKING`, a 20k/OSL=300 test, and the complete official tests below all passed; the failure was not reproducible.

### Official performance validation

| Workload | Metric | Before | After Fix 1 | Delta |
|---|---:|---:|---:|---:|
| C=5, 2.5k/1.5k | Successful requests | 50/50 | 50/50 | stable |
| C=5, 2.5k/1.5k | Mean TTFT | 1384.91 ms | 1295.53 ms | -89.38 ms (-6.45%) |
| C=5, 2.5k/1.5k | Mean TPOT | 36.32 ms | 35.29 ms | -1.03 ms (-2.84%) |
| C=5, 2.5k/1.5k | Total TGS | 356.85 tok/s | 365.95 tok/s | +9.10 tok/s (+2.55%) |
| C=1, 20k/300 | Successful requests | 10/10 | 10/10 | stable |
| C=1, 20k/300 | Mean TTFT | 2663.99 ms | 2685.94 ms | +21.95 ms (+0.82%) |
| C=1, 20k/300 | Mean TPOT | 35.82 ms | 35.37 ms | -0.45 ms (-1.26%) |
| C=1, 20k/300 | Total TGS | 1464.25 tok/s | 1475.10 tok/s | +10.85 tok/s (+0.74%) |

Artifacts:

- Before, C=5: `/data/feihu/bench_results/official_Qwen3.5-27B-FP8_c5_isl2500_osl1500_20260714_225623`
- After, C=5: `/data/feihu/bench_results/official_Qwen3.5-27B-FP8_c5_isl2500_osl1500_20260715_000417`
- Before, C=1: `/data/feihu/bench_results/official_Qwen3.5-27B-FP8_c1_isl20k_osl300_20260714_232028`
- After, C=1: `/data/feihu/bench_results/official_Qwen3.5-27B-FP8_c1_isl20k_osl300_20260714_235952`

### Decision

**KEEP.** The 50-request test shows a material improvement to both TTFT and TPOT, and the independent long workload confirms the TPOT/throughput direction. The +22 ms long-prompt TTFT difference is below 1% and is outweighed by the measured decode and total-throughput gains.

## Next step

The accepted change removes the recurrent scale-conversion overhead. The remaining long-ISL gap is now primarily the eager 20k prefill forward (roughly 2.56--2.65 s in xLLM's `[PREFILL_FWD]` traces versus SGLang's 2.48 s end-to-end TTFT), plus the still higher FP8 decode cost. Further work should port SGLang's complete MUSA quantizer (non-caching vector load/store intrinsics and its per-hidden-dimension launch policy) or replace the decomposed quantize-plus-GEMM sequence with a fused operation; changing the lane count alone is not beneficial.

## Fix 2 experiment: direct 16-lane quantizer port (rejected)

### Change tested

The xLLM activation quantizer was temporarily changed to the SGLang-style 16-lane/8-value mapping with 16 groups per 256-thread block. Tensor layout, row-major scale output, and the mate GEMM contract were unchanged.

### Rebuild and correctness

- Rebuild: PASS via `./_build_cuda_graph_musa.sh` (`build_logs/build_cuda_graph_musa_20260715_001818.log`).
- FP8 correctness: PASS on the deterministic arithmetic smoke (`build_logs/correctness_fp8_quant16_20260715_0020*.log`).

### Performance result and decision

The exact C=1 long-ISL test regressed relative to the accepted Fix 1 binary:

- Fix 1: mean TTFT 2685.94 ms, mean TPOT 35.37 ms, total TGS 1475.10 tok/s.
- 16-lane experiment: mean TTFT 2697.26 ms, mean TPOT 35.76 ms, total TGS 1461.30 tok/s.
- All 10 measured requests succeeded in both runs.

**REJECT and revert.** The lane count cannot be ported independently: SGLang's gain comes with its specialized non-caching vector loads/stores and shape-aware 2D scheduler. The final rebuild restores the original 32-lane kernel plus Fix 1, and a final FP8 correctness pass succeeds (`build_logs/correctness_fp8_final_fix1_20260715_*.log`).

A second shape-aware variant (16 lanes, SGLang-style subwarp selection: 8 for 40 groups, 16 for 80/136 groups) was also rebuilt (`build_logs/build_cuda_graph_musa_20260715_003256.log`) and passed correctness (`build_logs/correctness_fp8_quant16_shapeaware_20260715_*.log`). Its exact long measurement was 2672.90 ms TTFT / 35.46 ms TPOT / 1473.61 tok/s, statistically neutral versus Fix 1's 2685.94 / 35.37 / 1475.10; its C=5 warmup TPOT was already worse (35.67 ms versus 35.24 ms). It was therefore reverted as well; the final rebuild is `build_logs/build_cuda_graph_musa_20260715_004401.log`.

## Accepted Fix 1 versus SGLang after the change

| Workload | Metric | xLLM after Fix 1 | SGLang | Remaining xLLM gap |
|---|---:|---:|---:|---:|
| C=5, 2.5k/1.5k | Mean TTFT | 1295.53 ms | 1270.35 ms | +25.18 ms (+1.98%) |
| C=5, 2.5k/1.5k | Mean TPOT | 35.29 ms | 34.21 ms | +1.08 ms (+3.16%) |
| C=5, 2.5k/1.5k | Total TGS | 365.95 tok/s | 381.35 tok/s | xLLM 95.96% of SGLang |
| C=1, 20k/300 | Mean TTFT | 2685.94 ms | 2483.32 ms | +202.62 ms (+8.16%) |
| C=1, 20k/300 | Mean TPOT | 35.37 ms | 34.49 ms | +0.88 ms (+2.55%) |
| C=1, 20k/300 | Total TGS | 1475.10 tok/s | 1586.32 tok/s | xLLM 92.99% of SGLang |
