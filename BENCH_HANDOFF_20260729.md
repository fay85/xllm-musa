# Qwen3.5-27B BF16 graph benchmark handoff

## Scope

Compare the current post-refactor MUSA build with the immediate pre-refactor
baseline. The user changed the C5 protocol to **4 warmup waves + 20 measured
requests** (wave size 5), rather than 50 measured requests.

## Authoritative runtime

- SSH host: `mccxadmin@10.121.38.92`
- Container: `xllm-musa2.9.1-sdk5.1-dev`
- Active source: `/workspace/xllm-musa-main-sync`
- Branch/HEAD: `feat/musa-main-sync`, `a8f564e4`, plus the large staged merge
  index used by the current binary (the build is not attributable to HEAD alone)
- Binary: `/workspace/xllm-musa-main-sync/build/lib.linux-x86_64-cpython-310/xllm/xllm`
- Model: `/workspace/model_weights/Qwen3.5-27B` (BF16)
- Device: `MUSA_VISIBLE_DEVICES=1`

All server and benchmark processes must run inside the container. Do not use
the host-mounted Windows checkout for builds or validation.

## Runtime environment required for the binary

```bash
LD_LIBRARY_PATH=/workspace/mudnn_3.4.0/lib:/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/musa/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/openmpi/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib
LD_PRELOAD=/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so:/workspace/artifacts/libmusa_backend_init.so:/opt/intel/oneapi/mkl/lib/intel64/libmkl_core.so.2
FLASHINFER_OPS_PATH=/workspace/mate_cached_ops
XLLM_USE_FA3=1
XLLM_USE_FA3_DECODE=1
XLLM_USE_CUSTOM_PREFILL_CONV=1
XLLM_TOKEN_MAJOR_PREFILL_CONV=1
XLLM_FUSED_GDN_QK_L2NORM=1
XLLM_PACKED_PREFILL_PIECEWISE=0
XLLM_PREFILL_FWD_TIMING=0
XLLM_PREFILL_STEP_TIMING=0
XLLM_PREFILL_BREAKDOWN=0
XLLM_REQUEST_TIMING=0
XLLM_KINETO_PROFILE_PREFILL=0
XLLM_SCHED_PACK_LOG=0
XLLM_TILELANG_LIB=/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so
```

Missing `LD_PRELOAD` causes `PrivateUse1HooksInterface` startup failure.
Missing `FLASHINFER_OPS_PATH` causes a runtime FATAL on the first matmul URI
lookup. Both failures are environment issues, not benchmark results.

## Matched historical baselines

Pre-split shared-CUDA-path baseline:

- C1 result directory:
  `/workspace/bench_results/official_Qwen3.5-27B_c1_isl20k_osl300_20260724_114204`
  - graph on, packed prefill, adaptive oneshot on, ISL=20000, OSL=300,
    4 warmup + 10 measure, C=1
  - mean TTFT 3500.07 ms; mean TPOT 46.677 ms; output 17.185 tok/s
- C5 result files:
  `/workspace/bench_results/use_cuda_off_phase1_c5_matched_20260724/result.json`
  and
  `/workspace/bench_results/use_cuda_off_phase1_c5_repeat_20260724/result.json`
  - graph + piecewise + packed prefill, C=5, ISL=2500, OSL=1500,
    prefix=200, wave size 5, 4 warmup waves + 10 measure waves (50 requests)
  - average of the two runs: mean TTFT 1522.646 ms, barrier TTFT 1524.012 ms,
    mean TPOT 52.503 ms, TGS 249.290 tok/s

An earlier current C5 run with 50 measured requests completed 50/50, but the
user changed the requested sample size. Its recorded values were mean TTFT
2297.557 ms, barrier TTFT 2298.690 ms, mean TPOT 57.986 ms, TGS 224.078
tok/s; do not use it as the final 20-request comparison.

## Correct C5 server options

For a strict historical-baseline match, use graph on, piecewise graph on,
adaptive oneshot off, packed prefill on, `max_tokens_per_batch=16384`,
`max_tokens_per_chunk_for_prefill=8192`, `max_seqs_per_batch=8`,
`max_concurrent_requests=8`, and `max_linear_state_cache_slots=0`. The old run
resolved an effective conv-cache shape with 10 slots. Verify the resolved
`Master init options` line in the server log before sending requests.

The historical July 24 binary predates the `XLLM_USE_FA3_DECODE` flag. The
current best-performance run keeps `XLLM_USE_FA3_DECODE=1`, so report this as
an attention-backend difference rather than claiming a one-variable code-only
A/B.

## 20-request C5 benchmark command

After starting the server on port 18133 with the options above:

```bash
cd /workspace/xllm-musa-main-sync
mkdir -p /workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/c5_clean_20req
python3 benchmark_c5_barrier_waves.py \
  --host 127.0.0.1 --port 18133 \
  --model Qwen3.5-27B \
  --tokenizer /workspace/model_weights/Qwen3.5-27B \
  --input-len 2500 --output-len 1500 --prefix-len 200 \
  --wave-size 5 --warmup-waves 4 --num-waves 4 \
  --seed 44002 --warmup-seed 44001 \
  --result-json /workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/c5_clean_20req/result.json \
  --benchmark-lib-dir /workspace \
  2>&1 | tee /workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/c5_clean_20req/bench.log
```

`num-waves=4` with wave size 5 gives 20 measured requests. Keep the result
JSON and benchmark log under `c5_clean_20req/`; do not overwrite historical
baseline files.

The strict 20-request run completed successfully at
`/workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/c5_clean_20req/result.json`:
20/20 successful, mean TTFT 2398.70 ms, barrier TTFT 2399.53 ms, mean TPOT
58.19 ms, and TGS 223.10 tok/s. Against the matched-baseline average (TTFT
1522.646 ms, barrier 1524.012 ms, TPOT 52.503 ms, TGS 249.290 tok/s), this is
TTFT +57.5%, TPOT +10.8%, and TGS -10.5%. The server was stopped after the
run.

## C1 follow-up requirement

The first clean C1 run used `max_concurrent_requests=4`, so the engine
resolved `max_seqs_per_batch=4`; it is not a strict match to the old adaptive
oneshot baseline. If rerunning C1, set both `max_seqs_per_batch=8` and
`max_concurrent_requests=8`, and use the launcher’s adaptive-oneshot setting.
Verify `enable_adaptive_prefill_oneshot=1` in `Master init options` (and the
`[SCHED_PACK] adaptive_oneshot=1` log) before measuring.

## Current diagnosis clues

The current C5 50-request run showed about +10.4% TPOT regression. Two likely
post-refactor hotspots identified by read-only audit are:

1. `xllm/core/kernels/musa/llm_decode_metadata_update.cu`: an extra
   single-thread cumulative metadata kernel launch each decode step.
2. `xllm/core/runtime/llm_worker_impl.cpp`: removal of the persistent MUSA
   `compute_stream_` path, potentially adding per-step FFI/stream sync.

Do not modify either file as part of this benchmark-only handoff; profile or
run an explicitly approved A/B experiment next.
