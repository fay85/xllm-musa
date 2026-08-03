# MUSA defaults and piecewise-prefill validation (2026-08-03)

## Scope and authoritative environment

- Branch: `codex/dev-optimizations-sync-20260802`
- Parent commit: `3a6fee5942f68dadec7201681fc9bbc7667e4c93`
- Host/container: `dev92` / `xllm-musa2.9.1-sdk5.1-dev`
- Checkout: `/workspace/xllm-git-master-HEAD`
- Build: `USE_MUSA=ON`, `USE_CUDA=OFF`
- Existing vcpkg root: `/workspace/vcpkg-xllm` (no dependency download)
- Validated binary SHA256:
  `4c92dd3e7c40dc9968579510fa8a008cb1e27bcac3fadbb4aae99f1587452f51`
- Model: `/workspace/model_weights/Qwen3.5-27B-FP8`

## Defaults implemented

MUSA builds now use the validated defaults even when the binary is launched
without performance flags:

| Setting | MUSA default | Non-MUSA default |
|---|---:|---:|
| graph execution | on | unchanged: off |
| decode graph without padding | on | unchanged: off |
| piecewise prefill graph | on | unchanged: off |
| graph VMM pool | off | unchanged: on |
| maximum graph tokens | 8192 | unchanged: 2048 |
| packed prefill | off | unchanged: off |

The MUSA FA3 selector now defaults to FA3 for BF16 activation shapes with
`head_dim=256` and GQA ratio 6 or 8. Explicit rollback remains available:

```bash
XLLM_USE_FA3=0 XLLM_USE_FA3_DECODE=0
```

The tracked `run_xllm_musa.sh` additionally fixes the runtime environment and
sets these explicit defaults:

```text
ENABLE_GRAPH=1
ENABLE_GRAPH_DECODE_NO_PADDING=1
ENABLE_GRAPH_VMM_POOL=0
ENABLE_PREFILL_PIECEWISE_GRAPH=1
ENABLE_PACKED_PREFILL=0
MAX_TOKENS_FOR_GRAPH_MODE=8192
XLLM_USE_FA3=1
XLLM_USE_FA3_DECODE=1
XLLM_MUSA_POOL_COMPUTE_STREAM=1
XLLM_PIECEWISE_CAPTURE_FA3=1
XLLM_GDN_DECODE_BACKEND=mate
```

The launcher resolves the binary relative to its own checkout, uses only
`MUSA_VISIBLE_DEVICES` for device selection, validates the exact GQA6/GQA8
Mate FA3 artifacts, and rejects incomplete MTP configuration before starting
the server.

## Exact-2048 piecewise profile

Protocol: C=1, two unprofiled warmups, then one online-Kineto profiled replay
with exactly 2048 server-side prompt tokens and one generated token. The same
binary, fixed requests, and GPU were used for all arms.

The three profile arms used parent binary SHA256
`7c649f6cc27434ac217aadcb47e78aa6875c9827a403aa2b15cc15273a0ea794`.
They explicitly enabled FA3 and the graph settings, so the final change to
absent-environment defaults does not alter the profiled execution path.

Artifact root:

```text
/workspace/bench_results/exact2048_replay_profile_20260803_113000
```

| Arm | Prefill structure | Server TTFT | Main stream wait | Visible FA3 GPU time |
|---|---:|---:|---:|---:|
| eager | eager kernels | 229 ms | 146.405 ms after eager submission | 5.658 ms |
| default full capture | 1 graph / 0 runners | 228 ms | 223.485 ms | captured/opaque |
| forced split | 17 graphs / 16 runners / 33 instructions | 229 ms | 220.926 ms | 5.734 ms |

For the forced-split arm (`XLLM_PIECEWISE_CAPTURE_FA3=0`):

- 17 prefill `musaGraphLaunch` calls consumed 358.208 microseconds of CPU time.
- Submitting the 17 graph segments and 16 external runners spanned about
  3.324 milliseconds.
- The 16 external FA3 kernels consumed 5.734 milliseconds of GPU time.
- The remaining time is graph-internal GPU computation, which Kineto does not
  expose as individual kernels on this MUSA stack.

Conclusion: the 33 graph/eager boundaries are not the main C1 TTFT bottleneck.
The current bounded full-FA3 capture already reduces the safe exact-2048 C1
case to one graph, but total TTFT remains compute-bound at about 228-229 ms.

Trace summary:

```text
/workspace/bench_results/exact2048_replay_profile_20260803_113000/trace_summary.json
```

## Correctness gate

Protocol: one C=2 wave, 2000 content tokens per request, deterministic sampling,
thinking disabled. The launcher received only model/GPU/port/capacity inputs;
all performance settings above came from its defaults.

Result:

```text
17 * 23                         -> FINAL: 391
125 red balls + 267 blue balls -> FINAL: 392
```

Artifact:

```text
/workspace/bench_results/defaults_c2_correctness_final_20260803_121000
```

During launcher validation, a reversed `LD_LIBRARY_PATH` construction was
found to put `/usr/local/musa/lib` ahead of the required
`/workspace/mudnn_3.4.0/lib`. That produced non-finite logits and an
`invalid multinomial distribution` abort. Preserving the declared library
priority fixed the failure; the final C2 gate and the long benchmark below
both pass without NaN, graph abort, or FATAL logs.

## C1 ISL=2k, OSL=2k performance gate

Protocol: closed-loop C=1, 4 warmup requests, 10 measured requests,
`temperature=0.9`, `top_k=20`, `top_p=0.95`, and `ignore_eos=true`.

| Metric | Pre-change artifact | New defaults | Delta |
|---|---:|---:|---:|
| mean TTFT | 299.813 ms | 298.883 ms | -0.310% |
| mean TPOT | 29.009 ms | 29.057 ms | +0.165% |
| input TGS | 34.312 tok/s | 34.256 tok/s | -0.162% |
| output TGS | 34.312 tok/s | 34.256 tok/s | -0.162% |
| total TGS | 68.623 tok/s | 68.512 tok/s | -0.162% |

All 10 measured requests completed with exactly 2000 output tokens. These
sub-percent changes are noise-level and do not indicate a performance
regression.

Artifacts:

```text
old: /workspace/bench_results/postmerge_c1_2k2k_20260803_110000/result.json
new: /workspace/bench_results/defaults_c1_2k2k_20260803_122000/result.json
```

Throughput uses the client protocol lengths:

```text
input_tgs  = measured_input_tokens / measured_duration
output_tgs = measured_output_tokens / measured_duration
total_tgs  = (measured_input_tokens + measured_output_tokens)
             / measured_duration
```

## Known limits and rollback guidance

- Packed prefill remains off by default. Do not enable FP8 packed piecewise B=2
  until its known NaN/correctness issue is separately fixed.
- `XLLM_PIECEWISE_CAPTURE_FA3=0` restores the 17-graph/16-runner diagnostic
  path. It is useful for profiling, not faster for the validated C1 workload.
- Setting both FA3 variables to zero restores the FA2 prefill/decode paths.
- Setting `ENABLE_GRAPH=0` restores eager execution.
- The Mate library and exact GQA-specific FA3 metadata, prefill, decode, and
  16/32/64 split-combine operators are startup requirements when FA3 is on.

## Validation checklist

- `bash -n run_xllm_musa.sh`: pass
- launcher missing-argument/fail-fast smoke: pass
- Qwen3-8B shared `XLLM_USE_FA3=0` rollback smoke: pass
- GQA6 and GQA8 exact Mate-op preflight: pass
- MUSA incremental rebuild and link: pass
- C2 deterministic semantic correctness: pass
- C1 4-warmup/10-measure 2k+2k benchmark: pass, no regression
- server logs: no FATAL, NaN, multinomial error, or graph abort
- post-test GPU memory: all devices at 0 MiB
