# MUSA packed-prefill bucket fix handoff (2026-07-31)

## Scope and authoritative runtime

This handoff records the current Qwen3.5-27B BF16 packed-piecewise prefill
fix.  The authoritative checkout and build are inside the 92 container:

- SSH: `ssh dev92`
- Container: `xllm-musa2.9.1-sdk5.1-dev`
- Source: `/workspace/xllm-git-master`
- Branch: `b2-mate-prefill-fix`
- Binary: `/workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310/xllm/xllm`
- Model: `/workspace/model_weights/Qwen3.5-27B` (BF16)

Build only with:

```bash
ssh dev92 docker exec -e MAX_JOBS=16 -e NINJA_TARGET=xllm \
  -w /workspace/xllm-git-master xllm-musa2.9.1-sdk5.1-dev \
  bash ./_build_cuda_graph_musa.sh
```

The 20:35 build (`build_cuda_graph_musa_20260731_203355.log`) completed
without compiler errors.  The worktree is intentionally dirty from the
larger MUSA refactor; no commit was made for this handoff.

## Changes exercised by the current binary

1. Packed pure-prefill graphs use a stable token bucket plus effective batch
   size as the cache key.  Live CU/host sequence metadata is updated on replay
   instead of forcing a graph capture for every exact prompt length.
2. Piecewise capture owns stable GDN query/key/value/gate/beta/output,
   final-state, KKT, and matmul intermediate buffers.  The direct in-graph
   variable-length Mate experiment is not used because it produced incorrect
   B>1 output.
3. FA3/GDN-only piecewise replay skips the expensive FlashInfer `plan_info`
   update; regular FA2/chunked runners still request it.
4. The token-major causal-conv kernel initializes only the live bucket tail
   that it owns.  This removes the per-layer full-buffer memset while keeping
   packed replay deterministic.  `gdn_output_` tail clearing remains in the
   runner because it is required for downstream GDN state/output safety.
5. The MUSA launcher defaults to graph + piecewise graph + packed prefill,
   `XLLM_USE_FA3=1`, `XLLM_USE_FA3_DECODE=1`, and
   `XLLM_GDN_DECODE_BACKEND=mate`.  `XLLM_MAX_PACKED_PREFILL_SEQS=2` is the
   default scheduler cap, preserving decode capacity while producing the
   reusable C=5 `2+2+1` prefill pattern.
6. The C++ GDN decode fallback is also hardcoded to `mate` when
   `XLLM_GDN_DECODE_BACKEND` is absent; `fused` and `reference` remain explicit
   diagnostic overrides.
7. A short-prompt correctness hole was found during cold-start validation:
   Qwen3.5 B>1 packed piecewise at the 64-token bucket corrupted output.  Only
   Qwen3.5 packed buckets below 128 tokens now take eager prefill; this does
   not affect the production 2k bucket.

## Correctness evidence

All checks used `MAXTOK=256`, so the expected `391` token is present.

- Port 18201, C=1/C=4: standard `correctness_check.sh` PASS.
- Port 18211, C=5: standard check PASS before and after the long replay test;
  all five concurrent responses contained the expected answer and no garbage.
- C=5 packed replay stress: 50/50 requests completed with output length 32,
  no `MultinomialOut`, NaN, or FATAL lines in the server log.

`concurrent_match_golden=False` in the script is informational; the semantic
checks (`concurrent_expected_substring`, `concurrent_not_garbage`) passed.

## Performance measurements

All measurements below use ISL=2000, prefix=2000, graph/piecewise/packed on,
`max_tokens_per_batch=16384`, and output length 16 unless noted.

| Workload | Result JSON | Success | Mean TTFT | Mean TPOT | Throughput |
|---|---|---:|---:|---:|---:|
| C=1, 2 warmup + 5 measure | `.../c5_cap2/c1_warm2_measure5.json` | 5/5 | 385.91 ms | 46.15 ms | 1627.40 TGS |
| C=4, 2 warmup + 5 measure | `.../c5_cap2/c4_warm2_measure5_latest.json` | 20/20 | 1065.91 ms | 73.41 ms | 3462.66 TGS |
| C=5, 2 warmup + 5 measure | `.../c5_cap2/c5_warm2_measure5.json` | 25/25 | 1216.46 ms | 89.93 ms | 3694.95 TGS |
| C=5, 10 replay waves, OSL=32 | `.../c5_cap2/c5_replay10_output32.json` | 50/50 | 1213.30 ms | 69.90 ms | 2851.81 TGS |
| C=5, default Mate (no GDN env), 2 warmup + 5 measure | `.../default_mate/c5_2k_warm2_measure5.json` | 25/25 | 1209.82 ms | 89.52 ms | 3713.10 TGS |
| C=1, default Mate (no GDN env), 2 warmup + 5 measure | `.../default_mate/c1_2k_warm2_measure5.json` | 5/5 | 384.52 ms | 45.77 ms | 1636.80 TGS |

The full paths are under:
`/workspace/bench_results/prefill_bucket_fix_20260731/c5_cap2/`.

The C=5 capture log shows exactly one packed B=2 capture (`actual=4106`,
`bucket=4352`, 65 graph segments/64 runners), one B=1 capture
(`actual=2053`, `bucket=2304`, 33 segments/16 runners), and one decode B=5
capture.  Subsequent waves replay those graphs without additional prefill
captures.  The staged client `2+2+1` test also completed 25/25; its higher
1.32 s mean TTFT includes the intentional 10 ms inter-group release delay and
is not the simultaneous-arrival comparison.

The final 21:28 build (`build_cuda_graph_musa_20260731_212833.log`) has no
compiler errors.  (The preceding 21:16 build was the first functional build
of the small-bucket guard.)  A no-env direct-binary run confirmed
`XLLM_GDN_DECODE_BACKEND` was absent from `/proc/<pid>/environ`, yet C=5
correctness still passed after the small-bucket guard; this validates the C++
Mate default independently of the launcher.

For diagnosis, the unguarded new binary was intentionally tested at the
64-token packed bucket and failed 4/5 semantic checks (`!!!!` output) while
the same binary with `XLLM_PACKED_PREFILL_PIECEWISE=0` passed.  After the
`<128` eager guard, the same short C=5 check passed, and the 2k benchmark still
captured `actual=4114 -> bucket=4352` with the normal 65/64 packed graph.

## Next safe steps

1. Keep the current tail-clear and cap=2 implementation as the correctness
   baseline; do not reintroduce the direct in-graph varlen Mate path.
2. If further optimization is needed, profile the 33/16 and 65/64 replay
   segments with `XLLM_PREFILL_BREAKDOWN=1` or Kineto, then A/B one change at a
   time.  Do not remove GDN output-tail initialization without a long C=5
   replay and correctness check.
3. Before committing, stage only the intended MUSA bucket files; the checkout
   contains unrelated dirty refactor files and ignored launcher scripts.
