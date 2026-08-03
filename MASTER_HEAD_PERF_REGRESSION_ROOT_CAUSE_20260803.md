# master-HEAD performance regression root cause (2026-08-03)

## Outcome

The large merge regression is fixed. The historical 700 ms memory was correct:

- historical strict-cap1 eager: 745.05 ms mean TTFT;
- historical strict-cap1 piecewise: 731.47 ms mean TTFT.

A same-time, same-GPU4 binary swap proved that the integration regression was
real rather than machine drift:

| Binary | Eager TTFT | Piecewise TTFT |
|---|---:|---:|
| source e26 | 735.66 ms | 750.05 ms |
| master-HEAD integration before fixes | 892.07 ms | 857.69 ms |

The clean repaired binary now reaches:

| GPU | Mode | Mean TTFT | Mean prefill forward |
|---|---|---:|---:|
| GPU4 | eager | 733.07 ms | 228.79 ms |
| GPU4 | split piecewise | 743.15 ms | 233.11 ms |
| GPU3, committed state | eager | 721.14 ms | 225.55 ms |
| GPU3, committed state | split piecewise | 727.50 ms | 227.98 ms |

Therefore the merge-induced eager regression is closed. The current
piecewise-less-than-eager gate is not closed: piecewise is 6.36 ms slower on
GPU3 and 10.08 ms slower on GPU4 in these same-time runs.

## Causal findings

### 1. Scheduler cap semantics were lost during scheduler unification

The integrated scheduler did not retain the local
`XLLM_MAX_PREFILL_SEQS` admission behavior. At external C=5 this allowed
multiple prefills into one forward and invalidated comparisons against the
strict-cap1 reference.

Restoring the cap reduced the cap2 graph run from 1215.15 ms to 1003.82 ms.
This is an integration conflict around the scheduler-unification lineage, not
evidence that every change in that upstream commit is bad.

### 2. FP8 host synchronization optimization was not ported

The MUSA block-FP8 GEMM path retained two host stream synchronizations per
GEMM. A prefill executes about 256 such GEMMs.

Porting the asynchronous default from
`3e05b26a perf(musa): snapshot prefill and eager optimizations` reduced eager
TTFT from 892.07 ms to 843.75 ms. The rollback is
`XLLM_FP8_FORCE_HOST_SYNC=1`.

### 3. The decisive remaining regression was missing fused MUSA SwiGLU

The new master activation path had not retained the MUSA-specific
`at::swish_glu` dispatch from:

`ec37b9de perf: cut qwen3.5 prefill ttft with swish_glu and dense fa3`.

Component timing isolated the complete remaining forward gap:

| Component | source e26 | integration before SwiGLU fix |
|---|---:|---:|
| full attention | 23.33 ms | 23.22 ms |
| GDN | 74.69 ms | 73.61 ms |
| MLP gate/up | 78.77 ms | 78.85 ms |
| MLP down | 38.00 ms | 37.63 ms |
| MLP activation | 9.42 ms | 39.20 ms |

Restoring `at::swish_glu` reduced eager TTFT from 815.56 ms to 733.07 ms
and forward from 256.95 ms to 228.79 ms. This closes the same-time gap to the
source binary (735.66 ms / 229.64 ms).

### 4. A block-FP8 output-buffer port had an ordering bug

During adaptation to the new `linear.cpp` ABI, `params.output` was assigned
after `fp8_block_matmul(params)`, so the preallocated output buffer was never
passed to the operation. Moving the assignment before the call reduced eager
TTFT from 842.27 ms to 815.56 ms in the controlled run. This was an
uncommitted integration error, not an upstream commit regression.

### 5. The GDN closure mainly helps piecewise

Porting the stride-aware/partial-KKT GDN closure reduced piecewise TTFT from
861.47 ms to 831.09 ms before the SwiGLU fix, while eager remained effectively
unchanged (843.75 ms versus 842.27 ms). This confirms that the GDN work was
needed but was not the common eager regression.

## Piecewise investigation status

The current split replay is real piecewise execution: each bucket contains
17 graphs and 16 FA3 runners. A diagnostic full-FA3 path reduced this to one
graph and zero runners, but did not recover enough performance and complicated
the correctness boundary, so the diagnostic change was reverted.

Zero-padding exact2048 tests show that the remaining direction is not caused
only by bucket padding:

| Binary | Full piecewise forward | Eager forward |
|---|---:|---:|
| repaired master-HEAD | 219.32 ms | 216.99 ms |
| source e26, same time/GPU | 220.76 ms | 218.88 ms |

Because the unchanged source binary shows the same approximately 2 ms graph
penalty in the current time window, this residual is not attributable to the
master-HEAD merge. It is a current MUSA graph replay/runtime behavior and
requires a separate device/runtime-level investigation. No claim that
piecewise is currently faster than eager is made.

## Build and artifacts

- branch: `codex/dev-optimizations-sync-20260802`
- build invariant: `USE_MUSA=ON`, `USE_CUDA=OFF`
- committed-state binary:
  `/workspace/bench_results/dev_to_master_head_sync_20260802/binaries/post_commit_ca0d17d3/xllm`
- SHA256:
  `7c649f6cc27434ac217aadcb47e78aa6875c9827a403aa2b15cc15273a0ea794`
- build log:
  `build_logs/build_cuda_graph_musa_20260803_021609.log`
- GPU4 eager:
  `/workspace/bench_results/dev_to_master_head_sync_20260802/current_gpu4_cap1/eager_swish_glu_fixed`
- GPU4 piecewise:
  `/workspace/bench_results/dev_to_master_head_sync_20260802/current_gpu4_cap1/piecewise_swish_glu_fixed`
- committed-state GPU3 eager/piecewise:
  `/workspace/bench_results/dev_to_master_head_sync_20260802/post_commit_ca0d17d3_gpu3`
