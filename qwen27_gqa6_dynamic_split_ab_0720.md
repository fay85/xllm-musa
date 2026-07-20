# Qwen3.5-27B GQA6 dynamic-split A/B (2026-07-20)

## Conclusion

GQA6 FA3 decode already had a dynamic split implementation in the cached Mate
metadata kernel. xLLM disabled it by passing `num_splits=1` to the older
four-output GQA6 metadata ABI, while the forward kernel received
`num_splits=0`. The ABI itself was not the blocker.

The final policy is adaptive:

- batch size <= 2: use Mate dynamic split;
- batch size > 2 and KV length < 8192: use one split;
- batch size > 2 and KV length >= 8192: use Mate dynamic split;
- `XLLM_FA3_GQA6_DYNAMIC_SPLIT=0`: force the original one-split path.

This keeps the large C=1 long-context win and avoids enabling extra split
reduction in the short-context C=5 workload, where the full server A/B did not
show an end-to-end gain.

## Implementation evidence

- SGLang's MUSA FA3 path passes `num_splits=0` to both scheduler metadata and
  forward by default.
- The generated GQA6 metadata source contains Mate's complete
  `get_num_splits` heuristic and writes the four tensors used by xLLM:
  `num_splits_dynamic`, `batch_table`, `num_m_blocks`, and
  `num_nheads_in_l2`.
- For B=1, KV=20480, the GQA6 metadata changes from `[1, 0, 1, 1]` to
  `[13, 0, 1, 1]` when `num_splits` changes from 1 to 0.
- Fixed and dynamic outputs at that shape have max absolute difference
  `0.000244140625` and mean absolute difference `1.861e-05`, consistent with
  BF16 reduction-order noise.

## Isolated FA3 decode sweep

Shape: Q heads=24, KV heads=4, head dim=256, page size=64, BF16. Times are
median synchronized forward times; scheduler metadata generation is outside
the timed forward, matching graph replay.

| Batch | KV length | Fixed split=1 (ms) | Dynamic split | Dynamic (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,500 | 0.1847 | 13 | 0.1516 | 1.22x |
| 1 | 4,096 | 0.2112 | 13 | 0.1568 | 1.35x |
| 1 | 8,192 | 0.2795 | 13 | 0.1665 | 1.68x |
| 1 | 12,288 | 0.3532 | 13 | 0.1738 | 2.03x |
| 1 | 20,480 | 0.5003 | 13 | 0.1927 | 2.60x |
| 5 | 2,500 | 0.1854 | 3 | 0.1663 | 1.12x |
| 5 | 4,096 | 0.2109 | 3 | 0.1846 | 1.14x |
| 5 | 8,192 | 0.2745 | 3 | 0.2568 | 1.07x |
| 5 | 20,480 | 0.4691 | 3 | 0.4154 | 1.13x |

Qwen3.5-27B has 16 full-attention layers. At C=1/KV=20k, the isolated
per-layer saving predicts about 4.9 ms per decode step, which agrees with the
server TPOT result below.

## C=1 long-context server A/B

Configuration: Qwen3.5-27B BF16, graph mode on, ISL=20k, OSL=300, C=1,
prefix cache off, 4 warmup requests and 10 measured requests. Both arms use
the same rebuilt binary; only `XLLM_FA3_GQA6_DYNAMIC_SPLIT` changes.

| Metric | Fixed split=1 | Dynamic | Change |
|---|---:|---:|---:|
| Successful requests | 10/10 | 10/10 | neutral |
| Mean TTFT | 3458.75 ms | 3465.45 ms | +0.19% |
| Mean TPOT | 49.99 ms | 45.07 ms | **-9.84%** |
| Mean latency | 18406.30 ms | 16938.22 ms | **-7.98%** |
| Total TGS (prefill + decode) | 1073.68 tok/s | 1164.05 tok/s | **+8.42%** |

Artifacts:

- fixed: `/workspace/bench_results/qwen27_bf16_gqa6_split1_c1_isl20k_osl300_4w10m_20260720`
- dynamic: `/workspace/bench_results/qwen27_bf16_gqa6_dynamic_c1_isl20k_osl300_4w10m_20260720`

TTFT is neutral, as expected for a decode-only change. The TPOT improvement
matches the kernel-level prediction.

## C=5 short-context server A/B

Configuration: Qwen3.5-27B BF16, graph mode on, ISL=2.5k, OSL=1.5k, strict
closed-loop C=5 barrier waves, packed-prefill cap=2, prefix cache off, 4 warmup
waves (20 requests) and 10 measured waves (50 requests).

| Metric | Fixed split=1 | Dynamic at all lengths | Change |
|---|---:|---:|---:|
| Successful requests | 50/50 | 50/50 | neutral |
| Mean TTFT | 1440.69 ms | 1434.90 ms | -0.40% |
| Mean TPOT | 51.04 ms | 51.28 ms | +0.47% |
| Mean latency | 77797.52 ms | 78073.88 ms | +0.36% |
| Total TGS (prefill + decode) | 255.24 tok/s | 254.25 tok/s | -0.39% |

Artifacts:

- fixed: `/workspace/bench_results/qwen27_bf16_gqa6_split1_c5_barrier_4w10m_20260720`
- dynamic: `/workspace/bench_results/qwen27_bf16_gqa6_dynamic_c5_barrier_4w10m_20260720`

The small isolated B=5 kernel win did not survive at server level. The final
adaptive policy therefore selects exactly the fixed metadata path for this
workload because B=5 and KV stays below 8192. A post-rebuild graph smoke using
the adaptive default completed 5/5 requests with no server errors:
`/workspace/bench_results/qwen27_bf16_gqa6_adaptive_c5_smoke_20260720`.

## Build and correctness

- Rebuilt inside `xllm-musa2.9.1-sdk5.1-dev` using
  `_build_cuda_graph_musa.sh` and `mcc_wrapper`.
- Graph-mode deterministic correctness check passed all HTTP, non-garbage,
  token-count, and expected-answer checks (`391`).
- Dynamic B=1 long-context completed 10/10 requests.
- Dynamic-at-all-lengths B=5 completed 50/50 requests.
- Final adaptive B=5 graph smoke completed 5/5 requests.

During the final clean rebuild, the newly cherry-picked KV-cache estimator
referenced MLU RDMA-only types that are not available in the current MUSA
tree. The MLU RDMA allocation code is now compiled only under `USE_MLU`;
MUSA keeps the existing generic capacity and linear-state calculation and
does not compile or link the MLU RDMA file.
