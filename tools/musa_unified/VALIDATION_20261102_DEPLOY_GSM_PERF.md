# 9n deployment simulation and DeepSeek validation (2026-11-02)

## Deployment-tool simulation

Used the tracked `9n-deploy-tools/ea-xy/xllm/musa/distributed/simulate_2p1d.sh`
with `2p1d_current_cp1.json` and a fresh run root:

```text
/data/nfs_shared/jd/musa_2p1d_deploy_sim_20261102/
```

The three role artifact checks passed:

```text
MUSA PD artifacts verified: d90
MUSA PD artifacts verified: p94
MUSA PD artifacts verified: p96
```

The generated plans were saved as `d90.plan.json`, `p94.plan.json`, and
`p96.plan.json`. This is a deployment/init/plan simulation only; it did not
start the 90/94/96 production topology.

## 90 host condition

90 GPU3 remains unsuitable for model acceptance because of the previously
reproduced standalone device-memory read/write corruption. The 90 GPU table
was idle during this run, but no reset was performed and no 2P1D model service
was started there.

## 1P1D full GSM8K

Because 90 was excluded, the fixed Native V2 Service experiment used 94
Prefill + 96 Decode, with Qwen Service Router/Coordinator and the fixed Native
V2 cache-group descriptor. The full frozen `dataset_test.jsonl` contains 1319
questions.

| Mode | Result | Notes |
|---|---:|---|
| concurrency 1 | **1277/1319 = 96.82%** | No HTTP errors; complete responses and numeric answer scoring |
| concurrency 16 | **920/1319 = 69.75%** | Many requests ended after `We`; this is a batch/Graph/state-reuse failure mode, not a valid production accuracy result |

Evidence:

```text
/data/nfs_shared/jd/unified_toolkit_20261031/smoke94_final/
  ds_qwen_service_probe_control4/gsm1319_c1/summary.json
  ds_qwen_service_probe_control4/gsm1319_c1/results.jsonl
  ds_qwen_service_probe_control4/gsm1319/summary.json
  ds_qwen_service_probe_control4/gsm1319/results.jsonl
```

The concurrency-1 failures were retained in `failed_indices`; the score is
close to the historical 96% baseline. The concurrency-16 result must not be
used as the model accuracy score until the existing batch/Graph truncation is
fixed.

## 1P1D strict performance

The same 94/96 1P1D service was tested with exact `20K/100`,
`cached_tokens=16000` (80% prefix), strict `1..50` output, and a fresh run
directory.

| Concurrency | Strict gate | TTFT mean | TPOT mean | Output throughput |
|---:|---|---:|---:|---:|
| B1 | passed | **489.67 ms** | **9.11 ms** | **71.68 tok/s** |
| B16 | passed | **3898.26 ms** | **11.23 ms** | **191.32 tok/s** |

The B16 warm stage also passed all 32 requests with exact cache accounting;
its mean TPOT was `15.73 ms` under the warm-up/capture stage, while the formal
B16 stage was `11.23 ms`.

Evidence:

```text
/data/nfs_shared/jd/unified_toolkit_20261031/smoke94_final/
  ds_qwen_service_probe_control4/perf20k_b1/summary.json
  ds_qwen_service_probe_control4/perf20k_b16/summary.json
```

These are **1P1D** numbers, not the unavailable healthy three-host 2P1D
numbers. A full 2P1D run remains blocked by the 90 GPU3 hardware condition.
