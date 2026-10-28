#!/bin/bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-/workspace/decode_tp2_dev_artifacts/bin/xllm.native_tp2_pf_tp_slices}
run=${RUN:-/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/sweep_fp8_tp2_repro_$(date +%Y%m%d_%H%M%S)}
wl=/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv
sha256sum "$bin"
python3 "$here/launch_pd_comparison.py" start \
  --topology tp2 \
  --run "$run" \
  --prefill-bin "$bin" \
  --decode-bin "$bin" \
  --model /data/nfs_shared/models/Qwen3.8-27B-FP8 \
  --model-id Qwen3.8-27B-FP8 \
  --staging-mib 512 \
  --pcp-custom-ar 1 || { echo LAUNCH_FAILED; exit 1; }
for c in 6 8 10 12 14 16 18; do
  echo "=== C$c $(date +%T)"
  python3 "$here/bench_until_steady.py" \
    --dataset "$wl/workload_c$c.json" \
    --concurrency "$c" \
    --server-log-dir "$run" \
    --out "$run/bench" \
    --model-id Qwen3.8-27B-FP8 \
    --topology fp8_pfcp2_tp2 || echo "C$c FAILED"
done
echo "SWEEP_DONE $(date +%T)"
python3 "$here/launch_pd_comparison.py" stop --run "$run"
echo STOPPED
