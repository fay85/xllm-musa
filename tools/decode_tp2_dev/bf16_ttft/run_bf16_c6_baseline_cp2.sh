#!/bin/bash
# BF16 C6 baseline: prefill DP3 CP2 with 24 linear-state slots, two TP1 decodes with 8.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-/workspace/decode_tp2_dev_artifacts/bin/xllm.native_tp2_pf_tp_slices}
run=${RUN:-/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/c6_bf16_cp2_repro_$(date +%Y%m%d_%H%M%S)}
trap 'python3 "$here/launch_pd_comparison.py" stop --run "$run" || true' EXIT
python3 "$here/launch_pd_comparison.py" start --topology tp1 --run "$run" \
  --prefill-bin "$bin" --decode-bin "$bin" \
  --model /data/nfs_shared/models/Qwen3.8-27B --model-id Qwen3.8-27B \
  --memory 0.90 --prefill-cp-size 2 --prefill-state-cache-slots 24 \
  --decode-state-cache-slots 8 --staging-mib 512 --pcp-custom-ar 1
python3 "$here/bench_c6_until_steady.py" --server-log-dir "$run" --out "$run/bench" \
  --model-id Qwen3.8-27B --topology bf16_pfcp2_two_tp1
