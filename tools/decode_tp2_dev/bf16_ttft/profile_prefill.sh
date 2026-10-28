#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dst=${BIN:-/workspace/decode_tp2_dev_artifacts/bin/xllm.native_tp2_pf_tp_slices}
run=${RUN:-/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/c6_bf16_prof_$(date +%Y%m%d_%H%M%S)}
trap 'python3 "$here/launch_pd_comparison.py" stop --run "$run" || true' EXIT
python3 "$here/launch_pd_comparison.py" start \
  --topology tp1 \
  --run "$run" \
  --prefill-bin "$dst" \
  --decode-bin "$dst" \
  --model /data/nfs_shared/models/Qwen3.8-27B \
  --model-id Qwen3.8-27B \
  --memory 0.90 \
  --prefill-cp-size 1 \
  --prefill-state-cache-slots 96 \
  --decode-state-cache-slots 4 \
  --staging-mib 512 \
  --pcp-custom-ar 1 \
  --online-profile-dir "$run/prof"
python3 "$here/bench_c6_until_steady.py" \
  --server-log-dir "$run" \
  --out "$run/bench" \
  --model-id Qwen3.8-27B \
  --topology pftp2_prof_warmup || true
python3 "$here/profile_one_request.py"
ls -la "$run/prof" || true
python3 "$here/parse_kineto_trace.py" "$run/prof" || true
echo PROFILE_DONE
