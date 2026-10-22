#!/usr/bin/env bash
set -euo pipefail

script_dir=${DSPARK_REPRO_SCRIPTS:-/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark/logs/dspark-repro-scripts.5vTKvK}
run_dir=${1:?Supply the service DSPARK_RUN directory}
first=${2:-0}
last=${3:-7}
[[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || exit 2
if ! ((first == 0 && last == 7)) && ! ((first >= 2 && last - first == 5)); then
  echo "Use 0 7 for a fresh run, or six consecutive variants for a declared repeat." >&2
  exit 2
fi
[[ -f "$run_dir/launch_manifest.txt" ]] || {
  echo "Missing launch_manifest.txt: use the matching service directory." >&2
  exit 2
}
url=${DSPARK_URL:-http://127.0.0.1:${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}/v1/chat/completions}
model_path=${MODEL_PATH:-/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt}
curl -fsS --max-time 5 "${url%/v1/chat/completions}/health"

run_case() {
  local name=$1
  shift
  local json="$run_dir/$name.json"
  local log="$run_dir/$name.client.log"
  if [[ -e "$json" || -e "$log" ]]; then
    echo "Refusing to overwrite $name. Use a fresh run directory." >&2
    exit 2
  fi
  local status=0
  timeout --signal=TERM --kill-after=10s 180 \
    python3 "$script_dir/benchmark_dspark.py" --model-path "$model_path" \
      --url "$url" --output "$json" "$@" >"$log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$run_dir/$name.exit_code"
  if ((status != 0)); then
    echo "$name failed (exit=$status); retain $log and rank logs." >&2
    exit "$status"
  fi
  python3 "$script_dir/check_result.py" "$json" >"$run_dir/$name.accuracy_gate.json"
}

if ((first == 0)); then
  for warmup in 1 2; do
    run_case "warmup_$warmup" --prompt-tokens 64 --output-tokens 64 --count-to 32
  done
fi
for ((variant=first; variant<=last; ++variant)); do
  run_case "formal_variant_$variant" --prompt-tokens 2500 --output-tokens 1500 \
    --count-to 750 --prefix-cache-rate 0.65 --prompt-variant "$variant"
done
hot_first=$first
if ((first == 0)); then hot_first=2; fi
python3 "$script_dir/summarize_window.py" "$run_dir" "$hot_first" "$last" \
  | tee "$run_dir/window_${hot_first}_${last}.summary.json"
