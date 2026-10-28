#!/bin/bash
# Runs every concurrency level on a freshly started stack and prints one
# table for all levels. Results and logs go to $RESULTS_DIR/sweep_<time>.
# Usage: run_sweep.sh [CONCURRENCY ...]   (default: 5 6 12)
# For 20k: export VL_PROMPTS=/root/xllm_vl_bench/vl_prompts_text20k_pm1k.json
# then run_sweep; bench.sh auto-passes --isl/--shared from the JSON.
set -u
source "$(dirname "$0")/env.sh"

levels=${*:-5 6 12}
out=$RESULTS_DIR/sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$out"
for c in $levels; do
  # An instance occasionally times out during engine init; one retry covers it.
  started=0
  for attempt in 1 2; do
    bash "$KIT_DIR/stop_stack.sh" > /dev/null
    sleep 5
    if bash "$KIT_DIR/start_stack.sh" > "$out/start_c${c}_$attempt.log" 2>&1; then
      started=1
      break
    fi
  done
  if [[ $started == 0 ]]; then
    echo "C=$c: the stack did not start, see $out/start_c${c}_*.log"
    continue
  fi
  # Instances keep registering with the master for a while after READY.
  sleep 45
  echo "C=$c: $(readlink -f "$RUNS_DIR/current")"
  bash "$KIT_DIR/bench.sh" "$c" --out "$out/c$c.json" > "$out/c$c.log" 2>&1 ||
    echo "C=$c: the benchmark failed, see $out/c$c.log"
done
bash "$KIT_DIR/stop_stack.sh" > /dev/null
echo "results: $out"
python3 "$KIT_DIR/report.py" "$out"/c*.json
