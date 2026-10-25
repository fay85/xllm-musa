#!/bin/bash
# Benchmarks the running stack at one concurrency: master warm-up bursts,
# 8 sequential warm-up requests, 4 x C closed-loop warm-up requests, then 60
# measured requests. Extra arguments go to vl_bench.py (e.g. --out FILE).
# Usage: bench.sh CONCURRENCY [vl_bench.py arguments...]
set -euo pipefail
source "$(dirname "$0")/env.sh"

concurrency=${1:?usage: bench.sh CONCURRENCY [vl_bench.py arguments...]}
shift
python3 -u "$KIT_DIR/vl_bench.py" --url "$MASTER_URL" --model "$MODEL_ID" --tokenizer "$MODEL" \
  --data-dir "$DATA_DIR" --prompts "$PROMPTS" --server-log-dir "$(readlink -f "$RUNS_DIR/current")" \
  --concurrency "$concurrency" --requests 60 --warm 8 --warm-master 2 --warm-closed $((4 * concurrency)) "$@"
