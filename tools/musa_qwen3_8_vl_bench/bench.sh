#!/bin/bash
# Benchmarks the running stack at one concurrency: master warm-up bursts,
# 8 sequential warm-up requests, 4 x C closed-loop warm-up requests, then 60
# measured requests. Extra arguments go to vl_bench.py (e.g. --out FILE).
# Usage: bench.sh CONCURRENCY [vl_bench.py arguments...]
#
# isl/shared: if you do not pass --isl/--shared on the CLI, this script reads
# them from the prompts JSON (fields "isl" / "shared"). Known filenames are a
# fallback (text20k* -> 22035/16384, *isl40960*/*40k* -> 40960/32768). So
#   export VL_PROMPTS=/root/xllm_vl_bench/vl_prompts_text20k_pm1k.json
#   bash bench.sh 6
# is enough; no need to also pass --isl/--shared. Explicit CLI still wins.
# VL_ISL / VL_SHARED env vars override JSON/filename when CLI is absent.
set -euo pipefail
source "$(dirname "$0")/env.sh"

concurrency=${1:?usage: bench.sh CONCURRENCY [vl_bench.py arguments...]}
shift

has_isl=0
has_shared=0
for a in "$@"; do
  case "$a" in
    --isl|--isl=*) has_isl=1 ;;
    --shared|--shared=*) has_shared=1 ;;
  esac
done

auto_args=()
if [[ $has_isl == 0 || $has_shared == 0 ]]; then
  json_isl=""
  json_shared=""
  if [[ -f "$PROMPTS" ]]; then
    # Prefer metadata embedded in the prompts JSON.
    read -r json_isl json_shared < <(
      python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
isl = d.get("isl", "")
shared = d.get("shared", "")
print("" if isl in (None, "") else isl, "" if shared in (None, "") else shared)
' "$PROMPTS"
    ) || true
  fi
  # Filename map when JSON has no isl/shared (or file not yet built).
  base=$(basename "$PROMPTS")
  case "$base" in
    *text20k*|*20k_pm1k*|*20k_img*)
      [[ -z $json_isl ]] && json_isl=22035
      [[ -z $json_shared ]] && json_shared=16384
      ;;
    *isl40960*|*40k*)
      [[ -z $json_isl ]] && json_isl=40960
      [[ -z $json_shared ]] && json_shared=32768
      ;;
  esac
  # VL_ISL / VL_SHARED override JSON/filename; CLI already handled above.
  [[ -n ${VL_ISL-} ]] && json_isl=$VL_ISL
  [[ -n ${VL_SHARED-} ]] && json_shared=$VL_SHARED

  if [[ $has_isl == 0 && -n $json_isl ]]; then
    auto_args+=(--isl "$json_isl")
  fi
  if [[ $has_shared == 0 && -n $json_shared ]]; then
    auto_args+=(--shared "$json_shared")
  fi
  if [[ ${#auto_args[@]} -gt 0 ]]; then
    echo "bench.sh: auto isl/shared from prompts -> ${auto_args[*]} (PROMPTS=$PROMPTS)" >&2
  fi
fi

python3 -u "$KIT_DIR/vl_bench.py" --url "$MASTER_URL" --model "$MODEL_ID" --tokenizer "$MODEL" \
  --data-dir "$DATA_DIR" --prompts "$PROMPTS" --server-log-dir "$(readlink -f "$RUNS_DIR/current")" \
  --concurrency "$concurrency" --requests 60 --warm 8 --warm-master 2 --warm-closed $((4 * concurrency)) \
  "${auto_args[@]}" "$@"

