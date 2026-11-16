#!/usr/bin/env bash
# Benchmarks the running GLM-5.3-Flash VLM service for 2k input / 1 output.
# Extra arguments go to vl_bench.py (e.g. --requests 16 --out FILE).
# Usage: bench.sh [CONCURRENCY] [vl_bench.py arguments...]
# Images are disabled by default; use --num-images N to include them.
#
# isl/shared/osl: if you do not pass --isl/--shared/--osl on the CLI, this
# script reads isl from the prompts JSON or infers it from the filename.
# Shared prefixes are disabled by default (shared=0); osl defaults to 1.
# Override with GLM5_ISL / GLM5_SHARED / GLM5_OSL or their VL_ equivalents.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ $# -gt 0 && "$1" != -* ]]; then
  concurrency=$1
  shift
else
  concurrency=${CONC:-1}
fi
requests=${REQUESTS:-$concurrency}

has_isl=0
has_shared=0
has_osl=0
has_num_images=0
for a in "$@"; do
  case "$a" in
    --isl|--isl=*) has_isl=1 ;;
    --shared|--shared=*) has_shared=1 ;;
    --osl|--osl=*) has_osl=1 ;;
    --num-images|--num-images=*) has_num_images=1 ;;
  esac
done

auto_args=()
if [[ $has_isl == 0 ]]; then
  json_isl=""
  if [[ -f "$PROMPTS" ]]; then
    read -r json_isl < <(
      python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
isl = d.get("isl", "")
print("" if isl in (None, "") else isl)
' "$PROMPTS"
    ) || true
  fi
  base=$(basename "$PROMPTS")
  case "$base" in
    *isl2048*|*2k*)
      [[ -z $json_isl ]] && json_isl=2048
      ;;
    *text20k*|*20k_pm1k*|*20k_img*)
      [[ -z $json_isl ]] && json_isl=22035
      ;;
    *isl40960*|*40k*)
      [[ -z $json_isl ]] && json_isl=40960
      ;;
    *)
      [[ -z $json_isl ]] && json_isl=${ISL:-2048}
      ;;
  esac
  [[ -n ${VL_ISL-} ]] && json_isl=$VL_ISL
  [[ -n ${GLM5_ISL-} ]] && json_isl=$GLM5_ISL

  if [[ $has_isl == 0 && -n $json_isl ]]; then
    auto_args+=(--isl "$json_isl")
  fi
fi

if [[ $has_shared == 0 ]]; then
  auto_args+=(--shared "${SHARED:-0}")
fi
if [[ $has_osl == 0 ]]; then
  auto_args+=(--osl "${OSL:-1}")
fi
if [[ $has_num_images == 0 ]]; then
  auto_args+=(--num-images "${NUM_IMAGES:-0}")
fi

if [[ ${#auto_args[@]} -gt 0 ]]; then
  echo "bench.sh: auto args -> ${auto_args[*]} (PROMPTS=$PROMPTS)" >&2
fi

python3 -u "$KIT_DIR/vl_bench.py" --url "$MASTER_URL" --model "$MODEL_ID" --tokenizer "$MODEL" \
  --data-dir "$DATA_DIR" --prompts "$PROMPTS" --server-log-dir "$LOG_DIR" \
  --concurrency "$concurrency" --requests "$requests" --warm "$WARM" \
  --warm-master "$WARM_MASTER" --warm-closed "$WARM_CLOSED" \
  "${auto_args[@]}" "$@"
