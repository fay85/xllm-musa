#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
exec python3 -u profile_pd.py --url "${GLM5_URL:-http://10.121.34.85:28888}" \
  --model "$MODEL_ID" --tokenizer "$MODEL" --data-dir "$DATA_DIR" \
  --prompts "$PROMPTS" --isl "$ISL" --osl "${GLM5_OSL:-16}" --num-images "$NUM_IMAGES" "$@"
