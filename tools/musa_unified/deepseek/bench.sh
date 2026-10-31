#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
concurrency=${1:?usage: bench.sh 1,16 --url http://CONTROL:24888 --out NEW_DIRECTORY [options]}
shift
model=${DS_MODEL_PATH:-/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt}
exec python3 "$(dirname "$0")/bench.py" --concurrency "$concurrency" \
  --model "${DS_MODEL_ID:-DeepSeek-V4-Flash-0731-FP8-mt}" --tokenizer "$model" "$@"
