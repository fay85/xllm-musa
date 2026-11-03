#!/bin/bash
# Runs the VL correctness check (vl_check.py) against the running stack.
set -euo pipefail
source "$(dirname "$0")/env.sh"

python3 -u "$KIT_DIR/vl_check.py" --url "$MASTER_URL" --model "$MODEL_ID" --prompts "$PROMPTS" "$@"
