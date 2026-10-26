#!/bin/bash
# One-time preparation on the serving machine:
#   1. download the news articles and photos into $DATA_DIR (needs internet;
#      on an offline machine run fetch_data.py elsewhere and copy the result),
#   2. write the tokenizer directory the master loads,
#   3. build the 68 prompts (8 warm-up + 60 measured) into $PROMPTS.
set -euo pipefail
source "$(dirname "$0")/env.sh"

python3 "$KIT_DIR/fetch_data.py" --out "$DATA_DIR"
python3 "$KIT_DIR/master_tokenizer.py" --model "$MODEL" --out "$MASTER_TOKENIZER"
python3 "$KIT_DIR/vl_bench.py" --tokenizer "$MODEL" --data-dir "$DATA_DIR" --prompts "$PROMPTS" \
  --warm 8 --requests 60 --dry-run
