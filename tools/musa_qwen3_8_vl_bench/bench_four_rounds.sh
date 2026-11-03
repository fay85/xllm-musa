#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ $# -eq 1 ]] || { echo "usage: $0 CONCURRENCY" >&2; exit 2; }
[[ -f $FOUR_ROUNDS_DATASET && -f $FOUR_ROUNDS_RUNNER ]] || {
  echo "missing frozen four-round dataset/runner; install the release bundle" >&2
  exit 1
}
RUN=$(readlink -f "$RUNS_DIR/current")
[[ -f $RUN/stack_info.txt ]] || { echo "start the canonical stack first" >&2; exit 1; }
mkdir -p "$RESULTS_DIR"
OUT=$RESULTS_DIR/${PROFILE}_c${1}_$(date +%Y%m%d_%H%M%S).json
exec python3 "$FOUR_ROUNDS_RUNNER" --dataset "$FOUR_ROUNDS_DATASET" \
  --concurrency "$1" --server-log-dir "$RUN" --out "$OUT" \
  --url "$MASTER_URL" --profile "$PROFILE"
