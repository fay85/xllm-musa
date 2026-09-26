#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$project_root"
run_dir=$(realpath -m -- "${1:?provide a new logs/reservation directory}")
case "$run_dir" in
  "$project_root"/logs/*) ;;
  *) echo "Reservation records must stay inside this checkout's logs" >&2; exit 2 ;;
esac
[[ ! -e "$run_dir" ]] || { echo "Reservation directory already exists" >&2; exit 2; }
mkdir -p "$run_dir"
exec > "$run_dir/holder.log" 2>&1
printf '%s\n' "$$" > "$run_dir/parent.pid"
pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for device in 0 1 2 3 4 5 6 7; do
  env MUSA_VISIBLE_DEVICES="$device" RESERVATION_DEVICE="$device" \
    PYTHONDONTWRITEBYTECODE=1 python3 -u -c '
import os
import time
import torch
import torch_musa
from scripts.logger import logger

free_bytes, total_bytes = torch.musa.mem_get_info()
if free_bytes < total_bytes - 1024**3:
    raise SystemExit("GPU is not idle; refusing duplicate/competing reservation")
allocation = torch.zeros((8 * 1024**3 // 2,), dtype=torch.bfloat16, device="musa:0")
torch.musa.synchronize()
logger.info("HC_POST_GOAL_HOLDER device=%s bytes=%d pid=%d",
            os.environ["RESERVATION_DEVICE"], allocation.numel() * allocation.element_size(), os.getpid())
while True:
    time.sleep(30)
' &
  pids+=("$!")
  printf '%s %s\n' "$device" "$!" >> "$run_dir/children.pid"
done
# A failing child releases this reservation group, never an unrelated process.
wait -n "${pids[@]}"
