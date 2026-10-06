#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
run=${1:?usage: stop_role.sh EXACT_ROLE_RUN_DIR}
[[ "$run" == /* && -f "$run/pids.tsv" && -f "$run/launch_manifest.txt" ]]
role=$(sed -n 's/^role=//p' "$run/launch_manifest.txt")
case "$role" in
  PREFILL) [[ "$(hostname -s)" == worker38094 ]] ;;
  DECODE) [[ "$(hostname -s)" == worker38090 ]] ;;
  *) exit 2 ;;
esac
expected=$(sed -n 's/^xllm_bin=//p' "$run/launch_manifest.txt")
[[ -n "$expected" && "$expected" == /* ]]
[[ "$(wc -l < "$run/pids.tsv")" == 8 ]]
# Validate every live rank before sending even the first signal.
while IFS=$'\t' read -r rank pid executable; do
  [[ "$pid" =~ ^[0-9]+$ && "$rank" =~ ^[0-7]$ && "$executable" == "$expected" ]]
  [[ -e "/proc/$pid" ]] || continue
  [[ "$(readlink -f "/proc/$pid/exe")" == "$expected" ]]
  tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "--instance_role=$role"
done < "$run/pids.tsv"
while IFS=$'\t' read -r rank pid executable; do
  [[ "$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)" == "$expected" ]] || continue
  printf 'TERM role=%s rank=%s pid=%s at=%s\n' "$role" "$rank" "$pid" "$(date -Is)" >> "$run/operator_stop.log"
  kill -TERM "$pid"
done < "$run/pids.tsv"
echo 'Sent TERM only to matching ranks. Check GPU/ports before restarting; no forced KILL or reset is performed.'
