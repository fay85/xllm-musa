#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-auto}
if [[ $mode == standalone ]]; then exec bash start_service.sh stop; fi
[[ $mode == auto || $mode == pd ]] || { echo "Usage: $0 [pd|standalone]" >&2; exit 2; }
found=0
for role in decode prefill; do
  dir=pd_logs/$role
  generation=$(cat "$dir/generation" 2>/dev/null || :)
  if [[ -z $generation ]]; then
    if [[ -s $dir/pids.txt ]]; then
      echo "Legacy $role instance: run start_${role}.sh stop on its own host." >&2
      exit 1
    fi
    continue
  fi
  [[ $generation =~ ^[[:xdigit:]-]{36}$ ]] || exit 1
  found=1
  echo "Stopping $role ($generation)"
  touch "$dir/stop.$generation"
  for ((i=0;i<120;i++)); do
    [[ -f $dir/stopped.$generation ]] && break
    sleep 1
  done
  [[ -f $dir/stopped.$generation ]] || { echo "No $role acknowledgement. See $dir/stop-watch.log" >&2; exit 1; }
  echo "$role stopped"
done
if [[ $found == 0 && $mode == auto ]]; then exec bash start_service.sh stop; fi
