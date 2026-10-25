#!/bin/bash
# Stops exactly the processes start_stack.sh recorded for the latest run, or
# for the run directory given as $1: SIGTERM first, SIGKILL after 10 s.
set -u
source "$(dirname "$0")/env.sh"

pids_file=${1:-$RUNS_DIR/current}/pids.txt
[[ -f $pids_file ]] || { echo "nothing to stop: no $pids_file"; exit 0; }

while read -r name pid; do
  kill -0 "$pid" 2>/dev/null && kill "$pid" && echo "TERM $name $pid"
done < "$pids_file"
sleep 10
while read -r name pid; do
  if kill -0 "$pid" 2>/dev/null && [[ $(ps -o stat= -p "$pid") != Z* ]]; then
    kill -9 "$pid" && echo "KILL $name $pid"
  fi
done < "$pids_file"
