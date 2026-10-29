#!/bin/bash
# Stops exactly the processes start_stack.sh recorded for the latest run, or
# for the run directory given as $1: SIGTERM first, SIGKILL after 10 s.
set -u
source /workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench/env.sh

pids_file=${1:-$RUNS_DIR/current}/pids.txt
[[ -f $pids_file ]] || { echo "nothing to stop: no $pids_file"; exit 0; }

# Stop Engine processes while discovery/coordinator services still exist.
# Then stop the front/control plane, and etcd last. Own only recorded PIDs.
stop_group() {
  local pattern=$1
  while read -r name pid; do
    [[ $name =~ $pattern ]] || continue
    kill -0 "$pid" 2>/dev/null && kill "$pid" && echo "TERM $name $pid"
  done < "$pids_file"
  for _ in $(seq 10); do
    local alive=0
    while read -r name pid; do
      [[ $name =~ $pattern ]] || continue
      if kill -0 "$pid" 2>/dev/null && [[ $(ps -o stat= -p "$pid") != Z* ]]; then
        alive=1
      fi
    done < "$pids_file"
    [[ $alive -eq 0 ]] && return 0
    sleep 1
  done
  while read -r name pid; do
    [[ $name =~ $pattern ]] || continue
    if kill -0 "$pid" 2>/dev/null && [[ $(ps -o stat= -p "$pid") != Z* ]]; then
      kill -9 "$pid" && echo "KILL $name $pid"
    fi
  done < "$pids_file"
}
stop_group '^(prefill|decode)'
stop_group '^(master|coordinator)$'
stop_group '^etcd$'
