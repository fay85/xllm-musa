#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
role=$1
action=${2:-start}
case "$role:$action" in
  PREFILL:watch|DECODE:watch|PREFILL:start|PREFILL:stop|PREFILL:status|DECODE:start|DECODE:stop|DECODE:status) ;;
  *) echo "Usage: start_{prefill,decode}.sh [start|stop|status]" >&2; exit 2 ;;
esac
# Keep concurrent starts/stops on this host from replacing each other's PID list.
if [[ ${GLM5_ROLE_LOCKED:-} != "$role" ]]; then
  exec flock --nonblock --close "/tmp/glm53-pd-$role.lock" env GLM5_ROLE_LOCKED="$role" bash "$PWD/pd_role.sh" "$role" "$action"
fi
export GLM5_INSTANCE_ROLE=$role GLM5_ENABLE_DISAGG_PD=true GLM5_KV_CACHE_TRANSFER_MODE=PULL
export GLM5_ETCD_ADDR=10.121.34.85:22379 GLM5_NNODES=8
export XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS=${XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS:-90}
export GLM5_WAIT_TIMEOUT_SEC=${GLM5_WAIT_TIMEOUT_SEC:-600}
if [[ $role == PREFILL ]]; then
  export GLM5_ENABLE_GRAPH="${GLM5_ENABLE_GRAPH:-${GLM5_PREFILL_GRAPH:-true}}"
  export GLM5_MAX_SEQS_PER_BATCH="${GLM5_PREFILL_MAX_SEQS_PER_BATCH:-${GLM5_MAX_SEQS_PER_BATCH:-16}}"
  export GLM5_ENABLE_SCHEDULE_OVERLAP="${GLM5_ENABLE_SCHEDULE_OVERLAP:-false}"
  export GLM5_HOST=10.121.34.85 GLM5_PORT=19100 GLM5_MASTER_ADDR=10.121.34.85:29907
  export GLM5_TRANSFER_LISTEN_PORT=26100 GLM5_PD_RPC_PORT=27107 GLM5_LOG_DIR=$PWD/pd_logs/prefill
  export GLM5_CP_SIZE=${GLM5_CP_SIZE:-8} GLM5_EP_SIZE=${GLM5_EP_SIZE:-8}
  export XLLM_GLM5_PREFILL_CP=$((GLM5_CP_SIZE > 1)) XLLM_GLM5_PREFILL_EP=$((GLM5_EP_SIZE > 1))
else
  export GLM5_MAX_SEQS_PER_BATCH="${GLM5_DECODE_MAX_SEQS_PER_BATCH:-${GLM5_MAX_SEQS_PER_BATCH:-16}}"
  export GLM5_ENABLE_SCHEDULE_OVERLAP="${GLM5_DECODE_SCHEDULE_OVERLAP:-true}"
  export GLM5_HOST=10.121.34.87 GLM5_PORT=19200 GLM5_MASTER_ADDR=10.121.34.87:29908
  export GLM5_TRANSFER_LISTEN_PORT=26200 GLM5_PD_RPC_PORT=27108 GLM5_LOG_DIR=$PWD/pd_logs/decode
  export GLM5_ENABLE_GRAPH="${GLM5_DECODE_GRAPH:-${GLM5_ENABLE_GRAPH:-true}}"
  # Decode can opt into expert parallelism independently of Prefill. Keep the
  # historical TP-style decode default at 1 until an EP A/B has completed.
  export GLM5_CP_SIZE=1 GLM5_EP_SIZE="${GLM5_DECODE_EP_SIZE:-1}" \
    XLLM_GLM5_PREFILL_CP=0 XLLM_GLM5_PREFILL_EP=0
fi
ip -4 addr show | grep -F " $GLM5_HOST/" >/dev/null || { echo "$role requires $GLM5_HOST" >&2; exit 1; }
generation_file=$GLM5_LOG_DIR/generation
generation=$(cat "$generation_file" 2>/dev/null || :)
if [[ -n ${GLM5_EXPECT_GENERATION:-} && $generation != "$GLM5_EXPECT_GENERATION" ]]; then
  exit 0
fi
if [[ $action == watch ]]; then
  [[ -s $GLM5_LOG_DIR/pids.txt ]] || exit 1
  pid=$(head -1 "$GLM5_LOG_DIR/pids.txt")
  [[ $(readlink "/proc/$pid/fd/2") == "$GLM5_LOG_DIR/rank_0.log" ]] || exit 1
else
  if [[ $role == PREFILL && $action == start ]]; then bash pd_control.sh start; fi
  bash start_service.sh "$action"
fi
case "$action" in
  start|watch)
    generation=$(cat /proc/sys/kernel/random/uuid)
    echo "$generation" > "$generation_file.tmp"
    mv "$generation_file.tmp" "$generation_file"
    setsid env -u GLM5_ROLE_LOCKED bash "$PWD/pd_stop_watch.sh" "$role" "$generation" \
      >"$GLM5_LOG_DIR/stop-watch.log" 2>&1 </dev/null &
    ;;
  stop)
    if [[ $role == PREFILL ]]; then bash pd_control.sh stop; fi
    if [[ -n $generation ]]; then
      rm -f "$generation_file" "$GLM5_LOG_DIR/stop.$generation"
      touch "$GLM5_LOG_DIR/stopped.$generation"
    fi
    ;;
esac
