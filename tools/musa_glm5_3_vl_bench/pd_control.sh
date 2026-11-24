#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dir=$PWD/pd_logs/control
mkdir -p "$dir"
host=10.121.34.85
etcd=http://$host:22379
peer=http://127.0.0.1:22380
bins=/opt/xllm_service/bin
if [[ ${1:-start} == stop ]]; then
  for name in coordinator router etcd; do
    pidfile=$dir/$name.pid
    [[ -s $pidfile ]] || continue
    pid=$(cat "$pidfile")
    if [[ $(readlink "/proc/$pid/fd/1" || :) == "$dir/$name.log" ]]; then
      kill "$pid" || true
      for ((i=0;i<50;i++)); do
        [[ $(readlink "/proc/$pid/fd/1" || :) == "$dir/$name.log" ]] || break
        sleep 0.1
      done
      if [[ $(readlink "/proc/$pid/fd/1" || :) == "$dir/$name.log" ]]; then
        kill -KILL "$pid"
        for ((i=0;i<50;i++)); do
          [[ $(readlink "/proc/$pid/fd/1" || :) == "$dir/$name.log" ]] || break
          sleep 0.1
        done
        [[ $(readlink "/proc/$pid/fd/1" || :) != "$dir/$name.log" ]] || exit 1
      fi
    fi
    rm -f "$pidfile"
  done
  exit
fi
launch() {
  local name=$1
  shift
  if [[ -s $dir/$name.pid && $(readlink "/proc/$(cat "$dir/$name.pid")/fd/1" || :) == "$dir/$name.log" ]]; then
    return
  fi
  setsid "$@" >"$dir/$name.log" 2>&1 </dev/null &
  echo $! >"$dir/$name.pid"
}
wait_url() {
  for ((i=0;i<60;i++)); do
    curl --noproxy '*' -fsSo /dev/null --max-time 2 "$1" 2>/dev/null && return
    kill -0 "$(cat "$dir/$2.pid")" 2>/dev/null || break
    sleep 1
  done
  tail -20 "$dir/$2.log" >&2
  return 1
}
launch etcd "$bins/etcd" --name=glm53 --data-dir=/tmp/glm53-pd-etcd \
  --listen-client-urls="$etcd" --advertise-client-urls="$etcd" \
  --listen-peer-urls="$peer" --initial-advertise-peer-urls="$peer" --initial-cluster="glm53=$peer"
wait_url "$etcd/health" etcd
token_file=/tmp/glm53-pd-internal-token
if [[ ! -s $token_file ]]; then (umask 077; openssl rand -hex 32 >"$token_file"); fi
common=(--server_host="$host" --etcd_addr="$host:22379" --block_size=256
  --tokenizer_path="${GLM5_MODEL_PATH:-/data/nfs_shared/models/GLM-5.3-Flash}"
  --native_pd_routing_policy=STATIC_BUCKET --kv_route_mode=DISABLED
  --internal_api_token="$(cat "$token_file")" --logtostderr=true
  --provider_ensure_link_timeout_ms=120000 --provider_adjacency_prewarm_timeout_ms=120000
  --flow_max_queue_wait_ms=150000
  --default_request_deadline_ms=300000 --max_request_deadline_ms=300000)
for spec in 'router ROUTER 28888 28889' 'coordinator COORDINATOR 28890 28891'; do
  read -r name role http rpc <<<"$spec"
  launch "$name" env LD_LIBRARY_PATH=/opt/xllm_service/lib:/usr/lib:/usr/lib/x86_64-linux-gnu \
    "$bins/xllm_master_serving" "${common[@]}" --service_deployment_role="$role" \
    --http_server_port="$http" --rpc_server_port="$rpc"
done
wait_url "http://$host:28888/livez" router
