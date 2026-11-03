#!/bin/bash
# Start xllm-service control plane only: etcd + master (ROUTER) + coordinator.
# Topology this serves: DP3 CP2 Prefill + DP2 TP1 Decode (engines via start_stack).
# Config: /workspace/config.json (via env.sh / load_config.sh).
# Order: 1) this script  2) full engine stack -- or use start_stack.sh which does both.
# Does NOT start Prefill/Decode engines. Does NOT check GPU free.
set -u
source /workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench/env.sh || exit $?

die() {
  echo "ERROR: $*" >&2
  [[ -n ${RUN:-} ]] && echo "stop the processes started so far with stop_stack.sh" >&2
  exit 1
}

for f in "$MASTER_BIN" "$ETCD_BIN" "$RUST_TOKENIZERS_LIB"; do
  [[ -f $f ]] || die "missing $f (see env.sh / config.json paths.*)"
done
if [[ ! -f $MASTER_TOKENIZER/tokenizer_config.json ]]; then
  if [[ -f /root/xllm_vl_bench/master_tokenizer/tokenizer_config.json ]]; then
    MASTER_TOKENIZER=/root/xllm_vl_bench/master_tokenizer
  elif [[ -f $MODEL/tokenizer_config.json ]]; then
    MASTER_TOKENIZER=$MODEL
  fi
fi
[[ -f $MASTER_TOKENIZER/tokenizer_config.json ]] ||
  die "missing $MASTER_TOKENIZER, run prepare_data.sh first" 

ports=("$ETCD_CLIENT_PORT" "$ETCD_PEER_PORT" "$MASTER_HTTP_PORT" "$MASTER_RPC_PORT" 64890 64891)
listening=$(ss -ltn)
for p in "${ports[@]}"; do
  grep -q ":$p " <<< "$listening" && die "port $p is in use"
done

RUN=$RUNS_DIR/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN/master_lib"
ln -sfn "$RUN" "$RUNS_DIR/current"
ln -sf "$RUST_TOKENIZERS_LIB" "$RUN/master_lib/librust_tokenizers.so"
{
  echo "mode=service_only"
  echo "config_json=${CONFIG_JSON:-/workspace/config.json} loaded=${CONFIG_JSON_LOADED:-0}"
  echo "master_bin=$MASTER_BIN md5=$(md5sum < "$MASTER_BIN" | cut -c1-32)"
  echo "etcd_bin=$ETCD_BIN"
  echo "profile=$PROFILE service_vmodule=$SERVICE_VMODULE"
  echo "batch_credit_max_grant_units=$BATCH_CREDIT_MAX_GRANT_UNITS"
  echo "batch_credit_safety_permyriad=$BATCH_CREDIT_SAFETY_PERMYRIAD"
  echo "master_url=$MASTER_URL"
} > "$RUN/stack_info.txt"

service_logging=()
if [[ -n ${SERVICE_VMODULE:-} ]]; then
  service_logging+=(--vmodule="$SERVICE_VMODULE")
fi

setsid "$ETCD_BIN" --name=vl-bench --data-dir="$RUN/etcd_data" \
  --listen-client-urls="http://127.0.0.1:$ETCD_CLIENT_PORT" \
  --advertise-client-urls="http://127.0.0.1:$ETCD_CLIENT_PORT" \
  --listen-peer-urls="http://127.0.0.1:$ETCD_PEER_PORT" \
  --initial-advertise-peer-urls="http://127.0.0.1:$ETCD_PEER_PORT" \
  --initial-cluster="vl-bench=http://127.0.0.1:$ETCD_PEER_PORT" \
  --initial-cluster-token="vl-bench-$(basename "$RUN")" --initial-cluster-state=new \
  > "$RUN/etcd.log" 2>&1 < /dev/null &
echo "etcd $!" >> "$RUN/pids.txt"
sleep 3

LD_LIBRARY_PATH=$RUN/master_lib:/usr/lib:/usr/lib/x86_64-linux-gnu setsid "$MASTER_BIN" \
  --server_host="$HOST" --etcd_addr="127.0.0.1:$ETCD_CLIENT_PORT" \
  --http_server_port="$MASTER_HTTP_PORT" --rpc_server_port="$MASTER_RPC_PORT" \
  --tokenizer_path="$MASTER_TOKENIZER" --block_size=64 \
  --http_server_num_threads=4 --rpc_server_num_threads=4 --num_threads=4 --logtostderr=true \
  "${service_logging[@]}" \
  --batch_credit_max_grant_units="$BATCH_CREDIT_MAX_GRANT_UNITS" \
  --batch_credit_safety_permyriad="$BATCH_CREDIT_SAFETY_PERMYRIAD" \
  --batch_credit_max_grant_units_per_router_tenant="$BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT" \
  --internal_api_token=lab-local-token --service_deployment_role=ROUTER \
  --native_pd_routing_policy=STATIC_BUCKET --load_balance_policy="${LOAD_BALANCE_POLICY:-CAR}" \
  --kv_route_mode=DISABLED --tool_call_parser="${TOOL_CALL_PARSER:-qwen3_coder}" \
  --first_output_watchdog_timeout_ms="${VL_FIRST_OUTPUT_WATCHDOG_MS:-60000}" \
  --max_nonstream_retry_wasted_device_ms="${VL_DEVICE_TIME_BUDGET_MS:-5000}" \
  --flow_max_queue_wait_ms="${VL_FLOW_MAX_QUEUE_WAIT_MS:-15000}" \
  --default_request_deadline_ms="${VL_REQUEST_DEADLINE_MS:-300000}" \
  --max_request_deadline_ms="${VL_REQUEST_DEADLINE_MS:-300000}" \
  > "$RUN/master.log" 2>&1 < /dev/null &
echo "master $!" >> "$RUN/pids.txt"
sleep 2

LD_LIBRARY_PATH=$RUN/master_lib:/usr/lib:/usr/lib/x86_64-linux-gnu setsid "$MASTER_BIN" \
  --server_host="$HOST" --etcd_addr="127.0.0.1:$ETCD_CLIENT_PORT" \
  --http_server_port=64890 --rpc_server_port=64891 \
  --tokenizer_path="$MASTER_TOKENIZER" --block_size=64 \
  --num_threads=2 --logtostderr=true \
  "${service_logging[@]}" \
  --batch_credit_max_grant_units="$BATCH_CREDIT_MAX_GRANT_UNITS" \
  --batch_credit_safety_permyriad="$BATCH_CREDIT_SAFETY_PERMYRIAD" \
  --batch_credit_max_grant_units_per_router_tenant="$BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT" \
  --internal_api_token=lab-local-token --service_deployment_role=COORDINATOR \
  --native_pd_routing_policy=STATIC_BUCKET --load_balance_policy="${LOAD_BALANCE_POLICY:-CAR}" \
  --kv_route_mode=DISABLED --tool_call_parser="${TOOL_CALL_PARSER:-qwen3_coder}" \
  --first_output_watchdog_timeout_ms="${VL_FIRST_OUTPUT_WATCHDOG_MS:-60000}" \
  --max_nonstream_retry_wasted_device_ms="${VL_DEVICE_TIME_BUDGET_MS:-5000}" \
  --flow_max_queue_wait_ms="${VL_FLOW_MAX_QUEUE_WAIT_MS:-15000}" \
  --default_request_deadline_ms="${VL_REQUEST_DEADLINE_MS:-300000}" \
  --max_request_deadline_ms="${VL_REQUEST_DEADLINE_MS:-300000}" \
  > "$RUN/coordinator.log" 2>&1 < /dev/null &
echo "coordinator $!" >> "$RUN/pids.txt"
sleep 3

for _ in $(seq 180); do
  curl -fsS -o /dev/null -m 2 "$MASTER_URL/readyz" && break
  sleep 1
done
curl -fsS -o /dev/null -m 2 "$MASTER_URL/readyz" || die "master is not ready on $MASTER_URL/readyz (see $RUN/master.log)"
echo "logs: $RUN"
echo "config: ${CONFIG_JSON:-/workspace/config.json}"
echo "master: $MASTER_URL"
echo READY_SERVICE
