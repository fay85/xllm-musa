#!/usr/bin/env bash
set -euo pipefail

# Local DSpark launcher for memory experiments.  It intentionally reuses the
# same binary, AOT objects, preload set and TP8 arguments as the documented
# launcher, while allowing chunked-prefill/cache tuning inside this checkout.
run_dir=${1:?run directory}
max_tokens=${2:?max tokens per batch}
max_memory=${3:?max memory utilization}
xllm_bin=${DSPARK_BIN:?set DSPARK_BIN}
model_path=${MODEL_PATH:?set MODEL_PATH}
port=${XLLM2_EXTERNAL_TP8_HTTP_PORT:-18861}
master=${XLLM2_EXTERNAL_TP8_MASTER_ADDR:-127.0.0.1:20861}
ops_path=${XLLM2_EXTERNAL_TP8_OPS_PATH:-/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops}
custom_ar=${XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO:-/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so}
kv_cache_dtype=${XLLM2_EXTERNAL_TP8_KV_CACHE_DTYPE:-auto}
ep_size=${XLLM2_EXTERNAL_TP8_EP_SIZE:-1}
dp_size=${XLLM2_EXTERNAL_TP8_DP_SIZE:-1}
host_blocks_factor=${XLLM2_EXTERNAL_TP8_HOST_BLOCKS_FACTOR:-0}
num_speculative_tokens=${XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS:-5}
disable_draft=${XLLM2_EXTERNAL_TP8_DISABLE_DRAFT:-false}
enable_chunked=${XLLM2_EXTERNAL_TP8_CHUNKED_PREFILL:-true}
speculative_algorithm=${XLLM2_EXTERNAL_TP8_SPECULATIVE_ALGORITHM:-DSpark}
enable_graph=${XLLM2_EXTERNAL_TP8_GRAPH:-true}
enable_prefix_cache=${XLLM2_EXTERNAL_TP8_PREFIX_CACHE:-true}
enable_in_batch_prefix_cache=${XLLM2_EXTERNAL_TP8_IN_BATCH_PREFIX_CACHE:-false}
prefill_threshold=${XLLM2_EXTERNAL_TP8_PREFILL_THRESHOLD:-0.95}
prefill_chunk=${XLLM2_EXTERNAL_TP8_PREFILL_CHUNK:-$max_tokens}
max_graph_tokens=${XLLM2_EXTERNAL_TP8_MAX_GRAPH_TOKENS:-2048}
graph_vmm_pool=${XLLM2_EXTERNAL_TP8_GRAPH_VMM_POOL:-true}
enable_online_profile=${XLLM2_EXTERNAL_TP8_ONLINE_PROFILE:-false}
# Check resources before starting any rank: master+1 is the TP TCPStore port.
# Never stop an existing process here; the experiment owner must release it.
[[ -x "$xllm_bin" ]] || { echo "Missing executable: $xllm_bin" >&2; exit 2; }
python3 - "$port" "$master" <<'PY'
import socket, sys
http_port = int(sys.argv[1])
master_host, master_port = sys.argv[2].rsplit(':', 1)
ports = [http_port, int(master_port), int(master_port) + 1]
if len(set(ports)) != len(ports):
    raise SystemExit('HTTP/master/TCPStore ports must be distinct')
held = []
try:
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        held.append(sock)
        sock.bind((master_host, port))
except OSError as exc:
    raise SystemExit(f'Port preflight failed at {master_host}:{port}: {exc}')
finally:
    for sock in held:
        sock.close()
PY
# Consume ps output fully under pipefail (rg -q can terminate ps with SIGPIPE).
if ps -eo args= | rg '^python[0-9.]* .*musa_gpu_reservation\.py( |$)|^bash (tools/|.*/tools/)reserve_idle_musa\.sh( |$)' >/dev/null; then
  echo "GPU reservation is still running; release it before launching ranks." >&2
  exit 2
fi
draft_args=(--draft_model="$model_path")
if [[ "$disable_draft" == true ]]; then
  num_speculative_tokens=0
  draft_args=()
fi
mkdir -p "$run_dir"
[[ ! -e "$run_dir/launch_manifest.txt" ]] || { echo "run directory already used" >&2; exit 2; }
preload="$custom_ar:/data/nfs_shared/xllm2_artifacts/libasio_gcc.so:/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so"
library_path="/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/openmpi/lib:/usr/local/lib/python3.10/dist-packages/tilelang/lib:${LD_LIBRARY_PATH:-}"
# AOT modules are loaded via dlopen, not linked into the executable. Record
# their content as well as the binary hash so shared-cache updates cannot be
# mistaken for source-only accuracy/performance changes.
find "$ops_path" -maxdepth 2 -type f \( -name '*.so' -o -name 'manifest.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum > "$run_dir/ops_artifacts.sha256"
{
  date --iso-8601=seconds
  echo "xllm_bin=$xllm_bin"
  git rev-parse HEAD
  git diff --stat
  sha256sum "$xllm_bin"
  echo "model_path=$model_path"
  echo "master_addr=$master"
  echo "http_port=$port"
  echo "topology=TP8/EP${ep_size}/DP${dp_size}/CP1"
  echo "graph=$enable_graph"
  echo "schedule_overlap=false"
  echo "prefix_cache=$enable_prefix_cache"
  echo "max_tokens_per_batch=$max_tokens"
  echo "max_memory_utilization=$max_memory"
  echo "max_tokens_per_chunk_for_prefill=$prefill_chunk"
  echo "max_tokens_for_graph_mode=$max_graph_tokens"
  echo "graph_vmm_pool=$graph_vmm_pool"
  echo "enable_online_profile=$enable_online_profile"
  echo "kv_cache_dtype=$kv_cache_dtype"
  echo "host_blocks_factor=$host_blocks_factor"
  echo "num_speculative_tokens=$num_speculative_tokens"
  echo "disable_draft=$disable_draft"
  echo "enable_chunked_prefill=$enable_chunked"
  echo "speculative_algorithm=$speculative_algorithm"
  echo "enable_graph=$enable_graph"
  echo "prefill_scheduling_memory_usage_threshold=$prefill_threshold"
  echo "block_size=128"
  echo "ops_path=$ops_path"
  echo "ops_artifact_hashes=$run_dir/ops_artifacts.sha256"
  sha256sum "$run_dir/ops_artifacts.sha256"
  echo "custom_ar_so=$custom_ar"
  sha256sum "$custom_ar"
  echo "routing_environment_begin"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
  # Only allowlisted execution flags; never dump unrelated credentials.
  env | LC_ALL=C sort | rg '^(XLLM_DSV4_|XLLM_GRAPH_|XLLM_MUSA_GRAPH_|XLLM_KEEP_HOST_MIRROR_REFRESH=)' || true
  echo "routing_environment_end"
} > "$run_dir/launch_manifest.txt"

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT
launch_rank() {
  local rank=$1
  env LD_PRELOAD="$preload" LD_LIBRARY_PATH="$library_path" \
    TORCH_MCCL_ASYNC_ERROR_HANDLING=0 TORCH_MCCL_ENABLE_MONITORING=0 \
    MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FLASHINFER_OPS_PATH="$ops_path" \
    XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS=120 \
    "$xllm_bin" --model="$model_path" --model_id=DeepSeek-V4-Flash-FP8 \
    --backend=llm --host=127.0.0.1 --port="$port" \
    --master_node_addr="$master" --nnodes=8 --node_rank="$rank" \
    --dp_size="$dp_size" --ep_size="$ep_size" --cp_size=1 --communication_backend=mccl \
    --block_size=128 --max_memory_utilization="$max_memory" \
    --kv_cache_dtype="$kv_cache_dtype" \
    --host_blocks_factor="$host_blocks_factor" \
    --enable_prefix_cache="$enable_prefix_cache" \
    --enable_in_batch_prefix_cache="$enable_in_batch_prefix_cache" \
    --max_tokens_per_batch="$max_tokens" --max_seqs_per_batch=1 \
    --max_concurrent_requests=1 --num_response_handling_threads=4 \
    --enable_chunked_prefill="$enable_chunked" \
    --max_tokens_per_chunk_for_prefill="$prefill_chunk" --enable_shm=false \
    --prefill_scheduling_memory_usage_threshold="$prefill_threshold" \
    --enable_schedule_overlap=false --num_speculative_tokens="$num_speculative_tokens" \
    --speculative_algorithm="$speculative_algorithm" --enable_mtp_draft_body_tp1=false \
    "${draft_args[@]}" \
    --enable_graph="$enable_graph" \
    --enable_graph_vmm_pool="$graph_vmm_pool" --enable_graph_mode_decode_no_padding=true \
    --max_tokens_for_graph_mode="$max_graph_tokens" --enable_online_profile="$enable_online_profile" \
    --profile_backend=torch --profile_dir="$run_dir/profile" \
    > "$run_dir/rank${rank}.log" 2>&1 &
  pids+=("$!")
}
for rank in 1 2 3 4 5 6 7; do launch_rank "$rank"; done
launch_rank 0
wait -n "${pids[@]}"
