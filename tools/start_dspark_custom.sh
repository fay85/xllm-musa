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
mkdir -p "$run_dir"
[[ ! -e "$run_dir/launch_manifest.txt" ]] || { echo "run directory already used" >&2; exit 2; }
preload="$custom_ar:/data/nfs_shared/xllm2_artifacts/libasio_gcc.so:/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so"
library_path="/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/openmpi/lib:/usr/local/lib/python3.10/dist-packages/tilelang/lib:${LD_LIBRARY_PATH:-}"
{
  date --iso-8601=seconds
  echo "xllm_bin=$xllm_bin"
  sha256sum "$xllm_bin"
  echo "model_path=$model_path"
  echo "master_addr=$master"
  echo "http_port=$port"
  echo "topology=TP8/EP${ep_size}/DP${dp_size}/CP1"
  echo "graph=true"
  echo "schedule_overlap=false"
  echo "prefix_cache=true"
  echo "num_speculative_tokens=5"
  echo "speculative_algorithm=DSpark"
  echo "max_tokens_per_batch=$max_tokens"
  echo "max_memory_utilization=$max_memory"
  echo "kv_cache_dtype=$kv_cache_dtype"
  echo "host_blocks_factor=$host_blocks_factor"
  echo "num_speculative_tokens=$num_speculative_tokens"
  echo "disable_draft=$disable_draft"
  echo "enable_chunked_prefill=$enable_chunked"
  echo "speculative_algorithm=$speculative_algorithm"
  echo "enable_graph=$enable_graph"
  echo "block_size=128"
  echo "enable_chunked_prefill=true"
  echo "ops_path=$ops_path"
  echo "custom_ar_so=$custom_ar"
  sha256sum "$custom_ar"
} > "$run_dir/launch_manifest.txt"

pids=()
draft_args=(--draft_model="$model_path")
if [[ "$disable_draft" == true ]]; then
  num_speculative_tokens=0
  draft_args=()
fi
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
    --enable_prefix_cache=true --enable_in_batch_prefix_cache=false \
    --max_tokens_per_batch="$max_tokens" --max_seqs_per_batch=1 \
    --max_concurrent_requests=1 --num_response_handling_threads=4 \
    --enable_chunked_prefill="$enable_chunked" \
    --max_tokens_per_chunk_for_prefill="$max_tokens" --enable_shm=false \
    --enable_schedule_overlap=false --num_speculative_tokens="$num_speculative_tokens" \
    --speculative_algorithm="$speculative_algorithm" --enable_mtp_draft_body_tp1=false \
    "${draft_args[@]}" \
    --enable_graph="$enable_graph" \
    --enable_graph_vmm_pool=true --enable_graph_mode_decode_no_padding=true \
    --max_tokens_for_graph_mode=2048 --enable_online_profile=false \
    --profile_backend=torch --profile_dir="$run_dir/profile" \
    > "$run_dir/rank${rank}.log" 2>&1 &
  pids+=("$!")
}
for rank in 1 2 3 4 5 6 7; do launch_rank "$rank"; done
launch_rank 0
wait -n "${pids[@]}"
