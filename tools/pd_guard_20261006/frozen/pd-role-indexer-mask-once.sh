#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

role=${1:?usage: pd-role-clean.sh PREFILL|DECODE RUN_DIR}
run_dir=${2:?usage: pd-role-clean.sh PREFILL|DECODE RUN_DIR}
fixed_data_ports=()
if [[ -n "${MCK_DATA_PORTS:-}" ]]; then
  IFS=, read -r -a fixed_data_ports <<< "$MCK_DATA_PORTS"
  [[ ${#fixed_data_ports[@]} == 8 ]]
  for value in "${fixed_data_ports[@]}"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -ge 1024 && "$value" -lt 32768 ]]
  done
fi
unset MC_MIN_RPC_PORT MC_MAX_RPC_PORT MC_MIN_PRC_PORT MC_MAX_PRC_PORT
root=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005
common=/data/nfs_shared/jiaming.wang/pd_n6q7_r3_20260927
bin=${DSPARK_BIN:-$root/xllm}
expected_bin_sha=${DSPARK_BIN_SHA256:?set the independently verified candidate SHA256}
# Byte-for-byte copies from host28's accepted N6/q7 run, not a reconstructed
# baseline. Fail before launching any rank if either is absent or changed.
reference_manifest=$root/reference_v128.launch_manifest.txt
reference_ops=$common/reference_n6q7.ops_artifacts.sha256
reference_manifest_sha=35f6158cb8b18d89bcc2885d1762c4c25e50b6c7ea88b586545bf3d6b66715c7
reference_ops_sha=c9f9a0b32cb87c33e3edcce537c42f82cb3ec05d3c6bac82659151f35df29d21
reference_ops_root=/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops
model=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
ops=/data/nfs_shared/jiaming.wang/gsm8k_goal_20260919_1555/layerdiff_20260921/report_deps_20260925/cached_ops
ar=/data/nfs_shared/jiaming.wang/gsm8k_goal_20260919_1555/layerdiff_20260921/report_deps_20260925/libxllm_custom_ar_sglang_current.so
etcd=10.121.38.92:2489
graph=${GRAPH:-true}
case "$graph" in true|false) ;; *) echo 'GRAPH must be true or false' >&2; exit 2 ;; esac
prefix_cache=${PREFIX_CACHE:-true}
case "$prefix_cache" in true|false) ;; *) echo 'PREFIX_CACHE must be true or false' >&2; exit 2 ;; esac
max_seqs=${MAX_SEQS_PER_BATCH:-16}
max_concurrent=${MAX_CONCURRENT_REQUESTS:-16}
max_tokens=${MAX_TOKENS_PER_BATCH:-10032}
request_threads=${REQUEST_THREADS:-4}
prefill_chunk=${PREFILL_CHUNK:-8192}
overlap=${OVERLAP:-true}
[[ "$role" != PREFILL ]] || overlap=false
prefill_threshold=${PREFILL_THRESHOLD:-0.95}
[[ "$max_seqs" =~ ^[0-9]+$ && "$max_concurrent" =~ ^[0-9]+$ && "$max_tokens" =~ ^[0-9]+$ && "$request_threads" =~ ^[0-9]+$ ]] || { echo 'batch/thread values must be integers' >&2; exit 2; }
if [[ "$role" == PREFILL || "$role" == DECODE ]]; then
  : "${PREFIX_CACHE_EFFECTIVE:=$prefix_cache}"
fi

# Erase inherited experiment toggles, including the obsolete MUSA-prefixed
# variables in the older PD launcher. They must not silently affect this run.
while IFS= read -r name; do
  case "$name" in
    XLLM_DSV4_*|XLLM_DSPARK_*|XLLM_GRAPH_*|XLLM_MUSA_*|XLLM2_EXTERNAL_TP8_*|XLLM_KEEP_HOST_MIRROR_REFRESH)
      unset "$name" ;;
  esac
done < <(compgen -e)
unset OMP_NUM_THREADS MKL_NUM_THREADS LD_PRELOAD

check_hash() {
  local expected=$1 file=$2 actual
  [[ -r "$file" ]] || { echo "Missing dependency: $file" >&2; exit 2; }
  read -r actual _ < <(sha256sum -- "$file")
  [[ "$actual" == "$expected" ]] || {
    echo "Artifact hash mismatch: $file" >&2; exit 2;
  }
}
[[ "$expected_bin_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'DSPARK_BIN_SHA256 must contain a complete lowercase SHA256' >&2; exit 2;
}
[[ -x "$bin" ]] || { echo "Missing executable: $bin" >&2; exit 2; }
check_hash "$expected_bin_sha" "$bin"
check_hash "$reference_manifest_sha" "$reference_manifest"
check_hash "$reference_ops_sha" "$reference_ops"
check_hash c2578d935edc3af6b4a39f124d06c8e13b68d345bc7a07c27f5581ce310082a5 "$ar"
check_hash a5cdd3ccb6ee6c1d385fda7a4ca363c5b2688cbc4e69265841b329f0cf3b5d6d "$common/libasio.so"

# Preserve the complete accepted v128 routing environment, not the old q7 subset.
mapfile -t routing_env < <(grep '^XLLM_DSV4_' "$reference_manifest")
[[ ${#routing_env[@]} -gt 60 ]] || { echo 'Truncated v128 routing environment' >&2; exit 2; }
for entry in "${routing_env[@]}"; do
  [[ "$entry" =~ ^XLLM_DSV4_[A-Z0-9_]+=[0-9]+$ ]] || {
    echo 'Invalid entry in frozen routing environment' >&2; exit 2;
  }
done

apply_diag_override() {
  local name=$1 value=$2
  local filtered=()
  for entry in "${routing_env[@]}"; do
    [[ "$entry" == "$name="* ]] || filtered+=("$entry")
  done
  filtered+=("$name=$value")
  routing_env=("${filtered[@]}")
}
[[ "${DIAG_COMPACT_PREFILL_FP8:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_COMPACT_PREFILL_FP8 "$DIAG_COMPACT_PREFILL_FP8"
[[ "${DIAG_PREFILL_INDEXER_BF16_FP32:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_INDEXER_BF16_FP32 "$DIAG_PREFILL_INDEXER_BF16_FP32"
[[ "${DIAG_PREFILL_PREPARE_ONCE:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_PREPARE_ONCE "$DIAG_PREFILL_PREPARE_ONCE"
apply_diag_override XLLM_DSV4_REQUIRE_PREFILL_PREPARE_ONCE "$DIAG_PREFILL_PREPARE_ONCE"
[[ "${DIAG_HC_TILED_PREFILL:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_HC_TILED_PREFILL "$DIAG_HC_TILED_PREFILL"
[[ "${DIAG_PREFILL_GATE_BF16_FP32:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_GATE_BF16_FP32 "$DIAG_PREFILL_GATE_BF16_FP32"
[[ "${DIAG_LONG_MOE_VECTOR_RESTORE:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_LONG_MOE_VECTOR_RESTORE "$DIAG_LONG_MOE_VECTOR_RESTORE"
[[ "${DIAG_PREFILL_REDUCE_SCATTER:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_NATIVE_REDUCE_SCATTER "$DIAG_PREFILL_REDUCE_SCATTER"
[[ "${DIAG_LONG_MOE_WARP_SWIGLU:?set 0 or 1}" =~ ^[01]$ ]]
[[ "$role" == PREFILL || "$DIAG_LONG_MOE_WARP_SWIGLU" == 0 ]]
apply_diag_override XLLM_DSV4_LONG_MOE_WARP_SWIGLU "$DIAG_LONG_MOE_WARP_SWIGLU"
[[ "${DIAG_INDEXER_ROPE_COPY:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_INDEXER_ROPE_COPY "$DIAG_INDEXER_ROPE_COPY"
[[ "${DIAG_MAIN_ROPE_COPY:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_MAIN_ROPE_COPY "$DIAG_MAIN_ROPE_COPY"
[[ "${DIAG_INDEXER_MASK_ONCE:?set 0 or 1}" =~ ^[01]$ ]]
apply_diag_override XLLM_DSV4_PREFILL_INDEXER_MASK_ONCE "$DIAG_INDEXER_MASK_ONCE"
if [[ "${DIAG_GRAPH_PER_REQUEST_RANGES:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_GRAPH_PER_REQUEST_RANGES 1
fi
if [[ "${DIAG_DISABLE_BATCHED_MATE:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY 0
fi
if [[ "${DIAG_DISABLE_MATE_GROUPED_METADATA:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_MATE_GROUPED_METADATA 0
fi
if [[ "${DIAG_DISABLE_BATCHED_INDEXER:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_GRAPH_BATCHED_INDEXER 0
fi
if [[ "${DIAG_SKIP_SP_MOE_ALLGATHER:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_SP_SKIP_MOE_ALLGATHER 1
fi
if [[ "${DIAG_EAGER_TRACE:-0}" == 1 ]]; then
  [[ "$graph" == false ]] || { echo 'Eager trace must not run under Graph' >&2; exit 2; }
  apply_diag_override XLLM_DSV4_MUSA_TRACE 1
fi
if [[ "${DIAG_LAYER4_BOUNDARY:-0}" == 1 ]]; then
  [[ "$graph" == false && "$role" == DECODE ]] || { echo 'Boundary probe is eager Decode-only' >&2; exit 2; }
  apply_diag_override XLLM_DSV4_LAYER4_PROBE_DIR "$run_dir/layer4_probe"
fi
if [[ "${DIAG_DISABLE_GRAPH_Q6_COMPRESSOR_STATE:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_DISABLE_GRAPH_Q6_COMPRESSOR_STATE 1
fi
if [[ "${DIAG_DISABLE_GRAPH_Q6_INDEXER_STATE:-0}" == 1 ]]; then
  apply_diag_override XLLM_DSV4_DISABLE_GRAPH_Q6_INDEXER_STATE 1
fi

if [[ "${DSPARK_PREFILL_SP_MULTI_SEQUENCE:-0}" == 1 ]]; then routing_env+=("XLLM_DSV4_PREFILL_SP_MULTI_SEQUENCE=1"); fi
if [[ "${DSPARK_LONG_PREFILL_RENDEZVOUS:-0}" == 1 ]]; then routing_env+=("XLLM_DSV4_LONG_PREFILL_RENDEZVOUS=1"); fi
# Rebind only the accepted artifact root. Every SO and metadata manifest must
# match the single-host reference bytes; no mixed AOT fallback is permitted.
while read -r digest source; do
  [[ "$source" == "$reference_ops_root/"* ]] || {
    echo "Unexpected reference artifact: $source" >&2; exit 2;
  }
  check_hash "$digest" "$ops/${source#"$reference_ops_root/"}"
done < "$reference_ops"

if [[ "$role" == PREFILL ]]; then
  host=10.121.38.94
  master=10.121.38.94:20861
  http=18861
  transfer=26000
  pd_port=7777
elif [[ "$role" == DECODE ]]; then
  host=10.121.38.90
  master=10.121.38.90:20862
  http=18862
  transfer=26100
  pd_port=7787
else
  echo "unsupported role: $role" >&2
  exit 2
fi

[[ ! -e "$run_dir" ]] || { echo "run directory already exists: $run_dir" >&2; exit 2; }
lib="$common:$ops:/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:/workspace/MCCL_2.3.0/mccl/lib:/usr/local/openmpi/lib:/usr/local/lib/python3.10/dist-packages/tilelang/lib"
preload="$ar:/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so"
for dependency in \
  /usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so \
  /usr/local/lib/python3.10/dist-packages/tvm_ffi/lib/libtvm_ffi.so \
  /usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so \
  "$model/config.json" "$model/tokenizer.json"; do
  [[ -r "$dependency" ]] || { echo "Missing dependency: $dependency" >&2; exit 2; }
done

# No model/GPU work during preflight. This checks local binds, the existing
# Control RPC and etcd health, and refuses any competing visible GPU process.
python3 - "$host" "$master" "$http" "$transfer" "$pd_port" "$model" <<'PY'
import glob, json, os, socket, sys, urllib.request
host, master, http, transfer, pd_port, model = sys.argv[1:]
for path in glob.glob('/proc/[0-9]*/fd/*'):
    try:
        device = os.readlink(path)
    except (OSError, PermissionError):
        continue
    if device.startswith(('/dev/musa', '/dev/mtgpu')):
        raise SystemExit(f'Visible GPU process still owns {device}: {path}')
master_port = int(master.rsplit(':', 1)[1])
ports = [master_port, master_port + 1]
ports += list(range(int(http), int(http) + 8))
ports += list(range(int(transfer), int(transfer) + 8))
ports += list(range(int(pd_port), int(pd_port) + 8))
if len(ports) != len(set(ports)):
    raise SystemExit('Role ports overlap')
held = []
try:
    for port in ports:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.append(sock)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise SystemExit(f'Cannot bind role endpoint {host}:{port}: {exc}')
finally:
    for sock in held:
        sock.close()
with socket.create_connection(('10.121.38.92', 29889), timeout=3):
    pass
with urllib.request.urlopen('http://10.121.38.92:2489/health', timeout=3) as response:
    health = json.load(response)
if health.get('health') not in (True, 'true'):
    raise SystemExit(f'etcd is not healthy: {health}')
indices = glob.glob(model + '/*.safetensors.index.json')
if not indices:
    raise SystemExit('Missing model safetensors index')
for index in indices:
    with open(index) as file:
        shards = set(json.load(file)['weight_map'].values())
    missing = [name for name in shards if not os.path.isfile(model + '/' + name)]
    if missing:
        raise SystemExit(f'Missing model shards: {missing}')
print('Static/container preflight passed; host GPU-idleness check is still required.')
PY

# Record loader resolution before any rank starts. Do not use inherited paths.
# `libmusa_python.so` is a Python extension and cannot be preloaded into the
# standalone ELF while running ldd (`PyInstanceMethod_Type` is unresolved).
# The ranks still receive the historical preload; dependency resolution itself
# only needs the curated library search path.
loader_output=$(env LD_LIBRARY_PATH="$lib" ldd "$bin")
if [[ "$loader_output" == *'not found'* ]]; then
  printf '%s\n' "$loader_output" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "$run_dir")"
mkdir -- "$run_dir"
if [[ "${DIAG_LAYER4_BOUNDARY:-0}" == 1 ]]; then mkdir "$run_dir/layer4_probe"; fi
printf '%s\n' "$loader_output" > "$run_dir/ldd.txt"
cp -- "$reference_manifest" "$run_dir/reference_n6q7.launch_manifest.txt"
cp -- "$reference_ops" "$run_dir/reference_n6q7.ops_artifacts.sha256"
find "$ops" -maxdepth 2 -type f \( -name '*.so' -o -name manifest.json \) \
  -print0 | sort -z | xargs -0 -r sha256sum > "$run_dir/ops_artifacts.sha256"
{
  date --iso-8601=seconds
  printf 'role=%s\nrun_dir=%s\nxllm_bin=%s\nbinary_sha256=%s\n' "$role" "$run_dir" "$bin" "$expected_bin_sha"
  printf 'fixed_mooncake_data_ports=%s\n' "${MCK_DATA_PORTS:-dynamic}"
  printf 'enable_online_profile=true\nprofile_backend=torch\nprofile_dir=%s/profile\n' "$run_dir"
  printf 'reference_manifest_sha256=%s\nreference_ops_sha256=%s\n' "$reference_manifest_sha" "$reference_ops_sha"
  printf 'source_base=b5cc2b94679db6770b7ed8b6e96d1cb922eaa042+v128+MUSA_PD_port\n'
  printf 'v128_parent_binary_sha256=a90f6412a1576f6266444ca5a7b2a20b0796bd8fa8760a18ec392a238c987d57\n'
  printf 'model_path=%s\nmodel_id=DeepSeek-V4-Flash-0731-FP8-mt\n' "$model"
  printf 'master_addr=%s\nhttp_port=%s\ntransfer_port=%s\ndisagg_pd_port=%s\netcd=%s\n' "$master" "$http" "$transfer" "$pd_port" "$etcd"
  printf '%s\n' 'topology=TP8/EP1/DP1/CP1' 'draft_graph=false' "schedule_overlap=$overlap" \
    "graph=$graph" "prefix_cache_requested=$prefix_cache" "prefix_cache_effective=$prefix_cache" \
    "max_tokens_per_batch=$max_tokens" 'max_memory_utilization=0.81' \
    "max_seqs_per_batch=$max_seqs" "max_concurrent_requests=$max_concurrent" "request_threads=$request_threads" \
    "max_tokens_per_chunk_for_prefill=$prefill_chunk" 'max_tokens_for_graph_mode=2048' \
    'graph_vmm_pool=true' 'kv_cache_dtype=auto' 'host_blocks_factor=0' \
    'num_speculative_tokens=6' 'target_query_width=7' 'speculative_algorithm=DSpark' \
    'enable_chunked_prefill=true' "prefill_scheduling_memory_usage_threshold=$prefill_threshold" \
    'block_size=128' 'kv_transfer=Mooncake/PULL' 'kv_split_size=1' \
    'OMP_NUM_THREADS=4' 'MKL_NUM_THREADS=4' 'MUSA_MODULE_LOADING=EAGER'
  printf 'ops_path=%s\ncustom_ar_so=%s\nLD_LIBRARY_PATH=%s\nLD_PRELOAD=%s\n' "$ops" "$ar" "$lib" "$preload"
  printf '%s\n' "${routing_env[@]}"
  sha256sum "$bin" "$ar" "$common/libasio.so" "${BASH_SOURCE[0]}" "$run_dir/ops_artifacts.sha256"
} > "$run_dir/launch_manifest.txt"
sha256sum "$run_dir/launch_manifest.txt" > "$run_dir/launch_manifest.sha256"

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for rank in 0 1 2 3 4 5 6 7; do
  rank_http=$((http + rank))
  rank_transfer=$((transfer + rank))
  rank_data_env=()
  if [[ ${#fixed_data_ports[@]} != 0 ]]; then
    rank_data_env=("MC_MIN_RPC_PORT=${fixed_data_ports[rank]}" "MC_MAX_RPC_PORT=${fixed_data_ports[rank]}")
    printf '%s\t%s\n' "$rank" "${fixed_data_ports[rank]}" >> "$run_dir/rank_data_ports.tsv"
  fi
  env LD_PRELOAD="$preload" LD_LIBRARY_PATH="$lib" MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FLASHINFER_OPS_PATH="$ops" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUSA_MODULE_LOADING=EAGER \
    "${routing_env[@]}" "${rank_data_env[@]}" \
    TORCH_MCCL_ASYNC_ERROR_HANDLING=0 TORCH_MCCL_ENABLE_MONITORING=0 \
    XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS=120 \
    "$bin" --model="$model" --model_id=DeepSeek-V4-Flash-0731-FP8-mt \
      --backend=llm --task=generate --host="$host" --port="$rank_http" \
      --master_node_addr="$master" --nnodes=8 --node_rank="$rank" \
      --dp_size=1 --ep_size=1 --cp_size=1 --communication_backend=mccl \
      --block_size=128 --max_memory_utilization=0.81 --kv_cache_dtype=auto \
      --enable_prefix_cache="$prefix_cache" --enable_in_batch_prefix_cache=false \
      --max_tokens_per_batch="$max_tokens" --max_seqs_per_batch="$max_seqs" --max_concurrent_requests="$max_concurrent" \
      --num_request_handling_threads="$request_threads" --num_response_handling_threads=4 \
      --max_tokens_per_chunk_for_prefill="$prefill_chunk" --enable_chunked_prefill=true \
      --prefill_scheduling_memory_usage_threshold="$prefill_threshold" --host_blocks_factor=0 \
      --enable_shm=false --enable_schedule_overlap="$overlap" --num_speculative_tokens=6 \
      --speculative_algorithm=DSpark --enable_mtp_draft_body_tp1=false \
      --draft_model="$model" --enable_graph="$graph" --enable_graph_vmm_pool=true \
      --enable_graph_mode_decode_no_padding=true --max_tokens_for_graph_mode=2048 \
      --enable_disagg_pd=true --instance_role="$role" \
      --kv_cache_transfer_type=Mooncake --kv_cache_transfer_mode=PULL --kv_split_size=1 \
      --etcd_addr="$etcd" --transfer_listen_port="$rank_transfer" \
      --disagg_pd_port="$pd_port" --enable_online_profile=true \
      --profile_backend=torch --profile_dir="$run_dir/profile" \
      >"$run_dir/rank${rank}.log" 2>&1 < /dev/null &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$rank" "${pids[-1]}" "$bin" >> "$run_dir/pids.tsv"
done
date --iso-8601=seconds > "$run_dir/started"
printf 'ROLE=%s RUN=%s MASTER=%s HTTP=%s TRANSFER=%s GRAPH=%s K=6\n' \
  "$role" "$run_dir" "$master" "$http" "$transfer" "$graph"
set +e
wait -n -p finished_pid "${pids[@]}"
wait_status=$?
set -e
{
  date --iso-8601=ns
  printf 'FIRST_CHILD_EXIT_PID=%s\nFIRST_CHILD_EXIT_STATUS=%s\n' "${finished_pid:-unknown}" "$wait_status"
  ps -o pid,ppid,stat,lstart,args -p "$(IFS=,; echo "${pids[*]}")" || true
} > "$run_dir/first_child_exit.txt"
exit "$wait_status"
