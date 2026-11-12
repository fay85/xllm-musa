#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-start}"

XLLM_BIN="${GLM5_XLLM_BIN:-${REPO_ROOT}/build/lib.linux-x86_64-cpython-310/xllm/xllm}"
MODEL_PATH="${GLM5_MODEL_PATH:-/data/nfs_shared/models/GLM-5.3-Flash}"
MODEL_ID="${GLM5_MODEL_ID:-glm-5.3-flash}"
BACKEND="${GLM5_BACKEND:-vlm}"
HOST="${GLM5_HOST:-127.0.0.1}"
BASE_PORT="${GLM5_PORT:-18901}"
MASTER_ADDR="${GLM5_MASTER_ADDR:-127.0.0.1:29888}"
NNODES="${GLM5_NNODES:-8}"
MAX_MEMORY_UTILIZATION="${GLM5_MAX_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS_PER_BATCH="${GLM5_MAX_TOKENS_PER_BATCH:-4096}"
MAX_SEQS_PER_BATCH="${GLM5_MAX_SEQS_PER_BATCH:-16}"
ENABLE_GRAPH="${GLM5_ENABLE_GRAPH:-true}"
KV_CACHE_DTYPE="${GLM5_KV_CACHE_DTYPE:-auto}"
# GLM MegaMoE is enabled by default; set 0 to select the original TP experts.
export XLLM_GLM5_MEGA_MOE="${XLLM_GLM5_MEGA_MOE:-1}"
PREFILL_GRAPH_MAX_ENTRIES="${GLM5_PREFILL_GRAPH_MAX_ENTRIES:-8}"
ENABLE_CHUNKED_PREFILL="${GLM5_ENABLE_CHUNKED_PREFILL:-false}"
MAX_TOKENS_PER_CHUNK="${GLM5_MAX_TOKENS_PER_CHUNK:--1}"
PREFILL_BATCH_WAIT_MS="${GLM5_PREFILL_BATCH_WAIT_MS:-0}"
# Graph capture uses exact GLM5 token shapes for prefill and exact decode batch
# shapes.  The executor enforces model-specific safety checks for unsupported
# shapes; this limit only prevents unexpectedly large graph allocations.
MAX_TOKENS_FOR_GRAPH_MODE="${GLM5_MAX_TOKENS_FOR_GRAPH_MODE:-4096}"
ENABLE_PREFILL_PIECEWISE_GRAPH="${GLM5_ENABLE_PREFILL_PIECEWISE_GRAPH:-true}"
LOG_DIR="${GLM5_LOG_DIR:-/tmp/glm5_3_flash_run}"
WAIT_READY="${GLM5_WAIT_READY:-1}"
WAIT_TIMEOUT_SEC="${GLM5_WAIT_TIMEOUT_SEC:-180}"
TORCH_PROFILE="${GLM5_TORCH_PROFILE:-0}"
PROFILE_DIR="${GLM5_PROFILE_DIR:-${LOG_DIR}/traces}"
PROFILE_CALLSTACK="${GLM5_PROFILE_CALLSTACK:-0}"
CALLSTACK_PRELOAD_SO="${GLM5_CALLSTACK_PRELOAD_SO:-}"

export FLASHINFER_OPS_PATH="${GLM5_FLASHINFER_OPS_PATH:-${REPO_ROOT}/build/mate_sparse_ops}"
OPS_BACKUP_DIR="${GLM5_OPS_BACKUP_DIR:-/opt/mate027_cache/glm5_3_mate_sparse_ops}"

ensure_sparse_ops_cache() {
  local required=(
    "prepare_kda_dtype_bf16_gate_1_varlen_1_normqk_1_seqlens_i64_arch_mp31/prepare_kda_dtype_bf16_gate_1_varlen_1_normqk_1_seqlens_i64_arch_mp31.so"
    "sparse_mla_bf16_v1_mp31_h8/sparse_mla_bf16_v1_mp31_h8.so"
    "sparse_mla_nope_fp8_v1_mp31_h8_splits64/sparse_mla_nope_fp8_v1_mp31_h8_splits64.so"
    "glm_kpool_score_v1/glm_kpool_score_v1.so"
    "gemm_ops/gemm_ops.so"
  )
  local missing=0
  for rel in "${required[@]}"; do
    if [[ ! -f "${FLASHINFER_OPS_PATH}/${rel}" ]]; then
      missing=1
      break
    fi
  done
  [[ "${missing}" == "0" ]] && return 0

  echo "Populating missing MUSA operator cache at ${FLASHINFER_OPS_PATH}..."
  mkdir -p "${FLASHINFER_OPS_PATH}"
  if [[ -d "${OPS_BACKUP_DIR}" ]]; then
    cp -rf "${OPS_BACKUP_DIR}"/* "${FLASHINFER_OPS_PATH}/"
    return 0
  fi
  [[ -d /opt/mate027_cache/.cache/mate/0.2.7/mp31/cached_ops ]] && cp -rn /opt/mate027_cache/.cache/mate/0.2.7/mp31/cached_ops/* "${FLASHINFER_OPS_PATH}/" 2>/dev/null || true
  [[ -d /opt/mate027_cache/.cache/mate/0.2.8/mp31/cached_ops ]] && cp -rf /opt/mate027_cache/.cache/mate/0.2.8/mp31/cached_ops/* "${FLASHINFER_OPS_PATH}/" 2>/dev/null || true
  [[ -d /opt/mate027_cache/mp31/cached_ops_jd_k8d ]] && cp -rf /opt/mate027_cache/mp31/cached_ops_jd_k8d/* "${FLASHINFER_OPS_PATH}/" 2>/dev/null || true
  MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 python3 "${SCRIPT_DIR}/compile_musa_kda.py" --ops-path "${FLASHINFER_OPS_PATH}" --jobs 4
  MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 python3 "${SCRIPT_DIR}/compile_musa_mhc.py" --ops-path "${FLASHINFER_OPS_PATH}"
  MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 python3 "${SCRIPT_DIR}/compile_musa_sparse_mla.py" --ops-path "${FLASHINFER_OPS_PATH}"
}

ensure_ops_cache() {
  ensure_sparse_ops_cache
  if [[ "${XLLM_GLM5_MEGA_MOE}" != "0" ]]; then
    MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 \
      python3 "${SCRIPT_DIR}/compile_musa_mega_moe.py" --ops-path "${FLASHINFER_OPS_PATH}"
  fi
  # Older cache backups predate the MATE mHC finish modules. Check these even
  # when the sparse operator cache was already complete or restored above.
  local name
  for name in mhc_pre_big_fuse_glm53_eps6_v1 mhc_pre_big_fuse_glm53_eps5_v1; do
    if [[ ! -f "${FLASHINFER_OPS_PATH}/${name}/${name}.so" ]]; then
      MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 python3 \
        "${SCRIPT_DIR}/compile_musa_mhc.py" --ops-path "${FLASHINFER_OPS_PATH}"
      break
    fi
  done
  for name in mhc_pre_big_fuse_glm53_eps6_v1 mhc_pre_big_fuse_glm53_eps5_v1; do
    [[ -f "${FLASHINFER_OPS_PATH}/${name}/${name}.so" ]] || {
      echo "Missing MATE mHC module: ${name}" >&2
      return 1
    }
  done
}

PID_FILE="${LOG_DIR}/pids.txt"

stop_service() {
  if [[ -f "${PID_FILE}" ]]; then
    echo "Stopping GLM-5.3-Flash workers from ${PID_FILE}..."
    xargs -r kill -9 < "${PID_FILE}" 2>/dev/null || true
    : > "${PID_FILE}"
  else
    echo "No PID file found at ${PID_FILE}."
  fi
}

status_service() {
  if [[ ! -f "${PID_FILE}" ]] || [[ ! -s "${PID_FILE}" ]]; then
    echo "Service is not running (no active PIDs in ${PID_FILE})."
    return 1
  fi
  echo "Recorded PIDs in ${PID_FILE}:"
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  PID ${pid}: RUNNING"
    else
      echo "  PID ${pid}: STOPPED"
    fi
  done < "${PID_FILE}"
}

case "${ACTION}" in
  stop)
    stop_service
    exit 0
    ;;
  status)
    status_service
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac

if [[ ! -x "${XLLM_BIN}" ]]; then
  echo "Error: xllm binary not found or not executable: ${XLLM_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Error: model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

ensure_ops_cache
mkdir -p "${LOG_DIR}"
stop_service

PROFILE_ARGS=()
if [[ "${TORCH_PROFILE}" == "1" ]]; then
  mkdir -p "${PROFILE_DIR}"
  PROFILE_ARGS=(
    "--enable_online_profile=true"
    "--profile_backend=torch"
    "--profile_dir=${PROFILE_DIR}"
  )
  if [[ "${PROFILE_CALLSTACK}" == "1" ]] && [[ -n "${CALLSTACK_PRELOAD_SO}" ]] && [[ -f "${CALLSTACK_PRELOAD_SO}" ]]; then
    export LD_PRELOAD="${CALLSTACK_PRELOAD_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi

echo "Binary: ${XLLM_BIN}"
echo "Starting ${MODEL_ID} (${NNODES} ranks, backend=${BACKEND}) from ${MODEL_PATH}..."
echo "Logs: ${LOG_DIR}/rank_*.log"
if [[ "${TORCH_PROFILE}" == "1" ]]; then
  echo "Torch profile dir: ${PROFILE_DIR}"
  echo "LD_PRELOAD: ${LD_PRELOAD:-<none>}"
fi

for ((r = 0; r < NNODES; ++r)); do
  RANK_PORT=$((BASE_PORT + r))
  RANK_LOG="${LOG_DIR}/rank_${r}.log"
  : > "${RANK_LOG}"
  setsid "${XLLM_BIN}" \
    --model="${MODEL_PATH}" \
    --model_id="${MODEL_ID}" \
    --backend="${BACKEND}" \
    --task=generate \
    --host="${HOST}" \
    --port="${RANK_PORT}" \
    --nnodes="${NNODES}" \
    --node_rank="${r}" \
    --master_node_addr="${MASTER_ADDR}" \
    --communication_backend=mccl \
    --max_memory_utilization="${MAX_MEMORY_UTILIZATION}" \
    --max_tokens_per_batch="${MAX_TOKENS_PER_BATCH}" \
    --max_seqs_per_batch="${MAX_SEQS_PER_BATCH}" \
    --reasoning_parser=glm5 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    --max_tokens_per_chunk_for_prefill="${MAX_TOKENS_PER_CHUNK}" \
    --enable_schedule_overlap=false \
    --prefill_batch_wait_ms="${PREFILL_BATCH_WAIT_MS}" \
    --enable_graph="${ENABLE_GRAPH}" \
    --kv_cache_dtype="${KV_CACHE_DTYPE}" \
    --max_tokens_for_graph_mode="${MAX_TOKENS_FOR_GRAPH_MODE}" \
    --enable_prefill_piecewise_graph="${ENABLE_PREFILL_PIECEWISE_GRAPH}" \
    --musa_prefill_graph_max_entries="${PREFILL_GRAPH_MAX_ENTRIES}" \
    --enable_shm=false \
    "${PROFILE_ARGS[@]}" \
    > "${RANK_LOG}" 2>&1 < /dev/null &
  echo $! >> "${PID_FILE}"
done

echo "Launched PIDs: $(tr '\n' ' ' < "${PID_FILE}")"

if [[ "${WAIT_READY}" != "1" ]]; then
  exit 0
fi

echo "Waiting up to ${WAIT_TIMEOUT_SEC}s for HTTP server on http://${HOST}:${BASE_PORT} ..."
for ((elapsed = 1; elapsed <= WAIT_TIMEOUT_SEC; ++elapsed)); do
  if grep -q "Application startup complete" "${LOG_DIR}/rank_0.log" 2>/dev/null; then
    echo "Service ready after ${elapsed}s: http://${HOST}:${BASE_PORT}"
    exit 0
  fi
  if grep -q -E "Check failed|SIGSEGV|SIGABRT" "${LOG_DIR}"/rank_*.log 2>/dev/null; then
    echo "Error: worker crash detected during startup!" >&2
    grep -n -C 5 -E "Check failed|SIGSEGV|SIGABRT" "${LOG_DIR}"/rank_*.log >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Error: timed out after ${WAIT_TIMEOUT_SEC}s waiting for service startup. Check ${LOG_DIR}/rank_0.log" >&2
exit 1
