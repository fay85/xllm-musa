#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-start}"

XLLM_BIN="${GLM5_XLLM_BIN:-${REPO_ROOT}/build/lib.linux-x86_64-cpython-310/xllm/xllm}"
MODEL_PATH="${GLM5_MODEL_PATH:-/data/nfs_shared/models/GLM-5.3-Flash}"
MODEL_ID="${GLM5_MODEL_ID:-glm-5.3-flash}"
HOST="${GLM5_HOST:-127.0.0.1}"
BASE_PORT="${GLM5_PORT:-18901}"
MASTER_ADDR="${GLM5_MASTER_ADDR:-127.0.0.1:29888}"
NNODES="${GLM5_NNODES:-8}"
MAX_MEMORY_UTILIZATION="${GLM5_MAX_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS_PER_BATCH="${GLM5_MAX_TOKENS_PER_BATCH:-4096}"
MAX_SEQS_PER_BATCH="${GLM5_MAX_SEQS_PER_BATCH:-16}"
LOG_DIR="${GLM5_LOG_DIR:-/tmp/glm5_3_flash_run}"
WAIT_READY="${GLM5_WAIT_READY:-1}"
WAIT_TIMEOUT_SEC="${GLM5_WAIT_TIMEOUT_SEC:-180}"

export FLASHINFER_OPS_PATH="${GLM5_FLASHINFER_OPS_PATH:-${REPO_ROOT}/build/mate_sparse_ops}"

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

mkdir -p "${LOG_DIR}"
stop_service

echo "Binary: ${XLLM_BIN}"
echo "Starting ${MODEL_ID} (${NNODES} ranks) from ${MODEL_PATH}..."
echo "Logs: ${LOG_DIR}/rank_*.log"

for ((r = 0; r < NNODES; ++r)); do
  RANK_PORT=$((BASE_PORT + r))
  RANK_LOG="${LOG_DIR}/rank_${r}.log"
  : > "${RANK_LOG}"
  setsid "${XLLM_BIN}" \
    --model="${MODEL_PATH}" \
    --model_id="${MODEL_ID}" \
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
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_graph=false \
    --enable_shm=false \
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
