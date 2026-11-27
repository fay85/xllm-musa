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
EP_SIZE="${GLM5_EP_SIZE:-8}"
CP_SIZE="${GLM5_CP_SIZE:-8}"
MAX_MEMORY_UTILIZATION="${GLM5_MAX_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS_PER_BATCH="${GLM5_MAX_TOKENS_PER_BATCH:-32768}"
MAX_SEQS_PER_BATCH="${GLM5_MAX_SEQS_PER_BATCH:-16}"
ENABLE_GRAPH="${GLM5_ENABLE_GRAPH:-true}"
KV_CACHE_DTYPE="${GLM5_KV_CACHE_DTYPE:-fp8_e4m3}"
DRAFT_MODEL_PATH="${GLM5_DRAFT_MODEL_PATH:-}"
NUM_SPECULATIVE_TOKENS="${GLM5_NUM_SPECULATIVE_TOKENS:-6}"
SPECULATIVE_ALGORITHM="${GLM5_SPECULATIVE_ALGORITHM:-MTP}"
export XLLM_GLM5_PREFILL_EP="${XLLM_GLM5_PREFILL_EP:-1}"
export XLLM_GLM5_PREFILL_CP="${XLLM_GLM5_PREFILL_CP:-1}"
export GLM5_KV_CACHE_DTYPE="${GLM5_KV_CACHE_DTYPE:-fp8_e4m3}"
export XLLM_MUSA_POOL_COMPUTE_STREAM="${XLLM_MUSA_POOL_COMPUTE_STREAM:-1}"
export XLLM_MUSA_MCCL_EAGER_P2P="${XLLM_MUSA_MCCL_EAGER_P2P:-1}"
PREFILL_GRAPH_MAX_ENTRIES="${GLM5_PREFILL_GRAPH_MAX_ENTRIES:-16}"
ENABLE_CHUNKED_PREFILL="${GLM5_ENABLE_CHUNKED_PREFILL:-true}"
MAX_TOKENS_PER_CHUNK="${GLM5_MAX_TOKENS_PER_CHUNK:-32768}"
PREFILL_BATCH_WAIT_MS="${GLM5_PREFILL_BATCH_WAIT_MS:-0}"
# Graph captures can use every token row admitted by the batch budget.  Set a
# smaller value explicitly when reserving graph memory for short prompts.
MAX_TOKENS_FOR_GRAPH_MODE="${GLM5_MAX_TOKENS_FOR_GRAPH_MODE:-${MAX_TOKENS_PER_BATCH}}"
ENABLE_PREFILL_PIECEWISE_GRAPH="${GLM5_ENABLE_PREFILL_PIECEWISE_GRAPH:-true}"
ENABLE_PREFIX_CACHE="${GLM5_ENABLE_PREFIX_CACHE:-true}"
ENABLE_SCHEDULE_OVERLAP="${GLM5_ENABLE_SCHEDULE_OVERLAP:-false}"
# Leave launch blocking unset by default, including for Torch profiling.
# MUSA_LAUNCH_BLOCKING=1 on the service command opts into synchronous debugging.
if [[ ${MUSA_LAUNCH_BLOCKING:-0} == 0 ]]; then
  unset MUSA_LAUNCH_BLOCKING
elif [[ $MUSA_LAUNCH_BLOCKING == 1 ]]; then
  ENABLE_GRAPH=false
  ENABLE_PREFILL_PIECEWISE_GRAPH=false
  ENABLE_SCHEDULE_OVERLAP=false
fi
ENABLE_DISAGG_PD="${GLM5_ENABLE_DISAGG_PD:-false}"
INSTANCE_ROLE="${GLM5_INSTANCE_ROLE:-DEFAULT}"
KV_CACHE_TRANSFER_MODE="${GLM5_KV_CACHE_TRANSFER_MODE:-PUSH}"
TRANSFER_LISTEN_PORT="${GLM5_TRANSFER_LISTEN_PORT:-26000}"
ETCD_ADDR="${GLM5_ETCD_ADDR:-}"
LOG_DIR="${GLM5_LOG_DIR:-/tmp/glm5_3_flash_run}"
WAIT_READY="${GLM5_WAIT_READY:-1}"
WAIT_TIMEOUT_SEC="${GLM5_WAIT_TIMEOUT_SEC:-180}"
TORCH_PROFILE="${GLM5_TORCH_PROFILE:-0}"
PROFILE_DIR="${GLM5_PROFILE_DIR:-${LOG_DIR}/traces}"

export FLASHINFER_OPS_PATH="${GLM5_FLASHINFER_OPS_PATH:-${REPO_ROOT}/build/mate_sparse_ops}"
OPS_BACKUP_DIR="${GLM5_OPS_BACKUP_DIR:-/opt/mate027_cache/glm5_3_mate_sparse_ops}"

ensure_sparse_ops_cache() {
  local required=(
    "prepare_kda_dtype_bf16_gate_1_varlen_1_normqk_1_seqlens_i64_arch_mp31/prepare_kda_dtype_bf16_gate_1_varlen_1_normqk_1_seqlens_i64_arch_mp31.so"
    "sparse_mla_bf16_v1_mp31_h8/sparse_mla_bf16_v1_mp31_h8.so"
    "sparse_mla_nope_fp8_v1_mp31_h8_splits64/sparse_mla_nope_fp8_v1_mp31_h8_splits64.so"
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
  if (( NUM_SPECULATIVE_TOKENS > 0 )); then
    local heads=$((64 / NNODES))
    local verify_module="glm5_kda_verify_bf16_h${heads}"
    if [[ ! -f "${FLASHINFER_OPS_PATH}/${verify_module}/${verify_module}.so" ]]; then
      MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 \
        python3 "${SCRIPT_DIR}/compile_musa_kda.py" --verify-heads "${heads}" --ops-path "${FLASHINFER_OPS_PATH}"
    fi
  fi
  if [[ ! -f "${FLASHINFER_OPS_PATH}/glm_kpool_nonpaged_v1/glm_kpool_nonpaged_v1.so" ]]; then
    MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 \
      python3 "${SCRIPT_DIR}/compile_musa_sparse_mla.py" --indexer-only --ops-path "${FLASHINFER_OPS_PATH}"
  fi
  local module
  for module in moe_fused_gate mate_mega_moe_runtime_utils_layout3 glm53_mega_c4096_v2_{stage{1..3},decode{1..2}}; do
    if [[ ! -f "${FLASHINFER_OPS_PATH}/${module}/${module}.so" ]]; then
      MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 \
        python3 "${SCRIPT_DIR}/compile_musa_mega_moe.py" --ops-path "${FLASHINFER_OPS_PATH}"
      break
    fi
  done
  # Older cache backups predate the MATE mHC finish modules. Check these even
  # when the sparse operator cache was already complete or restored above.
  local name
  local mhc_modules=(mhc_pre_big_fuse_glm53_{eps6,eps5}{,_small}_v1)
  for name in "${mhc_modules[@]}"; do
    if [[ ! -f "${FLASHINFER_OPS_PATH}/${name}/${name}.so" ]]; then
      MATE_WORKSPACE_BASE=/opt/mate027_cache MATE_MUSA_ARCH_LIST=3.1 python3 \
        "${SCRIPT_DIR}/compile_musa_mhc.py" --ops-path "${FLASHINFER_OPS_PATH}"
      break
    fi
  done
  for name in "${mhc_modules[@]}"; do
    [[ -f "${FLASHINFER_OPS_PATH}/${name}/${name}.so" ]] || {
      echo "Missing MATE mHC module: ${name}" >&2
      return 1
    }
  done
}

PID_FILE="${LOG_DIR}/pids.txt"

stop_service() {
  if [[ -f "${PID_FILE}" || "$ENABLE_DISAGG_PD" == true ]]; then
    mkdir -p "$LOG_DIR"
    echo "Stopping GLM-5.3-Flash workers from ${PID_FILE}..."
    if [[ "$ENABLE_DISAGG_PD" == true ]]; then
      python3 - "$LOG_DIR" "$INSTANCE_ROLE" <<'PY_STOP'
import os
from pathlib import Path
import signal
import sys
import time

log_dir = Path(sys.argv[1]).resolve()
role = f"--instance_role={sys.argv[2]}".encode()

def is_rank(pid: int) -> bool:
    proc = Path(f"/proc/{pid}")
    try:
        cmd = (proc / "cmdline").read_bytes().split(b"\0")
        if role not in cmd or b"--enable_disagg_pd=true" not in cmd:
            return False
        log = (proc / "fd/2").resolve()
        return log.parent == log_dir and log.match("rank_*.log")
    except (OSError, RuntimeError):
        return False

pids = [int(name) for name in os.listdir("/proc") if name.isdigit() and is_rank(int(name))]
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
for _ in range(100):
    if not any(is_rank(pid) for pid in pids):
        break
    time.sleep(0.1)
else:
    sys.exit(f"Workers still active: {pids}")
print(f"Stopped {len(pids)} local {sys.argv[2]} ranks")
PY_STOP
    else
      xargs -r kill -9 < "${PID_FILE}" 2>/dev/null || true
    fi
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

if [[ "${XLLM_GLM5_PREFILL_EP}" != "0" && "${EP_SIZE}" != "8" ]]; then
  echo "Error: XLLM_GLM5_PREFILL_EP requires GLM5_EP_SIZE=8, got ${EP_SIZE}." >&2
  exit 1
fi
if [[ "${XLLM_GLM5_PREFILL_CP}" != "0" && "${CP_SIZE}" != "8" ]]; then
  echo "Error: XLLM_GLM5_PREFILL_CP requires GLM5_CP_SIZE=8, got ${CP_SIZE}." >&2
  exit 1
fi
if [[ "${EP_SIZE}" != "1" && "${NNODES}" != "8" ]]; then
  echo "Error: GLM5_EP_SIZE=${EP_SIZE} currently requires GLM5_NNODES=8." >&2
  exit 1
fi
if [[ "${CP_SIZE}" != "1" && "${NNODES}" != "8" ]]; then
  echo "Error: GLM5_CP_SIZE=${CP_SIZE} currently requires GLM5_NNODES=8." >&2
  exit 1
fi
if [[ "${CP_SIZE}" != "1" && "${ENABLE_GRAPH}" == "true" ]]; then
  if [[ "${ENABLE_CHUNKED_PREFILL}" != "true" ]]; then
    echo "Error: graph CP prefill requires GLM5_ENABLE_CHUNKED_PREFILL=true." >&2
    exit 1
  fi
  if [[ "${MAX_TOKENS_PER_CHUNK}" -lt 1 ||
        "${MAX_TOKENS_PER_CHUNK}" -gt "${MAX_TOKENS_FOR_GRAPH_MODE}" ]]; then
    echo "Error: graph CP prefill requires 1 <= GLM5_MAX_TOKENS_PER_CHUNK <= ${MAX_TOKENS_FOR_GRAPH_MODE}." >&2
    exit 1
  fi
fi

ensure_ops_cache
mkdir -p "${LOG_DIR}"
stop_service

# Release all ranks if any worker fails during startup.
cleanup_failed_start() {
  local status=$?
  if [[ "${status}" != "0" ]]; then
    stop_service
  fi
}
trap cleanup_failed_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PD_ARGS=()
if [[ "$ENABLE_DISAGG_PD" == true ]]; then
  PD_ARGS=(--disagg_pd_port="${GLM5_PD_RPC_PORT:-7777}" --enable_native_service_adapter=true
    --vmodule=disagg_pd_scheduler=1,mooncake_kv_cache_transfer=1 --usercode_in_pthread=true)
fi
if [[ $INSTANCE_ROLE == DECODE && $ENABLE_GRAPH == true ]]; then
  PD_ARGS+=(--enable_graph_vmm_pool=false)
fi
SPECULATIVE_ARGS=()
if [[ "${NUM_SPECULATIVE_TOKENS}" -gt 0 ]]; then
  [[ -n "${DRAFT_MODEL_PATH}" ]] || DRAFT_MODEL_PATH="${MODEL_PATH}"
  SPECULATIVE_ARGS=(--draft_model="${DRAFT_MODEL_PATH}"
    --num_speculative_tokens="${NUM_SPECULATIVE_TOKENS}"
    --speculative_algorithm="${SPECULATIVE_ALGORITHM}")
  if [[ ${INSTANCE_ROLE} == DECODE ]]; then
    # GLM-5.3 TP8 vocab shards have 19360 rows, short of a full FP8 tile.
    SPECULATIVE_ARGS+=(--enable_mtp_draft_fp8_lm_head=false
      --enable_mtp_target_fp8_lm_head=false)
  fi
fi

PROFILE_ARGS=()
if [[ "${TORCH_PROFILE}" == "1" ]]; then
  mkdir -p "${PROFILE_DIR}"
  PROFILE_ARGS=(
    "--enable_online_profile=true"
    "--profile_backend=torch"
    "--profile_dir=${PROFILE_DIR}"
  )
fi

echo "Binary: ${XLLM_BIN}"
echo "MUSA_LAUNCH_BLOCKING=${MUSA_LAUNCH_BLOCKING:-unset} graph=${ENABLE_GRAPH} prefill_piecewise_graph=${ENABLE_PREFILL_PIECEWISE_GRAPH} schedule_overlap=${ENABLE_SCHEDULE_OVERLAP}"
echo "Starting ${MODEL_ID} (${NNODES} ranks, backend=${BACKEND}) from ${MODEL_PATH}..."
if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  echo "Speculative decoding: algorithm=${SPECULATIVE_ALGORITHM} draft_tokens=${NUM_SPECULATIVE_TOKENS} draft_model=${DRAFT_MODEL_PATH}"
fi
echo "Parallel contract: ep_size=${EP_SIZE} cp_size=${CP_SIZE} kv_split_size=1 prefill_ep=${XLLM_GLM5_PREFILL_EP} prefill_cp=${XLLM_GLM5_PREFILL_CP} disagg_pd=${ENABLE_DISAGG_PD} instance_role=${INSTANCE_ROLE}"
echo "Logs: ${LOG_DIR}/rank_*.log"
if [[ "${TORCH_PROFILE}" == "1" ]]; then
  echo "Torch profile dir: ${PROFILE_DIR}"
fi

for ((r = 0; r < NNODES; ++r)); do
  RANK_PORT=$((BASE_PORT + r))
  RANK_TRANSFER_PORT=$((TRANSFER_LISTEN_PORT + r))
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
    --ep_size="${EP_SIZE}" \
    --cp_size="${CP_SIZE}" \
    --kv_split_size=1 \
    --master_node_addr="${MASTER_ADDR}" \
    --communication_backend=mccl \
    --max_memory_utilization="${MAX_MEMORY_UTILIZATION}" \
    --max_tokens_per_batch="${MAX_TOKENS_PER_BATCH}" \
    --max_seqs_per_batch="${MAX_SEQS_PER_BATCH}" \
    --reasoning_parser=glm5 \
    --enable_prefix_cache="${ENABLE_PREFIX_CACHE}" \
    --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    --max_tokens_per_chunk_for_prefill="${MAX_TOKENS_PER_CHUNK}" \
    --enable_schedule_overlap="${ENABLE_SCHEDULE_OVERLAP}" \
    --prefill_batch_wait_ms="${PREFILL_BATCH_WAIT_MS}" \
    --enable_graph="${ENABLE_GRAPH}" \
    --kv_cache_dtype="${KV_CACHE_DTYPE}" \
    --max_tokens_for_graph_mode="${MAX_TOKENS_FOR_GRAPH_MODE}" \
    --enable_prefill_piecewise_graph="${ENABLE_PREFILL_PIECEWISE_GRAPH}" \
    --musa_prefill_graph_max_entries="${PREFILL_GRAPH_MAX_ENTRIES}" \
    --enable_disagg_pd="${ENABLE_DISAGG_PD}" \
    --instance_role="${INSTANCE_ROLE}" \
    --kv_cache_transfer_mode="${KV_CACHE_TRANSFER_MODE}" \
    --transfer_listen_port="${RANK_TRANSFER_PORT}" \
    --etcd_addr="${ETCD_ADDR}" \
    --enable_shm=false \
    "${PD_ARGS[@]}" "${SPECULATIVE_ARGS[@]}" "${PROFILE_ARGS[@]}" \
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
  if grep -q -E "Check failed|SIGSEGV|SIGABRT|out of memory|terminate called" "${LOG_DIR}"/rank_*.log 2>/dev/null; then
    echo "Error: worker crash detected during startup!" >&2
    grep -n -C 5 -E "Check failed|SIGSEGV|SIGABRT|out of memory|terminate called" "${LOG_DIR}"/rank_*.log >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Error: timed out after ${WAIT_TIMEOUT_SEC}s waiting for service startup. Check ${LOG_DIR}/rank_0.log" >&2
exit 1
