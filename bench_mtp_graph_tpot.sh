#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/kill_zombie_xllm.sh"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/mtp_bench_${RUN_TS}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
export FLASHINFER_OPS_PATH="${FLASHINFER_OPS_PATH:-/workspace/mate_cached_ops}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b) DEFAULT_MAX_MEMORY_UTILIZATION=0.90 ;;
  *) DEFAULT_MAX_MEMORY_UTILIZATION=0.70 ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-1}"
export ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-true}"
export IGNORE_EOS="${IGNORE_EOS:-1}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/model_weights/Qwen3.5-27B-mtp}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-1}"
export SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-MTP}"
PORT_BASE="${PORT_BASE:-8200}"
MASTER_BASE="${MASTER_BASE:-9800}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
run_xllm_correctness() {
  local tag="$1" port="$2" master="$3"; shift 3
  local logf="$LOG_DIR/${tag}_correctness.log"
  log "=== CORRECTNESS $tag ==="
  kill_zombie_xllm "$port" "$master" 2>/dev/null || true; sleep 2
  if ! env "$@" PORT="$port" MASTER_PORT="$master" START_SERVER=1 STOP_SERVER=1 MAXTOK=256 \
    bash correctness_check.sh >"$logf" 2>&1; then
    log "FAIL correctness $tag"; tail -20 "$logf" | tee -a "$SUMMARY"; return 1
  fi
  if grep -q "CORRECTNESS: PASS" "$logf"; then
    log "PASS correctness $tag"; grep -E "391|CORRECTNESS" "$logf" | tail -3 | tee -a "$SUMMARY"; return 0
  fi
  log "FAIL correctness $tag (no PASS)"; tail -20 "$logf" | tee -a "$SUMMARY"; return 1
}
run_xllm_tpot() {
  local tag="$1" port="$2" master="$3"; shift 3
  local logf="$LOG_DIR/${tag}_tpot.log"
  log "=== TPOT $tag (ISL=$INPUT_LEN OSL=$OUTPUT_LEN C=1) ==="
  kill_zombie_xllm "$port" "$master" 2>/dev/null || true; sleep 2
  if ! env "$@" PORT="$port" MASTER_PORT="$master" INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" CONCURRENCY_LEVELS="1" \
    bash conc_eval.sh >"$logf" 2>&1; then
    log "FAIL tpot $tag"; tail -30 "$logf" | tee -a "$SUMMARY"; return 1
  fi
  local line; line="$(grep -E ">>> C=1:" "$logf" | tail -1 || true)"
  if [[ -n "$line" ]]; then
    log "RESULT $tag: $line"
    python3 - <<PY | tee -a "$SUMMARY"
import re
line = """$line"""
m = re.search(r"per-req-avg=([0-9.]+) tok/s", line)
if m:
    tps = float(m.group(1)); print(f"  approx_Tpot_ms={1000.0/tps:.2f}  decode_tok_s={tps:.2f}")
PY
    return 0
  fi
  log "FAIL tpot $tag (no result)"; tail -30 "$logf" | tee -a "$SUMMARY"; return 1
}
log "MTP benchmark -> $LOG_DIR"
run_xllm_correctness "mtp_fused_graph" "$PORT_BASE" "$MASTER_BASE" XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_correctness "mtp_fused_eager" "$((PORT_BASE+1))" "$((MASTER_BASE+1))" XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_tpot "mtp_fused_eager" "$((PORT_BASE+2))" "$((MASTER_BASE+2))" XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_tpot "mtp_host_eager" "$((PORT_BASE+3))" "$((MASTER_BASE+3))" XLLM_MATE_GDN_MTP=0 ENABLE_GRAPH=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_tpot "mtp_fused_graph" "$((PORT_BASE+4))" "$((MASTER_BASE+4))" XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_tpot "mtp_host_graph" "$((PORT_BASE+5))" "$((MASTER_BASE+5))" XLLM_MATE_GDN_MTP=0 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" || true
run_xllm_tpot "no_mtp_graph" "$((PORT_BASE+6))" "$((MASTER_BASE+6))" XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 NUM_SPECULATIVE_TOKENS=0 || true
log "=== DONE $SUMMARY ==="; echo "BENCH_DONE $LOG_DIR"
