#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/kill_zombie_xllm.sh"

TRACE_DECODE_STEPS="${TRACE_DECODE_STEPS:-32}"
WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-1}"
MAX_TOKENS=$((WARMUP_DECODE_STEPS + TRACE_DECODE_STEPS + 2))

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/profile_results/decode_eager_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b) DEFAULT_MAX_MEMORY_UTILIZATION=0.90 ;;
  *) DEFAULT_MAX_MEMORY_UTILIZATION=0.70 ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
export MAX_CONCURRENT_REQUESTS=1
export ENABLE_GRAPH="${ENABLE_GRAPH:-0}"
export ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-false}"
export NUM_SPECULATIVE_TOKENS=0
export PORT="${PORT:-8093}"
export MASTER_PORT="${MASTER_PORT:-9749}"

export XLLM_ENABLE_TORCH_KINETO_PROFILE="${XLLM_ENABLE_TORCH_KINETO_PROFILE:-1}"
export XLLM_ENABLE_KINETO_TRACE="${XLLM_ENABLE_KINETO_TRACE:-0}"
export XLLM_KINETO_WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS}"
export XLLM_KINETO_TRACE_DECODE_STEPS="${TRACE_DECODE_STEPS}"
export XLLM_TORCH_KINETO_TRACE_PATH="${RESULTS_DIR}/decode_eager.pt.trace.json"
export XLLM_KINETO_TRACE_PATH="${RESULTS_DIR}/decode_eager.libkineto.trace.json"
export XLLM_KINETO_SUMMARY_PATH="${RESULTS_DIR}/decode_eager_summary.txt"

LOG="${RESULTS_DIR}/profile.log"
{
echo "############################################################"
echo "# Decode-only eager profile (XllmKinetoProfiler)"
echo "# date               : $(date)"
echo "# ENABLE_GRAPH       : ${ENABLE_GRAPH}"
echo "# WARMUP_DECODE_STEPS: ${WARMUP_DECODE_STEPS}"
echo "# TRACE_DECODE_STEPS : ${TRACE_DECODE_STEPS}"
echo "# request max_tokens : ${MAX_TOKENS}"
echo "############################################################"
} | tee "$LOG"

kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || true
bash run_xllm_musa.sh --background --port "$PORT" --device 0 >>"$LOG" 2>&1
SERVER_LOG="${LOG_DIR:-log}/xllm_${MODEL_NAME}.log"
w=0
until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null \
      || curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  sleep 5; w=$((w+5))
  [ "$w" -ge 600 ] && { echo "FAIL: server not ready" | tee -a "$LOG"; exit 1; }
done
echo "==> server ready after ${w}s" | tee -a "$LOG"

URL="http://127.0.0.1:${PORT}/v1/chat/completions"
curl -s -m 600 -X POST "$URL" -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly.\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0,\"ignore_eos\":true}" \
  | tee "${RESULTS_DIR}/response.json" >>"$LOG"

if [[ "${XLLM_ENABLE_TORCH_KINETO_PROFILE}" == "1" ]]; then
  TRACE_PATH="${XLLM_TORCH_KINETO_TRACE_PATH}"
else
  TRACE_PATH="${XLLM_KINETO_TRACE_PATH}"
fi
for _ in $(seq 1 90); do
  [[ -f "${TRACE_PATH}" ]] && break
  sleep 2
done
[[ -f "${TRACE_PATH}" ]] || { echo FAIL no trace; grep -i kineto "$SERVER_LOG" | tail -20; exit 1; }

cp "${TRACE_PATH}" "${RESULTS_DIR}/decode_eager.perfetto.json"
python3 - <<PY | tee -a "$LOG"
import json
from collections import Counter, defaultdict
path = "${RESULTS_DIR}/decode_eager.perfetto.json"
with open(path) as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])
aten=Counter(); dur=defaultdict(float); scopes=Counter()
for e in events:
    if not isinstance(e,dict) or e.get("ph")!="X": continue
    name=e.get("name",""); cat=e.get("cat",""); d=e.get("dur",0)
    if cat=="cpu_op" and name.startswith("aten::"):
        aten[name]+=1; dur[name]+=d
    if "xllm" in name or "decode" in name.lower() or cat=="user_annotation":
        scopes[name]+=1
print("trace_events",len(events))
print("=== user scopes ===")
for n,c in scopes.most_common(20): print(c,n)
print("=== top aten by dur ms ===")
for name,d in sorted(dur.items(), key=lambda x:-x[1])[:20]:
    print(f"{d/1e3:.1f}ms {aten[name]}x {name}")
PY
kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || true
echo "PERFETTO=${RESULTS_DIR}/decode_eager.perfetto.json"
echo "PROFILE_DECODE_EAGER_DONE"
