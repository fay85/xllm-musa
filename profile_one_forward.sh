#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/kill_zombie_xllm.sh"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/profile_results/one_forward_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b) DEFAULT_MAX_MEMORY_UTILIZATION=0.90 ;;
  *) DEFAULT_MAX_MEMORY_UTILIZATION=0.70 ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
export MAX_CONCURRENT_REQUESTS=1
export ENABLE_GRAPH="${ENABLE_GRAPH:-0}"
export ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-false}"
export PORT="${PORT:-8093}"
export MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9749}"
export ENABLE_PROFILE=1
export PROFILE_DIR="$RESULTS_DIR"
export NUM_SPECULATIVE_TOKENS=0
MASTER_PORT="${MASTER_PORT:-9749}"

kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || true
echo "==> RESULTS_DIR=$RESULTS_DIR"
bash run_xllm_musa.sh --background --port "$PORT" --device 0

URL="http://127.0.0.1:${PORT}/v1/chat/completions"
readyprobe() {
  curl -s -m 60 -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null | grep -q completion_tokens
}
w=0
until readyprobe; do sleep 5; w=$((w+5)); [ "$w" -ge 600 ] && { echo FAIL ready; exit 1; }; done
echo "==> server ready ${w}s"

curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/start_profile"
echo
curl -s -m 300 -X POST "$URL" -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Reply with only the number.\"}],\"max_tokens\":1,\"temperature\":0}" \
  | tee "${RESULTS_DIR}/response.json"
echo
sleep 1
curl -s -m 120 -X POST "http://127.0.0.1:${PORT}/stop_profile"
echo
sleep 2
ls -lah "$RESULTS_DIR"
TRACE=""
for f in "$RESULTS_DIR"/*.pt.trace.json.gz "$RESULTS_DIR"/*.pt.trace.json; do [ -f "$f" ] && TRACE="$f" && break; done
PERFETTO="${RESULTS_DIR}/one_forward.perfetto.json"
if [ -z "$TRACE" ]; then echo "FAIL no trace"; exit 1; fi
if [[ "$TRACE" == *.gz ]]; then gunzip -c "$TRACE" > "$PERFETTO"; else cp "$TRACE" "$PERFETTO"; fi
echo "PERFETTO=$PERFETTO"
ls -lah "$PERFETTO"
python3 -c "import json;d=json.load(open('$PERFETTO'));ev=d.get('traceEvents',d if isinstance(d,list) else []);print('trace_events',len(ev))"
kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || true
