#!/usr/bin/env bash
# Reproducible C=1 A/B benchmark for Qwen3.5-27B-FP8 MTP on/off on MUSA.
# Run this script inside the xLLM MUSA development container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-0.8}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/bench_results/qwen35_27b_fp8_mtp_ab_$(date +%Y%m%d_%H%M%S)}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}-mtp}"

INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-2048}"
PREFIX_LEN="${PREFIX_LEN:-200}"
WARMUP_WAVES="${WARMUP_WAVES:-4}"
MEASURE_WAVES="${MEASURE_WAVES:-10}"
SEED="${SEED:-48002}"
WARMUP_SEED="${WARMUP_SEED:-48001}"
PORT_BASE="${PORT_BASE:-31102}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-19782}"

mkdir -p "$RESULT_ROOT"

SERVER_PID=""
SERVER_LOG=""

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "missing model config: $MODEL_PATH/config.json" >&2
  exit 1
fi

if [[ ! -f "$DRAFT_MODEL_PATH/model.safetensors.index.json" ]]; then
  mkdir -p "$DRAFT_MODEL_PATH"
  python3 tools/export_mtp.py \
    --input-dir "$MODEL_PATH" \
    --output-dir "$DRAFT_MODEL_PATH"
fi

stop_server() {
  local pid="${SERVER_PID:-}"
  if [[ -z "$pid" ]]; then
    return
  fi

  # The server is launched in its own session. Stop only that process group so
  # concurrent xLLM experiments on the same host are left untouched.
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
  SERVER_LOG=""
}

cleanup() {
  stop_server
}
trap cleanup EXIT

assert_ports_available() {
  python3 - "$@" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    port = int(value)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as error:
        raise SystemExit(f"port {port} is already in use: {error}") from error
    finally:
        sock.close()
PY
}

wait_for_server() {
  local port="$1"
  local timeout_s=420
  local elapsed_s=0
  local url="http://127.0.0.1:${port}/health"
  until curl -sf "$url" >/dev/null 2>&1; do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before becoming ready; see ${SERVER_LOG}" >&2
      return 1
    fi
    sleep 5
    elapsed_s=$((elapsed_s + 5))
    if (( elapsed_s >= timeout_s )); then
      echo "server did not become ready on port ${port}" >&2
      return 1
    fi
  done
}

run_case() {
  local tag="$1"
  local port="$2"
  local master_port="$3"
  local speculative_tokens="$4"
  local case_dir="${RESULT_ROOT}/${tag}"

  mkdir -p "$case_dir/log"
  stop_server
  assert_ports_available "$port" "$master_port"

  export MUSA_VISIBLE_DEVICES
  export MODEL_NAME
  export MODEL_ROOT
  export MAX_MEMORY_UTILIZATION
  export MAX_CONCURRENT_REQUESTS=1
  export ENABLE_SCHEDULE_OVERLAP=false
  export ENABLE_GRAPH=1
  export ENABLE_GRAPH_VMM_POOL=0
  export PORT="$port"
  export MASTER_NODE_ADDR="127.0.0.1:${master_port}"
  export LOG_DIR="${case_dir}/log"
  export NUM_SPECULATIVE_TOKENS="$speculative_tokens"

  if (( speculative_tokens > 0 )); then
    export SPECULATIVE_ALGORITHM=MTP
    export DRAFT_MODEL_PATH
    export XLLM_ENABLE_GRAPH_MTP=1
    export XLLM_MATE_GDN_MTP=1
  else
    unset SPECULATIVE_ALGORITHM
    unset XLLM_ENABLE_GRAPH_MTP
    unset XLLM_MATE_GDN_MTP
  fi

  SERVER_LOG="${case_dir}/server.log"
  setsid bash run_xllm_musa.sh --port "$port" --device 0 \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" >"${case_dir}/server.pid"
  wait_for_server "$port"

  BENCH_TEMPERATURE=0 BENCH_TOP_K=0 BENCH_TOP_P=1 \
    python3 benchmark_c5_barrier_waves.py \
      --host 127.0.0.1 \
      --port "$port" \
      --model "$MODEL_NAME" \
      --tokenizer "$MODEL_PATH" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --prefix-len "$PREFIX_LEN" \
      --wave-size 1 \
      --warmup-waves "$WARMUP_WAVES" \
      --num-waves "$MEASURE_WAVES" \
      --seed "$SEED" \
      --warmup-seed "$WARMUP_SEED" \
      --result-json "${case_dir}/benchmark.json" \
      | tee "${case_dir}/benchmark.log"

  stop_server
}

printf 'result_root=%s\nmodel=%s\ninput_len=%s\noutput_len=%s\n' \
  "$RESULT_ROOT" "$MODEL_NAME" "$INPUT_LEN" "$OUTPUT_LEN" \
  > "${RESULT_ROOT}/metadata.txt"

run_case "mtp_off" "$PORT_BASE" "$MASTER_PORT_BASE" 0
run_case "mtp_on" "$((PORT_BASE + 1))" "$((MASTER_PORT_BASE + 1))" 1

python3 - "${RESULT_ROOT}/mtp_off/benchmark.json" "${RESULT_ROOT}/mtp_on/benchmark.json" <<'PY'
import json
import sys

labels = ("MTP off", "MTP on")
keys = ("mean_ttft_ms", "mean_tpot_ms", "mean_latency_ms", "output_throughput")
for label, path in zip(labels, sys.argv[1:]):
    with open(path) as f:
        result = json.load(f)
    print(label)
    for key in keys:
        print(f"  {key}: {result[key]}")
PY
