#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/root/.virtualenvs/sglang-default/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/model_weights/Qwen3.5-27B-FP8}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
TOKENIZER="${TOKENIZER:-$MODEL_PATH}"
CLIENT="${CLIENT:-/tmp/benchmark_c5_barrier_waves.py}"
BENCHMARK_LIB_DIR="${BENCHMARK_LIB_DIR:-/sgl-workspace/bench_parallel}"
SGLANG_KILL="${SGLANG_KILL:-/sgl-workspace/sglang/scripts/killall_sglang.sh}"
OUT="${OUT:-/workspace/bench_results/sglang_fp8_c5_cap2_barrier_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-31002}"
SERVER_PID=""
INPUT_LEN="${INPUT_LEN:-2500}"
OUTPUT_LEN="${OUTPUT_LEN:-1500}"
PREFIX_LEN="${PREFIX_LEN:-200}"
WAVE_SIZE="${WAVE_SIZE:-5}"
WARMUP_WAVES="${WARMUP_WAVES:-1}"
NUM_WAVES="${NUM_WAVES:-10}"
RELEASE_PARTITION="${RELEASE_PARTITION:-}"
INTER_GROUP_DELAY_MS="${INTER_GROUP_DELAY_MS:-0}"
INTER_WAVE_DELAY_MS="${INTER_WAVE_DELAY_MS:-0}"

mkdir -p "$OUT"

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -x "$SGLANG_KILL" ]]; then
    bash "$SGLANG_KILL" >/dev/null 2>&1 || true
  else
    pkill -f "sglang.launch_server" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
cleanup
sleep 3

if ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
  echo "Port ${PORT} is already in use; refusing to benchmark a stale server."
  exit 1
fi

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"

nohup "$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --tp-size 1 \
  --device musa \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --linear-attn-backend flashinfer \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --prefill-max-requests 2 \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.8 \
  --max-running-requests 8 \
  --sampling-backend flashinfer \
  --decode-log-interval 1 \
  >"$OUT/server.log" 2>&1 &
SERVER_PID="$!"
echo "PID=$SERVER_PID"

URL="http://127.0.0.1:${PORT}/v1/chat/completions"
waited=0
until curl -s -m 20 -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
  | grep -q completion_tokens; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "The newly launched SGLang server exited before readiness."
    tail -100 "$OUT/server.log" || true
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
  if ((waited >= 900)); then
    tail -100 "$OUT/server.log" || true
    exit 1
  fi
done
if ! grep -q "prefill_max_requests=2" "$OUT/server.log"; then
  echo "The ready server log does not confirm prefill_max_requests=2."
  exit 1
fi
echo "SERVER_READY after ${waited}s"

release_args=()
if [[ -n "$RELEASE_PARTITION" ]]; then
  release_args+=(
    --release-partition "$RELEASE_PARTITION"
    --inter-group-delay-ms "$INTER_GROUP_DELAY_MS"
  )
fi

"$PY" "$CLIENT" \
  --benchmark-lib-dir "$BENCHMARK_LIB_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --prefix-len "$PREFIX_LEN" \
  --wave-size "$WAVE_SIZE" \
  --warmup-waves "$WARMUP_WAVES" \
  --num-waves "$NUM_WAVES" \
  --seed 44002 \
  --warmup-seed 44001 \
  --result-json "$OUT/result.json" \
  --inter-wave-delay-ms "$INTER_WAVE_DELAY_MS" \
  "${release_args[@]}" \
  2>&1 | tee "$OUT/client.log"

{
  echo "prefill_cap=$(grep -m1 -o 'prefill_max_requests=[^,)]*' "$OUT/server.log" || true)"
  echo "cached_prefills=$(grep 'Prefill batch' "$OUT/server.log" | grep -vc '#cached-token: 0' || true)"
  echo "errors=$(grep -cE 'Traceback|OutOfMemory|MUSA error|Scheduler hit an exception' "$OUT/server.log" || true)"
} | tee "$OUT/log_checks.txt"
echo "RESULTS_DIR=$OUT"
