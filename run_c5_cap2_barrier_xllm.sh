#!/usr/bin/env bash
set -euo pipefail

cd /workspace/xllm-git-master

MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
MODEL_PATH="${MODEL_PATH:-/workspace/model_weights/Qwen3.5-27B-FP8}"
TOKENIZER="${TOKENIZER:-$MODEL_PATH}"
OUT="${OUT:-/workspace/bench_results/xllm_fp8_c5_cap2_barrier_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8092}"
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

source ./kill_zombie_xllm.sh

cleanup() {
  kill_zombie_xllm "$PORT" 9748 9749 >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

export MODEL_NAME MODEL_PATH
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export PORT
export MASTER_PORT=9748
export LOG_DIR="$OUT"
export ENABLE_GRAPH=1
export ENABLE_GRAPH_VMM_POOL=0
export ENABLE_PREFILL_PIECEWISE_GRAPH=1
export ENABLE_PACKED_PREFILL=1
export ENABLE_CHUNKED_PREFILL=true
export ENABLE_SCHEDULE_OVERLAP=true
export ENABLE_PREFIX_CACHE=false
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b)
    # BF16 35B-A3B needs the larger post-weight-load KV-cache budget.
    DEFAULT_MAX_MEMORY_UTILIZATION=0.90
    ;;
  *)
    DEFAULT_MAX_MEMORY_UTILIZATION=0.70
    ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
export MAX_TOKENS_FOR_GRAPH_MODE=8192
export MAX_TOKENS_PER_CHUNK_FOR_PREFILL=8192
export MAX_TOKENS_PER_BATCH=16384
export NUM_SPECULATIVE_TOKENS=0
export IGNORE_EOS=1
export XLLM_USE_FA3=1
export XLLM_USE_CUSTOM_PREFILL_CONV=1
export XLLM_PACKED_PREFILL_PIECEWISE=0
export XLLM_MAX_PACKED_PREFILL_SEQS=2
export XLLM_SCHED_PACK_LOG="${XLLM_SCHED_PACK_LOG:-1}"
export XLLM_PREFILL_FWD_TIMING="${XLLM_PREFILL_FWD_TIMING:-0}"
export XLLM_PREFILL_BREAKDOWN="${XLLM_PREFILL_BREAKDOWN:-0}"
export XLLM_GDN_BREAKDOWN="${XLLM_GDN_BREAKDOWN:-0}"
export XLLM_REQUEST_TIMING="${XLLM_REQUEST_TIMING:-0}"
export XLLM_PREFILL_EMPTY_CACHE="${XLLM_PREFILL_EMPTY_CACHE:-0}"
export XLLM_TILELANG_LIB=/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so
export FLASHINFER_OPS_PATH="${FLASHINFER_OPS_PATH:-/workspace/mate_cached_ops}"

bash run_xllm_musa.sh --background --port "$PORT"

SERVER_LOG="$OUT/xllm_${MODEL_NAME}.log"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
waited=0
until curl -s -m 20 -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
  | grep -q completion_tokens; do
  sleep 5
  waited=$((waited + 5))
  if ((waited >= 480)); then
    tail -100 "$SERVER_LOG" || true
    exit 1
  fi
done
echo "SERVER_READY after ${waited}s"

release_args=()
if [[ -n "$RELEASE_PARTITION" ]]; then
  release_args+=(
    --release-partition "$RELEASE_PARTITION"
    --inter-group-delay-ms "$INTER_GROUP_DELAY_MS"
  )
fi

python3 benchmark_c5_barrier_waves.py \
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

cp "$SERVER_LOG" "$OUT/server.log"
{
  echo "sched_pack_count=$(tr -d '\000' < "$SERVER_LOG" | grep -c '\[SCHED_PACK\]' || true)"
  echo "prefill_fwd_count=$(tr -d '\000' < "$SERVER_LOG" | grep -c '\[PREFILL_FWD\]' || true)"
  echo "errors=$(tr -d '\000' < "$SERVER_LOG" | grep -cE 'FATAL|OutOfMemory|MUSA error' || true)"
} | tee "$OUT/log_checks.txt"
echo "RESULTS_DIR=$OUT"
