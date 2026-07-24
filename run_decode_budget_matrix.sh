#!/usr/bin/env bash
# Phase-1 canonical decode budget: isolating sampling matrix at C=1 and C=5.
# Uses official benchmark_serving_parallel.py with BENCH_* sampling overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/logs/decode_budget_${RUN_TS}}"
mkdir -p "$OUT_ROOT/results"
SUMMARY="$OUT_ROOT/summary.txt"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export PORT="${PORT:-8092}"
export MASTER_PORT="${MASTER_PORT:-9748}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export BENCH_MODEL_ID="${BENCH_MODEL_ID:-$MODEL_NAME}"
export TOKENIZER="${TOKENIZER:-${MODEL_ROOT}/${MODEL_NAME}}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b) DEFAULT_MAX_MEMORY_UTILIZATION=0.90 ;;
  *) DEFAULT_MAX_MEMORY_UTILIZATION=0.70 ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
export BENCH_DIR="${BENCH_DIR:-/workspace}"
export HOST="${HOST:-127.0.0.1}"
export PY="${PY:-python3}"

export INPUT_LEN="${INPUT_LEN:-2500}"
export OUTPUT_LEN="${OUTPUT_LEN:-1500}"
export SEED="${SEED:-42001}"
export PREFIX_CACHE_RATE="${PREFIX_CACHE_RATE:-0}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
export WARMUP_PROMPTS="${WARMUP_PROMPTS:-5}"
# 10-prompt slices isolate sampling arms (±0.2–0.4 ms). Use NUM_PROMPTS=50
# for a gold-style confirmation run.
export NUM_PROMPTS="${NUM_PROMPTS:-10}"

export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-5}"
export MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-5}"
export ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
export ENABLE_PREFILL_PIECEWISE_GRAPH="${ENABLE_PREFILL_PIECEWISE_GRAPH:-0}"
export ENABLE_PACKED_PREFILL="${ENABLE_PACKED_PREFILL:-0}"
export XLLM_USE_FA3="${XLLM_USE_FA3:-1}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-0}"
export SKIP_SERVER_RESTART="${SKIP_SERVER_RESTART:-0}"

# arm_name|temperature|top_k|top_p
ARMS=(
  "greedy|0.0|0|1.0"
  "temp_only|0.9|0|1.0"
  "topk_only|0.9|20|1.0"
  "official|0.9|20|0.95"
)
# shellcheck disable=SC2206
CONCURRENCY_LEVELS=(${CONCURRENCY_LEVELS:-1 5})

wait_server() {
  local start
  start=$(date +%s)
  echo "waiting for server http://${HOST}:${PORT} ..." | tee -a "$SUMMARY"
  while true; do
    if curl -fsS --max-time 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "server is ready" | tee -a "$SUMMARY"
      return 0
    fi
    if [[ $(( $(date +%s) - start )) -ge "${WAIT_TIMEOUT:-900}" ]]; then
      echo "server did not become ready within ${WAIT_TIMEOUT:-900}s" | tee -a "$SUMMARY"
      return 1
    fi
    sleep 2
  done
}

run_bench() {
  local c="$1"
  local arm_name="$2"
  local temperature="$3"
  local top_k="$4"
  local top_p="$5"
  local num_prompts="$6"
  local tag="$7"
  local log="$OUT_ROOT/${tag}.log"

  echo "" | tee -a "$SUMMARY"
  echo "========== ${tag} C=${c} arm=${arm_name} T=${temperature} top_k=${top_k} top_p=${top_p} n=${num_prompts} ==========" | tee -a "$SUMMARY"

  cd "$BENCH_DIR"
  BENCH_TEMPERATURE="$temperature" \
  BENCH_TOP_K="$top_k" \
  BENCH_TOP_P="$top_p" \
  timeout "$REQUEST_TIMEOUT" "$PY" benchmark_serving_parallel.py \
    --model "$BENCH_MODEL_ID" \
    --tokenizer "$TOKENIZER" \
    --dataset-name random \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$num_prompts" \
    --prefix-cache-rate "$PREFIX_CACHE_RATE" \
    --host "$HOST" \
    --port "$PORT" \
    --endpoint /v1/chat/completions \
    --backend openai-chat \
    --request-rate "$c" \
    --seed "$SEED" \
    --trust-remote-code \
    --disable-tqdm \
    --save-result \
    --result-dir "$OUT_ROOT/results" \
    2>&1 | tee "$log"

  grep -E "Mean TPOT|Median TPOT|Mean TTFT|Successful requests|Output token throughput" "$log" \
    | tee -a "$SUMMARY" || true
}

{
  echo "############################################################"
  echo "# xLLM Phase-1 decode budget matrix"
  echo "# date: $(date)"
  echo "# ISL/OSL: ${INPUT_LEN}/${OUTPUT_LEN}"
  echo "# warmup=${WARMUP_PROMPTS} measure=${NUM_PROMPTS} seed=${SEED}"
  echo "# piecewise=${ENABLE_PREFILL_PIECEWISE_GRAPH} packed=${ENABLE_PACKED_PREFILL} FA3=${XLLM_USE_FA3}"
  echo "# out: ${OUT_ROOT}"
  echo "############################################################"
} | tee "$SUMMARY"

if [[ "$SKIP_SERVER_RESTART" != "1" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/kill_zombie_xllm.sh"
  kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" >>"$SUMMARY" 2>&1 || true
  sleep 3

  cd "$SCRIPT_DIR"
  MAX_CONCURRENT_REQUESTS="$MAX_CONCURRENT_REQUESTS" \
  MAX_SEQS_PER_BATCH="$MAX_SEQS_PER_BATCH" \
  ENABLE_GRAPH="$ENABLE_GRAPH" \
  ENABLE_PREFILL_PIECEWISE_GRAPH="$ENABLE_PREFILL_PIECEWISE_GRAPH" \
  ENABLE_PACKED_PREFILL="$ENABLE_PACKED_PREFILL" \
  NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
  XLLM_USE_FA3="$XLLM_USE_FA3" \
  MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
  MODEL_ROOT="$MODEL_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  PORT="$PORT" \
  bash run_xllm_musa.sh --background >>"$SUMMARY" 2>&1

  wait_server
else
  wait_server
fi

for c in "${CONCURRENCY_LEVELS[@]}"; do
  run_bench "$c" "official" "0.9" "20" "0.95" "$WARMUP_PROMPTS" "warmup_c${c}_official"

  for arm in "${ARMS[@]}"; do
    IFS='|' read -r arm_name temperature top_k top_p <<<"$arm"
    run_bench "$c" "$arm_name" "$temperature" "$top_k" "$top_p" \
      "$NUM_PROMPTS" "measure_c${c}_${arm_name}"
  done
done

echo "" | tee -a "$SUMMARY"
echo "Done. Artifacts under ${OUT_ROOT}" | tee -a "$SUMMARY"
