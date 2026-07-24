#!/bin/bash
# Phase G: validate prefix cache on shared-prefix workloads (not fairness benches).
# Starts xLLM with prefix ON, sends N requests that share a long system prompt,
# and prints TTFT for cold vs warm (cached) hits.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
# shellcheck source=../kill_zombie_xllm.sh
source "$ROOT/kill_zombie_xllm.sh"

PORT="${PORT:-8093}"
MASTER_PORT="${MASTER_PORT:-9750}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
case "${MODEL_NAME,,}" in
  qwen3.5-35b-a3b) DEFAULT_MAX_MEMORY_UTILIZATION=0.90 ;;
  *) DEFAULT_MAX_MEMORY_UTILIZATION=0.70 ;;
esac
export MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-$DEFAULT_MAX_MEMORY_UTILIZATION}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
NUM_WARM="${NUM_WARM:-4}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

export MUSA_VISIBLE_DEVICES
export MODEL_NAME
export ENABLE_PREFIX_CACHE=true
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
export ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
export ENABLE_PREFILL_PIECEWISE_GRAPH="${ENABLE_PREFILL_PIECEWISE_GRAPH:-1}"
export XLLM_USE_FA3="${XLLM_USE_FA3:-1}"
export XLLM_MATE_GDN_PREFILL="${XLLM_MATE_GDN_PREFILL:-1}"

kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || true
MASTER_NODE_ADDR="127.0.0.1:${MASTER_PORT}" PORT="$PORT" \
  bash run_xllm_musa.sh --background --port "$PORT"

ready() {
  curl -s -m 20 -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
    2>/dev/null | grep -q completion_tokens
}
w=0
until ready; do
  sleep 5
  w=$((w + 5))
  [ "$w" -ge 420 ] && { echo "server FAIL"; exit 1; }
done
echo "server ready after ${w}s"

python3 - <<PY
import json, os, time, urllib.request
url = "$URL"
model = "$MODEL_NAME"
prefix_tokens = int("$PREFIX_TOKENS")
filler = ("The shared system context for prefix-cache validation. " * 200)
system = (filler * ((prefix_tokens // 20) + 1))[: prefix_tokens * 4]
users = [
    "Summarize the system context in one sentence.",
    "List three keywords from the system context.",
    "Is the system context empty? Answer yes or no.",
    "Reply with OK if you received the system context.",
]

def one(user: str):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    ttft_proxy = (time.perf_counter() - t0) * 1000.0
    text = data["choices"][0]["message"]["content"]
    return ttft_proxy, text[:80]

print(f"shared_prefix_chars={len(system)} num_warm={$NUM_WARM}")
cold_ms, cold_txt = one(users[0])
print(f"cold_ttft_proxy_ms={cold_ms:.1f} text={cold_txt!r}")
warm = []
for i in range(int("$NUM_WARM")):
    ms, txt = one(users[(i + 1) % len(users)])
    warm.append(ms)
    print(f"warm[{i}]_ttft_proxy_ms={ms:.1f} text={txt!r}")
print(f"warm_mean_ms={sum(warm)/len(warm):.1f} cold_ms={cold_ms:.1f} "
      f"speedup={cold_ms / (sum(warm)/len(warm)):.2f}x")
print("PASS: prefix-cache shared-prefix bench completed "
      "(expect warm << cold when prefix hits)")
PY
