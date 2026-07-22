#!/bin/bash
# Fast correctness smoke test: one prompt, temp=0, single run.
#
# Usage (server already running):
#   bash correctness_check.sh
#
# Usage (start server first):
#   START_SERVER=1 bash correctness_check.sh
#
# Usage (also run C=2 concurrent same-prompt check):
#   CONCURRENCY=2 bash correctness_check.sh
#
# Env: MUSA_VISIBLE_DEVICES=1  MODEL_NAME=Qwen3.5-27B  PORT=8092
#      QUESTION=...  EXPECTED_SUBSTR=391  MAXTOK=256  STOP_SERVER=1
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=kill_zombie_xllm.sh
source "$(dirname "$0")/kill_zombie_xllm.sh"

export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
export FLASHINFER_OPS_PATH="${FLASHINFER_OPS_PATH:-/workspace/mate_cached_ops}"
REAL_MUSA_LIB="${REAL_MUSA_LIB:-${SCRIPT_DIR}/runtime_libs/musa}"
if [[ -d "$REAL_MUSA_LIB" ]]; then
  export LD_LIBRARY_PATH="${REAL_MUSA_LIB}:${LD_LIBRARY_PATH:-}"
fi

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-1}"
export ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-true}"
# Graph-mode bring-up: MTP off by default (graph MTP is buggy). Override with env.
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-0}"
export ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
export ENABLE_GRAPH_VMM_POOL="${ENABLE_GRAPH_VMM_POOL:-0}"
export XLLM_USE_FA3="${XLLM_USE_FA3:-1}"
PORT="${PORT:-8092}"
MASTER_PORT="${MASTER_PORT:-9748}"
export MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:${MASTER_PORT}}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
QUESTION="${QUESTION:-What is 17 multiplied by 23? Reply with only the number.}"
MAXTOK="${MAXTOK:-256}"
CONCURRENCY="${CONCURRENCY:-1}"
START_SERVER="${START_SERVER:-0}"
STOP_SERVER="${STOP_SERVER:-0}"

readyprobe() {
  curl -s -m 30 -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
    2>/dev/null | grep -q completion_tokens
}

stop_xllm() { kill_zombie_xllm "$@"; }

if [[ "$START_SERVER" == "1" ]]; then
  kill_zombie_xllm "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))" || { echo "FAIL: requested ports are still busy; aborting startup"; exit 1; }
  echo "==> starting xLLM on port ${PORT} (GPU ${MUSA_VISIBLE_DEVICES})"
  bash run_xllm_musa.sh --background --port "$PORT"
  w=0
  until readyprobe; do
    sleep 5
    w=$((w + 5))
    [ "$w" -ge 420 ] && { echo "FAIL: server did not become ready"; exit 1; }
  done
  echo "==> server ready after ${w}s"
elif ! readyprobe; then
  echo "FAIL: no server on ${URL} (set START_SERVER=1 to launch)"
  exit 1
else
  echo "==> using existing server at ${URL}"
fi

export URL MODEL_NAME QUESTION MAXTOK CONCURRENCY
PYTHONUNBUFFERED=1 python3 -u - <<'PY'
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = os.environ["URL"]
MODEL = os.environ["MODEL_NAME"]
QUESTION = os.environ["QUESTION"]
MAXTOK = int(os.environ["MAXTOK"])
CONCURRENCY = int(os.environ["CONCURRENCY"])

# Optional: if question looks like 17*23, expect 391 in output.
EXPECTED_SUBSTR = os.environ.get("EXPECTED_SUBSTR", "391" if "17" in QUESTION and "23" in QUESTION else "")


def is_garbage(text: str) -> bool:
    if not text:
        return True
    s = re.sub(r"\s", "", text)
    if not s:
        return True
    if len(set(s)) <= 2 and len(s) >= 4:
        return True
    if sum(1 for c in s if c in "!?") > len(s) * 0.5:
        return True
    return False


def ask(tag: str):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": QUESTION}],
            "max_tokens": MAXTOK,
            "temperature": 0.0,
        }
    ).encode()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"}
            ),
            timeout=300,
        ) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage") or {}
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {
            "tag": tag,
            "ok": True,
            "content": content,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": time.perf_counter() - t0,
            "error": "",
        }
    except Exception as exc:
        return {
            "tag": tag,
            "ok": False,
            "content": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": time.perf_counter() - t0,
            "error": repr(exc),
        }


print(f"# correctness_check: model={MODEL} maxtok={MAXTOK} temp=0.0")
print(f"# question: {QUESTION}")

r1 = ask("run1")
print(f"\n[run1] ok={r1['ok']} pt={r1['prompt_tokens']} ct={r1['completion_tokens']} "
      f"lat={r1['latency_s']:.1f}s")
if r1["ok"]:
    print("  answer:")
    print(r1["content"])
else:
    print(f"  error: {r1['error']}")

conc_results = []
if CONCURRENCY > 1:
    print(f"\n[concurrent C={CONCURRENCY}] same prompt, temp=0")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(ask, f"c{i}") for i in range(CONCURRENCY)]
        for fut in as_completed(futs):
            conc_results.append(fut.result())
    for r in sorted(conc_results, key=lambda x: x["tag"]):
        snip = (r["content"] or r["error"]).replace("\n", " ")[:80]
        print(f"  [{r['tag']}] ok={r['ok']} ct={r['completion_tokens']} lat={r['latency_s']:.1f}s | {snip}")

checks = []
checks.append(("run1_http", r1["ok"]))
checks.append(("run1_not_garbage", r1["ok"] and not is_garbage(r1["content"])))
checks.append(("run1_has_tokens", r1["ok"] and r1["completion_tokens"] > 0))
if EXPECTED_SUBSTR:
    checks.append(("expected_substring", r1["ok"] and EXPECTED_SUBSTR in r1["content"]))

if CONCURRENCY > 1 and conc_results:
    golden = r1["content"] if r1["ok"] else ""
    all_ok = all(r["ok"] for r in conc_results)
    all_match = all(r["content"] == golden for r in conc_results if r["ok"])
    all_not_garbage = all(
        r["ok"] and not is_garbage(r["content"]) for r in conc_results
    )
    checks.append(("concurrent_all_ok", all_ok))
    checks.append(("concurrent_not_garbage", all_not_garbage))
    if EXPECTED_SUBSTR:
        all_expected = all(
            r["ok"] and EXPECTED_SUBSTR in r["content"] for r in conc_results
        )
        checks.append(("concurrent_expected_substring", all_expected))
    print(f"  [INFO] concurrent_match_golden={all_ok and all_match and bool(golden)}")

print("\n######## CHECKS ########")
passed = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        passed = False

print(f"\nCORRECTNESS: {'PASS' if passed else 'FAIL'}")
sys.exit(0 if passed else 1)
PY
rc=$?

if [[ "$STOP_SERVER" == "1" ]]; then
  stop_xllm "$PORT" "$MASTER_PORT"
fi

exit "$rc"
