#!/bin/bash
# Concurrency throughput evaluation with long prompt + long decode.
#
# Env overrides:
#   INPUT_LEN=512   target user prompt tokens (chat template, model tokenizer)
#   OUTPUT_LEN=512  max completion tokens per request
#   CONCURRENCY_LEVELS="16"  space-separated list (default: 16 32 64)
#   HTTP_TIMEOUT=5400       per-request urllib timeout in seconds (default: 1800)
#   MUSA_VISIBLE_DEVICES=1  PORT=8092
set -u
# Use the script's own directory so this works for either xllm_0526 or xllm-git-master.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=kill_zombie_xllm.sh
source "$SCRIPT_DIR/kill_zombie_xllm.sh"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs conc_eval_full_answer
LOG="logs/conc_eval_${RUN_TS}.log"
ANSWER_DIR="conc_eval_full_answer/${RUN_TS}"
mkdir -p "$ANSWER_DIR"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
export ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-true}"
# Graph-mode bring-up: MTP off by default (graph MTP is buggy). Override with env.
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-0}"
export ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
export ENABLE_GRAPH_VMM_POOL="${ENABLE_GRAPH_VMM_POOL:-0}"
export XLLM_USE_FA3="${XLLM_USE_FA3:-1}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
# Throughput runs need full-length decode; Qwen3.5 may EOS early otherwise.
export IGNORE_EOS="${IGNORE_EOS:-1}"
PORT="${PORT:-8092}"
MASTER_PORT="${MASTER_PORT:-9748}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

{
echo "############################################################"
echo "# Concurrency throughput evaluation"
echo "# date          : $(date)"
echo "# hostname      : $(hostname)"
echo "# INPUT_LEN     : ${INPUT_LEN}"
echo "# OUTPUT_LEN    : ${OUTPUT_LEN}"
echo "# concurrency   : ${CONCURRENCY_LEVELS:-16 32 64}"
echo "# answer dir    : ${ANSWER_DIR}"
echo "# container uname: $(uname -a)"
echo "# torch / torch_musa:"
python3 -c "import torch,torch_musa; print('   torch=%s  torch_musa=%s' % (torch.__version__, torch_musa.__version__))"
echo "# xllm binary   : $(ls -l build/lib.linux-x86_64-cpython-310/xllm/xllm 2>/dev/null | awk '{print $5, $6, $7, $8, $9}')"
echo "############################################################"
} | tee "$LOG"

readyprobe() {
  curl -s -m 20 -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
    2>/dev/null | grep -q completion_tokens
}
live_xllm_pid() {
  ps -o pid=,stat= -C xllm 2>/dev/null | awk '$2!="Z"{print $1; exit}'
}

kill_zombie_xllm "$PORT" "$MASTER_PORT"
bash run_xllm_musa.sh --background --port "$PORT" >>"$LOG" 2>&1
expected_pid=$(grep '==> PID:' "$LOG" | tail -1 | awk '{print $NF}')
w=0
until readyprobe; do
  sleep 5
  w=$((w + 5))
  [ "$w" -ge 420 ] && { echo "server FAIL" | tee -a "$LOG"; exit 1; }
done
live_pid=$(live_xllm_pid)
if [[ -n "${expected_pid:-}" && "${live_pid:-}" != "$expected_pid" ]]; then
  echo "server PID mismatch: expected=${expected_pid} live=${live_pid:-none}" | tee -a "$LOG"
  exit 1
fi
echo "server ready after ${w}s; serving PID: ${live_pid:-unknown}" | tee -a "$LOG"
if [[ -n "${live_pid:-}" && -r "/proc/$live_pid/cmdline" ]]; then
  echo "server cmdline: $(tr '\0' ' ' < "/proc/$live_pid/cmdline")" | tee -a "$LOG"
fi

export LOG URL MODEL_NAME MODEL_ROOT INPUT_LEN OUTPUT_LEN ANSWER_DIR CONCURRENCY_LEVELS
PYTHONUNBUFFERED=1 python3 -u - <<'PY' 2>&1 | tee -a "$LOG"
import json
import os
import re
import time
import urllib.request
import concurrent.futures as cf
from pathlib import Path

URL = os.environ["URL"]
MODEL = os.environ["MODEL_NAME"]
MODEL_ROOT = os.environ["MODEL_ROOT"]
INPUT_LEN = int(os.environ["INPUT_LEN"])
OUTPUT_LEN = int(os.environ["OUTPUT_LEN"])
LONG_CONTEXT_THRESHOLD = 512
ANSWER_DIR = Path(os.environ["ANSWER_DIR"])
CONCURRENCY_LEVELS = [
    int(x) for x in os.environ.get("CONCURRENCY_LEVELS", "16 32 64").split()
]
ANSWER_DIR.mkdir(parents=True, exist_ok=True)

QUESTION = (
    "\n\nBased only on the context above, summarize the main ideas in "
    "three or four sentences."
)
CONTEXT_PARAGRAPHS = [
    (
        "Modern transformer language models combine multi-head self-attention with "
        "position-wise feed-forward layers. Attention computes pairwise affinities "
        "between query and key vectors, applies a softmax over keys, and averages "
        "value vectors to produce contextualized token representations. "
    ),
    (
        "During prefill, the model processes the entire prompt in parallel and "
        "materializes key-value cache entries for every layer. During decode, each "
        "new token attends to cached keys and values from prior positions, which "
        "makes incremental generation efficient but memory-intensive at long context. "
    ),
    (
        "Rotary position embeddings inject relative position information into query "
        "and key projections without adding absolute position vectors to the residual "
        "stream. This design scales well to long contexts and is widely used in "
        "recent open-weight decoder-only models. "
    ),
    (
        "Grouped-query attention reduces KV-cache footprint by sharing key and value "
        "heads across multiple query heads. The trade-off is mild quality loss in "
        "some settings, but the memory savings enable higher batch concurrency and "
        "longer sequences on fixed GPU capacity. "
    ),
    (
        "Mixture-of-experts feed-forward layers activate only a small subset of "
        "experts per token, increasing model capacity without proportionally "
        "increasing compute. Serving systems must route tokens to experts efficiently "
        "and balance load across devices when expert parallelism is enabled. "
    ),
    (
        "Hybrid architectures interleave full attention blocks with linear-time "
        "recurrent or state-space layers to reduce quadratic cost on very long "
        "inputs. These designs aim to preserve quality on long-context workloads "
        "while improving throughput for prefill-heavy serving scenarios. "
    ),
    (
        "Production inference stacks batch requests dynamically, overlap scheduling "
        "with execution, and separate prefill from decode when possible. Throughput "
        "metrics must account for prompt length, decode length, concurrency, and "
        "whether outputs are truncated by end-of-sequence tokens. "
    ),
    (
        "Quantization lowers memory bandwidth pressure by storing weights in fewer "
        "bits, but kernel support and accuracy vary by hardware backend. Evaluating "
        "a serving stack requires checking not only tokens per second, but also "
        "correctness, tail latency, and stability under concurrent long prompts. "
    ),
]
POOL = [
    "What is the capital of France, and what river runs through it?",
    "Explain what a binary search algorithm does in two sentences.",
    "Write a one-line Python function that returns the factorial of n.",
    "What is 17 multiplied by 23? Show your reasoning briefly.",
    "Translate 'Good morning, how are you?' into French and Japanese.",
    "Name the largest planet in the solar system and one of its moons.",
    "What is the boiling point of water at sea level in Celsius?",
    "Give three synonyms for the word 'happy'.",
    "What year did the first human land on the Moon?",
    "Briefly explain what photosynthesis is.",
    "Convert 100 kilometers to miles (approx).",
    "What is the chemical symbol for gold?",
    "List the first five prime numbers.",
    "Who wrote the play 'Romeo and Juliet'?",
    "What is the square root of 144?",
    "Name a programming language created by Google.",
]


_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        model_path = os.path.join(MODEL_ROOT, MODEL)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    return _TOKENIZER


def encode_chat_user_prompt(user_content: str) -> list[int]:
    tok = get_tokenizer()
    messages = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def build_long_prompt(target_tokens: int) -> str:
    tok = get_tokenizer()
    question = QUESTION
    body_ids: list[int] = []
    para_idx = 0

    while len(encode_chat_user_prompt(tok.decode(body_ids, skip_special_tokens=True) + question)) < target_tokens:
        para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
        body_ids.extend(tok.encode(para, add_special_tokens=False))
        para_idx += 1
        if para_idx > 10000:
            raise RuntimeError(f"failed to reach target prompt length {target_tokens}")

    while body_ids and len(
        encode_chat_user_prompt(tok.decode(body_ids, skip_special_tokens=True) + question)
    ) > target_tokens:
        body_ids.pop()

    while len(
        encode_chat_user_prompt(tok.decode(body_ids, skip_special_tokens=True) + question)
    ) < target_tokens:
        para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
        for token_id in tok.encode(para, add_special_tokens=False):
            trial = body_ids + [token_id]
            if len(
                encode_chat_user_prompt(
                    tok.decode(trial, skip_special_tokens=True) + question
                )
            ) > target_tokens:
                break
            body_ids = trial
            if len(
                encode_chat_user_prompt(
                    tok.decode(body_ids, skip_special_tokens=True) + question
                )
            ) == target_tokens:
                break
        para_idx += 1

    return tok.decode(body_ids, skip_special_tokens=True) + question


def prompt_for_request(req_index: int) -> str:
    if INPUT_LEN >= LONG_CONTEXT_THRESHOLD:
        return LONG_PROMPT
    if req_index == 0:
        return LONG_PROMPT
    return POOL[req_index % len(POOL)]


LONG_PROMPT = build_long_prompt(INPUT_LEN)
BUILT_PROMPT_TOKENS = len(encode_chat_user_prompt(LONG_PROMPT))
print(
    f"# built long prompt: target={INPUT_LEN} "
    f"actual_chat_tokens={BUILT_PROMPT_TOKENS}"
)
if INPUT_LEN >= LONG_CONTEXT_THRESHOLD and BUILT_PROMPT_TOKENS < int(INPUT_LEN * 0.95):
    raise RuntimeError(
        f"built prompt too short: {BUILT_PROMPT_TOKENS} tokens, "
        f"expected>={int(INPUT_LEN * 0.95)}"
    )


def ask(i: int, prompt: str, max_tokens: int, temperature: float = 0.7):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": os.environ.get("IGNORE_EOS", "0") == "1",
        }
    ).encode()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"}
            ),
            timeout=int(os.environ.get("HTTP_TIMEOUT", "1800")),
        ) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage") or {}
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {
            "i": i,
            "content": content,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        return {
            "i": i,
            "content": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": time.perf_counter() - t0,
            "error": repr(exc),
        }


def is_garbage(text: str | None) -> bool:
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


def save_answer(c: int, req_idx: int, result: dict) -> Path:
    path = ANSWER_DIR / f"C{c}_req{req_idx:03d}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"concurrency={c}\n")
        f.write(f"request={req_idx}\n")
        f.write(f"prompt_tokens={result['prompt_tokens']}\n")
        f.write(f"completion_tokens={result['completion_tokens']}\n")
        f.write(f"latency_s={result['latency_s']:.3f}\n")
        if result["error"]:
            f.write(f"error={result['error']}\n")
        f.write("\n--- answer ---\n")
        f.write(result["content"] or "")
        f.write("\n")
    return path


def warmup():
    print(
        f"\n######## WARMUP  input_len~={INPUT_LEN}  output_len={OUTPUT_LEN} ########"
    )
    r = ask(0, LONG_PROMPT, OUTPUT_LEN, temperature=0.0)
    tag = "ERR " if r["error"] else ("GARB" if is_garbage(r["content"]) else "ok  ")
    print(
        f"  [{tag}] warmup: pt={r['prompt_tokens']} ct={r['completion_tokens']} "
        f"lat={r['latency_s']:.1f}s"
    )
    if r["error"]:
        raise RuntimeError(f"warmup failed: {r['error']}")
    if is_garbage(r["content"]):
        # Set ALLOW_GARBAGE_WARMUP=1 to downgrade the garbage check to a warning
        # so we can validate that graph-mode capture/replay completes end-to-end
        # even while correctness is being debugged separately. Default 0 keeps
        # the original hard-fail.
        if os.environ.get("ALLOW_GARBAGE_WARMUP", "0") != "1":
            raise RuntimeError("warmup produced garbage output")
        print("  WARN: warmup produced garbage output (ALLOW_GARBAGE_WARMUP=1; continuing)")
    if r["prompt_tokens"] < int(INPUT_LEN * 0.95):
        raise RuntimeError(
            f"warmup prompt too short: pt={r['prompt_tokens']} "
            f"expected>={int(INPUT_LEN * 0.95)}"
        )
    if r["completion_tokens"] < int(OUTPUT_LEN * 0.95):
        raise RuntimeError(
            f"warmup incomplete: ct={r['completion_tokens']} expected>={int(OUTPUT_LEN * 0.95)}"
        )
    path = save_answer(0, 0, r)
    print(f"  warmup answer saved: {path}")
    print("  warmup complete; starting benchmark sweeps")


def run(c: int):
    prompts = [prompt_for_request(i) for i in range(c)]

    results = {}
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(ask, i, prompts[i], OUTPUT_LEN) for i in range(c)]
        for fut in cf.as_completed(futs):
            r = fut.result()
            results[r["i"]] = r

    wall = time.perf_counter() - t0
    total_pt = sum(results[i]["prompt_tokens"] for i in results)
    total_ct = sum(results[i]["completion_tokens"] for i in results)
    ng = sum(is_garbage(results[i]["content"]) for i in results)
    ne = sum(1 for i in results if results[i]["error"])
    lats = [results[i]["latency_s"] for i in results]

    print(
        f"\n######## C={c}  input_len~={INPUT_LEN}  output_len={OUTPUT_LEN}  temp=0.7 ########"
    )
    for i in range(c):
        r = results[i]
        tag = "ERR " if r["error"] else ("GARB" if is_garbage(r["content"]) else "ok  ")
        snip = (r["error"] if r["error"] else (r["content"] or "")).replace("\n", " ")[:90]
        out_path = save_answer(c, i + 1, r)
        print(
            f"  [{tag}] req{i + 1:>3}: pt={r['prompt_tokens']:>4} ct={r['completion_tokens']:>3} "
            f"lat={r['latency_s']:5.1f}s | {snip} | saved={out_path.name}"
        )

    ok_count = max(1, c - ne)
    print(
        f"  >>> C={c}: wall={wall:.2f}s  total_prompt_tokens={total_pt}  "
        f"total_completion_tokens={total_ct}  "
        f"AGGREGATE={(total_pt + total_ct) / wall:.1f} tok/s  "
        f"per-req-avg={(total_ct / ok_count) / (sum(lats) / len(lats)):.2f} tok/s  "
        f"garbage={ng}/{c}  errors={ne}"
    )


warmup()
for c in CONCURRENCY_LEVELS:
    run(c)
PY

stop_xllm
echo "" | tee -a "$LOG"
echo "LOG SAVED TO: $(pwd)/$LOG" | tee -a "$LOG"
echo "ANSWERS SAVED TO: $(pwd)/$ANSWER_DIR" | tee -a "$LOG"
echo "CONC_EVAL_DONE"
