#!/bin/bash
# SGLang conc_eval mirror for sglang-wf: graph on, MTP off (no speculative flags).
#
# Run from host:
#   docker cp /data/feihu/xllm-git-master/conc_eval_sglang.sh sglang-wf:/tmp/
#   docker exec sglang-wf bash /tmp/conc_eval_sglang.sh
#
# Env overrides (same semantics as xllm conc_eval.sh):
#   INPUT_LEN=512  OUTPUT_LEN=512  CONCURRENCY_LEVELS="1 2"
#   MUSA_VISIBLE_DEVICES=1  PORT=30000  RESTART_SERVER=1
set -u

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-/tmp/conc_eval_sglang}"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/conc_eval_sglang_${RUN_TS}.log"
ANSWER_DIR="$LOG_ROOT/answers_${RUN_TS}"
mkdir -p "$ANSWER_DIR"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export MODEL_API_ID="${MODEL_API_ID:-${MODEL_ROOT}/${MODEL_NAME}}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
export IGNORE_EOS="${IGNORE_EOS:-1}"
PORT="${PORT:-30000}"
RESTART_SERVER="${RESTART_SERVER:-1}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

SGLANG_PY="${SGLANG_PY:-/root/.virtualenvs/sglang-default/bin/python}"
SGLANG_KILL="${SGLANG_KILL:-/sgl-workspace/sglang/scripts/killall_sglang.sh}"

launch_sglang() {
  nohup "${SGLANG_PY}" -m sglang.launch_server \
    --model-path "${MODEL_ROOT}/${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --tp-size 1 \
    --device musa \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --linear-attn-backend flashinfer \
    --mamba-scheduler-strategy extra_buffer \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-32}" \
    --mem-fraction-static 0.8 \
    --disable-overlap-schedule \
    --sampling-backend flashinfer \
    --decode-log-interval 1 \
    >>"$LOG" 2>&1 &
  echo "==> SGLang PID: $!"
}

readyprobe() {
  curl -s -m 20 -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_API_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
    2>/dev/null | grep -q completion_tokens
}

live_sglang_pid() {
  pgrep -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null | head -1
}

{
echo "############################################################"
echo "# SGLang conc_eval (graph on, MTP off)"
echo "# date          : $(date)"
echo "# hostname      : $(hostname)"
echo "# INPUT_LEN     : ${INPUT_LEN}"
echo "# OUTPUT_LEN    : ${OUTPUT_LEN}"
echo "# concurrency   : ${CONCURRENCY_LEVELS:-1 2}"
echo "# MODEL_API_ID  : ${MODEL_API_ID}"
echo "# MUSA_VISIBLE_DEVICES: ${MUSA_VISIBLE_DEVICES}"
echo "# PORT          : ${PORT}"
echo "# answer dir    : ${ANSWER_DIR}"
echo "# torch / torch_musa:"
"${SGLANG_PY}" -c "import torch,torch_musa; print('   torch=%s  torch_musa=%s' % (torch.__version__, torch_musa.__version__))" 2>/dev/null \
  || "${SGLANG_PY}" -c "import torch; print('   torch=%s' % torch.__version__)"
echo "############################################################"
} | tee "$LOG"

if [[ "$RESTART_SERVER" == "1" ]]; then
  if [[ -x "$SGLANG_KILL" ]]; then
    bash "$SGLANG_KILL" >>"$LOG" 2>&1 || true
  else
    pkill -f "sglang.launch_server" 2>/dev/null || true
  fi
  sleep 3
  launch_sglang
else
  if ! readyprobe; then
    echo "no server on ${URL}; set RESTART_SERVER=1" | tee -a "$LOG"
    exit 1
  fi
  echo "==> using existing SGLang server on port ${PORT}" | tee -a "$LOG"
fi

w=0
until readyprobe; do
  sleep 5
  w=$((w + 5))
  [ "$w" -ge 600 ] && { echo "server FAIL" | tee -a "$LOG"; exit 1; }
done
echo "server ready after ${w}s; PID: $(live_sglang_pid)" | tee -a "$LOG"
if pid=$(live_sglang_pid); then
  echo "server cmdline: $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || echo n/a)" | tee -a "$LOG"
fi

export LOG URL MODEL_API_ID MODEL_NAME MODEL_ROOT INPUT_LEN OUTPUT_LEN ANSWER_DIR CONCURRENCY_LEVELS
PYTHONUNBUFFERED=1 "${SGLANG_PY}" -u - <<'PY' 2>&1 | tee -a "$LOG"
import json
import os
import re
import time
import urllib.request
import concurrent.futures as cf
from pathlib import Path

URL = os.environ["URL"]
MODEL = os.environ["MODEL_API_ID"]
MODEL_ROOT = os.environ["MODEL_ROOT"]
MODEL_NAME = os.environ["MODEL_NAME"]
INPUT_LEN = int(os.environ["INPUT_LEN"])
OUTPUT_LEN = int(os.environ["OUTPUT_LEN"])
ANSWER_DIR = Path(os.environ["ANSWER_DIR"])
CONCURRENCY_LEVELS = [
    int(x) for x in os.environ.get("CONCURRENCY_LEVELS", "1 2").split()
]
ANSWER_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_CACHE = os.environ.get(
    "PROMPT_CACHE",
    os.path.join(os.environ.get("MODEL_ROOT", "/workspace/model_weights"), "..", "conc_eval_prompt_512.json"),
)
PROMPT_CACHE = os.path.abspath(PROMPT_CACHE)

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

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        model_path = os.path.join(MODEL_ROOT, MODEL_NAME)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return _TOKENIZER

def encode_chat_user_prompt(user_content: str) -> list[int]:
    tok = get_tokenizer()
    messages = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

def chat_len_for_body_ids(body_ids: list[int]) -> int:
    tok = get_tokenizer()
    text = tok.decode(body_ids, skip_special_tokens=True)
    return len(encode_chat_user_prompt(text + QUESTION))

def build_long_prompt(target_tokens: int) -> str:
    tok = get_tokenizer()
    question = QUESTION
    overhead = chat_len_for_body_ids([])
    body_ids: list[int] = []
    para_idx = 0
    while chat_len_for_body_ids(body_ids) < target_tokens:
        para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
        body_ids.extend(tok.encode(para, add_special_tokens=False))
        para_idx += 1
        if para_idx > 10000:
            raise RuntimeError(f"failed to reach target prompt length {target_tokens}")
    lo, hi = 0, len(body_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if chat_len_for_body_ids(body_ids[:mid]) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    body_ids = body_ids[:lo]
    para_idx = 0
    while chat_len_for_body_ids(body_ids) < target_tokens:
        para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
        for token_id in tok.encode(para, add_special_tokens=False):
            trial = body_ids + [token_id]
            if chat_len_for_body_ids(trial) > target_tokens:
                break
            body_ids = trial
            if chat_len_for_body_ids(body_ids) == target_tokens:
                break
        para_idx += 1
        if para_idx > 10000:
            raise RuntimeError(f"failed to fine-tune prompt length to {target_tokens}")
    return tok.decode(body_ids, skip_special_tokens=True) + question

if os.path.isfile(PROMPT_CACHE):
    with open(PROMPT_CACHE, encoding="utf-8") as f:
        cached = json.load(f)
    LONG_PROMPT = cached["prompt"]
    BUILT_PROMPT_TOKENS = int(cached.get("meta", {}).get("actual_chat_tokens", len(encode_chat_user_prompt(LONG_PROMPT))))
    print(f"# loaded cached prompt: {PROMPT_CACHE} target={INPUT_LEN} actual_chat_tokens={BUILT_PROMPT_TOKENS}")
else:
    print(f"# no prompt cache at {PROMPT_CACHE}; building (may be slow in sglang-wf)")
    LONG_PROMPT = build_long_prompt(INPUT_LEN)
    BUILT_PROMPT_TOKENS = len(encode_chat_user_prompt(LONG_PROMPT))
    print(f"# built long prompt: target={INPUT_LEN} actual_chat_tokens={BUILT_PROMPT_TOKENS}")

def ask(i: int, prompt: str, max_tokens: int, temperature: float = 0.7):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "ignore_eos": os.environ.get("IGNORE_EOS", "1") == "1",}).encode()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=int(os.environ.get("HTTP_TIMEOUT", "5400"))) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage") or {}
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {"i": i, "content": content, "prompt_tokens": int(usage.get("prompt_tokens", 0)), "completion_tokens": int(usage.get("completion_tokens", 0)), "latency_s": time.perf_counter() - t0, "error": None}
    except Exception as exc:
        return {"i": i, "content": None, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": time.perf_counter() - t0, "error": repr(exc)}

def is_garbage(text):
    if not text: return True
    s = re.sub(r"\s", "", text)
    if not s: return True
    if len(set(s)) <= 2 and len(s) >= 4: return True
    if sum(1 for c in s if c in "!?") > len(s) * 0.5: return True
    return False

def save_answer(c, req_idx, result):
    path = ANSWER_DIR / f"C{c}_req{req_idx:03d}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"concurrency={c}\nrequest={req_idx}\nprompt_tokens={result['prompt_tokens']}\ncompletion_tokens={result['completion_tokens']}\nlatency_s={result['latency_s']:.3f}\n")
        if result["error"]: f.write(f"error={result['error']}\n")
        f.write("\n--- answer ---\n" + (result["content"] or "") + "\n")
    return path

def warmup():
    print(f"\n######## WARMUP  input_len~={INPUT_LEN}  output_len={OUTPUT_LEN} ########")
    r = ask(0, LONG_PROMPT, OUTPUT_LEN, temperature=0.0)
    tag = "ERR " if r["error"] else ("GARB" if is_garbage(r["content"]) else "ok  ")
    tpot_ms = 1000.0 * r["latency_s"] / max(1, r["completion_tokens"])
    print(f"  [{tag}] warmup: pt={r['prompt_tokens']} ct={r['completion_tokens']} lat={r['latency_s']:.1f}s TPOT={tpot_ms:.1f}ms")
    if r["error"]: raise RuntimeError(f"warmup failed: {r['error']}")
    if is_garbage(r["content"]): raise RuntimeError("warmup produced garbage output")
    if r["prompt_tokens"] < int(INPUT_LEN * 0.95): raise RuntimeError(f"warmup prompt too short: pt={r['prompt_tokens']}")
    if r["completion_tokens"] < int(OUTPUT_LEN * 0.95): raise RuntimeError(f"warmup incomplete: ct={r['completion_tokens']}")
    print(f"  warmup answer saved: {save_answer(0, 0, r)}")
    print("  warmup complete; starting benchmark sweeps")

def run(c):
    prompts = [LONG_PROMPT] * c
    results = {}
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(ask, i, prompts[i], OUTPUT_LEN) for i in range(c)]
        for fut in cf.as_completed(futs):
            r = fut.result(); results[r["i"]] = r
    wall = time.perf_counter() - t0
    total_pt = sum(results[i]["prompt_tokens"] for i in results)
    total_ct = sum(results[i]["completion_tokens"] for i in results)
    ng = sum(is_garbage(results[i]["content"]) for i in results)
    ne = sum(1 for i in results if results[i]["error"])
    lats = [results[i]["latency_s"] for i in results]
    tpots = [1000.0 * results[i]["latency_s"] / max(1, results[i]["completion_tokens"]) for i in results if not results[i]["error"] and results[i]["completion_tokens"] > 0]
    mean_tpot_ms = sum(tpots) / len(tpots) if tpots else float("nan")
    print(f"\n######## C={c}  input_len~={INPUT_LEN}  output_len={OUTPUT_LEN}  temp=0.7 ########")
    for i in range(c):
        r = results[i]
        tag = "ERR " if r["error"] else ("GARB" if is_garbage(r["content"]) else "ok  ")
        snip = (r["error"] if r["error"] else (r["content"] or "")).replace("\n", " ")[:90]
        tpot_ms = 1000.0 * r["latency_s"] / max(1, r["completion_tokens"])
        print(f"  [{tag}] req{i + 1:>3}: pt={r['prompt_tokens']:>4} ct={r['completion_tokens']:>3} lat={r['latency_s']:5.1f}s TPOT={tpot_ms:5.1f}ms | {snip}")
    ok_count = max(1, c - ne)
    decode_tpot_batch_ms = 1000.0 * wall / max(1, total_ct)
    print(f"  >>> C={c}: wall={wall:.2f}s  total_prompt_tokens={total_pt}  total_completion_tokens={total_ct}  AGGREGATE={(total_pt + total_ct) / wall:.1f} tok/s  per-req-avg={(total_ct / ok_count) / (sum(lats) / len(lats)):.2f} tok/s  mean_TPOT={mean_tpot_ms:.1f}ms  batch_decode_TPOT={decode_tpot_batch_ms:.1f}ms  garbage={ng}/{c}  errors={ne}")

warmup()
for c in CONCURRENCY_LEVELS:
    run(c)
PY

echo "" | tee -a "$LOG"
echo "LOG SAVED TO: $LOG" | tee -a "$LOG"
echo "ANSWERS SAVED TO: $ANSWER_DIR" | tee -a "$LOG"
echo "CONC_EVAL_SGLANG_DONE"
