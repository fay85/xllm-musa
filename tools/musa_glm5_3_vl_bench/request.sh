#!/usr/bin/env bash
set -euo pipefail

HOST="${GLM5_HOST:-127.0.0.1}"
PORT="${GLM5_PORT:-18901}"
MODEL_ID="${GLM5_MODEL_ID:-glm-5.3-flash}"
MODE="${GLM5_MODE:-${MODE:-chat}}"
PROMPT="${1:-${GLM5_PROMPT:-${PROMPT:-请用一句话介绍北京。}}}"
MAX_TOKENS="${GLM5_MAX_TOKENS:-${MAX_TOKENS:-1024}}"
TEMPERATURE="${GLM5_TEMPERATURE:-${TEMPERATURE:-0.0}}"
REASONING_EFFORT="${GLM5_REASONING_EFFORT:-${REASONING_EFFORT:-low}}"

BASE_URL="http://${HOST}:${PORT}"

case "${MODE}" in
  chat)
    PAYLOAD="$(python3 - "${MODEL_ID}" "${PROMPT}" "${MAX_TOKENS}" "${TEMPERATURE}" "${REASONING_EFFORT}" <<'PY'
import json
import sys

model_id, prompt, max_tokens, temperature, reasoning_effort = sys.argv[1:6]
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": float(temperature),
}
if reasoning_effort:
    payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
    curl -sS "${BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "${PAYLOAD}"
    echo ""
    ;;
  completions)
    PAYLOAD="$(python3 - "${MODEL_ID}" "${PROMPT}" "${MAX_TOKENS}" "${TEMPERATURE}" <<'PY'
import json
import sys

model_id, prompt, max_tokens, temperature = sys.argv[1:5]
print(
    json.dumps(
        {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        },
        ensure_ascii=False,
    )
)
PY
)"
    curl -sS "${BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "${PAYLOAD}"
    echo ""
    ;;
  *)
    echo "Error: unsupported MODE='${MODE}' (expected 'chat' or 'completions')." >&2
    exit 1
    ;;
esac
