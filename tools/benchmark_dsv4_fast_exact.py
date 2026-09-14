#!/usr/bin/env python3
"""Fast exact-length single-request probe for long DSV4 prompts."""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
    p.add_argument("--prompt-tokens", type=int, required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--count-to", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True,
                                        local_files_only=True)
    instruction = (f"Count from 1 to {a.count_to}, separated by spaces. "
                   "Output only the numbers.")
    bos = "<｜begin▁of▁sentence｜>"
    user = "<｜User｜>"
    assistant = "<｜Assistant｜>"

    def ids(content: str) -> list[int]:
        rendered = bos + user + content + assistant + "</think>"
        return tok.encode(rendered, add_special_tokens=False)

    base = len(ids(instruction))
    filler = max(0, a.prompt_tokens - base)
    for _ in range(8):
        content = ("x " * filler) + instruction
        delta = a.prompt_tokens - len(ids(content))
        filler = max(0, filler + delta)
        if delta == 0:
            break
    content = ("x " * filler) + instruction
    actual = len(ids(content))
    if actual != a.prompt_tokens:
        raise RuntimeError(f"cannot build exact prompt: {actual} != {a.prompt_tokens}")
    payload = {"model": a.model, "messages": [{"role": "user", "content": content}],
               "chat_template_kwargs": {"enable_thinking": False},
               "temperature": 0, "max_tokens": a.output_tokens,
               "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    start = time.perf_counter(); first = last = None; chunks = []; usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                text = choice.get("delta", {}).get("content")
                if text:
                    now = time.perf_counter(); first = first or now; last = now
                    chunks.append(text)
    if first is None or last is None:
        result = {"error": "stream completed without a content token",
                  "prompt_tokens_built": actual,
                  "output_tokens_requested": a.output_tokens,
                  "count_to": a.count_to}
    else:
        text = "".join(chunks)
        nums = [int(x) for x in re.findall(r"\d+", text)]
        expected = list(range(1, a.count_to + 1))
        result = {"model": a.model, "model_path": str(a.model_path),
                  "request_payload": payload, "url": a.url,
                  "prompt_tokens_built": actual,
                  "output_tokens_requested": a.output_tokens,
                  "count_to": a.count_to,
                  "metrics": {"usage": usage, "finish_reason": finish,
                              "completion_tokens": usage.get("completion_tokens"),
                              "completion_tokens_source": "usage",
                              "ttft_ms": round((first - start) * 1000, 3),
                              "e2e_ms": round((time.perf_counter() - start) * 1000, 3),
                              "tpot_ms": round((last - first) * 1000 /
                                               max(usage.get("completion_tokens", 1) - 1, 1), 3)},
                  "accuracy": {"passed": nums == expected,
                               "strict_text_passed": text.split() == [str(x) for x in expected],
                               "expected_count": len(expected), "actual_count": len(nums)},
                  "content": text, "content_prefix": text[:512], "content_suffix": text[-512:]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result.get(k) for k in ("error", "prompt_tokens_built",
                                                  "output_tokens_requested", "metrics", "accuracy")},
                     ensure_ascii=False, indent=2))
    raise SystemExit(1 if "error" in result else 0)


if __name__ == "__main__":
    main()
