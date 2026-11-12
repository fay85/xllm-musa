# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a 2048-token multi-chunk PD request, independently of performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import requests
from tokenizers import Tokenizer


def _render(content: str) -> str:
    return "<｜begin▁of▁sentence｜><｜User｜>" + content + "<｜Assistant｜></think>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    short = "只输出计算结果，不要解释：7 × 8 = ?"
    short_count = len(tokenizer.encode(_render(short), add_special_tokens=False).ids)
    if short_count != 17:
        raise ValueError(
            f"Template/tokenizer differs from verified short usage: {short_count} != 17"
        )
    prefix = "下面的占位内容与问题无关，请忽略它，不要复述。\n"
    prompt = None
    token_ids = []
    for padding in range(1800, 2100):
        candidate = prefix + " x" * padding + "\n" + short
        encoded = tokenizer.encode(_render(candidate), add_special_tokens=False).ids
        if len(encoded) == 2048:
            prompt, token_ids = candidate, encoded
            break
    if prompt is None:
        raise ValueError(
            "No exact 2048-token candidate; inspect tokenizer rather than guessing length"
        )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking_mode": "chat"},
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    (args.output / "request.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2)
    )
    (args.output / "rendered_token_ids.json").write_text(json.dumps(token_ids))
    started = time.monotonic()
    content, finish, usage = "", None, None
    with requests.post(
        args.url.rstrip("/") + "/v1/chat/completions",
        json=body,
        stream=True,
        timeout=(10, 900),
    ) as response:
        (args.output / "http.json").write_text(
            json.dumps(
                {"status": response.status_code, "headers": dict(response.headers)},
                indent=2,
            )
        )
        response.raise_for_status()
        with (args.output / "sse.jsonl").open("x") as stream:
            for raw in response.iter_lines():
                if not raw:
                    continue
                text = raw.decode()
                stream.write(
                    json.dumps(
                        {"elapsed_s": time.monotonic() - started, "raw": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                if text == "data: [DONE]":
                    break
                if not text.startswith("data: "):
                    continue
                event = json.loads(text[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content += choice.get("delta", {}).get("content") or ""
                    finish = choice.get("finish_reason") or finish
    passed = (
        content.strip() == "56"
        and finish == "stop"
        and usage is not None
        and usage.get("prompt_tokens") == 2048
    )
    result = {
        "status": "PASS_2048_INPUT_SHORT_ANSWER" if passed else "FAIL",
        "expected": "56",
        "content": content,
        "finish_reason": finish,
        "usage": usage,
        "elapsed_s": time.monotonic() - started,
        "scope": "single long multi-chunk correctness request; max_tokens16, not formal 2048/256 performance or full accuracy",
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    logger.info("Long PD gate answer=%r usage=%s pass=%s", content, usage, passed)
    if not passed:
        raise AssertionError(result)


if __name__ == "__main__":
    main()
