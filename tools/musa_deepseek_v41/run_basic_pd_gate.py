# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two short deterministic native PD smoke questions with complete SSE logs."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "launch.json").exists():
        raise ValueError("Existing attempt: inspect results instead of overwriting")
    (args.output / "launch.json").write_text(
        json.dumps(
            {
                "started_at": datetime.datetime.now().astimezone().isoformat(),
                "url": args.url,
                "model": args.model,
                "purpose": "basic integration only, not formal performance",
            },
            indent=2,
        )
    )
    records = []
    for index, (question, expected) in enumerate(
        (
            ("只输出计算结果，不要解释：1.3 + 2.9 = ?", "4.2"),
            ("只输出计算结果，不要解释：7 × 8 = ?", "56"),
        ),
        1,
    ):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "top_p": 1,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking_mode": "chat"},
            "max_tokens": 96,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        (args.output / f"question{index}_request.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2)
        )
        started = time.monotonic()
        content, reasoning, finish, usage = "", "", None, None
        with requests.post(
            args.url.rstrip("/") + "/v1/chat/completions",
            json=body,
            stream=True,
            timeout=(10, 600),
        ) as response:
            (args.output / f"question{index}_http.json").write_text(
                json.dumps(
                    {"status": response.status_code, "headers": dict(response.headers)},
                    indent=2,
                )
            )
            if response.status_code != 200:
                (args.output / f"question{index}_http_error.txt").write_text(
                    response.text
                )
            response.raise_for_status()
            with (args.output / f"question{index}_sse.jsonl").open("x") as stream:
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    text = raw.decode("utf-8")
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
                        delta = choice.get("delta", {})
                        content += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                        finish = choice.get("finish_reason") or finish
        record = {
            "id": index,
            "question": question,
            "expected": expected,
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish,
            "usage": usage,
            "elapsed_s": time.monotonic() - started,
            "passed": content.strip() == expected and finish == "stop",
        }
        records.append(record)
        (args.output / "answers.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2)
        )
        logger.info(
            "Basic PD question%d answer=%r expected=%s pass=%s",
            index,
            content,
            expected,
            record["passed"],
        )
        if not record["passed"]:
            raise AssertionError(record)
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS_TWO_ARITHMETIC_QUESTIONS_ONLY",
                "answers": records,
                "scope": "does not prove full model accuracy or PD cache transfer; verify all-rank logs separately",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
