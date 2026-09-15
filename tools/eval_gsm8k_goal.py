#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Run a deterministic GSM8K subset against an OpenAI-compatible endpoint."""

import argparse
import hashlib
import json
import os
import random
import re
import time
import sys
import urllib.error
import urllib.request
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        if "/" in value:
            return str(Fraction(value))
        number = Decimal(value)
        if not number.is_finite():
            return None
        if number == number.to_integral_value():
            return str(number.to_integral_value())
        return format(number.normalize(), "f")
    except (ValueError, ArithmeticError):
        return value


def _get_reference(answer: str) -> str | None:
    marker = re.search(r"####\s*([^\n]+)", answer)
    values = NUMBER_RE.findall(marker.group(1)) if marker else []
    return _normalize(values[-1]) if values else None


def _get_prediction(content: str) -> str | None:
    markers = re.findall(r"####\s*([^\n]+)", content)
    if markers:
        values = NUMBER_RE.findall(markers[-1])
        if values:
            return _normalize(values[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", content)
    if boxed:
        values = NUMBER_RE.findall(boxed[-1])
        if values:
            return _normalize(values[-1])
    values = NUMBER_RE.findall(content)
    return _normalize(values[-1]) if values else None


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=("none", "high", "max"), default="high")
    parser.add_argument("--prompt-style", choices=("baseline", "goal"), default="baseline")
    parser.add_argument("--enable-thinking", choices=("true", "false"), default="false")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--extra-instruction", default="")
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x]
    else:
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.samples))
    if not indices or len(set(indices)) != len(indices):
        parser.error("indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= len(rows):
        parser.error("index outside dataset")
    artifacts = [args.output, args.output.with_suffix('.jsonl'),
                 args.output.with_suffix('.config.json')]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("Refusing to overwrite experiment output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(indices=indices, dataset_sha256=hashlib.sha256(data).hexdigest(),
                  dataset_rows=len(rows), scoring="last ####, boxed, then last numeric; normalized exact match",
                  oracle_retries=False, temperature=0, top_p=1)
    artifacts[2].write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    results = []
    for index in indices:
        row = rows[index]
        if args.prompt_style == "baseline":
            prompt = (
                "Solve the following grade-school math problem. Preserve every number "
                "and condition from the question. Verify the arithmetic, then put only "
                "the final numeric result after `####`.\n\nQuestion: " + row["question"]
            )
        else:
            prompt = (
                "Solve this grade-school math problem. Preserve every number and condition. "
                "Calculate carefully and end with #### followed by one numeric answer. "
                "Do not discuss another problem.\n\nQuestion: " + row["question"]
            )
        if args.extra_instruction:
            prompt = args.extra_instruction + "\n\n" + prompt
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "chat_template_kwargs": {
                "thinking_mode": "chat",
                "reasoning_effort": args.reasoning_effort,
                "enable_thinking": args.enable_thinking == "true",
            },
        }
        started = time.monotonic()
        record = {"dataset_index": index, "reference": _get_reference(row["answer"]),
                  "question": row["question"], "request_payload": payload}
        try:
            response = _post_json(args.url, payload, args.timeout)
            content = response["choices"][0]["message"].get("content", "")
            predicted = _get_prediction(content)
            record.update(
                prediction=predicted, correct=predicted is not None and predicted == record["reference"],
                content=content, response=response,
                finish_reason=response["choices"][0].get("finish_reason"),
                usage=response.get("usage"), latency_seconds=round(time.monotonic() - started, 3))
        except urllib.error.HTTPError as error:
            record.update(correct=False, prediction=None, error=repr(error),
                          error_body=error.read().decode(errors="replace"),
                          latency_seconds=round(time.monotonic() - started, 3))
        except Exception as error:
            record.update(correct=False, prediction=None, error=repr(error), latency_seconds=round(time.monotonic() - started, 3))
        results.append(record)
        with artifacts[1].open('a') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"[{len(results):03d}/{len(indices)}] {'PASS' if record['correct'] else 'FAIL'} "
                    f"index={index} pred={record.get('prediction')} ref={record['reference']}")
        if "error" in record:
            logger.error("Stopping on transport error; run is incomplete, not a model accuracy result")
            break
    correct = sum(item["correct"] for item in results)
    errors = sum("error" in item for item in results)
    complete = len(results) == len(indices) and errors == 0
    summary = {"config": config, "indices": indices, "correct": correct,
               "incorrect": len(results) - correct, "request_errors": errors,
               "completed_requests": len(results), "planned_requests": len(indices),
               "accuracy": correct / len(results) if complete else None,
               "complete": complete, "reasoning_effort": args.reasoning_effort}
    args.output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()
