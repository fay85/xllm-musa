#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Generate all GSM8K responses using the archived sgl-eval request protocol.

This is a serial HTTP adapter, not a replacement for sgl-eval's math grader.
Keep raw requests/responses; official scoring is a separate, reference-only
CPU step. A lightweight boxed-number score is labeled separately.
"""

import argparse
import collections
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import logger


def last_boxed(text: str) -> str | None:
    """Strict balanced-brace extraction; no last-number/#### fallback."""
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        depth += int(text[index] == "{") - int(text[index] == "}")
        if depth == 0:
            return text[opening + 1:index].strip()
    return None


def numeric_value(value: str | int | None) -> Fraction | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace(r"\,", "")
    text = text.strip("$").strip()
    fraction = re.fullmatch(r"\\(?:d?frac)\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
    try:
        if fraction:
            return Fraction(int(fraction[1]), int(fraction[2]))
        if re.fullmatch(r"[-+]?\d+/[-+]?\d+", text):
            return Fraction(text)
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            number = Decimal(text)
            return Fraction(number) if number.is_finite() else None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def numeric_exact(text: str, expected: str | int) -> bool:
    predicted = numeric_value(last_boxed(text))
    target = numeric_value(expected)
    return predicted is not None and target is not None and predicted == target


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_progress(run: Path, records: list[dict], planned: int) -> dict:
    completed = len(records)
    result = {
        "planned": planned, "completed": completed,
        "complete": completed == planned,
        "numeric_boxed_correct": sum(row["numeric_boxed_correct"] for row in records),
        "numeric_boxed_accuracy_completed_only":
            sum(row["numeric_boxed_correct"] for row in records) / completed if completed else None,
        "request_errors": sum("error" in row for row in records),
        "finish_reasons": dict(collections.Counter(row["finish_reason"] for row in records)),
        "grading_note": "provisional strict numeric boxed; official NeMo symbolic score is separate",
    }
    temporary = run / "progress.tmp.json"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(run / "progress.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--thinking-mode", choices=("omitted", "chat"), default="omitted",
                        help="chat adds thinking=false to match SGLang's default chat mode")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import yaml

    dataset = _read_rows(args.dataset)
    if len(dataset) != 1319:
        parser.error("this full-evaluation adapter requires exactly 1319 rows")
    template = yaml.safe_load(args.prompt_yaml.read_text())["user"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()), "dataset_rows": len(dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "prompt_yaml": str(args.prompt_yaml.resolve()),
        "prompt_yaml_sha256": hashlib.sha256(args.prompt_yaml.read_bytes()).hexdigest(),
        "user_template": template, "url": args.url, "model": args.model,
        "temperature": 0, "top_p": 0.95, "max_tokens": 1024,
        "chat_template_kwargs": {"thinking": False} if args.thinking_mode == "chat" else None,
        "thinking_mode_control": args.thinking_mode, "reasoning_effort": None,
        "system_message": None, "seed": None, "n_repeats": 1,
        "client_concurrency": 1, "request_order": "dataset index ascending",
        "timeout_seconds": args.timeout, "transport_retries": 0,
        "reference_in_model_request": False,
    }
    raw = args.run_dir / "responses.raw.jsonl"
    attempts = args.run_dir / "requests.jsonl"
    config_path = args.run_dir / "eval_config.json"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            parser.error("resume configuration differs")
        records = _read_rows(raw)
        if len(_read_rows(attempts)) != len(records):
            parser.error("an unrecorded in-flight attempt exists; inspect before retrying")
        if [row["id"] for row in records] != [f"gsm8k-{i}" for i in range(len(records))]:
            parser.error("resume IDs are not a contiguous unique prefix")
    else:
        if any(path.exists() for path in (raw, attempts, config_path)):
            parser.error("refusing to overwrite evaluation output")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        (args.run_dir / "eval_adapter.snapshot.py").write_bytes(Path(__file__).read_bytes())
        (args.run_dir / "dataset.snapshot.jsonl").write_bytes(args.dataset.read_bytes())
        (args.run_dir / "prompt.snapshot.yaml").write_bytes(args.prompt_yaml.read_bytes())
        records = []
    consecutive_errors = 0
    with raw.open("a", buffering=1) as output, attempts.open("a", buffering=1) as request_log:
        for index in range(len(records), len(dataset)):
            example = dataset[index]
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": template.format(
                    problem=example["problem"], examples="")}],
                "temperature": 0, "top_p": 0.95, "max_tokens": 1024,
            }
            if args.thinking_mode == "chat":
                payload["chat_template_kwargs"] = {"thinking": False}
            started = time.time()
            request_log.write(json.dumps({"id": f"gsm8k-{index}", "payload": payload,
                                          "started": started}, ensure_ascii=False) + "\n")
            row = {"id": f"gsm8k-{index}", "problem": example["problem"],
                   "expected_answer": example["expected_answer"],
                   "request_payload": payload, "generation_start_time": started}
            try:
                request = urllib.request.Request(args.url,
                    data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    body = json.loads(response.read())
                choice = body["choices"][0]
                text = choice["message"].get("content") or ""
                usage = body.get("usage", {})
                row.update(generation=text, response=body, usage=usage,
                           reasoning_content=choice["message"].get("reasoning_content"),
                           finish_reason=choice.get("finish_reason"),
                           num_generated_tokens=usage.get("completion_tokens"),
                           prompt_tokens=usage.get("prompt_tokens"))
                consecutive_errors = 0
            except Exception as error:
                detail = error.read().decode(errors="replace") if isinstance(error, urllib.error.HTTPError) else ""
                row.update(generation="", finish_reason="error",
                           error=f"{type(error).__name__}: {error}", error_body=detail,
                           num_generated_tokens=0, prompt_tokens=0)
                consecutive_errors += 1
            row["generation_end_time"] = time.time()
            row["latency_seconds"] = row["generation_end_time"] - started
            row["numeric_boxed_prediction"] = last_boxed(row["generation"])
            row["numeric_boxed_correct"] = "error" not in row and numeric_exact(
                row["generation"], example["expected_answer"])
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(row)
            progress = _write_progress(args.run_dir, records, len(dataset))
            logger.info("%s completed=%d/1319 numeric_boxed=%d errors=%d finish=%s tokens=%s",
                        row["id"], len(records), progress["numeric_boxed_correct"],
                        progress["request_errors"], row["finish_reason"], row["num_generated_tokens"])
            if consecutive_errors >= 3:
                logger.error("Three consecutive transport/API failures; preserving partial run")
                break
    final = _write_progress(args.run_dir, records, len(dataset))
    print(json.dumps(final, ensure_ascii=False, indent=2))
    raise SystemExit(0 if final["complete"] else 1)


if __name__ == "__main__":
    main()
