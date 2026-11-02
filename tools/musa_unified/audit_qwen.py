# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Audit Qwen benchmark bodies and per-request server token/cache evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _stats(logs: dict[str, str], request_id: str) -> tuple[str, dict]:
    pattern = re.compile(re.escape(request_id) + r"(?=[,\s])")
    hits = [
        (name, line)
        for name, text in logs.items()
        for line in text.splitlines()
        if pattern.search(line) and "request.cpp:" in line and "generated_tokens:" in line
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one completion record for {request_id}; found {len(hits)}")
    name, line = hits[0]
    values = {
        key: int(value)
        for key, value in re.findall(r"\b(prompt_tokens|num_prefix_cache_tokens|generated_tokens): (\d+)", line)
    }
    reason = re.search(r"finish_reason: ([^,\s]+)", line)
    values["finish_reason"] = reason.group(1) if reason else None
    return name, values


def audit(result: dict, prompts: dict, p_logs: dict[str, str], d_logs: dict[str, str]) -> dict:
    rows = []
    requests = result["requests"]
    if len(requests) != 60 or result["summary"]["measured_requests"] != 60:
        raise ValueError("This audit expects the stock bench.sh 60-request formal window")
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("Duplicate request IDs")
    for index, request in enumerate(requests):
        p_name, prefill = _stats(p_logs, request["request_id"])
        d_name, decode = _stats(d_logs, request["request_id"])
        body = request.get("text")
        checks = dict(
            input_tokens=prefill["prompt_tokens"] == prompts["prompt_lengths"][index + 8],
            cached_tokens=prefill["num_prefix_cache_tokens"] == 16384,
            output_tokens=decode["generated_tokens"] == request["completion_tokens"] == 100,
            finish=decode["finish_reason"] == "length",
            body=isinstance(body, str) and bool(body.strip()) and re.search(r"!{8,}", body) is None,
        )
        rows.append(
            dict(
                request_id=request["request_id"],
                passed=all(checks.values()),
                checks=checks,
                prefill=p_name,
                decode=d_name,
                prefill_stats=prefill,
                decode_stats=decode,
            )
        )
    return dict(
        passed=all(row["passed"] for row in rows) and result["summary"]["server_errors"] == 0,
        passed_count=sum(row["passed"] for row in rows),
        requests=rows,
        steady_no_new_capture=result["summary"]["lazy_captures_during_measurement"] == 0,
        benchmark=result["summary"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--server-log-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    p_logs = {path.name: path.read_text(errors="replace") for path in args.server_log_dir.glob("prefill*_r0.log")}
    d_logs = {path.name: path.read_text(errors="replace") for path in args.server_log_dir.glob("decode*.log")}
    results = {str(path): audit(json.loads(path.read_text()), prompts, p_logs, d_logs) for path in args.result}
    qa = None
    if args.qa_dir:
        qa = json.loads((args.qa_dir / "summary.json").read_text())
        qa["bad_body_files"] = []
        for path in args.qa_dir.glob("result_*.json"):
            body = json.loads(path.read_text()).get("result", {}).get("text", "")
            if not body.strip() or re.search(r"!{8,}", body):
                qa["bad_body_files"].append(path.name)
    passed = all(result["passed"] for result in results.values())
    if qa is not None:
        passed = passed and qa["passed"] == qa["total"] == 68 and not qa["bad_body_files"]
    with args.out.open("x") as stream:
        json.dump(dict(passed=passed, results=results, qa=qa), stream, indent=2)
    logger.info("Qwen body/token/cache audit passed=%s; result=%s", passed, args.out)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
