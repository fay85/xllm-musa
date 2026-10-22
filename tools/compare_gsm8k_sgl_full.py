#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Audit complete paired GSM8K results without changing answers or scores."""

import argparse
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


def _rows(path: Path) -> dict[str, dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = {f"gsm8k-{index}" for index in range(1319)}
    if len(items) != 1319 or {row["id"] for row in items} != expected:
        raise ValueError(f"not complete unique coverage: {path}")
    return {row["id"]: row for row in items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xllm-scored", type=Path, required=True)
    parser.add_argument("--reference-scored", type=Path, required=True)
    args = parser.parse_args()
    candidate = _rows(args.xllm_scored)
    reference = _rows(args.reference_scored)
    raw = _rows(args.run_dir / "responses.raw.jsonl")
    attempts = [json.loads(line) for line in (args.run_dir / "requests.jsonl").read_text().splitlines()]
    if len(attempts) != 1319 or {row["id"] for row in attempts} != set(raw):
        raise ValueError("not exactly one recorded request per test example")
    config = json.loads((args.run_dir / "eval_config.json").read_text())
    if config["max_tokens"] != 1024 or config["top_p"] != 0.95 or config["temperature"] != 0:
        raise ValueError("sampling protocol mismatch")
    prompt_counts = json.loads((args.run_dir / "reference_prompt_tokens.json").read_text())
    buckets: dict[str, list[str]] = {
        "both_correct": [], "xllm_only_correct": [], "sglang_only_correct": [], "both_incorrect": []}
    cases = []
    prompt_count_mismatches = []
    prompt_count_unavailable = []
    category_counts = collections.Counter()
    prompt_buckets = {name: {"examples": 0, "xllm_correct": 0, "sglang_correct": 0}
                      for name in ("le128", "129to256", "gt256", "unknown")}
    short_incorrect_ids = []
    for index in range(1319):
        key = f"gsm8k-{index}"
        x = candidate[key]
        s = reference[key]
        if x["problem"] != s["problem"] or str(x["expected_answer"]) != str(s["expected_answer"]):
            raise ValueError(f"problem/reference mismatch: {key}")
        if x["generation"] != raw[key]["generation"]:
            raise ValueError(f"scoring changed generation: {key}")
        payload = raw[key]["request_payload"]
        expected_payload = {
            "model": config["model"], "messages": [{"role": "user", "content":
                config["user_template"].format(problem=x["problem"], examples="")}],
            "temperature": 0, "top_p": 0.95, "max_tokens": 1024,
            "chat_template_kwargs": {"thinking": False},
        }
        if payload != expected_payload:
            raise ValueError(f"unexpected extra prompt/control: {key}")
        if "error" in raw[key] or raw[key].get("prompt_tokens") is None:
            prompt_count_unavailable.append(key)
        elif raw[key].get("prompt_tokens") != prompt_counts["prompt_tokens_by_index"][index]:
            prompt_count_mismatches.append(key)
        xc = bool(x["symbolic_correct"])
        sc = bool(s["symbolic_correct"])
        prompt_tokens = raw[key].get("prompt_tokens")
        size_group = ("unknown" if prompt_tokens is None else "le128" if prompt_tokens <= 128
                      else "129to256" if prompt_tokens <= 256 else "gt256")
        prompt_buckets[size_group]["examples"] += 1
        prompt_buckets[size_group]["xllm_correct"] += int(xc)
        prompt_buckets[size_group]["sglang_correct"] += int(sc)
        if not xc and 0 < (x.get("num_generated_tokens") or 0) <= 8:
            short_incorrect_ids.append(key)
        bucket = ("both_correct" if xc and sc else "xllm_only_correct" if xc else
                  "sglang_only_correct" if sc else "both_incorrect")
        buckets[bucket].append(key)
        if not xc:
            category = ("request_error" if "error" in x or x.get("finish_reason") == "error" else
                        "truncated_incorrect" if x.get("finish_reason") == "length" else
                        "no_boxed_answer" if x.get("predicted_answer") is None else
                        "boxed_answer_mismatch")
            category_counts[category] += 1
        else:
            category = None
        if bucket != "both_correct":
            cases.append({"id": key, "bucket": bucket, "failure_category": category,
                          "problem": x["problem"], "expected_answer": x["expected_answer"],
                          "xllm_prediction": x.get("predicted_answer"),
                          "sglang_prediction": s.get("predicted_answer"),
                          "xllm_generation": x["generation"], "sglang_generation": s["generation"],
                          "xllm_finish_reason": x.get("finish_reason"),
                          "xllm_generated_tokens": x.get("num_generated_tokens")})
    xc = len(buckets["both_correct"]) + len(buckets["xllm_only_correct"])
    sc = len(buckets["both_correct"]) + len(buckets["sglang_only_correct"])
    log = (args.run_dir / "rank0.log").read_text(errors="replace")
    # Performance warmup may share this service, so Graph evidence must be
    # inside the GSM8K generation interval, not borrowed from the counting run.
    started = min(row["generation_start_time"] for row in raw.values())
    ended = max(row["generation_end_time"] for row in raw.values())
    launch_lines = (args.run_dir / "launch_manifest.txt").read_text().splitlines()
    launch_time = launch_lines[0]
    launch_fields = dict(line.split("=", 1) for line in launch_lines if "=" in line)
    target_width = int(launch_fields["num_speculative_tokens"]) + 1
    log_timezone = datetime.fromisoformat(launch_time).tzinfo
    if log_timezone is None:
        raise ValueError("launch timestamp must carry the server log timezone")
    evaluation_lines = []
    for line in log.splitlines():
        match = re.match(r"[IWEF](\d{8} \d{2}:\d{2}:\d{2}\.\d{6}) ", line)
        if match is None:
            continue
        timestamp = datetime.strptime(match[1], "%Y%m%d %H:%M:%S.%f").replace(
            tzinfo=log_timezone).timestamp()
        if started <= timestamp <= ended:
            evaluation_lines.append(line)
    evaluation_log = "\n".join(evaluation_lines)
    graph_route_events = evaluation_log.count(
        f"[DSV4_GRAPH_KEY] bucket={target_width} actual_tokens={target_width} query_width={target_width}")
    q6_route_events = evaluation_log.count("[DSV4_GRAPH_KEY] bucket=6 actual_tokens=6 query_width=6")
    result = {
        "complete": True, "examples": 1319, "unique_ids": 1319,
        "recorded_attempts": len(attempts), "request_payloads_match_protocol": True,
        "xllm_correct": xc, "xllm_accuracy_percent": 100 * xc / 1319,
        "sglang_correct": sc, "sglang_accuracy_percent": 100 * sc / 1319,
        "difference_percentage_points": 100 * (xc - sc) / 1319,
        "paired_counts": {name: len(ids) for name, ids in buckets.items()},
        "paired_ids": buckets, "xllm_failure_categories": dict(category_counts),
        "prompt_length_groups": prompt_buckets,
        "incorrect_with_le8_output_tokens": short_incorrect_ids,
        "prompt_token_count_mismatches": prompt_count_mismatches,
        "prompt_token_usage_unavailable_ids": prompt_count_unavailable,
        "xllm_total_prompt_tokens": sum(row.get("prompt_tokens", 0) or 0 for row in raw.values()),
        "reference_total_prompt_tokens": prompt_counts["total_tokens"],
        "xllm_request_errors": sum("error" in row for row in raw.values()),
        "xllm_truncated": sum(row.get("finish_reason") == "length" for row in raw.values()),
        "graph_q6_timed_replay_observed":
            "actual_num_tokens=6 padded_num_tokens=6 replay_ms=" in evaluation_log,
        "graph_q6_route_events_during_evaluation": q6_route_events,
        "graph_q6_route_observed": q6_route_events > 0,
        "graph_target_query_width": target_width,
        "graph_target_route_events_during_evaluation": graph_route_events,
        "graph_target_route_observed": graph_route_events > 0,
        "graph_evidence_note": (
            "Only the GSM8K time interval is counted. DSV4_GRAPH_KEY is emitted "
            "after eager/fallback predicates, immediately before graph lookup "
            "and replay or capture+replay in the recorded executor source. "
            "Timed replay additionally requires diagnostic synchronization; "
            "its absence alone does not mean Graph is disabled."),
        "forced_eager_compact_count": evaluation_log.count("target compact verify forced eager"),
        "raw_sha256": hashlib.sha256((args.run_dir / "responses.raw.jsonl").read_bytes()).hexdigest(),
        "scope_note": "xLLM Graph+DSpark B1 serial vs historical SGLang Graph target-only client32; no speed comparison",
    }
    for name in ["comparison.json", "disagreement_cases.jsonl"]:
        if (args.run_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {name}")
    (args.run_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.run_dir / "disagreement_cases.jsonl").open("w") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "paired_ids"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
