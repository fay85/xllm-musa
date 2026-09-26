#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Audit the joint fixed-100 and q6 hot-window goal, not counting alone."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from eval_gsm8k_goal import _get_prediction, _get_reference


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("baseline_results", type=Path)
    args = parser.parse_args()
    checks: dict[str, bool] = {}
    launch = (args.run_dir / "launch_manifest.txt").read_text()
    fields = dict(line.split("=", 1) for line in launch.splitlines() if "=" in line)
    manifest = json.loads((args.candidate_dir / "manifest.json").read_text())
    binary = args.candidate_dir / "xllm"
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    checks["same_binary"] = (
        Path(fields["xllm_bin"]).resolve() == binary.resolve()
        and binary_sha == manifest["binary_sha256"] and binary_sha in launch)
    checks["q6_graph_draft_configuration"] = all(fields.get(key) == value for key, value in {
        "num_speculative_tokens": "5", "disable_draft": "false",
        "graph": "true", "prefix_cache": "true", "topology": "TP8/EP1/DP1/CP1",
        "XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY": "1",
        "XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER": "0",
    }.items())
    accuracy = _records(args.run_dir / "fixed100.jsonl")
    config = json.loads((args.run_dir / "fixed100.config.json").read_text())
    baseline = _records(args.baseline_results)
    dataset_bytes = Path(config["dataset"]).read_bytes()
    dataset = [json.loads(line) for line in dataset_bytes.splitlines()]
    checks["fixed_100_indices_and_dataset"] = (
        len(accuracy) == len(baseline) == 100
        and [row["dataset_index"] for row in accuracy] == config["indices"]
        and config["indices"] == [row["dataset_index"] for row in baseline]
        and hashlib.sha256(dataset_bytes).hexdigest() == config["dataset_sha256"])
    correct = 0
    failures = []
    aligned = True
    for row, old in zip(accuracy, baseline):
        original = dataset[row["dataset_index"]]
        reference = _get_reference(original["answer"])
        aligned = aligned and row["question"] == old["question"] == original["question"]
        aligned = aligned and str(row["reference"]) == str(old["reference"]) == reference
        payload = row["request_payload"]
        aligned = aligned and payload["temperature"] == 0 and payload["max_tokens"] == 2048
        aligned = aligned and payload["top_p"] == 1
        aligned = aligned and row.get("usage", {}).get("prompt_tokens") == old["usage"]["prompt_tokens"]
        passed = "error" not in row and _get_prediction(row.get("content", "")) == reference
        correct += int(passed)
        if not passed:
            failures.append(row["dataset_index"])
    checks["aligned_questions_references_and_inputs"] = aligned
    checks["accuracy_95_of_100"] = correct >= 95 and len(accuracy) == 100
    checks["no_request_errors"] = not any("error" in row for row in accuracy)
    perf = [json.loads((args.run_dir / "formal_2_7" / f"replay_{i}.json").read_text())
            for i in range(2, 8)]
    expected = [str(value) for value in range(1, 751)]
    checks["six_strict_hot_requests"] = all(
        row.get("content", "").split() == expected
        and row.get("finish_reason") == "length"
        and row.get("usage", {}).get("prompt_tokens") == 2500
        and row.get("usage", {}).get("completion_tokens") == 1500
        and row.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") == 1536
        and all(math.isfinite(row.get(key, math.nan)) and row.get(key, 0) > 0
                for key in ("ttft_ms", "tpot_ms", "e2e_ms")) for row in perf)
    ttft = statistics.mean(row["ttft_ms"] for row in perf)
    tpot = statistics.mean(row["tpot_ms"] for row in perf)
    checks["ttft_lt_250_ms"] = ttft < 250
    checks["tpot_lt_13_ms"] = tpot < 13
    server_log = (args.run_dir / "rank0.log").read_text(errors="replace")
    checks["q6_replay_without_short_eager"] = (
        "actual_num_tokens=6 padded_num_tokens=6 replay_ms=" in server_log
        and "target compact verify forced eager" not in server_log)
    result = {
        "goal_gate": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "binary_sha256": binary_sha,
        "correct": correct, "total_questions": len(accuracy),
        "failed_indices": failures, "ttft_avg_ms": ttft, "tpot_avg_ms": tpot,
        "scope": "fixed GSM8K100 and B1 2500/1500 hot variants2-7, same launch",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["goal_gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
