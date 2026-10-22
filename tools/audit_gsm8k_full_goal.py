#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Audit one complete Graph+DSpark GSM8K run and its same-service hot replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-perf", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--unit-log", type=Path, required=True)
    parser.add_argument("--min-accuracy-percent", type=float, default=96.0)
    parser.add_argument("--max-tpot-ms", type=float, default=13.0)
    parser.add_argument("--max-ttft-ms", type=float, default=250.0)
    parser.add_argument("--expected-unit-tests", type=int, default=14)
    parser.add_argument("--expected-draft-tokens", type=int, default=5)
    parser.add_argument(
        "--permit-scored-truncation", action="store_true",
        help="Use the requested accuracy gate without additionally demanding zero length finishes; "
             "retain every truncated response and its unchanged original score, and report strict gate separately.")
    args = parser.parse_args()
    if not (0 < args.min_accuracy_percent <= 100 and args.max_tpot_ms > 0
            and args.max_ttft_ms > 0 and args.expected_unit_tests >= 14
            and 1 <= args.expected_draft_tokens <= 16):
        parser.error("invalid acceptance thresholds")
    run = args.run_dir.resolve()
    output = run / "acceptance.json"
    if not run.is_relative_to(ROOT / "logs") or output.exists():
        parser.error("a new acceptance output inside this checkout's logs is required")
    launch = (run / "launch_manifest.txt").read_text()
    fields = dict(line.split("=", 1) for line in launch.splitlines() if "=" in line)
    manifest = _json(args.candidate_dir / "manifest.json")
    binary = args.candidate_dir / "xllm"
    binary_sha = _sha(binary)
    config = _json(run / "eval_config.json")
    scored = _rows(run / "xllm.scored.jsonl")
    raw = _rows(run / "responses.raw.jsonl")
    attempts = _rows(run / "requests.jsonl")
    summary = _json(run / "xllm.summary.json")
    comparison = _json(run / "comparison.json")
    reference = _json(args.reference_summary)
    ids = [f"gsm8k-{index}" for index in range(1319)]
    correct = sum(row["symbolic_correct"] for row in scored)
    perf_dir = run / "perf_after_gsm_formal"
    perf = [_json(perf_dir / f"replay_{index}.json") for index in range(2, 8)]
    perf_summary = _json(perf_dir / "summary.json")
    baseline = _json(args.baseline_perf)
    ttft = statistics.mean(row["ttft_ms"] for row in perf)
    tpot = statistics.mean(row["tpot_ms"] for row in perf)
    expected_counting = [str(value) for value in range(1, 751)]
    unit_log = args.unit_log.read_text()
    checks = {
        "single_recorded_binary": binary_sha == manifest["binary_sha256"]
            and binary_sha in launch
            and Path(fields["xllm_bin"]).resolve() == binary.resolve(),
        "compiled_sources_unchanged": all(_sha(ROOT / row["path"]) == row["sha256"]
            for row in manifest["sources"]),
        "recorded_kernel_archive_unchanged": _sha(Path(manifest["kernel_archive"]))
            == manifest["kernel_archive_sha256"],
        "required_graph_draft_configuration": all(fields.get(key) == value for key, value in {
            "topology": "TP8/EP1/DP1/CP1", "graph": "true", "disable_draft": "false",
            "num_speculative_tokens": str(args.expected_draft_tokens), "speculative_algorithm": "DSpark",
            "XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY": "1",
            "XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER": "0",
            "model_path": "/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt",
        }.items()),
        "exactly_one_attempt_for_each_of_1319": all(
            [row["id"] for row in records] == ids for records in (raw, attempts, scored)),
        "scoring_did_not_change_generation": all(
            a["generation"] == b["generation"] for a, b in zip(raw, scored)),
        "complete_original_reference_grader": summary["complete"]
            and not summary["partial_diagnostic"]
            and summary["scorer_source_sha256"] == reference["scorer_source_sha256"],
        "raw_evidence_hashes_match": _sha(run / "responses.raw.jsonl")
            == summary["input_sha256"] == comparison["raw_sha256"],
        "dataset_unchanged": _sha(Path(config["dataset"])) == config["dataset_sha256"]
            == summary["dataset_sha256"] == reference["dataset_sha256"],
        "unaltered_request_protocol": comparison["request_payloads_match_protocol"]
            and config["transport_retries"] == 0
            and not config["reference_in_model_request"],
        "accuracy_meets_requested_threshold": correct >= math.ceil(1319 * args.min_accuracy_percent / 100)
            and correct == summary["symbolic_correct"] == comparison["xllm_correct"],
        "no_errors_or_truncation": summary["request_errors"] == summary["truncated"] == 0,
        "no_request_errors": summary["request_errors"] == 0,
        "truncation_count_matches_raw": summary["truncated"]
            == sum(row.get("finish_reason") == "length" for row in raw),
        "matching_prompt_token_counts": not comparison["prompt_token_count_mismatches"]
            and not comparison["prompt_token_usage_unavailable_ids"],
        "graph_target_during_gsm8k_without_forced_eager":
            comparison.get("graph_target_query_width", 6) == args.expected_draft_tokens + 1
            and comparison.get("graph_target_route_events_during_evaluation",
                               comparison["graph_q6_route_events_during_evaluation"]) > 0
            and comparison["forced_eager_compact_count"] == 0,
        "six_strict_hot_requests": all(row["content"].split() == expected_counting
            and row["finish_reason"] == "length"
            and row["usage"]["prompt_tokens"] == 2500
            and row["usage"]["completion_tokens"] == 1500
            and row["usage"]["prompt_tokens_details"]["cached_tokens"] == 1536 for row in perf),
        "hot_ttft_below_requested_limit": 0 < ttft < args.max_ttft_ms,
        "hot_tpot_below_requested_limit": 0 < tpot < args.max_tpot_ms,
        "perf_summary_matches_raw": abs(ttft - perf_summary["ttft_avg_ms"]) < 1e-8
            and abs(tpot - perf_summary["tpot_avg_ms"]) < 1e-8
            and abs(9000 / perf_summary["duration_seconds"] - perf_summary["output_tgs"]) < 1e-8,
        "at_least_90_percent_historical_hot_throughput":
            perf_summary["output_tgs"] >= 0.9 * baseline["output_tgs"],
        "expected_kernel_tests_including_hc_post":
            f"[  PASSED  ] {args.expected_unit_tests} tests." in unit_log
            and re.search(r"\[\s*(FAILED|SKIPPED)\s*\]", unit_log) is None
            and "[       OK ] DeepseekV4HCPostTest.GraphReplayWithAliasedResidual" in unit_log,
    }
    # The user's percentage target permits wrong answers at the fixed token
    # limit. Zero truncations is a separate, stricter quality condition; never
    # remove these rows, retry them, or replace their scores to meet that check.
    strict_gate = "PASS" if all(checks.values()) else "FAIL"
    required_checks = {name: passed for name, passed in checks.items()
                       if not (args.permit_scored_truncation
                               and name == "no_errors_or_truncation")}
    result = {
        "goal_gate": "PASS" if all(required_checks.values()) else "FAIL",
        "strict_zero_truncation_gate": strict_gate,
        "checks": checks,
        "required_check_names": list(required_checks),
        "truncated_ids": [row["id"] for row in raw if row.get("finish_reason") == "length"],
        "requested_thresholds": {
            "min_accuracy_percent": args.min_accuracy_percent,
            "max_tpot_ms_exclusive": args.max_tpot_ms,
            "max_ttft_ms_exclusive": args.max_ttft_ms,
            "expected_unit_tests": args.expected_unit_tests,
            "expected_draft_tokens": args.expected_draft_tokens,
            "target_query_width": args.expected_draft_tokens + 1,
            "require_zero_truncations_in_addition_to_accuracy": not args.permit_scored_truncation,
        },
        "binary_sha256": binary_sha, "correct": correct, "examples": len(raw),
        "accuracy_percent": 100 * correct / 1319,
        "ttft_avg_ms": ttft, "tpot_avg_ms": tpot,
        "output_tgs_8gpu": perf_summary["output_tgs"],
        "total_tgs_8gpu": perf_summary["total_tgs"],
        "historical_hot_throughput_retained_percent":
            100 * perf_summary["output_tgs"] / baseline["output_tgs"],
        "remaining_sglang_only_ids": comparison["paired_ids"]["sglang_only_correct"],
        "scope": "single full1319 boxed/NeMo pass plus same-service B1 2500/1500 Prefix0.65; not all cases correct",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    logger.info("Full goal evidence audit: %s (%s)", result["goal_gate"], output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["goal_gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
