#!/usr/bin/env python3
"""Merge primary and reference-free repair records, then score the result."""

import argparse
import json
from pathlib import Path
import statistics


def load_latest(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["dataset_index"]] = record
    return records


def percentile95(values):
    if len(values) <= 1:
        return values[0] if values else 0
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--repair", action="append", default=[])
    parser.add_argument(
        "--audit",
        help="Current arithmetic audit JSONL; stale arithmetic repairs are ignored.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    primary = load_latest(args.primary)
    repairs = [(str(Path(path).resolve()), load_latest(path)) for path in args.repair]
    current_arithmetic_indices = None
    if args.audit:
        current_arithmetic_indices = {
            json.loads(line)["dataset_index"]
            for line in Path(args.audit).read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("has_arithmetic_discrepancy")
        }
    output = []
    for index in sorted(primary):
        source = primary[index]
        # A timed-out request is retained by the resumable evaluator as a
        # checkpoint placeholder.  It is retried on the next service segment
        # and must not be scored as a completed answer in the interim merge.
        if source.get("error") or source.get("selected_stage") == "transport-error":
            continue
        prediction = source.get("prediction")
        selected_stage = source.get("selected_stage", "primary")
        repair_latency = 0.0
        applied_repair = None
        for repair_path, candidates in repairs:
            candidate = candidates.get(index)
            if candidate is None or not candidate.get("selected", False):
                continue
            # The audit is regenerated after parser/rule changes.  A selected
            # arithmetic candidate from an older run must not survive when
            # the current audit no longer flags its primary response.
            if (
                current_arithmetic_indices is not None
                and candidate.get("repair_strategy_version") is not None
                and index not in current_arithmetic_indices
            ):
                continue
            if candidate.get("selection_uses_reference", False):
                raise ValueError(
                    f"repair selection unexpectedly uses reference: {repair_path}"
                )
            prediction = candidate.get("prediction")
            selected_stage = f"repair:{Path(repair_path).name}"
            repair_latency += candidate.get("latency_seconds", 0)
            applied_repair = {
                "path": repair_path,
                "prediction": prediction,
                "latency_seconds": candidate.get("latency_seconds", 0),
            }
        merged = {
            "dataset_index": index,
            "question": source.get("question"),
            "primary_prediction": source.get("prediction"),
            "prediction": prediction,
            "reference": source.get("reference"),
            "correct": prediction == source.get("reference"),
            "selected_stage": selected_stage,
            "primary_latency_seconds": source.get("latency_seconds", 0),
            "repair_latency_seconds": round(repair_latency, 3),
            "total_model_latency_seconds": round(
                source.get("latency_seconds", 0) + repair_latency, 3
            ),
            "applied_repair": applied_repair,
            "selection_uses_reference": False,
        }
        output.append(merged)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )
    latencies = [r["total_model_latency_seconds"] for r in output]
    primary_correct_became_wrong = sum(
        r["primary_prediction"] == r["reference"] and not r["correct"]
        for r in output
    )
    primary_wrong_became_correct = sum(
        r["primary_prediction"] != r["reference"] and r["correct"]
        for r in output
    )
    changed_predictions = sum(
        r["primary_prediction"] != r["prediction"] for r in output
    )
    summary = {
        "primary": str(Path(args.primary).resolve()),
        "repairs": [path for path, _ in repairs],
        "records": len(output),
        "correct": sum(r["correct"] for r in output),
        "incorrect": sum(not r["correct"] for r in output),
        "accuracy": sum(r["correct"] for r in output) / len(output) if output else 0,
        "repairs_applied": sum(r["applied_repair"] is not None for r in output),
        "changed_predictions": changed_predictions,
        "primary_wrong_became_correct": primary_wrong_became_correct,
        "primary_correct_became_wrong": primary_correct_became_wrong,
        "total_model_latency_seconds": round(sum(latencies), 3),
        "average_model_latency_seconds": (
            round(statistics.mean(latencies), 3) if latencies else 0
        ),
        "p50_model_latency_seconds": (
            round(statistics.median(latencies), 3) if latencies else 0
        ),
        "p95_model_latency_seconds": round(percentile95(latencies), 3),
        "effective_questions_per_minute": (
            round(60 * len(output) / sum(latencies), 3)
            if latencies and sum(latencies)
            else 0
        ),
        "failed_indices": [r["dataset_index"] for r in output if not r["correct"]],
        "selection_uses_reference": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
