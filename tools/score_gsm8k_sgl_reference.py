#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Score saved generations with the existing sgl-eval/NeMo CPU evaluator.

Run in the read-only reference container with PYTHONDONTWRITEBYTECODE=1.
Only output files below the supplied xLLM run directory are created.
"""

import argparse
import asyncio
import collections
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import logger
from eval_gsm8k_sgl_protocol import numeric_exact


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def _score(args: argparse.Namespace) -> None:
    from sgl_eval._vendored.nemo_skills.evaluator.math import MathEvaluator
    from sgl_eval._vendored.nemo_skills import math_grader

    dataset = _rows(args.dataset)
    # Read once: partial diagnostics can inspect a growing generation file.
    # The hash must describe the bytes actually parsed, not a later append.
    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line]
    if args.limit is not None:
        if not args.partial_diagnostic or args.limit <= 0 or len(rows) < args.limit:
            raise ValueError("limit requires a partial diagnostic with enough saved rows")
        rows = rows[:args.limit]
    if len(dataset) != 1319 or not rows:
        raise ValueError("expected the complete source dataset and nonempty responses")
    if len(rows) != 1319 and not args.partial_diagnostic:
        raise ValueError("refusing to call a partial run a full evaluation")
    valid_ids = {f"gsm8k-{i}" for i in range(1319)}
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or not ids.issubset(valid_ids):
        raise ValueError("duplicate or unknown IDs")
    if not args.partial_diagnostic and ids != valid_ids:
        raise ValueError("IDs are not unique complete GSM8K coverage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / f"{args.label}.scored.jsonl"
    summary_path = args.output_dir / f"{args.label}.summary.json"
    if scored_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite score evidence")
    evaluator = MathEvaluator(config={})
    scored = []
    old_score_changes = []
    with scored_path.open("w", buffering=1) as output:
        for index, row in enumerate(rows):
            example = dataset[int(row["id"].split("-")[-1])]
            if row["problem"] != example["problem"] or str(row["expected_answer"]) != str(example["expected_answer"]):
                raise ValueError(f"dataset content mismatch: {row['id']}")
            original_correct = row.get("symbolic_correct")
            prediction = await evaluator.eval_single({
                "generation": row.get("generation", ""),
                "expected_answer": example["expected_answer"],
            })
            result = dict(row)
            result.update(predicted_answer=prediction["predicted_answer"],
                          symbolic_correct=bool(prediction["symbolic_correct"]),
                          numeric_boxed_correct=numeric_exact(row.get("generation", ""), example["expected_answer"]))
            if "error" in row or row.get("finish_reason") == "error":
                result["symbolic_correct"] = False
                result["numeric_boxed_correct"] = False
            if original_correct is not None and original_correct != result["symbolic_correct"]:
                old_score_changes.append(row["id"])
            scored.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0:
                logger.info("%s CPU scoring %d/%d", args.label, index + 1, len(rows))
    failure_categories = collections.Counter()
    for row in scored:
        if row["symbolic_correct"]:
            continue
        if "error" in row or row.get("finish_reason") == "error":
            category = "request_error"
        elif row.get("finish_reason") == "length":
            category = "truncated_incorrect"
        elif row["predicted_answer"] is None:
            category = "no_boxed_answer"
        else:
            category = "boxed_answer_mismatch"
        failure_categories[category] += 1
    versions = {}
    for name in ["sgl-eval", "math-verify", "latex2sympy2-extended", "sympy"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_paths = [Path(inspect.getfile(MathEvaluator)), Path(inspect.getfile(math_grader))]
    correct = sum(row["symbolic_correct"] for row in scored)
    numeric = sum(row["numeric_boxed_correct"] for row in scored)
    result = {
        "complete": ids == valid_ids and not args.partial_diagnostic,
        "partial_diagnostic": args.partial_diagnostic,
        "diagnostic_limit": args.limit,
        "examples": len(rows), "unique_ids": len(ids),
        "symbolic_correct": correct, "symbolic_accuracy_percent": 100 * correct / len(rows),
        "numeric_boxed_correct": numeric, "numeric_boxed_accuracy_percent": 100 * numeric / len(rows),
        "scoring_disagreement_ids": [row["id"] for row in scored
                                     if row["symbolic_correct"] != row["numeric_boxed_correct"]],
        "historical_symbolic_score_changes": old_score_changes,
        "failed_ids": [row["id"] for row in scored if not row["symbolic_correct"]],
        "failure_categories": dict(failure_categories),
        "finish_reasons": dict(collections.Counter(row.get("finish_reason") for row in scored)),
        "request_errors": sum("error" in row or row.get("finish_reason") == "error" for row in scored),
        "truncated": sum(row.get("finish_reason") == "length" for row in scored),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "scorer_source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "versions": versions,
        "scoring": "unchanged sgl-eval vendored MathEvaluator(config={}); supplemental numeric boxed exact-match",
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--partial-diagnostic", action="store_true",
                        help="Score a selected subset; always label it non-complete diagnostic")
    parser.add_argument("--limit", type=int,
                        help="Score the first N saved rows; requires --partial-diagnostic")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not args.output_dir.resolve().is_relative_to(project / "logs"):
        parser.error("outputs must remain in this checkout's logs directory")
    if not args.label.replace("_", "").isalnum():
        parser.error("label must be alphanumeric/underscore")
    asyncio.run(_score(args))


if __name__ == "__main__":
    main()
