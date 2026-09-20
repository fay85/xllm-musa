#!/usr/bin/env python3
"""Reference-free semantic-risk routing for recurring GSM8K conventions."""

import argparse
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_audit_module():
    path = HERE / "dspark_gsm8k_arithmetic_audit.py"
    spec = importlib.util.spec_from_file_location("dsv4_arithmetic_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(question, content):
    question_lower = question.lower()
    content_lower = content.lower()
    categories = []
    if (
        re.search(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+years?\s+ago\b",
            question_lower,
        )
        and re.search(r"\b(?:age|aged|old|years old)\b", question_lower)
    ):
        categories.append("past_age_alignment")
    if (
        "since" in question_lower
        and "years old" in question_lower
        and re.search(r"\bnow\b", question_lower)
    ):
        categories.append("elapsed_age_endpoint")
    if (
        "because" in question_lower
        and re.search(r"\bper\s+invited\b", question_lower)
        and re.search(r"(?:do(?:es)?n't|do not|does not|no[- ]show)", question_lower)
    ):
        categories.append("causal_rate_already_applied")
    if (
        "monthly interest" in question_lower
        and not re.search(r"\bcompound(?:ed|ing)?\b", question_lower)
        and re.search(r"\bafter\s+\d+\s+months?\b", question_lower)
    ):
        categories.append("simple_monthly_interest")
    if "half a dozen" in question_lower:
        categories.append("half_dozen_cardinality")
    if (
        re.search(r"\b(?:empty|used)\s+(?:pens?|bottles?|wrappers?)\b", question_lower)
        and re.search(r"\b(?:make|trade|exchange|turn)\b", question_lower)
        and re.search(r"\b(?:new|full)\b", question_lower)
    ):
        categories.append("recursive_exchange")
    stall_counts = re.findall(r"\b(ten|twenty|\d+)\s+stalls?\b", question_lower)
    if len(set(stall_counts)) > 1:
        categories.append("conflicting_container_count")
    uncertainty_phrases = (
        "not specified",
        "no additional information",
        "ambiguous",
        "cannot determine",
        "possibly the intended",
        "likely intended",
    )
    if any(phrase in content_lower for phrase in uncertainty_phrases):
        categories.append("explicit_uncertainty")
    return categories


def category_supported_by_draft(category, content):
    """Require the draft to exhibit the convention before routing a repair."""
    content_lower = content.lower()
    patterns = {
        "past_age_alignment": r"\b(?:years? ago|age|aged|years old)\b",
        "elapsed_age_endpoint": r"\b(?:since|now|years? old|elapsed)\b",
        "causal_rate_already_applied": r"\b(?:per invited|no[- ]show|attend|because)\b",
        "simple_monthly_interest": r"\b(?:monthly interest|interest|principal|months?)\b",
        "half_dozen_cardinality": r"\b(?:half a dozen|6|six)\b",
        "recursive_exchange": r"\b(?:exchange|trade|used|empty|remainder|left)\b",
    }
    pattern = patterns.get(category)
    return bool(pattern and re.search(pattern, content_lower))


def filter_actionable_categories(categories, content):
    """Drop topic-only hits that the generated draft never actually uses."""
    return [
        category
        for category in categories
        if category in {"explicit_uncertainty", "conflicting_container_count"}
        or category_supported_by_draft(category, content)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--exclude-results",
        help="JSONL repair results whose selected indices should not be routed again.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    audit = load_audit_module()
    excluded_indices = set()
    if args.exclude_results:
        for line in Path(args.exclude_results).read_text(encoding="utf-8").splitlines():
            if line.strip():
                repair = json.loads(line)
                if repair.get("selected", False):
                    excluded_indices.add(repair["dataset_index"])

    output = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["dataset_index"] in excluded_indices:
            continue
        attempts = record.get("attempts", [])
        all_content = "\n".join(
            attempt.get("content", "") for attempt in attempts
        ) or audit.selected_content(record)
        categories = filter_actionable_categories(
            classify(record.get("question", ""), all_content),
            all_content,
        )
        if categories:
            output.append(
                {
                    "dataset_index": record["dataset_index"],
                    "prediction": record.get("prediction"),
                    "categories": categories,
                    "actionable": any(
                        category not in {
                            "explicit_uncertainty",
                            "conflicting_container_count",
                        }
                        for category in categories
                    ),
                    "selection_uses_reference": False,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "risks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )
    summary = {
        "input": str(Path(args.input).resolve()),
        "exclude_results": (
            str(Path(args.exclude_results).resolve()) if args.exclude_results else None
        ),
        "excluded_indices": sorted(excluded_indices),
        "records_routed": len(output),
        "actionable_records": sum(r["actionable"] for r in output),
        "routed_indices": [r["dataset_index"] for r in output],
        "actionable_indices": [r["dataset_index"] for r in output if r["actionable"]],
        "selection_uses_reference": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
