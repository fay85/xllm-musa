#!/usr/bin/env python3
"""Reference-free repairs for explicitly classified GSM8K conventions."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
import urllib.request

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
STRATEGY_VERSION = 3
CATEGORY_INSTRUCTIONS = {
    "past_age_alignment": (
        "When a relationship is stated at a past time, shift every person's age "
        "to that same past time before applying the relationship, then shift the "
        "requested person back to the requested time."
    ),
    "elapsed_age_endpoint": (
        "For vacationing or working since age A and now age B, count elapsed full "
        "years as B-A unless the question explicitly asks for inclusive birthdays."
    ),
    "causal_rate_already_applied": (
        "When a per-invited-person rate is stated because some invitees do not "
        "attend, the rate already accounts for absence; do not apply it a second time."
    ),
    "simple_monthly_interest": (
        "In a grade-school problem, monthly interest over a stated number of months "
        "is simple interest unless compounding is explicitly stated: principal times "
        "monthly rate times months."
    ),
    "half_dozen_cardinality": "A dozen is 12 items, so half a dozen is 6 items.",
    "recursive_exchange": (
        "For exchanging a fixed number of used items for one new usable item, repeat "
        "the exchange after each new item is used until fewer than one group remains."
    ),
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _compact_demonstration_answer(answer):
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    marked = [line for line in lines if line.startswith("####")]
    if marked:
        return "\n".join(lines[:2] + marked[-1:])
    return "\n".join(lines[:3])


def _prompt(question, categories, examples):
    demonstrations = "\n\n".join(
        f"Question: {row['question']}\nAnswer:\n{_compact_demonstration_answer(row['answer'])}"
        for row in examples
    )
    instructions = "\n".join(
        f"- {CATEGORY_INSTRUCTIONS[category]}"
        for category in categories
        if category in CATEGORY_INSTRUCTIONS
    )
    return (
        "Re-solve only the final grade-school word problem. Apply the explicit "
        "interpretation rules below, copy every named quantity before calculating, "
        "check each operation, use at most six short lines, and end with exactly "
        "`#### <numeric answer>`. Do not discuss ambiguity or explain the rule. The "
        "solved examples demonstrate output format only; do not copy their facts.\n\n"
        f"Interpretation rules:\n{instructions}\n\n"
        f"{demonstrations}\n\nQuestion: {question}\nAnswer:\n"
    )


def _percentile95(values):
    if len(values) <= 1:
        return values[0] if values else 0
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--primary-results", required=True)
    parser.add_argument("--risks", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-items", type=int, default=8)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_new_items <= 0:
        parser.error("token and per-service limits must be positive")

    primary_eval = _load_module(
        "dsv4_primary_eval", HERE / "dspark_gsm8k_consensus_eval.py"
    )
    audit = _load_module(
        "dsv4_arithmetic_audit", HERE / "dspark_gsm8k_arithmetic_audit.py"
    )
    base = primary_eval.load_base()
    rows = pq.read_table(args.dataset, columns=["question", "answer"]).to_pylist()
    examples = pq.read_table(
        Path(args.train_dataset),
        columns=["question", "answer"],
    ).slice(0, 4).to_pylist()

    primary = {}
    for line in Path(args.primary_results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            primary[record["dataset_index"]] = record
    risks = []
    for line in Path(args.risks).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("actionable", False):
                risks.append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    existing = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["dataset_index"]] = record

    config = {
        "dataset": str(Path(args.dataset).resolve()),
        "primary_results": str(Path(args.primary_results).resolve()),
        "risks": str(Path(args.risks).resolve()),
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "max_new_items_per_service": args.max_new_items,
        "strategy_version": STRATEGY_VERSION,
        "selection_policy": (
            "route only explicit convention categories; select a response only if "
            "it has a strict final marker and no exact-arithmetic contradiction"
        ),
        "selection_uses_reference": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output = results_path.open("a" if args.resume else "w", encoding="utf-8")
    new_items = 0
    for position, risk in enumerate(risks, 1):
        index = risk["dataset_index"]
        prior = existing.get(index)
        if prior is not None and (
            prior.get("selected", False)
            or (
                prior.get("strategy_version") == STRATEGY_VERSION
                and "error" not in prior
            )
        ):
            continue
        row = rows[index]
        record = {
            "position": position,
            "dataset_index": index,
            "question": row["question"],
            "primary_prediction": primary[index].get("prediction"),
            "categories": risk["categories"],
            "strategy_version": STRATEGY_VERSION,
            "selection_uses_reference": False,
        }
        started = time.monotonic()
        try:
            response = _post_json(
                args.url,
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": _prompt(
                                row["question"], risk["categories"], examples
                            ),
                        }
                    ],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "chat_template_kwargs": {
                        "thinking_mode": "chat",
                        "reasoning_effort": "none",
                    },
                },
                args.timeout,
            )
            choice = response["choices"][0]
            content = choice["message"].get("content", "")
            prediction = primary_eval.extract_marked_prediction(base, content)
            discrepancies = audit.audit_content(content)
            # A draft may contain a transcription typo in prose while its
            # strict final result is still independently reproducible from the
            # original question and the routed convention.  Ask the exact
            # arithmetic propagator before rejecting that candidate.
            propagated = None
            if prediction is not None and discrepancies:
                arithmetic_repair = _load_module(
                    "dsv4_arithmetic_repair",
                    HERE / "dspark_gsm8k_arithmetic_repair_eval.py",
                )
                propagated = arithmetic_repair.propagated_correction(
                    base, audit, row["question"], content, prediction
                )
            selected = prediction is not None and (
                not discrepancies or propagated == prediction
            )
            record.update(
                {
                    "content": content,
                    "prediction": prediction if selected else None,
                    "candidate_prediction": prediction,
                    "valid_final_marker": prediction is not None,
                    "arithmetic_discrepancies": discrepancies,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "selected": selected,
                }
            )
        except Exception as error:
            record.update({"error": repr(error), "prediction": None, "selected": False})

        reference = base.extract_reference(row["answer"])
        record["reference"] = reference
        record["correct"] = record.get("prediction") == reference
        record["latency_seconds"] = round(time.monotonic() - started, 3)
        existing[index] = record
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        new_items += 1
        print(
            f"[{position:04d}/{len(risks)}] "
            f"{'PASS' if record['correct'] else 'FAIL'} index={index} "
            f"primary={record['primary_prediction']} repair={record.get('prediction')} "
            f"ref={reference} categories={','.join(risk['categories'])} "
            f"latency={record['latency_seconds']}s",
            flush=True,
        )
        if new_items >= args.max_new_items:
            break
    output.close()

    results = [
        existing[risk["dataset_index"]]
        for risk in risks
        if risk["dataset_index"] in existing
    ]
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in results),
        encoding="utf-8",
    )
    completed = [record for record in results if "error" not in record]
    latencies = [record["latency_seconds"] for record in completed]
    summary = {
        **config,
        "routed_records": len(risks),
        "completed": len(completed),
        "request_errors": len(results) - len(completed),
        "selected": sum(record.get("selected", False) for record in completed),
        "correct": sum(record.get("correct", False) for record in completed),
        "incorrect": sum(not record.get("correct", False) for record in completed),
        "accuracy": (
            sum(record.get("correct", False) for record in completed) / len(completed)
            if completed
            else 0
        ),
        "total_latency_seconds": round(sum(latencies), 3),
        "average_latency_seconds": (
            round(statistics.mean(latencies), 3) if latencies else 0
        ),
        "p95_latency_seconds": round(_percentile95(latencies), 3),
        "failed_indices": [
            record["dataset_index"]
            for record in completed
            if not record.get("correct", False)
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
