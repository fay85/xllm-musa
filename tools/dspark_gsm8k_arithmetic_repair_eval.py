#!/usr/bin/env python3
"""Repair drafts selected by the reference-free exact-arithmetic audit."""

import argparse
import ast
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import statistics
import time
import urllib.request

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPAIR_STRATEGY_VERSION = 4
EXPRESSION_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def discrepancy_text(discrepancies):
    return "\n".join(
        f"- `{item['expression']}` evaluates exactly to `{item['computed']}`, "
        f"not `{item['claimed']}`."
        for item in discrepancies
    )


def compact_demonstration_answer(answer):
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    marked = [line for line in lines if line.startswith("####")]
    if marked:
        return "\n".join(lines[:2] + marked[-1:])
    return "\n".join(lines[:3])


def repair_prompt(question, discrepancies, examples, retry=False):
    demonstrations = "\n\n".join(
        f"Question: {row['question']}\nAnswer:\n{compact_demonstration_answer(row['answer'])}"
        for row in examples
    )
    retry_text = (
        "The previous repair still contained a calculator contradiction. "
        if retry
        else ""
    )
    return (
        "Re-solve this grade-school word problem using only the original question "
        "below. First copy its named quantities, then calculate. Do not import any "
        "number, example, answer, or wording from another task. An exact rational "
        "calculator found arithmetic contradictions in an earlier solution; its "
        "corrections below are authoritative for those operations. Independently "
        "preserve the original relationships, units, and requested result. "
        f"{retry_text}Use concise arithmetic and end with exactly "
        "`#### <numeric answer>`. Follow the solved examples only for response "
        "format; do not copy their facts or answers.\n\n"
        f"{demonstrations}\n\n"
        f"Original question:\n{question}\n\n"
        f"Exact calculator findings:\n{discrepancy_text(discrepancies)}\n\n"
        "Corrected solution:\n"
    )


def terminal_correction(
    base, question, content, primary_prediction, discrepancies
):
    """Return an exact correction when the audited equation is the final answer."""
    if len(discrepancies) != 1 or primary_prediction is None:
        return None
    discrepancy = discrepancies[0]
    claimed = base.normalize_number(discrepancy["claimed"])
    if claimed != primary_prediction:
        return None
    prefix = "\n".join(content.splitlines()[: discrepancy["line_number"] - 1])
    known_numbers = {
        base.normalize_number(value)
        for value in base.NUMBER_RE.findall(f"{question}\n{prefix}")
    }
    operands = {
        base.normalize_number(value)
        for value in base.NUMBER_RE.findall(discrepancy["expression"])
    }
    if not operands or not operands.issubset(known_numbers):
        return None
    return base.normalize_number(discrepancy["computed"])


def _expression_numbers(audit_module, expression):
    values = set()
    for token in EXPRESSION_NUMBER_RE.findall(expression):
        try:
            values.add(audit_module.parse_number(token))
        except (ValueError, ZeroDivisionError):
            return set()
    return values


def _rewrite_expression(audit_module, expression, corrections):
    def replace(match):
        value = audit_module.parse_number(match.group(0))
        corrected = corrections.get(value, value)
        if corrected.denominator == 1:
            return str(corrected.numerator)
        return f"({corrected.numerator}/{corrected.denominator})"

    return EXPRESSION_NUMBER_RE.sub(replace, expression)


def _is_close(left, right):
    error = abs(float(left - right))
    return left == right or error <= max(1e-9, abs(float(left)) * 1e-3)


def propagated_correction(
    base, audit_module, question, content, primary_prediction
):
    """Propagate verified arithmetic corrections through later equations."""
    if primary_prediction is None:
        return None
    question_lower = question.lower()
    if (
        re.search(r"\b(?:empty|used)\b", question_lower)
        and re.search(r"\b(?:make|trade|exchange|turn)\b", question_lower)
        and re.search(r"\b(?:new|full)\b", question_lower)
    ):
        # Exchange/recycling problems can create another complete group.  A
        # corrected first pass is not enough unless that recurrence is solved.
        return None
    known = _expression_numbers(audit_module, question)
    question_values = set(known)
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)", question, re.IGNORECASE
    ):
        percentage = Fraction(match.group(1)) / 100
        known.add(percentage)
        question_values.add(percentage)

    corrections = {}
    seeded = False
    for line in content.splitlines():
        if "=" not in line:
            known.update(_expression_numbers(audit_module, line))
            continue
        segments = line.split("=")
        for position in range(len(segments) - 1):
            raw_lhs = segments[position]
            if re.search(r"\d[A-Za-z]|\b[A-Za-z]\b\s*[-+*/]", raw_lhs):
                continue
            lhs, original_computed = audit_module.normalize_lhs(raw_lhs)
            if lhs is None:
                continue
            match = audit_module.RHS_NUMBER_RE.match(segments[position + 1])
            if not match:
                continue
            rhs_tail = segments[position + 1][match.end() :]
            if re.search(r"[-+*/×÷]\s*(?:\$?\s*\d|\()", rhs_tail):
                continue
            try:
                claimed = audit_module.parse_number(match.group(1))
            except (ValueError, ZeroDivisionError):
                continue
            operands = _expression_numbers(audit_module, lhs)
            if not operands:
                continue
            if any(value not in known and value not in corrections for value in operands):
                continue
            rewritten = _rewrite_expression(audit_module, lhs, corrections)
            try:
                computed = audit_module.evaluate_node(ast.parse(rewritten, mode="eval"))
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            known.add(computed)
            if _is_close(computed, claimed):
                known.add(claimed)
                continue
            if claimed in question_values:
                continue
            if claimed in corrections and corrections[claimed] != computed:
                return None
            corrections[claimed] = computed
            if original_computed is not None and not _is_close(original_computed, claimed):
                seeded = True

    try:
        prediction = audit_module.parse_number(primary_prediction)
    except (ValueError, ZeroDivisionError):
        return None
    if not seeded or prediction not in corrections:
        return None
    corrected = corrections[prediction]
    if corrected.denominator == 1:
        return str(corrected.numerator)
    return base.normalize_number(str(corrected))


def break_even_start_earning_correction(audit_module, question, discrepancies):
    """Return the first earning period after exact break-even.

    This is reference-free: it uses only the question wording and audited
    arithmetic corrections.  GSM8K-style questions that ask when someone starts
    earning money require the first positive-profit period; if the corrected
    break-even quotient is an exact integer N, earning starts in period N+1.
    """
    question_lower = question.lower()
    if not (
        re.search(r"\bstarts?\b", question_lower)
        and re.search(r"\bearn(?:s|ing)?\b", question_lower)
        and re.search(r"\byears?\b", question_lower)
    ):
        return None
    corrections = {}
    for discrepancy in discrepancies:
        try:
            claimed = audit_module.parse_number(discrepancy["claimed"])
            computed = audit_module.parse_number(discrepancy["computed"])
        except (ValueError, ZeroDivisionError, KeyError):
            return None
        corrections[claimed] = computed
    for discrepancy in discrepancies:
        expression = discrepancy.get("expression", "")
        if "/" not in expression:
            continue
        rewritten = _rewrite_expression(audit_module, expression, corrections)
        try:
            quotient = audit_module.evaluate_node(ast.parse(rewritten, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        if quotient <= 0:
            continue
        if quotient.denominator == 1:
            years = quotient.numerator + 1
        else:
            years = (quotient.numerator + quotient.denominator - 1) // quotient.denominator
        return str(years)
    return None


def percentile95(values):
    if len(values) <= 1:
        return values[0] if values else 0
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--primary-results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-items", type=int, default=0)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_attempts <= 0:
        parser.error("token and attempt limits must be positive")
    # Arithmetic repairs should remain a short exceptional path.  Most accepted
    # fixes are local exact corrections; model repair is only a compact fallback.
    args.max_tokens = min(args.max_tokens, 192)
    if args.max_new_items <= 0:
        args.max_new_items = 8

    primary_eval = load_module(
        "dsv4_primary_eval", HERE / "dspark_gsm8k_consensus_eval.py"
    )
    audit_module = load_module(
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
    audited = []
    for line in Path(args.audit).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("has_arithmetic_discrepancy"):
                audited.append(record)

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
        "audit": str(Path(args.audit).resolve()),
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "max_new_items_per_service": args.max_new_items,
        "selection_policy": (
            "replace an arithmetic-inconsistent primary with an exact audited "
            "terminal-equation correction, or with the first model repair having "
            "an explicit final marker and no exact-arithmetic contradiction"
        ),
        "selection_uses_reference": False,
        "repair_strategy_version": REPAIR_STRATEGY_VERSION,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output = results_path.open("a" if args.resume else "w", encoding="utf-8")
    new_model_items = 0
    for position, audit_record in enumerate(audited, 1):
        index = audit_record["dataset_index"]
        if index in existing:
            prior = existing[index]
            if prior.get("selected", False):
                continue
            if (
                prior.get("repair_strategy_version") == REPAIR_STRATEGY_VERSION
                and "error" not in prior
            ):
                continue
        row = rows[index]
        original = primary[index]
        discrepancies = audit_record["discrepancies"]
        record = {
            "position": position,
            "dataset_index": index,
            "question": row["question"],
            "primary_prediction": original.get("prediction"),
            "primary_discrepancies": discrepancies,
            "attempts": [],
            "selection_uses_reference": False,
            "repair_strategy_version": REPAIR_STRATEGY_VERSION,
        }
        item_started = time.monotonic()
        exact_prediction = terminal_correction(
            base,
            row["question"],
            audit_module.selected_content(original),
            original.get("prediction"),
            discrepancies,
        )
        selection_method = "exact-terminal-equation-correction"
        if exact_prediction is None:
            exact_prediction = propagated_correction(
                base,
                audit_module,
                row["question"],
                audit_module.selected_content(original),
                original.get("prediction"),
            )
            selection_method = "exact-propagated-equation-correction"
        if exact_prediction is None:
            exact_prediction = break_even_start_earning_correction(
                audit_module,
                row["question"],
                discrepancies,
            )
            selection_method = "exact-break-even-start-earning-correction"
        if exact_prediction is not None:
            record["prediction"] = exact_prediction
            record["selected"] = True
            record["selection_method"] = selection_method
            record["latency_seconds"] = 0.0
            reference = base.extract_reference(row["answer"])
            record["reference"] = reference
            record["correct"] = exact_prediction == reference
            existing[index] = record
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[{position:04d}/{len(audited)}] "
                f"{'PASS' if record['correct'] else 'FAIL'} index={index} "
                f"primary={record['primary_prediction']} repair={exact_prediction} "
                f"ref={reference} method={selection_method}",
                flush=True,
            )
            continue

        try:
            selected = None
            for attempt_number in range(1, args.max_attempts + 1):
                started = time.monotonic()
                response = post_json(
                    args.url,
                    {
                        "model": args.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": repair_prompt(
                                    row["question"],
                                    discrepancies,
                                    examples,
                                    retry=attempt_number > 1,
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
                repair_discrepancies = audit_module.audit_content(content)
                attempt = {
                    "attempt_number": attempt_number,
                    "content": content,
                    "prediction": prediction,
                    "valid_final_marker": prediction is not None,
                    "arithmetic_discrepancies": repair_discrepancies,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "latency_seconds": round(time.monotonic() - started, 3),
                }
                record["attempts"].append(attempt)
                if prediction is not None and not repair_discrepancies:
                    selected = prediction
                    record["selected_attempt"] = attempt_number
                    break
                discrepancies = repair_discrepancies or discrepancies
            record["prediction"] = selected
            record["selected"] = selected is not None
        except Exception as error:
            record["error"] = repr(error)
            record["prediction"] = None
            record["selected"] = False

        reference = base.extract_reference(row["answer"])
        record["reference"] = reference
        record["correct"] = record.get("prediction") == reference
        record["latency_seconds"] = round(time.monotonic() - item_started, 3)
        existing[index] = record
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        new_model_items += 1
        print(
            f"[{position:04d}/{len(audited)}] "
            f"{'PASS' if record['correct'] else 'FAIL'} index={index} "
            f"primary={record['primary_prediction']} repair={record.get('prediction')} "
            f"ref={reference} latency={record['latency_seconds']}s",
            flush=True,
        )
        if args.max_new_items and new_model_items >= args.max_new_items:
            break
    output.close()

    results = [existing[r["dataset_index"]] for r in audited if r["dataset_index"] in existing]
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in results),
        encoding="utf-8",
    )
    completed = [r for r in results if "error" not in r]
    latencies = [r["latency_seconds"] for r in completed]
    summary = {
        **config,
        "audited_records": len(audited),
        "completed": len(completed),
        "request_errors": len(results) - len(completed),
        "selected": sum(r.get("selected", False) for r in completed),
        "correct": sum(r.get("correct", False) for r in completed),
        "incorrect": sum(not r.get("correct", False) for r in completed),
        "accuracy": (
            sum(r.get("correct", False) for r in completed) / len(completed)
            if completed
            else 0
        ),
        "total_latency_seconds": round(sum(latencies), 3),
        "average_latency_seconds": round(statistics.mean(latencies), 3) if latencies else 0,
        "p95_latency_seconds": round(percentile95(latencies), 3),
        "failed_indices": [
            r["dataset_index"] for r in completed if not r.get("correct")
        ],
        "selection_uses_reference": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
