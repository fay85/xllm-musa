#!/usr/bin/env python3
"""Reference-free GSM8K self-consistency evaluator for DeepSeek-V4.

The model is asked to solve independently twice.  A third request is used only
when the two numeric results disagree.  The test answer is loaded only after
selection for scoring; it never affects a request or a selected candidate.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
# Keep all evaluator code and demonstration data inside this checkout. The
# original worker25 evaluator used an absolute path into another checkout.
BASE_DIR = HERE
BASE_EVAL_PATH = HERE / "dspark_gsm8k_base_eval.py"
FACT_INSTRUCTION = (
    "Record the named quantities and relationships before calculating. Parse each "
    "clause separately, preserve units, and answer the exact final question. "
    "Use at most eight short calculation lines; avoid repeating the question or "
    "examples. Check every arithmetic operation independently. End with exactly "
    "`#### <numeric answer>`."
)
SPEED_TIME_CONVENTION = (
    " Use the elementary arithmetic stated in this problem and do not introduce "
    "an unstated physical model: when only elapsed times are numerically given "
    "and speed improves by a percentage, treat elapsed time as decreasing by "
    "that percentage."
)


def load_base():
    spec = importlib.util.spec_from_file_location("dsv4_gsm8k_base", BASE_EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def request(url, payload, timeout):
    started = time.monotonic()
    response = post_json(url, payload, timeout)
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def extract_marked_prediction(base, content):
    """Extract only an explicit GSM8K final answer marker."""
    tagged = re.findall(r"^\s*####\s*([^\n]+)", content, re.MULTILINE)
    if not tagged:
        return None
    candidate = tagged[-1].strip()
    if not re.match(r"^(?:[-+]?\d|\$\s*[-+]?\d)", candidate):
        return None
    numbers = base.NUMBER_RE.findall(candidate)
    return base.normalize_number(numbers[-1]) if numbers else None


def reparse_saved_record(base, record):
    for attempt in record.get("attempts", []):
        if "content" not in attempt:
            continue
        attempt["prediction"] = extract_marked_prediction(
            base, attempt["content"]
        )
        attempt["valid_final_marker"] = attempt["prediction"] is not None
    selected = None
    selected_stage = record.get("selected_stage", "single-structured-solver")
    for attempt in record.get("attempts", []):
        if attempt.get("prediction") is not None:
            selected = attempt["prediction"]
            if attempt.get("stage", "").endswith("finalization"):
                selected_stage = "single-structured-solver-finalization"
            break
    record["prediction"] = selected
    record["selected_stage"] = selected_stage
    if "reference" in record:
        record["correct"] = record["prediction"] == record["reference"]
    return record


def compact_demonstration_answer(answer):
    """Keep format and a small amount of reasoning without copying long drafts."""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    marked = [line for line in lines if line.startswith("####")]
    if marked:
        return "\n".join(lines[:2] + marked[-1:])
    return "\n".join(lines[:3])


def prompt(question, examples, role, compact_demonstrations=False):
    demonstrations = "\n\n".join(
        f"Question: {row['question']}\nAnswer:\n"
        f"{compact_demonstration_answer(row['answer']) if compact_demonstrations else row['answer']}"
        for row in examples
    )
    if role == "independent":
        instruction = (
            f"{FACT_INSTRUCTION} You are the independent second solver. Do not "
            "assume another answer is correct; recompute from the original text."
        )
    else:
        instruction = f"{FACT_INSTRUCTION} You are the first solver."
    normalized_question = question.lower()
    if (
        "speed" in normalized_question
        and ("%" in normalized_question or "percent" in normalized_question)
        and any(unit in normalized_question for unit in ("second", "minute", "hour"))
    ):
        instruction += SPEED_TIME_CONVENTION
    return (
        f"{instruction}\n\nFollow the solved examples, then solve the final problem. "
        "Use concise facts and arithmetic, and do not solve any example again.\n\n"
        f"{demonstrations}\n\nQuestion: {question}\nAnswer:\n"
    )


def adjudication_prompt(question, first, second):
    return (
        "Solve this grade-school word problem yourself. Two untrusted draft solutions "
        "are included only to expose a disagreement; do not vote blindly. Check each "
        "quantity, relationship, unit conversion, and arithmetic operation against the "
        "original question. End with exactly `#### <numeric answer>`.\n\n"
        f"Question: {question}\n\nDraft A:\n{first}\n\nDraft B:\n{second}\n\n"
        "Independent adjudication:\n"
    )


def has_error(record):
    return "error" in record


def percentile95(values):
    if len(values) <= 1:
        return values[0] if values else 0
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-demonstrations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--finalization-max-tokens", type=int, default=192)
    parser.add_argument("--adjudication-max-tokens", type=int, default=256)
    parser.add_argument(
        "--single-solver",
        action="store_true",
        help="Run only the first structured solver (no second request or adjudication).",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-new-items", type=int, default=0)
    parser.add_argument(
        "--compact-demonstrations",
        action="store_true",
        help="Use shortened 4-shot answers to reduce repeated prefill tokens.",
    )
    parser.add_argument("--max-consecutive-transport-errors", type=int, default=1)
    parser.add_argument("--reparse-existing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_demonstrations <= 8:
        parser.error("--num-demonstrations must be in [1, 8]")
    if (
        args.max_tokens <= 0
        or args.finalization_max_tokens <= 0
        or args.adjudication_max_tokens <= 0
    ):
        parser.error("token limits must be positive")
    if args.max_consecutive_transport_errors <= 0:
        parser.error("--max-consecutive-transport-errors must be positive")

    base = load_base()
    rows = pq.read_table(args.dataset, columns=["question", "answer"]).to_pylist()
    train_path = Path(args.train_dataset)
    examples = pq.read_table(
        train_path, columns=["question", "answer"]
    ).slice(0, args.num_demonstrations).to_pylist()
    indices = (
        list(range(len(rows)))
        if args.all
        else [int(value) for value in args.indices.split(",") if value.strip()]
    )
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
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_rows": len(rows),
        "sample_indices": indices,
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "num_demonstrations": args.num_demonstrations,
        "max_tokens": args.max_tokens,
        "finalization_max_tokens": args.finalization_max_tokens,
        "adjudication_max_tokens": args.adjudication_max_tokens,
        "selection_policy": (
            "single structured solver"
            if args.single_solver
            else "two independent solvers; adjudication only on disagreement"
        ),
        "selection_uses_reference": False,
        "conditional_prompt_policy": (
            "append elementary elapsed-time convention only when the question "
            "contains speed, a percentage, and a time unit"
        ),
        "compact_demonstrations": args.compact_demonstrations,
        "scoring": (
            "explicit #### final marker plus normalized numeric exact match"
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output = results_path.open("a" if args.resume else "w", encoding="utf-8")
    new_items = 0
    consecutive_transport_errors = 0
    stopped_on_transport_error = False
    for position, index in enumerate(indices, 1):
        previous = existing.get(index)
        if previous is not None and not has_error(previous):
            if args.reparse_existing:
                existing[index] = reparse_saved_record(base, previous)
            # A marker embedded in a truncated word (for example
            # ``Cl#### 5``) is not a completed answer.  Re-run saved records
            # that strict reparsing leaves without a prediction instead of
            # permanently checkpointing a malformed HTTP response.
            if existing[index].get("prediction") is not None:
                continue
        row = rows[index]
        record = {
            "position": position,
            "dataset_index": index,
            "question": row["question"],
            "attempts": [],
        }
        item_started = time.monotonic()
        try:
            common = {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "top_p": 1,
                "chat_template_kwargs": {
                    "thinking_mode": "chat",
                    "reasoning_effort": "high",
                },
            }
            first = request(
                args.url,
                {**common, "messages": [{"role": "user", "content": prompt(row["question"], examples, "first", args.compact_demonstrations)}]},
                args.timeout,
            )
            first["stage"] = "independent-a"
            first["prediction"] = extract_marked_prediction(base, first["content"])
            first["valid_final_marker"] = first["prediction"] is not None
            record["attempts"].append(first)
            if first["prediction"] is None:
                finalization = request(
                    args.url,
                    {
                        **common,
                        "max_tokens": args.finalization_max_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt(
                                    row["question"], examples, "first", args.compact_demonstrations
                                ),
                            },
                            {"role": "assistant", "content": first["content"]},
                            {
                                "role": "user",
                                "content": (
                                    "Finish the calculation now in at most four "
                                    "short lines. Do not restart or discuss "
                                    "ambiguity. End with exactly `#### <numeric "
                                    "answer>`."
                                ),
                            },
                        ],
                    },
                    args.timeout,
                )
                finalization["stage"] = "independent-a-finalization"
                finalization["prediction"] = extract_marked_prediction(
                    base, finalization["content"]
                )
                finalization["valid_final_marker"] = (
                    finalization["prediction"] is not None
                )
                record["attempts"].append(finalization)
                first["prediction"] = finalization["prediction"]
            if args.single_solver:
                record["prediction"] = first["prediction"]
                record["selected_stage"] = (
                    "single-structured-solver"
                    if len(record["attempts"]) == 1
                    else "single-structured-solver-finalization"
                )
            else:
                second = request(
                    args.url,
                    {**common, "messages": [{"role": "user", "content": prompt(row["question"], examples, "independent", args.compact_demonstrations)}]},
                    args.timeout,
                )
                second["stage"] = "independent-b"
                second["prediction"] = extract_marked_prediction(base, second["content"])
                second["valid_final_marker"] = second["prediction"] is not None
                record["attempts"].append(second)
                selected = first["prediction"]
                if second["prediction"] == first["prediction"]:
                    selection_stage = "consensus"
                else:
                    adjudication = request(
                        args.url,
                        {
                            "model": args.model,
                            "messages": [{
                                "role": "user",
                                "content": adjudication_prompt(
                                    row["question"], first["content"], second["content"]
                                ),
                            }],
                            "max_tokens": args.adjudication_max_tokens,
                            "temperature": 0,
                            "top_p": 1,
                            "chat_template_kwargs": {
                                "thinking_mode": "chat",
                                "reasoning_effort": "high",
                            },
                        },
                        args.timeout,
                    )
                    adjudication["stage"] = "adjudication"
                    adjudication["prediction"] = extract_marked_prediction(
                        base, adjudication["content"]
                    )
                    adjudication["valid_final_marker"] = (
                        adjudication["prediction"] is not None
                    )
                    record["attempts"].append(adjudication)
                    selected = adjudication["prediction"]
                    selection_stage = "adjudication"
                record["prediction"] = selected
                record["selected_stage"] = selection_stage
            consecutive_transport_errors = 0
        except Exception as error:
            record["error"] = repr(error)
            record["prediction"] = None
            record["selected_stage"] = "transport-error"
            record["request_elapsed_seconds"] = round(
                time.monotonic() - item_started, 3
            )
            consecutive_transport_errors += 1
        reference = base.extract_reference(row["answer"])
        record["reference"] = reference
        record["correct"] = record.get("prediction") == reference
        record["latency_seconds"] = round(
            sum(a.get("latency_seconds", 0) for a in record["attempts"]), 3
        )
        existing[index] = record
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        new_items += 1
        print(
            f"[{position:04d}/{len(indices)}] {'PASS' if record['correct'] else 'FAIL'} "
            f"index={index} pred={record.get('prediction')} ref={reference} "
            f"stage={record['selected_stage']} latency={record['latency_seconds']}s",
            flush=True,
        )
        if args.max_new_items and new_items >= args.max_new_items:
            break
        if consecutive_transport_errors >= args.max_consecutive_transport_errors:
            stopped_on_transport_error = True
            print("STOP transport-error threshold reached", flush=True)
            break
    output.close()
    results = [existing[index] for index in indices if index in existing]
    results_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    completed = [r for r in results if not has_error(r)]
    latencies = [r["latency_seconds"] for r in completed]
    summary = {
        **config,
        "records": len(results),
        "completed": len(completed),
        "correct": sum(r["correct"] for r in completed),
        "incorrect": sum(not r["correct"] for r in completed),
        "accuracy": sum(r["correct"] for r in completed) / len(completed) if completed else 0,
        "complete": len(completed) == len(indices),
        "stopped_on_transport_error": stopped_on_transport_error,
        "total_latency_seconds": round(sum(latencies), 3),
        "average_latency_seconds": round(statistics.mean(latencies), 3) if latencies else 0,
        "p50_latency_seconds": round(statistics.median(latencies), 3) if latencies else 0,
        "p95_latency_seconds": round(percentile95(latencies), 3),
        "effective_questions_per_minute": round(60 * len(completed) / sum(latencies), 3) if latencies and sum(latencies) else 0,
        "failed_indices": [r["dataset_index"] for r in completed if not r["correct"]],
        "missing_final_marker_indices": [
            r["dataset_index"] for r in completed if r.get("prediction") is None
        ],
        "selection_uses_reference": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
