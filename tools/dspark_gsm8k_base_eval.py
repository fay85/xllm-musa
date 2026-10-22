#!/usr/bin/env python3
"""Evaluate a fixed GSM8K subset with DeepSeek-V4 reasoning modes."""

import argparse
import ast
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import time
import urllib.request

import pyarrow.parquet as pq


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def normalize_number(value):
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "").rstrip(". ")
    try:
        if "/" in value:
            return str(Fraction(value))
        decimal = Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return value
    if decimal == decimal.to_integral_value():
        return str(decimal.to_integral_value())
    return format(decimal.normalize(), "f")


def extract_reference(answer):
    match = re.search(r"####\s*([^\n]+)", answer)
    numbers = NUMBER_RE.findall(match.group(1)) if match else []
    return normalize_number(numbers[-1]) if numbers else None


def extract_prediction(text, first_final_marker=False):
    tagged = re.findall(r"####\s*([^\n]+)", text)
    if tagged:
        numbers = NUMBER_RE.findall(tagged[0] if first_final_marker else tagged[-1])
        if numbers:
            return normalize_number(numbers[-1])
    boxed = BOXED_RE.findall(text)
    if boxed:
        numbers = NUMBER_RE.findall(boxed[-1])
        if numbers:
            return normalize_number(numbers[-1])
    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def evaluate_expression_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_expression_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_expression_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = evaluate_expression_node(node.left)
        right = evaluate_expression_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


def extract_expression_prediction(text):
    candidates = re.findall(r"```(?:python)?\s*([^`]+)```", text, re.IGNORECASE)
    candidates.extend(line.strip() for line in text.splitlines() if line.strip())
    for candidate in candidates:
        candidate = re.sub(
            r"^(?:expression|answer|result)\s*:\s*", "", candidate, flags=re.IGNORECASE
        )
        candidate = candidate.replace("^", "**").replace("×", "*").replace("÷", "/")
        candidate = candidate.strip().rstrip(". ")
        try:
            value = evaluate_expression_node(ast.parse(candidate, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        return normalize_number(str(value))
    return None


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--indices", default="")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="thinking")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="high")
    parser.add_argument(
        "--prompt-style",
        choices=("reasoning", "short-answer", "expression", "few-shot"),
        default="reasoning",
    )
    parser.add_argument("--few-shot-dataset", default="")
    parser.add_argument("--few-shot-count", type=int, default=8)
    parser.add_argument("--extra-instruction", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.dataset, columns=["question", "answer"]).to_pylist()
    few_shot_rows = []
    if args.prompt_style == "few-shot":
        if not args.few_shot_dataset:
            parser.error("--few-shot-dataset is required for --prompt-style=few-shot")
        few_shot_rows = pq.read_table(
            args.few_shot_dataset, columns=["question", "answer"]
        ).slice(0, args.few_shot_count).to_pylist()
    if args.indices:
        selected = [int(value) for value in args.indices.split(",")]
    else:
        selected = sorted(random.Random(args.seed).sample(range(len(rows)), args.samples))

    config = {
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_rows": len(rows),
        "sample_indices": selected,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking_mode": args.thinking_mode,
        "reasoning_effort": args.reasoning_effort,
        "prompt_style": args.prompt_style,
        "few_shot_dataset": (
            str(Path(args.few_shot_dataset).resolve()) if args.few_shot_dataset else ""
        ),
        "few_shot_count": len(few_shot_rows),
        "extra_instruction": args.extra_instruction,
        "scoring": "normalized numeric exact match on assistant content",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for position, dataset_index in enumerate(selected, 1):
            row = rows[dataset_index]
            reference = extract_reference(row["answer"])
            if args.prompt_style == "short-answer":
                prompt = (
                    "Return only the numeric answer to the exact question below. "
                    "Do not explain, rewrite the question, or substitute any number. "
                    "Calculate internally and output one number only.\n\n"
                    f"Question: {row['question']}\nNumeric answer:"
                )
            elif args.prompt_style == "expression":
                prompt = (
                    "Translate the exact grade-school word problem below into one valid "
                    "Python arithmetic expression whose value is the requested answer. "
                    "Preserve every number and condition. Output the expression only: no "
                    "words, variables, code fence, or final evaluated number.\n\n"
                    f"Question: {row['question']}\nExpression:"
                )
            elif args.prompt_style == "few-shot":
                demonstrations = "\n\n".join(
                    f"Question: {item['question']}\nAnswer:\n{item['answer']}"
                    for item in few_shot_rows
                )
                prompt = (
                    "Solve the final grade-school math problem by following the examples. "
                    "Preserve every number and condition. End the solution with exactly "
                    "`#### <numeric answer>`.\n\n"
                    f"{demonstrations}\n\nQuestion: {row['question']}\nAnswer:\n"
                )
            else:
                prompt = (
                    "Solve the following grade-school math problem. Preserve every number "
                    "and condition from the question. Verify the arithmetic, then put only "
                    "the final numeric result after `####`.\n\n"
                    f"Question: {row['question']}"
                )
            if args.extra_instruction:
                prompt = f"{args.extra_instruction}\n\n{prompt}"
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "chat_template_kwargs": {
                    "thinking_mode": args.thinking_mode,
                    "reasoning_effort": args.reasoning_effort,
                },
            }
            if args.prompt_style == "few-shot":
                payload["stop"] = ["\n\nQuestion:"]
            started = time.monotonic()
            record = {
                "position": position,
                "dataset_index": dataset_index,
                "question": row["question"],
                "reference": reference,
            }
            try:
                response = post_json(args.url, payload, args.timeout)
                choice = response["choices"][0]
                message = choice["message"]
                content = message.get("content", "")
                prediction = (
                    extract_expression_prediction(content)
                    if args.prompt_style == "expression"
                    else extract_prediction(
                        content, first_final_marker=args.prompt_style == "few-shot"
                    )
                )
                record.update(
                    prediction=prediction,
                    correct=prediction == reference,
                    content=content,
                    reasoning_content=message.get("reasoning_content", ""),
                    finish_reason=choice.get("finish_reason"),
                    usage=response.get("usage"),
                    latency_seconds=round(time.monotonic() - started, 3),
                )
            except Exception as error:
                record.update(
                    prediction=None,
                    correct=False,
                    error=repr(error),
                    latency_seconds=round(time.monotonic() - started, 3),
                )
            results.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            state = "PASS" if record["correct"] else "FAIL"
            print(
                f"[{position:02d}/{len(selected)}] {state} index={dataset_index} "
                f"pred={record.get('prediction')} ref={reference} "
                f"latency={record['latency_seconds']:.3f}s",
                flush=True,
            )

    correct = sum(result["correct"] for result in results)
    summary = {
        **config,
        "correct": correct,
        "incorrect": len(results) - correct,
        "request_errors": sum("error" in result for result in results),
        "accuracy": correct / len(results),
        "failed_indices": [
            result["dataset_index"] for result in results if not result["correct"]
        ],
        "total_latency_seconds": round(
            sum(result["latency_seconds"] for result in results), 3
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
