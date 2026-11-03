# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Check five-image ordering, retaining the exact request and complete reply."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from vl_bench import stream_chat
from vl_qa_prompt import build_qa_prompt, image_uris

KEYWORDS = {
    "01_us_capitol.jpg": ("capitol",),
    "02_supreme_court.jpg": ("supreme court",),
    "03_kremlin.jpg": ("kremlin",),
    "04_oil_tanker.jpg": ("tanker", "ship"),
    "05_budget_airline.jpg": ("ryanair", "airplane", "aircraft", "plane"),
}
QUESTION = (
    "Question: This conversation contains five images. List them in order of appearance, "
    "one short line each, saying what each image shows."
)


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _ask(args: argparse.Namespace, prompt: str, index: int) -> dict[str, Any]:
    content = build_qa_prompt(prompt, QUESTION)
    uris = image_uris(content)
    order = [uri.rsplit("/", 1)[-1] for uri in uris]
    if any(name not in KEYWORDS for name in order):
        raise ValueError(f"Unknown images: {order}")
    request_id = f"vl-qa-fixed-{args.run_id}-{index}"
    record = dict(
        index=index,
        source_index=args.prompt_count - 1 - index,
        original_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        image_uris=uris,
        request_payload=dict(
            model=args.model,
            messages=[dict(role="user", content=content)],
            max_tokens=args.max_tokens,
            temperature=0,
            stream=True,
            ignore_eos=True,
            chat_template_kwargs=dict(enable_thinking=False),
            request_id=request_id,
            x_request_id=request_id,
        ),
    )
    _write(args.output / f"request_{index:02d}.json", record)
    try:
        result = stream_chat(args.url, args.model, content, args.max_tokens, request_id, args.timeout)
        # Preserve the original scorer while retaining the full response separately.
        answer = re.split(r"user|assistant", result["text"])[0].lower()
        position = 0
        missing = []
        for name in order:
            hits = [answer.find(key, position) for key in KEYWORDS[name] if answer.find(key, position) >= 0]
            if not hits:
                missing.append(name)
            else:
                position = min(hits) + 1
        outcome = dict(
            index=index,
            passed=not missing,
            missing=missing,
            answer_scored=answer,
            result=result,
        )
    except Exception as error:
        outcome = dict(index=index, passed=False, error=f"{type(error).__name__}: {error}")
    _write(args.output / f"result_{index:02d}.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("vl_qa_results") / str(time.time_ns()))
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())["prompts"]
    if not 0 < args.count <= len(prompts) or args.concurrency < 1:
        parser.error("Invalid count or concurrency")
    for prompt in prompts:
        build_qa_prompt(prompt, QUESTION)
    args.output.mkdir(exist_ok=False, parents=True)
    args.prompt_count = len(prompts)
    args.run_id = time.time_ns()
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for start in range(0, args.count, args.concurrency):
            indices = range(start, min(start + args.concurrency, args.count))
            outcomes.extend(pool.map(lambda i: _ask(args, prompts[-1 - i], i), indices))
    summary = dict(
        passed=sum(row["passed"] for row in outcomes),
        total=args.count,
        failed_indices=[row["index"] for row in outcomes if not row["passed"]],
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        protocol="five intact images; unchanged original keyword/order scorer; ignore_eos=true",
        prompt_file_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        constructor_sha256=hashlib.sha256(Path(__file__).with_name("vl_qa_prompt.py").read_bytes()).hexdigest(),
    )
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["failed_indices"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
