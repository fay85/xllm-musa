#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Measure one DeepSeek-V4 streaming request with deterministic token lengths."""

from __future__ import annotations

import argparse
import json
import sys
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


CONTEXT = (
    "Transformer inference has a prefill phase and a decode phase. "
    "The prefill phase processes all prompt tokens, while decode appends one "
    "token at a time and reuses paged key-value cache entries. Sparse attention "
    "uses an indexer to select compressed history and combines it with a recent "
    "sliding window. Expert-parallel mixture-of-experts layers route each token "
    "to selected experts and use collective communication to combine results. "
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import logger


def _chat_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    bos = tokenizer.bos_token or "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    rendered = (
        bos
        + "<\uff5cUser\uff5c>"
        + prompt
        + "<\uff5cAssistant\uff5c></think>"
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _counting_instruction(count_to: int) -> str:
    return (
        f"\n\nIgnore the background text. Output only the consecutive integers "
        f"from 1 through {count_to}, separated by one ASCII space. Do not add "
        "explanations or punctuation."
    )


def _build_prompt(tokenizer: Any, input_tokens: int, count_to: int) -> tuple[str, int]:
    instruction = _counting_instruction(count_to)
    context_ids = tokenizer.encode(CONTEXT, add_special_tokens=False)
    body_ids: list[int] = []
    while len(_chat_token_ids(tokenizer, tokenizer.decode(body_ids) + instruction)) < input_tokens:
        body_ids.extend(context_ids)

    while body_ids:
        prompt = tokenizer.decode(body_ids, skip_special_tokens=True) + instruction
        token_count = len(_chat_token_ids(tokenizer, prompt))
        if token_count <= input_tokens:
            break
        body_ids.pop()

    prompt = tokenizer.decode(body_ids, skip_special_tokens=True) + instruction
    token_count = len(_chat_token_ids(tokenizer, prompt))
    if abs(token_count - input_tokens) > 2:
        raise RuntimeError(
            "could not build prompt within two tokens of the target: "
            f"target={input_tokens}, actual={token_count}"
        )
    return prompt, token_count


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length


def _repeat_token_ids(source_ids: list[int], token_count: int) -> list[int]:
    if not source_ids or token_count <= 0:
        return []
    repeat_count = (token_count + len(source_ids) - 1) // len(source_ids)
    return (source_ids * repeat_count)[:token_count]


def _fit_counting_variant(
    tokenizer: Any,
    input_tokens: int,
    count_to: int,
    common_body_tokens: int,
    variant_index: int,
) -> tuple[str, int]:
    context_ids = tokenizer.encode(CONTEXT, add_special_tokens=False)
    common_ids = _repeat_token_ids(context_ids, common_body_tokens)
    common_text = tokenizer.decode(common_ids, skip_special_tokens=True)
    variant_sentence = (
        f"\nBenchmark background branch {variant_index} begins here. "
        f"Its deterministic marker is branch-{variant_index}-only. "
        "This sentence is background and must not affect the requested answer. "
    )
    variant_ids = tokenizer.encode(
        variant_sentence * (input_tokens + 32), add_special_tokens=False
    )
    instruction = _counting_instruction(count_to)

    def render(suffix_tokens: int) -> tuple[str, int]:
        suffix_text = tokenizer.decode(
            variant_ids[:suffix_tokens], skip_special_tokens=True
        )
        prompt = common_text + suffix_text + instruction
        return prompt, len(_chat_token_ids(tokenizer, prompt))

    low = 0
    high = min(len(variant_ids), input_tokens * 2 + 256)
    while low < high:
        middle = (low + high + 1) // 2
        _, token_count = render(middle)
        if token_count <= input_tokens:
            low = middle
        else:
            high = middle - 1

    candidates: list[tuple[int, str, int]] = []
    for suffix_tokens in range(max(0, low - 12), min(len(variant_ids), low + 12) + 1):
        prompt, token_count = render(suffix_tokens)
        candidates.append((abs(token_count - input_tokens), prompt, token_count))
    _, prompt, token_count = min(candidates, key=lambda item: item[0])
    if abs(token_count - input_tokens) > 2:
        raise RuntimeError(
            "could not build prefix-cache prompt variant within two tokens: "
            f"target={input_tokens}, actual={token_count}, "
            f"variant={variant_index}"
        )
    return prompt, token_count


def _build_prompt_variants(
    tokenizer: Any,
    input_tokens: int,
    count_to: int,
    prefix_cache_rate: float,
    num_variants: int,
    start_index: int = 0,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """Build exact-length prompts with a measured shared chat-token prefix."""
    if not 0.0 < prefix_cache_rate < 1.0:
        raise ValueError("prefix_cache_rate must be between zero and one")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    target_common_tokens = int(input_tokens * prefix_cache_rate)
    common_body_tokens = max(1, target_common_tokens - 16)
    calibration_prefix = 0
    for _ in range(8):
        calibration_prompts = [
            _fit_counting_variant(
                tokenizer,
                input_tokens,
                count_to,
                common_body_tokens,
                variant_index,
            )[0]
            for variant_index in (0, 1)
        ]
        calibration_ids = [
            _chat_token_ids(tokenizer, prompt) for prompt in calibration_prompts
        ]
        calibration_prefix = _common_prefix_length(
            calibration_ids[0], calibration_ids[1]
        )
        prefix_error = target_common_tokens - calibration_prefix
        if abs(prefix_error) <= 2:
            break
        common_body_tokens = max(1, common_body_tokens + prefix_error)

    variants = [
        _fit_counting_variant(
            tokenizer,
            input_tokens,
            count_to,
            common_body_tokens,
            variant_index,
        )
        for variant_index in range(start_index, start_index + num_variants)
    ]
    variant_ids = [_chat_token_ids(tokenizer, prompt) for prompt, _ in variants]
    measured_prefix = calibration_prefix
    if len(variant_ids) > 1:
        measured_prefix = min(
            _common_prefix_length(variant_ids[0], token_ids)
            for token_ids in variant_ids[1:]
        )
    metadata = {
        "configured_prefix_cache_rate": prefix_cache_rate,
        "target_common_chat_tokens": target_common_tokens,
        "calibrated_common_chat_tokens": calibration_prefix,
        "measured_variant_common_chat_tokens": measured_prefix,
        "measured_variant_common_rate": round(measured_prefix / input_tokens, 6),
        "common_body_tokens": common_body_tokens,
        "start_index": start_index,
        "num_variants": num_variants,
    }
    logger.info("Prefix-cache prompt metadata: %s", metadata)
    return variants, metadata


def _parse_stream(
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
    ignore_eos: bool,
    skip_special_tokens: bool,
    record_stream_chunks: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "skip_special_tokens": skip_special_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_content_at: float | None = None
    last_content_at: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    chunk_count = 0
    stream_chunks: list[dict[str, Any]] = []
    stream_errors: list[Any] = []

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error") is not None:
                stream_errors.append(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text is None:
                text = choice.get("text")
            if not text:
                continue
            now = time.perf_counter()
            if first_content_at is None:
                first_content_at = now
            last_content_at = now
            content_parts.append(str(text))
            chunk_count += 1
            if record_stream_chunks:
                stream_chunks.append(
                    {
                        "elapsed_ms": round((now - started) * 1000.0, 3),
                        "content": str(text),
                    }
                )

    ended = time.perf_counter()
    if first_content_at is None:
        details = {
            "errors": stream_errors,
            "usage": usage,
            "finish_reason": finish_reason,
            "nonempty_content_chunks": chunk_count,
        }
        raise RuntimeError(
            "stream completed without a content token: "
            + json.dumps(details, ensure_ascii=False)
        )

    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens > 0:
        completion_tokens_source = "usage"
    elif finish_reason == "length":
        # xLLM may omit usage in streaming responses, while the service log
        # still reports the requested max_tokens as generated_tokens.
        completion_tokens = output_tokens
        completion_tokens_source = "requested_max_tokens_on_length"
    else:
        completion_tokens = chunk_count
        completion_tokens_source = "nonempty_sse_chunks"
    ttft_ms = (first_content_at - started) * 1000.0
    e2e_ms = (ended - started) * 1000.0
    tpot_ms = (
        (ended - first_content_at) * 1000.0 / (completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )
    return {
        "usage": usage,
        "completion_tokens": completion_tokens,
        "completion_tokens_source": completion_tokens_source,
        "stream_content_chunks": chunk_count,
        "finish_reason": finish_reason,
        "ttft_ms": round(ttft_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "tgs_tokens_per_second": round(1000.0 / tpot_ms, 3) if tpot_ms else 0.0,
        "content": "".join(content_parts),
        "last_content_ms": round((last_content_at - started) * 1000.0, 3)
        if last_content_at is not None
        else None,
        "stream_chunks": stream_chunks if record_stream_chunks else None,
    }


def _parse_non_stream(
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
    ignore_eos: bool,
    skip_special_tokens: bool,
) -> dict[str, Any]:
    """Issue the same request without SSE to isolate response-stream races."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "skip_special_tokens": skip_special_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    ended = time.perf_counter()

    if body.get("error") is not None:
        raise RuntimeError(
            "non-stream request failed: "
            + json.dumps(body["error"], ensure_ascii=False)
        )
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(
            "non-stream response has no choice: "
            + json.dumps(body, ensure_ascii=False)
        )
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text") or ""
    usage = body.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "usage": usage,
        "completion_tokens": completion_tokens,
        "completion_tokens_source": "usage" if completion_tokens else "unavailable",
        "stream_content_chunks": None,
        "finish_reason": choice.get("finish_reason"),
        "ttft_ms": None,
        "e2e_ms": round((ended - started) * 1000.0, 3),
        "tpot_ms": None,
        "tgs_tokens_per_second": None,
        "content": str(content),
        "last_content_ms": None,
        "stream_chunks": None,
    }


def _validate_counting(content: str, count_to: int) -> dict[str, Any]:
    numbers = [int(value) for value in re.findall(r"\d+", content)]
    expected = list(range(1, count_to + 1))
    mismatch_index = next(
        (index for index, pair in enumerate(zip(numbers, expected)) if pair[0] != pair[1]),
        None,
    )
    if mismatch_index is None and len(numbers) != len(expected):
        mismatch_index = min(len(numbers), len(expected))
    window_start = max(0, (mismatch_index or 0) - 4)
    window_end = min(
        max(len(numbers), len(expected)), (mismatch_index or 0) + 5
    )
    return {
        "passed": numbers == expected,
        "expected_count": count_to,
        "actual_count": len(numbers),
        "first_numbers": numbers[:10],
        "last_numbers": numbers[-10:],
        "mismatch_index": mismatch_index,
        "mismatch_expected": (
            expected[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(expected)
            else None
        ),
        "mismatch_actual": (
            numbers[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(numbers)
            else None
        ),
        "mismatch_expected_window": (
            expected[window_start:window_end] if mismatch_index is not None else []
        ),
        "mismatch_actual_window": (
            numbers[window_start:window_end] if mismatch_index is not None else []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-tokens", type=int, default=3500)
    parser.add_argument("--output-tokens", type=int, default=1500)
    parser.add_argument("--count-to", type=int, default=750)
    parser.add_argument(
        "--prefix-cache-rate",
        type=float,
        default=0.0,
        help="Build a prompt family with this measured common-prefix rate.",
    )
    parser.add_argument(
        "--prompt-variant",
        type=int,
        default=0,
        help="Deterministic suffix variant used with --prefix-cache-rate.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Match throughput benchmarks that always decode max_tokens.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Keep special tokens in streamed text for correctness diagnosis.",
    )
    parser.add_argument(
        "--allow-accuracy-failure",
        action="store_true",
        help="Return success after recording an expected diagnostic mismatch.",
    )
    parser.add_argument(
        "--record-stream-chunks",
        action="store_true",
        help="Record nonempty SSE chunk text and arrival time for TPOT diagnostics.",
    )
    parser.add_argument(
        "--non-stream",
        action="store_true",
        help="Use one non-streaming response to isolate SSE response races.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    prompt_metadata: dict[str, Any] | None = None
    if args.prefix_cache_rate > 0.0:
        prompts, prompt_metadata = _build_prompt_variants(
            tokenizer,
            args.input_tokens,
            args.count_to,
            args.prefix_cache_rate,
            num_variants=1,
            start_index=args.prompt_variant,
        )
        prompt, prompt_tokens = prompts[0]
    else:
        prompt, prompt_tokens = _build_prompt(
            tokenizer, args.input_tokens, args.count_to
        )
    if args.non_stream:
        result = _parse_non_stream(
            args.url,
            args.model,
            prompt,
            args.output_tokens,
            args.timeout,
            args.ignore_eos,
            not args.include_special_tokens,
        )
    else:
        result = _parse_stream(
            args.url,
            args.model,
            prompt,
            args.output_tokens,
            args.timeout,
            args.ignore_eos,
            not args.include_special_tokens,
            args.record_stream_chunks,
        )
    content = result.pop("content")
    report = {
        "model": args.model,
        "url": args.url,
        "prompt_tokens_built": prompt_tokens,
        "output_tokens_requested": args.output_tokens,
        "count_to": args.count_to,
        "prompt_variant": args.prompt_variant,
        "prefix_cache_prompt": prompt_metadata,
        "ignore_eos": args.ignore_eos,
        "stream": not args.non_stream,
        "skip_special_tokens": not args.include_special_tokens,
        "metrics": result,
        "accuracy": _validate_counting(content, args.count_to),
        "content_prefix": content[:256],
        "content_suffix": content[-256:],
    }
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_text, flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report_text + "\n", encoding="utf-8")
    return 0 if report["accuracy"]["passed"] or args.allow_accuracy_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
