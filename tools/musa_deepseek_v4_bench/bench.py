#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Run one closed-loop DeepSeek V4 text benchmark with exact token counts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

_CONTEXT = (
    "Transformer inference separates prompt processing from token generation. "
    "The prefill worker processes the complete input and sends its cache to "
    "the decode worker. Both stages use the same model and tokenizer. "
)
_INSTRUCTION = (
    "\n\nIgnore the background text. Output only consecutive integers "
    "from 1 through 1000 separated by one ASCII space. "
    "Do not add explanations or punctuation."
)


class _Tokenizer:
    """Match the C++ template's single-user, no-tools, non-thinking branch."""

    def __init__(self, directory: Path) -> None:
        from tokenizers import Tokenizer

        if json.loads((directory / "config.json").read_text(encoding="utf-8"))["model_type"] != "deepseek_v4":
            raise ValueError("Expected a DeepSeek V4 model/tokenizer directory")
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        bos = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8")).get("bos_token")
        self._bos = (bos.get("content") if isinstance(bos, dict) else bos) or "<｜begin▁of▁sentence｜>"

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def prefix_ids(self, text: str) -> list[int]:
        return self.encode(self._bos + "<｜User｜>" + text)

    def chat_ids(self, text: str) -> list[int]:
        return self.prefix_ids(text + "<｜Assistant｜></think>")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lcp(left: list[int], right: list[int]) -> int:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p90": percentile(0.90),
        "p99": percentile(0.99),
    }


def _shared_prefix(tokenizer: _Tokenizer, nonce: str, filler: list[int], target: int) -> str:
    header = nonce + "\nShared background notes (ignore for the answer):\n"
    estimate = target - len(tokenizer.prefix_ids(header + "\n\n"))
    for offset in [0] + [x for distance in range(1, 49) for x in (-distance, distance)]:
        if estimate + offset < 0:
            continue
        text = header + tokenizer.decode(filler[: estimate + offset])
        for padding in range(5):
            candidate = text + " x" * padding + "\n\n"
            if len(tokenizer.prefix_ids(candidate)) == target:
                return candidate
    raise ValueError(f"Cannot fit {target} shared chat-prefix tokens")


def _make_case(
    tokenizer: _Tokenizer,
    args: argparse.Namespace,
    request_id: str,
    shared: str,
    filler: list[int],
    primer: list[int] | None,
) -> tuple[dict[str, Any], list[int]]:
    marker = _sha(request_id.encode())
    header = shared + marker + "\nPrivate background notes (ignore for the answer):\n"
    estimate = args.input_tokens - len(tokenizer.chat_ids(header + _INSTRUCTION))
    for offset in [0] + [x for distance in range(1, 33) for x in (-distance, distance)]:
        if estimate + offset < 0:
            continue
        content = header + tokenizer.decode(filler[: estimate + offset])
        trial = content + _INSTRUCTION
        trial_ids = tokenizer.chat_ids(trial)
        gap = args.input_tokens - len(trial_ids)
        for padding in [0] if gap == 0 else [gap, gap - 1, gap + 1]:
            if not 0 <= padding <= 32:
                continue
            prompt = content + " x" * padding + _INSTRUCTION
            ids = trial_ids if padding == 0 else tokenizer.chat_ids(prompt)
            if len(ids) != args.input_tokens:
                continue
            lcp = _lcp(primer, ids) if primer is not None else args.input_tokens
            cached = 0
            if shared and primer is not None:
                cached = lcp // args.block_size * args.block_size
                if abs(lcp - args.shared_tokens) > 8 or cached != args.hot_cached_tokens:
                    raise ValueError(f"Unexpected shared token boundary: {request_id}, LCP={lcp}")
            return {
                "id": request_id,
                "prompt_tokens_built": len(ids),
                "prompt_sha256": _sha(prompt.encode()),
                "primer_lcp_tokens": lcp,
                "expected_cached_tokens": cached,
                "payload": {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"thinking": False},
                    "temperature": 0,
                    "max_tokens": args.output_tokens,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            }, ids
    raise ValueError(f"Cannot fit exactly {args.input_tokens} chat tokens: {request_id}")


def _prepare(args: argparse.Namespace, tokenizer: _Tokenizer) -> tuple[list[dict[str, Any]], str]:
    canonical = " ".join(str(value) for value in range(1, 1001))
    expected = tokenizer.decode(tokenizer.encode(canonical)[: args.output_tokens])
    words = expected.split()
    if (
        len(tokenizer.encode(expected)) != args.output_tokens
        or len(words) < 40
        or words != [str(value) for value in range(1, len(words) + 1)]
    ):
        raise ValueError("Counting text does not round-trip the requested output token budget")
    run_id = uuid.uuid4().hex
    unit = tokenizer.encode(_CONTEXT)
    filler = tokenizer.encode(_CONTEXT * ((args.input_tokens + 512) // len(unit) + 20))
    shared = _shared_prefix(tokenizer, _sha(run_id.encode()), filler, args.shared_tokens) if args.prefix_ratio else ""
    plans = [("warmup", 1, args.warmup)] if args.warmup else []
    if shared:
        plans.append(("probe", 1, 1))
    plans.append(("formal", args.concurrency, args.requests))
    stages, primer, unique_blocks = [], None, set()
    for name, concurrency, count in plans:
        cases = []
        for index in range(count):
            case, ids = _make_case(tokenizer, args, f"{run_id}_{name}_{index:04d}", shared, filler, primer)
            prefix = ids[: args.hot_cached_tokens + args.block_size]
            block_hash = _sha(json.dumps(prefix, separators=(",", ":")).encode())
            if block_hash in unique_blocks:
                raise ValueError("Repeated first nonshared cache block; cache-hit expectation would be ambiguous")
            unique_blocks.add(block_hash)
            primer = ids if primer is None else primer
            cases.append(case)
        stages.append({"name": name, "concurrency": concurrency, "cases": cases})
    _write_json(args.out / "payloads.json", stages)
    _write_json(
        args.out / "manifest.json",
        {
            "run_id": run_id,
            "script_sha256": _sha(Path(__file__).read_bytes()),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "tokenizer_sha256": {
                name: _sha((args.tokenizer / name).read_bytes())
                for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
            },
            "payloads_sha256": _sha((args.out / "payloads.json").read_bytes()),
            "expected_visible_output": expected,
            "shared_prefix_sha256": _sha(shared.encode()),
            "shared_chat_tokens": args.shared_tokens,
            "hot_cached_tokens": args.hot_cached_tokens,
            "expected_hot_cache_fraction": args.hot_cached_tokens / args.input_tokens,
            "protocol": {
                "template": "deepseek-v4-cpp",
                "thinking": False,
                "retries": 0,
                "warmup_requests": args.warmup + bool(shared),
                "warmup_excluded": True,
                "tpot": "(last content - first content) / (completion_tokens - 1)",
                "percentiles": "Linear interpolation at fraction * (sample_count - 1)",
                "throughput": "validated completion usage / complete formal window wall; entire endpoint",
            },
        },
    )
    return stages, expected


async def _request(
    client: Any, args: argparse.Namespace, tokenizer: _Tokenizer, case: dict[str, Any], directory: Path, expected: str
) -> dict[str, Any]:
    import aiohttp

    started = time.perf_counter()
    row: dict[str, Any] = {"id": case["id"], "started_unix": time.time(), "started_monotonic": started}
    first = last = None
    usage: dict[str, Any] = {}
    finish, done, content, events, raw_bytes, data_lines = None, False, [], [], bytearray(), []

    def consume(data: str, arrived: float) -> None:
        nonlocal first, last, usage, finish, done
        if done:
            raise ValueError("SSE data after [DONE]")
        event = "[DONE]" if data == "[DONE]" else json.loads(data)
        events.append({"elapsed_ms": (arrived - started) * 1000, "event": event})
        if event == "[DONE]":
            done = True
            return
        if not isinstance(event, dict):
            raise ValueError("SSE event must be a JSON object")
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("SSE choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("SSE choice must be an object")
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected additional choice")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("SSE delta must be an object")
            if delta.get("reasoning_content") or delta.get("reasoning"):
                raise ValueError("Unexpected thinking output")
            text = delta.get("content")
            if text:
                if not isinstance(text, str):
                    raise ValueError("Non-text content in SSE")
                first = arrived if first is None else first
                last = arrived
                content.append(text)

    try:
        async with client.post(args.url, json=case["payload"]) as response:
            row.update(http_status=response.status, response_headers=dict(response.headers))
            if response.status != 200 or response.content_type != "text/event-stream":
                raw_bytes.extend(await response.read())
                raise RuntimeError(f"HTTP {response.status}, type={response.content_type}")
            async for raw in response.content:
                arrived = time.perf_counter()
                raw_bytes.extend(raw)
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                elif not line and data_lines:
                    consume("\n".join(data_lines), arrived)
                    data_lines.clear()
            if data_lines:
                consume("\n".join(data_lines), time.perf_counter())
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, KeyError, RuntimeError) as error:
        # Persist failed HTTP attempts and stop dispatch; there are no retries.
        row["error"] = f"{type(error).__name__}: {error}"
    ended = time.perf_counter()
    generated = "".join(content)
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    visible_tokens = len(tokenizer.encode(generated))
    checks = {
        "http_and_stream_ok": "error" not in row and done,
        "input_usage_exact": type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] == args.input_tokens,
        "output_usage_exact": type(usage.get("completion_tokens")) is int
        and usage["completion_tokens"] == args.output_tokens,
        "total_usage_exact": type(usage.get("total_tokens")) is int
        and usage["total_tokens"] == args.input_tokens + args.output_tokens,
        "cache_exact": type(cached) is int and cached == case["expected_cached_tokens"],
        "visible_output_tokens_exact": visible_tokens == args.output_tokens,
        "counting_text_exact": generated == expected,
        "finish_length": finish == "length",
        "content_timestamps": first is not None and last is not None and last > first,
    }
    row.update(
        content=generated,
        usage=usage,
        cached_tokens=cached,
        expected_cached_tokens=case["expected_cached_tokens"],
        visible_output_tokens=visible_tokens,
        finish_reason=finish,
        done_received=done,
        ended_monotonic=ended,
        ttft_ms=(first - started) * 1000 if first is not None else None,
        tpot_ms=(last - first) * 1000 / (args.output_tokens - 1) if first is not None else None,
        e2e_ms=(ended - started) * 1000,
        checks=checks,
        passed=all(checks.values()),
    )
    with (directory / (case["id"] + ".sse")).open("xb") as stream:
        stream.write(raw_bytes)
    _write_json(directory / (case["id"] + ".events.json"), events)
    _write_json(directory / (case["id"] + ".response.json"), row)
    return row


async def _run_stage(
    client: Any, args: argparse.Namespace, tokenizer: _Tokenizer, stage: dict[str, Any], expected: str
) -> dict[str, Any]:
    directory = args.out / stage["name"]
    directory.mkdir()
    pending, rows, stop = iter(stage["cases"]), [], asyncio.Event()
    started = time.perf_counter()

    async def worker() -> None:
        while not stop.is_set():
            case = next(pending, None)
            if case is None:
                return
            row = await _request(client, args, tokenizer, case, directory, expected)
            rows.append(row)
            logger.info("%s %s passed=%s", stage["name"], case["id"], row["passed"])
            if not row["passed"]:
                stop.set()

    await asyncio.gather(*(worker() for _ in range(stage["concurrency"])))
    ended = time.perf_counter()
    valid = [row for row in rows if row["passed"]]
    passed = len(valid) == len(stage["cases"])
    summary = {
        "name": stage["name"],
        "concurrency": stage["concurrency"],
        "planned": len(stage["cases"]),
        "attempted": len(rows),
        "passed_count": len(valid),
        "passed": passed,
        "failed_ids": [row["id"] for row in rows if not row["passed"]],
        "started_monotonic": started,
        "ended_monotonic": ended,
        "wall_seconds": ended - started,
        "validated_input_tokens": len(valid) * args.input_tokens,
        "validated_output_tokens": len(valid) * args.output_tokens,
        "qps": len(valid) / (ended - started) if passed else None,
        "output_token_throughput": len(valid) * args.output_tokens / (ended - started) if passed else None,
        "cached_tokens_counts": {
            str(value): sum(row["cached_tokens"] == value for row in valid)
            for value in sorted({row["cached_tokens"] for row in valid})
        },
        "performance_eligible": passed and stage["name"] == "formal",
    }
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[metric] for row in valid]
        summary[metric] = _stats(values) if passed else None
    _write_json(directory / "summary.json", summary)
    return summary


async def _execute(
    args: argparse.Namespace, tokenizer: _Tokenizer, stages: list[dict[str, Any]], expected: str
) -> dict[str, Any]:
    import aiohttp

    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        trust_env=False,
        connector=aiohttp.TCPConnector(limit=args.concurrency),
    ) as client:
        for stage in stages:
            result = await _run_stage(client, args, tokenizer, stage, expected)
            results.append(result)
            if not result["passed"]:
                break
    passed = len(results) == len(stages) and all(row["passed"] for row in results)
    summary = {
        "passed": passed,
        "stages": results,
        "formal": results[-1] if passed else None,
        "formal_requests": args.requests,
        "warmup_requests": args.warmup + bool(args.prefix_ratio),
        "manifest_sha256": _sha((args.out / "manifest.json").read_bytes()),
    }
    _write_json(args.out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full http(s)://HOST:PORT/v1/chat/completions URL")
    parser.add_argument("--model", required=True, help="Registered model ID returned by /v1/models")
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Model directory containing tokenizer.json and configs"
    )
    parser.add_argument("--input-tokens", type=int, default=20000, help="Includes the C++ chat template")
    parser.add_argument("--output-tokens", type=int, choices=(100, 1000, 1024), default=100)
    parser.add_argument(
        "--prefix-ratio", type=float, default=0.65, help="Shared input-token fraction; 0 selects cold prompts"
    )
    parser.add_argument("--block-size", type=int, default=128, help="Server prefix-cache block size")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=64, help="Formal requests, excluding warmup and probe")
    parser.add_argument(
        "--warmup", type=int, default=1, help="Sequential warmups; hot mode adds one mandatory cache probe"
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write prompts and manifest without HTTP; use a new --out for execution",
    )
    args = parser.parse_args()
    url = urlsplit(args.url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.path != "/v1/chat/completions"
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        parser.error("--url must be an HTTP(S) chat-completions endpoint without credentials, query, or fragment")
    if (
        args.input_tokens < 256
        or args.block_size < 1
        or not 1 <= args.concurrency <= 16
        or args.requests < args.concurrency
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.prefix_ratio)
        or not 0 <= args.prefix_ratio < 1
    ):
        parser.error(
            "Require input>=256, block>0, concurrency=1..16, requests>=concurrency, warmup>=0, timeout>0, ratio in [0,1)"
        )
    args.shared_tokens = int(args.input_tokens * args.prefix_ratio)
    args.hot_cached_tokens = args.shared_tokens // args.block_size * args.block_size
    if args.prefix_ratio and (
        args.warmup < 1 or args.hot_cached_tokens < args.block_size or args.input_tokens - args.shared_tokens < 256
    ):
        parser.error("Hot mode needs warmup>=1, at least one shared cache block, and 256 private tokens")
    args.tokenizer, args.out = args.tokenizer.resolve(), args.out.resolve()
    if args.out.exists():
        parser.error("--out already exists; original evidence is never overwritten")
    tokenizer = _Tokenizer(args.tokenizer)
    args.out.mkdir(parents=True, exist_ok=False)
    stages, expected = _prepare(args, tokenizer)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "inference_executed": False, "output_dir": str(args.out)}))
        return 0
    summary = asyncio.run(_execute(args, tokenizer, stages, expected))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        logger.exception("Benchmark failed; existing evidence is retained: %s", error)
        raise SystemExit(1)
