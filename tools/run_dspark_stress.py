#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Bounded concurrent streaming tests; persist every attempt, never retry errors."""
import argparse
import asyncio
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


async def _request(client: aiohttp.ClientSession, url: str, case: dict) -> dict:
    row = dict(case)
    row["generation_start_time"] = time.time()
    started = time.perf_counter()
    first = last = None
    chunks = []
    usage = {}
    finish = None
    try:
        async with client.post(url, json=case["request_payload"]) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:2048]}")
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                    text = choice.get("delta", {}).get("content")
                    if text:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        chunks.append(text)
            if first is None or not usage or finish is None:
                raise RuntimeError("stream ended without content/usage/finish")
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    text = "".join(chunks)
    tokens = usage.get("completion_tokens", 0)
    row.update(generation=text, usage=usage, finish_reason=finish,
               generation_end_time=time.time(), latency_seconds=time.perf_counter() - started,
               prompt_tokens=usage.get("prompt_tokens"), num_generated_tokens=tokens,
               ttft_ms=(first - started) * 1000 if first else None,
               tpot_ms=(last - first) * 1000 / (tokens - 1) if first and tokens > 1 else None,
               exact_prompt_usage=usage.get("prompt_tokens") == case["prompt_tokens_built"])
    if case["kind"] == "counting":
        row["strict_counting_pass"] = text.split() == [str(i) for i in range(case["count_from"], case["count_to"] + 1)]
        row["exact_output_usage"] = tokens == case["output_tokens_requested"]
    return row


def _summary(rows: list[dict], planned: int, duration: float, concurrency: int) -> dict:
    good = [r for r in rows if "error" not in r]
    ttfts = [r["ttft_ms"] for r in good if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in good if r["tpot_ms"] is not None]
    inputs = sum(r["usage"].get("prompt_tokens", 0) for r in good)
    outputs = sum(r["usage"].get("completion_tokens", 0) for r in good)
    return {"planned": planned, "attempted": len(rows), "complete": len(rows) == planned,
            "client_concurrency_limit": concurrency, "physical_batch": "must verify server logs separately",
            "http_successes": len(good), "request_errors": len(rows) - len(good),
            "strict_counting_passes": sum(bool(r.get("strict_counting_pass")) for r in rows),
            "input_usage_mismatches": [r["id"] for r in good if not r["exact_prompt_usage"]],
            "output_usage_mismatches": [r["id"] for r in good if r.get("exact_output_usage") is False],
            "failure_ids": [r["id"] for r in rows if "error" in r or r.get("strict_counting_pass") is False],
            "finish_reasons": dict(collections.Counter(r.get("finish_reason") for r in rows)),
            "duration_seconds": duration, "ttft_avg_ms": statistics.mean(ttfts) if ttfts else None,
            "tpot_avg_ms": statistics.mean(tpots) if tpots else None,
            "input_token_throughput": inputs / duration if duration else None,
            "output_token_throughput": outputs / duration if duration else None,
            "total_token_throughput": (inputs + outputs) / duration if duration else None,
            "throughput_scope": "8GPU aggregate, completed usage / complete wall interval, includes gaps/errors",
            "gsm8k_accuracy": "use original NeMo offline scorer; not inferred from HTTP success"}


async def _run(args: argparse.Namespace) -> None:
    run = args.output_dir.resolve()
    if not run.is_relative_to(ROOT / "logs") or run.exists():
        raise ValueError("use a new output directory inside logs")
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    cases = cases[args.start:args.start + args.limit] if args.limit else cases[args.start:]
    if not cases:
        raise ValueError("no cases selected")
    run.mkdir(parents=True)
    manifest = dict(vars(args), dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    transport_retries=0, request_count=len(cases), started=time.time())
    (run / "config.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout),
                                    connector=aiohttp.TCPConnector(limit=args.concurrency)) as client:
        for _ in range(300):
            try:
                async with client.get(args.url.rsplit("/v1/", 1)[0] + "/health",
                                      timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("readiness deadline")
        rows = []
        stop = asyncio.Event()
        iterator = iter(cases)
        started = time.perf_counter()
        with (run / "attempts.jsonl").open("x", buffering=1) as attempts, \
             (run / "responses.raw.jsonl").open("x", buffering=1) as output:
            async def worker() -> None:
                while not stop.is_set():
                    case = next(iterator, None)
                    if case is None:
                        return
                    attempts.write(json.dumps({"id": case["id"], "started": time.time()}, ensure_ascii=False) + "\n")
                    row = await _request(client, args.url, case)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    logger.info("completed=%d/%d id=%s tokens=%s count_ok=%s error=%s",
                                len(rows), len(cases), row["id"], row["num_generated_tokens"],
                                row.get("strict_counting_pass"), row.get("error"))
                    (run / "progress.json").write_text(json.dumps(
                        _summary(rows, len(cases), time.perf_counter() - started, args.concurrency), indent=2) + "\n")
                    # Avoid flooding a crashed server. Every already submitted
                    # request is still recorded; unattempted requests are explicit.
                    if sum("error" in r for r in rows) >= args.stop_errors:
                        stop.set()
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        summary = _summary(rows, len(cases), time.perf_counter() - started, args.concurrency)
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("Run summary: %s", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:19467/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--stop-errors", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 16 or args.timeout <= 0:
        parser.error("invalid concurrency/timeout")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
