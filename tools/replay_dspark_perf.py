#!/usr/bin/env python3
"""Replay saved chat payloads without prompt construction between requests."""

import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def stream_request(url, payload, expected_count):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    first = last = None
    usage = {}
    finish = None
    chunks = []
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks.append(content)
    end = time.perf_counter()
    count = usage.get("completion_tokens", 0)
    text = "".join(chunks)
    expected = [str(i) for i in range(1, expected_count + 1)]
    return {
        "usage": usage, "finish_reason": finish, "content": text,
        "strict_text_passed": text.split() == expected,
        "ttft_ms": (first - start) * 1000 if first is not None else None,
        "tpot_ms": ((last - first) * 1000 / (count - 1)
                    if first is not None and count > 1 else None),
        "e2e_ms": (end - start) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--last", type=int, default=7)
    args = parser.parse_args()
    if args.first < 0 or args.last < args.first:
        parser.error("invalid request range")
    # All preparation is outside the measured interval, just as model loading
    # and warmup are outside the serving benchmark interval.
    records = []
    for variant in range(args.first, args.last + 1):
        path = args.input_dir / f"formal_variant_{variant}.json"
        record = json.loads(path.read_text())
        if not record.get("accuracy", {}).get("strict_text_passed"):
            raise ValueError(f"reference correctness failed: {path}")
        records.append((variant, path, record))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.perf_counter()
    for variant, path, record in records:
        try:
            result = stream_request(args.url, record["request_payload"],
                                    record["count_to"])
            result["exact_usage"] = (
                result["usage"].get("prompt_tokens") == record["prompt_tokens_built"]
                and result["usage"].get("completion_tokens")
                == record["output_tokens_requested"])
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        result.update(variant=variant, source=str(path),
                      request_payload=record["request_payload"])
        results.append(result)
        if "error" in result:
            break
    duration = time.perf_counter() - started
    # Persist outside the measurement interval, including every failed result.
    for result in results:
        path = args.output_dir / f"replay_{result['variant']}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    valid = len(results) == len(records) and all(
        result.get("strict_text_passed") and result.get("exact_usage")
        and result.get("finish_reason") == "length"
        and all(isinstance(result.get(key), (int, float))
                and math.isfinite(result[key]) and result[key] > 0
                for key in ("ttft_ms", "tpot_ms", "e2e_ms"))
        for result in results)
    summary = {
        "global_batch": 1, "total_requests_planned": len(records),
        "total_requests_completed": len(results),
        "strict_pass_count": sum(bool(x.get("strict_text_passed")) for x in results),
        "duration_seconds": duration, "gate": "PASS" if valid else "FAIL",
        "throughput_denominator": "whole sequential replay wall time, including HTTP gaps",
        "prefix_input_tokens_included": True,
    }
    if valid:
        summary.update(
            ttft_avg_ms=statistics.mean(x["ttft_ms"] for x in results),
            tpot_avg_ms=statistics.mean(x["tpot_ms"] for x in results),
            input_tgs=sum(x["usage"]["prompt_tokens"] for x in results) / duration,
            output_tgs=sum(x["usage"]["completion_tokens"] for x in results) / duration,
            total_tgs=sum(x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"]
                          for x in results) / duration,
            cached_tokens=[x["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                           for x in results])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
