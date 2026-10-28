#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare and replay identical TP2 decode workloads for MCCL/custom-AR comparisons."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.logger import logger


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.mean(values) if values else None,
            "p50": _percentile(values, 50), "p95": _percentile(values, 95),
            "max": max(values) if values else None}


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return _sha(_canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"}))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["schema_version"] != 1 or manifest["manifest_sha256"] != _manifest_digest(manifest):
        raise ValueError("Manifest schema or SHA256 is invalid")
    first_blocks = set()
    for case in manifest["cases"]:
        ids, text = case["token_ids"], case["prompt_text"]
        if len(ids) != manifest["isl"] or not text.isascii():
            raise ValueError(f"Invalid input length or non-ASCII text: {case['key']}")
        if _sha(_canonical(ids)) != case["token_ids_sha256"] or _sha(text.encode()) != case["text_sha256"]:
            raise ValueError(f"Input hash mismatch: {case['key']}")
        block = tuple(ids[:64])
        if block in first_blocks:
            raise ValueError("Independent prepared cases share their first KV block")
        first_blocks.add(block)


def _prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    if args.manifest.exists():
        raise FileExistsError(f"Preserve the existing manifest: {args.manifest}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    cases = []
    unit = "The quick brown fox jumps over the lazy dog. "
    for concurrency in (1, 4, 8):
        # Eight prepared requests per group allow the same manifest to supply
        # every requested concurrency. Distinct C groups prevent cross-C hits.
        for mode, group_count in (("warm", 1), ("cold", args.warmups + args.rounds)):
            for group in range(group_count):
                for index in range(8):
                    key = f"c{concurrency}-{mode}-g{group}-r{index}"
                    header = f"Benchmark {args.seed} {key}. Continue the following text.\n"
                    if len(tokenizer.encode(header, add_special_tokens=False)) >= 64:
                        raise ValueError("Prepared header must fit inside the first KV block")
                    ids = tokenizer.encode(header + unit * (args.isl + 1), add_special_tokens=False)[:args.isl]
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if tokenizer.encode(text, add_special_tokens=False) != ids or not text.isascii():
                        raise ValueError(f"Token decode/re-encode mismatch: {key}")
                    cases.append({"key": key, "concurrency": concurrency, "cache_mode": mode,
                                  "group": group, "request_index": index, "token_ids": ids,
                                  "prompt_text": text, "token_ids_sha256": _sha(_canonical(ids)),
                                  "text_sha256": _sha(text.encode())})
    manifest = {"schema_version": 1, "model": args.model, "model_dir": str(args.model_dir),
                "seed": args.seed, "isl": args.isl, "osl": args.osl,
                "warmups": args.warmups, "rounds": args.rounds,
                "sampling": {"temperature": 0, "ignore_eos": True, "add_special_tokens": False},
                "cases": cases}
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    _validate_manifest(manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    _save(args.manifest, manifest)
    logger.info(f"Prepared {len(cases)} exact-length round-trippable ASCII inputs")
    print(json.dumps({"manifest": str(args.manifest), "manifest_sha256": manifest["manifest_sha256"],
                      "cases": len(cases), "isl": args.isl, "osl": args.osl}))


def _request(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case: dict[str, Any],
    wave: str,
    measured: bool,
    barrier: threading.Barrier,
    origin: list[float],
) -> dict[str, Any]:
    key = f"{wave}-r{case['request_index']}"
    request_id = f"tp2-{manifest['manifest_sha256'][:12]}-{key}"
    body = {"model": manifest["model"],
            "messages": [{"role": "user", "content": case.get("request_prompt_text", case["prompt_text"])}],
            "max_tokens": manifest["osl"], **manifest["sampling"],
            "stream": True, "stream_options": {"include_usage": True},
            "request_id": request_id, "x_request_id": request_id}
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    payload = _canonical(body)
    request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", data=payload, headers=headers)
    _save(args.output / f"{key}.request.json", {"body": body, "headers": headers,
                                              "body_sha256": _sha(payload)})
    result: dict[str, Any] = {
        "key": key, "case_key": case["key"], "measured": measured,
        "request_id": request_id, "manifest_sha256": manifest["manifest_sha256"],
        "token_ids_sha256": case["token_ids_sha256"], "prompt_text_sha256": case["text_sha256"],
        "request_body_sha256": _sha(payload), "events": [], "server_response_ids": [],
        "usage": None, "finish_reasons": [], "done_seen": False,
    }
    texts: list[str] = []
    first_content: float | None = None
    last_content: float | None = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    barrier.wait(timeout=60)
    start = time.perf_counter()
    result["start_offset_ms"] = (start - origin[0]) * 1000
    try:
        with (args.output / f"{key}.sse").open("wb") as raw_file:
            with opener.open(request, timeout=args.timeout) as response:
                result["http_status"], result["response_headers"] = response.status, dict(response.headers)
                data_lines: list[str] = []

                def consume_event(now: float) -> None:
                    nonlocal first_content, last_content
                    data = "\n".join(data_lines)
                    data_lines.clear()
                    if data == "[DONE]":
                        result["done_seen"] = True
                        return
                    message = json.loads(data)
                    result["events"].append({"elapsed_ms": (now - start) * 1000, "message": message})
                    if message.get("id") and message["id"] not in result["server_response_ids"]:
                        result["server_response_ids"].append(message["id"])
                    if message.get("error"):
                        raise RuntimeError(f"Streaming error: {message['error']}")
                    if message.get("usage"):
                        result["usage"] = message["usage"]
                    for choice in message.get("choices", []):
                        content = choice.get("text")
                        if content is None:
                            content = (choice.get("delta") or {}).get("content")
                        if content:
                            texts.append(content)
                            first_content = now if first_content is None else first_content
                            last_content = now
                        if choice.get("finish_reason"):
                            result["finish_reasons"].append(choice["finish_reason"])

                for raw in response:
                    now = time.perf_counter()
                    raw_file.write(raw)
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        consume_event(now)
                if data_lines:
                    consume_event(time.perf_counter())
    except urllib.error.HTTPError as error:
        result["http_status"] = error.code
        result["http_error_body"] = error.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTPError: {error}"
    except (OSError, TimeoutError, ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    end = time.perf_counter()
    result["end_offset_ms"] = (end - origin[0]) * 1000
    result["latency_ms"] = (end - start) * 1000
    result["first_content_ms"] = None if first_content is None else (first_content - start) * 1000
    result["last_content_ms"] = None if last_content is None else (last_content - start) * 1000
    result["text"] = "".join(texts)
    result["text_sha256"] = _sha(result["text"].encode())
    denominator = manifest["osl"] - 1
    result["eof_amortized_tpot_ms"] = (
        (end - first_content) * 1000 / denominator if first_content is not None and denominator > 0 else None)
    result["content_amortized_tpot_ms"] = (
        (last_content - first_content) * 1000 / denominator
        if first_content is not None and last_content is not None and denominator > 0 else None)
    usage = result["usage"] or {}
    result["reported_cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result["valid"] = (
        "error" not in result and result.get("http_status") == 200 and result["done_seen"]
        and usage.get("prompt_tokens") == manifest["isl"] and usage.get("completion_tokens") == manifest["osl"]
        and result["finish_reasons"] == ["length"] and bool(result["text"])
    )
    _save(args.output / f"{key}.result.json", result)
    return result


def _wave(args: argparse.Namespace, manifest: dict[str, Any], mode: str, index: int) -> dict[str, Any]:
    measured = index >= manifest["warmups"]
    phase = "measure" if measured else "warmup"
    phase_index = index - manifest["warmups"] if measured else index
    wave = f"c{args.concurrency}-{mode}-{phase}{phase_index}"
    group = 0 if mode == "warm" else index
    cases = sorted(
        (case for case in manifest["cases"] if case["concurrency"] == args.concurrency
         and case["cache_mode"] == mode and case["group"] == group),
        key=lambda case: case["request_index"],
    )[:args.concurrency]
    if len(cases) != args.concurrency:
        raise ValueError(f"Prepared cases are missing for {wave}")
    origin: list[float] = []
    barrier = threading.Barrier(args.concurrency + 1, action=lambda: origin.append(time.perf_counter()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_request, args, manifest, case, wave, measured, barrier, origin) for case in cases]
        barrier.wait(timeout=60)
        results = [future.result() for future in futures]
    wall_ms = max(result["end_offset_ms"] for result in results)
    count = sum((result["usage"] or {}).get("completion_tokens", 0) for result in results)
    summary = {"wave": wave, "measured": measured, "all_valid": all(item["valid"] for item in results),
               "concurrency": args.concurrency, "cache_mode": mode, "wall_ms": wall_ms,
               "output_tokens": count, "output_tokens_per_s": count * 1000 / wall_ms,
               "ttft_ms": _stats([item["first_content_ms"] for item in results if item["first_content_ms"] is not None]),
               "eof_amortized_tpot_ms": _stats([item["eof_amortized_tpot_ms"] for item in results
                                                if item["eof_amortized_tpot_ms"] is not None]),
               "content_amortized_tpot_ms": _stats([item["content_amortized_tpot_ms"] for item in results
                                                    if item["content_amortized_tpot_ms"] is not None]),
               "request_ids": [item["request_id"] for item in results],
               "start_spread_ms": max(item["start_offset_ms"] for item in results)
                                  - min(item["start_offset_ms"] for item in results)}
    _save(args.output / f"{wave}.summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["all_valid"]:
        raise RuntimeError(f"Wave {wave} failed; raw HTTP/SSE/result evidence is retained")
    return summary


def _run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / "run.json", {
        "manifest": str(args.manifest), "manifest_sha256": manifest["manifest_sha256"],
        "script_sha256": _sha(Path(__file__).read_bytes()), "base_url": args.base_url,
        "concurrency": args.concurrency, "cache_mode": args.cache_mode,
        "timing_notes": "TTFT is first nonempty text. Amortized TPOT uses OSL-1; MTP SSE bursts can contain several tokens. "
                        "p95 summarizes request-level averages, not individual token intervals. No stalls are removed.",
    })
    summaries = []
    modes = ("cold", "warm") if args.cache_mode == "both" else (args.cache_mode,)
    for mode in modes:
        for index in range(manifest["warmups"] + manifest["rounds"]):
            summaries.append(_wave(args, manifest, mode, index))
            # Prefix publication / teardown occurs outside the timed wave.
            time.sleep(args.settle_s)
    aggregate = {}
    for mode in modes:
        waves = [wave for wave in summaries if wave["measured"] and wave["cache_mode"] == mode]
        selected = [json.loads(path.read_text()) for path in args.output.glob(f"c{args.concurrency}-{mode}-measure*.result.json")]
        aggregate[mode] = {
            "measured_requests": len(selected), "all_valid": all(item["valid"] for item in selected),
            "output_tokens_per_s": sum(wave["output_tokens"] for wave in waves) * 1000
                                   / sum(wave["wall_ms"] for wave in waves),
            "ttft_ms": _stats([item["first_content_ms"] for item in selected]),
            "eof_amortized_tpot_ms": _stats([item["eof_amortized_tpot_ms"] for item in selected]),
            "content_amortized_tpot_ms": _stats([item["content_amortized_tpot_ms"] for item in selected]),
            "reported_cached_tokens": [item["reported_cached_tokens"] for item in selected],
        }
    _save(args.output / "aggregate.json", aggregate)
    print(json.dumps(aggregate), flush=True)


def _native_logs(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    matches: dict[str, list[dict[str, str]]] = {}
    pattern = re.compile(r"x-request-id:\s*([^,]*),.*?\brequest_id:\s*([^,]+),")
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                match = pattern.search(line)
                if match:
                    matches.setdefault(match.group(1), []).append(
                        {"file": str(path), "line": str(line_number), "native_request_id": match.group(2),
                         "text": line.rstrip()})
    return matches


def _validate_output(args: argparse.Namespace) -> None:
    sides = []
    for directory, log_paths in ((args.left, args.left_log), (args.right, args.right_log)):
        results = {}
        for path in directory.rglob("*.result.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["key"] in results:
                raise ValueError(f"Duplicate replay key: {result['key']}")
            results[result["key"]] = result
        sides.append((results, _native_logs(log_paths)))
    left, left_logs = sides[0]
    right, right_logs = sides[1]
    comparisons = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        comparisons.append({
            "key": key, "measured": a["measured"] and b["measured"],
            "same_manifest": a["manifest_sha256"] == b["manifest_sha256"],
            "same_token_ids": a["token_ids_sha256"] == b["token_ids_sha256"],
            "same_text_input": a["prompt_text_sha256"] == b["prompt_text_sha256"],
            "same_body": a["request_body_sha256"] == b["request_body_sha256"],
            "both_valid": a["valid"] and b["valid"], "same_output_text": a["text_sha256"] == b["text_sha256"],
            "left_native_matches": left_logs.get(a["request_id"], []),
            "right_native_matches": right_logs.get(b["request_id"], []),
            "left_response_ids": a["server_response_ids"], "right_response_ids": b["server_response_ids"],
        })
    same_keys = bool(left) and left.keys() == right.keys()
    operational = same_keys and all(all(item[k] for k in (
        "same_manifest", "same_token_ids", "same_text_input", "same_body", "both_valid")) for item in comparisons)
    report = {
        "operational_replay_validation": operational, "same_request_set": same_keys,
        "left_only": sorted(left.keys() - right.keys()), "right_only": sorted(right.keys() - left.keys()),
        "full_text_equivalence": operational and all(item["same_output_text"] for item in comparisons),
        "different_output_keys": [item["key"] for item in comparisons if not item["same_output_text"]],
        "native_matching_requested": bool(args.left_log or args.right_log),
        "comparisons": comparisons,
        "interpretation": "Text differences are retained; identical temperature does not establish numerical equivalence "
                          "across different batch schedules or kernels. Native matches use exact x-request-id, never timestamps.",
    }
    _save(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}), flush=True)
    if not operational or (args.require_identical_text and not report["full_text_equivalence"]):
        raise RuntimeError("Replay validation failed; inspect the retained comparison report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare", aliases=["prepare-only"], help="CPU-only deterministic tokenization; no HTTP")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--model-dir", type=Path, default=Path("/data/nfs_shared/models/Qwen3.8-27B-FP8"))
    prepare.add_argument("--model", default="Qwen3.8-27B-FP8")
    prepare.add_argument("--seed", default="20261027-v1")
    prepare.add_argument("--isl", type=int, default=2048)
    prepare.add_argument("--osl", type=int, default=40)
    prepare.add_argument("--warmups", type=int, default=2)
    prepare.add_argument("--rounds", type=int, default=3)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--base-url", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--concurrency", type=int, choices=(1, 4, 8), required=True)
    run.add_argument("--cache-mode", choices=("cold", "warm", "both"), default="both")
    run.add_argument("--settle-s", type=float, default=2)
    run.add_argument("--timeout", type=float, default=300)
    compare = subparsers.add_parser("validate-output")
    compare.add_argument("--left", required=True, type=Path)
    compare.add_argument("--right", required=True, type=Path)
    compare.add_argument("--left-log", type=Path, action="append", default=[])
    compare.add_argument("--right-log", type=Path, action="append", default=[])
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--require-identical-text", action="store_true")
    args = parser.parse_args()
    if args.action in ("prepare", "prepare-only"):
        if args.isl < 64 or args.osl < 2 or args.warmups < 1 or args.rounds < 1:
            parser.error("Require ISL>=64, OSL>=2, at least one warmup and one measured round")
        _prepare(args)
    elif args.action == "run":
        if args.settle_s < 0 or args.timeout <= 0:
            parser.error("Settle time must be nonnegative and timeout must be positive")
        _run(args)
    else:
        _validate_output(args)


if __name__ == "__main__":
    main()
