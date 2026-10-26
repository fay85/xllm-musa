#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Prepare and execute an 80-percent prefix-sharing (20K or 40K) sweep."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--repo-root", type=Path, default=Path.cwd())
bootstrap_args, _ = bootstrap.parse_known_args()
sys.path.insert(0, str(bootstrap_args.repo_root.resolve()))
from scripts.logger import logger

CONTEXT = (
    "Transformer inference separates prompt processing from token generation. "
    "The prefill worker processes the complete input and sends its cache to "
    "the decode worker. Both stages use the same model and tokenizer. "
)


class _DeepseekTokenizer:
    """Use the same tokenizer.json vocabulary as the service's Rust tokenizer."""

    def __init__(self, directory: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        config = json.loads((directory / "tokenizer_config.json").read_text())
        bos = config.get("bos_token")
        self.bos_token = bos.get("content") if isinstance(bos, dict) else bos

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chat_ids(tokenizer: Any, prompt: str, template: str) -> list[int]:
    if template == "huggingface":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, thinking=False, enable_thinking=False)
    # Exact single-user, no-tools, thinking=false branch of the service's
    # DeepseekV4CppTemplate. The real model has no HF chat_template.
    rendered = ((tokenizer.bos_token or "<｜begin▁of▁sentence｜>")
                + "<｜User｜>" + prompt + "<｜Assistant｜></think>")
    return tokenizer.encode(rendered, add_special_tokens=False)



INPUT_TOKENS = 40000
OUTPUT_TOKENS = 100
SHARED_CHAT_TOKENS = 32000
BLOCK_TOKENS = 128
EXPECTED_CACHED_TOKENS = (SHARED_CHAT_TOKENS // BLOCK_TOKENS) * BLOCK_TOKENS
ROUNDS = 2
REQUESTS_PER_ROUND = 32


def _ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _stage_plan(max_concurrency: int = 5) -> list[dict[str, Any]]:
    stages = [{"name": "primer", "phase": "primer", "concurrency": 1, "count": 2},
              {"name": "warm_probe", "phase": "probe", "concurrency": 1, "count": 2}]
    for round_index in range(1, ROUNDS + 1):
        order = (range(max_concurrency, 0, -1) if round_index == 2
                 else range(1, max_concurrency + 1))
        stages.extend({"name": f"b{concurrency}_r{round_index}", "phase": "formal",
                       "round": round_index, "concurrency": concurrency,
                       "count": REQUESTS_PER_ROUND} for concurrency in order)
    return stages


def _prefix_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode((tokenizer.bos_token or "<\uff5cbegin\u2581of\u2581sentence\uff5c>")
                            + "<\uff5cUser\uff5c>" + text)


def _make_shared_prefix(tokenizer: Any, run_nonce: str,
                        filler: list[int],
                        shared_tokens: int = SHARED_CHAT_TOKENS) -> tuple[str, list[int]]:
    header = run_nonce + "\nShared background notes (ignore for the answer):\n"
    estimate = shared_tokens - len(_prefix_ids(tokenizer, header + "\n\n"))
    offsets = [0] + [offset for distance in range(1, 49) for offset in (-distance, distance)]
    for offset in offsets:
        length = estimate + offset
        text = header + tokenizer.decode(filler[:length])
        for padding in range(5):
            candidate = text + " x" * padding + "\n\n"
            ids = _prefix_ids(tokenizer, candidate)
            if len(ids) == shared_tokens:
                return candidate, ids
    raise ValueError(f"Cannot construct exactly {shared_tokens} shared chat-prefix tokens")


def _lcp(left: list[int], right: list[int]) -> int:
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def _make_case(tokenizer: Any, args: argparse.Namespace, stage: dict[str, Any],
               request_id: str, shared: str, filler: list[int],
               primer_ids: list[int] | None) -> tuple[dict[str, Any], list[int]]:
    marker = hashlib.sha256(request_id.encode()).hexdigest()
    header = shared + marker + "\nPrivate background notes (ignore for the answer):\n"
    instruction = ("\n\nIgnore the background text. Output only consecutive integers "
                   "from 1 through 1000 separated by one ASCII space. "
                   "Do not add explanations or punctuation.")
    shared_tokens = args.input_tokens * 4 // 5
    cached_tokens = shared_tokens // BLOCK_TOKENS * BLOCK_TOKENS
    estimate = args.input_tokens - len(_chat_ids(tokenizer, header + instruction, args.template))
    offsets = [0] + [offset for distance in range(1, 33) for offset in (-distance, distance)]
    for offset in offsets:
        content = header + tokenizer.decode(filler[:estimate + offset])
        trial = content + instruction
        trial_ids = _chat_ids(tokenizer, trial, args.template)
        gap = args.input_tokens - len(trial_ids)
        for padding in ([0] if gap == 0 else [gap, gap - 1, gap + 1]):
            if padding < 0 or padding > 32:
                continue
            prompt = content + " x" * padding + instruction
            ids = trial_ids if padding == 0 else _chat_ids(tokenizer, prompt, args.template)
            if len(ids) != args.input_tokens:
                continue
            prefix_length = args.input_tokens if primer_ids is None else _lcp(primer_ids, ids)
            if primer_ids is not None and not (
                    abs(prefix_length - shared_tokens) <= 8
                    and prefix_length // BLOCK_TOKENS * BLOCK_TOKENS == cached_tokens):
                raise ValueError(f"Unexpected shared token boundary {prefix_length}: {request_id}")
            if prompt[len(shared):len(shared) + len(marker)] != marker or prompt.count(marker) != 1:
                raise ValueError("The unique suffix marker must occur once immediately after the shared prefix")
            return {"stage": stage["name"], "id": request_id,
                    "prompt_tokens_built": len(ids),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "shared_prefix_sha256": hashlib.sha256(shared.encode()).hexdigest(),
                    "unique_suffix_marker": marker, "primer_lcp_tokens": prefix_length,
                    "expected_cached_tokens": 0 if stage["phase"] == "primer" else cached_tokens,
                    "payload": {"model": args.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "chat_template_kwargs": {"thinking": False}, "temperature": 0,
                                "max_tokens": OUTPUT_TOKENS, "ignore_eos": True,
                                "stream": True, "stream_options": {"include_usage": True}}}, ids
    raise ValueError(f"Cannot fit exactly {args.input_tokens} tokens: {request_id}")


def _output_contract(tokenizer: Any) -> dict[str, Any]:
    ids = tokenizer.encode(" ".join(str(value) for value in range(1, 1001)))
    expected = tokenizer.decode(ids[:OUTPUT_TOKENS])
    if (expected.split() != [str(value) for value in range(1, 51)]
            or tokenizer.encode(expected) != ids[:OUTPUT_TOKENS]):
        raise ValueError("The 100-token counting contract must round-trip exactly 1..50")
    return {"expected_visible_text": expected, "visible_token_count": OUTPUT_TOKENS,
            "canonical_token_ids_sha256": _ids_hash(ids[:OUTPUT_TOKENS]),
            "text_sha256": hashlib.sha256(expected.encode()).hexdigest()}


def _cache_summary(rows: list[dict[str, Any]], input_tokens: int) -> dict[str, Any]:
    values = [row["cached_tokens"] for row in rows if type(row.get("cached_tokens")) is int]
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    total = sum(values)
    return {"reported_requests": len(values), "missing_or_invalid_requests": len(rows) - len(values),
            "counts": counts, "total_cached_tokens": total,
            "hit_requests": sum(value > 0 for value in values),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "fraction_of_input_tokens": total / input_tokens if input_tokens else None}


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = _DeepseekTokenizer(args.tokenizer)
    output = _output_contract(tokenizer)
    run_id = uuid.uuid4().hex
    run_nonce = hashlib.sha256(run_id.encode()).hexdigest()
    unit_length = len(tokenizer.encode(CONTEXT))
    filler = tokenizer.encode(CONTEXT * ((args.input_tokens + 512) // unit_length + 20))
    shared_tokens = args.input_tokens * 4 // 5
    cached_tokens = shared_tokens // BLOCK_TOKENS * BLOCK_TOKENS
    shared, shared_ids = _make_shared_prefix(tokenizer, run_nonce, filler, shared_tokens)
    stages = _stage_plan(args.max_concurrency)
    formal_requests = args.max_concurrency * ROUNDS * REQUESTS_PER_ROUND
    primer_ids = None
    ids_seen: set[str] = set()
    marker_seen: set[str] = set()
    hashes_seen: set[str] = set()
    lcp_min, lcp_max = args.input_tokens, 0
    count = 0
    with (args.output_dir / "payloads.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for stage in stages:
            for index in range(stage["count"]):
                request_id = f"{run_id}_{stage['name']}_{index:03d}"
                case, token_ids = _make_case(tokenizer, args, stage, request_id, shared, filler, primer_ids)
                if primer_ids is None:
                    primer_ids = token_ids
                else:
                    lcp_min = min(lcp_min, case["primer_lcp_tokens"])
                    lcp_max = max(lcp_max, case["primer_lcp_tokens"])
                if (request_id in ids_seen or case["unique_suffix_marker"] in marker_seen
                        or case["prompt_sha256"] in hashes_seen):
                    raise ValueError("Duplicate request, suffix marker, or complete prompt")
                ids_seen.add(request_id)
                marker_seen.add(case["unique_suffix_marker"])
                hashes_seen.add(case["prompt_sha256"])
                stream.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            logger.info("Prepared %s: %d exact-length prompts", stage["name"], stage["count"])
    if count != formal_requests + 4:
        raise ValueError("Unexpected total prepared request count")
    prefix = {"run_nonce": run_nonce, "requested_ratio": 0.8,
              "target_chat_tokens": shared_tokens, "block_tokens": BLOCK_TOKENS,
              "expected_cached_tokens": cached_tokens,
              "expected_cached_fraction": cached_tokens / args.input_tokens,
              "snapshot_rule": "floor(floor(prompt_tokens * 80 / 100) / 128) * 128",
              "shared_prompt_text": shared,
              "shared_prefix_sha256": hashlib.sha256(shared.encode()).hexdigest(),
              "shared_chat_prefix_token_count": len(shared_ids),
              "shared_chat_prefix_sha256": _ids_hash(shared_ids),
              "token_ids_hash_encoding": "UTF-8 json.dumps(ids,separators=(',',':'))",
              "nonprimer_lcp_min": lcp_min, "nonprimer_lcp_max": lcp_max,
              "required_prefill_environment": {"XLLM_DSV4_CHUNK_PREFIX_SNAPSHOT": "1"}}
    manifest = {"schema_version": 1, "prepared_unix": time.time(), "run_id": run_id,
                "run_nonce": run_nonce, "script_sha256": _digest(Path(__file__)),
                "source_client_sha256": "c683bcbfc5045bc6abcf0ec8388faa44a552602cbe3c5b7ffadd0a563419ed45",
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "tokenizer_sha256": {name: _digest(args.tokenizer / name)
                                     for name in ("tokenizer.json", "tokenizer_config.json", "config.json")},
                "payloads_file": "payloads.jsonl", "payloads_sha256": _digest(args.output_dir / "payloads.jsonl"),
                "requests_including_warmup": count, "formal_requests": formal_requests,
                "prefix": prefix, "output_contract": output, "stages": stages,
                "protocol": {"template": "deepseek-v4-cpp", "thinking": False,
                             "ignore_eos": True, "retries": 0, "warmup_excluded": True,
                             "primer_excluded": True, "warm_probe_excluded": True,
                             "concurrency_order": [args.batches, list(reversed(args.batches))],
                             "cache_policy": "primer=0; probe/formal=exact fixed 80-percent snapshot boundary",
                             "ttft": "first content minus HTTP request start",
                             "tpot": "(last content - first content) / (actual completion_tokens - 1)",
                             "throughput": "actual usage / complete stage wall; two 8-device Prefills plus one 8-device Decode",
                             "uncached_input_throughput": "(prompt usage - cached usage) / complete stage wall"}}
    _write_json(args.output_dir / "manifest.json", manifest)
    result = {"prepared": True, "inference_executed": False, "requests": count,
              "formal_requests": formal_requests, "output_dir": str(args.output_dir),
              "manifest_sha256": _digest(args.output_dir / "manifest.json"),
              "payloads_sha256": manifest["payloads_sha256"],
              "shared_chat_tokens": len(shared_ids), "expected_cached_tokens": cached_tokens,
              "nonprimer_lcp_min": lcp_min, "nonprimer_lcp_max": lcp_max}
    _write_json(args.output_dir / "prepare_result.json", result)
    return result


def _prepared_cases(directory: Path, stages: list[dict[str, Any]]) -> Any:
    with (directory / "payloads.jsonl").open(encoding="utf-8") as stream:
        for stage in stages:
            cases = []
            for _ in range(stage["count"]):
                line = stream.readline()
                if not line:
                    raise ValueError("Prepared payload stream ended early")
                case = json.loads(line)
                if case["stage"] != stage["name"]:
                    raise ValueError("Prepared stage ordering changed")
                cases.append(case)
            yield dict(stage, cases=cases)
        if stream.read(1):
            raise ValueError("Prepared payload stream has extra data")


async def _request(client: Any, url: str, case: dict[str, Any],
                   directory: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"id": case["id"], "started_unix": time.time(),
                           "started_monotonic": started}
    first = last = None
    usage: dict[str, Any] = {}
    finish = None
    content: list[str] = []
    done = False
    events: list[dict[str, Any]] = []
    raw_bytes = bytearray()
    data_lines: list[str] = []

    def consume(data: str, arrived: float) -> None:
        nonlocal first, last, usage, finish, done
        if data == "[DONE]":
            done = True
            return
        event = json.loads(data)
        events.append({"elapsed_ms": (arrived - started) * 1000, "event": event})
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected additional choice")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if delta.get("reasoning_content"):
                raise ValueError("Unexpected thinking output in non-thinking run")
            text = delta.get("content")
            if text:
                first = arrived if first is None else first
                last = arrived
                content.append(text)

    try:
        async with client.post(url, json=case["payload"]) as response:
            row["http_status"] = response.status
            row["response_headers"] = dict(response.headers)
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
    except Exception as error:
        # The HTTP boundary retains failed attempts and never retries them.
        row["error"] = f"{type(error).__name__}: {error}"
    ended = time.perf_counter()
    generated = "".join(content)
    expected_output = args.expected_output
    visible_tokens = len(args.runtime_tokenizer.encode(generated))
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    tokens = usage.get("completion_tokens")
    checks = {
        "http_and_stream_ok": "error" not in row and done,
        "input_usage_exact": usage.get("prompt_tokens") == args.input_tokens,
        "output_usage_exact": tokens == args.output_tokens,
        "total_usage_exact": usage.get("total_tokens") == args.input_tokens + args.output_tokens,
        "expected_cache_exact": type(cached) is int and cached == case["expected_cached_tokens"],
        "visible_output_tokens_exact": visible_tokens == args.output_tokens,
        "finish_length": finish == "length",
        "strict_counting_prefix": generated == expected_output,
        "content_timestamps": first is not None and last is not None and last > first,
    }
    row.update(content=generated, usage=usage, cached_tokens=cached,
               expected_cached_tokens=case["expected_cached_tokens"],
               primer_lcp_tokens=case["primer_lcp_tokens"],
               visible_output_tokens=visible_tokens,
               finish_reason=finish, done_received=done,
               ttft_ms=(first - started) * 1000 if first is not None else None,
               tpot_ms=((last - first) * 1000 / (tokens - 1)
                        if first is not None and isinstance(tokens, int)
                        and tokens > 1 else None),
               e2e_ms=(ended - started) * 1000,
               first_content_ms=(first - started) * 1000 if first is not None else None,
               last_content_ms=(last - started) * 1000 if last is not None else None,
               ended_monotonic=ended, checks=checks, passed=all(checks.values()))
    (directory / (case["id"] + ".sse")).write_bytes(raw_bytes)
    _write_json(directory / (case["id"] + ".events.json"), events)
    _write_json(directory / (case["id"] + ".response.json"), row)
    return row


async def _run_stage(client: Any, args: argparse.Namespace,
                     stage: dict[str, Any], output: Path) -> dict[str, Any]:
    directory = output / stage["name"]
    directory.mkdir()
    cases = iter(stage["cases"])
    rows: list[dict[str, Any]] = []
    stop = asyncio.Event()
    started = time.perf_counter()

    async def worker() -> None:
        while not stop.is_set():
            case = next(cases, None)
            if case is None:
                return
            row = await _request(client, args.url, case, directory, args)
            rows.append(row)
            logger.info("%s %s passed=%s", stage["name"], case["id"], row["passed"])
            if not row["passed"]:
                stop.set()

    await asyncio.gather(*(worker() for _ in range(stage["concurrency"])))
    ended = time.perf_counter()
    wall = ended - started
    actual_input = sum(row["usage"].get("prompt_tokens", 0) for row in rows)
    actual_output = sum(row["usage"].get("completion_tokens", 0) for row in rows)
    summary: dict[str, Any] = {
        "name": stage["name"], "concurrency": stage["concurrency"],
        "planned": len(stage["cases"]), "attempted": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "passed": len(rows) == len(stage["cases"]) and all(row["passed"] for row in rows),
        "failed_ids": [row["id"] for row in rows if not row["passed"]],
        "started_monotonic": started, "ended_monotonic": ended,
        "wall_seconds": wall, "actual_input_tokens": actual_input,
        "actual_output_tokens": actual_output,
        "input_token_throughput": actual_input / wall,
        "output_token_throughput": actual_output / wall,
        "total_token_throughput": (actual_input + actual_output) / wall,
        "throughput_scope": "2P1D, two 8-device Prefills plus one 8-device Decode, actual usage / complete client wall",
    }
    summary["cache"] = _cache_summary(rows, actual_input)
    summary["uncached_input_token_throughput"] = (actual_input - summary["cache"]["total_cached_tokens"]) / wall
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[metric] for row in rows
                  if isinstance(row[metric], (int, float)) and math.isfinite(row[metric])]
        summary[metric] = {"mean": statistics.mean(values), "median": statistics.median(values),
                           "min": min(values), "max": max(values)} if values else None
    _write_json(directory / "summary.json", summary)
    return summary



async def _execute(args: argparse.Namespace, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import aiohttp

    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout), trust_env=False,
                                     connector=aiohttp.TCPConnector(limit=16)) as client:
        for stage in _prepared_cases(args.output_dir, stages):
            result = await _run_stage(client, args, stage, args.output_dir)
            results.append(result)
            if not result["passed"]:
                break
    return results


def _execute_prepared(cli: argparse.Namespace) -> dict[str, Any]:
    directory = cli.output_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((directory / "prepare_result.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(**manifest["arguments"])
    if (manifest["schema_version"] != 1 or manifest["script_sha256"] != _digest(Path(__file__))
            or manifest["stages"] != _stage_plan(args.max_concurrency)
            or prepared["manifest_sha256"] != _digest(directory / "manifest.json")
            or manifest["payloads_sha256"] != _digest(directory / "payloads.jsonl")
            or prepared["payloads_sha256"] != manifest["payloads_sha256"]):
        raise ValueError("Prepared script, manifest, stage plan, or payload bytes changed")
    if (args.input_tokens not in (20000, 40000) or args.output_tokens != OUTPUT_TOKENS
            or not 1 <= args.max_concurrency <= 16
            or args.batches != list(range(1, args.max_concurrency + 1)) or args.rounds != ROUNDS
            or args.requests_per_round != REQUESTS_PER_ROUND
            or args.template != "deepseek-v4-cpp" or Path(args.output_dir) != directory):
        raise ValueError("Prepared benchmark dimensions or directory changed")
    args.output_dir, args.tokenizer = directory, Path(args.tokenizer)
    for name, digest in manifest["tokenizer_sha256"].items():
        if _digest(args.tokenizer / name) != digest:
            raise ValueError("Prepared tokenizer bytes changed: " + name)
    args.runtime_tokenizer = _DeepseekTokenizer(args.tokenizer)
    output = _output_contract(args.runtime_tokenizer)
    if output != manifest["output_contract"]:
        raise ValueError("Prepared output contract changed")
    args.expected_output = output["expected_visible_text"]
    # The marker is exclusive, so concurrent invocations or retries cannot send
    # another copy of a prepared request set.
    _write_json(directory / "execution.json", {
        "started_unix": time.time(), "manifest_sha256": _digest(directory / "manifest.json"),
        "payloads_sha256": manifest["payloads_sha256"], "script_sha256": manifest["script_sha256"],
        "prepared_requests_rebuilt": False, "argv": sys.argv})
    results = asyncio.run(_execute(args, manifest["stages"]))
    passed = len(results) == len(manifest["stages"]) and all(row["passed"] for row in results)
    aggregate: dict[str, Any] = {"passed": passed, "stages": results, "formal": {},
                                "manifest_sha256": _digest(directory / "manifest.json"),
                                "payloads_sha256": manifest["payloads_sha256"]}
    for concurrency in args.batches:
        selected = [row for row in results if row["name"].startswith(f"b{concurrency}_")]
        cache_total = sum(row["cache"]["total_cached_tokens"] for row in selected)
        actual_input = sum(row["actual_input_tokens"] for row in selected)
        counts: dict[str, int] = {}
        for row in selected:
            for key, count in row["cache"]["counts"].items():
                counts[key] = counts.get(key, 0) + count
        aggregate["formal"][f"b{concurrency}"] = {
            "rounds_completed": len(selected),
            "valid": len(selected) == ROUNDS and all(row["passed"] for row in selected),
            "requests": sum(row["attempted"] for row in selected),
            "median_of_round_means": {
                metric: statistics.median(row[metric]["mean"] for row in selected)
                for metric in ("ttft_ms", "tpot_ms", "e2e_ms")
                if selected and all(row[metric] is not None for row in selected)},
            "median_output_token_throughput": statistics.median(
                row["output_token_throughput"] for row in selected) if selected else None,
            "cache": {"counts": counts, "total_cached_tokens": cache_total,
                      "fraction_of_input_tokens": cache_total / actual_input if actual_input else None}}
    _write_json(directory / "summary.json", aggregate)
    _write_json(directory / "execution_result.json", {"completed": True, "passed": passed,
                "ended_unix": time.time(), "stages_completed": len(results),
                "summary_sha256": _digest(directory / "summary.json")})
    return {"completed": True, "passed": passed, "stages_completed": len(results),
            "output_dir": str(directory), "formal": aggregate["formal"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--input-tokens", type=int, choices=(20000, 40000))
    parser.add_argument("--max-concurrency", type=int, choices=range(1, 17))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.execute_prepared:
        if (args.base_url or args.model or args.tokenizer
                or args.input_tokens is not None or args.max_concurrency is not None):
            parser.error("execute-prepared uses only the frozen URL/model/tokenizer; omit overrides")
        result = _execute_prepared(args)
    else:
        if not args.base_url or not args.model or args.tokenizer is None:
            parser.error("prepare-only requires --base-url, --model, and --tokenizer")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("timeout must be positive and finite")
        origin = args.base_url.rstrip("/")
        parts = urlsplit(origin)
        if (parts.scheme not in {"http", "https"} or not parts.netloc
                or parts.path not in {"", "/v1"} or parts.query or parts.fragment):
            parser.error("base-url must be an HTTP origin, optionally ending in /v1")
        args.tokenizer = args.tokenizer.resolve()
        if json.loads((args.tokenizer / "config.json").read_text()).get("model_type") != "deepseek_v4":
            raise ValueError("Expected DeepSeek V4 tokenizer and chat template")
        args.url = origin.removesuffix("/v1") + "/v1/chat/completions"
        args.input_tokens = args.input_tokens or INPUT_TOKENS
        args.max_concurrency = args.max_concurrency or 5
        args.template, args.output_tokens = "deepseek-v4-cpp", OUTPUT_TOKENS
        args.batches = list(range(1, args.max_concurrency + 1))
        args.rounds, args.requests_per_round = ROUNDS, REQUESTS_PER_ROUND
        result = _prepare(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("prepared") or result.get("passed") else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Prefix sweep failed; existing evidence is retained")
        sys.exit(1)
