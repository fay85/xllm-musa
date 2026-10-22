#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Replay frozen GSM8K payloads serially. Never load references or score answers."""

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


FROZEN_SHA = "3cf14a90c1f64a3a2e76cd920adc13624278d824dc23001b3cb18302ad4020c0"
CONTROL_URL = "http://10.121.38.92:29888/v1/chat/completions"
MODEL_ALIAS = "DeepSeek-V4-Flash-0731-FP8-mt"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def validate_payload(payload):
    expected = {"model", "messages", "temperature", "top_p", "max_tokens", "chat_template_kwargs"}
    if set(payload) != expected:
        raise ValueError("Unexpected frozen request fields")
    if (payload["model"] != "DeepSeek-V4-Flash-FP8" or
            payload["temperature"] != 0 or payload["top_p"] != 0.95 or
            payload["max_tokens"] != 1024 or
            payload["chat_template_kwargs"] != {"thinking": False}):
        raise ValueError("Frozen protocol differs from accepted baseline")
    messages = payload["messages"]
    if len(messages) != 1 or set(messages[0]) != {"role", "content"} or messages[0]["role"] != "user":
        raise ValueError("Expected a single unchanged user prompt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=[10, 20, 1319], required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Validate and save payloads only; zero HTTP requests")
    args = parser.parse_args()
    frozen = args.assets / "gsm8k_frozen" / "requests.json"
    source_bytes = frozen.read_bytes()
    if sha(source_bytes) != FROZEN_SHA:
        raise ValueError("Frozen requests hash mismatch")
    originals = json.loads(source_bytes)
    if len(originals) != 1319:
        raise ValueError("Expected all 1319 frozen requests")
    for payload in originals:
        validate_payload(payload)
    planned = []
    for index, source in enumerate(originals[:args.count]):
        effective = copy.deepcopy(source)
        effective["model"] = MODEL_ALIAS
        if {k: v for k, v in effective.items() if k != "model"} != {k: v for k, v in source.items() if k != "model"}:
            raise ValueError("Only the model alias may be changed")
        frozen_body = json.dumps(source, ensure_ascii=False).encode("utf-8")
        http_body = json.dumps(effective, ensure_ascii=False).encode("utf-8")
        planned.append((index, source, effective, frozen_body, http_body))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "PD GSM8K full evaluation" if args.count == 1319 else "PD GSM8K partial diagnostic",
        "frozen_sha256": FROZEN_SHA, "frozen_path": str(frozen),
        "runner_sha256": sha(Path(__file__).read_bytes()), "source_model": originals[0]["model"],
        "effective_model": MODEL_ALIAS, "only_payload_change": "model alias",
        "url": CONTROL_URL, "concurrency": 1, "retry_count": 0,
        "count": args.count, "partial_diagnostic": args.count != 1319,
        "dry_run": args.dry_run, "timeout_seconds": args.timeout,
        "references_loaded": False,
        "payloads": [{"index": i, "id": f"gsm8k-{i}", "frozen_body_sha256": sha(frozen_body),
                       "http_body_sha256": sha(http_body)}
                      for i, _, _, frozen_body, http_body in planned],
    }
    write_json(args.output_dir / "manifest.json", plan)
    for index, frozen_payload, effective_payload, _, _ in planned:
        write_json(args.output_dir / f"payload_{index}.json", frozen_payload)
        write_json(args.output_dir / f"http_payload_{index}.json", effective_payload)
    if args.dry_run:
        write_json(args.output_dir / "complete.json", {"http_requests_sent": 0, "answers_generated": 0, "dry_run": True})
        print(json.dumps({"output_dir": str(args.output_dir), "dry_run": True, "requests_sent": 0}))
        return
    outcomes = []
    with (args.output_dir / "transport.jsonl").open("x", encoding="utf-8", buffering=1) as transport:
        for index, frozen_payload, effective_payload, _, body in planned:
            request = urllib.request.Request(CONTROL_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
            start = time.perf_counter()
            status = None
            raw = b""
            error = None
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    status = response.status
                    raw = response.read()
                saved = json.loads(raw)
                if not isinstance(saved, dict) or (not saved.get("choices") and not saved.get("error")):
                    raise ValueError("Malformed HTTP completion response")
            except urllib.error.HTTPError as exc:
                status, raw = exc.code, exc.read()
                error = f"HTTPError: {exc}"
                saved = {"error": error, "http_status": status}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                saved = {"error": error, "http_status": status}
            elapsed = time.perf_counter() - start
            with (args.output_dir / f"response_{index}.http.body").open("xb") as output:
                output.write(raw)
            write_json(args.output_dir / f"response_{index}.json", saved)
            row = {"id": f"gsm8k-{index}", "index": index, "http_status": status,
                   "elapsed_seconds": elapsed, "error": error or saved.get("error"),
                   "raw_body_sha256": sha(raw)}
            outcomes.append(row)
            transport.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "complete.json", {
        "http_requests_sent": len(outcomes), "expected_count": args.count,
        "transport_errors": sum(bool(x["error"]) for x in outcomes),
        "all_planned_requests_recorded": len(outcomes) == args.count,
        "partial_diagnostic": args.count != 1319, "dry_run": False,
        "note": "Generation completion only; official scoring is a separate CPU step",
    })


if __name__ == "__main__":
    main()
