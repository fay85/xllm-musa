# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture one native 1P1D request after the completed official baseline."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import requests


def _control(url: str, action: str) -> dict:
    started = time.monotonic()
    response = requests.post(url.rstrip("/") + "/" + action, timeout=(10, 300))
    record = {
        "url": url,
        "action": action,
        "status": response.status_code,
        "body": response.text,
        "elapsed_s": time.monotonic() - started,
    }
    return record


def _controls(urls: list[str], action: str) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(lambda url: _control(url, action), urls))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefill-control", required=True)
    parser.add_argument("--decode-control", required=True)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_identity = None
    if not args.diagnostic:
        if (
            args.baseline_commit is None
            or re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit) is None
        ):
            raise ValueError("Require the already recorded baseline commit")
        if args.performance is None:
            raise ValueError("Require the completed official performance directory")
        finished = json.loads(
            (args.performance / "client_runs_finished.json").read_text()
        )
        if finished["status"] != "PASS_ALL_FOUR_PHASES_EXACT_NATIVE_TOKEN_COUNTS":
            raise ValueError("Do not profile during incomplete formal performance")
        baseline_identity = json.loads((args.performance / "identity.json").read_text())
    if list(args.profile_root.rglob("*.pt.trace.json")) or list(
        args.profile_root.rglob("native_cpp_stacks_*.jsonl")
    ):
        raise ValueError("Preserve prior traces; use a new service/profile directory")
    args.output.mkdir(parents=True, exist_ok=False)
    body = json.loads(args.dataset.read_text().splitlines()[0])
    body["max_tokens"] = 8
    if body["temperature"] != 0 or not body["ignore_eos"]:
        raise ValueError("Unexpected fixed profiling protocol")
    (args.output / "request.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2)
    )
    launch = {
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "baseline_commit": args.baseline_commit,
        "baseline_identity": baseline_identity,
        "diagnostic": args.diagnostic,
        "purpose": "256-input/8-output diagnostic only; overhead present, excluded from formal performance",
    }
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2))
    controls = [args.prefill_control, args.decode_control]
    started = []
    request_error = None
    content, usage, finish = "", None, None
    try:
        started = _controls(controls, "start_profile")
        (args.output / "start_profile.json").write_text(json.dumps(started, indent=2))
        if not all(row["status"] == 200 for row in started):
            raise RuntimeError("Native P/D profile start failed")
        begin = time.monotonic()
        with requests.post(
            args.url.rstrip("/") + "/v1/chat/completions",
            json=body,
            stream=True,
            timeout=(10, 900),
        ) as response:
            response.raise_for_status()
            with (args.output / "sse.jsonl").open("x") as stream:
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    text = raw.decode()
                    stream.write(
                        json.dumps(
                            {"elapsed_s": time.monotonic() - begin, "raw": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    if text == "data: [DONE]":
                        break
                    if not text.startswith("data: "):
                        continue
                    event = json.loads(text[6:])
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        content += choice.get("delta", {}).get("content") or ""
                        finish = choice.get("finish_reason") or finish
    except (requests.RequestException, RuntimeError, ValueError) as error:
        request_error = repr(error)
        logger.exception(
            "Native profiling request failed; stop requested and raw files retained"
        )
    finally:
        stopped = _controls(controls, "stop_profile")
        (args.output / "stop_profile.json").write_text(json.dumps(stopped, indent=2))
    archive = []
    for role in ("sj9", "sj10"):
        source = args.profile_root / role
        traces = list(source.glob("*.pt.trace.json"))
        stacks = list(source.glob("native_cpp_stacks_*.jsonl"))
        if len(traces) != 8 or len(stacks) != 8:
            request_error = (
                f"Incomplete {role} profile: {len(traces)} traces, {len(stacks)} stacks"
            )
        destination = args.output / "raw" / role
        destination.mkdir(parents=True, exist_ok=False)
        for path in traces + stacks:
            target = destination / path.name
            shutil.move(path, target)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            archive.append(
                {
                    "role": role,
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    (args.output / "archive_manifest.json").write_text(json.dumps(archive, indent=2))
    okay = (
        request_error is None
        and all(row["status"] == 200 for row in stopped)
        and usage is not None
        and usage.get("prompt_tokens") == 256
        and usage.get("completion_tokens") == 8
    )
    result = {
        "status": (
            "PROFILE_REQUEST_SAVED_REQUIRES_ALL_RANK_TRACE_VALIDATION"
            if okay
            else "FAIL"
        ),
        "content": content,
        "finish_reason": finish,
        "usage": usage,
        "error": request_error,
        "scope": "validate actual16rank CPU/GPU traces and native PC correlation separately",
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    logger.info("Native profile request: %s", result)
    if not okay:
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
