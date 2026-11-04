# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Run one GTest suite, retaining per-case XML for every CTest attempt."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.logger import logger


def _classify_result(return_code: int, xml_path: Path, suite: str) -> tuple[int, dict[str, int]]:
    counts = {"tests": 0, "failed": 0, "skipped": 0}
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        logger.exception("Missing or invalid GTest XML: %s", xml_path)
        return 1, counts
    cases = root.findall(".//testcase")
    counts["tests"] = len(cases)
    counts["failed"] = sum(bool(case.findall("failure") or case.findall("error")) for case in cases)
    counts["skipped"] = sum(case.find("skipped") is not None for case in cases)
    if (
        return_code != 0
        or not cases
        or counts["failed"]
        or int(root.get("failures", "0"))
        or int(root.get("errors", "0"))
    ):
        return 1, counts
    if root.tag != "testsuites" or int(root.get("tests", "-1")) != len(cases):
        return 1, counts
    if any(case.get("classname") != suite or case.get("status") != "run" for case in cases):
        return 1, counts
    return (77 if counts["skipped"] == counts["tests"] else 0), counts


def _run_suite(binary: str, suite: str, output_root: Path, extra_args: list[str]) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{suite}-", dir=output_root))
    xml_path = attempt / "results.xml"
    command = [binary, f"--gtest_filter={suite}.*", f"--gtest_output=xml:{xml_path}", *extra_args]
    logger.info("Grouped GTest results: %s", attempt)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)

    def _forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signum)
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, _forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return_code = process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    try:
        result, counts = _classify_result(return_code, xml_path, suite)
    except (ValueError, TypeError):
        logger.exception("Invalid GTest counts: %s", xml_path)
        result, counts = 1, {}
    summary = {
        "suite": suite,
        "binary_exit_code": return_code,
        "exit_code": result,
        "wall_seconds": time.monotonic() - started,
        "xml": str(xml_path),
        **counts,
    }
    (attempt / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Grouped GTest %s: %s", suite, summary)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return _run_suite(args.binary, args.suite, args.output_root, extra)


if __name__ == "__main__":
    raise SystemExit(_main())
