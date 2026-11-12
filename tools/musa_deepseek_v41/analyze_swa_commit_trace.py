# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Link SWA commit CPU scopes through actual runtime correlation to GPU events."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _analyze(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [
        x
        for x in events
        if x.get("name") == "V41/Attention/SWA_commit"
        and x.get("ph") == "X"
        and x.get("cat") in ("cpu_op", "user_annotation")
    ]
    if not scopes:
        raise ValueError("No actual SWA commit scope")
    by_thread: dict[int, list[dict]] = defaultdict(list)
    gpu: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category in ("cpu_op", "user_annotation", "privateuse1_runtime"):
            by_thread[event["tid"]].append(event)
        if "kernel" in category.lower() or "gpu_memcpy" in category.lower():
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                gpu[correlation].append(event)
    for items in by_thread.values():
        items.sort(key=lambda x: x["ts"])
    stamps = {tid: [x["ts"] for x in items] for tid, items in by_thread.items()}
    rows = []
    for scope in scopes:
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        thread = scope["tid"]
        children = [
            x
            for x in by_thread[thread][
                bisect_left(stamps[thread], begin) : bisect_right(stamps[thread], end)
            ]
            if x is not scope and x["ts"] + x.get("dur", 0) <= end + 0.001
        ]
        cpu = Counter(x["name"] for x in children if x.get("cat") == "cpu_op")
        runtime = [x for x in children if x.get("cat") == "privateuse1_runtime"]
        correlations = {x.get("args", {}).get("correlation") for x in runtime}
        associated = [
            event for correlation in correlations for event in gpu.get(correlation, [])
        ]
        kernels = [x for x in associated if "kernel" in x.get("cat", "").lower()]
        if not kernels:
            raise AssertionError(
                "No GPU kernel linked through actual runtime correlation"
            )
        rows.append(
            {
                "ts_us": begin,
                "cpu_scope_us": scope["dur"],
                "cpu_ops": dict(cpu),
                "runtime_calls": dict(Counter(x["name"] for x in runtime)),
                "gpu_kernels": dict(Counter(x["name"] for x in kernels)),
                "gpu_kernel_count": len(kernels),
                "gpu_kernel_us_sum": sum(x["dur"] for x in kernels),
                "linked_gpu_memcpy_count": len(associated) - len(kernels),
            }
        )
    return {
        "status": "PASS_ACTUAL_RUNTIME_CORRELATION_FOR_EVERY_SWA_COMMIT",
        "trace": str(path),
        "scope_count": len(rows),
        "median_cpu_scope_us": statistics.median(x["cpu_scope_us"] for x in rows),
        "cpu_scope_us_sum": sum(x["cpu_scope_us"] for x in rows),
        "gpu_kernel_count": sum(x["gpu_kernel_count"] for x in rows),
        "gpu_kernel_us_sum": sum(x["gpu_kernel_us_sum"] for x in rows),
        "cpu_op_counts": dict(sum((Counter(x["cpu_ops"]) for x in rows), Counter())),
        "runtime_call_counts": dict(
            sum((Counter(x["runtime_calls"]) for x in rows), Counter())
        ),
        "gpu_kernel_counts": dict(
            sum((Counter(x["gpu_kernels"]) for x in rows), Counter())
        ),
        "scopes": rows,
        "interpretation": "Observed profiled durations include instrumentation overhead; not a formal TPOT estimate. GPU links use runtime correlation, not External id directly.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output; retain prior analysis")
    result = _analyze(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info(
        "Verified %d SWA scopes and %d correlated GPU kernels",
        result["scope_count"],
        result["gpu_kernel_count"],
    )


if __name__ == "__main__":
    main()
