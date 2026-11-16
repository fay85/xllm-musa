# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine real Kineto intervals and symbolized native entry counts in Perfetto.

Offline only. Requires perfetto==0.59.1. StackSample weights are COUNT, never
invented CPU-time samples or estimated durations for individual native frames.
The original Kineto JSON and native backtrace JSONL remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_records(
    path: Path,
    include_aten: bool = False,
    record_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    stacks: dict[int, list[dict[str, Any]]] = {}
    records = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "stack":
                stacks[row["stack_id"]] = row["frames"]
            elif row["kind"] == "record" and row["name"].startswith(
                ("V41/", "aten::") if include_aten else ("V41/",)
            ):
                if record_ids is not None and row["record_id"] not in record_ids:
                    continue
                frames = [
                    frame
                    for frame in stacks[row["stack_id"]]
                    if frame["module"].endswith("/bin/xllm")
                ]
                if not frames:
                    raise ValueError("Semantic entry has no actual xllm native frame")
                records.append({**row, "native_frames": frames})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--native-stacks", type=Path, required=True)
    parser.add_argument("--resolved-symbols", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--step", choices=("first", "last"), required=True)
    parser.add_argument(
        "--step-prefix", default="V41/step/",
        help="Actual semantic root prefix; use a DSpark layer for draft analysis",
    )
    parser.add_argument(
        "--scope", help="Optional actual CPU semantic scope within the chosen step"
    )
    parser.add_argument("--scope-index", type=int, default=0)
    parser.add_argument(
        "--record-id", type=int,
        help="Select an actual RecordFunction ID and its correlated GPU kernels; "
        "use the native sidecar to establish the caller before selecting it",
    )
    parser.add_argument(
        "--include-aten",
        action="store_true",
        help="Include actual correlated ATen entry backtraces as COUNT; "
        "the extra leaf is the real RecordFunction name, not a native PC symbol",
    )
    parser.add_argument(
        "--include-runtime",
        action="store_true",
        help="Show actual runtime/driver API intervals on a separate track; "
        "GraphLaunch and stream waits are CPU spans, not graph GPU nodes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output; do not overwrite trace evidence")
    if len(args.binary_sha256) != 64:
        raise ValueError("Supply the SHA256 of the binary used for symbolization")
    events = json.loads(args.trace.read_text())["traceEvents"]
    steps = [
        row
        for row in events
        if row.get("ph") == "X"
        and row.get("cat") in ("cpu_op", "user_annotation")
        and row.get("name", "").startswith(args.step_prefix)
    ]
    # The replay-based speculative path calls target.forward recursively to
    # rebuild an accepted prefix. Select the outer request step, not its last
    # nested replay: otherwise the comparison silently omits current verify.
    outer_steps = []
    boundaries: dict[tuple[int, int], float] = {}
    for row in sorted(steps, key=lambda item: (item["ts"], -item["dur"])):
        key = row["pid"], row["tid"]
        end = row["ts"] + row["dur"]
        if end <= boundaries.get(key, float("-inf")) + 0.001:
            continue
        outer_steps.append(row)
        boundaries[key] = end
    if not outer_steps:
        raise ValueError(f"No outer semantic scope with prefix {args.step_prefix!r}")
    step = outer_steps[0 if args.step == "first" else -1]
    begin, end = step["ts"], step["ts"] + step["dur"]
    selected_scope = None
    if args.record_id is not None:
        if args.scope:
            raise ValueError("Use either --record-id or --scope")
        matches = [
            row for row in events
            if row.get("ph") == "X"
            and row.get("cat") in ("cpu_op", "user_annotation")
            and row.get("args", {}).get("Record function id") == args.record_id
        ]
        if len(matches) != 1:
            raise ValueError("RecordFunction ID must select exactly one actual event")
        step = selected_scope = matches[0]
        begin, end = step["ts"], step["ts"] + step["dur"]
    if args.scope:
        candidates = [
            row
            for row in events
            if row.get("ph") == "X"
            and row.get("cat") in ("cpu_op", "user_annotation")
            and row.get("name") == args.scope
            and row.get("tid") == step["tid"]
            and begin <= row["ts"]
            and row["ts"] + row["dur"] <= end + 0.001
        ]
        selected_scope = candidates[args.scope_index]
        begin, end = selected_scope["ts"], selected_scope["ts"] + selected_scope["dur"]
    selected = [
        row
        for row in events
        if row.get("ph") == "X"
        and row.get("dur", 0) > 0
        and begin <= row["ts"]
        and row["ts"] + row["dur"] <= end + 0.001
        and (
            row.get("cat") in ("cpu_op", "user_annotation")
            or "kernel" in row.get("cat", "").lower()
            or (args.include_runtime and row.get("cat", "").endswith(
                ("_runtime", "_driver")
            ))
        )
    ]
    if selected_scope is not None:
        # A native async launch may complete after its CPU scope has returned.
        # Link through the actual runtime correlation instead of inventing a
        # nearby GPU interval or truncating it to the CPU duration.
        correlations = {
            row["args"]["correlation"]
            for row in events
            if row.get("ph") == "X"
            and (
                row.get("cat", "").endswith("_runtime")
                or row.get("cat", "").endswith("_driver")
            )
            and row.get("tid") == selected_scope["tid"]
            and begin <= row["ts"]
            and row["ts"] + row.get("dur", 0) <= end + 0.001
            and row.get("args", {}).get("correlation") is not None
        }
        gpu = [
            row
            for row in events
            if row.get("ph") == "X"
            and "kernel" in row.get("cat", "").lower()
            and row.get("args", {}).get("correlation") in correlations
        ]
        if not gpu:
            raise ValueError("No GPU kernel linked to the actual CPU semantic scope")
        selected = [
            row for row in selected
            if row.get("cat") in ("cpu_op", "user_annotation")
            or (args.include_runtime and row.get("cat", "").endswith(
                ("_runtime", "_driver")
            ))
        ]
        selected += gpu
    symbols = json.loads(args.resolved_symbols.read_text())
    trace = proto.Trace()
    defaults = trace.packet.add()
    defaults.trusted_packet_sequence_id = 2
    defaults.sequence_flags = 1
    defaults.trace_packet_defaults.stack_sample_defaults.source = (
        "xllm.native_cpp.RecordFunction_entry_backtrace.COUNT"
    )
    def _track_key(row: dict) -> tuple[int, int, str]:
        category = row.get("cat", "")
        kind = "gpu" if "kernel" in category.lower() else (
            "runtime" if category.endswith(("_runtime", "_driver")) else "cpu"
        )
        return row["pid"], row["tid"], kind

    track_ids: dict[tuple[int, int, str], int] = {}
    for row in selected:
        key = _track_key(row)
        if key in track_ids:
            continue
        track_ids[key] = len(track_ids) + 1
        packet = trace.packet.add()
        packet.trusted_packet_sequence_id = 1
        packet.track_descriptor.uuid = track_ids[key]
        prefix = (
            "GPU kernel stream "
            if "kernel" in row.get("cat", "").lower()
            else (
                "CPU runtime / driver thread " if key[2] == "runtime"
                else "Native C++ / ATen thread "
            )
        )
        packet.track_descriptor.name = prefix + str(row["tid"])
    packets = []
    for row in selected:
        uuid = track_ids[_track_key(row)]
        for stamp, kind in (
            (round(row["ts"] * 1000), 1),
            (round((row["ts"] + row["dur"]) * 1000), 2),
        ):
            packet = proto.TracePacket(timestamp=stamp, trusted_packet_sequence_id=1)
            event = packet.track_event
            event.type, event.track_uuid = kind, uuid
            if kind == 1:
                event.name = row["name"]
                event.categories.append(row.get("cat", ""))
                for name, value in row.get("args", {}).items():
                    annotation = event.debug_annotations.add()
                    annotation.name = str(name)
                    annotation.string_value = json.dumps(value, ensure_ascii=False)
            packets.append(
                (
                    stamp,
                    0 if kind == 2 else 1,
                    -row["dur"] if kind == 1 else row["dur"],
                    packet,
                )
            )
    by_id = {
        row["args"]["Record function id"]: row
        for row in selected
        if "Record function id" in row.get("args", {})
    }
    count = 0
    for record in _native_records(
        args.native_stacks, args.include_aten, set(by_id)
    ):
        event = by_id.get(record["record_id"])
        if event is None:
            continue
        if event["name"] != record["name"] or event["tid"] != record["os_thread"]:
            raise ValueError("Record ID matched a different operator or OS thread")
        packet = proto.TracePacket(
            timestamp=round(event["ts"] * 1000), trusted_packet_sequence_id=2
        )
        sample = packet.stack_sample
        sample.task_context.pid = step["pid"]
        sample.task_context.tid = record["os_thread"]
        sample.execution_context.mode = 1
        sample.primary_descriptor.name = (
            "Real native semantic + ATen entries (COUNT, not CPU time)"
            if args.include_aten
            else "Real native semantic-scope entries (COUNT, not CPU time)"
        )
        sample.primary_descriptor.unit = 9
        sample.primary_descriptor.description = "Actual libc backtrace at RecordFunction entry; one count per semantic scope"
        sample.primary_weight = 1
        for frame in reversed(record["native_frames"]):
            symbol, location = symbols[str(frame["module_offset"])]
            target = sample.callstack.frames.add()
            target.function_name = symbol
            if location != "??:?":
                target.source_file = location
        if record["name"].startswith("aten::"):
            # This is the actual matched RecordFunction operation label,
            # explicitly distinct from the symbolized native PC frames above.
            sample.callstack.frames.add().function_name = record["name"]
        packets.append((packet.timestamp, 2, 0, packet))
        count += 1
    if not count:
        raise ValueError("No actual native entry correlated with the selected trace")
    for _, _, _, packet in sorted(packets, key=lambda item: item[:3]):
        trace.packet.add().CopyFrom(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(trace.SerializeToString())
    verification = {}
    if args.verify:
        from perfetto.trace_processor import TraceProcessor

        with TraceProcessor(trace=str(args.output)) as processor:
            slices = list(processor.query("SELECT COUNT(*) AS n FROM slice"))[0].n
            samples = list(processor.query("SELECT COUNT(*) AS n FROM stack_sample"))[
                0
            ].n
            errors = list(
                processor.query(
                    "SELECT name,value FROM stats WHERE severity='error' AND value>0"
                )
            )
        if slices != len(selected) or samples != count or errors:
            raise ValueError(f"Trace Processor mismatch: {slices}/{samples}/{errors}")
        verification = {"slices": slices, "native_entry_samples": samples, "errors": 0}
    receipt = {
        "status": "CONVERTED_AND_VERIFIED" if args.verify else "CONVERTED_NOT_VERIFIED",
        "trace_sha256": _sha(args.trace),
        "native_stacks_sha256": _sha(args.native_stacks),
        "symbols_sha256": _sha(args.resolved_symbols),
        "symbolization_binary_sha256": args.binary_sha256,
        "selected_step": step,
        "step_selection": (
            "actual RecordFunction ID; GPU kernels linked through runtime correlation"
            if args.record_id is not None
            else "outermost matching semantic scope; nested calls retained"
        ),
        "selected_record_id": args.record_id,
        "step_prefix": args.step_prefix,
        "selected_scope": selected_scope,
        "gpu_selection": (
            "runtime correlation for scope"
            if selected_scope
            else "within full step window"
        ),
        "actual_intervals": len(selected),
        "native_entry_samples": count,
        "include_aten_entries": args.include_aten,
        "include_runtime_intervals": args.include_runtime,
        "operation_leaf_provenance": (
            "Actual same-ID RecordFunction names, not invented native symbols"
            if args.include_aten
            else None
        ),
        "output_sha256": _sha(args.output),
        "verification": verification,
        "scope": "Actual CPU/GPU intervals plus real native semantic entry COUNT; not CPU-time sampling",
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)
    )
    logger.info(
        "Converted %s actual intervals and %s native entries", len(selected), count
    )


if __name__ == "__main__":
    main()
