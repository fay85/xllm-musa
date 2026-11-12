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


def _native_records(path: Path) -> list[dict[str, Any]]:
    stacks: dict[int, list[dict[str, Any]]] = {}
    records = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "stack":
                stacks[row["stack_id"]] = row["frames"]
            elif row["kind"] == "record" and row["name"].startswith("V41/"):
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output; do not overwrite trace evidence")
    if len(args.binary_sha256) != 64:
        raise ValueError("Supply the SHA256 of the binary used for symbolization")
    events = json.loads(args.trace.read_text())["traceEvents"]
    steps = [row for row in events if row.get("name", "").startswith("V41/step/")]
    step = steps[0 if args.step == "first" else -1]
    begin, end = step["ts"], step["ts"] + step["dur"]
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
        )
    ]
    symbols = json.loads(args.resolved_symbols.read_text())
    trace = proto.Trace()
    defaults = trace.packet.add()
    defaults.trusted_packet_sequence_id = 2
    defaults.sequence_flags = 1
    defaults.trace_packet_defaults.stack_sample_defaults.source = (
        "xllm.native_cpp.RecordFunction_entry_backtrace.COUNT"
    )
    track_ids: dict[tuple[int, int], int] = {}
    for row in selected:
        key = (row["pid"], row["tid"])
        if key in track_ids:
            continue
        track_ids[key] = len(track_ids) + 1
        packet = trace.packet.add()
        packet.trusted_packet_sequence_id = 1
        packet.track_descriptor.uuid = track_ids[key]
        prefix = (
            "GPU kernel stream "
            if "kernel" in row.get("cat", "").lower()
            else "Native C++ / ATen thread "
        )
        packet.track_descriptor.name = prefix + str(row["tid"])
    packets = []
    for row in selected:
        uuid = track_ids[(row["pid"], row["tid"])]
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
    for record in _native_records(args.native_stacks):
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
            "Real native semantic-scope entries (COUNT, not CPU time)"
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
        "actual_intervals": len(selected),
        "native_entry_samples": count,
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
