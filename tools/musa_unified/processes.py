# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Own only recorded processes: boot ID, PID start ticks, executable and argv hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def identity(pid: int) -> dict | None:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (proc / "cmdline").read_bytes()
        return dict(
            pid=pid,
            start_ticks=int(stat[19]),
            state=stat[0],
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            executable=os.readlink(proc / "exe"),
            argv_sha256=hashlib.sha256(argv).hexdigest(),
            argv_bytes=len(argv),
            argv0=argv.split(b"\0", 1)[0].decode() if argv else "",
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(row: dict) -> bool:
    current = identity(row["pid"])
    return bool(
        current
        and current["state"] != "Z"
        and all(current[key] == row[key] for key in ["start_ticks", "boot_id", "executable", "argv_sha256"])
    )


def record(run: Path, name: str, pid: int, executable: Path) -> dict:
    run = run.resolve(strict=True)
    expected = str(executable.resolve(strict=True))
    deadline = time.monotonic() + 10
    previous = None
    while time.monotonic() < deadline:
        row = identity(pid)
        if (
            row
            and row["executable"] == expected
            and row["argv_bytes"] > 0
            and str(Path(row["argv0"]).resolve()) == expected
        ):
            stable = (row["boot_id"], row["start_ticks"], row["executable"], row["argv_sha256"])
            if stable == previous:
                break
            previous = stable
        else:
            previous = None
        time.sleep(0.02)
    else:
        raise RuntimeError(f"PID {pid} did not exec expected binary {expected}")
    stdout = Path(os.readlink(Path("/proc") / str(pid) / "fd/1"))
    if not stdout.is_relative_to(run):
        raise RuntimeError(f"Process stdout does not belong to this run: {stdout}")
    row["name"] = name
    with (run / (name + ".identity.json")).open("x") as stream:
        json.dump(row, stream, indent=2)
    return row


def stop(run: Path) -> None:
    run = run.resolve(strict=True)
    rows = [json.loads(path.read_text()) for path in sorted(run.glob("*.identity.json"))]
    if not rows:
        raise RuntimeError("No identity records; refusing to stop from bare/stale PID numbers")
    for row in rows:
        current = identity(row["pid"])
        if (
            current
            and current["state"] != "Z"
            and current["boot_id"] == row["boot_id"]
            and current["start_ticks"] == row["start_ticks"]
            and not matches(row)
        ):
            raise RuntimeError(f"Same process has changed identity: PID {row['pid']}; inspect before stopping")
    for group in [
        ("prefill", "decode"),
        ("master", "coordinator", "control"),
        ("etcd",),
    ]:
        selected = [row for row in rows if row["name"].startswith(group)]
        for row in selected:
            if matches(row):
                os.kill(row["pid"], signal.SIGTERM)
                logger.info("TERM %s PID %s", row["name"], row["pid"])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(matches(row) for row in selected):
            time.sleep(0.2)
        for row in selected:
            if matches(row):
                os.kill(row["pid"], signal.SIGKILL)
                logger.warning("KILL own unresponsive process %s", row["pid"])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(matches(row) for row in rows):
        time.sleep(0.1)
    remaining = [row for row in rows if matches(row)]
    (run / "stop_result.json").write_text(json.dumps({"remaining": remaining}, indent=2))
    if remaining:
        raise RuntimeError("Own processes remain; inspect stop_result.json without GPU reset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "stop"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--expected-exe", type=Path)
    args = parser.parse_args()
    if args.action == "record":
        if not args.name or args.pid is None or args.expected_exe is None:
            parser.error("record requires --name, --pid, --expected-exe")
        if not args.name.replace("_", "").isalnum():
            parser.error("invalid process name")
        record(args.run, args.name, args.pid, args.expected_exe)
    else:
        stop(args.run)


if __name__ == "__main__":
    main()
