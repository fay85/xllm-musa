#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Plan and manage an isolated, single-instance, two-process MUSA decode."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterator
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.logger import logger

ROOT = Path(__file__).resolve().parent


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    return {"pid": pid, "state": fields[0], "pgid": int(fields[2]),
            "session": int(fields[3]), "start_ticks": fields[19],
            "exe": os.readlink(proc / "exe") if fields[0] != "Z" else "",
            "argv": (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
            "wait_status": int(fields[49]) if fields[0] == "Z" else None}


def _same_identity(current: dict[str, Any], recorded: dict[str, Any]) -> bool:
    keys = ("pid", "exe", "argv", "start_ticks", "pgid", "session")
    return current["state"] != "Z" and all(current[key] == recorded[key] for key in keys)


def _current(recorded: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _identity(recorded["pid"])
    except FileNotFoundError:
        return None


def _members(pgid: int) -> list[dict[str, Any]]:
    members = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            current = _identity(int(proc.name))
        except FileNotFoundError:
            continue
        if current["pgid"] == pgid and current["state"] != "Z":
            members.append(current)
    return members


def _assert_container(config: dict[str, Any]) -> None:
    actual = os.readlink("/proc/1/ns/pid")
    if actual != config["identity"]["pid_namespace"]:
        raise ValueError("PID namespace differs from the pinned decode_TP2_dev container")
    if socket.gethostname() != config["identity"]["hostname"]:
        raise ValueError("Unexpected hostname; inspect container identity before running")


def _validate(config: dict[str, Any]) -> None:
    if config["container"] != "decode_TP2_dev" or config["gpus"] != [6, 7]:
        raise ValueError("This harness is restricted to decode_TP2_dev and physical GPUs 6/7")
    if config["mode"] not in ("standalone", "pd"):
        raise ValueError("mode must be standalone or pd")
    if not isinstance(config["custom_all_reduce"], bool):
        raise ValueError("custom_all_reduce must be a JSON boolean")
    if type(config["mtp_k"]) is not int or type(config["concurrency"]) is not int:
        raise ValueError("MTP width and concurrency must be JSON integers")
    if config["mtp_k"] < 0 or config["concurrency"] < 1:
        raise ValueError("Invalid MTP width or concurrency")
    if not 0 < config["memory_utilization"] <= 1:
        raise ValueError("memory_utilization must be in (0, 1]")
    if len(config["ports"]["transfer"]) != 2 or len(config["ports"]["mooncake"]) != 2:
        raise ValueError("TP2 requires distinct transfer/RPC settings for two ranks")
    if config["mode"] == "pd":
        address = config.get("private_etcd", "")
        if not re.fullmatch(r"(?:127\.0\.0\.1|10\.121\.38\.92):[0-9]+", address):
            raise ValueError("PD requires an explicit private local etcd endpoint")
        port = int(address.rsplit(":", 1)[1])
        if port in (64379, 64380) or not 1 <= port <= 65535:
            raise ValueError("PD must not register with the active main stack's etcd")
    ports = _ports(config)
    if any(not 1 <= port <= 65535 for port in ports) or len(ports) != len(set(ports)):
        raise ValueError("Configured port windows overlap or are outside 1..65535")
    for minimum, maximum in config["ports"]["mooncake"]:
        if minimum > maximum or not (maximum < 32768 or minimum > 60999):
            raise ValueError("Mooncake RPC range overlaps its rejected ephemeral interval")
    binary = Path(config["paths"]["binary"])
    if not binary.is_relative_to(Path("/workspace/decode_tp2_dev_artifacts/bin")):
        raise ValueError("Refusing an executable outside the isolated development binary directory")


def _ports(config: dict[str, Any]) -> list[int]:
    ports = config["ports"]
    # Source currently consumes master+1/+2 and +5/+6 for world, TP and
    # single-rank stores. Reserve a wider window for encoder-DP variants.
    result = [ports["http"], ports["disagg"]]
    result.extend(range(ports["master"], ports["master"] + 17))
    result.extend(ports["transfer"])
    for minimum, maximum in ports["mooncake"]:
        result.extend(range(minimum, maximum + 1))
    return result


def _listeners() -> set[int]:
    ports = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A":
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports


def _check_ports(config: dict[str, Any]) -> None:
    requested = _ports(config)
    busy = sorted(set(requested) & _listeners())
    if busy:
        raise ValueError(f"Existing host listeners occupy requested ports: {busy}")
    for port in requested:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((config["host"], port))


def _gpu_snapshot(text: str) -> dict[str, Any]:
    devices: dict[int, dict[str, int]] = {}
    processes = []
    for line in text.splitlines():
        match = re.match(r"^(\d+)\s+.+\|[^|]+\|\s*(\d+)%\s+(\d+)MiB\(", line)
        if match:
            gpu, utilization, memory = map(int, match.groups())
            devices[gpu] = {"utilization": utilization, "used_mib": memory}
        match = re.match(r"^(\d+)\s+(\d+)\s+(.+?)\s+(<?\d+)MiB\s*$", line)
        if match:
            gpu_text, pid_text, command, memory_text = match.groups()
            processes.append({"gpu": int(gpu_text), "host_pid": int(pid_text),
                              "command": command, "used_mib_upper_bound": int(memory_text.lstrip("<"))})
    return {"devices": devices, "processes": processes, "raw": text}


def _check_gpus(snapshot: dict[str, Any], gpus: list[int]) -> None:
    for gpu in gpus:
        device = snapshot["devices"].get(gpu)
        if device is None:
            raise ValueError(f"GPU{gpu} cannot be identified in mthreads-gmi")
        busy_processes = [p for p in snapshot["processes"]
                          if p["gpu"] == gpu and p["used_mib_upper_bound"] > 1]
        if device["used_mib"] > 1 or device["utilization"] != 0 or busy_processes:
            raise ValueError(f"GPU{gpu} is occupied: {device}; processes={busy_processes}")


def _environment(config: dict[str, Any], rank: int) -> dict[str, str]:
    keep = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "MUSA_HOME", "MUDNN_HOME", "MCCL_HOME")
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    environment.update(config["environment"])
    environment.update({"MUSA_VISIBLE_DEVICES": "6,7", "CUDA_VISIBLE_DEVICES": "6,7",
                        "MTHREADS_VISIBLE_DEVICES": "6,7", "XLLM_BIN": config["paths"]["binary"],
                        "XLLM_MUSA_ENABLE_TP_CUSTOM_AR": "1" if config["custom_all_reduce"] else "0",
                        "MC_MIN_RPC_PORT": str(config["ports"]["mooncake"][rank][0]),
                        "MC_MAX_RPC_PORT": str(config["ports"]["mooncake"][rank][1])})
    environment["LD_PRELOAD"] = config["paths"]["preload"]
    environment["LD_LIBRARY_PATH"] = config["paths"]["mooncake_lib"] + ":" + config["environment"]["LD_LIBRARY_PATH"]
    return environment


def _command(config: dict[str, Any], rank: int) -> list[str]:
    paths, ports = config["paths"], config["ports"]
    flags: dict[str, Any] = {
        "model": paths["model"], "model_id": config["model_id"], "backend": "vlm",
        "host": config["host"], "port": ports["http"],
        "master_node_addr": f"{config['host']}:{ports['master']}",
        "nnodes": 2, "node_rank": rank, "dp_size": 1, "cp_size": 1,
        "block_size": 64, "max_memory_utilization": config["memory_utilization"],
        "max_concurrent_requests": config["concurrency"], "max_seqs_per_batch": config["concurrency"],
        "max_tokens_per_batch": 32768, "max_tokens_per_chunk_for_prefill": 2048,
        "max_tokens_for_graph_mode": 4096, "enable_prefill_piecewise_graph": False,
        "enable_graph_vmm_pool": False, "enable_graph": True,
        "enable_graph_mode_decode_no_padding": True, "enable_schedule_overlap": True,
        "enable_prefix_cache": True, "enable_chunked_prefill": True,
        "enable_mtp_draft_body_tp1": False, "enable_mm_encoder_dp": False,
        "random_seed": config["random_seed"], "transfer_listen_port": ports["transfer"][rank],
        "disagg_pd_port": ports["disagg"], "kv_split_size": 1,
        "enable_disagg_pd": config["mode"] == "pd",
        "instance_role": "DECODE" if config["mode"] == "pd" else "DEFAULT",
        "usercode_in_pthread": True,
    }
    if config["mtp_k"]:
        flags.update({"draft_model": paths["model"], "speculative_algorithm": "MTP",
                      "num_speculative_tokens": config["mtp_k"]})
        if config.get("draft_vocab"):
            flags["mtp_draft_vocab_path"] = config["draft_vocab"]
    if config["mode"] == "pd":
        flags.update({"etcd_addr": config["private_etcd"], "kv_cache_transfer_mode": "PULL",
                      "enable_native_service_adapter": True})
    return [paths["binary"], *[f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
                               for key, value in flags.items()]]


def _plan(config: dict[str, Any]) -> dict[str, Any]:
    _validate(config)
    return {"container": config["container"], "mode": config["mode"],
            "gpus": config["gpus"], "custom_all_reduce": config["custom_all_reduce"],
            "endpoint": f"http://{config['host']}:{config['ports']['http']}",
            "reserved_host_ports": _ports(config),
            "ranks": [{"rank": rank, "argv": _command(config, rank),
                       "environment": _environment(config, rank)} for rank in range(2)]}


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    plan = _plan(config)
    _assert_container(config)
    _check_ports(config)
    snapshot = _gpu_snapshot(subprocess.check_output(["mthreads-gmi"], text=True))
    _check_gpus(snapshot, config["gpus"])
    for key in ("binary", "preload"):
        if not Path(config["paths"][key]).is_file():
            raise ValueError(f"Missing {key}: {config['paths'][key]}")
    if not (Path(config["paths"]["model"]) / "config.json").is_file():
        raise ValueError("Model directory has no config.json")
    for key in ("FLASHINFER_OPS_PATH", "MATE_MUBIN_DIR", "MATE_HOME"):
        if not Path(config["environment"][key]).is_dir():
            raise ValueError(f"Missing runtime directory {key}")
    if not (Path(config["paths"]["mooncake_lib"]) / "libasio.so").is_file():
        raise ValueError("Isolated build libasio.so is absent")
    if config.get("draft_vocab"):
        if not Path(config["draft_vocab"]).is_file():
            raise ValueError("Configured draft vocabulary is absent")
        logger.warning("TP2 draft vocabulary currently truncates each rank to 128 rows; compare acceptance separately")
    plan["binary_sha256"] = _sha256(Path(config["paths"]["binary"]))
    plan["gpu_snapshot"] = snapshot
    return plan


@contextmanager
def _exclusive(runtime: Path) -> Iterator[None]:
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / "launcher.lock"
    with lock.open("x") as marker:
        marker.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def _signal_owned(record: dict[str, Any], sig: int) -> None:
    members = _members(record["pgid"])
    if not members:
        return
    current = _current(record)
    if current is None or not _same_identity(current, record):
        raise ValueError(f"Rank leader identity changed; refusing to signal group {record['pgid']}")
    if record["pid"] != record["pgid"] or record["pid"] != record["session"]:
        raise ValueError("Recorded process is not its own session/group leader")
    if any(member["session"] != record["session"] for member in members):
        raise ValueError("Unexpected process in owned group")
    os.killpg(record["pgid"], sig)


def _stop_state(state: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    records = list(state["processes"].values())
    for record in records:
        current = _current(record)
        if current is not None and current["state"] != "Z" and not _same_identity(current, record):
            raise ValueError("A recorded PID was reused; refusing all stop signals")
    for record in reversed(records):
        _signal_owned(record, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_members(r["pgid"]) for r in records):
        time.sleep(0.25)
    killed = []
    for record in reversed(records):
        if _members(record["pgid"]):
            _signal_owned(record, signal.SIGKILL)
            killed.append(record["pid"])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_members(r["pgid"]) for r in records):
        time.sleep(0.1)
    result = [{"rank": rank, "pid": record["pid"], "final": _current(record),
               "sigkill_sent": record["pid"] in killed, "remaining_members": _members(record["pgid"])}
              for rank, record in state["processes"].items()]
    if any(item["remaining_members"] for item in result):
        raise RuntimeError("Owned group did not exit; inspect stop evidence")
    return result


def _start(config: dict[str, Any], runtime: Path, timeout: float) -> dict[str, Any]:
    active = runtime / "active.json"
    with _exclusive(runtime):
        if active.exists():
            raise ValueError("active.json exists; inspect status and stop before another start")
        plan = _preflight(config)
        run = runtime / (time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        run.mkdir()
        state = {"run_dir": str(run), "created_epoch": time.time(), "plan": plan, "processes": {}}
        _save(run / "config.json", config)
        try:
            for rank in plan["ranks"]:
                with (run / f"rank{rank['rank']}.log").open("xb") as log:
                    child = subprocess.Popen(rank["argv"], env=rank["environment"], cwd=run,
                                             stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                state["processes"][str(rank["rank"])] = _identity(child.pid)
                _save(active, state)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                for record in state["processes"].values():
                    current = _current(record)
                    if current is None or current["state"] == "Z":
                        raise RuntimeError("A rank exited during startup; inspect rank logs")
                    if not _same_identity(current, record):
                        raise ValueError("Rank identity changed during startup")
                text = (run / "rank0.log").read_text(errors="replace")
                if "Application startup complete" in text:
                    state["ready_epoch"] = time.time()
                    _save(active, state)
                    _save(run / "state.json", state)
                    return state
                time.sleep(0.5)
            raise TimeoutError("Rank0 did not reach application startup completion")
        except (OSError, ValueError, RuntimeError, TimeoutError):
            state["startup_failed_epoch"] = time.time()
            try:
                state["stop_result"] = _stop_state(state, 10)
            finally:
                _save(run / "state.json", state)
                _save(active, state)
            raise


def _status(runtime: Path) -> dict[str, Any]:
    state = json.loads((runtime / "active.json").read_text())
    return {"run_dir": state["run_dir"], "stopped_epoch": state.get("stopped_epoch"),
            "ranks": {rank: {"recorded_pid": record["pid"], "current": _current(record),
                             "identity_matches": (current := _current(record)) is not None
                             and _same_identity(current, record)}
                      for rank, record in state["processes"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "start", "status", "stop"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = Path(config["runtime_directory"])
    if args.action == "plan":
        result = _plan(config)
    elif args.action == "preflight":
        result = _preflight(config)
    elif args.action == "start":
        result = _start(config, runtime, args.timeout)
    elif args.action == "status":
        result = _status(runtime)
    else:
        _assert_container(config)
        with _exclusive(runtime):
            active = runtime / "active.json"
            state = json.loads(active.read_text())
            state["stop_result"] = _stop_state(state, min(args.timeout, 30))
            state["stopped_epoch"] = time.time()
            _save(Path(state["run_dir"]) / "state.json", state)
            active.unlink()
            result = state
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.CalledProcessError):
        logger.exception("TP2 launcher failed")
        raise SystemExit(1)
