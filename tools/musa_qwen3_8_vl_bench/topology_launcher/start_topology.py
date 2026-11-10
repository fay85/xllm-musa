#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Explicit topology plans over the read-only delivered runtime bundle.

--validate-plan is a standard-library-only static check, without bundle loading,
port inspection, directory creation or process launch. --plan-only renders safe
effective argv after sourcing the bundle env, without launching anything.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
EXPECTED_ENGINE_MD5 = "3e464a5b77092efe063631d333ce6532"
DEFAULT_BUNDLE = "/workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110"
DEFAULT_WORK = "/workspace/topology_sweep_1110"
# Below the ephemeral range 32768-60999. Ports inside that range lose the bind
# when an outbound socket is assigned the same source port during startup.
PORT_BASES = {
    "p_http": 18000, "p_tcpstore": 18100, "p_pd": 18300,
    "p_transfer": 18400, "p_mooncake": 33000,
    "d_http": 18600, "d_tcpstore": 18700, "d_pd": 19100,
    "d_transfer": 19200, "d_mooncake": 44000,
    "master_http": 64888, "master_rpc": 64889,
    "coordinator_http": 64890, "coordinator_rpc": 64891,
    "etcd_client": 64379, "etcd_peer": 64380,
}
FATAL = re.compile(r"Check failed|terminate called|core dumped|out of memory|^F\d{8}", re.I | re.M)


def _int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_plan(raw, allow_pd_cp2=False, allow_split_host=False):
    name = raw.get("topology_name") or raw.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("topology_name must be a safe nonempty label")
    groups = []
    used_gpus = set()
    names = {"master", "coordinator", "etcd"}
    role_indices = collections.Counter()
    source_groups = raw.get("groups")
    if not isinstance(source_groups, list) or not source_groups:
        raise ValueError("groups must be a nonempty list")
    gpu_count = _int(raw.get("gpu_count", 8), "gpu_count")
    for source in source_groups:
        role = source.get("role", "").upper()
        if role not in {"DEFAULT", "PREFILL", "DECODE"}:
            raise ValueError(f"Unsupported role {role!r}")
        role_index = role_indices["DECODE" if role == "DECODE" else "P"]
        role_indices["DECODE" if role == "DECODE" else "P"] += 1
        group_name = source.get("name", ("decode" if role == "DECODE" else "aggregate" if role == "DEFAULT" else "prefill") + str(role_index))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", group_name) or group_name in names:
            raise ValueError(f"Invalid or duplicate group name {group_name!r}")
        names.add(group_name)
        gpus = source.get("visible_gpus", source.get("physical_gpu_ids"))
        if isinstance(gpus, str):
            gpus = [int(x) for x in gpus.split(",")]
        if not isinstance(gpus, list) or not gpus:
            raise ValueError(f"{group_name}: visible_gpus must be an integer list or comma list")
        gpus = [_int(x, group_name + ".visible_gpus", 0) for x in gpus]
        if len(set(gpus)) != len(gpus) or any(x >= gpu_count for x in gpus):
            raise ValueError(f"{group_name}: duplicate or out-of-range physical GPU")
        if used_gpus.intersection(gpus) and not allow_split_host:
            raise ValueError(f"{group_name}: GPU overlaps another group")
        if not allow_split_host:
            used_gpus.update(gpus)
        world = _int(source.get("world_size", source.get("nnodes", len(gpus))), group_name + ".world_size")
        dp = _int(source.get("dp_size", 1), group_name + ".dp_size")
        cp = _int(source.get("cp_size", 1), group_name + ".cp_size")
        if world != len(gpus) or world % (dp * cp):
            raise ValueError(f"{group_name}: world_size must equal visible GPU count and be divisible by dp*cp")
        tp = world // (dp * cp)
        if role == "DEFAULT" and cp != 1:
            raise ValueError("GATED: VLM aggregated CP2/CP×TP is forbidden; do not launch")
        if role == "DECODE" and cp != 1:
            raise ValueError("Decode cp_size must remain 1")
        if role != "DEFAULT" and dp != 1 and not allow_split_host:
            raise ValueError("Internal DP for PD roles is not enabled by this launcher")
        if role == "PREFILL" and cp != 1 and not (allow_pd_cp2 and cp == 2 and tp == 1 and world == dp * cp):
            raise ValueError("GATED: P CP2 is reserved; root must explicitly pass --allow-pd-cp2 (world2/cp2/tp1 only)")
        groups.append({"name": group_name, "role": role, "visible_gpus": gpus,
                       "physical_gpu_ids": gpus, "world_size": world, "nnodes": world,
                       "dp_size": dp, "cp_size": cp, "tp_size_derived": tp, "role_index": role_index})
    roles = {g["role"] for g in groups}
    mode = "aggregate" if roles == {"DEFAULT"} else "pd"
    if mode == "pd" and ("DEFAULT" in roles or not {"PREFILL", "DECODE"}.issubset(roles)):
        raise ValueError("PD mode needs P and D groups; DEFAULT must not mix with PD roles")
    if raw.get("mode", mode) != mode:
        raise ValueError("mode disagrees with group roles")
    bases = dict(PORT_BASES)
    for key, value in raw.get("port_bases", {}).items():
        if key not in bases:
            raise ValueError(f"Unknown port base {key}")
        bases[key] = _int(value, "port_bases." + key)
    plan = {"schema_version": 1, "topology_name": name, "mode": mode,
            "concurrency": _int(raw.get("concurrency", 16), "concurrency"),
            "benchmark_concurrency": raw.get("benchmark_concurrency", 8), "gpu_count": gpu_count,
            "groups": groups, "port_bases": bases,
            "gate": {"aggregate_cp_allowed": False, "pd_cp2_explicitly_allowed": allow_pd_cp2}}
    plan["port_reservations"], plan["ranks"] = allocate_ports(plan)
    return plan


def allocate_ports(plan):
    reservations = {}
    ranks = []
    bases = plan["port_bases"]
    def reserve(port, owner):
        if not 1 <= port <= 65535:
            raise ValueError(f"Port outside1..65535: {port} ({owner})")
        if port in reservations:
            raise ValueError(f"Internal port collision {port}: {reservations[port]} versus {owner}")
        reservations[port] = owner
    for key in ("master_http", "master_rpc", "coordinator_http", "coordinator_rpc", "etcd_client", "etcd_peer"):
        reserve(bases[key], key)
    for group in plan["groups"]:
        family = "d" if group["role"] == "DECODE" else "p"
        index = group["role_index"]
        ports = {"http": bases[family + "_http"] + index,
                 "tcpstore": bases[family + "_tcpstore"] + 100 * index,
                 "pd": bases[family + "_pd"] + index}
        reserve(ports["http"], group["name"] + ":http")
        reserve(ports["pd"], group["name"] + ":pd")
        for port in range(ports["tcpstore"], ports["tcpstore"] + 33):
            reserve(port, group["name"] + ":tcpstore-reserved")
        group["ports"] = ports
        for rank in range(group["world_size"]):
            transfer = bases[family + "_transfer"] + 100 * index + 10 * rank
            mooncake = bases[family + "_mooncake"] + 1000 * index + 20 * rank
            reserve(transfer, group["name"] + f":rank{rank}:transfer")
            for port in range(mooncake, mooncake + 17):
                reserve(port, group["name"] + f":rank{rank}:mooncake")
            ranks.append({"name": group["name"] + f"_r{rank}", "group": group["name"],
                          "role": group["role"], "node_rank": rank, "world_size": group["world_size"],
                          "cp_size": group["cp_size"], "dp_size": group["dp_size"],
                          "tp_size_derived": group["tp_size_derived"], "visible_gpus": group["visible_gpus"],
                          "logname": group["name"] + f"_r{rank}.log",
                          "ports": {**ports, "transfer": transfer, "mooncake_min": mooncake, "mooncake_max": mooncake + 16}})
    return [{"port": p, "owner": reservations[p]} for p in sorted(reservations)], ranks


def load_bundle_env(bundle, model, host):
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    if model:
        env["MODEL_DIR"] = str(model)
    if host:
        env["HOST_ADDR"] = host
    process = subprocess.run(["bash", "-c", 'set -e; set -a; source "$1/env.sh"; env -0', "topology-env", str(bundle)],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError("Bundle env failed: " + process.stderr.decode(errors="replace"))
    values = {}
    for item in process.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            values[key.decode()] = value.decode()
    for key in ("XLLM_BIN", "MASTER_BIN", "ETCD_BIN", "MODEL", "HOST", "MUSA_OPS_DIR", "MASTER_TOKENIZER"):
        if not values.get(key):
            raise RuntimeError("Bundle env omitted " + key)
    values["MODEL"] = str(Path(values["MODEL"]).resolve())
    return values


def relocated_env(env, work, name):
    env = dict(env)
    cache = work / "cache/topology_launcher/mate027"
    source = Path(env["MUSA_OPS_DIR"])
    env["MATE_WORKSPACE_BASE"] = str(cache)
    env["MUSA_OPS_DIR"] = str(cache / "mp31" / source.name)
    env["FLASHINFER_OPS_PATH"] = env["MUSA_OPS_DIR"]
    env["RUNS_DIR"] = str(work / "runs" / name)
    env["RESULTS_DIR"] = str(work / "results" / name)
    for key, relative in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache/xdg",
                          "HF_HOME": "cache/hf", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions"}.items():
        env[key] = str(work / relative)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "16"
    env.pop("LD_PRELOAD", None)
    return env, source


def engine_command(rank, plan, env, etcd_host="127.0.0.1"):
    p = rank["ports"]
    role = rank["role"]
    decode = role == "DECODE"
    argv = [env["XLLM_BIN"], f'--model={env["MODEL"]}', f'--model_id={env["MODEL_ID"]}',
            f'--host={env["HOST"]}', "--backend=vlm", "--random_seed=-1", "--max_body_size=268435456",
            "--block_size=64", f'--max_concurrent_requests={plan["concurrency"]}', "--max_seqs_per_batch=16",
            "--max_encoder_cache_size=1024", "--tool_call_parser=qwen3_coder", "--enable_graph_vmm_pool=true",
            f'--enable_disagg_pd={str(plan["mode"] == "pd").lower()}', "--enable_native_service_adapter=true",
            "--kv_cache_transfer_mode=PULL", "--kv_cache_dtype=auto", "--kv_split_size=1",
            f'--etcd_addr={etcd_host}:{plan["port_bases"]["etcd_client"]}',
            "--vmodule=disagg_pd_scheduler=1", "--usercode_in_pthread=true",
            f'--draft_model={env["MODEL"]}', "--speculative_algorithm=MTP", "--num_speculative_tokens=3",
            f'--mtp_draft_vocab_path={env["DRAFT_VOCAB"]}', f'--instance_role={role}',
            f'--nnodes={rank["world_size"]}', f'--node_rank={rank["node_rank"]}',
            f'--cp_size={rank["cp_size"]}', f'--dp_size={rank["dp_size"]}',
            f'--port={p["http"]}', f'--master_node_addr={env["HOST"]}:{p["tcpstore"]}',
            f'--transfer_listen_port={p["transfer"]}', f'--disagg_pd_port={p["pd"]}',
            f'--max_memory_utilization={"0.85" if decode else "0.9"}',
            f'--max_linear_state_cache_slots={12 if decode else 32}',
            f'--max_tokens_per_batch={32768 if decode else 10240}',
            f'--max_tokens_per_chunk_for_prefill={2048 if decode else 10240}',
            "--max_tokens_for_graph_mode=4096", f'--enable_prefill_piecewise_graph={str(not decode).lower()}']
    if not decode:
        argv += ["--prefill_batch_wait_ms=0", "--enable_in_batch_prefix_cache=false", "--musa_prefill_graph_token_stride=64"]
    if role == "DEFAULT":
        argv += ["--enable_service_routing=true"]
    gpu_string = ",".join(map(str, rank["visible_gpus"]))
    overrides = {"MUSA_VISIBLE_DEVICES": gpu_string, "CUDA_VISIBLE_DEVICES": gpu_string,
                 "MTHREADS_VISIBLE_DEVICES": gpu_string, "LD_PRELOAD": env["MUSA_PRELOAD"],
                 "FLASHINFER_OPS_PATH": env["MUSA_OPS_DIR"], "MC_MIN_RPC_PORT": str(p["mooncake_min"]),
                 "MC_MAX_RPC_PORT": str(p["mooncake_max"]),
                 "XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS": "180"}
    return {**rank, "argv": argv, "env_overrides": overrides}


def control_commands(plan, env, run, etcd_host=""):
    p = plan["port_bases"]
    listen_host = "0.0.0.0" if etcd_host else "127.0.0.1"
    advertise_host = etcd_host or "127.0.0.1"
    etcd = [env["ETCD_BIN"], "--name=topology-bench", f'--data-dir={run}/etcd_data',
            f'--listen-client-urls=http://{listen_host}:{p["etcd_client"]}', f'--advertise-client-urls=http://{advertise_host}:{p["etcd_client"]}',
            f'--listen-peer-urls=http://127.0.0.1:{p["etcd_peer"]}', f'--initial-advertise-peer-urls=http://127.0.0.1:{p["etcd_peer"]}',
            f'--initial-cluster=topology-bench=http://127.0.0.1:{p["etcd_peer"]}',
            "--initial-cluster-token=topology-" + Path(run).name, "--initial-cluster-state=new"]
    result = [{"name": "etcd", "argv": etcd, "env_overrides": {}}]
    for name, role in (("master", "ROUTER"), ("coordinator", "COORDINATOR")):
        http = p["master_http" if name == "master" else "coordinator_http"]
        rpc = p["master_rpc" if name == "master" else "coordinator_rpc"]
        argv = [env["MASTER_BIN"], "--server_host=0.0.0.0", f'--etcd_addr=127.0.0.1:{p["etcd_client"]}',
                f'--http_server_port={http}', f'--rpc_server_port={rpc}', f'--tokenizer_path={env["MASTER_TOKENIZER"]}',
                "--block_size=64", f'--num_threads={4 if name == "master" else 2}', "--logtostderr=true",
                "--batch_credit_max_grant_units=8192", "--batch_credit_safety_permyriad=10000",
                "--batch_credit_max_grant_units_per_router_tenant=8192", "--internal_api_token=lab-local-token",
                f'--service_deployment_role={role}', "--native_pd_routing_policy=STATIC_BUCKET",
                "--load_balance_policy=CAR", "--kv_route_mode=DISABLED", "--tool_call_parser=qwen3_coder",
                "--first_output_watchdog_timeout_ms=60000", "--max_nonstream_retry_wasted_device_ms=5000",
                "--flow_max_queue_wait_ms=15000", "--default_request_deadline_ms=300000", "--max_request_deadline_ms=300000"]
        if name == "master":
            argv += ["--http_server_num_threads=4", "--rpc_server_num_threads=4"]
        if env.get("SERVICE_VMODULE"):
            argv += ["--vmodule=" + env["SERVICE_VMODULE"]]
        result.append({"name": name, "argv": argv,
                       "env_overrides": {"LD_LIBRARY_PATH": str(run) + "/master_lib:" + env["MASTER_LIBRARY_PATH"]}})
    return result


def safe_commands(commands):
    return [{**r, "argv": ["--internal_api_token=<local-lab-default>" if a.startswith("--internal_api_token=") else a for a in r["argv"]]} for r in commands]


def hashes(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def check_assets(bundle, env, expected_md5, input_counts):
    identities = {}
    for key in ("XLLM_BIN", "MASTER_BIN", "ETCD_BIN", "RUST_TOKENIZERS_LIB", "MUSA_PRELOAD", "DRAFT_VOCAB"):
        path = Path(env[key])
        if not path.is_file():
            raise RuntimeError(f"Missing runtime asset {key}: {path}")
        if key in {"XLLM_BIN", "MASTER_BIN", "ETCD_BIN"} and not os.access(path, os.X_OK):
            raise RuntimeError("Non-executable binary: " + str(path))
        identities[key] = hashes(path)
    if identities["XLLM_BIN"]["md5"] != expected_md5:
        raise RuntimeError("Engine MD5 differs from the explicitly selected new binary")
    for key in ("MODEL", "MATE_HOME", "MATE_MUBIN_DIR", "MASTER_TOKENIZER"):
        if not Path(env[key]).is_dir():
            raise RuntimeError(f"Missing runtime directory {key}: {env[key]}")
    args = [sys.executable, str(bundle / "scripts/prepare_data.py"), "--model-dir", env["MODEL"], "--check-only", "--require-ready"]
    if input_counts:
        args.append("--verify-token-counts")
    process = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode:
        raise RuntimeError("Input preflight failed: " + (process.stdout + process.stderr)[-6000:])
    return identities, json.loads(process.stdout)


def check_ports(plan):
    process = subprocess.run(["ss", "-H", "-ltn"], check=True, text=True, stdout=subprocess.PIPE)
    occupied = set()
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            match = re.search(r":(\d+)$", fields[3])
            if match:
                occupied.add(int(match[1]))
    conflicts = [r for r in plan["port_reservations"] if r["port"] in occupied]
    if conflicts:
        raise RuntimeError("Occupied reserved ports; no process started: " + json.dumps(conflicts))


def prepare_cache(source, dest, engine_md5):
    if not source.is_dir():
        raise RuntimeError("Missing read-only source cache: " + str(source))
    marker = dest / ".topology_cache_source.json"
    identity = {"source": str(source.resolve()), "engine_md5": engine_md5}
    if dest.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise RuntimeError("Existing experiment cache lacks matching ownership marker; left untouched: " + str(dest))
        return "reused_owned_experiment_cache"
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=dest.name + ".copy.", dir=dest.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=False)
        (staging / marker.name).write_text(json.dumps(identity, indent=2) + "\n")
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return "copied_source_cache_into_experiment"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def spawn(run, command, env):
    args = [sys.executable, "-S", str(HERE / "scripts/spawn_owned.py"), "--run", str(run), "--name", command["name"]]
    for key, value in command["env_overrides"].items():
        args += ["--env", key + "=" + value]
    process = subprocess.run(args + ["--"] + command["argv"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode:
        raise RuntimeError(f"Spawn {command['name']} failed: " + (process.stdout + process.stderr)[-6000:])
    record = json.loads(process.stdout)
    print(f"STARTED {command['name']} pid={record['pid']}", flush=True)


def current_identity(record):
    try:
        proc = Path("/proc") / str(record["pid"])
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and os.readlink(proc / "exe") == record["exe"] and stat[19] == record["starttime"]
    except FileNotFoundError:
        return False


def inspect_started(run):
    records = json.loads((run / "identities.json").read_text())
    for record in records:
        if not current_identity(record):
            raise RuntimeError("Owned process exited or changed identity: " + record["name"])
        log = run / (record["name"] + ".log")
        text = log.read_text(errors="replace") if log.exists() else ""
        if FATAL.search(text):
            raise RuntimeError("Fatal startup log: " + str(log))


def wait_engines(run, names, deadline):
    pending = set(names)
    last_progress = 0
    while pending and time.monotonic() < deadline:
        inspect_started(run)
        pending = {name for name in pending if "Application startup complete" not in (run / (name + ".log")).read_text(errors="replace")}
        if pending and time.monotonic() - last_progress > 30:
            print("WAITING " + ",".join(sorted(pending)), flush=True)
            last_progress = time.monotonic()
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError("Bounded engine startup timeout: " + ",".join(sorted(pending)))


def wait_http(url, deadline, run):
    while time.monotonic() < deadline:
        inspect_started(run)
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError("Bounded health wait timed out: " + url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path(DEFAULT_BUNDLE))
    parser.add_argument("--work-root", type=Path, default=Path(DEFAULT_WORK))
    parser.add_argument("--run-dir", type=Path, help="Root-owned unique absent/empty run directory under work-root; overrides automatic naming/current")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--etcd-advertise", default="", help="Reachable etcd address. Empty keeps 127.0.0.1.")
    parser.add_argument("--only", default="all", help="all, or a comma list of control,prefill,decode")
    parser.add_argument("--expected-engine-md5", default=EXPECTED_ENGINE_MD5)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--allow-pd-cp2", action="store_true")
    parser.add_argument("--allow-split-host", action="store_true")
    parser.add_argument("--validate-plan", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-cpu-token-counts", action="store_true", help="Explicit diagnostic override; normal preflight checks88 counts")
    args = parser.parse_args()
    raw = json.loads(args.plan.read_text())
    plan = normalize_plan(raw, args.allow_pd_cp2, args.allow_split_host)
    if args.startup_timeout <= 0:
        raise ValueError("startup-timeout must be positive")
    if args.validate_plan:
        print(json.dumps(plan, indent=2))
        return 0
    bundle = args.bundle_dir.resolve()
    work = args.work_root.resolve()
    if work.is_relative_to(bundle):
        raise ValueError("Experiment work-root must be outside the read-only delivery bundle")
    requested_run = args.run_dir or (Path(raw["run_dir"]) if raw.get("run_dir") else None)
    if requested_run is not None:
        requested_run = requested_run.resolve()
        if not requested_run.is_relative_to(work) or requested_run == work:
            raise ValueError("Explicit run-dir must be a child of experiment work-root")
        if requested_run.is_relative_to(bundle):
            raise ValueError("Explicit run-dir must not write inside the read-only bundle")
        if requested_run.exists() and (not requested_run.is_dir() or any(requested_run.iterdir())):
            raise ValueError("Explicit run-dir must be absent or empty; refusing reuse")
    source_env = load_bundle_env(bundle, args.model_dir, args.host)
    env, source_cache = relocated_env(source_env, work, plan["topology_name"])
    dry_run = str(requested_run or (work / "runs" / plan["topology_name"] / "<RUN>"))
    commands = control_commands(plan, env, dry_run) + [engine_command(rank, plan, env) for rank in plan["ranks"]]
    if args.plan_only:
        print(json.dumps({"effective_plan": plan, "commands": safe_commands(commands), "cache_source": str(source_cache), "cache_destination": env["MUSA_OPS_DIR"]}, indent=2))
        return 0
    binaries, inputs = check_assets(bundle, env, args.expected_engine_md5, not args.skip_cpu_token_counts)
    if not source_cache.is_dir():
        raise RuntimeError("Missing source cache: " + str(source_cache))
    check_ports(plan)
    if args.check_only:
        print(json.dumps({"status": "preflight_passed_no_start", "effective_plan": plan, "bin_hashes": binaries,
                          "cpu_inputs_verified": inputs["cpu_token_counts"]["verified"], "cache_copy_pending": not Path(env["MUSA_OPS_DIR"]).exists()}, indent=2))
        return 0
    cache_action = prepare_cache(source_cache, Path(env["MUSA_OPS_DIR"]), binaries["XLLM_BIN"]["md5"])
    for key in ("RUNS_DIR", "RESULTS_DIR", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_EXTENSIONS_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if requested_run is None:
        run = Path(tempfile.mkdtemp(prefix=dt.datetime.now().strftime("%Y%m%d_%H%M%S."), dir=env["RUNS_DIR"]))
    else:
        requested_run.mkdir(parents=True, exist_ok=True)
        run = requested_run
    (run / "master_lib").mkdir()
    (run / "master_lib/librust_tokenizers.so").symlink_to(Path(env["RUST_TOKENIZERS_LIB"]).resolve())
    current = Path(env["RUNS_DIR"]) / "current"
    if current.exists() and not current.is_symlink():
        raise RuntimeError("Refusing to replace a real current directory: " + str(current))
    temporary_current = current.with_name("current." + str(os.getpid()) + ".tmp")
    temporary_current.symlink_to(run)
    temporary_current.replace(current)
    write_json(run / "plan.json", raw)
    write_json(run / "effective_plan.json", plan)
    write_json(run / "bin_hashes.json", binaries)
    write_json(run / "inputs_preflight.json", inputs)
    write_json(run / "cache_preparation.json", {"action": cache_action, "source": str(source_cache), "destination": env["MUSA_OPS_DIR"]})
    etcd_host = args.etcd_advertise or "127.0.0.1"
    only = {part.strip().lower() for part in args.only.split(",") if part.strip()}
    if not only or "all" in only:
        only = {"control", "prefill", "decode"}
    controls = control_commands(plan, env, str(run), args.etcd_advertise)
    engines = [engine_command(rank, plan, env, etcd_host) for rank in plan["ranks"]]
    engines = [item for item in engines if item["role"].lower() in only]
    if "control" not in only:
        controls = []
    write_json(run / "effective_commands.json", safe_commands(controls + engines))
    allowed = [k for k in env if k.startswith(("XLLM_", "MATE_")) or k in {"MODEL", "MODEL_ID", "MASTER_TOKENIZER", "MUSA_OPS_DIR", "FLASHINFER_OPS_PATH", "MUSA_BACKEND_INIT_SO", "LD_LIBRARY_PATH", "MASTER_LIBRARY_PATH", "OMP_NUM_THREADS", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_EXTENSIONS_DIR", "RUNS_DIR", "RESULTS_DIR"}]
    write_json(run / "effective_environment.json", {k: env[k] for k in sorted(allowed)})
    print("RUN " + str(run), flush=True)
    deadline = time.monotonic() + args.startup_timeout
    try:
        if controls:
            spawn(run, controls[0], env)
            wait_http(f'http://127.0.0.1:{plan["port_bases"]["etcd_client"]}/health', min(deadline, time.monotonic() + 30), run)
            spawn(run, controls[1], env)
            time.sleep(2)
            spawn(run, controls[2], env)
            time.sleep(3)
        phases = []
        if "decode" in only:
            phases.append([r for r in engines if r["role"] == "DECODE"])
        if "prefill" in only:
            phases.append([r for r in engines if r["role"] != "DECODE"])
        for phase in phases:
            for command in phase:
                spawn(run, command, env)
            if phase:
                wait_engines(run, [r["name"] for r in phase if r["node_rank"] == 0], deadline)
        if controls:
            wait_http(f'http://127.0.0.1:{plan["port_bases"]["master_http"]}/readyz', deadline, run)
        inspect_started(run)
        write_json(run / "READY.json", {"status": "ready_not_inference_qualified", "run": str(run), "topology_name": plan["topology_name"],
                                        "master_url": f'http://127.0.0.1:{plan["port_bases"]["master_http"]}', "results_dir": env["RESULTS_DIR"]})
        print("READY " + str(run), flush=True)
    except BaseException as exc:
        write_json(run / "STARTUP_FAILURE.json", {"error": repr(exc), "cleanup": "only recorded identities"})
        if (run / "identities.json").exists():
            cleanup = subprocess.run([sys.executable, "-S", str(HERE / "scripts/stop_owned.py"), "--run", str(run), "--timeout", "15", "--force"], env=env)
            write_json(run / "STARTUP_CLEANUP.json", {"returncode": cleanup.returncode})
        raise
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    raise SystemExit(main())
