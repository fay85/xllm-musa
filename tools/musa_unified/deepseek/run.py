# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Local-host, identity-tracked launch of the frozen DeepSeek 2P1D+PCP recipe."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import resource
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.logger import logger
from tools.musa_unified.artifacts import check_packages, digest
from tools.musa_unified.processes import matches, record, stop


def validate(config: dict) -> None:
    if config.get("schema") != 1 or len(config.get("prefill", [])) != 2:
        raise ValueError("This recipe requires schema1 and exactly two Prefill instances")
    all_hosts = [config["control"], config["decode"], *config["prefill"]]
    for item in all_hosts:
        host = item["host"]
        if not host or host.startswith("REPLACE") or host in {"0.0.0.0", "::"}:
            raise ValueError("Set explicit reachable host IPs; control may share P or D host")
        socket.getaddrinfo(host, None)
        for key, value in item.items():
            if key.endswith("port") and (type(value) is not int or not 1024 <= value < 65000):
                raise ValueError(f"Invalid port {key}: {value}")
    if len({x["host"] for x in [config["decode"], *config["prefill"]]}) != 3:
        raise ValueError("P0, P1 and D each require a separate idle 8-GPU host")


def runtime_environment(pack: Path, role: str) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "USER",
        "LOGNAME",
        "MTHREADS_DRIVER_CAPABILITIES",
        "MUSA_DEVICE_ORDER",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    python_lib = "/usr/local/lib/python3.10/dist-packages"
    role_lib = pack / (f"service/{role}" if role == "deepseek" else role)
    libraries = [
        str(role_lib),
        "/workspace/mudnn_3.4.0/lib",
        "/workspace/MCCL_2.3.0/mccl/lib",
        f"{python_lib}/tvm_ffi/lib",
        f"{python_lib}/tilelang/lib",
        f"{python_lib}/z3/lib",
        f"{python_lib}/torch_musa/lib",
        f"{python_lib}/torch/lib",
        "/usr/local/musa/lib",
        "/usr/local/openmpi/lib",
        "/opt/intel/oneapi/mkl/lib/intel64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env.update(
        LD_LIBRARY_PATH=":".join(libraries),
        PYTHONPATH="",
        OMP_NUM_THREADS="16",
        MKL_NUM_THREADS="16",
    )
    if role == "deepseek":
        return env
    env.update(
        MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        MTHREADS_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        CUDA_VISIBLE_DEVICES="" if role == "decode" else "0,1,2,3,4,5,6,7",
        MUSA_HOME="/usr/local/musa",
        TORCH_MUSA_ARCH_LIST="31",
        MUSA_MODULE_LOADING="EAGER",
        FLASHINFER_OPS_PATH=str(pack / "cached_ops"),
        MATE_MUBIN_DIR=str(pack / "mate_mubin"),
        MATE_MUSA_ARCH_LIST="3.1",
        MATE_HOME="/opt/mate_0.2.7",
        XLLM_USE_FA3="1",
        XLLM_USE_FA3_DECODE="1",
        XLLM_MUSA_POOL_COMPUTE_STREAM="1",
        XLLM_PIECEWISE_CAPTURE_FA3="1",
        XLLM_PACKED_PREFILL_PIECEWISE="0",
        XLLM_SPEC_VERIFY_LARGE_BATCH_EAGER="0",
        MUSA_BACKEND_INIT_SO=str(pack / "bootstrap/libmusa_backend_init.so"),
        XLLM_MUSA_PYTHON_LIB=f"{python_lib}/torch_musa/lib/libmusa_python.so",
        XLLM_TILELANG_LIB=f"{python_lib}/tilelang/lib/libtilelang.so",
        LD_PRELOAD=f"{python_lib}/torch_musa/lib/libmusa_python.so:" + str(pack / "bootstrap/libmusa_backend_init.so"),
        TORCH_MCCL_ASYNC_ERROR_HANDLING="0",
        TORCH_MCCL_ENABLE_MONITORING="0",
        XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS="120",
    )
    return env


def plan(config: dict, role: str, run: Path) -> dict:
    validate(config)
    pack = Path(config["pack_root"])
    control = config["control"]
    etcd = f"{control['host']}:{control['etcd_port']}"
    processes = []
    if role == "control":
        env = runtime_environment(pack, "deepseek")
        client_url = "http://" + etcd
        peer_url = f"http://{control['host']}:{control['etcd_peer_port']}"
        etcd_args = [
            str(pack / "etcd/etcd"),
            "--name=xllm-bench",
            f"--data-dir={run / 'etcd_data'}",
            f"--listen-client-urls={client_url}",
            f"--advertise-client-urls={client_url}",
            f"--listen-peer-urls={peer_url}",
            f"--initial-advertise-peer-urls={peer_url}",
            f"--initial-cluster=xllm-bench={peer_url}",
            f"--initial-cluster-token={run.name}",
            "--initial-cluster-state=new",
        ]
        argv = [
            str(pack / "service/deepseek/xllm_master_serving"),
            f"--etcd_addr={etcd}",
            f"--http_server_port={control['http_port']}",
            f"--rpc_server_port={control['rpc_port']}",
            f"--tokenizer_path={config['model_path']}",
            "--logtostderr=true",
        ]
        processes = [
            dict(name="etcd", argv=etcd_args, environment=env),
            dict(name="control", argv=argv, environment=env),
        ]
        ports = [control[key] for key in ["http_port", "rpc_port", "etcd_port", "etcd_peer_port"]]
        host, health_port = control["host"], control["http_port"]
    else:
        kind = "decode" if role == "decode" else "prefill"
        node = config["decode"] if kind == "decode" else config["prefill"][int(role[-1])]
        profile = json.loads(Path(__file__).with_name("profile.json").read_text())["roles"][kind]
        env = runtime_environment(pack, kind)
        env["MATE_WORKSPACE_BASE"] = str(run / "mate_workspace")
        env.update(profile["environment"])
        ports = [node["pd_port"], *range(node["master_port"], node["master_port"] + 9)]
        for rank in range(8):
            flags = dict(
                profile["gflags"],
                model=config["model_path"],
                model_id=config["model_id"],
                draft_model=config["model_path"],
                host=node["host"],
                port=node["port"] + rank,
                master_node_addr=f"{node['host']}:{node['master_port']}",
                node_rank=rank,
                etcd_addr=etcd,
                transfer_listen_port=node["transfer_port"] + rank,
                disagg_pd_port=node["pd_port"],
                instance_role=kind.upper(),
            )
            rank_env = dict(
                env,
                MC_MIN_RPC_PORT=str(node["mooncake_port"] + rank),
                MC_MAX_RPC_PORT=str(node["mooncake_port"] + rank),
            )
            argv = [
                str(pack / kind / "xllm"),
                *[f"--{key}={value}" for key, value in flags.items()],
            ]
            processes.append(dict(name=f"{role}_rank{rank}", argv=argv, environment=rank_env))
            ports.extend(
                [
                    node["port"] + rank,
                    node["transfer_port"] + rank,
                    node["mooncake_port"] + rank,
                ]
            )
        host, health_port = node["host"], node["port"]
    return dict(
        role=role,
        host=host,
        ports=ports,
        health_url=f"http://{host}:{health_port}/v1/models",
        control_rpc_port=control["rpc_port"] if role == "control" else None,
        processes=processes,
        topology="2P1D+PCP: each P CP2xTP4, D TP8",
    )


def _preflight(config: dict, spec: dict) -> None:
    check_packages()
    with socket.socket() as probe:
        probe.bind((spec["host"], 0))
    for port in spec["ports"]:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))
    pack = Path(config["pack_root"])
    required = [Path(config["model_path"]) / "tokenizer.json"]
    required.extend(Path(row["argv"][0]) for row in spec["processes"])
    if spec["role"] == "control":
        required.append(pack / "service/deepseek/librust_tokenizers.so")
    else:
        kind = "decode" if spec["role"] == "decode" else "prefill"
        required.extend(
            [
                pack / kind / "libasio.so",
                pack / "cached_ops",
                pack / "mate_mubin",
                pack / "bootstrap/libmusa_backend_init.so",
            ]
        )
        text = subprocess.check_output(["mthreads-gmi"], text=True)
        if "No running processes found" not in text.split("Processes:", 1)[1]:
            raise RuntimeError("GPUs have existing processes; no automatic kill/reset is allowed")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)


def _wait_url(url: str, rows: list[dict], timeout: float) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not all(matches(row) for row in rows):
            raise RuntimeError("A launched process exited; inspect logs and stop remaining own processes")
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Not ready: {url}")


def _wait_rpc(host: str, port: int, rows: list[dict], timeout: float) -> None:
    """Legacy Control precedes engines; model HTTP availability is not its gate."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not all(matches(row) for row in rows):
            raise RuntimeError("Control/etcd exited during startup")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Control RPC not listening: {host}:{port}")


def start(config: dict, spec: dict, run: Path, timeout: float) -> None:
    _preflight(config, spec)
    run.mkdir(parents=True, exist_ok=False)
    current = Path(config["run_root"]) / spec["role"] / "current"
    current.parent.mkdir(parents=True, exist_ok=True)
    if current.is_symlink():
        current.unlink()
    elif current.exists():
        raise ValueError("current must be a symlink")
    current.symlink_to(run)
    if spec["role"] != "control":
        workspace = run / "mate_workspace"
        cache = workspace / ".cache/mate/0.2.7/mp31"
        cache.mkdir(parents=True)
        (cache / "cached_ops").symlink_to(Path(config["pack_root"]) / "cached_ops")
        (workspace / "mubin").symlink_to(Path(config["pack_root"]) / "mate_mubin")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    (run / "plan.json").write_text(json.dumps(spec, indent=2))
    (run / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for process in spec["processes"]:
        binary = Path(process["argv"][0])
        process["sha256"] = digest(binary)
        with (run / (process["name"] + ".log")).open("xb") as log:
            child = subprocess.Popen(
                process["argv"],
                env=process["environment"],
                cwd=run,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        rows.append(record(run, process["name"], child.pid, binary))
        if process["name"] == "etcd":
            _wait_url(
                f"http://{config['control']['host']}:{config['control']['etcd_port']}/health",
                rows,
                30,
            )
    (run / "plan_with_hashes.json").write_text(json.dumps(spec, indent=2))
    if spec["role"] == "control":
        _wait_rpc(spec["host"], spec["control_rpc_port"], rows, timeout)
    else:
        _wait_url(spec["health_url"], rows, timeout)
    (run / "ready.json").write_text(
        json.dumps(
            {
                "role_ready": True,
                "readiness_kind": "rpc" if spec["role"] == "control" else "http",
                "inference_verified": False,
            }
        )
    )
    logger.info(
        "Role ready: %s; run benchmark correctness gate before acceptance",
        spec["role"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("--role", choices=["control", "decode", "prefill0", "prefill1"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    role_root = Path(config["run_root"]) / args.role
    if args.action == "stop":
        stop(args.run_dir or (role_root / "current").resolve(strict=True))
        return
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run = (args.run_dir or role_root / tag).resolve()
    spec = plan(config, args.role, run)
    if args.plan:
        print(json.dumps(spec, indent=2))
        return
    if run.exists():
        raise FileExistsError(run)
    current = role_root / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError("current must be a symlink")
    start(config, spec, run, args.timeout)


if __name__ == "__main__":
    main()
