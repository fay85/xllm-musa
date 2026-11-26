# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one measured 8-rank role configuration in its runtime container."""

import argparse
import hashlib
from ipaddress import IPv4Address
import json
import os
from pathlib import Path
import re
import resource
import sys

HERE = Path(__file__).resolve().parent
ROLES = json.loads((HERE / "roles.json").read_text())


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if sys.version_info >= (3, 11) else _stream_hash(stream)


def _stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(data)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=ROLES["cases"])
    parser.add_argument("role", choices=("p", "d"))
    parser.add_argument("--run", type=Path, required=True, help="New shared run directory")
    parser.add_argument("--namespace", required=True, help="Unique etcd namespace")
    parser.add_argument("--p-ip", type=IPv4Address, required=True)
    parser.add_argument("--d-ip", type=IPv4Address, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", help="Required override for a locally rebuilt ELF")
    parser.add_argument("--driver", type=Path, required=True, help="Directory holding the 4.3.10 libmusa.so.1")
    parser.add_argument("--router-preload", type=Path,
                        help="Archived tokenizer pre-initializer .so, required on D")
    parser.add_argument("--ragged-provider", type=Path, help="G32RAGLIB1 root, required on M256 P")
    parser.add_argument("--mccl-ifname", help="Override recorded bond0 when required by the site")
    parser.add_argument("--rdma-devices", help="Override the role's recorded MC_TE_FILTERS")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.namespace):
        parser.error("Use a unique namespace containing letters, digits, dots, dashes or underscores")
    if args.p_ip == args.d_ip:
        parser.error("P and D must have distinct addresses")
    template = ROLES["templates"][ROLES["cases"][args.case][args.role]["template"]]
    if args.role == "d" and args.router_preload is None:
        parser.error("D requires --router-preload")
    if template["variant"] == "G32PM2561" and args.ragged_provider is None:
        parser.error("M256 P requires --ragged-provider")
    values = {
        "@P_IP@": str(args.p_ip), "@D_IP@": str(args.d_ip), "@RUN@": str(args.run.resolve()),
        "@MODEL@": str(args.model.absolute()), "@NAMESPACE@": args.namespace,
        "@DRIVER@": str(args.driver.absolute()), "@BINARY@": str(args.binary.absolute()),
        "@RAGGED_PROVIDER@": str(args.ragged_provider.absolute()) if args.ragged_provider else "",
    }

    def render(text: str) -> str:
        return re.sub(r"@[A-Z_]+@", lambda match: values[match[0]], text)

    config = {k: v for k, v in template.items()
              if k not in ("base_command", "base_environment", "rank_overrides")}
    config.update(binary=values["@BINARY@"], model=values["@MODEL@"], control_host=str(args.d_ip),
                  namespace=args.namespace, ranks=[], case=args.case, run_role=args.role)
    if args.binary_sha256:
        if not re.fullmatch(r"[a-f0-9]{64}", args.binary_sha256):
            parser.error("--binary-sha256 must be a lowercase SHA-256")
        config["binary_sha256"] = args.binary_sha256
    if args.router_preload:
        config["router_preload"] = str(args.router_preload.absolute())
    for row in template["rank_overrides"]:
        command = template["base_command"].copy()
        for index, value in row["command_overrides"].items():
            command[int(index)] = value
        env = {k: render(v) for k, v in (template["base_environment"] | row["environment_overrides"]).items()}
        if args.mccl_ifname:
            env["MCCL_SOCKET_IFNAME"] = args.mccl_ifname
        if args.rdma_devices:
            env["MC_TE_FILTERS"] = args.rdma_devices
        config["ranks"].append({"rank": row["rank"], "command": [render(v) for v in command],
                                "environment": env})
    if not args.prepare_only:
        assets = [
            args.model / "tokenizer.json", args.binary, args.driver / "libmusa.so.1",
            *(Path(path) for path in config["ranks"][0]["environment"]["LD_PRELOAD"].split(":")),
        ]
        if args.router_preload:
            assets.append(args.router_preload)
        if args.ragged_provider:
            assets += [args.ragged_provider / "provider-progress.json"]
            assets += [args.ragged_provider / "provider" / f"m{size}-hint{hint}" / file
                       for size in (128, 256) for hint in (0, 1)
                       for file in ("launcher.so", "kernel.mubin")]
        for asset in assets:
            if not asset.is_file():
                raise FileNotFoundError(asset)
        if _sha256(args.binary) != config["binary_sha256"]:
            raise ValueError("ELF hash differs from the archived or explicitly supplied hash")
        workload = json.loads((HERE / "workload.json").read_text())
        if _sha256(args.model / "tokenizer.json") != workload["tokenizer_sha256"]:
            raise ValueError("Tokenizer differs from the measured model view")
    directory = args.run.resolve() / {"p": "sj9", "d": "sj10"}[args.role]
    directory.mkdir(parents=True)
    (directory / "profile").mkdir()
    config_path = directory / "prepared_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.prepare_only:
        print(config_path)
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execv(sys.executable, [sys.executable, str(HERE / "pd_rank_supervisor.py"),
                             "--config", str(config_path), "--output", str(directory)])


if __name__ == "__main__":
    main()
