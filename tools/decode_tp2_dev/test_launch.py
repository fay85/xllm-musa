#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU checks for TP2 topology and no-interference admission boundaries."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import unittest
from unittest.mock import patch

import launch


class LauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((Path(__file__).parent / "config.json").read_text())

    def test_one_logical_instance_two_distinct_ranks(self) -> None:
        plan = launch._plan(self.config)
        rank0, rank1 = [entry["argv"] for entry in plan["ranks"]]
        for arguments in (rank0, rank1):
            self.assertIn("--nnodes=2", arguments)
            self.assertIn("--cp_size=1", arguments)
            self.assertIn("--dp_size=1", arguments)
            self.assertIn("--port=65100", arguments)
            self.assertNotIn("--mtp_draft_vocab_path=", " ".join(arguments))
            self.assertFalse(any(item.startswith("--etcd_addr=") for item in arguments))
        self.assertIn("--node_rank=0", rank0)
        self.assertIn("--node_rank=1", rank1)
        self.assertNotEqual(plan["ranks"][0]["environment"]["MC_MIN_RPC_PORT"],
                            plan["ranks"][1]["environment"]["MC_MIN_RPC_PORT"])

    def test_pd_publishes_native_registration_capabilities(self) -> None:
        self.config.update(mode="pd", private_etcd="127.0.0.1:57579")
        plan = launch._plan(self.config)
        for rank in plan["ranks"]:
            self.assertIn("--enable_native_service_adapter=true", rank["argv"])
            self.assertIn("--instance_role=DECODE", rank["argv"])

    def test_rejects_main_etcd_registration(self) -> None:
        self.config.update(mode="pd", private_etcd="127.0.0.1:64379")
        with self.assertRaisesRegex(ValueError, "active main"):
            launch._plan(self.config)

    def test_rejects_overlapping_and_ephemeral_ports(self) -> None:
        invalid = deepcopy(self.config)
        invalid["ports"]["http"] = invalid["ports"]["master"] + 6
        with self.assertRaisesRegex(ValueError, "overlap"):
            launch._plan(invalid)
        self.config["ports"]["mooncake"][0] = [34000, 34016]
        with self.assertRaisesRegex(ValueError, "ephemeral"):
            launch._plan(self.config)

    def test_tp_custom_ar_switch_does_not_change_topology(self) -> None:
        enabled = launch._plan(self.config)
        self.config["custom_all_reduce"] = False
        disabled = launch._plan(self.config)
        self.assertEqual(enabled["ranks"][0]["argv"], disabled["ranks"][0]["argv"])
        self.assertEqual(disabled["ranks"][0]["environment"]["XLLM_MUSA_ENABLE_TP_CUSTOM_AR"], "0")

    def test_detects_gpu_allocations_even_with_zero_utilization(self) -> None:
        raw = ("6 MTT S5000 |00000000:c3:00.0 |0% 74395MiB(81920MiB)\n"
               "7 MTT S5000 |00000000:e3:00.0 |0% 0MiB(81920MiB)\n"
               "6 9999 /existing/service 74395MiB\n")
        with self.assertRaisesRegex(ValueError, "occupied"):
            launch._check_gpus(launch._gpu_snapshot(raw), [6, 7])

    def test_pid_reuse_never_passes_identity_check(self) -> None:
        old = {"pid": 12, "exe": "/owned/xllm", "argv": ["xllm"],
               "start_ticks": "50", "pgid": 12, "session": 12, "state": "S"}
        new = {**old, "start_ticks": "51"}
        self.assertFalse(launch._same_identity(new, old))
        with patch.object(launch, "_current", return_value=new), patch.object(launch.os, "killpg") as kill:
            with self.assertRaisesRegex(ValueError, "PID was reused"):
                launch._stop_state({"processes": {"0": old}}, 0)
            kill.assert_not_called()

    def test_foreign_container_cannot_signal_processes(self) -> None:
        self.config["identity"]["pid_namespace"] = "pid:[FOREIGN]"
        with self.assertRaisesRegex(ValueError, "namespace differs"):
            launch._assert_container(self.config)

    def test_active_listener_refused_before_bind_or_launch(self) -> None:
        with patch.object(launch, "_listeners", return_value={65206}), patch.object(launch.socket, "socket") as sock:
            with self.assertRaisesRegex(ValueError, "65206"):
                launch._check_ports(self.config)
            sock.assert_not_called()

    def test_stops_exact_owned_cpu_child(self) -> None:
        child = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
        record = launch._identity(child.pid)
        try:
            result = launch._stop_state({"processes": {"0": record}}, 2)
            self.assertFalse(result[0]["sigkill_sent"])
            self.assertEqual(result[0]["remaining_members"], [])
            self.assertEqual(child.wait(timeout=3), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                current = launch._current(record)
                if current is not None and launch._same_identity(current, record):
                    os.killpg(record["pgid"], signal.SIGKILL)
                child.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
