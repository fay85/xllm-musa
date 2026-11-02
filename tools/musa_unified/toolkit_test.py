# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU checks for launch plans, process ownership, and benchmark acceptance gates."""

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.musa_unified import processes
from tools.musa_unified.artifacts import inventory
from tools.musa_unified.audit_qwen import audit
from tools.musa_unified.deepseek import bench
from tools.musa_unified.deepseek.run import _wait_rpc, plan, validate


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(100)) if text == "1 2" else []


class _Response:
    status = 200
    content_type = "text/event-stream"
    headers: dict = {}

    def __init__(self, cached: int, text: str) -> None:
        self.cached = cached
        self.text = text

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    @property
    def content(self) -> AsyncIterator[bytes]:
        async def stream() -> AsyncIterator[bytes]:
            for part in [self.text[:1], self.text[1:]]:
                await asyncio.sleep(0)
                yield ("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": part}}]}) + "\n").encode()
                yield b"\n"
            event = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 20000,
                    "completion_tokens": 100,
                    "total_tokens": 20100,
                    "prompt_tokens_details": {"cached_tokens": self.cached},
                },
            }
            yield ("data: " + json.dumps(event) + "\n").encode()
            yield b"\n"
            yield b"data: [DONE]\n"
            yield b"\n"

        return stream()


class _Client:
    def __init__(self, cached: int = 16000, text: str = "1 2") -> None:
        self.cached, self.text = cached, text

    def post(self, *args: object, **kwargs: object) -> _Response:
        return _Response(self.cached, self.text)


class ToolkitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(Path(__file__).with_name("deepseek").joinpath("cluster.example.json").read_text())
        self.config["control"]["host"] = "127.0.0.1"
        self.config["decode"]["host"] = "127.0.0.1"
        self.config["prefill"][0]["host"] = "127.0.0.2"
        self.config["prefill"][1]["host"] = "127.0.0.3"

    def test_role_shapes_and_rank_ports(self) -> None:
        for role, cp in [("prefill0", "2"), ("prefill1", "2"), ("decode", "1")]:
            spec = plan(self.config, role, Path("/tmp/test-plan"))
            self.assertEqual(len(spec["processes"]), 8)
            for rank, proc in enumerate(spec["processes"]):
                flags = dict(value[2:].split("=", 1) for value in proc["argv"][1:])
                self.assertEqual(flags["node_rank"], str(rank))
                self.assertEqual(flags["nnodes"], "8")
                self.assertEqual(flags.get("cp_size", "1"), cp)
                self.assertEqual(flags["num_speculative_tokens"], "6")
                self.assertEqual(proc["environment"]["XLLM_DSV4_PCP"], "1")
            self.assertEqual(len(spec["ports"]), len(set(spec["ports"])))

    def test_control_can_share_host_but_engines_cannot(self) -> None:
        validate(self.config)
        changed = copy.deepcopy(self.config)
        changed["prefill"][1]["host"] = changed["decode"]["host"]
        with self.assertRaises(ValueError):
            validate(changed)

    def test_placeholder_hosts_are_rejected(self) -> None:
        self.config["decode"]["host"] = "REPLACE_DECODE_IP"
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_stale_pid_after_reboot_is_not_owned(self) -> None:
        row = dict(
            pid=42,
            start_ticks=100,
            executable="/xllm",
            argv_sha256="a",
            boot_id="old",
            state="S",
        )
        current = dict(row, boot_id="new")
        with mock.patch.object(processes, "identity", return_value=current):
            self.assertFalse(processes.matches(row))
        with mock.patch.object(processes, "identity", return_value=dict(row, start_ticks=101)):
            self.assertFalse(processes.matches(row))
        with mock.patch.object(processes, "identity", return_value=row):
            self.assertTrue(processes.matches(row))

    def test_record_waits_for_nonempty_stable_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            executable = run / "engine"
            executable.write_text("test")
            good = dict(
                pid=42,
                start_ticks=100,
                state="S",
                boot_id="boot",
                executable=str(executable),
                argv0=str(executable),
                argv_bytes=20,
                argv_sha256="valid",
            )
            empty = dict(good, argv0="", argv_bytes=0, argv_sha256=hashlib.sha256(b"").hexdigest())
            with (
                mock.patch.object(processes, "identity", side_effect=[empty, good, good]) as observe,
                mock.patch.object(processes.os, "readlink", return_value=str(run / "engine.log")),
                mock.patch.object(processes.time, "sleep"),
            ):
                row = processes.record(run, "decode_test", 42, executable)
            self.assertEqual(observe.call_count, 3)
            self.assertEqual(row["argv_sha256"], "valid")

    def test_same_live_process_with_changed_argv_is_not_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            row = dict(
                pid=42,
                start_ticks=100,
                state="S",
                boot_id="boot",
                executable="/engine",
                argv_sha256="old",
                name="decode_test",
            )
            (run / "decode_test.identity.json").write_text(json.dumps(row))
            with (
                mock.patch.object(processes, "identity", return_value=dict(row, argv_sha256="new")),
                mock.patch.object(processes.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed identity"):
                    processes.stop(run)
                kill.assert_not_called()

    def test_control_waits_for_rpc_without_requiring_model_http(self) -> None:
        spec = plan(self.config, "control", Path("/tmp/control-plan"))
        self.assertEqual(spec["control_rpc_port"], self.config["control"]["rpc_port"])
        with (
            mock.patch("tools.musa_unified.deepseek.run.matches", return_value=True),
            mock.patch("tools.musa_unified.deepseek.run.socket.create_connection") as connect,
        ):
            _wait_rpc("127.0.0.1", 64989, [dict(pid=42)], 1)
            connect.assert_called_once_with(("127.0.0.1", 64989), timeout=2)

    def test_manifest_does_not_hash_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "payload").write_bytes(b"test")
            (root / "engine_manifest.json").write_text("old")
            first = inventory(root, frozenset({"engine_manifest.json"}))
            (root / "engine_manifest.json").write_text("new")
            self.assertEqual(first, inventory(root, frozenset({"engine_manifest.json"})))

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process ownership test")
    def test_record_and_stop_only_owned_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            with (run / "decode_test.log").open("wb") as log:
                child = subprocess.Popen(
                    ["/bin/sleep", "30"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            try:
                row = processes.record(run, "decode_test", child.pid, Path("/bin/sleep"))
                self.assertTrue(processes.matches(row))
                processes.stop(run)
                child.wait(timeout=5)
                self.assertFalse(processes.matches(row))
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)

    def test_warmup_precedes_each_formal_concurrency(self) -> None:
        args = argparse.Namespace(prefill_instances=2, rounds=2, batches=[1, 16], requests_per_round=32)
        stages = bench._stage_plan(args)
        for index, stage in enumerate(stages):
            if stage["phase"] == "formal":
                self.assertEqual(stages[index - 1]["phase"], "warmup")
                self.assertEqual(stages[index - 1]["concurrency"], stage["concurrency"])

    def _response(self, client: _Client) -> dict:
        args = argparse.Namespace(
            input_tokens=20000,
            output_tokens=100,
            expected_output="1 2",
            runtime_tokenizer=_Tokenizer(),
        )
        case = dict(id="case", payload={}, expected_cached_tokens=16000, primer_lcp_tokens=16000)
        with tempfile.TemporaryDirectory() as temp:
            return asyncio.run(bench._request(client, "http://unused", case, Path(temp), args))

    def test_good_response_passes_all_gates(self) -> None:
        self.assertTrue(self._response(_Client())["passed"])

    def test_wrong_cache_rejects_http_200(self) -> None:
        row = self._response(_Client(cached=0))
        self.assertFalse(row["passed"])
        self.assertFalse(row["checks"]["expected_cache_exact"])

    def test_wrong_text_rejects_exact_usage(self) -> None:
        row = self._response(_Client(text="1 3"))
        self.assertFalse(row["passed"])
        self.assertFalse(row["checks"]["strict_counting_prefix"])

    def test_qwen_audit_checks_every_formal_request(self) -> None:
        prompts = {"prompt_lengths": [22000] * 68}
        rows = [dict(request_id=f"case-{i}", completion_tokens=100, text="a useful answer") for i in range(60)]
        result = dict(
            requests=rows, summary=dict(measured_requests=60, server_errors=0, lazy_captures_during_measurement=0)
        )
        lines = [
            f"request.cpp:161] request_id: case-{i}, prompt_tokens: 22000, "
            "num_prefix_cache_tokens: 16384, generated_tokens: 100, finish_reason: length"
            for i in range(60)
        ]
        logs = {"rank.log": "\n".join(lines)}
        self.assertTrue(audit(result, prompts, logs, logs)["passed"])
        result["requests"][17]["text"] = "!" * 100
        checked = audit(result, prompts, logs, logs)
        self.assertFalse(checked["passed"])
        self.assertEqual(checked["passed_count"], 59)

    def test_qwen_audit_rejects_partial_window(self) -> None:
        with self.assertRaises(ValueError):
            audit(dict(requests=[], summary=dict(measured_requests=60)), {}, {}, {})


if __name__ == "__main__":
    unittest.main()
