"""Focused tests for the 2000/1024 goal, including the no-pooling rule."""
import json
import unittest
from pathlib import Path

from long_c1_two_restart_gate import judge


class GateTest(unittest.TestCase):
    def setUp(self):
        p = Path(__file__).resolve().parents[2] / "docs/zh/long_c1_20261119/S2Ret冻结参考.json"
        self.ref = json.loads(p.read_text())
        self.gs = []
        for i in (0, 1):
            rs = [dict(ttft_s=1., tpot_ms=15., e2e_s=16.345, id=f"{i}-{j}") for j in range(6)]
            self.gs.append(dict(uid=str(i), normalized_config_sha256="same",
                                launch_id={"sj9": str(i), "sj10": str(i)},
                                identity=dict(length=2000, output_tokens=1024, concurrency=1, profiler_on=False,
                                              token_ids_sha256=self.ref["token_ids_sha256"],
                                              dataset_sha256="d1310f52619cb1f8a449250e4fa602cfe2319a803b062c7e185b529014630b50"),
                                audit=dict(warmup=rs[:1], formal=rs[1:])))
    def test_pass(self):
        self.assertTrue(judge(self.ref, self.gs)["pass"])

    def test_no_pooling(self):
        for s in self.gs[1]["audit"]["formal"]:
            s["ttft_s"] = 1.100
        self.assertFalse(judge(self.ref, self.gs)["pass"])

    def test_e2e_equality_fails(self):
        for s in self.gs[0]["audit"]["formal"]:
            s["e2e_s"] = self.ref["means"]["e2e_s"]
        self.assertFalse(judge(self.ref, self.gs)["pass"])

    def test_changed_reference(self):
        self.ref["means"]["ttft_ms"] += 1
        with self.assertRaises(ValueError):
            judge(self.ref, self.gs)

    def test_one_restart_not_two(self):
        self.gs[1]["launch_id"]["sj10"] = self.gs[0]["launch_id"]["sj10"]
        with self.assertRaises(ValueError):
            judge(self.ref, self.gs)
