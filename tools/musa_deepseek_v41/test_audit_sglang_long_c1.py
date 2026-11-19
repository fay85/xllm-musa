"""Small synthetic coverage for the SGLang terminal join (no accelerator)."""
import base64
import hashlib
import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

from audit_sglang_long_c1 import inspect


class TerminalJoinTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.p = p = Path(t.name)
        self.log = p / "decode.log"
        ids = [3, 9]
        sha = hashlib.sha256(b"[3,9]\n").hexdigest()
        (p / "identity.json").write_text(json.dumps({"token_ids_sha256": sha}))
        (p / "tokenization_preflight.json").write_text(json.dumps({
            "same_ids": True, "response": {"tokens": ids}, "expected_token_ids_sha256": sha}))
        lines = []
        for phase in ("warmup", "formal"):
            rid = "decode-" + phase
            events = [{"id": rid, "choices": [{"delta": {"content": "ok"}, "finish_reason": "length"}]}]
            (p / phase).mkdir()
            with sqlite3.connect(p / phase / "benchmark_data.db") as db:
                db.execute("create table result(start_time,success,prompt_tokens,completion_tokens,"
                           "first_chunk_latency,latency,response_messages)")
                db.execute("insert into result values (0,1,2,1024,1,17,?)",
                           (base64.b64encode(pickle.dumps(events)),))
            meta = dict(id=rid, finish_reason={"type": "length"}, prompt_tokens=2, completion_tokens=1024,
                        cached_tokens=0, reasoning_tokens=0, num_retractions=0, spec_num_correct_drafts=697,
                        spec_num_proposed_drafts=1400, spec_verify_ct=280, dp_rank=0)
            lines.append(json.dumps(dict(event="request.finished", rid=rid,
                                         obj=dict(rid=rid, stream=True), out=dict(meta_info=meta))))
        self.log.write_text("\n".join(lines) + "\n")

    def test_join(self):
        self.assertEqual(inspect(self.p, self.log, 2, 1)["throughput"], 1024/17)

    def test_duplicate(self):
        self.log.write_text(self.log.read_text()*2)
        with self.assertRaises(ValueError):
            inspect(self.p, self.log, 2, 1)

    def test_reject_prefix_hit(self):
        self.log.write_text(self.log.read_text().replace('"cached_tokens": 0', '"cached_tokens": 1'))
        with self.assertRaises(ValueError):
            inspect(self.p, self.log, 2, 1)


if __name__ == "__main__":
    unittest.main()
