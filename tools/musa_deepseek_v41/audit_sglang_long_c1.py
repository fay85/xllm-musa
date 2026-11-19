"""Join frozen EvalScope C1 streams to SGLang Decode request.finished JSON metadata.

Enable --log-requests --log-requests-level 0 --log-requests-format json on Decode.
Router IDs are *not* Decode completion IDs. Service startup probes are not samples.
"""
import argparse
import base64
import hashlib
import io
import json
import sqlite3
import statistics
from pathlib import Path

from audit_long_c1 import PlainData, require


def inspect(directory, decode_log, length, count):
    identity = json.loads((directory / "identity.json").read_text())
    pre = json.loads((directory / "tokenization_preflight.json").read_text())
    ids = pre["response"]["tokens"]
    digest = hashlib.sha256((json.dumps(ids, separators=(",", ":")) + "\n").encode()).hexdigest()
    require(pre["same_ids"] and len(ids) == length and
            digest == pre["expected_token_ids_sha256"] == identity["token_ids_sha256"],
            "tokenization mismatch")
    terminals = {}
    with decode_log.open(errors="replace") as stream:
        for line in stream:
            if '"event": "request.finished"' not in line:
                continue
            record = json.loads(line[line.index("{"):])
            m = record["out"]["meta_info"]
            if not record["obj"]["stream"]:  # server /generate warmers are non-streaming
                continue
            rid = record["rid"]
            require(record["obj"]["rid"] == m["id"] == rid and rid not in terminals, "bad terminal ID")
            terminals[rid] = m

    samples = []
    for phase, size in (("warmup", 1), ("formal", count)):
        dbs = list((directory / phase).rglob("benchmark_data.db"))
        require(len(dbs) == 1, "missing or ambiguous DB")
        with sqlite3.connect(dbs[0].resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM result ORDER BY start_time").fetchall()
        require(len(rows) == size, f"wrong sample count for {phase}")
        for row in rows:
            require(row["success"] == 1 and row["prompt_tokens"] == length and
                    row["completion_tokens"] == 1024, "client failure or wrong length")
            events = PlainData(io.BytesIO(base64.b64decode(row["response_messages"], validate=True))).load()
            rids = {e["id"] for e in events if e.get("id")}
            require(len(rids) == 1, "ambiguous SSE id")
            rid = rids.pop()
            require(rid in terminals, f"no unique Decode terminal for {rid}")
            m = terminals.pop(rid)
            require(m["finish_reason"]["type"] == "length" and m["prompt_tokens"] == length and
                    m["completion_tokens"] == 1024 and m["cached_tokens"] == 0 and
                    m["reasoning_tokens"] == 0 and m["num_retractions"] == 0, f"bad terminal {rid}")
            accepted, proposed, verify = (m[k] for k in
                                         ("spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct"))
            require(0 <= accepted <= proposed and proposed > 0 and verify > 0, "missing speculative work")
            choices = [c for e in events for c in e.get("choices", [])]
            require({c["finish_reason"] for c in choices if c.get("finish_reason")} == {"length"},
                    "missing length finish")
            text = "".join(c.get("delta", {}).get("content") or "" for c in choices)
            ttft, e2e = row["first_chunk_latency"], row["latency"]
            require(text and 0 < ttft < e2e, "missing text or invalid timing")
            samples.append(dict(phase=phase, id=rid, accepted=accepted, proposed=proposed,
                                verify=verify, dp_rank=m["dp_rank"],
                                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                ttft_ms=ttft*1000, tpot_ms=(e2e-ttft)*1000/1023, e2e_s=e2e))
    require(not terminals, "extra streaming terminals in isolated log")
    rows = [s for s in samples if s["phase"] == "formal"]
    return dict(length=length, count=count, samples=samples, token_ids_sha256=digest,
                means={key: statistics.mean(r[key] for r in rows) for key in ("ttft_ms", "tpot_ms", "e2e_s")},
                throughput=1024*count/sum(r["e2e_s"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("decode_log", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--count", type=int, default=5)
    a = parser.parse_args()
    print(json.dumps(inspect(a.directory, a.decode_log, a.length, a.count), indent=2))
