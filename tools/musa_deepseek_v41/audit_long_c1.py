"""Join EvalScope streaming rows to native C1 sequence_terminals by completion id.

No network, no changes to measurements. Exit nonzero for incomplete, duplicate,
or mismatched groups (including warmup). Do not use request_id from the DB:
it can be NULL even when SSE chunks have a server-issued id.
"""
import argparse
import base64
import hashlib
import io
import json
import pickle
import re
import sqlite3
from pathlib import Path


class PlainData(pickle.Unpickler):
    """EvalScope stores JSON-like SSE events in base64-pickled lists."""

    def find_class(self, module, name):
        raise ValueError("non-data pickle in benchmark DB")

    def persistent_load(self, pid):
        raise ValueError("unexpected persistent object")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory, log, length, outputs, count):
    terminals = {}
    for line in log.open(errors="replace"):
        if "record_type: sequence_terminal," not in line:
            continue
        fields = dict(re.findall(r"(?:^|, )([\w-]+): ([^,\n]+)", line))
        rid = fields["request_id"]
        require(rid not in terminals, f"duplicate server terminal {rid}")
        terminals[rid] = fields

    seen = set()
    report = {}
    for phase, size in (("warmup", 1), ("formal", count)):
        dbs = list((directory / phase).rglob("benchmark_data.db"))
        require(len(dbs) == 1, f"{phase}: missing or ambiguous DB")
        with sqlite3.connect(dbs[0].resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = list(db.execute("SELECT * FROM result ORDER BY start_time"))
        require(len(rows) == size, f"{phase}: {len(rows)} != {size}")
        report[phase] = []
        for row in rows:
            require(row["success"] == 1 and row["prompt_tokens"] == length and
                    row["completion_tokens"] == outputs, "failed request or wrong length")
            events = PlainData(io.BytesIO(base64.b64decode(row["response_messages"],
                                                           validate=True))).load()
            ids = {e["id"] for e in events if e.get("id")}
            require(len(ids) == 1, f"inconsistent completion identity {ids}")
            rid = ids.pop()
            require(rid in terminals and rid not in seen, f"unmatched or repeated {rid}")
            seen.add(rid)
            choices = [c for e in events for c in e.get("choices", [])]
            text = "".join(c.get("delta", {}).get("content") or "" for c in choices)
            reasons = {c["finish_reason"] for c in choices if c.get("finish_reason")}
            usages = [e["usage"] for e in events if e.get("usage")]
            require(text and reasons == {"length"}, f"stream incomplete {rid}")
            # EvalScope can omit the usage-only SSE from response_messages.
            # The result row still carries usage; cross-check the native terminal.
            if usages:
                require(usages[-1]["prompt_tokens"] == length and
                        usages[-1]["completion_tokens"] == outputs, f"usage mismatch {rid}")
            fields = terminals[rid]
            require(fields["finish_reason"] == "length" and
                    int(fields["num_prefix_cache_tokens"]) == 0 and
                    int(fields["prompt_tokens"]) == length and
                    int(fields["generated_tokens"]) == outputs, f"terminal mismatch {rid}")
            ttft, latency = row["first_chunk_latency"], row["latency"]
            require(0 < ttft < latency, f"bad timing {rid}")
            report[phase].append(dict(id=rid, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      accepted=int(fields["speculative_accepted_tokens"]),
                                      proposed=int(fields["speculative_proposed_tokens"]),
                                      ttft_s=ttft, e2e_s=latency,
                                      tpot_ms=1000*(latency-ttft)/(outputs-1)))
    require(seen == set(terminals), "server log contains extra requests; supply isolated log")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("decode_rank0_log", type=Path)
    ap.add_argument("--length", required=True, type=int)
    ap.add_argument("--outputs", default=1024, type=int)
    ap.add_argument("--count", required=True, type=int)
    args = ap.parse_args()
    print(json.dumps(audit(args.directory, args.decode_rank0_log, args.length,
                           args.outputs, args.count), indent=2))
