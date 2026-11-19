"""Join bounded router lifecycle events with audited SSE and engine IDs (offline)."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re


def budgets(router_lines, terminal_lines, audit):
    ids = [row["id"] for phase in ("warmup", "formal") for row in audit[phase]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("missing or repeated audited response ID")
    terms = {}
    for line in terminal_lines:
        if "record_type: sequence_terminal," not in line:
            continue
        rid = re.search(r", request_id: ([^,]+),", line)
        if not rid or rid[1] not in ids:
            continue
        uid = re.search(r", x-request-id: ([^,]+),", line)
        if not uid or rid[1] in terms or uid[1] in terms.values():
            raise ValueError("terminal ID join is not one-to-one")
        terms[rid[1]] = uid[1]
    if set(terms) != set(ids):
        raise ValueError("missing audited terminal")
    events = defaultdict(list)
    wanted = set(terms.values())
    for line in router_lines:
        if "event=request_lifecycle payload=" not in line:
            continue
        v = json.loads(line.split("event=request_lifecycle payload=", 1)[1])
        if v.get("owner_role") != "EVENT_OWNER_ROLE_SERVICE":
            continue
        if v.get("global_request_id") in wanted:
            events[v["global_request_id"]].append(
                (v, datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")))
    out = []
    for rid in ids:
        ev = events[terms[rid]]
        def one(kind, result, *, duration=False):
            rows = [(v, t) for v, t in ev
                    if v.get("event_type") == "REQUEST_EVENT_TYPE_" + kind
                    and v.get("result") == "EVENT_RESULT_" + result
                    and (not duration or
                         v.get("stage_duration_validity") == "METRIC_VALIDITY_VALID")]
            if len(rows) != 1:
                raise ValueError(f"{rid}: expected one {kind}/{result}, got {len(rows)}")
            v, t = rows[0]
            if duration:
                ns = int(v["stage_duration_ns"])
                if ns < 0:
                    raise ValueError("negative duration")
                return ns / 1e6
            return t
        start = one("INGRESS", "STARTED")
        queue_ms = one("PREFILL_QUEUE", "ACCEPTED", duration=True)
        route_ms = one("ROUTE", "SUCCEEDED", duration=True)
        dispatch = one("P_DISPATCH", "ACCEPTED")
        log_ms = (dispatch - start).total_seconds() * 1000
        if log_ms < 0:
            raise ValueError("dispatch before ingress")
        out.append(dict(id=rid, global_request_id=terms[rid], ingress_to_queue_ms=queue_ms,
                        queue_to_route_ms=route_ms, ingress_to_dispatch_log_ms=round(log_ms)))
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("router_log", "decode_log", "joined_audit"):
        cli.add_argument(name, type=Path)
    a = cli.parse_args()
    with a.router_log.open() as router, a.decode_log.open() as decode:
        print(json.dumps(budgets(router, decode, json.loads(a.joined_audit.read_text())), indent=2))
