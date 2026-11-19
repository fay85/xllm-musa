"""Offline 2k goal gate; use ONLY after the group identity/terminal audits.

Input JSON: reference from audit_sglang_long_c1; two candidate group envelopes
(identity, normalized_config_sha256, launch_id, uid, audit from audit_long_c1).
Distinct launches MUST be verified from host launch receipts, not just names.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def judge(reference, groups):
    digest = hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != "9791fc7e15fb24e261569e0a1bddcbdd190f900b6f2d6732f5283ff52b1359fb":
        raise ValueError("not the frozen S2Ret reference")
    if reference["length"] != 2000 or reference["count"] != 5:
        raise ValueError("reference must be the frozen 2000/1024, 1+5")
    if len(groups) != 2 or len({g["uid"] for g in groups}) != 2:
        raise ValueError("need two distinct groups, not a pooled or repeated sample")
    for h in ("sj9", "sj10"):
        if len({g["launch_id"][h] for g in groups}) != 2:
            raise ValueError("both P and D must have restarted")
    if len({g["normalized_config_sha256"] for g in groups}) != 1:
        raise ValueError("execution configs differ")
    target = reference["means"]
    result, seen = [], set()
    for g in groups:
        identity = g["identity"]
        if not (identity["length"] == 2000 and identity["output_tokens"] == 1024 and
                identity["token_ids_sha256"] == reference["token_ids_sha256"] and
                identity["dataset_sha256"] == "d1310f52619cb1f8a449250e4fa602cfe2319a803b062c7e185b529014630b50" and
                identity["concurrency"] == 1 and not identity["profiler_on"]):
            raise ValueError("wrong workload")
        audit = g["audit"]
        rows = audit["formal"]
        if len(audit["warmup"]) != 1 or len(rows) != 5:
            raise ValueError("require all predeclared 1+5, including slow requests")
        for req in audit["warmup"] + rows:
            if req["id"] in seen:
                raise ValueError("repeated SSE id")
            seen.add(req["id"])
        metrics = {"ttft_ms": 1000 * statistics.mean(r["ttft_s"] for r in rows),
                   "tpot_ms": statistics.mean(r["tpot_ms"] for r in rows),
                   "e2e_s": statistics.mean(r["e2e_s"] for r in rows)}
        ok = (metrics["ttft_ms"] <= target["ttft_ms"] and
              metrics["tpot_ms"] <= target["tpot_ms"] and metrics["e2e_s"] < target["e2e_s"])
        result.append({"uid": g["uid"], "metrics": metrics, "pass": ok})
    return {"groups": result, "pass": all(g["pass"] for g in result)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frozen_reference", type=Path)
    p.add_argument("groups", type=Path, nargs=2)
    args = p.parse_args()
    result = judge(json.loads(args.frozen_reference.read_text()),
                   [json.loads(f.read_text()) for f in args.groups])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
