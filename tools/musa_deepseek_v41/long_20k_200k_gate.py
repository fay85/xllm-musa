# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Length-specific F/R gate, after raw request/terminal and identity collection.

The collector must rerun audit_long_c1 on the SQLite databases and isolated D
terminal log. This layer recomputes every metric and rejects reused starts,
config drift, partial groups, wrong length/reference and missing release.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def reference_metrics(reference: dict) -> dict:
    length = reference["length"]
    _require(length in (20000, 200000) and reference["outputs"] == 1024,
             "only the requested 20k/200k lengths with 1024 output")
    audits = reference["raw_audits"]
    hashes = reference["audit_sha256"]
    _require(len(audits) == len(hashes) == 2, "reference requires two rounds")
    if length == 20000:
        _require(hashes == [
            "5d6082ec0d0fcbe089a7e17c7804e7d0511a87e5deb49af7d931e46f22a48197",
            "e0ef7a75add0a7fbf38179038918a3d48fe64b1facb3a46b090a166a3d0cca29"],
                 "not the frozen S20A/S20B reference")
        _require(reference["token_ids_sha256"] ==
                 "6ee1cb788cedad06ba68d69bcbfe163bd649aaa1dff8e555911ee86bf30bdf27",
                 "wrong frozen 20k input")
    samples = []
    seen = set()
    for value, expected_sha in zip(audits, hashes):
        _require(canonical_sha(value) == expected_sha, "reference SHA mismatch")
        _require(value["length"] == length and value["count"] == 5 and
                 value["token_ids_sha256"] == reference["token_ids_sha256"],
                 "reference is from a different length/input")
        _require(len(value["samples"]) == 6 and
                 sum(row["phase"] == "warmup" for row in value["samples"]) == 1 and
                 sum(row["phase"] == "formal" for row in value["samples"]) == 5,
                 "reference must contain complete 1+5")
        for row in value["samples"]:
            _require(row["id"] not in seen, "duplicate reference SSE ID")
            seen.add(row["id"])
            _require(all(math.isfinite(row[key]) and row[key] > 0
                         for key in ("ttft_ms", "tpot_ms", "e2e_s")),
                     "invalid reference timing")
            _require(math.isclose(row["tpot_ms"],
                                 (1000 * row["e2e_s"] - row["ttft_ms"]) / 1023,
                                 rel_tol=1e-12, abs_tol=1e-10),
                     "reference timing definition mismatch")
        samples.extend(row for row in value["samples"] if row["phase"] == "formal")
    return {key: statistics.mean(row[key] for row in samples)
            for key in ("ttft_ms", "tpot_ms", "e2e_s")}


def judge(reference: dict, groups: list[dict]) -> dict:
    return _judge_two_registered_groups(reference, groups, reference_metrics(reference))


def _judge_two_registered_groups(reference: dict, groups: list[dict], target: dict) -> dict:
    """Shared integrity checks; callers must first independently verify their frozen reference."""
    _require(len(groups) == 2 and len({g["uid"] for g in groups}) == 2,
             "need two different predeclared F/R groups")
    _require([g["phase"] for g in groups] == ["F", "R"], "F then R required")
    registrations = [g.get("registration") for g in groups]
    _require(all(isinstance(r, dict) for r in registrations),
             "missing F/R registration")
    _require(canonical_sha(registrations[0]) == canonical_sha(registrations[1]),
             "F/R registration changed")
    registration = registrations[0]
    _require(registration.get("sha256") == canonical_sha(
        {k: v for k, v in registration.items() if k != "sha256"}),
        "registration SHA mismatch")
    _require(registration["rounds"] == [
        {"phase": g["phase"], "uid": g["uid"]} for g in groups] and
        registration["length"] == reference["length"] and
        registration["outputs"] == 1024,
        "round names or workload differ from predeclaration")
    registered_at = datetime.datetime.fromisoformat(registration["at"])
    _require(registered_at.tzinfo is not None, "registration requires timezone")
    for role in ("P", "D", "router"):
        starts = [g["starts"][role] for g in groups]
        _require(all(set(start) >= {"host", "container", "pid", "start_ticks", "at"}
                     and start["pid"] > 0 and int(start["start_ticks"]) > 0
                     for start in starts), "missing real start receipt")
        _require(canonical_sha(starts[0]) != canonical_sha(starts[1]) and
                 (starts[0]["pid"], starts[0]["start_ticks"]) !=
                 (starts[1]["pid"], starts[1]["start_ticks"]),
                 "P/D/router must all restart")
        _require(all(datetime.datetime.fromisoformat(start["at"]) > registered_at
                     for start in starts), "registration was not before launches")
    _require(len({g["execution_config_sha256"] for g in groups}) == 1,
             "execution config drift")
    _require(len({canonical_sha(g["assets"]) for g in groups}) == 1,
             "model, dependencies or ELF drift")
    seen = set()
    rounds = []
    for group in groups:
        identity = group["identity"]
        _require(identity["length"] == reference["length"] and
                 identity["output_tokens"] == 1024 and
                 identity["token_ids_sha256"] == reference["token_ids_sha256"] and
                 identity["dataset_sha256"] == reference["dataset_sha256"] and
                 identity["concurrency"] == 1 and identity["warmup"] == 1 and
                 identity["formal"] == 5 and identity["profiler_on"] is False,
                 "wrong candidate workload")
        _require(group["released"] == {"sj9": True, "sj10": True},
                 "missing both-host release evidence")
        _require(group["basic_answers"] == ["4.2", "56"], "basic function not passed")
        _require(group["raw_audit"]["length"] == reference["length"] and
                 group["raw_audit"]["outputs"] == 1024 and
                 group["raw_audit"]["source_sha256"], "missing raw audit provenance")
        audit = group["audit"]
        _require(len(audit["warmup"]) == 1 and len(audit["formal"]) == 5,
                 "require all 1+5 rows, no partial run or cherry-picking")
        for row in audit["warmup"] + audit["formal"]:
            _require(row["id"] not in seen, "repeated SSE ID")
            seen.add(row["id"])
            _require(all(math.isfinite(row[key]) and row[key] > 0
                         for key in ("ttft_s", "tpot_ms", "e2e_s")),
                     "invalid candidate timing")
            _require(math.isclose(row["tpot_ms"],
                                 1000 * (row["e2e_s"] - row["ttft_s"]) / 1023,
                                 rel_tol=1e-12, abs_tol=1e-10),
                     "candidate timing definition mismatch")
        metrics = {"ttft_ms": 1000 * statistics.mean(r["ttft_s"] for r in audit["formal"]),
                   "tpot_ms": statistics.mean(r["tpot_ms"] for r in audit["formal"]),
                   "e2e_s": statistics.mean(r["e2e_s"] for r in audit["formal"])}
        passed = (metrics["ttft_ms"] <= target["ttft_ms"] and
                  metrics["tpot_ms"] <= target["tpot_ms"] and
                  metrics["e2e_s"] < target["e2e_s"])
        rounds.append({"uid": group["uid"], "metrics": metrics, "pass": passed})
    return {"length": reference["length"], "target": target, "rounds": rounds,
            "pass": all(r["pass"] for r in rounds),
            "accuracy_status": "仅通过基础功能检查，未完成模型精度验收"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("groups", type=Path, nargs=2)
    args = parser.parse_args()
    result = judge(json.loads(args.reference.read_text()),
                   [json.loads(p.read_text()) for p in args.groups])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
