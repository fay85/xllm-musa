# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured rank/phase breakdown, with conservative bounds and explicit unknowns."""
from __future__ import annotations
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from calibrate_sampled_steps import requests


def _window(kv: int, length: int) -> str:
    index = kv-length
    return "early" if index <= 85 else "middle" if index <= 170 else "late"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args(); e=args.experiment
    phases=[]
    for mode in ["HOST","DEVICE"]:
        perf=e/"04_性能"/mode/"raw/performance"
        if not perf.exists():continue
        reqs={n:requests(perf,n) for n in [256,2048]}
        for host in ["sj9","sj10"]:
            scopes=e/"05_trace"/mode/host/"scopes.json"
            if not scopes.exists():continue
            data=json.loads(scopes.read_text())["measurements"]
            per_step=collections.defaultdict(list)
            for row in data:per_step[(row["pid"],row["thread"],row["step"])].append(row)
            for (pid,thread,step),rows in per_step.items():
                root=next(x for x in rows if x["scope"]=="V41/step")
                if host=="sj9":
                    # P CPU clocks do not align with the D-host client's clock.
                    # Retain all prefill scopes by real q/kv, without inventing
                    # exact per-request IDs or GPU/device start skew.
                    if root["q"] <= 1:continue
                    phase="prefill_all_q"+str(root["q"])+"_kv"+str(root["kv"])
                    length=-1;request_index=-1
                else:
                    if root["q"]!=1:continue
                    matches=[(length,i) for length,rr in reqs.items() for i,r in enumerate(rr) if r["start_time"] <= root["host_begin_ns"]/1e9 <= r["completed_time"]]
                    if not matches:continue
                    assert len(matches)==1,matches
                    length,request_index=matches[0];phase=_window(root["kv"],length)
                host_exclusive=collections.defaultdict(float)
                gpu_inclusive=collections.defaultdict(float)
                counts=collections.Counter();bad=collections.Counter()
                for row in rows:
                    name=row["scope"];host_exclusive[name]+=row["host_exclusive_ms"];counts[name]+=1
                    if row["device_event_status"]==0 and row["gpu_stream_span_ms"] is not None:gpu_inclusive[name]+=row["gpu_stream_span_ms"]
                    elif row["device_event_status"] is not None:bad[name]+=1
                for name in counts:
                    phases.append({"mode":mode,"host":host,"pid":pid,"step":step,"request_index":request_index,"input_tokens":length,"phase":phase,"q":root["q"],"kv":root["kv"],"scope":name,"invocations":counts[name],"host_exclusive_ms":host_exclusive[name],"gpu_inclusive_span_sum_ms":gpu_inclusive.get(name),"gpu_event_errors":bad[name],"note":"Inclusive device child sums are NOT additive critical-path shares."})
    out=e/"05_trace"/"phase_breakdown.json";out.write_text(json.dumps(phases,indent=2))
    grouped=collections.defaultdict(list)
    for row in phases:grouped[(row["mode"],row["host"],row["input_tokens"],row["phase"],row["scope"])].append(row)
    summary=[]
    for (mode,host,length,phase,scope),rows in grouped.items():
        gpu=[r["gpu_inclusive_span_sum_ms"] for r in rows if r["gpu_inclusive_span_sum_ms"] is not None]
        summary.append({"mode":mode,"host":host,"input_tokens":length,"phase":phase,"scope":scope,"rank_count":len({r["pid"] for r in rows}),"step_samples":len(rows),"host_exclusive_median_ms":statistics.median(r["host_exclusive_ms"] for r in rows),"gpu_inclusive_median_ms":statistics.median(gpu) if gpu else None,"gpu_error_count":sum(r["gpu_event_errors"] for r in rows),"p_low":0,"p_high":"unknown","critical_path_status":"NEEDS_DEPENDENCY_AND_OVERHEAD_REVIEW"})
    (e/"05_trace/phase_summary.json").write_text(json.dumps(summary,indent=2))
    if summary:
        with (e/"05_trace/critical_path.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({"measured_step_scope_rows":len(phases),"summary_rows":len(summary),"scope":"Native CPU/device ranges; p_high explicitly unknown, not a fabricated kernel share."}))


if __name__=="__main__":main()
