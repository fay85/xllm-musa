"""Attribute D rank-local kernels to q=6 target graph launches (not draft/eager).

Use the correlation ID, not just GPU timestamps: graph work runs asynchronously.
Reported costs are from the profiler-on request, never official serving latency.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def summarize(events):
    events = [e for e in events if e.get("ph") == "X"]
    steps = [e for e in events if e.get("cat") in {"cpu_op", "user_annotation"}
             and e["name"].startswith("V41/step/") and "/q=6/" in e["name"]]
    graphs = [e for e in events if e["name"] == "musaGraphLaunch"]
    kernels = defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel" and "correlation" in e.get("args", {}):
            kernels[e["args"]["correlation"]].append(e)
    if not steps:
        raise ValueError("no q=6 steps")
    costs, counts, rows, seen = Counter(), Counter(), [], set()
    for step in steps:
        launches = [g for g in graphs if step["pid"] == g["pid"] and step["tid"] == g["tid"]
                    and step["ts"] <= g["ts"] < step["ts"] + step["dur"]]
        if len(launches) != 1:
            raise ValueError(f"expected exactly one target launch for {step['name']}: {len(launches)}")
        corr = launches[0]["args"]["correlation"]
        nodes = kernels[corr]
        if not nodes or corr in seen:
            raise ValueError("unmatched or repeated target graph correlation")
        seen.add(corr)
        intervals = sorted((n["ts"], n["ts"] + n["dur"]) for n in nodes)
        end, union = intervals[0][0], 0
        for left, right in intervals:
            union += max(0, right - max(left, end))
            end = max(end, right)
        for n in nodes:
            counts[n["name"]] += 1
            costs[n["name"]] += n["dur"]
        rows.append(dict(step=step["name"], correlation=corr, nodes=len(nodes),
                         span_us=end - intervals[0][0], union_us=union))
    return dict(graphs=rows, mean_union_ms=mean(r["union_us"] for r in rows) / 1000,
                kernels=[dict(name=k, count=c, us_per_graph=costs[k]/len(rows))
                         for k, c in sorted(counts.items(), key=lambda p: -costs[p[0]])],
                scope="Single rank. Sum of kernel durations by name is not critical-path time.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(json.loads(args.trace.read_text())["traceEvents"]), indent=2))
