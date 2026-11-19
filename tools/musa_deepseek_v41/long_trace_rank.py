"""Summarize a *single rank* of a Chrome device trace; do not sum ranks."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(trace):
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    intervals = sorted((e["ts"], e["ts"]+e["dur"]) for e in kernels)
    union, end = 0, None
    for a, b in intervals:
        union += max(0, b-max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    per_name = Counter()
    counts = Counter()
    for e in kernels:
        counts[e["name"]] += 1
        per_name[e["name"]] += e["dur"]
    if not kernels:
        raise ValueError("no kernel events")
    return dict(trace=trace.name, kernel_count=len(kernels), kernel_union_ms=union/1000,
                kernel_span_ms=(end-intervals[0][0])/1000,
                top_by_aggregate_kernel_us=[dict(name=k, count=counts[k], kernel_us=v)
                                            for k, v in per_name.most_common(20)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    print(json.dumps(summarize(p.parse_args().trace), indent=2))
