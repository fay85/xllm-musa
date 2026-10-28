import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

prof_dir = Path(sys.argv[1])
files = sorted(prof_dir.glob("*.pt.trace.json"))
print("trace files", [(f.name, f.stat().st_size) for f in files])

BUCKETS = [
    ("allreduce", re.compile(r"all_?reduce|custom_ar|cross_device|mccl|nccl|two_?shot|one_?shot", re.I)),
    ("attention", re.compile(r"flash|fmha|attn|attention|mha|paged|varlen", re.I)),
    ("gdn", re.compile(r"gated_delta|delta_rule|chunk_|recurrent|gdn|fla_|kkt|solve_tril|wy_|cumsum|l2norm", re.I)),
    ("conv1d", re.compile(r"conv1d|causal_conv", re.I)),
    ("gemm", re.compile(r"gemm|matmul|mudnn.*mm|_mm_|cutlass|wmma|mma|xmma|linear", re.I)),
    ("norm/rope", re.compile(r"rms|norm|rope|rotary", re.I)),
    ("act", re.compile(r"silu|swish|glu|gelu|sigmoid|softplus|exp", re.I)),
    ("copy/elementwise", re.compile(r"copy|fill|cat|index|elementwise|vectorized|reduce|scatter|gather|add|mul|where|cast|contig", re.I)),
]


def bucket(name):
    for b, rx in BUCKETS:
        if rx.search(name):
            return b
    return "other"


for f in files:
    data = json.loads(f.read_text(errors="replace"))
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    cats = Counter(e.get("cat") for e in events if isinstance(e, dict))
    print(f"\n==== {f.name}: events {len(events)} cats {cats.most_common(12)}")
    dev = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"
           and str(e.get("cat", "")).lower() in ("kernel", "musa_kernel", "gpu_memcpy", "gpu_memset", "privateuse1_kernel")]
    if not dev:
        continue
    t0 = min(e["ts"] for e in dev)
    t1 = max(e["ts"] + e.get("dur", 0) for e in dev)
    total = sum(e.get("dur", 0) for e in dev)
    print(f"device events {len(dev)}, busy {total/1000:.1f} ms over window {(t1-t0)/1000:.1f} ms")
    by_bucket = defaultdict(float)
    by_name = defaultdict(lambda: [0.0, 0])
    for e in dev:
        n = e.get("name", "")
        by_bucket[bucket(n)] += e.get("dur", 0)
        by_name[n][0] += e.get("dur", 0)
        by_name[n][1] += 1
    for b, us in sorted(by_bucket.items(), key=lambda x: -x[1]):
        print(f"  {b:18s} {us/1000:9.1f} ms  {100*us/total:5.1f}%")
    print("  top kernels:")
    for n, (us, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:25]:
        print(f"    {us/1000:8.1f} ms  x{cnt:5d}  [{bucket(n)}] {n[:150]}")
