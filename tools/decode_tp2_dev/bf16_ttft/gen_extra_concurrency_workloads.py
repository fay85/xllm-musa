#!/usr/bin/env python3
"""Add workload_c{8,10,14}.json with the exact scheme of build_comparison_workload_stable.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

GEN = Path(__file__).resolve().with_name("build_comparison_workload_stable.py")
spec = importlib.util.spec_from_file_location("stable_builder", GEN)
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)

source = GEN.read_text()
source = source.replace("for concurrency in [6, 12, 16, 18]:", "for concurrency in [8, 10, 14]:")
source = source.replace("(ROOT / 'workload_manifest.json')", "(ROOT / 'workload_manifest_extra.json')")
assert "for concurrency in [8, 10, 14]:" in source and "workload_manifest_extra.json" in source
namespace = {"__name__": "extra_builder", "__file__": str(GEN)}
exec(compile(source, str(GEN), "exec"), namespace)
namespace["main"]()

root = Path("/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv")
for c in (6, 8, 10, 12, 14, 16, 18):
    d = json.loads((root / f"workload_c{c}.json").read_text())
    meta = d["measured_metadata"]
    print(c, len(d["measured_prompts"]), min(m["expanded_tokens"] for m in meta), max(m["expanded_tokens"] for m in meta),
          round(sum(m["expected_kv_hit_rate"] for m in meta) / len(meta), 4))
