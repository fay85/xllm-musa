import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench")
import vl_bench as bench

HOST = "10.121.38.92"
DATASET = "/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv/workload_c6.json"
PREFILL_PORTS = [28100, 28101, 28102]


def call(port, path):
    req = urllib.request.Request(f"http://{HOST}:{port}/{path}", data=b"{}",
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()[:200]
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]


prompts = json.loads(Path(DATASET).read_text())["measured_prompts"]
for port in PREFILL_PORTS:
    print("start", port, call(port, "start_profile"), flush=True)
time.sleep(1)
tag = str(time.time_ns())
results = []
for i in (0, 1, 2):
    t0 = time.perf_counter()
    r = bench.stream_chat(f"http://{HOST}:57888", "Qwen3.8-27B", prompts[i], 100, f"vl-prof{tag}-p{i}", 600)
    results.append((i, round((time.perf_counter() - t0) * 1000, 1), r.get("ttft_ms") if isinstance(r, dict) else None))
    time.sleep(1.5)
print("requests", results, flush=True)
time.sleep(2)
for port in PREFILL_PORTS:
    print("stop", port, call(port, "stop_profile"), flush=True)
