#!/usr/bin/env python3
"""C6 measured prompts until a timed wave adds no Lazy-capture lines."""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "/workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench")
import vl_bench as bench

DATASET = "/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv/workload_c6.json"


def capture_counts(log_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(glob.glob(f"{log_dir}/prefill*.log") + glob.glob(f"{log_dir}/decode*.log")):
        name = Path(path).name
        n = sum(1 for line in Path(path).read_text(errors="replace").splitlines() if "Lazy captur" in line)
        counts[name] = n
    return counts


def wave(args, prompts, tag, name):
    results = [None] * len(prompts)
    start, epoch = time.perf_counter(), time.time()

    def send(i: int) -> None:
        results[i] = bench.stream_chat(
            args.url, args.model, prompts[i], 100, f"vl-{tag}-{name}-{i}", args.timeout
        )

    bench._closed_loop(len(prompts), args.concurrency, send)
    wall = time.perf_counter() - start
    time.sleep(2)
    return results, wall, epoch


def fill_kv(summary, results, tag, name, dataset, log_dir: str):
    prefix = f"vl-{tag}-{name}"
    prefill = bench._request_stats(sorted(glob.glob(f"{log_dir}/prefill*_r0.log")), prefix)
    decode = bench._request_stats(sorted(glob.glob(f"{log_dir}/decode*.log")), prefix)
    ids = [r["request_id"] for r in results]
    prompt_tokens = sum(prefill[i][0] for i in ids if i in prefill)
    cached_tokens = sum(prefill[i][1] for i in ids if i in prefill)
    accepted = sum(decode[i][2] for i in ids if i in decode)
    proposed = sum(decode[i][3] for i in ids if i in decode)
    summary.update(
        kv_logged_requests=sum(1 for i in ids if i in prefill),
        kv_prompt_tokens=prompt_tokens,
        kv_cached_tokens=cached_tokens,
        kv_hit_rate=round(cached_tokens / prompt_tokens, 4) if prompt_tokens else None,
        mtp_acceptance=round(accepted / proposed, 4) if proposed else None,
        isl_min=min(x["expanded_tokens"] for x in dataset["measured_metadata"]),
        isl_max=max(x["expanded_tokens"] for x in dataset["measured_metadata"]),
    )
    return summary


def main() -> None:
    class Args:
        pass

    args = Args()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="Qwen3.8-27B")
    parser.add_argument("--topology", default="two_tp1")
    cli = parser.parse_args()
    args.model, args.isl, args.osl, args.timeout = cli.model_id, 22035, 100, 1800
    args.concurrency = 6
    args.url = "http://10.121.38.92:57888"
    args.server_log_dir = cli.server_log_dir
    out = Path(cli.out)
    dataset = json.loads(Path(DATASET).read_text())
    prompts = dataset["measured_prompts"]
    if len(prompts) != 60:
        raise SystemExit(f"expected 60 measured prompts, got {len(prompts)}")
    out.mkdir(parents=True, exist_ok=True)
    tag = str(time.time_ns())
    print(json.dumps({"capture_pass": "start", "requests": 60, "topology": cli.topology, "model": cli.model_id}), flush=True)
    _, capture_wall, _ = wave(args, prompts, tag, "capture")
    print(json.dumps({"capture_pass": "done", "wall_s": capture_wall}), flush=True)
    for attempt in range(1, 16):
        before = capture_counts(args.server_log_dir)
        name = f"c{attempt}"
        results, wall, epoch = wave(args, prompts, tag, name)
        after = capture_counts(args.server_log_dir)
        names = sorted(set(before) | set(after))
        deltas = {n: after.get(n, 0) - before.get(n, 0) for n in names}
        new_lines = sum(deltas.values())
        prefill_new = sum(count for name, count in deltas.items() if name.startswith("prefill"))
        decode_new = sum(count for name, count in deltas.items() if name.startswith("decode"))
        summary = bench._summarize(args, results, wall, epoch, tag)
        fill_kv(summary, results, tag, name, dataset, args.server_log_dir)
        summary.update(
            pass_index=attempt,
            new_capture_lines=new_lines,
            prefill_new_captures=prefill_new,
            decode_new_captures=decode_new,
            graph_capture_deltas=deltas,
            output_tokens_per_second=6000 / wall,
            topology=cli.topology,
        )
        (out / f"steady{attempt}.json").write_text(json.dumps({"summary": summary}) + "\n")
        print("SUMMARY " + json.dumps(summary), flush=True)
        if new_lines == 0:
            print(json.dumps({"steady": True, "pass": attempt}), flush=True)
            return
        print(json.dumps({
            "steady": False,
            "pass": attempt,
            "new_capture_lines": new_lines,
            "prefill_new_captures": prefill_new,
            "decode_new_captures": decode_new,
        }), flush=True)
    raise SystemExit("still capturing")


if __name__ == "__main__":
    main()
