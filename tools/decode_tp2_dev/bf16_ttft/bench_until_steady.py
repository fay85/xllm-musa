#!/usr/bin/env python3
"""Measured prompts at a given concurrency until a timed wave adds no Lazy-capture lines."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "/workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench")
import vl_bench as bench


def capture_counts(log_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(glob.glob(f"{log_dir}/prefill*.log") + glob.glob(f"{log_dir}/decode*.log")):
        counts[Path(path).name] = sum(
            1 for line in Path(path).read_text(errors="replace").splitlines() if "Lazy captur" in line)
    return counts


def wave(args, prompts, tag, name):
    results = [None] * len(prompts)
    start, epoch = time.perf_counter(), time.time()

    def send(i: int) -> None:
        results[i] = bench.stream_chat(args.url, args.model, prompts[i], args.osl, f"vl-{tag}-{name}-{i}", args.timeout)

    bench._closed_loop(len(prompts), args.concurrency, send)
    wall = time.perf_counter() - start
    time.sleep(2)
    return results, wall, epoch


def fill_kv(summary, results, tag, name, dataset, log_dir):
    prefix = f"vl-{tag}-{name}"
    prefill = bench._request_stats(sorted(glob.glob(f"{log_dir}/prefill*_r0.log")), prefix)
    decode = bench._request_stats(sorted(glob.glob(f"{log_dir}/decode*_r0.log")), prefix)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--server-log-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--max-passes", type=int, default=15)
    cli = parser.parse_args()

    class Args:
        pass

    args = Args()
    args.model, args.isl, args.osl, args.timeout = cli.model_id, 22035, 100, 1800
    args.concurrency = cli.concurrency
    args.url = "http://10.121.38.92:57888"
    args.server_log_dir = cli.server_log_dir
    dataset = json.loads(Path(cli.dataset).read_text())
    prompts = dataset["measured_prompts"]
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = str(time.time_ns())
    print(json.dumps({"capture_pass": "start", "concurrency": cli.concurrency, "requests": len(prompts),
                      "topology": cli.topology, "model": cli.model_id}), flush=True)
    _, capture_wall, _ = wave(args, prompts, tag, "capture")
    print(json.dumps({"capture_pass": "done", "wall_s": round(capture_wall, 1)}), flush=True)
    for attempt in range(1, cli.max_passes + 1):
        before = capture_counts(cli.server_log_dir)
        name = f"c{attempt}"
        results, wall, epoch = wave(args, prompts, tag, name)
        after = capture_counts(cli.server_log_dir)
        deltas = {n: after.get(n, 0) - before.get(n, 0) for n in sorted(set(before) | set(after))}
        prefill_new = sum(v for n, v in deltas.items() if n.startswith("prefill"))
        decode_new = sum(v for n, v in deltas.items() if n.startswith("decode"))
        summary = bench._summarize(args, results, wall, epoch, tag)
        fill_kv(summary, results, tag, name, dataset, cli.server_log_dir)
        summary.update(pass_index=attempt, new_capture_lines=prefill_new + decode_new,
                       prefill_new_captures=prefill_new, decode_new_captures=decode_new,
                       output_tokens_per_second=round(len(prompts) * args.osl / wall, 1),
                       topology=cli.topology, concurrency=cli.concurrency)
        (out / f"c{cli.concurrency}_pass{attempt}.json").write_text(json.dumps({"summary": summary}) + "\n")
        print("SUMMARY " + json.dumps(summary), flush=True)
        if prefill_new + decode_new == 0:
            print(json.dumps({"steady": True, "concurrency": cli.concurrency, "pass": attempt}), flush=True)
            return
    print(json.dumps({"steady": False, "concurrency": cli.concurrency}), flush=True)


if __name__ == "__main__":
    main()
