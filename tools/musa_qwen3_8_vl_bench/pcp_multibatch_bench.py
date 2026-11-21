#!/usr/bin/env python3
"""Microbenchmark & parity harness for Qwen3.8-27B-FP8 multi-batch PCP prefill."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from typing import Dict, List, Tuple


def post_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 1,
    temperature: float = 0.0,
) -> Tuple[float, str, int, int]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    content = body["choices"][0]["message"].get("content", "")
    usage = body.get("usage", {})
    return (
        latency_ms,
        content,
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def make_unique_prompt(salt: str, char_budget: int) -> str:
    chunks: List[str] = []
    idx = 0
    cur = 0
    while cur < char_budget:
        piece = (
            f"[{salt}-b{idx}] Context-parallel hybrid GDN + full-attention "
            "prefill splits variable-length sequences on 64-token boundaries "
            "and forwards recurrent linear state across CP ranks. "
        )
        chunks.append(piece)
        cur += len(piece)
        idx += 1
    return "".join(chunks)[:char_budget] + "\nReply with ONE word: OK."


def run_burst(
    base_url: str, model: str, prompts: List[str], max_tokens: int = 1
) -> Tuple[float, List[Tuple[float, str, int, int]]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        t0 = time.perf_counter()
        futures = [
            ex.submit(post_chat, base_url, model, p, max_tokens, 0.0)
            for p in prompts
        ]
        results = [f.result() for f in futures]
        wall_ms = (time.perf_counter() - t0) * 1000.0
    return wall_ms, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.8 multi-batch PCP prefill throughput & TTFT."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8", help="Model name")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup bursts per case")
    parser.add_argument("--iters", type=int, default=4, help="Measured bursts per case")
    parser.add_argument("--tag", default="run", help="Unique salt tag for cold prompts")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    summary: Dict[str, Dict[str, float]] = {}
    cold_cases = [
        ("cold_4x384", 4, 1050),
        ("cold_4x768", 4, 2200),
        ("cold_4x1400", 4, 4100),
        ("cold_4x1900", 4, 5600),
        ("cold_8x768", 8, 2200),
    ]

    for name, n_reqs, chars in cold_cases:
        for w in range(args.warmup):
            run_burst(
                args.url,
                args.model,
                [make_unique_prompt(f"{args.tag}-{name}-w{w}-{i}", chars) for i in range(n_reqs)],
            )
        walls, mean_lats, batch_toks = [], [], []
        for it in range(args.iters):
            wall_ms, res = run_burst(
                args.url,
                args.model,
                [make_unique_prompt(f"{args.tag}-{name}-r{it}-{i}", chars) for i in range(n_reqs)],
            )
            walls.append(wall_ms)
            mean_lats.append(sum(r[0] for r in res) / len(res))
            batch_toks.append(sum(r[2] for r in res))
        mw = sum(walls) / len(walls)
        mt = sum(batch_toks) / len(batch_toks)
        summary[name] = {
            "wall_ms": round(mw, 2),
            "mean_ttft_ms": round(sum(mean_lats) / len(mean_lats), 2),
            "batch_tokens": round(mt, 1),
            "tok_per_s": round(mt / (mw / 1000.0), 1),
        }

    prefixes = [make_unique_prompt(f"{args.tag}-warm-pfx-{i}", 8600) for i in range(4)]
    prefix_toks = 0
    for pfx in prefixes:
        _, _, ptok, _ = post_chat(args.url, args.model, pfx, max_tokens=1)
        prefix_toks += (ptok // 64) * 64

    for name, suffix_chars in [
        ("warm_4x_pfx_384new", 1050),
        ("warm_4x_pfx_768new", 2200),
        ("warm_4x_pfx_1400new", 4100),
    ]:
        for w in range(args.warmup):
            run_burst(
                args.url,
                args.model,
                [
                    prefixes[i] + "\n" + make_unique_prompt(f"{args.tag}-{name}-w{w}-{i}", suffix_chars)
                    for i in range(4)
                ],
            )
        walls, mean_lats, total_toks = [], [], []
        for it in range(args.iters):
            wall_ms, res = run_burst(
                args.url,
                args.model,
                [
                    prefixes[i] + "\n" + make_unique_prompt(f"{args.tag}-{name}-r{it}-{i}", suffix_chars)
                    for i in range(4)
                ],
            )
            walls.append(wall_ms)
            mean_lats.append(sum(r[0] for r in res) / len(res))
            total_toks.append(sum(r[2] for r in res))
        mw = sum(walls) / len(walls)
        mt = sum(total_toks) / len(total_toks)
        new_toks = max(1.0, mt - prefix_toks)
        summary[name] = {
            "wall_ms": round(mw, 2),
            "mean_ttft_ms": round(sum(mean_lats) / len(mean_lats), 2),
            "total_tokens": round(mt, 1),
            "uncached_tokens": round(new_toks, 1),
            "uncached_tok_per_s": round(new_toks / (mw / 1000.0), 1),
        }

    for k, v in summary.items():
        print(f"{k:24s} | {json.dumps(v)}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
