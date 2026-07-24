#!/usr/bin/env python3
"""Phase-1 steady-batch decode harness (profiling only; no xLLM source edits).

Prefills once (max_tokens=1), then runs a long decode generate at fixed batch
size B. Reports batch-step latency by subtracting the prefill-only wall time
from the full generate wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310")

os.environ.setdefault(
    "MUSA_VISIBLE_DEVICES", os.environ.get("MUSA_VISIBLE_DEVICES", "1")
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    filler = "word "
    text = (filler * (target_tokens + 64)).strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/model_weights/Qwen3.5-27B")
    parser.add_argument("--isl", type=int, default=2500)
    parser.add_argument("--decode-steps", type=int, default=512)
    parser.add_argument("--batch-sizes", default="1,2,4,5,8")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument(
        "--out",
        default="/workspace/xllm-git-master/logs/decode_budget_steady.json",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from xllm import LLM, SamplingParams

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.isl)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"prompt_tokens={prompt_len} target_isl={args.isl}", flush=True)

    print("Loading model...", flush=True)
    llm = LLM(
        model=args.model,
        devices="musa:0",
        block_size=64,
        max_memory_utilization=0.75,
        enable_prefix_cache=False,
        enable_chunked_prefill=True,
        enable_schedule_overlap=False,
        max_tokens_per_chunk_for_prefill=8192,
        max_seqs_per_batch=max(batch_sizes),
        max_concurrent_requests=max(batch_sizes),
        enable_graph=True,
        enable_graph_mode_decode_no_padding=True,
        enable_prefill_piecewise_graph=False,
        max_tokens_for_graph_mode=8192,
        use_cpp_chat_template=True,
        disable_log_stats=True,
    )
    print("Model loaded.", flush=True)

    def make_sp(max_tokens: int) -> SamplingParams:
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "ignore_eos": True,
        }
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
        if args.top_p < 1.0:
            kwargs["top_p"] = args.top_p
        return SamplingParams(**kwargs)

    print("=== warmup ===", flush=True)
    llm.generate([prompt], make_sp(args.warmup_steps), use_tqdm=False)

    results = []
    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size
        print(f"\n=== B={batch_size} prefill-only ===", flush=True)
        t0 = time.perf_counter()
        prefill_out = llm.generate(prompts, make_sp(1), use_tqdm=False)
        prefill_s = time.perf_counter() - t0
        prefill_tokens = sum(o.usage.prompt_tokens for o in prefill_out)

        print(
            f"=== B={batch_size} decode steps={args.decode_steps} ===",
            flush=True,
        )
        t1 = time.perf_counter()
        decode_out = llm.generate(
            prompts, make_sp(args.decode_steps), use_tqdm=False
        )
        total_s = time.perf_counter() - t1
        completion_tokens = sum(o.usage.completion_tokens for o in decode_out)
        decode_s = max(total_s - prefill_s, 1e-6)
        steps = max(args.decode_steps - 1, 1)
        batch_step_ms = (decode_s / steps) * 1000.0
        agg_tok_s = completion_tokens / total_s

        row = {
            "batch_size": batch_size,
            "prompt_tokens": prompt_len,
            "prefill_wall_s": prefill_s,
            "prefill_prompt_tokens": prefill_tokens,
            "decode_steps": args.decode_steps,
            "total_wall_s": total_s,
            "completion_tokens": completion_tokens,
            "decode_wall_s_est": decode_s,
            "batch_step_ms": batch_step_ms,
            "per_request_tpot_ms": batch_step_ms,
            "aggregate_tokens_per_s": agg_tok_s,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "isl": args.isl,
        "decode_steps": args.decode_steps,
        "batch_sizes": batch_sizes,
        "results": results,
        "mean_batch_step_ms": statistics.mean(r["batch_step_ms"] for r in results)
        if results
        else None,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    llm.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
