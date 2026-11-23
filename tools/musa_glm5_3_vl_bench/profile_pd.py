"""Capture one routed request on selected workers from both P/D roles."""

import argparse
import fcntl
import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import vl_bench

LOGS = Path(__file__).resolve().parent / "pd_logs"
ROLES = ("prefill", "decode")


def worker_address(role, rank):
    pattern = re.compile(rf"Worker {rank}: server address: ([0-9.]+:\d+)")
    with (LOGS / role / f"rank_{rank}.log").open(errors="replace") as lines:
        for line in lines:
            if match := pattern.search(line):
                return match[1]
    raise RuntimeError(f"Missing {role} worker address for rank {rank}")


def rpc(workers, method):
    def call(item):
        label, addr = item
        req = urllib.request.Request(
            f"http://{addr}/DistributeWorker/{method}", data=b"{}", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            if result.get("ok") is not True:
                raise RuntimeError(result)
        except Exception as error:
            return f"{label} {addr} {method}: {error}"

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        errors = list(filter(None, pool.map(call, workers.items())))
    if errors:
        raise RuntimeError("\n".join(errors))


def capture(args, ranks, out):
    workers = {(role, rank): worker_address(role, rank) for role in ROLES for rank in ranks}
    dirs = {"prefill": args.prefill_trace_dir, "decode": args.decode_trace_dir}
    for role, directory in dirs.items():
        if not directory.is_dir():
            raise RuntimeError(f"{role} trace directory {directory} missing. Start with GLM5_TORCH_PROFILE=1")
    prompt = vl_bench.load_prompts(args.prompts, args.tokenizer, args.data_dir, args.isl, 0, 1, args.num_images)[
        "prompts"
    ][0]
    os.environ["GLM5_PD_BENCH"] = "1"

    def request(tag):
        result = vl_bench.stream_chat(args.url, args.model, prompt, args.osl, f"pd-prof-{time.time_ns()}-{tag}", 300)
        if result["prompt_tokens"] != args.isl:
            raise RuntimeError(f"Expected {args.isl} prompt tokens, received {result['prompt_tokens']}")
        if args.osl > 1:
            result["tpot_ms"] = (result["last_ms"] - result["ttft_ms"]) / (args.osl - 1)
        print(tag, result["request_id"], "usage", args.isl, args.osl, "tpot_ms", result.get("tpot_ms"))
        (out / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    for i in range(args.warmups):
        request(f"warmup-{i + 1}")
    before = {role: set(directory.glob("*.pt.trace.json")) for role, directory in dirs.items()}
    try:
        rpc(workers, "StartProfile")
        request("profile")
    finally:
        rpc(workers, "StopProfile")

    summary = {}
    for (role, rank), addr in workers.items():
        deadline = time.monotonic() + 60
        while True:
            new = set(dirs[role].glob(f"xllm_rank{rank}_*.pt.trace.json")) - before[role]
            if new or time.monotonic() >= deadline:
                break
            time.sleep(1)
        if len(new) != 1:
            raise RuntimeError(f"Expected one new trace for {role} rank {rank}, found {new}")
        src = new.pop()
        (out / role).mkdir(exist_ok=True)
        dest = src.rename(out / role / src.name)
        with dest.open() as trace:
            events = json.load(trace)["traceEvents"]
        kernels = [ev for ev in events if ev.get("cat") == "kernel" and ev.get("ph") == "X"]
        if not kernels:
            raise RuntimeError(f"No GPU kernel events in {dest}")
        times = Counter()
        for ev in kernels:
            times[ev["name"]] += ev["dur"]
        summary[f"{role}-{rank}"] = {
            "worker": addr,
            "trace": str(dest),
            "kernel_count": len(kernels),
            "top_gpu_kernels_total_us": dict(times.most_common(12)),
        }
        print(role, rank, dest.name, len(kernels), "GPU kernels")
        del events, kernels
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("url", "model", "tokenizer", "data-dir", "prompts"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--isl", type=int, default=20000)
    p.add_argument("--osl", type=int, default=16)
    p.add_argument("--num-images", type=int, default=0)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--ranks", default="0", help="Same ranks on both roles, e.g. 0,1 or all")
    for role in ROLES:
        p.add_argument(f"--{role}-trace-dir", type=Path, default=LOGS / role / "traces")
    args = p.parse_args()
    if not re.fullmatch(r"all|[0-7](,[0-7])*", args.ranks):
        p.error("--ranks expects 0..7 separated by commas, or all")
    ranks = list(range(8)) if args.ranks == "all" else sorted(set(map(int, args.ranks.split(","))))
    if min(args.isl, args.osl) < 1 or min(args.num_images, args.warmups) < 0:
        p.error("Invalid length, image count or warmup count")
    root = LOGS / "profiles"
    root.mkdir(exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        out.mkdir()
        print(out, flush=True)
        capture(args, ranks, out)


if __name__ == "__main__":
    main()
