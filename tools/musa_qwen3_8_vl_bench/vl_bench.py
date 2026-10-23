"""Closed-loop benchmark for the Qwen3.8-27B-FP8 VL PD stack.

Each prompt is one 40,960-token chat message: a shared region (instructions,
news articles and five 854x480 photos, at least 32,864 tokens) followed by a
unique "Additional document N" and a question about it. The prefix cache
reuses the shared region in whole 8,192-token chunks, so once a prefill
instance has seen it, every request hits 32,768 of its 40,960 tokens (80%).

Protocol: master warm-up bursts, sequential warm-up requests that seed the
prefix and encoder caches of every prefill instance, a closed-loop warm-up at
the measured concurrency that captures the lazily built decode and verify
graphs, then the measured closed loop.
"""

import argparse
import datetime
import glob
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.logger import logger

# Image file and the index of the shared-region article it follows.
IMAGES = (
    ("01_us_capitol.jpg", 0),
    ("03_kremlin.jpg", 16),
    ("02_supreme_court.jpg", 21),
    ("04_oil_tanker.jpg", 23),
    ("05_budget_airline.jpg", 25),
)
# Vision tokens of one 854x480 image.
IMAGE_TOKENS = 405
HEADER = (
    "Below is a collection of news articles, some illustrated with photos, followed by an additional "
    "document. Read everything carefully and answer the question at the end.\n\n"
)
QUESTION = "\n\nQuestion: Summarize the key points of Additional document %d in a few sentences."
# Keeps the last reusable token inside the shared text whatever the tokenizer
# merges at the junction with the unique part.
_SHARED_SLACK_TOKENS = 96
# Per-paragraph estimates grow a prompt until it is this close to the target;
# exact counts take over from there.
_ESTIMATE_MARGIN_TOKENS = 512
_PAD_SLACK = 4
_MASTER_BURST = 16
_SERVER_ERRORS = re.compile(r"terminate called|core dumped|Check failed", re.IGNORECASE)
_LAZY_CAPTURE = re.compile(r"^I(\d{8}) (\d\d:\d\d:\d\d\.\d+) .*Lazy capturing")


def _article_text(article: dict) -> str:
    paras = article.get("paras") or []
    paras = paras if isinstance(paras, list) else [str(paras)]
    return article.get("title", "") + "\n\n" + "\n\n".join(paras) + "\n\n"


class PromptBuilder:
    """Builds prompts of exactly isl tokens as the instances count them."""

    def __init__(self, tokenizer_dir: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def count(self, content: str) -> int:
        """Prompt tokens of one user message: chat template plus expanded images."""
        num_images = content.count("<img>")
        text = re.sub(r"<img>.*?</img>", "<|vision_start|><|image_pad|><|vision_end|>", content)
        rendered = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self._length(rendered) + num_images * (IMAGE_TOKENS - 1)

    def build(
        self, articles: list[dict], image_dir: str, isl: int, shared_tokens: int, num_prompts: int
    ) -> tuple[list[str], int]:
        """Returns the prompts and the token count of their shared region."""
        by_article = {index: name for name, index in IMAGES}
        shared = HEADER
        used = 0
        for index, article in enumerate(articles):
            shared += _article_text(article)
            if index in by_article:
                shared += f"<img>file://{image_dir}/{by_article[index]}</img>\n\n"
            used = index + 1
            if index >= max(by_article) and self.count(shared) >= shared_tokens + _SHARED_SLACK_TOKENS:
                break
        pool = [p for a in articles[used:] for p in _article_text(a).split("\n\n") if p.strip()]
        if not pool or self.count(shared) < shared_tokens + _SHARED_SLACK_TOKENS:
            raise ValueError(f"{len(articles)} articles are too short for a {shared_tokens}-token shared region")
        pool_tokens = [self._length(p + "\n\n") for p in pool]
        stride = max(1, len(pool) // max(num_prompts, 1))
        prompts = []
        for i in range(num_prompts):
            question = QUESTION % (i + 1)
            body = f"Additional document {i + 1}:\n\n"
            cursor = i * stride
            estimate = self.count(shared + body + question)
            while estimate + pool_tokens[cursor % len(pool)] < isl - _ESTIMATE_MARGIN_TOKENS:
                estimate += pool_tokens[cursor % len(pool)]
                body += pool[cursor % len(pool)] + "\n\n"
                cursor += 1
            while self.count(shared + body + question) < isl:
                body += pool[cursor % len(pool)] + "\n\n"
                cursor += 1
            while self.count(shared + body + question) > isl and "\n\n" in body.rstrip("\n"):
                body = "\n\n".join(body.rstrip("\n").split("\n\n")[:-1]) + "\n\n"
            prompts.append(shared + body + self._pad(shared + body, question, isl) + question)
            logger.info(f"built prompt {i + 1}/{num_prompts}")
        return prompts, self.count(shared)

    def _length(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pad(self, head: str, question: str, isl: int) -> str:
        """Appends " ." up to isl tokens; an overshoot swaps the last " ." for a shorter tail."""
        pad = " ." * max(0, isl - self.count(head + question) - _PAD_SLACK)
        if self.count(head + pad + question) >= isl:
            pad = ""
        while self.count(head + pad + question) < isl:
            pad += " ."
        if self.count(head + pad + question) > isl:
            tails = (".", " ", "\n", "")
            pad = next((pad[:-2] + t for t in tails if self.count(head + pad[:-2] + t + question) == isl), pad)
        return pad


def load_prompts(path: str, tokenizer_dir: str, data_dir: str, isl: int, shared_tokens: int, num_prompts: int) -> dict:
    """Loads the prompt set from path, building and saving it first if needed."""
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("isl") == isl and len(cached["prompts"]) >= num_prompts:
            return cached
        logger.warning(f"{path} does not hold {num_prompts} prompts of {isl} tokens, rebuilding it")
    with open(os.path.join(data_dir, "articles.json")) as f:
        articles = json.load(f)
    image_dir = os.path.abspath(os.path.join(data_dir, "images"))
    builder = PromptBuilder(tokenizer_dir)
    prompts, shared = builder.build(articles, image_dir, isl, shared_tokens, num_prompts)
    lengths = sorted({builder.count(p) for p in prompts})
    if lengths != [isl]:
        raise RuntimeError(f"prompt lengths {lengths} differ from {isl}")
    prompt_set = {"isl": isl, "shared": shared, "lengths": lengths, "prompts": prompts}
    with open(path, "w") as f:
        json.dump(prompt_set, f, ensure_ascii=False)
    return prompt_set


def stream_chat(url: str, model: str, content: str, osl: int, request_id: str, timeout: float) -> dict:
    """Sends one streaming chat request and times its first and last content chunks."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": osl,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "request_id": request_id,
            "x_request_id": request_id,
        }
    ).encode()
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
    )
    start = time.perf_counter()
    first = last = None
    pieces = []
    completion_tokens = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            completion_tokens = (chunk.get("usage") or {}).get("completion_tokens", completion_tokens)
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    last = time.perf_counter()
                    first = last if first is None else first
                    pieces.append(piece)
    if first is None:
        raise RuntimeError(f"no content streamed for {request_id}")
    return {
        "request_id": request_id,
        "ttft_ms": (first - start) * 1e3,
        "last_ms": (last - start) * 1e3,
        "completion_tokens": completion_tokens or osl,
        "text": "".join(pieces),
    }


def _request_stats(log_paths: list[str], prefix: str) -> dict:
    """Per-request prompt, prefix-cache and speculative token counts from the server logs."""
    fields = ("prompt_tokens", "num_prefix_cache_tokens", "speculative_accepted_tokens", "speculative_proposed_tokens")
    pattern = re.compile(rf"x-request-id: ({re.escape(prefix)}[^,]*),")
    stats = {}
    for path in log_paths:
        with open(path, errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    values = [re.search(rf"{field}: (\d+)", line) for field in fields]
                    stats[match.group(1)] = tuple(int(v.group(1)) if v else 0 for v in values)
    return stats


def _server_log_events(log_dir: str, since_epoch: float) -> tuple[int, int]:
    """Lazy graph captures after since_epoch and fatal errors in all instance logs."""
    captures = errors = 0
    for path in glob.glob(os.path.join(log_dir, "*.log")):
        with open(path, errors="replace") as f:
            for line in f:
                if _SERVER_ERRORS.search(line):
                    errors += 1
                match = _LAZY_CAPTURE.match(line)
                if match:
                    stamp = datetime.datetime.strptime(match.group(1) + " " + match.group(2), "%Y%m%d %H:%M:%S.%f")
                    captures += stamp.timestamp() >= since_epoch
    return captures, errors


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(percent / 100.0 * (len(ordered) - 1))))]


def _closed_loop(num_requests: int, concurrency: int, send: Callable[[int], None]) -> None:
    """Runs send(index) for every index with concurrency requests in flight."""
    lock = threading.Lock()
    cursor = [0]
    failures = []

    def client() -> None:
        while True:
            with lock:
                index = cursor[0]
                cursor[0] += 1
            if index >= num_requests or failures:
                return
            try:
                send(index)
            except Exception as exc:
                failures.append(f"request {index}: {exc}")
                return

    threads = [threading.Thread(target=client) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))


def _master_burst(args: argparse.Namespace, request_prefix: str) -> list[float]:
    """Sends _MASTER_BURST tiny requests at once and returns their TTFTs."""
    ttfts = []

    def send(i: int) -> None:
        result = stream_chat(args.url, args.model, "Reply with OK.", 1, f"{request_prefix}{i}", args.timeout)
        ttfts.append(result["ttft_ms"])

    _closed_loop(_MASTER_BURST, _MASTER_BURST, send)
    return ttfts


def _warm_up(args: argparse.Namespace, prompts: list[str], tag: str) -> None:
    for burst in range(args.warm_master):
        ttfts = _master_burst(args, f"vl-{tag}-m{burst}-")
        logger.info(f"master warm-up burst {burst}: {len(ttfts)} ok, slowest TTFT {max(ttfts):.0f} ms")
    for start in range(0, args.warm, args.warm_concurrency):
        wave = range(start, min(start + args.warm_concurrency, args.warm))

        def send(i: int, wave: range = wave) -> None:
            result = stream_chat(args.url, args.model, prompts[wave[i]], 8, f"vl-{tag}-w{wave[i]}", args.timeout)
            logger.info(f"warm-up {wave[i]}: TTFT {result['ttft_ms']:.0f} ms")

        _closed_loop(len(wave), len(wave), send)
    if args.warm_closed:
        start = time.perf_counter()
        _closed_loop(
            args.warm_closed,
            args.concurrency,
            lambda i: stream_chat(
                args.url, args.model, prompts[i % args.warm], args.osl, f"vl-{tag}-wc{i}", args.timeout
            ),
        )
        logger.info(
            f"closed-loop warm-up: {args.warm_closed} requests at C={args.concurrency} "
            f"in {time.perf_counter() - start:.1f} s"
        )


def _summarize(args: argparse.Namespace, results: list[dict], wall: float, epoch: float, tag: str) -> dict:
    ttft = [r["ttft_ms"] for r in results]
    steady = ttft[args.concurrency :] if len(ttft) > args.concurrency else ttft
    tpot = [(r["last_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1) for r in results]
    summary = {
        "concurrency": args.concurrency,
        "measured_requests": len(results),
        "isl": args.isl,
        "osl": args.osl,
        "wall_s": round(wall, 1),
        "measure_start_epoch": round(epoch, 3),
        "request_rate": round(len(results) / wall, 3),
        "ttft_mean_ms": round(statistics.mean(ttft), 1),
        "ttft_p50_ms": round(_percentile(ttft, 50), 1),
        "ttft_p90_ms": round(_percentile(ttft, 90), 1),
        "ttft_p99_ms": round(_percentile(ttft, 99), 1),
        "ttft_max_ms": round(max(ttft), 1),
        "ttft_steady_mean_ms": round(statistics.mean(steady), 1),
        "tpot_mean_ms": round(statistics.mean(tpot), 2),
        "tpot_p50_ms": round(_percentile(tpot, 50), 2),
        "tpot_p90_ms": round(_percentile(tpot, 90), 2),
        "tpot_max_ms": round(max(tpot), 2),
    }
    if not args.server_log_dir:
        return summary
    prefill = _request_stats(sorted(glob.glob(os.path.join(args.server_log_dir, "prefill*_r0.log"))), f"vl-{tag}-c")
    decode = _request_stats(sorted(glob.glob(os.path.join(args.server_log_dir, "decode*.log"))), f"vl-{tag}-c")
    ids = [r["request_id"] for r in results]
    prompt_tokens = sum(prefill[i][0] for i in ids if i in prefill)
    cached_tokens = sum(prefill[i][1] for i in ids if i in prefill)
    accepted = sum(decode[i][2] for i in ids if i in decode)
    proposed = sum(decode[i][3] for i in ids if i in decode)
    captures, errors = _server_log_events(args.server_log_dir, epoch)
    summary.update(
        {
            "kv_logged_requests": sum(1 for i in ids if i in prefill),
            "kv_prompt_tokens": prompt_tokens,
            "kv_cached_tokens": cached_tokens,
            "kv_hit_rate": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else None,
            "mtp_acceptance": round(accepted / proposed, 4) if proposed else None,
            "lazy_captures_during_measurement": captures,
            "server_errors": errors,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:64888", help="master address")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8", help="model id to request")
    parser.add_argument("--tokenizer", required=True, help="model directory, for token counting")
    parser.add_argument("--data-dir", required=True, help="directory with articles.json and images/")
    parser.add_argument("--prompts", required=True, help="prompt set JSON, built if missing")
    parser.add_argument("--isl", type=int, default=40960)
    parser.add_argument("--shared", type=int, default=32768, help="minimum shared-region tokens")
    parser.add_argument("--osl", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--requests", type=int, default=60, help="measured requests")
    parser.add_argument("--warm", type=int, default=8, help="sequential warm-up requests, distinct prompts")
    parser.add_argument(
        "--warm-concurrency", type=int, default=1, help="warm-up requests in flight at once during --warm"
    )
    parser.add_argument(
        "--warm-master",
        type=int,
        default=2,
        help="bursts of concurrent tiny requests run first, so every master worker thread "
        "builds its tokenizer before measuring",
    )
    parser.add_argument(
        "--warm-closed",
        type=int,
        default=None,
        help="closed-loop warm-up requests at --concurrency cycling the warm-up prompts, so decode "
        "and verify meet every batch size before measuring (default: 4 x concurrency)",
    )
    parser.add_argument("--server-log-dir", default="", help="run directory of start_stack.sh, for KV hit stats")
    parser.add_argument("--out", default="", help="write the summary and per-request results here")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="only build or check the prompt set")
    args = parser.parse_args()
    if args.warm_closed is None:
        args.warm_closed = 4 * args.concurrency

    prompt_set = load_prompts(
        args.prompts, args.tokenizer, args.data_dir, args.isl, args.shared, args.warm + args.requests
    )
    prompts = prompt_set["prompts"][: args.warm + args.requests]
    logger.info(
        f"{len(prompts)} prompts of {prompt_set['lengths']} tokens, shared region {prompt_set['shared']} tokens, "
        f"{prompts[0].count('<img>')} images each"
    )
    if args.dry_run:
        return

    tag = str(int(time.time()))
    _warm_up(args, prompts, tag)
    measured = prompts[args.warm :]
    results = [None] * len(measured)
    start = time.perf_counter()
    epoch = time.time()

    def send(i: int) -> None:
        offset = time.perf_counter() - start
        results[i] = stream_chat(args.url, args.model, measured[i], args.osl, f"vl-{tag}-c{i}", args.timeout)
        results[i]["start_s"] = offset

    _closed_loop(len(measured), args.concurrency, send)
    wall = time.perf_counter() - start
    # Let the instances flush their request logs before reading them.
    time.sleep(2)
    for r in results:
        tpot = (r["last_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
        print(f"{r['request_id']} start={r['start_s']:.2f}s ttft={r['ttft_ms']:.0f}ms tpot={tpot:.2f}ms", flush=True)
    summary = _summarize(args, results, wall, epoch, tag)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "requests": results}, f, ensure_ascii=False)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
