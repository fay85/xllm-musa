#!/usr/bin/env python3
"""Closed-loop benchmark for the GLM-5.3-Flash VLM service on MUSA.

Default workload: 2,048 input tokens per request (including one 854x480 photo
and news article paragraphs) and 256 output tokens.
"""

import argparse
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.logger import logger

IMAGE_NAMES = (
    "01_us_capitol.jpg",
    "03_kremlin.jpg",
    "02_supreme_court.jpg",
    "04_oil_tanker.jpg",
    "05_budget_airline.jpg",
)
# Vision tokens of one 854x480 image after GLM-5.3-Flash smart_resize_tokens
# (resized to 504x868, patch_size=14, merge_size=2 -> 36x62 / 4 = 558 tokens).
DEFAULT_IMAGE_TOKENS = 558
HEADER = (
    "Below is a collection of news articles, some illustrated with photos, followed by an additional "
    "document. Read everything carefully and answer the question at the end.\n\n"
)
QUESTION = "\n\nQuestion: Summarize the key points of Additional document %d in a few sentences."
_SHARED_SLACK_TOKENS = 32
_PAD_SLACK = 4
_MASTER_BURST = 4
_SERVER_ERRORS = re.compile(r"terminate called|core dumped|Check failed", re.IGNORECASE)
_LAZY_CAPTURE = re.compile(r"^I(\d{8}) (\d\d:\d\d:\d\d\.\d+) .*Lazy capturing")
_GLOG_TIMESTAMP = re.compile(r"^[IWEF](\d{8}) (\d\d:\d\d:\d\d\.\d+)")

_FALLBACK_PARAS = [
    (
        f"Section {i}: Modern heterogeneous computing systems combine high-bandwidth memory, "
        "tensor processing cores, and low-latency interconnects to accelerate large multimodal "
        "language models across distributed nodes with predictable tail latency."
    )
    for i in range(1, 256)
]


def _smart_resize_image_tokens(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_factor: int = 2,
    min_tokens: int = 16,
    max_tokens: int = 8000,
) -> int:
    """Mirrors Qwen2VLImageProcessor / Glm5NextImageProcessor smart_resize_tokens."""
    factor = patch_size * merge_size
    pixels_per_token = temporal_factor * factor * factor
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(val: int) -> int:
        return ((val + factor - 1) // factor) * factor

    aligned_frames = temporal_factor
    aligned_h = align(height)
    aligned_w = align(width)
    budget = aligned_frames * aligned_h * aligned_w
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (temporal_factor * height * width))
        aligned_h = align(max(1, math.ceil(height * scale)))
        aligned_w = align(max(1, math.ceil(width * scale)))
        budget = aligned_frames * aligned_h * aligned_w
    if budget > max_pixels:
        low, high = 1, height
        best_h = best_w = factor
        while low <= high:
            mid_h = (low + high) // 2
            mid_w = max(1, math.floor(width * mid_h / height))
            cand_h, cand_w = align(mid_h), align(mid_w)
            if aligned_frames * cand_h * cand_w <= max_pixels:
                best_h, best_w = cand_h, cand_w
                low = mid_h + 1
            else:
                high = mid_h - 1
        aligned_h, aligned_w = best_h, best_w

    grid_t = aligned_frames // temporal_factor
    grid_h = aligned_h // patch_size
    grid_w = aligned_w // patch_size
    return (grid_t * grid_h * grid_w) // (merge_size * merge_size)


def _image_token_count(image_path: str) -> int:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        return _smart_resize_image_tokens(width, height)
    except Exception:
        return DEFAULT_IMAGE_TOKENS


def _extract_paragraphs(articles: list[dict]) -> list[str]:
    paras = []
    for article in articles:
        title = (article.get("title") or "").strip()
        if title:
            paras.append(title)
        raw_paras = article.get("paras") or []
        if not isinstance(raw_paras, list):
            raw_paras = [str(raw_paras)]
        for p in raw_paras:
            if isinstance(p, str) and p.strip():
                paras.append(p.strip())
    return paras or _FALLBACK_PARAS


class PromptBuilder:
    """Builds prompts of exactly isl tokens as GLM-5.3-Flash counts them."""

    def __init__(self, tokenizer_dir: str, reasoning_effort: str = "low") -> None:
        tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.isfile(tokenizer_json):
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(tokenizer_json)
            self._fast_file = True
        else:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            self._fast_file = False
        self._reasoning_effort = reasoning_effort.capitalize() if reasoning_effort else "Max"
        self._image_tokens_cache: dict[str, int] = {}

    def _image_tokens_for_url(self, url: str) -> int:
        if url not in self._image_tokens_cache:
            path = url[len("file://") :] if url.startswith("file://") else url
            self._image_tokens_cache[url] = _image_token_count(path)
        return self._image_tokens_cache[url]

    def _length(self, text: str) -> int:
        if self._fast_file:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def count(self, content: str) -> int:
        """Prompt tokens of one user message after GLM-5.3-Flash template + image expansion."""
        extra_image_tokens = 0
        for match in re.finditer(r"<img>(.*?)</img>", content):
            extra_image_tokens += self._image_tokens_for_url(match.group(1)) - 1
        text = re.sub(
            r"<img>.*?</img>",
            "<|begin_of_image|><|image|><|end_of_image|>",
            content,
        )
        rendered = (
            f"[gMASK]<sop><|system|>Reasoning Effort: {self._reasoning_effort}"
            f"<|user|>{text}<|assistant|><think>"
        )
        return self._length(rendered) + extra_image_tokens

    def _pad(self, head: str, question: str, isl: int) -> str:
        """Appends ' .' up to isl tokens; an overshoot swaps the last ' .' for a shorter tail."""
        pad = " ." * max(0, isl - self.count(head + question) - _PAD_SLACK)
        if self.count(head + pad + question) >= isl:
            pad = ""
        while self.count(head + pad + question) < isl:
            pad += " ."
        if self.count(head + pad + question) > isl:
            tails = (".", " ", "\n", "")
            pad = next((pad[:-2] + t for t in tails if self.count(head + pad[:-2] + t + question) == isl), pad)
        return pad

    def build(
        self,
        articles: list[dict],
        image_dir: str,
        isl: int,
        shared_tokens: int,
        num_prompts: int,
        num_images: int = 1,
    ) -> tuple[list[str], int]:
        """Returns the prompts and the token count of their shared region."""
        all_paras = _extract_paragraphs(articles)
        selected_images = []
        if num_images > 0 and os.path.isdir(image_dir):
            for idx in range(num_images):
                name = IMAGE_NAMES[idx % len(IMAGE_NAMES)]
                full_path = os.path.join(image_dir, name)
                if os.path.isfile(full_path):
                    selected_images.append(full_path)

        shared = HEADER
        for img_path in selected_images:
            cand = shared + f"<img>file://{img_path}</img>\n\n"
            if self.count(cand) <= isl - 128:
                shared = cand

        max_shared_budget = max(self.count(shared), min(shared_tokens + _SHARED_SLACK_TOKENS, isl - 160))
        used = 0
        while used < len(all_paras) and self.count(shared) < max_shared_budget:
            para = all_paras[used]
            cand = shared + para + "\n\n"
            if self.count(cand) <= max_shared_budget:
                shared = cand
                used += 1
            else:
                words = para.split()
                lo, hi = 0, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    piece = " ".join(words[:mid]) + "\n\n"
                    if self.count(shared + piece) <= max_shared_budget:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo > 0:
                    shared += " ".join(words[:lo]) + "\n\n"
                used += 1
                break

        pool = all_paras[used:] or all_paras
        stride = max(1, len(pool) // max(num_prompts, 1))
        prompts = []
        for i in range(num_prompts):
            question = QUESTION % (i + 1)
            body = f"Additional document {i + 1}:\n\n"
            cursor = i * stride
            while self.count(shared + body + pool[cursor % len(pool)] + "\n\n" + question) <= isl - 8:
                body += pool[cursor % len(pool)] + "\n\n"
                cursor += 1
            rem_words = pool[cursor % len(pool)].split()
            lo, hi = 0, len(rem_words)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                piece = " ".join(rem_words[:mid]) + "\n\n"
                if self.count(shared + body + piece + question) <= isl - 4:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 0:
                body += " ".join(rem_words[:lo]) + "\n\n"
            pad = self._pad(shared + body, question, isl)
            prompts.append(shared + body + pad + question)

        actual_shared = max(0, self.count(shared) - _SHARED_SLACK_TOKENS)
        return prompts, actual_shared


def load_prompts(
    path: str,
    tokenizer_dir: str,
    data_dir: str,
    isl: int,
    shared_tokens: int,
    num_prompts: int,
    num_images: int = 1,
) -> dict:
    """Loads the prompt set from path, building and saving it first if needed."""
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if (
            cached.get("isl") == isl
            and cached.get("num_images", num_images) == num_images
            and len(cached.get("prompts", [])) >= num_prompts
        ):
            return cached
        logger.warning(f"{path} does not hold {num_prompts} prompts of {isl} tokens, rebuilding it")

    articles_path = os.path.join(data_dir, "articles.json")
    if os.path.isfile(articles_path):
        with open(articles_path) as f:
            articles = json.load(f)
    else:
        articles = [{"title": "Benchmark Reference Document", "paras": _FALLBACK_PARAS}]

    image_dir = os.path.abspath(os.path.join(data_dir, "images"))
    builder = PromptBuilder(tokenizer_dir)
    prompts, shared = builder.build(articles, image_dir, isl, shared_tokens, num_prompts, num_images=num_images)
    lengths = sorted({builder.count(p) for p in prompts})
    if lengths != [isl]:
        raise RuntimeError(f"prompt lengths {lengths} differ from {isl}")
    prompt_set = {
        "isl": isl,
        "shared": shared,
        "num_images": prompts[0].count("<img>") if prompts else 0,
        "lengths": lengths,
        "prompts": prompts,
    }
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompt_set, f, ensure_ascii=False)
    return prompt_set


def _content_to_openai_mm(content: str) -> list[dict] | str:
    """Converts inline <img>URL</img> markers into OpenAI multimodal content items."""
    if "<img>" not in content:
        return content
    parts = []
    pos = 0
    for match in re.finditer(r"<img>(.*?)</img>", content):
        if match.start() > pos:
            parts.append({"type": "text", "text": content[pos : match.start()]})
        parts.append({"type": "image_url", "image_url": {"url": match.group(1)}})
        pos = match.end()
    if pos < len(content):
        parts.append({"type": "text", "text": content[pos:]})
    return parts


def stream_chat(
    url: str,
    model: str,
    content: str,
    osl: int,
    request_id: str,
    timeout: float,
    inline_image_tags: bool = False,
) -> dict:
    """Sends one streaming chat request and times its first and last generated token chunks."""
    msg_content = content if inline_image_tags else _content_to_openai_mm(content)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": msg_content}],
            "max_tokens": osl,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": False},
            "request_id": request_id,
            "x_request_id": request_id,
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
    )
    start = time.perf_counter()
    first = last = None
    pieces = []
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = None
    try:
        response_cm = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    with response_cm as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens", cached_tokens)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    pieces.append(piece)
    if first is None:
        raise RuntimeError(f"no content streamed for {request_id}")
    return {
        "request_id": request_id,
        "ttft_ms": (first - start) * 1e3,
        "last_ms": (last - start) * 1e3,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens or osl,
        "text": "".join(pieces),
    }


def _request_stats(log_paths: list[str], prefix: str) -> dict:
    """Per-request prompt, prefix-cache and speculative token counts from the server logs."""
    fields = ("prompt_tokens", "num_prefix_cache_tokens", "speculative_accepted_tokens", "speculative_proposed_tokens")
    pattern = re.compile(rf"x-request-id: ({re.escape(prefix)}[^,]*),")
    stats = {}
    for path in log_paths:
        if not os.path.isfile(path):
            continue
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
    candidates = (
        sorted(glob.glob(os.path.join(log_dir, "rank_*.log")))
        + sorted(glob.glob(os.path.join(log_dir, "prefill*.log")))
        + sorted(glob.glob(os.path.join(log_dir, "decode*.log")))
    )
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    for path in candidates:
        with open(path, errors="replace") as f:
            for line in f:
                if _SERVER_ERRORS.search(line):
                    ts_match = _GLOG_TIMESTAMP.match(line)
                    if ts_match:
                        stamp = datetime.datetime.strptime(
                            ts_match.group(1) + " " + ts_match.group(2), "%Y%m%d %H:%M:%S.%f"
                        )
                        if stamp.timestamp() >= since_epoch:
                            errors += 1
                    else:
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
    if num_requests <= 0:
        return
    worker_count = max(1, min(concurrency, num_requests))
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

    threads = [threading.Thread(target=client) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))


def _master_burst(args: argparse.Namespace, request_prefix: str) -> list[float]:
    """Sends a small burst of tiny requests at once and returns their TTFTs."""
    ttfts = []
    burst_size = max(1, min(_MASTER_BURST, args.concurrency))

    def send(i: int) -> None:
        result = stream_chat(
            args.url,
            args.model,
            "Reply with OK.",
            1,
            f"{request_prefix}{i}",
            args.timeout,
            inline_image_tags=args.inline_image_tags,
        )
        ttfts.append(result["ttft_ms"])

    _closed_loop(burst_size, burst_size, send)
    return ttfts


def _warm_up(args: argparse.Namespace, prompts: list[str], tag: str) -> None:
    for burst in range(args.warm_master):
        ttfts = _master_burst(args, f"vl-{tag}-m{burst}-")
        logger.info(f"master warm-up burst {burst}: {len(ttfts)} ok, slowest TTFT {max(ttfts):.0f} ms")
    for start in range(0, args.warm, max(1, args.warm_concurrency)):
        wave = range(start, min(start + max(1, args.warm_concurrency), args.warm))

        def send(i: int, wave: range = wave) -> None:
            result = stream_chat(
                args.url,
                args.model,
                prompts[wave[i]],
                min(8, args.osl),
                f"vl-{tag}-w{wave[i]}",
                args.timeout,
                inline_image_tags=args.inline_image_tags,
            )
            logger.info(f"warm-up {wave[i]}: TTFT {result['ttft_ms']:.0f} ms")

        _closed_loop(len(wave), len(wave), send)
    if args.warm_closed:
        warm_pool = max(1, args.warm)
        start = time.perf_counter()
        _closed_loop(
            args.warm_closed,
            args.concurrency,
            lambda i: stream_chat(
                args.url,
                args.model,
                prompts[i % warm_pool],
                args.osl,
                f"vl-{tag}-wc{i}",
                args.timeout,
                inline_image_tags=args.inline_image_tags,
            ),
        )
        logger.info(
            f"closed-loop warm-up: {args.warm_closed} requests at C={args.concurrency} "
            f"in {time.perf_counter() - start:.1f} s"
        )


def _summarize(args: argparse.Namespace, results: list[dict], wall: float, epoch: float, tag: str) -> dict:
    ttft = [r["ttft_ms"] for r in results]
    steady = ttft[args.concurrency :] if len(ttft) > args.concurrency else ttft
    tpot = [(r["last_ms"] - r["ttft_ms"]) / max(1, r["completion_tokens"] - 1) for r in results]
    e2e = [r["last_ms"] for r in results]
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r.get("prompt_tokens") or args.isl for r in results)
    summary = {
        "concurrency": args.concurrency,
        "measured_requests": len(results),
        "isl": args.isl,
        "osl": args.osl,
        "wall_s": round(wall, 1),
        "measure_start_epoch": round(epoch, 3),
        "request_rate": round(len(results) / wall, 3),
        "output_token_throughput": round(total_completion_tokens / wall, 2),
        "total_token_throughput": round((total_prompt_tokens + total_completion_tokens) / wall, 2),
        "ttft_mean_ms": round(statistics.mean(ttft), 1),
        "ttft_p50_ms": round(_percentile(ttft, 50), 1),
        "ttft_p90_ms": round(_percentile(ttft, 90), 1),
        "ttft_p99_ms": round(_percentile(ttft, 99), 1),
        "ttft_max_ms": round(max(ttft), 1),
        "ttft_steady_mean_ms": round(statistics.mean(steady), 1),
        "tpot_mean_ms": round(statistics.mean(tpot), 2),
        "tpot_p50_ms": round(_percentile(tpot, 50), 2),
        "tpot_p90_ms": round(_percentile(tpot, 90), 2),
        "tpot_p99_ms": round(_percentile(tpot, 99), 2),
        "tpot_max_ms": round(max(tpot), 2),
        "e2e_mean_ms": round(statistics.mean(e2e), 1),
        "e2e_p50_ms": round(_percentile(e2e, 50), 1),
        "e2e_p90_ms": round(_percentile(e2e, 90), 1),
        "e2e_p99_ms": round(_percentile(e2e, 99), 1),
        "e2e_max_ms": round(max(e2e), 1),
    }
    if not args.server_log_dir or not os.path.isdir(args.server_log_dir):
        return summary

    prefill_logs = sorted(glob.glob(os.path.join(args.server_log_dir, "prefill*_r0.log")))
    decode_logs = sorted(glob.glob(os.path.join(args.server_log_dir, "decode*.log")))
    rank0_logs = sorted(glob.glob(os.path.join(args.server_log_dir, "rank_0.log")))
    prefill = _request_stats(prefill_logs or rank0_logs, f"vl-{tag}-c")
    decode = _request_stats(decode_logs or rank0_logs, f"vl-{tag}-c")

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
    parser.add_argument("--url", default="http://127.0.0.1:18901", help="service HTTP address")
    parser.add_argument("--model", default="glm-5.3-flash", help="model id to request")
    parser.add_argument("--tokenizer", required=True, help="model directory, for token counting")
    parser.add_argument("--data-dir", required=True, help="directory with articles.json and images/")
    parser.add_argument("--prompts", required=True, help="prompt set JSON, built if missing")
    parser.add_argument("--isl", type=int, default=2048, help="input sequence length in tokens (default: 2048)")
    parser.add_argument("--shared", type=int, default=1024, help="minimum shared-region tokens (default: 1024)")
    parser.add_argument("--osl", type=int, default=256, help="output sequence length in tokens (default: 256)")
    parser.add_argument("--num-images", type=int, default=1, help="number of images per prompt (default: 1)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=4, help="measured requests")
    parser.add_argument("--warm", type=int, default=1, help="sequential warm-up requests, distinct prompts")
    parser.add_argument(
        "--warm-concurrency", type=int, default=1, help="warm-up requests in flight at once during --warm"
    )
    parser.add_argument(
        "--warm-master",
        type=int,
        default=0,
        help="bursts of concurrent tiny requests run first",
    )
    parser.add_argument(
        "--warm-closed",
        type=int,
        default=0,
        help="closed-loop warm-up requests at --concurrency cycling the warm-up prompts",
    )
    parser.add_argument(
        "--inline-image-tags",
        action="store_true",
        help="send <img>file://...</img> inline string instead of OpenAI image_url content list",
    )
    parser.add_argument("--server-log-dir", default="", help="log directory of start_service.sh, for server stats")
    parser.add_argument("--out", default="", help="write the summary and per-request results here")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="only build or check the prompt set")
    args = parser.parse_args()

    total_needed = max(1, args.warm + args.requests)
    prompt_set = load_prompts(
        args.prompts,
        args.tokenizer,
        args.data_dir,
        args.isl,
        args.shared,
        total_needed,
        num_images=args.num_images,
    )
    prompts = prompt_set["prompts"][:total_needed]
    logger.info(
        f"{len(prompts)} prompts of {prompt_set['lengths']} tokens, shared region {prompt_set['shared']} tokens, "
        f"{prompts[0].count('<img>')} images each"
    )
    if args.dry_run:
        return

    tag = str(int(time.time()))
    _warm_up(args, prompts, tag)
    measured = prompts[args.warm : args.warm + args.requests]
    results = [None] * len(measured)
    start = time.perf_counter()
    epoch = time.time()

    def send(i: int) -> None:
        offset = time.perf_counter() - start
        results[i] = stream_chat(
            args.url,
            args.model,
            measured[i],
            args.osl,
            f"vl-{tag}-c{i}",
            args.timeout,
            inline_image_tags=args.inline_image_tags,
        )
        results[i]["start_s"] = offset

    _closed_loop(len(measured), args.concurrency, send)
    wall = time.perf_counter() - start
    time.sleep(1)
    for r in results:
        tpot = (r["last_ms"] - r["ttft_ms"]) / max(1, r["completion_tokens"] - 1)
        print(
            f"{r['request_id']} start={r['start_s']:.2f}s "
            f"prompt_tokens={r.get('prompt_tokens') or args.isl} "
            f"completion_tokens={r['completion_tokens']} "
            f"ttft={r['ttft_ms']:.0f}ms tpot={tpot:.2f}ms e2e={r['last_ms']:.0f}ms",
            flush=True,
        )
    summary = _summarize(args, results, wall, epoch, tag)
    if args.out:
        if os.path.dirname(args.out):
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "requests": results}, f, ensure_ascii=False, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
