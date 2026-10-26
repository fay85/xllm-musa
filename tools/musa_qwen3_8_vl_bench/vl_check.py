"""VL correctness check against the running stack.

Sends the last ten benchmark prompts, five at a time, with the question
replaced by one asking for the five images in order, and checks that each
answer names the subject of every image in the order the images appear.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vl_bench import stream_chat

from scripts.logger import logger

KEYWORDS = {
    "01_us_capitol.jpg": ("capitol",),
    "02_supreme_court.jpg": ("supreme court",),
    "03_kremlin.jpg": ("kremlin",),
    "04_oil_tanker.jpg": ("tanker", "ship"),
    "05_budget_airline.jpg": ("ryanair", "airplane", "aircraft", "plane"),
}
QUESTION = (
    "Question: This conversation contains five images. List them in order of appearance, "
    "one short line each, saying what each image shows."
)
_NUM_QUESTIONS = 10
_CONCURRENCY = 5


def _ask(args: argparse.Namespace, prompt: str, index: int) -> tuple[bool, float, str]:
    order = [m.split("/")[-1] for m in re.findall(r"<img>([^<]+)</img>", prompt)]
    content = prompt[: prompt.rfind("Question:")] + QUESTION
    result = stream_chat(args.url, args.model, content, 120, f"vl-qa{index}-{int(time.time())}", args.timeout)
    # ignore_eos keeps the model generating past its answer into made-up turns.
    answer = re.split(r"user|assistant", result["text"])[0].lower()
    position = 0
    for name in order:
        hits = [answer.find(k, position) for k in KEYWORDS[name] if answer.find(k, position) >= 0]
        if not hits:
            return False, result["ttft_ms"], answer
        position = min(hits) + 1
    return True, result["ttft_ms"], answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="master address, e.g. http://HOST:64888")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8", help="model id to request")
    parser.add_argument("--prompts", required=True, help="prompt set built by vl_bench.py")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)["prompts"]
    passed = 0
    with concurrent.futures.ThreadPoolExecutor(_CONCURRENCY) as pool:
        for start in range(0, _NUM_QUESTIONS, _CONCURRENCY):
            indices = range(start, start + _CONCURRENCY)
            answers = pool.map(lambda i: _ask(args, prompts[-1 - i], i), indices)
            for index, (ok, ttft, answer) in zip(indices, answers):
                passed += ok
                text = answer.strip().replace("\n", " | ")[:300]
                print(f"{'PASS' if ok else 'FAIL'} q{index} ttft={ttft:.0f}ms: {text}", flush=True)
    print(f"VL_QA {passed}/{_NUM_QUESTIONS} passed", flush=True)
    if passed != _NUM_QUESTIONS:
        logger.error("some answers miss an image or list them out of order")
        sys.exit(1)


if __name__ == "__main__":
    main()
