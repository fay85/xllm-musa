#!/usr/bin/env python3
"""20k text plus five 480p images, with separate text and image KV hits.

Linear-state checkpoints on this stack are 8192 tokens apart, and a prompt of
this length holds two of them. Two images sit in the shared prefix; a second
group of prompts puts one more image inside the second checkpoint. Across the
measured set that is a 48% image hit and a 77% text hit.
"""

import json
import os
import re

from transformers import AutoTokenizer

IMAGE_NAMES = (
    "01_us_capitol.jpg",
    "03_kremlin.jpg",
    "02_supreme_court.jpg",
    "04_oil_tanker.jpg",
    "05_budget_airline.jpg",
)
IMAGE_COST = 407
TEXT_TOKENS = 20000
CHECKPOINT = 8192
CACHED_CHECKPOINTS = 2
SLACK = 96
# Measured requests: 36 prompts cache 2 images, 24 cache 3.
MEASURED_FAMILY_A = 36
MEASURED_FAMILY_B = 24
WARM = 8

TOKENIZER = "/data/nfs_shared/models/Qwen3.8-27B-FP8"
DATA_DIR = "/workspace/xllm_vl_bench_ref"
OUT = "/root/xllm_vl_bench/vl_prompts_text20k_img50.json"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def count(tokenizer, content: str) -> int:
    num_images = content.count("<img>")
    text = re.sub(r"<img>.*?</img>", "<|vision_start|><|image_pad|><|vision_end|>", content)
    rendered = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) + num_images * (IMAGE_COST - 3)


def image_tag(image_dir: str, name: str) -> str:
    return f"<img>file://{image_dir}/{name}</img>"


def add_text(tokenizer, content: str, n_tokens: int) -> str:
    if n_tokens < 0:
        raise RuntimeError(f"negative text budget {n_tokens}")
    if n_tokens == 0:
        return content
    blob = " ." * n_tokens
    got = count(tokenizer, content + blob) - count(tokenizer, content)
    while got < n_tokens:
        blob += " ."
        got += 1
    while got > n_tokens and blob.endswith(" ."):
        blob = blob[:-2]
        got -= 1
    final = count(tokenizer, content + blob) - count(tokenizer, content)
    if final != n_tokens:
        raise RuntimeError(f"text pad landed on {final}, wanted {n_tokens}")
    return content + blob


def article_blob(data_dir: str) -> str:
    with open(os.path.join(data_dir, "articles.json")) as handle:
        articles = json.load(handle)
    parts = []
    for article in articles:
        paras = article.get("paras") or []
        if not isinstance(paras, list):
            paras = [str(paras)]
        parts.append(article.get("title", ""))
        parts.extend(str(p) for p in paras if str(p).strip())
    return "\n\n".join(parts) + "\n\n"


def main() -> None:
    tokenizer = load_tokenizer()
    image_dir = os.path.abspath(os.path.join(DATA_DIR, "images"))
    images = [image_tag(image_dir, name) for name in IMAGE_NAMES]
    for name in IMAGE_NAMES:
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"missing image {path}")

    prefix_end = CACHED_CHECKPOINTS * CHECKPOINT + SLACK
    common_end = CHECKPOINT + SLACK
    common_text = common_end - 2 * IMAGE_COST
    header = (
        "Below is a collection of news articles, some illustrated with photos, followed by an additional "
        "document. Read everything carefully and answer the question at the end.\n\n"
    )
    base = count(tokenizer, header)
    if base >= common_text:
        raise RuntimeError(f"header is {base} tokens, common text budget is {common_text}")
    common = add_text(tokenizer, header, common_text - base)
    common += images[0] + images[1]
    if count(tokenizer, common) != common_end:
        raise RuntimeError(f"common prefix is {count(tokenizer, common)}, wanted {common_end}")

    family_a = add_text(tokenizer, common, prefix_end - count(tokenizer, common))
    family_b = common + images[2]
    family_b = add_text(tokenizer, family_b, prefix_end - count(tokenizer, family_b))
    if count(tokenizer, family_a) != prefix_end or count(tokenizer, family_b) != prefix_end:
        raise RuntimeError(
            f"family prefixes {count(tokenizer, family_a)} {count(tokenizer, family_b)}, wanted {prefix_end}"
        )

    text_at_a = prefix_end - 2 * IMAGE_COST
    text_at_b = prefix_end - 3 * IMAGE_COST
    unique_text_a = TEXT_TOKENS - text_at_a
    unique_text_b = TEXT_TOKENS - text_at_b
    isl = TEXT_TOKENS + 5 * IMAGE_COST

    def finish(prefix: str, doc_id: int, n_text: int, tail_images: list[str]) -> str:
        marker = f"\nAdditional document {doc_id} says the following. "
        marked = prefix + marker
        marker_tokens = count(tokenizer, marked) - count(tokenizer, prefix)
        if marker_tokens >= n_text:
            raise RuntimeError(f"marker uses {marker_tokens} tokens, tail budget is {n_text}")
        body = add_text(tokenizer, marked, n_text - marker_tokens)
        for tag in tail_images:
            body += tag
        got = count(tokenizer, body)
        if got != isl:
            raise RuntimeError(f"prompt {doc_id} is {got} tokens, wanted {isl}")
        return body

    warm_pattern = ["A", "B", "A", "B", "A", "B", "A", "B"]
    measured_pattern = (["A", "A", "A", "B", "B"] * 12)
    if measured_pattern.count("A") != MEASURED_FAMILY_A or measured_pattern.count("B") != MEASURED_FAMILY_B:
        raise RuntimeError("measured family mix is not 36/24")
    pattern = warm_pattern + measured_pattern

    prompts = []
    families = []
    for doc_id, family in enumerate(pattern, start=1):
        if family == "A":
            prompts.append(finish(family_a, doc_id, unique_text_a, images[2:]))
        else:
            prompts.append(finish(family_b, doc_id, unique_text_b, images[3:]))
        families.append(family)
        print(f"built {doc_id}/{len(pattern)} family {family}", flush=True)

    # Both images sit at the end of the common prefix, still inside the resumed
    # 16384 tokens. The 96-token slack after that checkpoint is not cached.
    cached_span = CACHED_CHECKPOINTS * CHECKPOINT
    common_text_tokens = common_end - 2 * IMAGE_COST
    family_cached_tokens = cached_span - common_end
    cached_text = {
        "A": common_text_tokens + family_cached_tokens,
        "B": common_text_tokens + family_cached_tokens - IMAGE_COST,
    }
    cached_images = {"A": 2, "B": 3}
    measured = families[WARM:]
    avg_text = sum(cached_text[f] for f in measured) / len(measured) / TEXT_TOKENS
    avg_images = sum(cached_images[f] for f in measured) / len(measured) / 5
    print(f"text hit {avg_text:.4f} image hit {avg_images:.4f} isl {isl}")

    payload = {
        "isl": isl,
        "shared": cached_span,
        "lengths": [isl],
        "text_tokens": TEXT_TOKENS,
        "image_cost": IMAGE_COST,
        "cached_text": cached_text,
        "cached_images": cached_images,
        "families": families,
        "warm": WARM,
        "expected_text_hit": round(avg_text, 4),
        "expected_image_hit": round(avg_images, 4),
        "prompts": prompts,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
