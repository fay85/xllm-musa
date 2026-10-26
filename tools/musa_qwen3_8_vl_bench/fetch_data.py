"""Downloads the benchmark corpus into one directory.

- articles.json: the news articles listed in article_urls.txt (lite.cnn.com
  and text.npr.org), kept in that order because the prompt builder places
  each photo after a fixed article index. Articles that changed or vanished
  since the reference fetch are reported; the prompts built from the result
  still have 40,960 tokens and the same 80% shared prefix.
- images/: the Wikimedia Commons photos listed in images.json, cropped to
  16:9 and resized to 854x480, which the model encodes as 405 tokens each.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.logger import logger

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (854, 480)
# Wikimedia asks automated clients to identify themselves.
_COMMONS_AGENT = "xllm-vl-bench/1.0 (https://github.com/jd-opensource/xllm)"
_BROWSER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
_BOILERPLATE = re.compile(
    r"(see full web article|full cnn experience|all rights reserved|©|copyright"
    r"|npr\.org|sign up|subscribe|terms of use|privacy policy)",
    re.IGNORECASE,
)
_MIN_PARAGRAPH_CHARS = 60
_COMMONS_WIDTH = 1920
_ATTEMPTS = 3
# md5 of articles.json as fetched on 2026-10-22 for the published results.
REFERENCE_ARTICLES_MD5 = "adc5d434686157b0c7b7b244e2d5862e"


class _ArticleParser(HTMLParser):
    """Collects the <p> paragraphs and the first non-empty <h1>/<h2>."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.paragraphs: list[str] = []
        self._paragraph_depth = 0
        self._in_heading = False
        self._text: list[str] = []
        self._heading: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "p":
            self._paragraph_depth += 1
            self._text = []
        elif tag in ("h1", "h2") and not self.title:
            self._in_heading = True
            self._heading = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._paragraph_depth:
            self._paragraph_depth -= 1
            text = " ".join("".join(self._text).split())
            if text:
                self.paragraphs.append(text)
        elif tag in ("h1", "h2") and self._in_heading:
            self._in_heading = False
            self.title = " ".join("".join(self._heading).split())

    def handle_data(self, data: str) -> None:
        if self._paragraph_depth:
            self._text.append(data)
        if self._in_heading:
            self._heading.append(data)


def _download(url: str, agent: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError:
            raise
        except OSError as exc:
            if attempt == _ATTEMPTS:
                raise
            logger.warning(f"attempt {attempt} for {url} failed ({exc}), retrying")
            time.sleep(5 * attempt)


def _fetch_article(url: str) -> dict:
    body, charset = _download(url, _BROWSER_AGENT)
    parser = _ArticleParser()
    parser.feed(body.decode(charset, "replace"))
    paragraphs = [p for p in parser.paragraphs if len(p) >= _MIN_PARAGRAPH_CHARS and not _BOILERPLATE.search(p)]
    return {
        "url": url,
        "title": parser.title,
        "paras": paragraphs,
        "words": sum(len(p.split()) for p in paragraphs),
    }


def _fetch_articles(url_file: str, out_path: str) -> None:
    """Fetches every "URL WORDS" line of url_file; WORDS is the reference length."""
    with open(url_file) as f:
        entries = [line.split() for line in f if line.strip()]
    articles = []
    changed = missing = 0
    for url, reference_words in entries:
        try:
            article = _fetch_article(url)
        except OSError as exc:
            logger.warning(f"skipping {url}: {exc}")
            missing += 1
            continue
        if not article["paras"]:
            logger.warning(f"skipping {url}: no paragraphs parsed")
            missing += 1
            continue
        if article["words"] != int(reference_words):
            logger.warning(f"{url} has {article['words']} words, {reference_words} in the reference fetch")
            changed += 1
        logger.info(f"{article['words']:5d} words  {url}")
        articles.append(article)
        time.sleep(0.3)
    with open(out_path, "w") as f:
        json.dump(articles, f, ensure_ascii=False, indent=1)
    with open(out_path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    if digest == REFERENCE_ARTICLES_MD5:
        logger.info("articles.json matches the reference fetch")
    else:
        logger.warning(
            f"articles.json differs from the reference fetch ({changed} articles changed, {missing} missing): "
            "prompts will differ from the reference set but keep their length and 80% shared prefix"
        )
    total_words = sum(a["words"] for a in articles)
    logger.info(f"wrote {len(articles)} articles ({total_words} words) to {out_path}")


def _crop_16_9(image: Image.Image, horizontal: str, vertical: str) -> Image.Image:
    width, height = image.size
    if width * 9 > height * 16:
        crop_width, crop_height = height * 16 // 9, height
    else:
        crop_width, crop_height = width, width * 9 // 16
    left = {"left": 0, "center": (width - crop_width) // 2, "right": width - crop_width}
    top = {"top": 0, "center": (height - crop_height) // 2, "bottom": height - crop_height}
    x, y = left[horizontal], top[vertical]
    return image.crop((x, y, x + crop_width, y + crop_height))


def _fetch_images(manifest: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest) as f:
        entries = json.load(f)
    for entry in entries:
        url = (
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            f"{urllib.parse.quote(entry['commons_file'])}?width={_COMMONS_WIDTH}"
        )
        body, _ = _download(url, _COMMONS_AGENT)
        source = Image.open(io.BytesIO(body)).convert("RGB")
        image = _crop_16_9(source, *entry["crop"]).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        path = os.path.join(out_dir, entry["file"])
        image.save(path, quality=90)
        logger.info(f"{entry['file']}: source {source.size} -> {image.size} ({entry['license']}, {entry['author']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="data directory to fill")
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    articles = os.path.join(args.out, "articles.json")
    if args.force or not os.path.exists(articles):
        _fetch_articles(os.path.join(KIT_DIR, "article_urls.txt"), articles)
    manifest = os.path.join(KIT_DIR, "images.json")
    with open(manifest) as f:
        names = [entry["file"] for entry in json.load(f)]
    images = os.path.join(args.out, "images")
    if args.force or not all(os.path.exists(os.path.join(images, n)) for n in names):
        try:
            _fetch_images(manifest, images)
        except OSError as exc:
            logger.error(
                f"cannot download the photos ({exc}); run fetch_data.py on a machine that reaches "
                f"commons.wikimedia.org and copy its images/ directory to {images}"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
