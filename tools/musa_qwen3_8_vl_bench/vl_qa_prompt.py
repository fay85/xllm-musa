# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Build image QA requests without dropping or changing image attachments."""

import re

_IMAGE = re.compile(r"<img>([^<>]+)</img>")


def image_uris(prompt: str) -> list[str]:
    """Return ordered URIs, rejecting incomplete or nested image tags."""
    matches = list(_IMAGE.finditer(prompt))
    remainder = _IMAGE.sub("", prompt)
    if re.search(r"</?img\b", remainder) or not matches:
        raise ValueError("Missing, malformed, or nested image tags")
    return [match.group(1) for match in matches]


def build_qa_prompt(prompt: str, question: str, expected_images: int = 5) -> str:
    """Replace only a trailing question; otherwise append after all images."""
    before = image_uris(prompt)
    if len(before) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(before)}")
    final_image_end = list(_IMAGE.finditer(prompt))[-1].end()
    question_start = prompt.rfind("Question:")
    head = prompt[:question_start] if question_start >= final_image_end else prompt
    content = head.rstrip() + "\n\n" + question
    if image_uris(content) != before:
        raise ValueError("QA construction changed image attachments or their order")
    return content
