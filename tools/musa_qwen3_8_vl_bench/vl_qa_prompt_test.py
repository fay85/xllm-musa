# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU regressions for the VL QA image-preserving input constructor."""

import argparse
import hashlib
import json
import unittest
from pathlib import Path

from vl_qa_prompt import build_qa_prompt, image_uris

IMAGES = "".join(f"<img>file:///image{i}.jpg</img>" for i in range(5))
QUESTION = "Question: Describe all five images in order."


class ImagePromptTest(unittest.TestCase):
    def test_missing_question_preserves_final_closing_tag(self) -> None:
        self.assertEqual(build_qa_prompt(IMAGES, QUESTION), IMAGES + "\n\n" + QUESTION)

    def test_trailing_question_replaced(self) -> None:
        self.assertEqual(
            build_qa_prompt(IMAGES + "\nQuestion: old", QUESTION),
            IMAGES + "\n\n" + QUESTION,
        )

    def test_question_in_body_not_removed(self) -> None:
        prompt = "Question: inside the article\n" + IMAGES
        self.assertTrue(build_qa_prompt(prompt, QUESTION).startswith(prompt))

    def test_incomplete_image_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_qa_prompt(IMAGES[:-1], QUESTION)

    def test_missing_image_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_qa_prompt(IMAGES.replace("<img>file:///image4.jpg</img>", ""), QUESTION)

    def test_nested_image_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_qa_prompt("<img>" + IMAGES + "</img>", QUESTION)

    def test_question_cannot_add_an_image(self) -> None:
        with self.assertRaises(ValueError):
            build_qa_prompt(IMAGES, QUESTION + "<img>file:///extra.jpg</img>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ImagePromptTest))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.prompts:
        prompts = json.loads(args.prompts.read_text())["prompts"]
        for prompt in prompts:
            content = build_qa_prompt(prompt, QUESTION)
            assert image_uris(content) == image_uris(prompt)
            assert len(image_uris(content)) == 5
        print(
            json.dumps(
                dict(
                    frozen_prompts_passed=len(prompts),
                    sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                )
            )
        )


if __name__ == "__main__":
    main()
