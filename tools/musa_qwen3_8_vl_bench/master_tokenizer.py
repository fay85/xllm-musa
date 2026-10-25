"""Builds the tokenizer directory the xllm-service master loads.

The master tokenizes every chat request itself before routing it, and its
Jinja engine cannot parse the Qwen3.8 chat template, so /v1/chat/completions
fails with the model directory. This directory holds the model's tokenizer
files unchanged plus a ChatML template that renders a single user message
the way the instances do with enable_thinking=False. The master reads the
template from tokenizer_config.json first, so both places carry it.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.logger import logger

_COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "vocab.json",
)
CHAT_TEMPLATE = (
    r"{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\n'"
    r" + message['content'] + '<|im_end|>\n' }}{%- endfor %}"
    r"{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n"
    r"</think>\n\n' }}{%- endif %}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model checkpoint directory")
    parser.add_argument("--out", required=True, help="directory to write")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in _COPIED_FILES:
        shutil.copyfile(os.path.join(args.model, name), os.path.join(args.out, name))
    with open(os.path.join(args.model, "tokenizer_config.json")) as f:
        config = json.load(f)
    config["chat_template"] = CHAT_TEMPLATE
    with open(os.path.join(args.out, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "chat_template.jinja"), "w") as f:
        f.write(CHAT_TEMPLATE)
    logger.info(f"master tokenizer written to {args.out}")


if __name__ == "__main__":
    main()
