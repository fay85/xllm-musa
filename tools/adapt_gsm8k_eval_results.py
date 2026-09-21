#!/usr/bin/env python3
"""Adapt the single-solver evaluator JSONL to the reference-free audit schema."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append(
            {
                "dataset_index": item["dataset_index"],
                "question": item.get("question", ""),
                "prediction": item.get("prediction"),
                "attempts": [
                    {
                        "stage": "single-solver",
                        "content": item.get("content", ""),
                    }
                ],
                "selected_stage": "single-solver",
                "selection_uses_reference": False,
            }
        )
    Path(args.output).write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
