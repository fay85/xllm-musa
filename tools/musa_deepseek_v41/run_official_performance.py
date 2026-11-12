# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged EvalScope perf with fixed requests and native usage checks."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from tokenizers import Tokenizer


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evalscope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if not (
        gate["status"] == "PASS"
        and len(gate["questions"]) == 2
        and all(row["passed"] for row in gate["questions"])
        and gate["binary_sha256"] == _sha(args.binary)
        and gate["endpoint"] == args.url
    ):
        raise ValueError("Require successful same-binary, same-endpoint basic gate")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "basic_gate.json").write_text(json.dumps(gate, indent=2))
    tokenizer = Tokenizer.from_file(str(args.tokenizer_directory / "tokenizer.json"))
    commands = []
    identity = {}
    for length in (256, 2048):
        source = args.inputs / f"input{length}_output256.jsonl"
        requests = [json.loads(line) for line in source.read_text().splitlines()]
        if len(requests) < 5:
            raise ValueError("Need five fixed requests per measured group")
        for body in requests:
            prompt = body["messages"][0]["content"]
            rendered = (
                "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜></think>"
            )
            actual = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
            if actual != length or body["model"] != args.model:
                raise ValueError(f"Request identity/token count mismatch: {actual}")
            if not (
                body["max_tokens"] == 256
                and body["ignore_eos"] is True
                and body["temperature"] == 0
                and body["top_p"] == 1
                and body["reasoning_effort"] == "none"
                and body["chat_template_kwargs"] == {"thinking_mode": "chat"}
            ):
                raise ValueError("Sampling protocol differs from fixed baseline")
        identity[str(length)] = {"sha256": _sha(source), "requests": len(requests)}
        for phase, count in (("warmup", 1), ("measured", 5)):
            out = args.output / f"input{length}_output256" / phase
            cmd = [
                str(args.evalscope),
                "perf",
                "--api",
                "openai",
                "--model",
                args.model,
                "--url",
                args.url.rstrip("/") + "/v1/chat/completions",
                "--dataset",
                "line_by_line",
                "--dataset-path",
                str(source),
                "--number",
                str(count),
                "--parallel",
                "1",
                "--num-workers",
                "1",
                "--db-commit-interval",
                "1",
                "--log-every-n-query",
                "1",
                "--warmup-num",
                "0",
                "--tokenizer-path",
                str(args.tokenizer_directory),
                "--stream",
                "--outputs-dir",
                str(out),
                "--no-timestamp",
                "--no-test-connection",
                "--connect-timeout",
                "20",
                "--read-timeout",
                "1800",
                "--total-timeout",
                "3600",
            ]
            commands.append(
                {
                    "input_tokens": length,
                    "phase": phase,
                    "count": count,
                    "directory": str(out),
                    "command": cmd,
                }
            )
    (args.output / "identity.json").write_text(
        json.dumps(
            {
                "binary_sha256": _sha(args.binary),
                "datasets": identity,
                "tokenizer_sha256": _sha(args.tokenizer_directory / "tokenizer.json"),
                "protocol": "C1 eager, each group one warmup then five measured, ignore_eos256",
            },
            indent=2,
        )
    )
    (args.output / "commands.json").write_text(json.dumps(commands, indent=2))
    results = []
    env = dict(
        os.environ,
        TOKENIZERS_PARALLELISM="false",
        LD_PRELOAD="",
        PYTHONPATH="",
        MUSA_VISIBLE_DEVICES="-1",
    )
    for item in commands:
        started = _now()
        (args.output / "active.json").write_text(
            json.dumps({"started_at": started, **item}, indent=2)
        )
        log_path = args.output / f"{item['phase']}_input{item['input_tokens']}.log"
        logger.info(
            "EvalScope %s input%s x%s",
            item["phase"],
            item["input_tokens"],
            item["count"],
        )
        with log_path.open("xb") as stream:
            result = subprocess.run(
                item["command"], stdout=stream, stderr=subprocess.STDOUT, env=env
            )
        record = {key: value for key, value in item.items() if key != "command"}
        record.update(
            started_at=started, finished_at=_now(), exit_code=result.returncode
        )
        results.append(record)
        (args.output / "status.json").write_text(json.dumps(results, indent=2))
        if result.returncode:
            raise RuntimeError(f"EvalScope failed: {log_path}")
        databases = list(Path(item["directory"]).rglob("benchmark_data.db"))
        if len(databases) != 1:
            raise ValueError(f"Expected one benchmark database: {databases}")
        with sqlite3.connect("file:" + str(databases[0]) + "?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT success,prompt_tokens,completion_tokens,latency,first_chunk_latency FROM result"
            ).fetchall()
        if not (
            len(rows) == item["count"]
            and all(
                row[0]
                and row[1] == item["input_tokens"]
                and row[2] == 256
                and row[3] > row[4] > 0
                for row in rows
            )
        ):
            raise ValueError(f"Stop after invalid native usage/results: {rows}")
        record["validated_rows"] = rows
        (args.output / "status.json").write_text(json.dumps(results, indent=2))
    (args.output / "client_runs_finished.json").write_text(
        json.dumps(
            {
                "status": "PASS_ALL_FOUR_PHASES_EXACT_NATIVE_TOKEN_COUNTS",
                "finished_at": _now(),
                "note": "Warmups excluded. Raw SQLite, streaming responses, command and request identities retained.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
