# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay the archived C1 or C16/C24/C48 workload against a running 1P1D pair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import statistics
from threading import Barrier
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "workload.json").read_text())


def _records(run: Path) -> dict:
    output = {}
    for host in ("sj9", "sj10"):
        with (run / host / "rank0.log").open(errors="replace") as log:
            for line in log:
                if "record_type: " not in line:
                    continue
                fields = dict(part.split(": ", 1)
                              for part in line.split("] ", 1)[-1].strip().split(", ")
                              if ": " in part)
                for rid in {fields.get("request_id"), fields.get("x-request-id")} - {None}:
                    output.setdefault(rid, {})[(host, line)] = fields
    return output


def _audit(row: dict, records: dict) -> bool:
    native = {}
    ids = set(row["ids"]) | {row["rid"]} | set(row["headers"].values())
    for rid in ids:
        native.update(records.get(rid, {}))
    row["native"] = [dict(host=host, **fields) for (host, _), fields in native.items()]
    for host, event in (("sj9", "pd_prefill_stage"), ("sj10", "sequence_terminal"),
                        ("sj10", "request_terminal")):
        selected = [fields for (origin, _), fields in native.items()
                    if origin == host and fields.get("record_type") == event]
        if len(selected) != 1:
            return False
        fields = selected[0]
        if event == "request_terminal":
            if fields.get("outcome") != "success":
                return False
        elif (fields.get("prompt_tokens") != str(row["expected_input"])
              or fields.get("num_prefix_cache_tokens") != "0"):
            return False
        if event == "sequence_terminal" and (
                fields.get("generated_tokens") != str(row["expected_output"])
                or fields.get("finish_reason") != "length"):
            return False
    return True


def _request(endpoint: str, prompt: str, expected: dict, tokens: int,
             output: Path, index: int, warm: bool) -> dict:
    url = urlsplit(endpoint)
    connection_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(url.hostname, url.port, timeout=1800)
    rid = str(uuid4())
    row = dict(rid=rid, warm=warm, prompt_sha256=expected["sha"],
               expected_input=expected["input_cpu"], expected_output=tokens)
    payload = json.dumps(dict(model="deepseek-v4.1-flash-1017-w4afp8-mt", prompt=prompt,
                              best_of=1, max_tokens=tokens, stream=True, temperature=0,
                              ignore_eos=True), ensure_ascii=False, separators=(",", ":")).encode()
    first = last = None
    ids, finishes = set(), []
    reasoning = False
    done = 0
    done_last = False
    stage = output / f"{index:03}-{rid}"
    try:
        with stage.with_suffix(".sse.jsonl").open("x") as stream:
            started = time.perf_counter()
            connection.request("POST", "/v1/completions", payload,
                               headers={"Content-Type": "application/json", "X-Request-ID": rid})
            response = connection.getresponse()
            row["status"] = response.status
            row["headers"] = {k.lower(): v for k, v in response.getheaders()
                              if k.lower() in ("x-request-id", "x-request-uid", "x-trace-id")}
            if response.status != 200:
                row["error"] = response.read(8192).decode(errors="replace")
            else:
                for chunk in response:
                    now = time.perf_counter()
                    line = chunk.decode().rstrip("\r\n")
                    if line:
                        stream.write(json.dumps(dict(ms=(now - started) * 1000, line=line)) + "\n")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    last = now
                    done_last = payload == "[DONE]"
                    if done_last:
                        done += 1
                        continue
                    frame = json.loads(payload)
                    if frame.get("id"):
                        ids.add(frame["id"])
                    for choice in frame.get("choices", []):
                        if choice.get("text") and first is None:
                            first = now
                        if choice.get("finish_reason"):
                            finishes.append(choice["finish_reason"])
                        reasoning |= bool(choice.get("reasoning_content")
                                          or (choice.get("delta") or {}).get("reasoning_content"))
    except (OSError, ValueError, http.client.HTTPException) as error:
        row["error"] = repr(error)
    finally:
        connection.close()
    row.update(ids=sorted(ids), done=done, done_last=done_last, finish=finishes, reasoning=reasoning)
    if first is not None and last is not None:
        row.update(ttft_ms=(first - started) * 1000, e2e_ms=(last - started) * 1000)
        if tokens > 1:
            row["tpot_ms"] = (last - first) * 1000 / (tokens - 1)
    stage.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def _ready(endpoint: str, run: Path) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        directories = [run / "sj9", run / "sj10"]
        if any((directory / "result.json").exists() for directory in directories):
            raise RuntimeError("A role exited before the benchmark")
        for directory in directories:
            rank_zero = directory / "rank0.log"
            if len(list(directory.glob("rank*.log"))) != 8 or not rank_zero.is_file():
                break
            with rank_zero.open("rb") as log:
                log.seek(max(0, rank_zero.stat().st_size - 131072))
                if b"Application startup complete" not in log.read():
                    break
        else:
            try:
                with urlopen(endpoint + "/readyz", timeout=3) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
        time.sleep(3)
    raise TimeoutError("P, D and Router did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("G32PEXPM256", "G32DRINGC1"))
    parser.add_argument("--endpoint", required=True, help="Router HTTP URL without an endpoint path")
    parser.add_argument("--run", type=Path, required=True, help="Shared role run directory")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, choices=(1, 1000), default=1000)
    args = parser.parse_args()
    if args.case == "G32DRINGC1" and args.output_tokens != 1000:
        parser.error("The ramp workload requires 1000 output tokens")
    endpoint = args.endpoint.rstrip("/")
    for directory, role in (("sj9", "p"), ("sj10", "d")):
        config = json.loads((args.run / directory / "prepared_config.json").read_text())
        if config.get("case") != args.case or config.get("run_role") != role:
            raise ValueError(f"{directory} is not running the requested configuration")
    cases = {"G32PEXPM256": {"c1": 2}, "G32DRINGC1": {"c16": 16, "c24": 24, "c48": 48}}
    counts = cases[args.case]
    inputs = SPEC["rows"][:max(counts.values())]
    prompts = []
    for row in inputs:
        prompt = (args.inputs / f"{row['i']:03}.txt").read_bytes()
        if hashlib.sha256(prompt).hexdigest() != row["sha"]:
            raise ValueError(f"Input hash mismatch for prompt {row['i']}")
        prompts.append(prompt.decode())
    _ready(endpoint, args.run)
    args.output.mkdir()
    summaries = {}
    for name, count in counts.items():
        stage = args.output / name
        stage.mkdir()

        def request(sequence: int, index: int, warm: bool = False) -> dict:
            return _request(endpoint, prompts[index], inputs[index], args.output_tokens,
                            stage, sequence, warm)

        if name == "c1":
            rows = []
            for sequence, index in enumerate((0, 0, 1)):
                result = request(sequence, index, sequence == 0)
                rows.append(result)
                if result.get("status") != 200 or result.get("done") != 1 or "ttft_ms" not in result:
                    break
        else:
            barrier = Barrier(count)

            def offer(index: int) -> dict:
                barrier.wait(timeout=25)
                if index:
                    time.sleep(SPEC["offset_seconds"][index])
                return request(index, index)

            with ThreadPoolExecutor(max_workers=count) as executor:
                rows = list(executor.map(offer, range(count)))
        complete = len(rows) == count + (name == "c1")
        for row in rows:
            row["sse_valid"] = (row.get("status") == 200 and len(row["ids"]) == 1
                                and row["finish"] == ["length"] and row["done"] == 1
                                and row["done_last"] and not row["reasoning"]
                                and "ttft_ms" in row and "error" not in row)
        for attempt in range(14):
            records = _records(args.run)
            for row in rows:
                row["audited"] = row["sse_valid"] and _audit(row, records)
            if all(row["audited"] for row in rows) or any(not row["sse_valid"] for row in rows):
                break
            if attempt < 13:
                time.sleep(4)
        (stage / "audited.json").write_text(json.dumps(rows, indent=2) + "\n")
        if not complete or not all(row["audited"] for row in rows):
            raise RuntimeError(f"{name} failed length, terminal or SSE audit. See {stage}")
        measured = [row for row in rows if not row["warm"]]
        summary = {"requests": len(measured)}
        for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
            if key in measured[0]:
                summary[key] = statistics.mean(row[key] for row in measured)
        summaries[name] = summary
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
