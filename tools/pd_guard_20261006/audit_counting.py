#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Check frozen 30K input, full output and throughput from raw records."""
import hashlib
import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(sys.argv[1])
    output = root / sys.argv[2]
    if output.exists() or output.parent != root:
        raise ValueError("A new root-level output filename is required")
    dataset_path = root / "logs/dspark_30k_1k_cases_20260926/cases_pd_model_id.jsonl"
    dataset = {row["id"]: row for row in
               (json.loads(line) for line in dataset_path.read_text().splitlines())}
    reports = []
    for specification in sys.argv[3:]:
        label, count_text = specification.split(":")
        count = int(count_text)
        raw = root / "logs" / label / "responses.raw.jsonl"
        summary_path = raw.parent / "summary.json"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        summary = json.loads(summary_path.read_text())
        failures = []
        total_tokens = 0
        for row in rows:
            source = dataset[row["id"]]
            expected = [str(value) for value in range(source["count_from"], source["count_to"] + 1)]
            usage = row.get("usage", {})
            passed = (len(expected) == 512 and row.get("generation", "").split() == expected
                      and row.get("request_payload") == source["request_payload"]
                      and row.get("http_status") == 200 and not row.get("error")
                      and row.get("finish_reason") == "length"
                      and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0
                      and usage.get("prompt_tokens") == 30000 and usage.get("completion_tokens") == 1024)
            if not passed:
                failures.append(row["id"])
            total_tokens += usage.get("completion_tokens", 0)
        throughput = total_tokens / summary["duration_seconds"]
        summary_ok = (summary["complete"] and summary["attempted"] == count
                      and len(rows) == count and len({row["id"] for row in rows}) == count
                      and summary["strict_counting_passes"] == count and summary["request_errors"] == 0
                      and not summary["input_usage_mismatches"] and not summary["output_usage_mismatches"]
                      and abs(throughput - summary["output_token_throughput"]) < 1e-6)
        reports.append(dict(label=label, passed=not failures and summary_ok, failed_ids=failures,
                            count=len(rows), actual_output_tokens=total_tokens, output_tok_s=throughput,
                            ttft_ms=summary["ttft_avg_ms"], tpot_ms=summary["tpot_avg_ms"],
                            raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                            summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest()))
    passed = bool(reports) and all(report["passed"] for report in reports)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(dict(passed=passed, reports=reports,
                       dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                       scope="16 GPU PD aggregate, full client window, actual cached=0"),
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(reports))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
