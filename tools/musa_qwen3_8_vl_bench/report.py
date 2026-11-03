"""Prints a markdown table from vl_bench.py --out result files."""

import argparse
import json


def _cell(summary: dict, key: str, fmt: str) -> str:
    value = summary.get(key)
    return "-" if value is None else format(value, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="JSON files written by vl_bench.py --out")
    args = parser.parse_args()

    summaries = []
    for path in args.results:
        with open(path) as f:
            summaries.append(json.load(f)["summary"])
    columns = (
        ("C", "concurrency", "d"),
        ("TTFT mean (ms)", "ttft_mean_ms", ".0f"),
        ("TTFT p50", "ttft_p50_ms", ".0f"),
        ("TTFT p90", "ttft_p90_ms", ".0f"),
        ("TTFT p99", "ttft_p99_ms", ".0f"),
        ("TTFT steady", "ttft_steady_mean_ms", ".0f"),
        ("TPOT mean (ms)", "tpot_mean_ms", ".2f"),
        ("TPOT p90", "tpot_p90_ms", ".2f"),
        ("QPS", "request_rate", ".2f"),
        ("KV hit", "kv_hit_rate", ".2%"),
        ("MTP accept", "mtp_acceptance", ".3f"),
        ("captures", "lazy_captures_during_measurement", "d"),
        ("errors", "server_errors", "d"),
    )
    print("| " + " | ".join(title for title, _, _ in columns) + " |")
    print("|" + "---|" * len(columns))
    for summary in sorted(summaries, key=lambda s: s["concurrency"]):
        print("| " + " | ".join(_cell(summary, key, fmt) for _, key, fmt in columns) + " |")


if __name__ == "__main__":
    main()
