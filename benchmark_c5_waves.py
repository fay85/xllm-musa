#!/usr/bin/env python3
"""Analyze explicit C=5 barrier waves by internal prefill partition.

This analyzer intentionally accepts only JSON produced by
``benchmark_c5_barrier_waves.py``. The older Joblib client replenishes worker
slots independently, so every five submission indices are not guaranteed to
be one arrival wave and must not be used for partition-conditioned scoring.

Use a dedicated server log that starts before the barrier client's warmup and
contains no other traffic. The analyzer requires the number of logged waves to
match ``warmup_waves + num_waves`` exactly; it never tail-aligns ambiguous logs
or infers scheduler partitions from TTFT tiers.

Example::

    python3 benchmark_c5_waves.py \
      --run xllm /results/xllm.json /results/xllm.server.log \
      --run sglang /results/sglang.json /results/sglang.server.log
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections.abc import Sequence


DEFAULT_PARTITIONS = (
    "5",
    "1+4",
    "2+3",
    "3+2",
    "4+1",
    "1+1+3",
    "1+2+2",
    "1+3+1",
    "2+1+2",
    "2+2+1",
    "3+1+1",
    "1+1+1+2",
    "1+1+2+1",
    "1+2+1+1",
    "2+1+1+1",
    "1+1+1+1+1",
)


@dataclasses.dataclass(frozen=True)
class RunSpec:
    engine: str
    result_json: pathlib.Path
    server_log: pathlib.Path

    @property
    def label(self) -> str:
        parent = self.result_json.parent
        if parent.name == "results":
            parent = parent.parent
            return parent.name
        return f"{parent.name}:{self.result_json.stem}"


@dataclasses.dataclass(frozen=True)
class PrefillGroup:
    size: int
    active_before: int


@dataclasses.dataclass(frozen=True)
class ResultWave:
    index: int
    values_ms: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class Wave:
    engine: str
    run_label: str
    index: int
    partition: str
    values_ms: tuple[float, ...]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.values_ms)

    @property
    def sorted_values_ms(self) -> tuple[float, ...]:
        return tuple(sorted(self.values_ms))


@dataclasses.dataclass(frozen=True)
class LoadedResult:
    wave_size: int
    warmup_waves: int
    waves: tuple[ResultWave, ...]
    comparison_metadata: tuple[tuple[str, object], ...]


@dataclasses.dataclass(frozen=True)
class Summary:
    count: int
    mean: float
    median: float
    stdev: float
    ci_low: float
    ci_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        required=True,
        metavar=("ENGINE", "RESULT_JSON", "SERVER_LOG"),
        help="Add a dedicated barrier-wave run for xLLM or SGLang.",
    )
    parser.add_argument(
        "--metric",
        choices=("barrier_ttft_ms", "ttft_ms"),
        default="barrier_ttft_ms",
        help="Per-request metric used in wave summaries.",
    )
    parser.add_argument(
        "--min-prefill-tokens",
        type=int,
        default=2000,
        help="Filter readiness probes; valid runs use prefix cache off.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--partitions",
        default=",".join(DEFAULT_PARTITIONS),
        help="Ordered strata eligible for the common-strata score.",
    )
    parser.add_argument(
        "--min-common-waves",
        type=int,
        default=10,
        help="Minimum waves per engine required for a scored stratum.",
    )
    return parser.parse_args()


def normalize_engine(value: str) -> str:
    lowered = value.lower()
    if "sglang" in lowered:
        return "sglang"
    if "xllm" in lowered:
        return "xllm"
    raise ValueError(f"unsupported engine {value!r}; expected xllm or sglang")


def finite_nonnegative(value: object, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return number


def load_barrier_result(path: pathlib.Path, metric: str) -> LoadedResult:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("protocol") != "closed_loop_barrier_waves":
        raise ValueError(
            f"{path}: unsupported protocol {result.get('protocol')!r}; "
            "use benchmark_c5_barrier_waves.py instead of the Joblib client"
        )

    wave_size = int(result.get("wave_size", 0))
    warmup_waves = int(result.get("warmup_waves", -1))
    num_waves = int(result.get("num_waves", 0))
    records = result.get("requests")
    if wave_size <= 0 or warmup_waves < 0 or num_waves <= 0:
        raise ValueError(f"{path}: invalid barrier-wave metadata")
    expected_measured = num_waves * wave_size
    expected_warmup = warmup_waves * wave_size
    if int(result.get("completed", -1)) != expected_measured:
        raise ValueError(
            f"{path}: completed does not equal {num_waves} * {wave_size}"
        )
    if int(result.get("warmup_completed", -1)) != expected_warmup:
        raise ValueError(
            f"{path}: warmup_completed does not equal "
            f"{warmup_waves} * {wave_size}"
        )
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: missing non-empty requests array")

    grouped: dict[int, list[dict[str, object]]] = collections.defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path}: malformed request record")
        if record.get("success") is not True:
            raise ValueError(f"{path}: incomplete wave contains a failed request")
        wave_index = int(record.get("wave_index", -1))
        if wave_index < 0:
            raise ValueError(f"{path}: invalid wave_index")
        grouped[wave_index].append(record)

    expected_indices = list(range(num_waves))
    if sorted(grouped) != expected_indices:
        raise ValueError(
            f"{path}: wave indices {sorted(grouped)} do not match "
            f"0..{num_waves - 1}"
        )

    waves: list[ResultWave] = []
    for wave_index in expected_indices:
        records_in_wave = sorted(
            grouped[wave_index], key=lambda record: int(record["order_in_wave"])
        )
        if len(records_in_wave) != wave_size:
            raise ValueError(
                f"{path}: wave {wave_index} has {len(records_in_wave)} "
                f"requests, expected {wave_size}"
            )
        orders = [int(record["order_in_wave"]) for record in records_in_wave]
        if orders != list(range(wave_size)):
            raise ValueError(f"{path}: wave {wave_index} has invalid request order")
        values = tuple(
            finite_nonnegative(
                record.get(metric), f"{path}: wave {wave_index} {metric}"
            )
            for record in records_in_wave
        )
        waves.append(ResultWave(index=wave_index, values_ms=values))

    reported_key = (
        "mean_barrier_ttft_ms" if metric == "barrier_ttft_ms" else "mean_ttft_ms"
    )
    reported = result.get(reported_key)
    calculated = statistics.fmean(
        value for wave in waves for value in wave.values_ms
    )
    if reported is not None and not math.isclose(
        calculated, float(reported), rel_tol=1e-7, abs_tol=1e-3
    ):
        raise ValueError(
            f"{path}: request {metric} mean {calculated:.6f} does not match "
            f"{reported_key}={float(reported):.6f}"
        )
    metadata_keys = (
        "protocol",
        "backend",
        "model_id",
        "tokenizer_id",
        "input_len",
        "output_len",
        "prefix_len",
        "wave_size",
        "warmup_waves",
        "seed",
        "warmup_seed",
        "temperature",
        "top_k",
        "top_p",
    )
    missing_metadata = [key for key in metadata_keys if key not in result]
    if missing_metadata:
        raise ValueError(
            f"{path}: missing comparison metadata: "
            + ", ".join(missing_metadata)
        )
    return LoadedResult(
        wave_size=wave_size,
        warmup_waves=warmup_waves,
        waves=tuple(waves),
        comparison_metadata=tuple(
            (key, result[key]) for key in metadata_keys
        )
        + (
            (
                "requested_release_partition",
                tuple(result.get("requested_release_partition", ())),
            ),
            (
                "inter_group_delay_ms",
                float(result.get("inter_group_delay_ms", 0.0)),
            ),
            (
                "inter_wave_delay_ms",
                float(result.get("inter_wave_delay_ms", 0.0)),
            ),
        ),
    )


def read_log_lines(path: pathlib.Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"scheduler log does not exist: {path}")
    return (
        path.read_text(encoding="utf-8", errors="replace")
        .replace("\0", "")
        .splitlines()
    )


def parse_sglang_prefill_groups(
    lines: Sequence[str], min_prefill_tokens: int
) -> list[PrefillGroup]:
    pattern = re.compile(
        r"Prefill batch, #new-seq: (?P<seqs>\d+), "
        r"#new-token: (?P<tokens>\d+), #cached-token: (?P<cached>\d+)"
    )
    groups: list[PrefillGroup] = []
    for line in lines:
        match = pattern.search(line)
        if not match or int(match.group("tokens")) < min_prefill_tokens:
            continue
        if int(match.group("cached")) != 0:
            raise ValueError(
                "SGLang scheduler log contains cached prefill tokens; rerun "
                "the fairness benchmark with radix/prefix cache disabled"
            )
        group_size = int(match.group("seqs"))
        if group_size > 0:
            running = re.search(r"#running-req: (\d+)", line)
            if running is None:
                raise ValueError("SGLang prefill line lacks #running-req")
            groups.append(
                PrefillGroup(
                    size=group_size,
                    active_before=int(running.group(1)),
                )
            )
    return groups


def parse_xllm_prefill_groups(
    lines: Sequence[str], min_prefill_tokens: int
) -> list[PrefillGroup]:
    groups: list[PrefillGroup] = []
    for line in lines:
        if "[SCHED_PACK]" not in line:
            continue
        prefill = re.search(r"\bn_prefill=(\d+)", line)
        chunked = re.search(r"\bn_chunked=(\d+)", line)
        decode = re.search(r"\bn_decode=(\d+)", line)
        tokens = re.search(r"\btoken_budget=(\d+)", line)
        running = re.search(r"\brunning_q=(\d+)", line)
        if not all((prefill, chunked, decode, tokens, running)):
            raise ValueError("malformed xLLM [SCHED_PACK] line")
        if int(tokens.group(1)) < min_prefill_tokens:
            continue
        if int(chunked.group(1)) != 0:
            raise ValueError(
                "xLLM scheduler log contains chunked prefills; request IDs "
                "are required before those can be classified safely"
            )
        if int(decode.group(1)) != 0:
            raise ValueError(
                "xLLM prefill log contains a mixed decode batch; rerun with "
                "the required graph-on homogeneous scheduler fast path"
            )
        group_size = int(prefill.group(1))
        if group_size > 0:
            groups.append(
                PrefillGroup(
                    size=group_size,
                    active_before=int(running.group(1)),
                )
            )
    return groups


def groups_to_partitions(
    groups: Sequence[PrefillGroup], wave_size: int
) -> list[str]:
    partitions: list[str] = []
    current: list[int] = []
    current_sum = 0
    for index, group in enumerate(groups):
        group_size = group.size
        if group_size <= 0 or group_size > wave_size:
            raise ValueError(f"invalid prefill group size {group_size} at {index}")
        if current_sum + group_size > wave_size:
            raise ValueError(
                "scheduler groups cross a wave boundary; the log is missing "
                "a group or contains unrelated traffic"
            )
        if group.active_before != current_sum:
            raise ValueError(
                f"scheduler group {index} reports {group.active_before} "
                f"active request(s), expected {current_sum}; the log is not "
                "an isolated barrier-wave run"
            )
        current.append(group_size)
        current_sum += group_size
        if current_sum == wave_size:
            partitions.append("+".join(str(value) for value in current))
            current = []
            current_sum = 0
    if current:
        raise ValueError(f"scheduler log ends with partial wave {current}")
    return partitions


def classify_run(
    spec: RunSpec, metric: str, min_prefill_tokens: int
) -> tuple[list[Wave], tuple[tuple[str, object], ...]]:
    result = load_barrier_result(spec.result_json, metric)
    lines = read_log_lines(spec.server_log)
    if spec.engine == "sglang":
        groups = parse_sglang_prefill_groups(lines, min_prefill_tokens)
    else:
        groups = parse_xllm_prefill_groups(lines, min_prefill_tokens)
    partitions = groups_to_partitions(groups, result.wave_size)
    expected = result.warmup_waves + len(result.waves)
    if len(partitions) != expected:
        raise ValueError(
            f"{spec.server_log}: found {len(partitions)} complete scheduler "
            f"waves, expected exactly {expected} (warmup + measurement)"
        )
    measured_partitions = partitions[result.warmup_waves :]
    waves = [
        Wave(
            engine=spec.engine,
            run_label=spec.label,
            index=result_wave.index,
            partition=partition,
            values_ms=result_wave.values_ms,
        )
        for result_wave, partition in zip(result.waves, measured_partitions)
    ]
    return waves, result.comparison_metadata


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def summarize(
    waves: Sequence[Wave], bootstrap_samples: int, seed: int
) -> Summary:
    if not waves:
        raise ValueError("cannot summarize an empty wave sample")
    values = [wave.mean_ms for wave in waves]
    by_run: dict[str, list[float]] = collections.defaultdict(list)
    for wave in waves:
        by_run[wave.run_label].append(wave.mean_ms)

    if len(values) == 1 or bootstrap_samples <= 0:
        ci_low = ci_high = values[0]
    else:
        rng = random.Random(seed)
        run_labels = sorted(by_run)
        bootstrap_means: list[float] = []
        for _ in range(bootstrap_samples):
            sampled_values: list[float] = []
            for run_label in rng.choices(run_labels, k=len(run_labels)):
                run_values = by_run[run_label]
                sampled_values.extend(
                    rng.choices(run_values, k=len(run_values))
                )
            bootstrap_means.append(statistics.fmean(sampled_values))
        bootstrap_means.sort()
        ci_low = percentile(bootstrap_means, 0.025)
        ci_high = percentile(bootstrap_means, 0.975)

    return Summary(
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def aggregate_waves(
    runs: Sequence[tuple[RunSpec, list[Wave]]]
) -> dict[str, dict[str, list[Wave]]]:
    aggregate: dict[str, dict[str, list[Wave]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for _, waves in runs:
        for wave in waves:
            aggregate[wave.engine][wave.partition].append(wave)
    return aggregate


def format_counts(waves: Sequence[Wave]) -> str:
    counts = collections.Counter(wave.partition for wave in waves)
    return ",".join(
        f"{partition}:{counts[partition]}" for partition in sorted(counts)
    )


def print_run_summary(runs: Sequence[tuple[RunSpec, list[Wave]]]) -> None:
    print("\nPer-run explicit-wave summary")
    print("engine  run                                      waves  mean_ms  partitions")
    for spec, waves in runs:
        print(
            f"{spec.engine:<7} {spec.label:<40.40} {len(waves):>5} "
            f"{statistics.fmean(wave.mean_ms for wave in waves):>8.2f}  "
            f"{format_counts(waves)}"
        )


def print_strata(
    aggregate: dict[str, dict[str, list[Wave]]],
    bootstrap_samples: int,
    seed: int,
) -> None:
    print("\nPartition-conditioned statistics (wave is the sampling unit)")
    print(
        "engine  partition  n   mean_ms  median_ms    sd_ms       "
        "hierarchical_bootstrap_95%_CI       sorted_request_means_ms"
    )
    for engine in sorted(aggregate):
        for partition in sorted(aggregate[engine]):
            waves = aggregate[engine][partition]
            summary = summarize(
                waves,
                bootstrap_samples,
                stable_seed(seed, f"{engine}:{partition}"),
            )
            sorted_means = [
                statistics.fmean(wave.sorted_values_ms[index] for wave in waves)
                for index in range(len(waves[0].values_ms))
            ]
            positions = ",".join(f"{value:.1f}" for value in sorted_means)
            print(
                f"{engine:<7} {partition:<10} {summary.count:>2} "
                f"{summary.mean:>9.2f} {summary.median:>10.2f} "
                f"{summary.stdev:>8.2f} "
                f"[{summary.ci_low:>8.2f}, {summary.ci_high:>8.2f}]  "
                f"{positions}"
            )


def print_common_comparison(
    aggregate: dict[str, dict[str, list[Wave]]],
    partitions: Sequence[str],
    min_waves: int,
) -> None:
    engines = sorted(aggregate)
    print("\nUniform common-strata comparison")
    if len(engines) != 2:
        print(f"Need exactly two engines; found {engines}.")
        return
    eligible = [
        partition
        for partition in partitions
        if all(
            len(aggregate[engine].get(partition, ())) >= min_waves
            for engine in engines
        )
    ]
    if not eligible:
        print(
            f"No requested partition has at least {min_waves} explicit "
            "waves for both engines. Keep collecting; no score is reported."
        )
        return
    weight = 1.0 / len(eligible)
    print(
        "eligible=" + ",".join(eligible) + f"; uniform_weight={weight:.6f}"
    )
    scores: dict[str, float] = {}
    for engine in engines:
        scores[engine] = sum(
            weight
            * statistics.fmean(
                wave.mean_ms for wave in aggregate[engine][partition]
            )
            for partition in eligible
        )
        print(f"{engine:<7} weighted_mean_ms={scores[engine]:.2f}")
    baseline, contender = engines
    print(
        f"delta({contender}-{baseline})_ms="
        f"{scores[contender] - scores[baseline]:+.2f}"
    )
    print(
        "formula: score = mean_p(mean_wave(mean_request(metric))); only "
        "partitions meeting the per-engine sample threshold are included"
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be nonnegative")
    if args.min_common_waves <= 0:
        raise ValueError("--min-common-waves must be positive")
    if args.min_prefill_tokens <= 0:
        raise ValueError("--min-prefill-tokens must be positive")

    specs = [
        RunSpec(
            engine=normalize_engine(engine),
            result_json=pathlib.Path(result_json),
            server_log=pathlib.Path(server_log),
        )
        for engine, result_json, server_log in args.run
    ]
    classified = [
        classify_run(
            spec,
            metric=args.metric,
            min_prefill_tokens=args.min_prefill_tokens,
        )
        for spec in specs
    ]
    reference_metadata = classified[0][1]
    for spec, (_, metadata) in zip(specs[1:], classified[1:]):
        if metadata != reference_metadata:
            reference = dict(reference_metadata)
            candidate = dict(metadata)
            mismatches = [
                key
                for key in reference
                if reference[key] != candidate.get(key)
            ]
            raise ValueError(
                f"{spec.result_json}: comparison metadata mismatch: "
                + ", ".join(
                    f"{key}={candidate.get(key)!r} "
                    f"(expected {reference[key]!r})"
                    for key in mismatches
                )
            )
    runs = [
        (spec, waves)
        for spec, (waves, _) in zip(specs, classified)
    ]
    print(f"metric={args.metric}")
    print_run_summary(runs)
    aggregate = aggregate_waves(runs)
    print_strata(aggregate, args.bootstrap_samples, args.seed)
    partitions = [value for value in args.partitions.split(",") if value]
    print_common_comparison(
        aggregate,
        partitions=partitions,
        min_waves=args.min_common_waves,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
