#!/usr/bin/env python3
"""Run deterministic, closed-loop waves with the official serving client.

Each wave submits ``--wave-size`` request threads before a common barrier and
waits for every full streaming response before releasing the next wave.  By
default all threads leave the barrier together.  ``--release-partition`` can
instead release staged subgroups, such as ``1+2+2``, separated by
``--inter-group-delay-ms``.  The random prompt generator, tokenizer setup,
request object, and OpenAI-chat request function are imported from
``/data/feihu/benchmark_serving_parallel.py`` and
``/data/feihu/backend_request_func_parallel.py``.

This controls client-side arrivals, but it does not force an engine's internal
prefill partition (for example, 1+4 versus 1+2+2).  Pair this result with the
server scheduler log when partition-conditioned results are needed.  Waiting
for every decode to finish also makes this a barrier-wave workload, not the
slot-refilling Joblib workload produced by the official client's
``--request-rate 5`` implementation.

Example for the current C=5 protocol::

    python3 benchmark_c5_barrier_waves.py \
      --host 127.0.0.1 --port 8092 \
      --model Qwen3.5-27B-FP8 \
      --tokenizer /workspace/model_weights/Qwen3.5-27B-FP8 \
      --input-len 2500 --output-len 1500 \
      --wave-size 5 --warmup-waves 1 --num-waves 10 \
      --result-json /tmp/qwen35_fp8_c5_barrier.json

Example with staged client arrivals::

    python3 benchmark_c5_barrier_waves.py \
      --host 127.0.0.1 --port 8092 \
      --model Qwen3.5-27B-FP8 \
      --tokenizer /workspace/model_weights/Qwen3.5-27B-FP8 \
      --input-len 2500 --output-len 1500 \
      --wave-size 5 --release-partition 1+2+2 \
      --inter-group-delay-ms 10 \
      --warmup-waves 1 --num-waves 10 \
      --result-json /tmp/qwen35_fp8_c5_staged.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import math
import os
import pathlib
import random
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any


DEFAULT_BENCHMARK_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class TimedRequest:
    """One request result with common-barrier timing metadata."""

    wave_index: int
    order_in_wave: int
    completion_order_in_wave: int
    release_at: float
    started_at: float
    finished_at: float
    output: Any


def parse_release_partition(value: str) -> tuple[int, ...]:
    """Parse a client release partition such as ``1+2+2``."""

    if not value:
        return ()
    try:
        partition = tuple(int(group_size) for group_size in value.split("+"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "release partition must contain positive integers joined by '+'"
        ) from error
    if not partition or any(group_size <= 0 for group_size in partition):
        raise argparse.ArgumentTypeError(
            "release partition must contain positive integers joined by '+'"
        )
    return partition


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer path or ID; defaults to --model.",
    )
    parser.add_argument("--input-len", type=int, default=2500)
    parser.add_argument("--output-len", type=int, default=1500)
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=200,
        help="Shared random-prompt prefix, matching the official client default.",
    )
    parser.add_argument("--wave-size", type=int, default=5)
    parser.add_argument(
        "--release-partition",
        type=parse_release_partition,
        default=(),
        metavar="GROUPS",
        help=(
            "Optional staged client release partition, for example 1+2+2; "
            "empty releases the complete wave simultaneously."
        ),
    )
    parser.add_argument(
        "--inter-group-delay-ms",
        type=float,
        default=0.0,
        help="Delay between consecutive staged release groups in milliseconds.",
    )
    parser.add_argument(
        "--inter-wave-delay-ms",
        type=float,
        default=0.0,
        help="Drain delay after one complete wave before releasing the next.",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        default=10,
        help="Number of measured waves.",
    )
    parser.add_argument(
        "--warmup-waves",
        type=int,
        default=1,
        help="Warmup waves discarded from aggregate metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44002,
        help="Measured-prompt seed, matching the current official protocol.",
    )
    parser.add_argument(
        "--warmup-seed",
        type=int,
        help="Warmup-prompt seed; defaults to --seed minus one.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--result-json",
        type=pathlib.Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--benchmark-lib-dir",
        type=pathlib.Path,
        default=DEFAULT_BENCHMARK_LIB_DIR,
        help="Directory containing the two official benchmark Python modules.",
    )
    args = parser.parse_args(argv)

    if args.wave_size <= 0:
        parser.error("--wave-size must be positive")
    if args.release_partition and sum(args.release_partition) != args.wave_size:
        parser.error("sum(--release-partition) must equal --wave-size")
    if args.inter_group_delay_ms < 0:
        parser.error("--inter-group-delay-ms cannot be negative")
    if args.inter_wave_delay_ms < 0:
        parser.error("--inter-wave-delay-ms cannot be negative")
    if args.num_waves <= 0:
        parser.error("--num-waves must be positive")
    if args.warmup_waves < 0:
        parser.error("--warmup-waves cannot be negative")
    if args.input_len <= 0 or args.output_len <= 0:
        parser.error("--input-len and --output-len must be positive")
    if not 0 <= args.prefix_len <= args.input_len:
        parser.error("--prefix-len must be in [0, --input-len]")
    if not args.endpoint.endswith("/chat/completions"):
        parser.error("--endpoint must end with /chat/completions")
    return args


def load_official_modules(directory: pathlib.Path) -> tuple[Any, Any]:
    """Import the official prompt and request implementations in-place."""

    benchmark_path = directory / "benchmark_serving_parallel.py"
    backend_path = directory / "backend_request_func_parallel.py"
    missing = [
        str(path)
        for path in (benchmark_path, backend_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "official benchmark module(s) not found: " + ", ".join(missing)
        )

    directory_text = str(directory.resolve())
    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)

    import backend_request_func_parallel as backend
    import benchmark_serving_parallel as benchmark

    return benchmark, backend


def execute_barrier_wave(
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    request_func: Callable[[Any], Any],
    request_inputs: Sequence[Any],
    failed_output_factory: Callable[[Any, Exception], Any],
    wave_index: int,
    release_partition: Sequence[int] = (),
    inter_group_delay_ms: float = 0.0,
) -> list[TimedRequest]:
    """Release one complete wave, returning results in submission order."""

    partition = tuple(release_partition) or (len(request_inputs),)
    if sum(partition) != len(request_inputs):
        raise ValueError("release partition must cover the complete wave")

    release_group_by_order: list[int] = []
    for group_index, group_size in enumerate(partition):
        release_group_by_order.extend([group_index] * group_size)

    release_state: dict[str, float] = {}
    completion_lock = threading.Lock()
    completion_count = 0

    def mark_release() -> None:
        release_state["at"] = time.perf_counter()

    barrier = threading.Barrier(len(request_inputs) + 1, action=mark_release)

    def invoke(order_in_wave: int, request_input: Any) -> TimedRequest:
        nonlocal completion_count
        barrier.wait()
        scheduled_release_at = release_state["at"] + (
            release_group_by_order[order_in_wave]
            * inter_group_delay_ms
            / 1000.0
        )
        remaining_delay = scheduled_release_at - time.perf_counter()
        if remaining_delay > 0:
            time.sleep(remaining_delay)
        started_at = time.perf_counter()
        try:
            output = request_func(request_input)
        except Exception as error:  # Preserve a result JSON for failed runs.
            output = failed_output_factory(request_input, error)
        finished_at = time.perf_counter()
        with completion_lock:
            completion_order = completion_count
            completion_count += 1
        return TimedRequest(
            wave_index=wave_index,
            order_in_wave=order_in_wave,
            completion_order_in_wave=completion_order,
            release_at=release_state["at"],
            started_at=started_at,
            finished_at=finished_at,
            output=output,
        )

    futures = [
        executor.submit(invoke, order, request_input)
        for order, request_input in enumerate(request_inputs)
    ]
    barrier.wait()
    return [future.result() for future in futures]


def generate_requests(
    *,
    benchmark: Any,
    tokenizer: Any,
    seed: int,
    count: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
) -> list[tuple[str, int, int]]:
    """Generate exactly the official random dataset for one benchmark arm."""

    random.seed(seed)
    benchmark.np.random.seed(seed)
    return benchmark.sample_rand_requests(
        input_len=input_len,
        num_requests=count,
        max_output_len=output_len,
        tokenizer=tokenizer,
        prefix_len=prefix_len,
    )


def create_request_inputs(
    *,
    backend: Any,
    requests: Sequence[tuple[str, int, int]],
    model: str,
    api_url: str,
) -> list[Any]:
    return [
        backend.RequestFuncInput(
            model=model,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        for prompt, prompt_len, output_len in requests
    ]


def run_waves(
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    request_func: Callable[[Any], Any],
    request_inputs: Sequence[Any],
    failed_output_factory: Callable[[Any, Exception], Any],
    wave_size: int,
    release_partition: Sequence[int] = (),
    inter_group_delay_ms: float = 0.0,
    inter_wave_delay_ms: float = 0.0,
) -> list[TimedRequest]:
    timed_results: list[TimedRequest] = []
    for offset in range(0, len(request_inputs), wave_size):
        wave_index = offset // wave_size
        timed_results.extend(
            execute_barrier_wave(
                executor=executor,
                request_func=request_func,
                request_inputs=request_inputs[offset : offset + wave_size],
                failed_output_factory=failed_output_factory,
                wave_index=wave_index,
                release_partition=release_partition,
                inter_group_delay_ms=inter_group_delay_ms,
            )
        )
        if offset + wave_size < len(request_inputs) and inter_wave_delay_ms > 0:
            time.sleep(inter_wave_delay_ms / 1000.0)
    return timed_results


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def metric_dict(metrics: Any) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {}
    for field in dataclasses.fields(metrics):
        value = getattr(metrics, field.name)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            value = finite_or_none(value)
        result[field.name] = value
    return result


def request_record(
    timed: TimedRequest,
    *,
    measurement_started_at: float,
    wave_size: int,
    actual_output_tokens: int,
) -> dict[str, Any]:
    output = timed.output
    generated_tokens = actual_output_tokens
    tpot_ms = None
    if generated_tokens > 1:
        tpot_ms = (
            (output.latency - output.ttft) / (generated_tokens - 1) * 1000.0
        )
    barrier_ttft_ms = None
    if output.success:
        barrier_ttft_ms = (
            timed.started_at - timed.release_at + output.ttft
        ) * 1000.0
    return {
        "request_index": timed.wave_index * wave_size + timed.order_in_wave,
        "wave_index": timed.wave_index,
        "order_in_wave": timed.order_in_wave,
        "completion_order_in_wave": timed.completion_order_in_wave,
        "success": bool(output.success),
        "input_tokens": int(output.prompt_len),
        "output_tokens": generated_tokens,
        "release_offset_ms": (timed.release_at - measurement_started_at) * 1000.0,
        "request_start_offset_ms": (timed.started_at - timed.release_at) * 1000.0,
        "completion_offset_ms": (timed.finished_at - timed.release_at) * 1000.0,
        "latency_ms": output.latency * 1000.0 if output.success else None,
        "ttft_ms": output.ttft * 1000.0 if output.success else None,
        "barrier_ttft_ms": barrier_ttft_ms,
        "tpot_ms": tpot_ms,
        "error": output.error,
    }


def summarize_waves(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    wave_indices = sorted({int(record["wave_index"]) for record in records})
    summaries: list[dict[str, Any]] = []
    for wave_index in wave_indices:
        wave = [record for record in records if record["wave_index"] == wave_index]
        successful = [record for record in wave if record["success"]]
        summaries.append(
            {
                "wave_index": wave_index,
                "request_indices": [record["request_index"] for record in wave],
                "completion_order": [
                    record["order_in_wave"]
                    for record in sorted(
                        wave, key=lambda item: item["completion_order_in_wave"]
                    )
                ],
                "completed": len(successful),
                "duration_ms": max(record["completion_offset_ms"] for record in wave),
                "mean_ttft_ms": (
                    statistics.fmean(record["ttft_ms"] for record in successful)
                    if successful
                    else None
                ),
                "mean_barrier_ttft_ms": (
                    statistics.fmean(
                        record["barrier_ttft_ms"] for record in successful
                    )
                    if successful
                    else None
                ),
            }
        )
    return summaries


def print_summary(result: dict[str, Any]) -> None:
    print("=" * 50)
    print(" Barrier-Wave Serving Benchmark Result ".center(50, "="))
    rows = (
        ("Successful requests", result["completed"]),
        ("Benchmark duration (s)", f"{result['duration']:.2f}"),
        ("Total input tokens", result["total_input_tokens"]),
        ("Total generated tokens", result["total_output_tokens"]),
        ("Request throughput (req/s)", f"{result['request_throughput']:.2f}"),
        ("Input token throughput (tok/s)", f"{result['input_throughput']:.2f}"),
        ("Output token throughput (tok/s)", f"{result['output_throughput']:.2f}"),
        ("Total TGS, prefill+decode (tok/s)", f"{result['total_tgs']:.2f}"),
        ("Mean latency (ms)", f"{result['mean_latency_ms']:.2f}"),
        ("Mean TTFT (ms)", f"{result['mean_ttft_ms']:.2f}"),
        ("Mean barrier-aligned TTFT (ms)", f"{result['mean_barrier_ttft_ms']:.2f}"),
        ("Mean TPOT (ms)", f"{result['mean_tpot_ms']:.2f}"),
    )
    for label, value in rows:
        print(f"{label:<40} {value}")
    print("=" * 50)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark, backend = load_official_modules(args.benchmark_lib_dir)
    tokenizer_id = args.tokenizer or args.model
    if "joyai" in f"{args.model} {tokenizer_id}".lower():
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(
            tokenizer_id,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        tokenizer = benchmark.AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=args.trust_remote_code,
        )

    warmup_seed = args.warmup_seed
    if warmup_seed is None:
        warmup_seed = args.seed - 1
    warmup_requests = generate_requests(
        benchmark=benchmark,
        tokenizer=tokenizer,
        seed=warmup_seed,
        count=args.warmup_waves * args.wave_size,
        input_len=args.input_len,
        output_len=args.output_len,
        prefix_len=args.prefix_len,
    )
    measured_requests = generate_requests(
        benchmark=benchmark,
        tokenizer=tokenizer,
        seed=args.seed,
        count=args.num_waves * args.wave_size,
        input_len=args.input_len,
        output_len=args.output_len,
        prefix_len=args.prefix_len,
    )

    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    request_func = backend.ASYNC_REQUEST_FUNCS["openai-chat"]

    def failed_output(request_input: Any, error: Exception) -> Any:
        return backend.RequestFuncOutput(
            success=False,
            prompt_len=request_input.prompt_len,
            error=f"{type(error).__name__}: {error}",
        )

    warmup_inputs = create_request_inputs(
        backend=backend,
        requests=warmup_requests,
        model=args.model,
        api_url=api_url,
    )
    measured_inputs = create_request_inputs(
        backend=backend,
        requests=measured_requests,
        model=args.model,
        api_url=api_url,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.wave_size,
        thread_name_prefix="barrier-wave",
    ) as executor:
        warmup_results = run_waves(
            executor=executor,
            request_func=request_func,
            request_inputs=warmup_inputs,
            failed_output_factory=failed_output,
            wave_size=args.wave_size,
            release_partition=args.release_partition,
            inter_group_delay_ms=args.inter_group_delay_ms,
            inter_wave_delay_ms=args.inter_wave_delay_ms,
        )
        if warmup_results and measured_inputs and args.inter_wave_delay_ms > 0:
            time.sleep(args.inter_wave_delay_ms / 1000.0)
        measurement_started_at = time.perf_counter()
        measured_results = run_waves(
            executor=executor,
            request_func=request_func,
            request_inputs=measured_inputs,
            failed_output_factory=failed_output,
            wave_size=args.wave_size,
            release_partition=args.release_partition,
            inter_group_delay_ms=args.inter_group_delay_ms,
            inter_wave_delay_ms=args.inter_wave_delay_ms,
        )
        duration = time.perf_counter() - measurement_started_at

    outputs = [timed.output for timed in measured_results]
    completed = sum(output.success for output in outputs)
    if completed:
        metrics, actual_output_lens = benchmark.calculate_metrics(
            input_requests=measured_requests,
            outputs=outputs,
            dur_s=duration,
            tokenizer=tokenizer,
        )
        result: dict[str, Any] = metric_dict(metrics)
        # The official chat client sets ignore_eos=true, so every successful
        # request generates its requested output length.  MTP can stream two
        # accepted tokens in one SSE chunk; counting chunks therefore
        # under-reports output tokens and over-reports TPOT.
        actual_output_lens = [
            request[2] if output.success else 0
            for request, output in zip(measured_requests, outputs)
        ]
        successful_output_lens = [
            output_len
            for output_len, output in zip(actual_output_lens, outputs)
            if output.success
        ]
        tpots = [
            (output.latency - output.ttft) / (output_len - 1)
            for output_len, output in zip(actual_output_lens, outputs)
            if output.success and output_len > 1
        ]
        result["total_output"] = sum(actual_output_lens)
        result["mean_output_len"] = statistics.fmean(successful_output_lens)
        result["median_output_len"] = statistics.median(
            successful_output_lens
        )
        result["max_output_len"] = max(successful_output_lens)
        result["output_throughput"] = sum(actual_output_lens) / duration
        result["mean_tpot_ms"] = statistics.fmean(tpots) * 1000.0
        result["median_tpot_ms"] = statistics.median(tpots) * 1000.0
    else:
        actual_output_lens = [0] * len(outputs)
        result = {
            "completed": 0,
            "total_input": 0,
            "total_output": 0,
            "request_throughput": 0.0,
            "input_throughput": 0.0,
            "output_throughput": 0.0,
            "mean_latency_ms": 0.0,
            "mean_ttft_ms": 0.0,
            "mean_tpot_ms": 0.0,
        }

    result["duration"] = duration
    result["total_input_tokens"] = int(result.pop("total_input"))
    result["total_output_tokens"] = int(result.pop("total_output"))
    result["total_tgs"] = (
        result["total_input_tokens"] + result["total_output_tokens"]
    ) / duration
    records = [
        request_record(
            timed,
            measurement_started_at=measurement_started_at,
            wave_size=args.wave_size,
            actual_output_tokens=actual_output_len,
        )
        for timed, actual_output_len in zip(
            measured_results, actual_output_lens
        )
    ]
    successful_barrier_ttfts = [
        record["barrier_ttft_ms"] for record in records if record["success"]
    ]
    result["mean_barrier_ttft_ms"] = (
        statistics.fmean(successful_barrier_ttfts)
        if successful_barrier_ttfts
        else 0.0
    )
    result.update(
        {
            "date": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "protocol": "closed_loop_barrier_waves",
            "backend": "openai-chat",
            "model_id": args.model,
            "tokenizer_id": tokenizer_id,
            "api_url": api_url,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "prefix_len": args.prefix_len,
            "wave_size": args.wave_size,
            "requested_release_partition": list(args.release_partition),
            "inter_group_delay_ms": args.inter_group_delay_ms,
            "inter_wave_delay_ms": args.inter_wave_delay_ms,
            "num_waves": args.num_waves,
            "warmup_waves": args.warmup_waves,
            "seed": args.seed,
            "warmup_seed": warmup_seed,
            "temperature": float(os.environ.get("BENCH_TEMPERATURE", "0.9")),
            "top_k": int(os.environ.get("BENCH_TOP_K", "20")),
            "top_p": float(os.environ.get("BENCH_TOP_P", "0.95")),
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": actual_output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "errors": [output.error for output in outputs],
            "requests": records,
            "waves": summarize_waves(records),
            "warmup_completed": sum(
                timed.output.success for timed in warmup_results
            ),
            "warmup_errors": [
                timed.output.error
                for timed in warmup_results
                if not timed.output.success
            ],
        }
    )

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print_summary(result)
    print(f"Result JSON: {args.result_json}")
    expected = len(measured_results)
    if result["completed"] != expected or result["warmup_completed"] != len(
        warmup_results
    ):
        print("One or more requests failed; inspect the result JSON.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
