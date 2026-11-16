# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the production V4 projection and V4.1 routing integration."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa
import v41_gate_projection_test_module as native


def _check_routes(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)


def _time_graph(fn: Callable[[], object]) -> dict[str, object]:
    for _ in range(3):
        fn()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        outputs = [fn() for _ in range(40)]
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (40 * 20))
    del graph, outputs
    gc.collect()
    return {"samples_us": samples, "median_us": statistics.median(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(3102)
    torch.musa.set_device(0)
    checks, graphs, timings = [], [], []
    alternate = torch.musa.Stream()
    with torch.inference_mode():
        for alternate_stream in [False, True]:
            stream = alternate if alternate_stream else torch.musa.default_stream()
            stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                for cache_fp32 in [False, True]:
                    for fp32_weight in [False, True]:
                        dtype = torch.float32 if fp32_weight else torch.bfloat16
                        weight = torch.randn(384, 5120, device="musa", dtype=dtype) / 64
                        bias = torch.randn(384, device="musa") / 8
                        old = native.Gate(weight, bias, cache_fp32, False)
                        new = native.Gate(weight, bias, cache_fp32, True)
                        for rows in [0, 1, 4, 6, 7]:
                            for fp32_input in [False, True]:
                                dtype = torch.float32 if fp32_input else torch.bfloat16
                                x = torch.randn(rows, 5120, device="musa", dtype=dtype)
                                case = {"rows": rows, "cache_fp32": cache_fp32,
                                        "fp32_weight": fp32_weight,
                                        "fp32_input": fp32_input,
                                        "alternate_stream": alternate_stream}
                                (args.output / "current_case.json").write_text(json.dumps(case))
                                _check_routes(new.forward(x), old.forward(x))
                                checks.append(case)
                        stream.synchronize()
        with torch.musa.stream(alternate):
            weight = torch.randn(384, 5120, device="musa", dtype=torch.bfloat16) / 64
            bias = torch.randn(384, device="musa") / 8
            old = native.Gate(weight, bias, False, False)
            new = native.Gate(weight, bias, False, True)
            for rows in [1, 4, 6]:
                x = torch.randn(rows, 5120, device="musa", dtype=torch.bfloat16)
                raw = native.projection(x, weight)
                torch.testing.assert_close(
                    raw, x.float() @ weight.float().T, rtol=1e-4, atol=1e-4)
                for _ in range(3):
                    new.forward(x)
                alternate.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph):
                    actual = new.forward(x)
                for update in range(3):
                    x.copy_(torch.randn_like(x))
                    graph.replay()
                    alternate.synchronize()
                    _check_routes(actual, old.forward(x))
                    graphs.append({"rows": rows, "update": update})
                del graph, actual
                gc.collect()
                for label, gate in [("off", old), ("on", new), ("reverse_off", old)]:
                    timings.append({"rows": rows, "label": label,
                                    **_time_graph(lambda: gate.forward(x))})
                (args.output / "partial.json").write_text(json.dumps(
                    {"checks": checks, "graphs": graphs, "timings": timings}, indent=2))
    result = {"status": "PASS", "routing_checks": len(checks),
              "dynamic_graph_checks": len(graphs), "checks": checks,
              "graphs": graphs, "timings": timings,
              "scope": "Production V4 MATE projection plus V4.1 selection; not full-model acceptance"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Production routing checks=%d, dynamic graph checks=%d", len(checks), len(graphs))


if __name__ == "__main__":
    main()
